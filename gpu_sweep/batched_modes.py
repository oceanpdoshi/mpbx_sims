"""Batched mode solves on one device (a GPU): several "slots" — one x-parity sector of one
wavelength each — go through the Newton / LOBPCG iteration together, in one vmapped,
jit-compiled call per Newton step.

This is sim_core/tfln_modes.solve_wavelength (and chunked_sweep/run_geometry._solve_one's use of
it) rearranged for a GPU; the mathematics is unchanged:
  * The eigensolve is mpbjax.modes.solve_omega2 (the same LOBPCG, tolerances and iteration cap),
    vmapped over slots. Each slot has its own k, parity sign, ε⁻¹ (= its wavelength) and LOBPCG
    block. The sign is a traced value, so ONE compiled program serves both sectors and every
    wavelength of a grid (one program per (grid, nev, batch size)).
  * The Newton iteration on k (tfln_modes.solve_k_checked) runs for all slots of a batch in
    lockstep: band by band, one batched eigensolve per Newton step; a slot whose |Δk| < k_tol is
    frozen (its result is recorded at that step, exactly where solve_k_checked stops) and rides
    along until every slot of the batch has converged.
  * A cold slot starts from the same random block lobpcg draws for X0=None (same key), so a cold
    solve here is the same computation as tfln_modes' cold solve.
  * The per-mode post-processing (power normalization, TE / LN fraction, edge diagnostics, HG00
    overlap, and the node-count labels of run_geometry.node_labels) runs on the device. No field
    leaves the GPU: per mode only scalars and two short lines (for the node count) come back.
check_gpu_local.py compares the results with tfln_modes.solve_wavelength on the CPU.

Device memory: one slot needs ~610-750 bytes per (pixel x band) while it is solved (XLA memory
analysis); `GridContext.batch_size` compiles the batched program, reads its exact buffer sizes and
runs fewer slots per call if it would not fit next to what is already on the device.
"""
from __future__ import annotations

import dataclasses
import datetime
import functools
import math
import time

import jax
import jax.numpy as jnp
import numpy as np

from mpbjax import analysis as An
from mpbjax import grid_2d
from mpbjax import modes as Mo
from mpbjax.helmholtz import HMkH, mag_mn

import sim_core.tfln_modes as M
from geometry_gen import tfln_rib as R

GiB = 1024 ** 3


# --------------------------------------------------------------------------------------
# Small dense eigensolves inside LOBPCG
# --------------------------------------------------------------------------------------
class _LinalgProxy:
    def __init__(self, impl):
        self._impl = impl

    def __getattr__(self, k):
        return getattr(jnp.linalg, k)

    def eigh(self, a, UPLO=None, symmetrize_input=True):
        v, w = jax.lax.linalg.eigh(a, lower=UPLO is None or UPLO == "L", symmetrize_input=symmetrize_input,
                                   implementation=self._impl)
        return w, v                                           # jnp.linalg.eigh's (eigenvalues, eigenvectors)


class _JnpProxy:
    def __init__(self, linalg):
        self.linalg = linalg

    def __getattr__(self, k):
        return getattr(jnp, k)


def set_eigh(implementation):
    """Which algorithm mpbjax's LOBPCG uses for its small (<= 3·nev square) Hermitian eigensolves.

    "default": jnp.linalg.eigh as shipped. On a GPU JAX then lets cuSOLVER pick, which for such small
    matrices is the Jacobi solver (syevj). Its stopping test is relative to the whole matrix norm, and
    mpbjax's Rayleigh-Ritz step puts ~1e12 on the diagonal for null directions, so the small
    eigenvalues come out less exact and LOBPCG needs more iterations (estimated 1.3-1.8x).
    "qr": cuSOLVER's QR-based syevd (exact to rounding; one call per matrix of a batch).
    On the CPU both are LAPACK heevd: no difference. Only mpbjax.eigensolver is affected. Call it
    before anything is compiled (the workers do, at start-up)."""
    import mpbjax.eigensolver as E
    impl = {"default": None, "qr": jax.lax.linalg.EighImplementation.QR,
            "jacobi": jax.lax.linalg.EighImplementation.JACOBI}[implementation]
    E.jnp = jnp if impl is None else _JnpProxy(_LinalgProxy(impl))


# --------------------------------------------------------------------------------------
# Compiled kernels
# --------------------------------------------------------------------------------------
@functools.partial(jax.jit, static_argnames=("grid", "nev", "tol", "maxiter", "use_parity"))
def newton_step(k, sign, einv, X, omega, band, key, *, grid, nev, tol, maxiter, use_parity):
    """One Newton step of tfln_modes.solve_k_checked for B slots at once.

    k (B,), sign (B,) ±1, einv (B, 3, 3, Nx, Ny) real, X (B, 2·Nx·Ny, nev) LOBPCG blocks,
    omega (B,), band: the band being converged (traced). Returns the new blocks, Δk of ``band``
    (k_new = k - Δk), the LOBPCG iteration counts and converged flags."""
    def one(k1, s1, ei, X0, om):
        parities = {0: s1} if use_parity else None
        _vals, _vecs, res = Mo.solve_omega2(k1, ei, grid, nev=nev, tol=tol, maxiter=maxiter, X0=X0,
                                            key=key, parities=parities)
        X1 = res.eigenvectors
        ev = jax.lax.dynamic_index_in_dim(X1, band, axis=1, keepdims=False)
        mag, mn = mag_mn(k1, grid)
        dw2dk = 2.0 * HMkH(ev, ei, mag, mn) / jnp.real(jnp.vdot(ev, ev))
        val = jax.lax.dynamic_index_in_dim(res.eigenvalues, band, keepdims=False)
        return X1, (val - om ** 2) / dw2dk, res.iterations, res.converged

    return jax.vmap(one)(k, sign, einv, X, omega)


@functools.partial(jax.jit, static_argnames=("grid",))
def canonicalize_and_ng(ks, evs, omega, einv, deps, *, grid):
    """Phase-canonicalized eigenvectors (solve_k_checked) and group indices (analysis.group_index)
    of the nev modes of one slot: ks (nev,), evs (nev, n). One mode at a time (lax.map)."""
    def one(args):
        k, ev = args
        evc = Mo.canonicalize_phase(ev, k, einv, grid)
        return evc, An.group_index_value(k, evc, omega, einv, deps, grid)

    return jax.lax.map(one, (ks, evs))


def _hg00(F, X, Y):
    """tfln_modes.hg00_overlap on the device."""
    I = jnp.abs(F) ** 2
    P = jnp.sum(I)
    Ps = jnp.where(P > 0, P, 1.0)
    x0, y0 = jnp.sum(X * I) / Ps, jnp.sum(Y * I) / Ps
    wx = 2.0 * jnp.sqrt(jnp.sum((X - x0) ** 2 * I) / Ps)
    wy = 2.0 * jnp.sqrt(jnp.sum((Y - y0) ** 2 * I) / Ps)
    G = jnp.exp(-((X - x0) / wx) ** 2 - ((Y - y0) / wy) ** 2)
    return jnp.where(P > 0, jnp.abs(jnp.sum(F * G)) ** 2 / (Ps * jnp.sum(G ** 2)), 0.0)


@functools.partial(jax.jit, static_argnames=("grid",))
def mode_diagnostics(ks, evs, omega, einv, nng, eps, ln_fill, edge_mask, ring, X, Y, *, grid):
    """Per mode (lax.map over ks (M,), evs (M, n)): the quantities tfln_modes.solve_wavelength
    computes from the power-normalized fields, plus the two lines run_geometry.node_labels counts
    sign changes on. Returns (P, te_frac, ln_frac, edge_frac, edge_ratio, hg00, line_x (M, Nx),
    line_y (M, Ny))."""
    dV = grid.dV

    def one(args):
        k, ev = args
        E = Mo.e_field(k, ev, einv, nng, grid)                 # = tfln_modes._power_normalized_fields
        H = Mo.h_field(k, ev, omega, einv, nng, grid)
        Sz = jnp.real(jnp.conj(E[0]) * H[1] - jnp.conj(E[1]) * H[0])
        P = jnp.sum(Sz) * dV
        E = E * jnp.where(P != 0, 1.0 / jnp.sqrt(jnp.abs(jnp.where(P != 0, P, 1.0))), 1.0)
        Ex2, Ey2 = jnp.abs(E[0]) ** 2, jnp.abs(E[1]) ** 2
        I2 = Ex2 + Ey2 + jnp.abs(E[2]) ** 2
        sx, sy = jnp.sum(Ex2), jnp.sum(Ey2)
        te = sx / (sx + sy)
        w = jnp.real(jnp.einsum("a...,ab...,b...->...", jnp.conj(E), eps, E))     # E*·ε·E
        ln_frac = jnp.sum(w * ln_fill) / jnp.sum(w)
        edge_frac = jnp.sum(jnp.where(edge_mask, I2, 0.0)) / jnp.sum(I2)
        edge_ratio = jnp.max(jnp.where(ring, I2, 0.0)) / jnp.max(I2)
        F = jnp.where(te >= 0.5, E[0], E[1])                   # dominant transverse component
        hg00 = _hg00(F, X, Y)
        # run_geometry.node_labels (it works on the stored complex64 fields)
        Fc = F.astype(jnp.complex64)
        i0 = jnp.argmax(jnp.abs(Fc))                           # flat, C order -> (ix, iy)
        ix = i0 // Fc.shape[1]
        Fr = jnp.real(Fc * jnp.exp(-1j * jnp.angle(Fc.reshape(-1)[i0])))
        jy = jnp.argmax(jnp.sum(Fr ** 2, axis=0))
        line_x = jax.lax.dynamic_index_in_dim(Fr, jy, axis=1, keepdims=False)
        line_y = jax.lax.dynamic_index_in_dim(Fr, ix, axis=0, keepdims=False)
        return P, te, ln_frac, edge_frac, edge_ratio, hg00, line_x, line_y

    return jax.lax.map(one, (ks, evs))


def cold_block(n, p):
    """The random starting block mpbjax.eigensolver.lobpcg draws for X0=None with key PRNGKey(0)
    (the key tfln_modes uses), so a cold slot repeats tfln_modes' cold solve exactly."""
    key = jax.random.PRNGKey(0)
    key, _ = jax.random.split(key)
    kr, ki = jax.random.split(key)
    return (jax.random.normal(kr, (n, p), dtype=jnp.float64)
            + 1j * jax.random.normal(ki, (n, p), dtype=jnp.float64)).astype(jnp.complex128)


def count_sign_changes(line):
    """run_geometry.node_labels' inner ``changes``."""
    a = np.abs(line)
    s = np.sign(line[a > 0.03 * a.max()])
    return int(np.sum(s[1:] != s[:-1]))


def is_oom(e):
    s = str(e)
    return "RESOURCE_EXHAUSTED" in s or "Out of memory" in s or "out of memory" in s


# --------------------------------------------------------------------------------------
# One grid on one device
# --------------------------------------------------------------------------------------
@dataclasses.dataclass
class LamData:
    """A wavelength's dielectric on the device (and what the post-processing needs)."""
    lam: float
    omega: float
    einv_np: np.ndarray                  # host copy (k_guess)
    einv: jax.Array
    deps: jax.Array
    nng: jax.Array
    eps: jax.Array
    n_cut: float


@dataclasses.dataclass
class Slot:
    """One x-parity sector of one wavelength. ``warm`` = (omega, [k], [ev], [ng]) of the same sector
    at a neighbouring wavelength (tfln_modes' warm-start state), or None (cold)."""
    lam: LamData
    sign: int
    warm: tuple | None = None


class GridContext:
    """The device-side state of one (geometry, grid): dielectric plan, coordinates, masks, compiled
    programs and the batch-size caps found for each nev."""

    def __init__(self, geometry, grid_info, align_frac, use_parity, k_tol, eig_tol, field_dtype="complex64",
                 headroom_gb=2.5, max_slots=None, log=print):
        g = grid_info
        self.grid = grid_2d(g["Lx"], g["Ly"], g["Nx"], g["Ny"])
        self.geom = R.align_to_grid(R.RibGeometry(**geometry), self.grid, frac=align_frac, warn=False)
        self.diel = R.RibDielectric(self.geom, self.grid)
        self.use_parity = use_parity
        self.sectors = (-1, +1) if use_parity else (0,)
        self.settings_kw = dict(use_parity=use_parity, warm_start=True, label_modes=False, store_fields=False,
                                field_dtype=field_dtype, k_tol=k_tol, eig_tol=eig_tol)
        base = M.SolverSettings(**self.settings_kw)
        self.maxiter, self.max_newton = base.maxiter, base.max_newton
        self.n = 2 * self.grid.Nx * self.grid.Ny
        self.headroom = headroom_gb * GiB
        self.max_slots = max_slots                # optional cap on slots per call (bench: lockstep vs one by one)
        self.log = log
        xs, ys = R.grid_coords(self.grid)
        self.xs, self.ys = xs, ys
        Xh, Yh = np.meshgrid(xs, ys, indexing="ij")
        ew = base.edge_width
        edge = (Xh - xs[0] < ew) | (xs[-1] - Xh < ew) | (Yh - ys[0] < ew) | (ys[-1] - Yh < ew)
        ring = np.zeros_like(edge)
        ring[[0, -1], :] = True
        ring[:, [0, -1]] = True
        self.X, self.Y = jnp.asarray(Xh), jnp.asarray(Yh)
        self.edge_mask, self.ring = jnp.asarray(edge), jnp.asarray(ring)
        self.ln_fill = jnp.asarray(self.geom.ln_fill(self.grid))
        self.key = jax.random.PRNGKey(0)
        self.device = jax.devices()[0]
        self._progs = {}                      # (bands, B) -> (compiled, bytes needed, compile s)
        self._cap = {}                        # bands -> max slots per call, set only when memory forced it
        self._bpb = 0.0                       # largest measured program bytes per (slot x band)
        self.compile_log = []

    # ---- device memory -----------------------------------------------------------------
    def mem_stats(self):
        try:
            return self.device.memory_stats() or {}
        except Exception:                                     # noqa: BLE001  (CPU backend)
            return {}

    def peak_gb(self):
        return self.mem_stats().get("peak_bytes_in_use", float("nan")) / GiB

    def budget(self):
        """Device bytes a new program may use: what is free (and, if the allocator reports it, no more
        than its largest free block: XLA's temporaries are one block), minus the headroom."""
        st = self.mem_stats()
        if "bytes_limit" not in st:
            return math.inf
        free = st["bytes_limit"] - st.get("bytes_in_use", 0)
        if st.get("largest_free_block_bytes"):
            free = min(free, st["largest_free_block_bytes"])
        return free - self.headroom

    def bands(self, nev):
        """Bands per solve: nev per sector with parity, 2·nev in the single unsplit solve (as on the CPU)."""
        return nev if self.use_parity else 2 * nev

    # ---- programs ----------------------------------------------------------------------
    def program(self, nev, B):
        """The compiled newton_step for B slots of nev bands (nev = bands per solve; compiled once)."""
        if (nev, B) not in self._progs:
            g = self.grid
            S = jax.ShapeDtypeStruct
            t0 = time.perf_counter()
            low = newton_step.lower(S((B,), jnp.float64), S((B,), jnp.float64),
                                    S((B, 3, 3, g.Nx, g.Ny), jnp.float64), S((B, self.n, nev), jnp.complex128),
                                    S((B,), jnp.float64), S((), jnp.int32), self.key,
                                    grid=g, nev=nev, tol=self.settings_kw["eig_tol"], maxiter=self.maxiter,
                                    use_parity=self.use_parity)
            comp = low.compile()
            need = 0
            try:
                ma = comp.memory_analysis()
                need = (ma.temp_size_in_bytes + ma.argument_size_in_bytes + ma.output_size_in_bytes
                        - getattr(ma, "alias_size_in_bytes", 0))
            except Exception:                                 # noqa: BLE001
                pass
            dt = time.perf_counter() - t0
            self._progs[(nev, B)] = (comp, need, dt)
            self._bpb = max(self._bpb, need / (B * nev))
            self.compile_log.append(dict(nev=nev, B=B, compile_s=round(dt, 2), need_gb=round(need / GiB, 3)))
            self.log(f"    compiled nev={nev} B={B}: {dt:.1f} s, needs {need / GiB:.2f} GiB "
                     f"(free for it: {self.budget() / GiB:.2f} GiB)")
        return self._progs[(nev, B)]

    def _extra(self, nev, B):
        """Device bytes a batch holds beside its program: the per-band eigenvector copies kept through
        the Newton loop and the stacked starting blocks (complex128, n x nev per slot each)."""
        return 2 * B * nev * self.n * 16

    def batch_size(self, nev, n_slots):
        """Slots per call for n_slots slots of nev bands: all of them if they fit, else the equal split
        into the fewest calls that fits. Only sizes that will run are compiled; a cap is remembered
        only when memory forced it (or an out-of-memory error, lower_cap)."""
        cap = min(n_slots, self._cap.get(nev, n_slots), self.max_slots or n_slots)
        if self._bpb and self.budget() < math.inf:            # first guess from programs compiled so far
            per = self._bpb * nev + self._extra(nev, 1)
            cap = max(1, min(cap, int(self.budget() // per)))
        while True:
            B = math.ceil(n_slots / math.ceil(n_slots / cap))
            _, need, _ = self.program(nev, B)
            free = self.budget() - self._extra(nev, B)
            if need <= free or B == 1:
                return B
            cap = max(1, min(B - 1, int(B * free / need)))
            self._cap[nev] = cap                              # a memory-forced limit
            self.log(f"    nev {nev}: {B} slots per call need {need / GiB:.2f} GiB, {free / GiB:.2f} GiB free "
                     f"-> at most {cap} per call")

    def lower_cap(self, nev, B):
        self._cap[nev] = max(1, B // 2)

    # ---- wavelengths ------------------------------------------------------------------
    def lam_data(self, lam):
        d = self.diel.at(lam)
        return LamData(lam=lam, omega=d["omega"], einv_np=d["einv"], einv=jnp.asarray(d["einv"]),
                       deps=jnp.asarray(d["deps"]), nng=jnp.asarray(d["nng"]), eps=jnp.asarray(d["eps"]),
                       n_cut=self.geom.cutoff_index(lam))

    # ---- solves -----------------------------------------------------------------------
    def solve_slots(self, slots, nev):
        """Newton/LOBPCG for every slot, in batches that fit on the device. ``nev`` = modes per parity
        sector (the chunk's nev); a solve has bands(nev) bands. Returns per slot dict(k, ev
        (bands, n) canonicalized, ng, diag [per band], sign) and a stats dict (batch sizes, calls)."""
        nev = self.bands(nev)
        out = [None] * len(slots)
        todo = list(range(len(slots)))
        stats = dict(batches=[], newton_calls=0)
        while todo:
            size = self.batch_size(nev, len(todo))
            group = todo[:size]
            try:
                res, calls = self._newton_batch([slots[i] for i in group], nev)
            except Exception as e:                            # noqa: BLE001
                if is_oom(e) and size > 1:
                    self.log(f"    out of device memory with {size} slots of nev {nev}: retrying with fewer")
                    self.lower_cap(nev, size)
                    continue
                raise
            for i, r in zip(group, res):
                out[i] = r
            stats["batches"].append(size)
            stats["newton_calls"] += calls
            todo = todo[size:]
        return out, stats

    def _newton_batch(self, slots, nev):
        """tfln_modes.solve_k_checked for len(slots) slots in lockstep (see the module doc)."""
        B = len(slots)
        comp, _, _ = self.program(nev, B)
        kguess = []
        blocks = []
        for s in slots:
            if s.warm is not None and len(s.warm[1]) == nev:
                om_p, k_p, ev_p, ng_p = s.warm
                kguess.append([k + (s.lam.omega - om_p) * g for k, g in zip(k_p, ng_p)])   # k ≈ k_p + n_g Δω
                blocks.append(jnp.stack(list(ev_p), axis=1))
            else:
                kguess.append(None)
                blocks.append(None)
        if any(b is None for b in blocks):                    # drawn per call (cheap), not kept on the device
            cold = cold_block(self.n, nev)
            blocks = [cold if b is None else b for b in blocks]
            del cold
        X = jnp.stack(blocks)
        del blocks
        sign = jnp.asarray([float(s.sign) for s in slots], dtype=jnp.float64)
        omega_h = np.array([s.lam.omega for s in slots], dtype=float)
        omega = jnp.asarray(omega_h)
        einv = jnp.stack([s.lam.einv for s in slots])
        k_prev = np.array([Mo.k_guess(s.lam.omega, s.lam.einv_np) if kguess[i] is None else np.nan
                           for i, s in enumerate(slots)])
        kmags = np.zeros((B, nev))
        evs = [[None] * nev for _ in range(B)]
        diag = [[None] * nev for _ in range(B)]
        calls = 0
        for b in range(nev):
            k = np.array([kguess[i][b] if kguess[i] is not None else k_prev[i] for i in range(B)], dtype=float)
            info = [dict(newton=0, lobpcg_iters=0, lobpcg_converged=True, newton_converged=False) for _ in range(B)]
            active = np.ones(B, bool)
            band = jnp.asarray(b, dtype=jnp.int32)
            for _ in range(self.max_newton):
                X, dk, iters, conv = comp(jnp.asarray(k), sign, einv, X, omega, band, self.key)
                calls += 1
                dk_h, it_h, cv_h = jax.device_get((dk, iters, conv))
                for i in np.flatnonzero(active):
                    info[i]["newton"] += 1
                    info[i]["lobpcg_iters"] = max(info[i]["lobpcg_iters"], int(it_h[i]))
                    info[i]["lobpcg_converged"] = bool(cv_h[i])
                    k[i] -= dk_h[i]
                    if abs(dk_h[i]) < self.settings_kw["k_tol"]:
                        info[i]["newton_converged"] = True
                        active[i] = False
                        evs[i][b] = X[i, :, b]                # the iterate solve_k_checked keeps
                    elif not np.isfinite(dk_h[i]):            # a NaN never converges: stop it here (not
                        active[i] = False                     # converged) instead of holding the whole batch
                        evs[i][b] = X[i, :, b]                # for max_newton steps
                if not active.any():
                    break
            for i in np.flatnonzero(active):                  # max_newton reached: the last iterate
                evs[i][b] = X[i, :, b]
            kmags[:, b] = k
            k_prev = k.copy()
            for i in range(B):
                diag[i][b] = info[i]
        del X
        res = []
        for i, s in enumerate(slots):
            evc, ng = canonicalize_and_ng(jnp.asarray(kmags[i]), jnp.stack(evs[i]), s.lam.omega, s.lam.einv,
                                          s.lam.deps, grid=self.grid)
            res.append(dict(k=kmags[i].copy(), ev=evc, ng=np.asarray(ng, dtype=float), diag=diag[i], sign=s.sign))
        return res, calls

    # ---- post-processing ----------------------------------------------------------------
    def build_sweep(self, ld: LamData, results, nev, t_solve, extra_meta=None):
        """The ModeSweep (one wavelength, no fields) that tfln_modes.solve_wavelength returns, the
        node-count labels of run_geometry.node_labels, and the new warm-start state."""
        modes = []
        for r in results:
            for j in range(len(r["k"])):
                modes.append((float(r["k"][j]), r["sign"], j, r))
        modes.sort(key=lambda m: -m[0])                       # descending k = descending n_eff
        ks = jnp.asarray([m[0] for m in modes])
        evs = jnp.stack([m[3]["ev"][m[2]] for m in modes])
        t0 = time.perf_counter()
        P, te, lnf, edge, ratio, hg00, lx, ly = jax.device_get(
            mode_diagnostics(ks, evs, ld.omega, ld.einv, ld.nng, ld.eps, self.ln_fill, self.edge_mask, self.ring,
                             self.X, self.Y, grid=self.grid))
        del evs
        rec = {k: [] for k in M.FLOAT_FIELDS + M.INT_FIELDS + M.BOOL_FIELDS + M.STR_FIELDS}
        labs = []
        for j, (k, sign, b, r) in enumerate(modes):
            info = r["diag"][b]
            neff = k / ld.omega
            t = float(te[j])
            rec["neff"].append(neff)
            rec["ng"].append(float(r["ng"][b]))
            rec["te_frac"].append(t)
            rec["ln_frac"].append(float(lnf[j]))
            rec["edge_frac"].append(float(edge[j]))
            rec["edge_db"].append(float(10 * np.log10(max(float(ratio[j]), 1e-30))))
            rec["hg00"].append(float(hg00[j]))
            rec["power_raw"].append(float(P[j]))
            rec["parity"].append(sign)
            rec["newton_iters"].append(info["newton"])
            rec["lobpcg_iters"].append(info["lobpcg_iters"])
            rec["guided"].append(bool(neff > ld.n_cut))
            rec["converged"].append(bool(info["newton_converged"] and info["lobpcg_converged"]))
            rec["hg_pol"].append("TE" if t >= 0.5 else "TM")      # label_modes=False, as in production
            rec["hg_m"].append(-1)
            rec["hg_n"].append(-1)
            rec["hg_err"].append(np.nan)
            labs.append(("TE" if t >= 0.5 else "TM", count_sign_changes(lx[j]), count_sign_changes(ly[j])))
        t_post = time.perf_counter() - t0
        data = {k: np.asarray(v)[None, :] for k, v in rec.items()}
        n_e, n_o = R.ln_indices(ld.lam)
        per_lam = dict(n_cutoff=[ld.n_cut], n_e=[n_e], n_o=[n_o], n_sio2=[R.sio2_index(ld.lam)],
                       solve_time=[t_solve + t_post])
        settings = M.SolverSettings(nev=nev if self.use_parity else 2 * nev, **self.settings_kw)
        meta = dict(geometry=self.geom.as_dict(),
                    grid=dict(Lx=self.grid.Lx, Ly=self.grid.Ly, Nx=self.grid.Nx, Ny=self.grid.Ny),
                    settings=settings.as_dict(),
                    material=dict(LN="mpbjax.materials.MgO_LiNbO3_matsuda (Matsuda 2015, 20 C)",
                                  rotation_R=R.R_XCUT.tolist(), rotation_convention="eps_lab = R^T eps_crystal R",
                                  substrate_cladding="mpbjax.materials.SiO2 (Malitson)"),
                    created=datetime.datetime.now().isoformat(timespec="seconds"),
                    backend=jax.default_backend(), devices=[str(dv) for dv in jax.devices()],
                    x64=bool(jax.config.jax_enable_x64),
                    timing=dict(eigensolve_s=t_solve, postprocess_s=t_post))
        meta.update(extra_meta or {})
        sw = M.ModeSweep([ld.lam], data, per_lam, self.xs, self.ys, None, meta)
        warm = {r["sign"]: (ld.omega, [float(x) for x in r["k"]], [r["ev"][j] for j in range(len(r["k"]))],
                            [float(x) for x in r["ng"]]) for r in results}
        return sw, labs, warm, t_post
