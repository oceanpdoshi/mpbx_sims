"""Forward (no adjoint) mode solves of the X-cut TFLN rib with mpbx.

    solve_wavelength(diel, lam, settings)   -> ModeSweep holding one wavelength
    run_sweep(geom, grid, lams, settings)   -> ModeSweep over all wavelengths (+ HDF5)
    track_mode(sweep, "TE00")               -> the TE00 branch across wavelength
    ModeSweep.save / ModeSweep.load / .to_csv / .table()

Per mode and wavelength the sweep stores:
    neff, ng           effective and group index. ng = c/v_g includes material
                       dispersion (mpbjax.analysis.group_index with ∂ε/∂ω), the same
                       definition as Tidy3D's "group index".
    te_frac            Σ|Ex|² / (Σ|Ex|² + Σ|Ey|²). TE = Ex = crystal Z, TM = Ey = crystal X.
                       Equals 1 - Tidy3D pol_fraction.te (the CSV "TE (Ex) fraction") for the
                       notebook's y-normal plane, i.e. ``zpol`` in hg_mode_idx_tracking.py:
                       Tidy3D's "TE" there is the vertical crystal-X component = TM here.
    hg_pol, hg_m, hg_n library Hermite-Gaussian fit label (mpbjax.analysis.hg_mode_label):
                       pol = TE/TM, m = order along x (width), n = order along y (thickness).
    hg_err             relative residual of that fit (0 = perfect HG; > ~0.3 unreliable).
    hg00               overlap of the dominant transverse component with a moment-matched
                       Gaussian (the HG00 test from Tidy3D_sims/hg_mode_idx_tracking.py).
    ln_frac            fraction of the electric energy inside the LN.
    edge_frac          fraction of Σ|E|² within ``edge_width`` of the window boundary, and
    edge_db            10·log10(max |E|² on the outer pixel ring / max |E|²): mode-clipping
                       diagnostics (the window is periodic, not PML-terminated).
    guided             neff > max(substrate, cladding, slab-mode) index (see RibGeometry.cutoff_index).
    parity             x-mirror sector the mode was solved in (H-field parity convention:
                       -1 holds TE00, TM10, TE20, ...; +1 holds TM00, TE10, TM20, ...; 0 = no split).
    E                  (3, Nx, Ny) complex field (Ex, Ey, Ez) in the lab frame, normalized to
                       ∫ Re(E* × H)·ẑ dA = 1 in mpbx natural units (ε0 = μ0 = c = 1,
                       H = (k+G)×E/ω, so |H| ≈ n|E|). This is NOT Tidy3D's normalization
                       (½∫Re(E* × H)·ẑ dA = 1 W with E in V/µm): E_tidy3d = sqrt(2·Z0)·E_here
                       ≈ 27.45·E_here. See ``to_si_fields``.
At each wavelength the modes are ordered by descending neff ("mode index" = that rank).
"""
from __future__ import annotations

import dataclasses
import datetime
import functools
import json
import math
import os
import time
import types

import jax
import jax.numpy as jnp
import numpy as np

from mpbjax import analysis as An
from mpbjax import modes as Mo
from mpbjax.helmholtz import HMkH, mag_mn

from geometry_gen.tfln_rib import (R_XCUT, RibDielectric, RibGeometry, grid_coords, ln_indices,
                      sio2_index)
from mpbjax import grid_2d


# --------------------------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class SolverSettings:
    nev: int = 8               # modes per wavelength (per x-parity sector if use_parity=True)
    # Solve the two x-mirror sectors separately (parities={0: ±1}). Needs an ODD Nx
    # (make_grid(..., odd_x=True)): on an even Nx the projected LOBPCG did not converge here
    # (mpbx GOTCHAS #33, Nyquist plane). With Nx odd it matched the full solve to 2e-15, but
    # was only ~15 % faster in one test, so it is off by default.
    use_parity: bool = False
    k_tol: float = 1e-8        # Newton stop |Δk| [1/µm] -> n_eff converged to ~k_tol·λ
    eig_tol: float = 1e-8      # LOBPCG relative residual tolerance
    maxiter: int = 1000        # LOBPCG iteration cap per eigensolve
    max_newton: int = 40       # Newton steps per band
    warm_start: bool = True    # seed each λ from the previous one (k extrapolated with n_g, eigvecs)
    label_modes: bool = True   # run the library HG fit (pol, m, n) on every mode
    hg_max_order: int = 6      # HG orders 0..6 in x and y (at 0.4 µm TM00..TM60 sit above TE00);
                               # ~2.5 s/mode, 3 -> ~0.7 s/mode but aliases TM40+ to "TM30"
    store_fields: bool = True  # keep E for every mode (needed for field plots / tracking overlaps)
    field_dtype: str = "complex64"
    edge_width: float = 0.25   # µm, boundary band for edge_frac

    def as_dict(self):
        return dataclasses.asdict(self)


# --------------------------------------------------------------------------------------
# Eigensolver: mpbjax.modes.solve_k's Newton iteration with a jit-compiled eigensolve
# --------------------------------------------------------------------------------------
# Mo.solve_k recompiles its LOBPCG while-loop on every eigensolve (mpbx GOTCHAS #21),
# about 0.2-0.3 s each, 10-40 times per wavelength. Wrapping Mo.solve_omega2 in jax.jit
# with the grid/sizes static compiles it once per (grid, nev, parity) and reuses it for
# every wavelength. The Newton driver below is the same algorithm as Mo.solve_k
# (mpbjax/modes.py), plus convergence bookkeeping that solve_k does not report.
@functools.partial(jax.jit, static_argnames=("grid", "nev", "tol", "maxiter", "parity"))
def _eigensolve(k, einv, X0, key, *, grid, nev, tol, maxiter, parity):
    parities = {0: parity} if parity else None
    vals, _vecs, res = Mo.solve_omega2(k, einv, grid, nev=nev, tol=tol, maxiter=maxiter,
                                       X0=X0, key=key, parities=parities)
    return vals, res.eigenvectors, res.iterations, res.converged


def solve_k_checked(omega, einv, grid, nev, *, parity=0, k_tol=1e-8, eig_tol=1e-8,
                    maxiter=1000, max_newton=40, kguess=None, Hguess=None):
    """Fixed-ω modes (``Mo.solve_k`` algorithm). Returns ``(kmags, evecs, diag)``.

    kguess : None (band 0 starts at ω·n_max, each later band at the previous band's k)
             or a per-band sequence of starting |k|.
    Hguess : None or a list of nev eigenvectors to start the LOBPCG block from.
    diag   : per band dict(newton, lobpcg_iters, lobpcg_converged, newton_converged).
    """
    omega = float(omega)
    einv_j = jnp.asarray(einv)
    key = jax.random.PRNGKey(0)
    X = None if Hguess is None else jnp.stack([jnp.asarray(h).reshape(-1) for h in Hguess], axis=1)
    k0 = Mo.k_guess(omega, einv_j) if kguess is None else float(kguess[0])
    kmags, evecs, diag = [], [], []
    for b in range(nev):
        k = float(kguess[b]) if kguess is not None else k0
        info = dict(newton=0, lobpcg_iters=0, lobpcg_converged=True, newton_converged=False)
        ev = None
        for _ in range(max_newton):
            vals, X, iters, conv = _eigensolve(jnp.asarray(k, dtype=jnp.float64), einv_j, X, key,
                                               grid=grid, nev=nev, tol=eig_tol, maxiter=maxiter,
                                               parity=parity)
            info["newton"] += 1
            info["lobpcg_iters"] = max(info["lobpcg_iters"], int(iters))
            info["lobpcg_converged"] = bool(conv)
            ev = X[:, b]
            mag, mn = mag_mn(k, grid)
            dw2dk = 2.0 * float(HMkH(ev, einv_j, mag, mn)) / float(jnp.real(jnp.vdot(ev, ev)))
            dk = (float(vals[b]) - omega ** 2) / dw2dk
            k -= dk
            if abs(dk) < k_tol:
                info["newton_converged"] = True
                break
        kmags.append(k)
        evecs.append(Mo.canonicalize_phase(ev, k, einv_j, grid))
        diag.append(info)
        k0 = k
    return np.asarray(kmags), evecs, diag


# --------------------------------------------------------------------------------------
# Per-mode diagnostics
# --------------------------------------------------------------------------------------
def hg00_overlap(F, xs, ys):
    """|<F, G>|² / (<F,F><G,G>) with G a Gaussian whose center and 1/e² widths match the
    first and second moments of |F|² (port of hg00_overlap in hg_mode_idx_tracking.py)."""
    F = np.asarray(F)
    I = np.abs(F) ** 2
    P = I.sum()
    if P <= 0:
        return 0.0
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    x0, y0 = (X * I).sum() / P, (Y * I).sum() / P
    wx = 2.0 * np.sqrt(((X - x0) ** 2 * I).sum() / P)
    wy = 2.0 * np.sqrt(((Y - y0) ** 2 * I).sum() / P)
    G = np.exp(-((X - x0) / wx) ** 2 - ((Y - y0) / wy) ** 2)
    return float(np.abs((F * G).sum()) ** 2 / (P * (G ** 2).sum()))


def _label_view(E, xs, ys, geom, max_pts=72):
    """Crop E to the region holding the mode (the core ± 0.5 µm, widened to ±3σ of the
    |E|² distribution for weakly confined modes) and decimate to <= max_pts per axis, so
    the library HG fit (32 Nelder-Mead fits per mode at max_order=3) stays ~0.5 s/mode."""
    E = np.asarray(E)
    I = np.sum(np.abs(E[:2]) ** 2, axis=0)
    P = I.sum()
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    x0, y0 = (X * I).sum() / P, (Y * I).sum() / P
    sx = math.sqrt(((X - x0) ** 2 * I).sum() / P)
    sy = math.sqrt(((Y - y0) ** 2 * I).sum() / P)
    hx = max(0.5 * max(geom.w_base, geom.w_top) + 0.5, 3 * sx)
    hy = max(0.5 * geom.t_ln + 0.5, 3 * sy)
    yc = 0.5 * (geom.y0 + geom.y_top)
    ix = np.flatnonzero(np.abs(xs - x0) <= hx)
    iy = np.flatnonzero(np.abs(ys - yc) <= hy)
    ix = ix[::max(1, math.ceil(len(ix) / max_pts))]
    iy = iy[::max(1, math.ceil(len(iy) / max_pts))]
    xv, yv = xs[ix], ys[iy]
    view = types.SimpleNamespace(xs=lambda: xv, ys=lambda: yv)   # all hg_mode_label needs
    return E[:, ix][:, :, iy], view


def _fit_ok(fit, xv, yv):
    """False for a numerically degenerate Hermite-Gaussian fit: non-finite or huge amplitude,
    a waist below half a pixel or wider than 4x the crop, a center outside the crop, or an
    exactly zero residual. (At order 6 a guided TE30 was 'fit' as TE06 with amplitude 1e162,
    wx = 2 nm and y0 outside the crop.)"""
    try:
        A, x0, y0, wx, wy = (float(fit[k]) for k in ("amplitude", "x0", "y0", "wx", "wy"))
    except (TypeError, KeyError, ValueError):
        return False
    dx, dy = xv[1] - xv[0], yv[1] - yv[0]
    Lx, Ly = xv[-1] - xv[0], yv[-1] - yv[0]
    return (np.isfinite([A, x0, y0, wx, wy]).all() and abs(A) < 1e6
            and 0.5 * dx <= wx <= 4 * Lx and 0.5 * dy <= wy <= 4 * Ly
            and xv[0] <= x0 <= xv[-1] and yv[0] <= y0 <= yv[-1]
            and float(fit["residual"]) > 0)


def label_mode(E, xs, ys, geom, max_order=3, occam_tol=0.03):
    """(pol, m, n, rel_error) from the library Hermite-Gaussian fits
    (mpbjax.analysis.hg_fit_residuals, the engine of hg_mode_label), with two guards:
    1. degenerate fits are discarded (see ``_fit_ok``);
    2. among the remaining (pol, m, n) whose error is within ``occam_tol`` of the best, the
       LOWEST order m + n wins. Without it, a weakly confined TE00 at λ = 2 µm was labeled
       "TE31" (a high-order template with nodes outside the crop fits a smooth lobe equally well).
    Returns ("", -1, -1, nan) if no fit survives."""
    Ec, view = _label_view(E, xs, ys, geom)
    xv, yv = np.asarray(view.xs()), np.asarray(view.ys())
    r = An.hg_fit_residuals(Ec, view, max_order=max_order)
    cands = [(m + n, float(err[m, n]), pol, m, n)
             for pol, err, fits in (("TE", r["err_TE"], r["fits_TE"]), ("TM", r["err_TM"], r["fits_TM"]))
             for m in range(err.shape[0]) for n in range(err.shape[1])
             if np.isfinite(err[m, n]) and _fit_ok(fits[m][n], xv, yv)]
    if not cands:
        return "", -1, -1, float("nan")
    e_min = min(c[1] for c in cands)
    _, e, pol, m, n = min((c for c in cands if c[1] <= e_min + occam_tol), key=lambda c: (c[0], c[1]))
    return pol, int(m), int(n), e


Z0_OHM = 376.730313412   # free-space impedance


def to_si_fields(E=None, H=None):
    """Convert stored fields (∫Re(E*×H)_z dA = 1 in mpbx natural units) to the Tidy3D / SI
    convention ½∫Re(E*×H_SI)_z dA = 1 W with E in V/µm, H in A/µm:
    E_SI = sqrt(2·Z0)·E,  H_SI = sqrt(2/Z0)·H."""
    out = []
    if E is not None:
        out.append(np.asarray(E) * math.sqrt(2.0 * Z0_OHM))
    if H is not None:
        out.append(np.asarray(H) * math.sqrt(2.0 / Z0_OHM))
    return out[0] if len(out) == 1 else tuple(out)


def relabel(sweep, max_order=6, occam_tol=0.03, verbose=True):
    """Recompute the Hermite-Gaussian labels of a sweep from its stored fields (no solve),
    e.g. with a higher ``max_order`` when high-order modes alias to a lower label (at 0.4 µm
    this guide has TM00..TM60 above TE00). Updates ``sweep`` in place and returns it."""
    if sweep.E is None:
        raise ValueError("relabel needs stored fields")
    geom = sweep.geometry
    t0 = time.perf_counter()
    g = sweep.data
    g["hg_pol"] = g["hg_pol"].astype("<U2")
    for i in range(sweep.n_lam):
        for j in range(sweep.n_modes):
            pol, m, n, e = label_mode(sweep.E[i, j].astype(complex), sweep.xs, sweep.ys, geom,
                                      max_order=max_order, occam_tol=occam_tol)
            g["hg_pol"][i, j], g["hg_m"][i, j], g["hg_n"][i, j], g["hg_err"][i, j] = pol, m, n, e
    sweep.meta.setdefault("settings", {})["hg_max_order"] = max_order
    if verbose:
        print(f"[relabel] {sweep.n_lam * sweep.n_modes} modes, max_order={max_order}: "
              f"{time.perf_counter() - t0:.1f} s")
    return sweep


def _power_normalized_fields(k, ev, d, grid):
    """E, H energy-normalized by mpbx, then rescaled so ∫ Re(E* × H)_z dA = 1 (mpbx units)."""
    E = np.asarray(Mo.e_field(k, ev, d["einv"], d["nng"], grid))
    H = np.asarray(Mo.h_field(k, ev, d["omega"], d["einv"], d["nng"], grid))
    Sz = np.real(np.conj(E[0]) * H[1] - np.conj(E[1]) * H[0])
    P = float(Sz.sum() * grid.dV)
    s = 1.0 / math.sqrt(abs(P)) if P != 0 else 1.0
    return E * s, H * s, P


# --------------------------------------------------------------------------------------
# Results container
# --------------------------------------------------------------------------------------
FLOAT_FIELDS = ("neff", "ng", "te_frac", "ln_frac", "edge_frac", "edge_db", "hg00", "hg_err", "power_raw")
INT_FIELDS = ("parity", "hg_m", "hg_n", "newton_iters", "lobpcg_iters")
BOOL_FIELDS = ("guided", "converged")
STR_FIELDS = ("hg_pol",)
PER_LAM = ("n_cutoff", "n_e", "n_o", "n_sio2", "solve_time")


class ModeSweep:
    """Modes over wavelength. Per-mode arrays are (n_lam, n_modes); ``E`` is
    (n_lam, n_modes, 3, Nx, Ny) or None. ``lam`` ascending; modes sorted by descending neff."""

    def __init__(self, lam, data, per_lam, xs, ys, E=None, meta=None):
        self.lam = np.asarray(lam, dtype=float)
        self.data = {k: np.asarray(v) for k, v in data.items()}
        self.per_lam = {k: np.asarray(v, dtype=float) for k, v in per_lam.items()}
        self.xs, self.ys = np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)
        self.E = E
        self.meta = dict(meta or {})

    # ---- convenient accessors -----------------------------------------------------
    def __getattr__(self, name):
        data = self.__dict__.get("data", {})
        if name in data:
            return data[name]
        per_lam = self.__dict__.get("per_lam", {})
        if name in per_lam:
            return per_lam[name]
        raise AttributeError(name)

    @property
    def n_lam(self):
        return len(self.lam)

    @property
    def n_modes(self):
        return self.data["neff"].shape[1]

    @property
    def labels(self):
        """(n_lam, n_modes) ASCII labels like 'TE00' from the HG fit ('' if not labeled)."""
        pol = self.data["hg_pol"].astype(str)
        m, n = self.data["hg_m"], self.data["hg_n"]
        return np.where(m >= 0, np.char.add(np.char.add(pol, m.astype(str)), n.astype(str)), "")

    @property
    def geometry(self):
        return RibGeometry(**self.meta["geometry"])

    @property
    def grid(self):
        g = self.meta["grid"]
        return grid_2d(g["Lx"], g["Ly"], g["Nx"], g["Ny"])

    def index_of(self, lam):
        """Index of the stored wavelength nearest to ``lam``."""
        return int(np.argmin(np.abs(self.lam - lam)))

    # ---- combine / reorder ----------------------------------------------------------
    @classmethod
    def concat(cls, parts):
        """Stack sweeps along wavelength (sorted). Refuses parts from a different geometry,
        grid or mode count, and duplicate wavelengths (e.g. stale checkpoint files)."""
        parts = [p for p in parts if p is not None]
        if not parts:
            raise ValueError("concat: no parts (no matching .h5 files?)")
        p0 = parts[0]
        has_E = [p.E is not None for p in parts]
        if any(has_E) and not all(has_E):
            raise ValueError("concat: some parts have fields and some do not "
                             "(load all with fields=True or all with fields=False)")
        for p in parts[1:]:
            for key in ("geometry", "grid"):
                if p.meta.get(key) != p0.meta.get(key):
                    raise ValueError(f"concat: {key} differs between parts "
                                     f"({p.meta.get(key)} vs {p0.meta.get(key)})")
            if p.n_modes != p0.n_modes:
                raise ValueError(f"concat: n_modes differs ({p.n_modes} vs {p0.n_modes})")
        parts = sorted(parts, key=lambda p: float(p.lam[0]))     # sort first: no extra copy
        lam = np.concatenate([p.lam for p in parts])
        if np.any(np.diff(lam) <= 0):
            raise ValueError("concat: duplicate or unsorted wavelengths (parts must not overlap)")
        cat = lambda arrs: np.concatenate(arrs, axis=0)
        data = {k: cat([p.data[k] for p in parts]) for k in p0.data}
        per_lam = {k: cat([p.per_lam[k] for p in parts]) for k in p0.per_lam}
        E = cat([p.E for p in parts]) if all(p.E is not None for p in parts) else None
        return cls(lam, data, per_lam, p0.xs, p0.ys, E, dict(p0.meta))

    @classmethod
    def gather(cls, directory, pattern=".h5", fields=True):
        """Combine single-λ checkpoint files (e.g. one per cluster array task)."""
        files = sorted(f for f in os.listdir(directory) if f.endswith(pattern))
        return cls.concat([cls.load(os.path.join(directory, f), fields=fields) for f in files])

    # ---- I/O ------------------------------------------------------------------------
    def save(self, path, compression="gzip"):
        """HDF5: per-mode datasets (n_lam, n_modes), /lam, /x, /y, /E (optional), and
        JSON metadata (geometry, grid, solver settings, material) in the root attrs."""
        import h5py
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with h5py.File(path, "w") as f:
            f.attrs["format"] = "tfln_modesweep/1"
            f.attrs["layout"] = (
                "per-mode datasets are (n_lam, n_modes), modes at each wavelength sorted by "
                "descending neff; E is (n_lam, n_modes, 3, Nx, Ny) = (Ex, Ey, Ez) in the mpbx lab "
                "frame (x horizontal = crystal Z, y vertical = crystal X, z propagation = crystal Y), "
                "normalized to int Re(E* x H)_z dA = 1 in mpbx natural units (eps0=mu0=c=1); "
                "multiply by sqrt(2*Z0) = 27.45 for Tidy3D's 1 W SI fields in V/um; "
                "x, y are pixel-center coordinates in um")
            f.attrs["meta"] = json.dumps(self.meta)
            f["lam"] = self.lam
            f["x"], f["y"] = self.xs, self.ys
            for k, v in self.data.items():
                f[k] = v.astype("S") if v.dtype.kind in "US" else v
            for k, v in self.per_lam.items():
                f[k] = v
            if self.E is not None:
                f.create_dataset("E", data=self.E, compression=compression,
                                 chunks=(1, 1) + self.E.shape[2:])
        return path

    @classmethod
    def load(cls, path, fields=True):
        import h5py
        with h5py.File(path, "r") as f:
            meta = json.loads(f.attrs["meta"])
            data = {}
            for k in FLOAT_FIELDS + INT_FIELDS + BOOL_FIELDS + STR_FIELDS:
                if k in f:
                    v = f[k][()]
                    data[k] = v.astype(str) if v.dtype.kind == "S" else v
            per_lam = {k: f[k][()] for k in PER_LAM if k in f}
            E = f["E"][()] if (fields and "E" in f) else None
            return cls(f["lam"][()], data, per_lam, f["x"][()], f["y"][()], E, meta)

    def rows(self):
        """One dict per (λ, mode), like Tidy3D's ModeSolverData.to_dataframe()."""
        labels = self.labels
        out = []
        for i, lam in enumerate(self.lam):
            for j in range(self.n_modes):
                r = dict(wavelength=lam, mode_index=j, label=labels[i, j])
                for k, v in self.data.items():
                    x = v[i, j]
                    r[k] = x.item() if hasattr(x, "item") else x
                r["n_cutoff"] = self.per_lam["n_cutoff"][i]
                out.append(r)
        return out

    def to_csv(self, path):
        import csv
        rows = self.rows()
        with open(path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        return path

    def to_dataframe(self):
        """pandas DataFrame indexed by (wavelength, mode_index) (needs pandas installed)."""
        import pandas as pd
        return pd.DataFrame(self.rows()).set_index(["wavelength", "mode_index"])

    def table(self, i=None):
        """Plain-text table of the modes at wavelength index ``i`` (all wavelengths if None)."""
        idx = range(self.n_lam) if i is None else [i]
        labels = self.labels
        lines = []
        for i in idx:
            lines.append(f"λ = {self.lam[i]:.4f} µm   cutoff n = {self.per_lam['n_cutoff'][i]:.4f}   "
                         f"(n_e {self.per_lam['n_e'][i]:.4f}, n_o {self.per_lam['n_o'][i]:.4f}, "
                         f"n_SiO2 {self.per_lam['n_sio2'][i]:.4f})   solve {self.per_lam['solve_time'][i]:.1f} s")
            lines.append("  idx label  neff      ng        TEfrac  HGerr  hg00   LNfrac edge%   edge_dB par guided conv")
            for j in range(self.n_modes):
                g = self.data
                lines.append(
                    f"  {j:3d} {labels[i, j]:5s} {g['neff'][i, j]:.6f}  {g['ng'][i, j]:.6f}  "
                    f"{g['te_frac'][i, j]:6.3f}  {g['hg_err'][i, j]:5.3f}  {g['hg00'][i, j]:5.3f}  "
                    f"{g['ln_frac'][i, j]:5.3f}  {100 * g['edge_frac'][i, j]:6.3f}  {g['edge_db'][i, j]:7.1f} "
                    f"{g['parity'][i, j]:+3d}  {str(bool(g['guided'][i, j])):5s}  {str(bool(g['converged'][i, j]))}")
        return "\n".join(lines)


# --------------------------------------------------------------------------------------
# One wavelength
# --------------------------------------------------------------------------------------
def solve_wavelength(diel: RibDielectric, lam, settings=SolverSettings(), warm=None):
    """All modes of ``diel`` at wavelength ``lam`` [µm]. Returns ``(ModeSweep, warm_state)``.

    ``warm`` is the warm_state returned for a neighbouring wavelength (or None)."""
    geom, grid = diel.geom, diel.grid
    if settings.use_parity and grid.Nx % 2 == 0:
        raise ValueError("use_parity=True needs an odd Nx (make_grid(..., odd_x=True)); on an even "
                         "Nx the mirror projector is inexact and LOBPCG does not converge (mpbx GOTCHAS #33)")
    t0 = time.perf_counter()
    d = diel.at(lam)
    om = d["omega"]
    sectors = (-1, +1) if settings.use_parity else (0,)

    modes, new_warm = [], {}
    for par in sectors:
        kguess = Hguess = None
        if warm is not None and par in warm:
            om_p, k_p, ev_p, ng_p = warm[par]
            kguess = [k + (om - om_p) * g for k, g in zip(k_p, ng_p)]   # k(ω) ≈ k_p + n_g Δω
            Hguess = ev_p
        km, evs, diag = solve_k_checked(om, d["einv"], grid, settings.nev, parity=par,
                                        k_tol=settings.k_tol, eig_tol=settings.eig_tol,
                                        maxiter=settings.maxiter, max_newton=settings.max_newton,
                                        kguess=kguess, Hguess=Hguess)
        ngs = []
        for k, ev, info in zip(km, evs, diag):
            ng = float(An.group_index(float(k), ev, om, d["einv"], d["deps"], grid))
            ngs.append(ng)
            modes.append(dict(k=float(k), ev=ev, ng=ng, parity=par, info=info))
        new_warm[par] = (om, [float(k) for k in km], evs, ngs)
    t_solve = time.perf_counter() - t0

    # ---- post-process every mode ------------------------------------------------------
    modes.sort(key=lambda m: -m["k"])
    xs, ys = grid_coords(grid)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    ln_fill = geom.ln_fill(grid)                        # area-weighted LN fraction per pixel
    ew = settings.edge_width
    edge_mask = ((X - xs[0] < ew) | (xs[-1] - X < ew) | (Y - ys[0] < ew) | (ys[-1] - Y < ew))
    ring = np.zeros_like(edge_mask)
    ring[[0, -1], :] = True
    ring[:, [0, -1]] = True
    n_cut = geom.cutoff_index(lam)

    rec = {k: [] for k in FLOAT_FIELDS + INT_FIELDS + BOOL_FIELDS + STR_FIELDS}
    Es = []
    for m in modes:
        k, ev = m["k"], m["ev"]
        E, _H, P = _power_normalized_fields(k, ev, d, grid)
        Ex2, Ey2 = np.abs(E[0]) ** 2, np.abs(E[1]) ** 2
        I2 = Ex2 + Ey2 + np.abs(E[2]) ** 2
        te = float(Ex2.sum() / (Ex2.sum() + Ey2.sum()))
        w = np.real(np.einsum("a...,ab...,b...->...", np.conj(E), d["eps"], E))   # E*·ε·E
        neff = k / om
        rec["neff"].append(neff)
        rec["ng"].append(m["ng"])
        rec["te_frac"].append(te)
        rec["ln_frac"].append(float((w * ln_fill).sum() / w.sum()))
        rec["edge_frac"].append(float(I2[edge_mask].sum() / I2.sum()))
        rec["edge_db"].append(float(10 * np.log10(max(I2[ring].max() / I2.max(), 1e-30))))
        rec["hg00"].append(hg00_overlap(E[0] if te >= 0.5 else E[1], xs, ys))
        rec["power_raw"].append(P)
        rec["parity"].append(m["parity"])
        rec["newton_iters"].append(m["info"]["newton"])
        rec["lobpcg_iters"].append(m["info"]["lobpcg_iters"])
        rec["guided"].append(bool(neff > n_cut))
        rec["converged"].append(bool(m["info"]["newton_converged"] and m["info"]["lobpcg_converged"]))
        if settings.label_modes:
            pol, hm, hn, herr = label_mode(E, xs, ys, geom, max_order=settings.hg_max_order)
            rec["hg_pol"].append(pol)
            rec["hg_m"].append(hm)
            rec["hg_n"].append(hn)
            rec["hg_err"].append(herr)
        else:
            rec["hg_pol"].append("TE" if te >= 0.5 else "TM")
            rec["hg_m"].append(-1)
            rec["hg_n"].append(-1)
            rec["hg_err"].append(np.nan)
        if settings.store_fields:
            Es.append(E.astype(settings.field_dtype))

    data = {k: np.asarray(v)[None, :] for k, v in rec.items()}
    n_e, n_o = ln_indices(lam)
    per_lam = dict(n_cutoff=[n_cut], n_e=[n_e], n_o=[n_o], n_sio2=[sio2_index(lam)],
                   solve_time=[time.perf_counter() - t0])
    E_all = np.stack(Es)[None] if settings.store_fields else None
    meta = dict(geometry=geom.as_dict(),
                grid=dict(Lx=grid.Lx, Ly=grid.Ly, Nx=grid.Nx, Ny=grid.Ny),
                settings=settings.as_dict(),
                material=dict(LN="mpbjax.materials.MgO_LiNbO3_matsuda (Matsuda 2015, 20 C)",
                              rotation_R=R_XCUT.tolist(), rotation_convention="eps_lab = R^T eps_crystal R",
                              substrate_cladding="mpbjax.materials.SiO2 (Malitson)"),
                created=datetime.datetime.now().isoformat(timespec="seconds"),
                backend=jax.default_backend(),
                devices=[str(dv) for dv in jax.devices()],
                x64=bool(jax.config.jax_enable_x64),
                timing=dict(eigensolve_s=t_solve))
    return ModeSweep([lam], data, per_lam, xs, ys, E_all, meta), new_warm


# --------------------------------------------------------------------------------------
# Wavelength sweep
# --------------------------------------------------------------------------------------
def run_sweep(geom: RibGeometry, grid, lams, settings=SolverSettings(), out=None,
              checkpoint_dir=None, verbose=True):
    """Solve every wavelength in ``lams`` [µm] (processed from long to short so that the
    warm start follows the mode set as it grows). Returns a ModeSweep (λ ascending).

    out            : HDF5 path for the combined result (optional)
    checkpoint_dir : if given, every wavelength is also written there as its own file
                     (crash-safe; the same layout one cluster array task would write;
                     recombine with ModeSweep.gather(checkpoint_dir))
    """
    diel = RibDielectric(geom, grid)
    order = sorted({float(l) for l in np.atleast_1d(lams)}, reverse=True)
    if verbose:
        print(f"[run_sweep] {len(order)} wavelengths, {grid.Nx}x{grid.Ny} px over "
              f"{grid.Lx:g}x{grid.Ly:g} µm, nev={settings.nev}"
              f"{' per parity sector' if settings.use_parity else ''}, backend={jax.default_backend()}")
    parts, warm = [], None
    t_all = time.perf_counter()
    for i, lam in enumerate(order):
        part, new_warm = solve_wavelength(diel, lam, settings,
                                          warm=warm if settings.warm_start else None)
        warm = new_warm
        parts.append(part)
        if checkpoint_dir:
            os.makedirs(checkpoint_dir, exist_ok=True)
            part.save(os.path.join(checkpoint_dir, f"lam_{lam * 1e3:08.2f}nm.h5"))
        if verbose:
            lab = part.labels[0]
            g = part.data
            te00 = [j for j in range(part.n_modes) if lab[j] == "TE00" and g["guided"][0, j]]
            s = (f"TE00 = idx {te00[0]} neff {g['neff'][0, te00[0]]:.5f} ng {g['ng'][0, te00[0]]:.4f}"
                 if te00 else "TE00 not found by label (raise nev?)")
            if te00 and te00[0] == part.n_modes - 1:
                s += "  [TE00 is the LAST solved band: no margin, consider a larger nev]"
            print(f"  [{i + 1}/{len(order)}] λ = {lam:.4f} µm  {part.per_lam['solve_time'][0]:6.1f} s  "
                  f"guided {int(g['guided'].sum())}/{part.n_modes}  "
                  f"max edge {100 * g['edge_frac'][0].max():.2e} %  "
                  f"{'' if g['converged'].all() else 'NOT CONVERGED  '}{s}", flush=True)
    sweep = ModeSweep.concat(parts)
    sweep.meta["timing"] = dict(total_s=time.perf_counter() - t_all)
    if out:
        sweep.save(out)
        if verbose:
            print(f"[run_sweep] saved {out}")
    return sweep


# --------------------------------------------------------------------------------------
# Tracking a mode across wavelength
# --------------------------------------------------------------------------------------
@dataclasses.dataclass
class ModeTrack:
    target: str
    lam: np.ndarray
    idx: np.ndarray            # mode index per wavelength (-1 = not found)
    neff: np.ndarray
    ng: np.ndarray
    score: np.ndarray          # hg00 overlap (for XX00 targets) or 1 - HG fit error
    overlap_prev: np.ndarray   # |<E_i, E_(i-1)>| / norms between consecutive picks (nan if no fields)
    label: np.ndarray          # library HG label of the pick
    flags: list

    def summary(self):
        out = [f"track {self.target}:  λ [µm]   idx  label   neff      ng        score  overlap_prev  flags"]
        for i, lam in enumerate(self.lam):
            out.append(f"  {lam:8.4f}  {self.idx[i]:3d}  {self.label[i]:5s}  {self.neff[i]:.6f}  "
                       f"{self.ng[i]:.6f}  {self.score[i]:.3f}  {self.overlap_prev[i]:8.4f}      "
                       f"{'; '.join(self.flags[i])}")
        return "\n".join(out)


def richardson(q_fine, q_coarse, ratio=2.0, order=2):
    """Grid-converged estimate from two runs whose pitches differ by ``ratio`` (coarse/fine)
    and whose interfaces sit at the SAME sub-pixel position (see tfln_rib.align_to_grid):
    q_inf = q_f + (q_f - q_c) / (ratio**order - 1). Returns (q_inf, |q_inf - q_f|), the
    second being the estimated discretization error of the fine run. Works on arrays,
    e.g. the neff / ng of the same tracked branch from two sweeps."""
    q_f, q_c = np.asarray(q_fine, float), np.asarray(q_coarse, float)
    q_inf = q_f + (q_f - q_c) / (ratio ** order - 1.0)
    return q_inf, np.abs(q_inf - q_f)


def track_mode(sweep: ModeSweep, target="TE00", min_pol=0.5, min_score=0.5, min_overlap=0.8):
    """Pick ``target`` (e.g. "TE00", "TM00", "TE10") at every wavelength.

    1. Candidates: guided modes of the right polarization (te_frac > min_pol for TE,
       < 1 - min_pol for TM).
    2. For the fundamental (m = n = 0): the candidate with the largest moment-matched
       HG00 overlap (must exceed min_score), i.e. the Tidy3D-era algorithm. For other
       targets: candidates whose library HG label equals ``target``, lowest fit error first.
    3. Cross-checks recorded in ``flags``: the library label of the pick, and (if fields
       were stored) the field overlap with the pick at the previous wavelength.
    """
    pol, m, n = target[:2], int(target[2]), int(target[3])
    want_te = pol == "TE"
    labels = sweep.labels
    g = sweep.data
    L = sweep.n_lam
    idx = np.full(L, -1)
    score = np.full(L, np.nan)
    flags = [[] for _ in range(L)]
    for i in range(L):
        pol_ok = (g["te_frac"][i] > min_pol) if want_te else (g["te_frac"][i] < 1 - min_pol)
        cand = np.flatnonzero(pol_ok & g["guided"][i].astype(bool))
        if m == 0 and n == 0:
            s = g["hg00"][i, cand]
            keep = cand[s > min_score]
            if keep.size:
                j = keep[np.argmax(g["hg00"][i, keep])]
                idx[i], score[i] = j, g["hg00"][i, j]
        else:
            keep = [j for j in cand if labels[i, j] == target]
            if keep:
                j = min(keep, key=lambda jj: g["hg_err"][i, jj])
                idx[i], score[i] = j, 1.0 - g["hg_err"][i, j]
        if idx[i] < 0:
            flags[i].append("not found")
        elif labels[i, idx[i]] and labels[i, idx[i]] != target:
            flags[i].append(f"HG-fit label is {labels[i, idx[i]]}")
    overlap = np.full(L, np.nan)
    if sweep.E is not None:
        for i in range(1, L):
            a, b = idx[i - 1], idx[i]
            if a >= 0 and b >= 0:
                Ea = sweep.E[i - 1, a, :2].astype(complex)
                Eb = sweep.E[i, b, :2].astype(complex)
                overlap[i] = abs(np.vdot(Ea, Eb)) / (np.linalg.norm(Ea) * np.linalg.norm(Eb))
                if overlap[i] < min_overlap:
                    flags[i].append(f"low overlap with previous λ ({overlap[i]:.2f})")
    sel = idx >= 0
    take = lambda arr: np.where(sel, arr[np.arange(L), np.maximum(idx, 0)], np.nan)
    lab = np.where(sel, labels[np.arange(L), np.maximum(idx, 0)], "")
    return ModeTrack(target, sweep.lam.copy(), idx, take(g["neff"]), take(g["ng"]), score,
                     overlap, lab, flags)
