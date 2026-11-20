"""Why does LOBPCG not converge on the GPU? One small eigensolve, taken apart.

    python gpu_sweep/diag_lobpcg_gpu.py                   everything on the CPU backend (sanity run, ~1 min)
    python gpu_sweep/diag_lobpcg_gpu.py --platform cuda   the GPU against the CPU (SuperCloud: --diag job)

Background: on the first V100 run (2026-11-20, `--check`), every LOBPCG eigensolve of the batched
path with JAX's default small eigensolver ran to its 1000-iteration cap without reaching the 1e-8
residual (the CPU needs ~50), n_eff came out 2.5e-5 off, and the "qr" variant did not finish the
same two wavelengths in 23 minutes. This script finds out where the GPU loses the convergence.
(Its first GPU run found it: mpbjax's 1e12 penalty on null directions in the Rayleigh-Ritz step
defeats cuSOLVER's norm-relative small eigensolvers. mpbjax now scales the penalty to the kept
block (eigensolver._null_penalty); the header line "mpbjax RR penalty" says which one is installed.)

Same geometry, grid (the --test grid, 105 x 120 px), wavelength and number of bands as --check, at
ONE fixed k (the Newton starting value), ONE parity sector, cold start; no Newton iteration, no
batching (vmap), no post-processing. Three parts:

1. Building blocks, GPU vs CPU on identical inputs: FFT, the Helmholtz operator, the
   preconditioner, a block Gram product, and the operator's Hermiticity on the GPU. All should
   agree to ~1e-15 relative; a much larger number points at the culprit.
2. The small dense eigensolves inside LOBPCG, on the REAL matrices of a GPU run (recorded): the
   whitening Gram matrices ZᴴZ and the Rayleigh-Ritz matrices (with mpbjax's penalty on null
   directions). Each is solved by cuSOLVER ("default", "qr", "jacobi") and by LAPACK (reference).
3. The residual history of the whole LOBPCG solve (the same lobpcg code; the LOBPCG cap is raised
   step by step: 1, 2, 5, ... iterations from the same start), for: LAPACK on the CPU backend, and
   on the GPU with each small eigensolver, including "host" (LAPACK through a host callback).

The last lines give a verdict: which GPU variants converge, and what that implies.
Exit status 0 if any GPU variant converges like the CPU, 1 if none does.
"""
import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [p for p in (HERE, os.path.join(ROOT, "chunked_sweep"), ROOT) if p not in sys.path]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--platform", default="cpu", choices=("cpu", "cuda"))
    ap.add_argument("--lam", type=float, default=0.60)
    ap.add_argument("--nev", type=int, default=4, help="bands per sector (as --check)")
    ap.add_argument("--sign", type=int, default=+1, choices=(-1, 1), help="x-parity sector")
    ap.add_argument("--caps", default="1,2,5,10,20,50,100,200,500,1000",
                    help="LOBPCG iteration caps of the residual history (part 3)")
    ap.add_argument("--variants", default="cpu,default,qr,jacobi,host", help="part 3 variants")
    ap.add_argument("--n-record", type=int, default=60, help="small matrices recorded for part 2")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    from utils.setup_jax import setup_jax
    if a.platform == "cuda":
        os.environ["JAX_PLATFORMS"] = "cuda,cpu"                  # GPU default, CPU for the references
        setup_jax(None, verbose=True)
    else:
        setup_jax("cpu", verbose=True)
    import jax
    import jax.numpy as jnp
    import numpy as np
    import batched_modes as BM
    import mpbjax.eigensolver as E
    import run_geometry_gpu as RGG
    import sweep_config as C
    from geometry_gen import tfln_rib as R
    from mpbjax import grid_2d
    from mpbjax import modes as Mo

    dev = jax.devices()[0]
    cpu = jax.devices("cpu")[0]
    on_gpu = dev.platform != "cpu"
    print(f"device under test: {dev} ({dev.platform}); reference: {cpu}")
    print(f"mpbjax RR penalty: {'scaled to the kept block (_null_penalty)' if hasattr(E, '_null_penalty') else '1e12 wall (old mpbjax: cuSOLVER will stall)'}")
    if on_gpu:
        print(f"  {getattr(dev, 'device_kind', '?')}")

    entry = dict(name="diag", geometry=dict(t_ln=0.4, etch_depth=0.4, w_top=1.8, sidewall_deg=73.0, t_clad=1.35))
    plan = RGG.gpu_plan(entry, "gpu", True, 0.05)               # the --check / --test grid
    g = plan["chunks"][0]["grid"]
    grid = grid_2d(g["Lx"], g["Ly"], g["Nx"], g["Ny"])
    geom = R.align_to_grid(R.RibGeometry(**entry["geometry"]), grid, frac=C.ALIGN_FRAC, warn=False)
    d = R.RibDielectric(geom, grid).at(a.lam)
    einv_np = np.asarray(d["einv"])
    omega = float(d["omega"])
    k = Mo.k_guess(omega, einv_np)
    nev, tol = a.nev, C.EIG_TOL
    print(f"grid {g['Nx']}x{g['Ny']} ({g['Lx']:.3f} x {g['Ly']:.3f} µm), λ {a.lam} µm, k {k:.6f} µm⁻¹ (Newton start), "
          f"{nev} bands, sector {a.sign:+d}, tol {tol:g}")

    put = jax.device_put
    rng = np.random.default_rng(0)

    def rel(x, ref):
        x, ref = np.asarray(x), np.asarray(ref)
        return float(np.max(np.abs(x - ref)) / max(np.max(np.abs(ref)), 1e-300))

    flag = lambda v, lim: "" if v <= lim else "   <-- LARGE"

    # ---- 1. building blocks --------------------------------------------------------------
    print("\n1. building blocks, device vs CPU (relative max difference; ~1e-15 expected)")
    from mpbjax.helmholtz import eps_ave_from_inv, helmholtz_apply, mag_mn, precond_apply
    sh = (2, g["Nx"], g["Ny"])
    H = rng.standard_normal(sh) + 1j * rng.standard_normal(sh)
    H2 = rng.standard_normal(sh) + 1j * rng.standard_normal(sh)
    F = rng.standard_normal(sh[1:]) + 1j * rng.standard_normal(sh[1:])
    V = rng.standard_normal((2 * g["Nx"] * g["Ny"], 3 * nev)) + 1j * rng.standard_normal((2 * g["Nx"] * g["Ny"], 3 * nev))

    def op(Hx, ei, kk):
        mag, mn = mag_mn(kk, grid)
        return helmholtz_apply(Hx, ei, mag, mn)

    def pc(Hx, ei, kk):
        mag, mn = mag_mn(kk, grid)
        return precond_apply(Hx, eps_ave_from_inv(ei), mag, mn)

    opj, pcj = jax.jit(op), jax.jit(pc)
    gram = jax.jit(lambda A: A.conj().T @ A)
    fft = jax.jit(jnp.fft.fft2)
    res1 = {}
    for name, f, args in (("fft2", fft, (F,)),
                          ("Helmholtz operator", opj, (H, einv_np, k)),
                          ("preconditioner", pcj, (H, einv_np, k)),
                          ("block Gram VᴴV", gram, (V,))):
        out_d = f(*[put(x, dev) if isinstance(x, np.ndarray) else x for x in args])
        out_c = f(*[put(x, cpu) if isinstance(x, np.ndarray) else x for x in args])
        res1[name] = rel(out_d, out_c)
        print(f"  {name:22s} {res1[name]:.2e}{flag(res1[name], 1e-12)}")
    fft_np = rel(fft(put(F, cpu)), np.fft.fft2(F))
    print(f"  {'fft2 (CPU vs numpy)':22s} {fft_np:.2e}{flag(fft_np, 1e-12)}")
    Ay, Ax = opj(put(H2, dev), put(einv_np, dev), k), opj(put(H, dev), put(einv_np, dev), k)
    lhs = complex(jnp.vdot(put(H, dev), Ay))
    rhs = complex(jnp.vdot(Ax, put(H2, dev)))
    herm = abs(lhs - rhs) / max(abs(lhs), 1e-300)
    print(f"  {'Hermiticity on device':22s} {herm:.2e}{flag(herm, 1e-12)}   (|<x,Ay> - <Ax,y>| / |<x,Ay>|)")

    # ---- LOBPCG runner -------------------------------------------------------------------
    sign = a.sign

    def run(cap, device):
        """One cold LOBPCG solve with at most ``cap`` iterations, on ``device`` (the same random start
        block as tfln_modes' and the batched path's cold solves: PRNGKey(0))."""
        def f(ei):
            key = jax.random.PRNGKey(0)                        # made inside: lives on ``device``
            _, _, r = Mo.solve_omega2(k, ei, grid, nev=nev, tol=tol, maxiter=cap, key=key, parities={0: sign})
            return r.eigenvalues, r.residuals, r.iterations, r.converged
        with jax.default_device(device):
            out = jax.jit(f)(put(einv_np, device))
            return [np.asarray(x) for x in jax.device_get(out)]

    def set_variant(v):
        if v in ("cpu", "default"):
            BM.set_eigh("default")
        else:
            BM.set_eigh(v)
        jax.clear_caches()

    # ---- 2. the small eigensolves on real LOBPCG matrices ----------------------------------
    print(f"\n2. small dense eigensolves on the first {a.n_record} matrices of a LOBPCG run on the device "
          f"(recorded through a host callback)")
    rec = []

    def rec_eigh(m):
        m = np.asarray(m)
        if len(rec) < a.n_record:
            rec.append(m.copy())
        return np.linalg.eigh(m)

    E.jnp = BM._JnpProxy(BM._HostLinalgProxy(rec_eigh))
    jax.clear_caches()
    run(max(1, a.n_record // 3), dev)
    BM.set_eigh("default")
    jax.clear_caches()
    p = nev
    kinds = {"p x p (start-up)": [], "Gram ZᴴZ (whitening)": [], "Rayleigh-Ritz (penalized)": []}
    n_big = 0
    for m in rec:
        if m.shape[-1] != 3 * p:
            kinds["p x p (start-up)"].append(m)
        else:
            (kinds["Gram ZᴴZ (whitening)"] if n_big % 2 == 0 else kinds["Rayleigh-Ritz (penalized)"]).append(m)
            n_big += 1
    impls = [("default", None)]
    if on_gpu:
        impls += [("qr", jax.lax.linalg.EighImplementation.QR), ("jacobi", jax.lax.linalg.EighImplementation.JACOBI)]
    print(f"  {'matrix kind':28s} {'n':>3s} " + " ".join(f"{nm + ' Δλ/|λ|':>16s} {nm + ' resid':>14s}" for nm, _ in impls))
    res2 = {}
    for kind, ms in kinds.items():
        if not ms:
            continue
        row = []
        for nm, impl in impls:
            e_val = e_res = 0.0
            for m in ms:
                w_ref = np.linalg.eigh(m)[0]
                md = put(m, dev)
                if impl is None:
                    w, vv = jnp.linalg.eigh(md)
                else:
                    vv, w = jax.lax.linalg.eigh(md, implementation=impl)
                w, vv = np.asarray(w), np.asarray(vv)
                sel = np.argsort(w_ref)[:p] if "Rayleigh" in kind else np.flatnonzero(w_ref > 1e-7 * w_ref.max())
                scale = np.maximum(np.abs(w_ref[sel]), 1e-300)
                e_val = max(e_val, float(np.max(np.abs(np.sort(w)[sel] - w_ref[sel]) / scale)))
                r = m @ vv - vv * w[None, :]
                e_res = max(e_res, float(np.max(np.linalg.norm(r[:, sel], axis=0) / scale)))
            res2[(kind, nm)] = (e_val, e_res)
            row.append(f"{e_val:16.2e} {e_res:14.2e}")
        print(f"  {kind:28s} {len(ms):3d} " + " ".join(row))
    print("  (Δλ/|λ|: eigenvalue error relative to the eigenvalue, for the eigenvalues LOBPCG uses: the p")
    print("   smallest Ritz values / the Gram eigenvalues above its 1e-7 cut. resid: |A v - λ v| / |λ|.")
    print("   For scale: the CPU backend against itself (LAPACK vs LAPACK) gives 0 / <= 2e-13 here.)")

    # ---- 3. residual histories -----------------------------------------------------------
    caps = sorted({int(x) for x in a.caps.split(",")})
    variants = [v.strip() for v in a.variants.split(",") if v.strip()]
    if not on_gpu:
        variants = [v for v in variants if v in ("cpu", "host")] or ["cpu"]
    print(f"\n3. LOBPCG residual history (max relative residual over the {nev} bands after N iterations; "
          f"converged at < {tol:g})")
    hist, final = {}, {}
    for v in variants:
        device = cpu if v == "cpu" else dev
        set_variant(v)
        rows, t0 = [], time.perf_counter()
        for cap in caps:
            w, r, it, conv = run(cap, device)
            rows.append((cap, int(it), float(np.max(r)), bool(conv), w))
            if conv:                                           # later caps would repeat the same solve
                break
        dt = time.perf_counter() - t0
        hist[v] = rows
        final[v] = rows[-1]
        cells = "  ".join(f"{c}:{rr:.1e}" for c, _, rr, _, _ in rows)
        cap, it, rr, conv, w = rows[-1]
        print(f"  {v + (' (CPU backend)' if v == 'cpu' else ''):22s} {cells}")
        print(f"  {'':22s} -> {'converged in ' + str(it) + ' iterations' if conv else 'NOT converged after ' + str(it)}"
              f", min residual {min(x[2] for x in rows):.1e}  ({dt:.1f} s incl. compiles)")
    BM.set_eigh("default")
    jax.clear_caches()

    ref = final.get("cpu")
    if ref is not None:
        print("\n  eigenvalues ω² vs the CPU run (relative):")
        for v in variants:
            if v != "cpu":
                print(f"  {v:22s} {rel(final[v][4], ref[4]):.2e}")

    # ---- verdict -------------------------------------------------------------------------
    print("\nVERDICT")
    gpu_vars = [v for v in variants if v != "cpu"]
    ok = [v for v in gpu_vars if final[v][3]]
    bad = [v for v in gpu_vars if not final[v][3]]
    cpu_it = ref[1] if ref is not None else None
    print(f"  CPU: {'converged in ' + str(cpu_it) if ref is not None and ref[3] else 'did NOT converge'}; "
          f"device variants converged: {ok or 'none'}; not converged: {bad or 'none'}")
    big1 = [n for n, x in res1.items() if x > 1e-12] + (["Hermiticity"] if herm > 1e-12 else [])
    if big1:
        print(f"  Building blocks differ from the CPU beyond rounding: {big1}. Look there first: LOBPCG")
        print("  cannot reach a 1e-8 residual if the operator itself is that inexact.")
    if not on_gpu:
        print("  (CPU-only sanity run: every variant should converge in the same number of iterations.)")
    elif "host" in ok and any(v in bad for v in ("default", "qr", "jacobi")):
        print("  The GPU converges once the small eigensolves are done by LAPACK on the host: cuSOLVER's")
        print("  small eigensolver is the cause. Fix: gpu_config.GPU_EIGH = \"host\", then re-run --check.")
    elif not ok:
        print("  No GPU variant converges, not even with LAPACK for the small eigensolves: the cause is in")
        print("  the large-vector part (operator, preconditioner, projections). See part 1 and the residual")
        print("  floor above.")
    elif set(ok) >= {"default", "qr"}:
        print("  LOBPCG converges on the GPU with cuSOLVER too" + (" (mpbjax's scaled penalty)." if hasattr(E, "_null_penalty") else "."))
        print("  gpu_config.GPU_EIGH may then be \"default\" or \"qr\" (no host round trips): set it, run --check,")
        print("  and let the benchmark (bench_gpu.py --eigh) decide between them and \"host\" on speed.")
        print("  If --check still fails, the cause is in the batched path (vmap, Newton loop, warm starts).")
    else:
        print(f"  Mixed: {ok} converge, {bad} do not. Use a converging variant for GPU_EIGH and re-run --check.")
    sys.exit(0 if (ok or not on_gpu) else 1)


if __name__ == "__main__":
    main()
