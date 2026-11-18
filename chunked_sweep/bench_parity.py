"""Does use_parity save time, or is it enough to pick FFT-friendly grid sizes?

    python bench_parity.py              both parts (~45 min on the 8 P-cores of an i7-12700)
    python bench_parity.py --part fft   part 1 only (~5 min)
    python bench_parity.py --part solve part 2 only

Part 1 (FFT sizes): time per LOBPCG iteration (fixed iteration count, tol = 0) of the
jit-compiled eigensolve on the chunk-1 window (~4 x 4 µm at 12.5 nm) for Nx = 320 (2^6·5),
319 (11·29), 321 (3·107, what "320 + 1 for parity" gives), 325 (5^2·13), all with Ny = 320,
nev = 6, without parity and (odd Nx only) with parity.

Part 2 (complete solves): solve_wavelength at λ = 0.50 µm (cold) and then 0.49 µm (warm start),
production settings, for
  * no parity, Nx = 320, nev = 12            (what chunk 1 would run with use_parity=False)
  * parity,    Nx = 325, nev = 6 per sector  (the same number of modes)
and reports wall time, Newton steps and LOBPCG iterations, and whether both found the same modes.

The process is pinned to `--cores` (default: one logical CPU per P-core of the i7-12700) before
JAX is imported. Results -> results/bench_parity.json
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def parse_cores(s):
    out = []
    for part in s.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b) + 1)) if b else [int(a)]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=("fft", "solve", "all"), default="all")
    ap.add_argument("--cores", default=None, help="e.g. 0-7 (default: 0,2,...,14 on Windows, all on Linux)")
    ap.add_argument("--iters", type=int, default=30)
    a = ap.parse_args()
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:                                                  # noqa: BLE001
            pass

    if a.cores:
        cores = parse_cores(a.cores)
    elif hasattr(os, "sched_getaffinity"):
        cores = sorted(os.sched_getaffinity(0))
    else:
        cores = list(range(0, 16, 2))
    if hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, cores)
    else:
        import psutil
        psutil.Process().cpu_affinity(cores)

    sys.path[:0] = [ROOT, os.path.join(ROOT, "handoff_tfln_modesolve")]
    from utils.setup_jax import setup_jax
    setup_jax("cpu", verbose=False)
    import jax
    import jax.numpy as jnp
    import numpy as np
    from mpbjax import grid_2d
    from geometry_gen import tfln_rib as R
    import tfln_modes as M

    out_path = os.path.join(HERE, "results", "bench_parity.json")
    res = json.load(open(out_path)) if os.path.exists(out_path) else {}
    res["cores"] = cores
    d = 0.0125
    Ny = 320

    def make(Nx):
        grid = grid_2d(Nx * d, Ny * d, Nx, Ny)
        geom = R.align_to_grid(R.RibGeometry(), grid, frac=1 / 3, warn=False)
        return grid, R.RibDielectric(geom, grid)

    if a.part in ("fft", "all"):
        rows = []
        for Nx in (320, 319, 321, 325):
            grid, diel = make(Nx)
            dd = diel.at(0.5)
            einv = jnp.asarray(dd["einv"])
            k = jnp.asarray(2.2 / 0.5)                     # mpbx units: k = n_eff / λ
            key = jax.random.PRNGKey(0)
            for par in ((0, -1) if Nx % 2 else (0,)):
                kw = dict(grid=grid, nev=6, tol=0.0, maxiter=a.iters, parity=par)
                t0 = time.perf_counter()
                jax.block_until_ready(M._eigensolve(k, einv, None, key, **kw))   # compile + run
                t_first = time.perf_counter() - t0
                t0 = time.perf_counter()
                jax.block_until_ready(M._eigensolve(k, einv, None, key, **kw))
                t = (time.perf_counter() - t0) / a.iters
                rows.append(dict(Nx=Nx, Ny=Ny, parity=par, nev=6, s_per_iter=t, first_call_s=t_first))
                print(f"Nx {Nx:4d}  parity {par:+d}  {1e3 * t:7.1f} ms/iter   (first call incl. compile {t_first:5.1f} s)",
                      flush=True)
            jax.clear_caches()
        res["fft"] = rows
        json.dump(res, open(out_path, "w"), indent=1)

    if a.part in ("solve", "all"):
        rows = []
        for Nx, par, nev in ((325, True, 6), (320, False, 12)):
            grid, diel = make(Nx)
            st = M.SolverSettings(nev=nev, use_parity=par, warm_start=True, label_modes=False, store_fields=False)
            warm = None
            for lam in (0.50, 0.49):
                t0 = time.perf_counter()
                sw, warm = M.solve_wavelength(diel, lam, st, warm=warm)
                t = time.perf_counter() - t0
                g = sw.data
                row = dict(Nx=Nx, Ny=Ny, parity=par, nev=nev, lam=lam, warm=lam != 0.50, t=t,
                           eig_s=sw.meta["timing"]["eigensolve_s"],
                           neff=g["neff"][0].tolist(), te_frac=g["te_frac"][0].tolist(),
                           newton=g["newton_iters"][0].tolist(), lobpcg=g["lobpcg_iters"][0].tolist(),
                           converged=bool(g["converged"].all()))
                rows.append(row)
                print(f"Nx {Nx}  parity {par!s:5s}  nev {nev:2d}  λ {lam:.2f} {'warm' if row['warm'] else 'cold'}: "
                      f"{t:7.1f} s (eigensolve {row['eig_s']:.1f} s)  Newton/mode {np.mean(row['newton']):.1f}  "
                      f"LOBPCG it/solve median {np.median(row['lobpcg']):.0f}  converged {row['converged']}", flush=True)
                res["solve"] = rows
                json.dump(res, open(out_path, "w"), indent=1)
            jax.clear_caches()
        for lam in (0.50, 0.49):
            a_, b_ = [r for r in rows if r["lam"] == lam]
            na, nb = np.array(a_["neff"]), np.array(b_["neff"])
            print(f"λ {lam}: max |Δn_eff| between the two mode sets {np.max(np.abs(np.sort(na) - np.sort(nb))):.2e} "
                  f"(different Nx: expect ~1e-5 from the window/pixel change, not a different mode)")
    print(f"-> {out_path}")


if __name__ == "__main__":
    main()
