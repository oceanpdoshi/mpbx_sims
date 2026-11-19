"""Check the GPU code path against the CPU reference on a small grid.

    python gpu_sweep/check_gpu_local.py                  both on the CPU backend (any machine, ~1-2 min)
    python gpu_sweep/check_gpu_local.py --platform cuda  reference on the CPU, batched path on the GPU
                                                         (the first thing to run on SuperCloud: --check job)

1. A warm-start chain of two wavelengths, solved by sim_core/tfln_modes.solve_wavelength (+ the
   node-count labels of run_geometry.node_labels) and by batched_modes (both sectors in one batch):
   n_eff, n_g, every diagnostic, the labels and the mode count must agree (LOBPCG iteration counts
   are printed: they may differ by a few between CPU and GPU, where sums are ordered differently).
2. Two cold wavelengths in ONE batched call (4 slots) against each solved alone (2 slots).
Exit status 0 = all checks passed.
"""
import argparse
import contextlib
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [p for p in (HERE, os.path.join(ROOT, "chunked_sweep"), ROOT) if p not in sys.path]

TOL = dict(neff=1e-7, ng=1e-6, te_frac=1e-6, ln_frac=1e-6, edge_frac=1e-6, edge_db=1e-3, hg00=1e-6, power_raw=1e-6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--platform", default="cpu", choices=("cpu", "cuda"))
    ap.add_argument("--nev", type=int, default=4)
    ap.add_argument("--lams", default="0.60,0.55")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    from utils.setup_jax import setup_jax
    if a.platform == "cuda":                                  # both backends in this process: GPU default,
        os.environ["JAX_PLATFORMS"] = "cuda,cpu"              # the reference pinned to the CPU below
        os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.5")
        setup_jax(None, verbose=True)
    else:
        setup_jax("cpu", verbose=True)
    import jax
    import numpy as np
    on_cpu = (jax.default_device(jax.devices("cpu")[0]) if a.platform == "cuda" else contextlib.nullcontext())
    print(f"reference on {jax.devices('cpu')[0]}, batched path on {jax.devices()[0]}")
    import batched_modes as BM
    import run_geometry as RG
    import run_geometry_gpu as RGG
    import sim_core.tfln_modes as M
    import sweep_config as C
    from geometry_gen import tfln_rib as R
    from mpbjax import grid_2d

    RG._W["np"] = np                                          # run_geometry.node_labels reads numpy from there
    entry = dict(name="check", geometry=dict(t_ln=0.4, etch_depth=0.4, w_top=1.8, sidewall_deg=73.0, t_clad=1.35))
    plan = RGG.gpu_plan(entry, "gpu", True, 0.05)
    g = plan["chunks"][0]["grid"]
    nev = a.nev
    lams = [float(x) for x in a.lams.split(",")]
    print(f"grid {g['Nx']}x{g['Ny']} ({g['Lx']:.3f} x {g['Ly']:.3f} µm), nev {nev}/sector, λ {lams}")

    # ---- reference: tfln_modes (on the CPU) ------------------------------------------------
    with on_cpu:
        grid = grid_2d(g["Lx"], g["Ly"], g["Nx"], g["Ny"])
        geom = R.align_to_grid(R.RibGeometry(**entry["geometry"]), grid, frac=C.ALIGN_FRAC, warn=False)
        diel = R.RibDielectric(geom, grid)
        st = M.SolverSettings(nev=nev, use_parity=True, warm_start=True, label_modes=False, store_fields=True,
                              field_dtype=C.FIELD_DTYPE, k_tol=C.K_TOL, eig_tol=C.EIG_TOL)
        ref, warm = [], None
        t0 = time.perf_counter()
        for lam in lams:
            sw, warm = M.solve_wavelength(diel, lam, st, warm=warm)
            ref.append((sw, RG.node_labels(sw.E[0], sw.data["te_frac"][0])))
        t_ref = time.perf_counter() - t0

    # ---- batched path ----------------------------------------------------------------------
    def batched(eigh):
        BM.set_eigh(eigh)
        jax.clear_caches()                                    # retrace with this eigensolver
        ctx = BM.GridContext(entry["geometry"], g, C.ALIGN_FRAC, True, C.K_TOL, C.EIG_TOL, C.FIELD_DTYPE,
                             headroom_gb=0.0)
        got, warm = [], None
        t0 = time.perf_counter()
        for lam in lams:
            ld = ctx.lam_data(lam)
            res, stats = ctx.solve_slots([BM.Slot(ld, s, (warm or {}).get(s)) for s in ctx.sectors], nev)
            sw, labs, warm, _ = ctx.build_sweep(ld, res, nev, 0.0)
            got.append((sw, labs, stats))
        print(f"reference {t_ref:.1f} s, batched path (LOBPCG eigh '{eigh}') {time.perf_counter() - t0:.1f} s "
              f"(incl. compiles)")
        return ctx, got

    if a.platform == "cuda":                                  # information: JAX's default small eigensolver
        _, got_d = batched("default")                         # (cuSOLVER Jacobi) vs QR, iterations vs the CPU
        for (sr, _), (sg, _, _) in zip(ref, got_d):
            print(f"  eigh 'default', λ = {sr.lam[0]:.3f}: max |Δn_eff| "
                  f"{np.max(np.abs(sr.data['neff'][0] - sg.data['neff'][0])) if sr.n_modes == sg.n_modes else float('nan'):.2e}"
                  f"; LOBPCG iterations CPU {int(sr.data['lobpcg_iters'].sum())}, GPU {int(sg.data['lobpcg_iters'].sum())}")
    ctx, got = batched("qr")                                  # gpu_config.GPU_EIGH: what production uses

    bad = 0
    for (sr, lr), (sg, lg, stt) in zip(ref, got):
        lam = sr.lam[0]
        print(f"\nλ = {lam:.3f} µm: {sr.n_modes} vs {sg.n_modes} modes, batches {stt['batches']}")
        if sr.n_modes != sg.n_modes:
            print("  MODE COUNT DIFFERS")
            bad += 1
            continue
        for k, tol in TOL.items():
            # power_raw: compare |P|. Its sign follows which of two mirror-symmetric pixels wins the
            # argmax in mpbjax's e_field vs h_field phase fixing (a rounding tie); the CPU path
            # sometimes stores -P for an odd mode. Only |P| is used (normalization).
            f = np.abs if k == "power_raw" else (lambda x: x)
            d = np.max(np.abs(f(sr.data[k][0]) - f(sg.data[k][0])))
            flag = "" if d <= tol else "   <-- FAIL"
            bad += bool(flag)
            print(f"  {k:10s} max |Δ| {d:.2e}{flag}")
        for k in ("parity", "guided", "converged"):
            same = np.array_equal(sr.data[k][0], sg.data[k][0])
            bad += not same
            print(f"  {k:10s} {'same' if same else 'DIFFERENT  <-- FAIL'}")
        it_r, it_g = sr.data["newton_iters"][0], sg.data["newton_iters"][0]
        lb_r, lb_g = sr.data["lobpcg_iters"][0], sg.data["lobpcg_iters"][0]
        print(f"  newton iters ref {[int(x) for x in it_r]} / batched {[int(x) for x in it_g]};  "
              f"lobpcg ref {[int(x) for x in lb_r]} / batched {[int(x) for x in lb_g]} "
              f"(sums {int(lb_r.sum())} / {int(lb_g.sum())})")
        if list(lr) != list(lg):
            print(f"  LABELS DIFFER: ref {lr}\n                 got {lg}   <-- FAIL")
            bad += 1
        else:
            print(f"  labels same: {[f'{p}{m}{n}' for p, m, n in lr]}")

    # ---- 2 wavelengths in one batched call vs alone -----------------------------------------
    lds = [ctx.lam_data(l) for l in lams]
    both, st_b = ctx.solve_slots([BM.Slot(ld, s) for ld in lds for s in ctx.sectors], nev)
    print(f"\nbatched cold solve of {lams}: slot batches {st_b['batches']}")
    for i, ld in enumerate(lds):
        alone, _ = ctx.solve_slots([BM.Slot(ld, s) for s in ctx.sectors], nev)
        for j in range(2):
            dk = np.max(np.abs(alone[j]["k"] - both[2 * i + j]["k"])) / ld.omega
            dng = np.max(np.abs(alone[j]["ng"] - both[2 * i + j]["ng"]))
            flag = "" if dk <= TOL["neff"] and dng <= TOL["ng"] else "   <-- FAIL"
            bad += bool(flag)
            print(f"  λ {ld.lam:.3f} sector {alone[j]['sign']:+d}: max |Δn_eff| {dk:.2e}, |Δn_g| {dng:.2e}{flag}")
    print(f"\n{'ALL CHECKS PASSED' if not bad else f'{bad} CHECK(S) FAILED'}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
