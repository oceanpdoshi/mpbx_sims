"""Quick GPU benchmark: time a few wavelengths per chunk on ONE GPU, project the full 0.4-2.0 µm
geometry on a 2-GPU node, and compare with a measured CPU run. Answers "is the GPU node >= 6x
faster?" in ~1 h instead of a full run (README §3).

    python gpu_sweep/bench_gpu.py --out DIR --cpu-run CPU_GEOMETRY_DIR                       the GPU grids (gpu_config)
    python gpu_sweep/bench_gpu.py --out DIR --cpu-run CPU_GEOMETRY_DIR --grids cpu-run        the CPU run's own grids
    python gpu_sweep/bench_gpu.py --out DIR --nev-scan 3,4,6 --scan-lam 1.5                   cost of more modes per sector
    python gpu_sweep/bench_gpu.py --out DIR --lanes-scan 1,2,3 --chunks D                     does batching more λ help?
  options
    --n-lam 3          wavelengths per lane (the first is solved cold, the rest warm-started)
    --chunks A,C       only these chunks (first letter or full name)
    --max-slots 1      one sector solve per call (no lockstep): compare with the default
    --eigh default     LOBPCG's small eigensolves as JAX ships them (gpu_config.GPU_EIGH = "qr" otherwise)
    --platform cpu     run on the CPU backend (to test this script; the timings are then meaningless)
Every timed unit is preceded by an untimed warm-up with the same batch shape, so compiles and
first-call costs stay out of the per-λ times; the projection adds the measured compile time once.

--grids gpu (default): gpu_config.GPU_CHUNKS (10 x 10 µm, 6.25 nm). The projected wall time of the
full run on N_GPUS GPUs is compared with the CPU run's measured wall time (from its job log, or
--cpu-hours): the 6x question at the GPU grids.
--grids cpu-run: the CPU run's exact grids and modes. Per chunk, node against node: the CPU node
solves 48 wavelengths at a time (its per-λ times, from its _modes.h5, were measured with all 48
workers busy) and the GPU node 2 x (lanes) at a time: an equal-work comparison.
Results: printed, and <out>/bench_<grids>.json.
"""
import argparse
import copy
import glob
import json
import math
import os
import shutil
import sys
import time
import types

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [p for p in (HERE, os.path.join(ROOT, "chunked_sweep"), ROOT) if p not in sys.path]
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import gpu_config as G  # noqa: E402
import run_geometry as RG  # noqa: E402
import run_geometry_gpu as RGG  # noqa: E402
import sweep_config as C  # noqa: E402

BASELINE = dict(t_ln=0.4, etch_depth=0.4, w_top=1.8, sidewall_deg=73.0, t_clad=1.35)   # the 2.44 h CPU run


def init_in_process(platform, max_slots=None):
    """What run_geometry_gpu._init_worker does, in this process (one GPU: CUDA_VISIBLE_DEVICES)."""
    if platform == "cuda":
        os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", str(G.MEM_FRACTION))
    dev = os.environ.get("CUDA_VISIBLE_DEVICES", "0") if platform == "cuda" else "cpu0"
    return RGG.init_jax(platform, dev, max_slots, verbose=True)


def plan_from_cpu_run(cpu_dir):
    """A run_geometry_gpu plan with the chunks, grids and settings of a finished CPU run."""
    import results_io as rio
    _, meta = rio.load_modes(cpu_dir)
    setup = meta.get("setup", {})
    chunks = []
    objs = [types.SimpleNamespace(**c) for c in meta["chunks"]]
    for c, o in zip(meta["chunks"], objs):
        lams = RG.chunk_lams(o, meta["step"])
        chunks.append(dict(c, targets=list(c["targets"]), lams=lams,
                           owned=[l for l in lams if RG.owner_chunk(l, objs) == c["name"]]))
    return dict(name=meta["name"], geometry=meta["geometry"], step=meta["step"], fields="none",
                use_parity=meta["use_parity"], align_frac=meta["align_frac"], k_tol=meta["k_tol"],
                eig_tol=meta["eig_tol"], nev_extra=setup.get("nev_extra", C.NEV_EXTRA),
                field_dtype=setup.get("field_dtype", C.FIELD_DTYPE), min_margin=C.MIN_MARGIN, lams_subset=None,
                chunks=chunks, warnings=[], engine="gpu (jax) bench", chunk_set="cpu-run")


def cpu_chunk_times(cpu_dir):
    """{chunk: [per-λ solve_time]} of a CPU run (measured with the node full of workers)."""
    import h5py
    import numpy as np
    import results_io as rio
    _, meta = rio.load_modes(cpu_dir)
    with h5py.File(os.path.join(cpu_dir, f"{meta['name']}_modes.h5"), "r") as f:
        ch = f["chunk"][()].astype(str)
        t = f["solve_time"][()]
    return {c: [float(x) for x in t[ch == c] if np.isfinite(x)] for c in sorted(set(ch))}


def pick_lams(chunk, n_lanes, n_lam, center=None):
    """n_lanes contiguous runs of n_lam wavelengths (long -> short) around ``center`` (default: the
    middle of the chunk), as many as the chunk has."""
    lams = sorted(chunk["lams"], reverse=True)
    n_lanes = max(1, min(n_lanes, len(lams) // max(1, n_lam)))   # keep n_lam per lane (warm samples)
    want = min(len(lams), n_lanes * n_lam)
    i_c = len(lams) // 2 if center is None else min(range(len(lams)), key=lambda i: abs(lams[i] - center))
    i0 = max(0, min(len(lams) - want, i_c - want // 2))
    sel = lams[i0:i0 + want]
    n_lanes = max(1, min(n_lanes, len(sel)))
    per = math.ceil(len(sel) / n_lanes)
    return [sel[i:i + per] for i in range(0, len(sel), per)]


def time_unit(plan, chunk, nev, lanes, work, label):
    """Run one work unit through run_geometry_gpu.run_unit (the production code) and return the
    per-λ timing lines it wrote. An untimed warm-up of the same batch shape (each lane's first λ)
    runs first, so compiles and first-call costs are measured apart (compile_s), not in the λ times."""
    import numpy as np
    plan = copy.deepcopy(plan)
    for c in plan["chunks"]:
        if c["name"] == chunk["name"]:
            c["nev"] = nev                                    # run_unit reads the lane's nev from the chunk
    t0 = time.perf_counter()
    wu = os.path.join(work + "_warmup")
    shutil.rmtree(wu, ignore_errors=True)
    os.makedirs(wu)
    _, _, _, st0 = RGG.run_unit(dict(grid=chunk["grid"], nev=nev, lanes=[dict(chunk=chunk["name"], lams=l[:1])
                                                                          for l in lanes]), plan, wu)
    t_warmup = time.perf_counter() - t0
    shutil.rmtree(wu, ignore_errors=True)
    unit = dict(grid=chunk["grid"], nev=nev, lanes=[dict(chunk=chunk["name"], lams=l) for l in lanes])
    shutil.rmtree(work, ignore_errors=True)
    os.makedirs(work)
    t0 = time.perf_counter()
    n, notes, fails, st = RGG.run_unit(unit, plan, work)
    wall = time.perf_counter() - t0
    recs = []
    for p in glob.glob(os.path.join(work, "timing_gpu*.jsonl")):
        recs += [json.loads(l) for l in open(p) if l.strip()]
    first = {l[0] for l in lanes}
    cold = [r["t"] for r in recs if r["lam"] in first]
    warm = [r["t"] for r in recs if r["lam"] not in first]
    comp = sum(c["compile_s"] for c in st0["compiles"] + st["compiles"])
    out = dict(label=label, chunk=chunk["name"], nev=nev, lanes=len(lanes), lams=lanes, solved=n,
               failed=len(fails), wall_s=round(wall, 1), warmup_s=round(t_warmup, 1), compile_s=round(comp, 1),
               compile_in_timed_s=round(sum(c["compile_s"] for c in st["compiles"]), 1),
               t_cold=float(np.mean(cold)) if cold else float("nan"), t_warm=float(np.mean(warm)) if warm else float("nan"),
               newton_max=max((r["newton_max"] for r in recs), default=0),
               lobpcg_max=max((r["lobpcg_max"] for r in recs), default=0),
               escalated=sum(r["escalated"] for r in recs), peak_gpu_gb=st["peak_gpu_gb"],
               slot_batches=sorted({tuple(b["sizes"]) for b in st["batches"]}),
               grid=f"{chunk['grid']['Nx']}x{chunk['grid']['Ny']}", notes=notes)
    print(f"  {label:28s} {out['grid']:>10s} nev {nev} lanes {len(lanes)}: cold {out['t_cold']:8.1f} s/λ, "
          f"warm {out['t_warm']:8.1f} s/λ  (unit {wall:.0f} s; warm-up {t_warmup:.0f} s incl. "
          f"{out['compile_s']:.0f} s compiling; slots per call {out['slot_batches']}; peak {out['peak_gpu_gb']:.1f} GiB; "
          f"newton<={out['newton_max']} lobpcg<={out['lobpcg_max']}"
          + (f"; {out['escalated']} escalated" if out["escalated"] else "") + ")", flush=True)
    return out


def project(plan, per_chunk, n_gpus, units_per_gpu, max_lanes):
    """Projected wall time [s] of the full plan on n_gpus GPUs: the production work units with
    per-λ times from the benchmark (cold for each lane's first λ), handed out in the order the
    driver submits them (each to the GPU that frees first). Returns (wall, LPT lower bound, units)."""
    units = RGG.make_units(plan, n_gpus, units_per_gpu, max_lanes)
    times = []
    for u in units:
        t = 0.0
        for r in u["lanes"]:
            b = per_chunk.get(r["chunk"])
            if b is None:
                return None, None, units
            tw = b["t_warm"] if math.isfinite(b["t_warm"]) else b["t_cold"]
            t += b["t_cold"] + (len(r["lams"]) - 1) * tw
        times.append(t)

    def schedule(ts):
        load = [0.0] * n_gpus
        for t in ts:
            load[load.index(min(load))] += t
        return max(load)

    return schedule(times), schedule(sorted(times, reverse=True)), units


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--grids", choices=("gpu", "cpu-run"), default="gpu")
    ap.add_argument("--cpu-run", default=None, help="geometry folder of a finished CPU run (chunked_sweep)")
    ap.add_argument("--cpu-hours", type=float, default=None, help="the CPU run's wall time (default: from its log)")
    ap.add_argument("--cpu-workers", type=int, default=None, help="the CPU run's workers (default: from its log)")
    ap.add_argument("--geom", default=None, help="JSON RibGeometry parameters (default: the CPU run's, else the 2.44 h baseline)")
    ap.add_argument("--chunks", default=None)
    ap.add_argument("--n-lam", type=int, default=3)
    ap.add_argument("--n-gpus", type=int, default=2, help="GPUs per node for the projection")
    ap.add_argument("--nev-scan", default=None)
    ap.add_argument("--scan-lam", type=float, default=1.5)
    ap.add_argument("--lanes-scan", default=None)
    ap.add_argument("--threshold", type=float, default=6.0)
    ap.add_argument("--max-slots", type=int, default=G.MAX_SLOTS_PER_CALL)
    ap.add_argument("--eigh", choices=("qr", "default", "jacobi"), default=None)
    ap.add_argument("--platform", choices=("cuda", "cpu"), default="cuda")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    if a.grids == "cpu-run" and not a.cpu_run:
        raise SystemExit("--grids cpu-run needs --cpu-run")
    os.makedirs(a.out, exist_ok=True)
    if a.eigh:
        os.environ["GPU_SWEEP_EIGH"] = a.eigh
    jax = init_in_process(a.platform, a.max_slots)
    print(f"[bench_gpu] device {jax.devices()[0]}, backend {jax.default_backend()}, "
          f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}", flush=True)
    import compare_runs as CR

    geometry = json.loads(a.geom) if a.geom else None
    if a.grids == "cpu-run":
        plan = plan_from_cpu_run(a.cpu_run)
    else:
        if geometry is None and a.cpu_run:
            import results_io as rio
            geometry = rio.load_modes(a.cpu_run)[1]["geometry"]
        geometry = geometry or BASELINE
        from geometry_gen.tfln_rib import RibGeometry
        import dataclasses
        g = {k: v for k, v in dataclasses.asdict(RibGeometry(**geometry)).items() if k in RG.GEOM_KEYS}
        plan = RGG.gpu_plan(dict(name=RG.geometry_name(g), geometry=g), "gpu", False, G.STEP)
    sel = None if not a.chunks else [s.strip() for s in a.chunks.split(",")]
    chunks = [c for c in plan["chunks"] if sel is None or any(c["name"] == s or c["name"].startswith(s) for s in sel)]
    print(f"[bench_gpu] {plan['name']}, grids '{a.grids}': "
          + "; ".join(f"{c['name']} {c['grid']['Nx']}x{c['grid']['Ny']} nev {c['nev']}" for c in chunks), flush=True)
    res = dict(geometry=plan["geometry"], grids=a.grids, device=str(jax.devices()[0]), per_chunk={}, scans=[],
               eigh=RGG.eigh_choice(), max_slots=a.max_slots)
    work_root = os.path.join(a.out, f"work_{a.grids}")

    # ---- every (selected) chunk at its production nev and batch size -----------------------
    if not (a.nev_scan or a.lanes_scan):
        print("\nper chunk (production nev and lanes):", flush=True)
        for c in chunks:
            g = c["grid"]
            L = RGG.lanes_for(c["nev"], g["Nx"] * g["Ny"], plan["use_parity"], G.MAX_LANES)
            lanes = pick_lams(c, L, a.n_lam)
            res["per_chunk"][c["name"]] = time_unit(plan, c, c["nev"], lanes, os.path.join(work_root, c["name"]), c["name"])

    # ---- scans ----------------------------------------------------------------------------
    scan_chunk = next((c for c in chunks if c["lam_min"] - 1e-9 <= a.scan_lam <= c["lam_max"] + 1e-9), chunks[0])
    if a.nev_scan:
        print(f"\nnev scan in {scan_chunk['name']} around λ = {a.scan_lam} µm:", flush=True)
        g = scan_chunk["grid"]
        for nev in [int(x) for x in a.nev_scan.split(",")]:
            L = RGG.lanes_for(nev, g["Nx"] * g["Ny"], plan["use_parity"], G.MAX_LANES)
            lanes = pick_lams(scan_chunk, L, a.n_lam, a.scan_lam)
            res["scans"].append(time_unit(plan, scan_chunk, nev, lanes, os.path.join(work_root, f"nev{nev}"),
                                          f"nev {nev} ({L} lanes)"))
    if a.lanes_scan:
        print(f"\nlanes scan in {scan_chunk['name']} (nev {scan_chunk['nev']}):", flush=True)
        for L in [int(x) for x in a.lanes_scan.split(",")]:
            lanes = pick_lams(scan_chunk, L, a.n_lam, a.scan_lam)
            res["scans"].append(time_unit(plan, scan_chunk, scan_chunk["nev"], lanes,
                                          os.path.join(work_root, f"lanes{L}"), f"{L} lanes"))

    # ---- projection and the comparison with the CPU run --------------------------------------
    if res["per_chunk"]:
        print("\nprojection:", flush=True)
        full = plan_from_cpu_run(a.cpu_run) if a.grids == "cpu-run" else plan
        wall, wall_lpt, units = project(full, res["per_chunk"], a.n_gpus, G.UNITS_PER_GPU, G.MAX_LANES)
        # one-time costs, measured in the warm-ups: every GPU process compiles each (nev, batch) program
        # once, both at the same time (and the job's compilation cache serves the second one)
        comp = sum(r["compile_s"] for r in res["per_chunk"].values())
        if wall is None:
            print("  (not all chunks benchmarked: no full-run projection)")
        else:
            res["projected_wall_h"] = (wall + comp) / 3600
            res["projected_wall_lpt_h"] = (wall_lpt + comp) / 3600
            res["compile_h"] = comp / 3600
            print(f"  full geometry ({sum(len(c['lams']) for c in full['chunks'])} λ, {len(units)} work units) on "
                  f"{a.n_gpus} GPU(s): ~{res['projected_wall_h']:.2f} h incl. {comp / 60:.1f} min compiling "
                  f"(best-case order {res['projected_wall_lpt_h']:.2f} h; + ~1-3 min start-up and merge)")
            if a.grids == "cpu-run":
                print("  (assumes each step of a unit costs what a full-width step measured here: units whose last "
                      "steps run fewer lanes are charged a bit too little)")
        if a.cpu_run:
            hc = a.cpu_hours or CR.wall_hours(a.cpu_run, r"\[run_geometry\]")
            nw = a.cpu_workers or CR.workers_from_log(a.cpu_run) or 48
            res["cpu_wall_h"], res["cpu_workers"] = hc, nw
            if hc and wall is not None:
                s = hc / res["projected_wall_h"]
                res["speedup_wall"] = s
                print(f"  CPU run: {hc:.2f} h on {nw} workers  ->  projected GPU-node speed-up {s:.2f}x "
                      f"({'meets' if s >= a.threshold else 'BELOW'} the {a.threshold:g}x threshold)")
            if a.grids == "cpu-run":
                import numpy as np
                ct = cpu_chunk_times(a.cpu_run)
                print("  equal work, node vs node (CPU: 48 λ at a time at its measured per-λ time; "
                      f"GPU: {a.n_gpus} x lanes at the warm per-λ time):")
                tot_c = tot_g = 0.0
                for c in full["chunks"]:
                    b = res["per_chunk"].get(c["name"])
                    tc = ct.get(c["name"])
                    if not b or not tc:
                        continue
                    rc = np.median(tc) / nw                       # CPU node-seconds per λ
                    tg = b["t_warm"] if math.isfinite(b["t_warm"]) else b["t_cold"]
                    rg = tg / a.n_gpus                            # GPU node-seconds per λ (t is per λ already)
                    n = len(c["owned"])
                    tot_c += n * rc
                    tot_g += n * rg
                    print(f"    {c['name']:15s} CPU {np.median(tc):7.0f} s/λ/worker -> {rc:6.2f} node-s/λ;  GPU "
                          f"{tg:7.1f} s/λ/GPU -> {rg:6.2f} node-s/λ;  GPU node {rc / rg:5.2f}x")
                if tot_g:
                    res["speedup_equal_work"] = tot_c / tot_g
                    print(f"    whole geometry, busy nodes: GPU node {tot_c / tot_g:.2f}x the CPU node")
    path = os.path.join(a.out, f"bench_{a.grids}{'_scan' if (a.nev_scan or a.lanes_scan) else ''}.json")
    with open(path, "w") as f:
        json.dump(res, f, indent=1, default=str)
    print(f"\n[bench_gpu] wrote {path}", flush=True)


if __name__ == "__main__":
    main()
