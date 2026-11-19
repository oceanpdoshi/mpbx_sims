"""Compare a CPU run and a GPU run of the same geometry: wall time (the 6x question) and results.

    python gpu_sweep/compare_runs.py CPU_GEOMETRY_DIR GPU_GEOMETRY_DIR
    python gpu_sweep/compare_runs.py CPU_DIR GPU_DIR --cpu-hours 2.44 --threshold 6 --out report.txt

CPU_GEOMETRY_DIR / GPU_GEOMETRY_DIR are geometry folders (<run>/<geometry name>/) written by
chunked_sweep/run_geometry.py and gpu_sweep/run_geometry_gpu.py.

Time: each run's wall time is read from its job log (<run>/logs/*.out: "[run_geometry] finished in
X h" / "[run_geometry_gpu] finished in X h"), or given with --cpu-hours / --gpu-hours. Two ratios:
  * wall-time speed-up = CPU wall / GPU wall, node against node (1 CPU node vs 1 GPU node), the number
    to hold against the 24-CPU-node vs 4-GPU-node threshold (6x by default);
  * busy-time speed-up = the CPU's summed per-λ worker time / its worker count, against the GPU's
    summed per-λ GPU time / its GPU count: what each node does when it is never idle (a sweep of
    many geometries keeps the CPU workers busy past a single geometry's long tail).
Results: per target mode, the differences GPU - CPU of n_eff and n_g at the wavelengths both runs
solved. With different grids this is (mostly) the CPU grid's discretization error (README §2.2 of
chunked_sweep: the coarser grid sits low, so GPU - CPU > 0 is expected); with the same grids it
must be at the solver tolerance (~1e-8).
"""
import argparse
import glob
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [p for p in (HERE, os.path.join(ROOT, "chunked_sweep"), ROOT) if p not in sys.path]
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")


def wall_hours(gdir, pattern):
    """Wall time [h] from the run's job logs (<run>/logs/*.out): the sum of the "finished in" times of
    all its jobs (a resumed run has one per job), or None. Warns where that can mislead: several
    jobs (resumed: a killed job prints nothing and is missing from the sum; a sweep: the jobs solved
    other geometries too). Give --cpu-hours / --gpu-hours then."""
    run = os.path.dirname(os.path.abspath(gdir))
    hits, logs = [], 0
    for p in glob.glob(os.path.join(run, "logs", "**", "*.out"), recursive=True):
        try:
            txt = open(p, encoding="utf-8", errors="replace").read()
        except OSError:
            continue
        if re.search(pattern + r" ", txt):
            logs += 1
        hits += [float(x) for x in re.findall(pattern + r" finished in ([0-9.]+) h", txt)]
    if not hits:
        return None
    if logs > 1 or len(hits) > 1:
        print(f"WARNING: {run} has {logs} job logs ({len(hits)} finished): using the sum {sum(hits):.3f} h. "
              f"For a resumed run add the killed jobs' time; for a sweep give --cpu-hours / --gpu-hours.")
    return sum(hits)


def workers_from_log(gdir):
    """Number of CPU worker processes ("48 CPUs -> 48 workers x 1") from the CPU run's log."""
    for p in glob.glob(os.path.join(os.path.dirname(os.path.abspath(gdir)), "logs", "**", "*.out"), recursive=True):
        m = re.search(r"CPUs -> (\d+) workers x", open(p, encoding="utf-8", errors="replace").read())
        if m:
            return int(m.group(1))
    return None


def gpu_count(gdir, name):
    p = os.path.join(gdir, f"{name}_gpu_timing.jsonl")
    if not os.path.exists(p):
        return None, []
    recs = [json.loads(l) for l in open(p) if l.strip()]
    return len({r["gpu"] for r in recs}), recs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cpu_dir")
    ap.add_argument("gpu_dir")
    ap.add_argument("--cpu-hours", type=float, default=None)
    ap.add_argument("--gpu-hours", type=float, default=None)
    ap.add_argument("--cpu-workers", type=int, default=None)
    ap.add_argument("--threshold", type=float, default=6.0, help="speed-up the GPU node must reach (default 6)")
    ap.add_argument("--out", default=None, help="also write the report to this file")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    import numpy as np
    import results_io as rio

    L = []
    out = L.append
    mc, meta_c = rio.load_modes(a.cpu_dir)
    mg, meta_g = rio.load_modes(a.gpu_dir)
    tc, tg = rio.load_tracks(a.cpu_dir), rio.load_tracks(a.gpu_dir)
    out(f"CPU run: {os.path.abspath(a.cpu_dir)}\n  complete {meta_c['complete']}, {mc.lam.nunique()} λ, backend "
        f"{meta_c.get('backend')}, code {meta_c.get('code_version')}, mpbx {meta_c.get('mpbx_version')}")
    out(f"GPU run: {os.path.abspath(a.gpu_dir)}\n  complete {meta_g['complete']}, {mg.lam.nunique()} λ, backend "
        f"{meta_g.get('backend')}, code {meta_g.get('code_version')}, mpbx {meta_g.get('mpbx_version')}")
    if meta_c["geometry"] != meta_g["geometry"]:
        out(f"WARNING: different geometries\n  CPU {meta_c['geometry']}\n  GPU {meta_g['geometry']}")
    out("\ngrids (chunk: Nx x Ny px = Lx x Ly µm, nev per sector):")
    for tag, meta in (("CPU", meta_c), ("GPU", meta_g)):
        for c in meta["chunks"]:
            g = c["grid"]
            out(f"  {tag} {c['name']:15s} {g['Nx']} x {g['Ny']} = {g['Nx'] * g['Ny'] / 1e6:.2f} Mpx = "
                f"{g['Lx']:.3f} x {g['Ly']:.3f} µm (dx {1e3 * g['dx']:.2f}, dy {1e3 * g['dy']:.2f} nm), nev {c['nev']}")

    # ---- time ------------------------------------------------------------------------------
    hc = a.cpu_hours or wall_hours(a.cpu_dir, r"\[run_geometry\]")
    hg = a.gpu_hours or wall_hours(a.gpu_dir, r"\[run_geometry_gpu\]")
    nw = a.cpu_workers or workers_from_log(a.cpu_dir)
    n_gpu, grecs = gpu_count(a.gpu_dir, meta_g["name"])
    import h5py

    def busy(gdir, name):                                     # Σ per-λ solve time of the tables' wavelengths
        with h5py.File(os.path.join(gdir, f"{name}_modes.h5"), "r") as f:
            return float(np.nansum(f["solve_time"][()]))

    busy_c, busy_g = busy(a.cpu_dir, meta_c["name"]), busy(a.gpu_dir, meta_g["name"])
    out("\ntime:")
    out(f"  CPU wall {hc if hc is not None else '?'} h; per-λ worker time summed {busy_c / 3600:.2f} h over "
        f"{nw or '?'} workers")
    out(f"  GPU wall {hg if hg is not None else '?'} h; per-λ GPU time summed {busy_g / 3600:.3f} h over "
        f"{n_gpu or '?'} GPU(s)")
    if hc and hg:
        s = hc / hg
        out(f"  WALL-TIME SPEED-UP (1 GPU node vs 1 CPU node): {s:.2f}x  -> "
            f"{'meets' if s >= a.threshold else 'BELOW'} the {a.threshold:g}x threshold")
    if nw and n_gpu and np.isfinite(busy_g) and busy_g > 0:
        s2 = (busy_c / nw) / (busy_g / n_gpu)
        out(f"  BUSY-TIME SPEED-UP (both nodes kept full, e.g. a many-geometry sweep): {s2:.2f}x  "
            f"(CPU {busy_c / nw / 3600:.3f} h vs GPU {busy_g / n_gpu / 3600:.3f} h per geometry)")
    out("  (the two runs may solve different grids: see above. Equal grids: run_geometry_gpu.py --chunks cpu)")

    # ---- results ----------------------------------------------------------------------------
    out("\nresults, GPU - CPU at the wavelengths both solved (median / max |.| / max signed, per target):")
    out(f"  {'target':7s} {'n λ':>4s}   {'Δn_eff median':>14s} {'max|Δn_eff|':>12s}   {'Δn_g median':>12s} {'max|Δn_g|':>10s}")
    for t in sorted(set(tc.target) & set(tg.target)):
        a_ = tc[tc.target == t].set_index(tc[tc.target == t].lam.round(6))
        b_ = tg[tg.target == t].set_index(tg[tg.target == t].lam.round(6))
        common = a_.index.intersection(b_.index)
        if not len(common):
            continue
        dn = (b_.loc[common, "neff"] - a_.loc[common, "neff"]).to_numpy(float)
        dg = (b_.loc[common, "ng"] - a_.loc[common, "ng"]).to_numpy(float)
        out(f"  {t:7s} {len(common):4d}   {np.median(dn):+14.2e} {np.max(np.abs(dn)):12.2e}   "
            f"{np.median(dg):+12.2e} {np.max(np.abs(dg)):10.2e}")
    lc, lg = set(mc.lam.round(6)), set(mg.lam.round(6))
    if lc != lg:
        out(f"  λ only in CPU run: {sorted(lc - lg)[:10]}{' ...' if len(lc - lg) > 10 else ''}; "
            f"only in GPU run: {sorted(lg - lc)[:10]}{' ...' if len(lg - lc) > 10 else ''}")
    lab_c = mc.assign(lam=mc.lam.round(6)).groupby("lam").label.apply(lambda s: sorted(s))
    lab_g = mg.assign(lam=mg.lam.round(6)).groupby("lam").label.apply(lambda s: sorted(s))
    diff = [l for l in lab_c.index.intersection(lab_g.index) if lab_c[l] != lab_g[l]]
    out(f"  wavelengths whose set of mode labels differs: {len(diff)}"
        + (f" (e.g. {diff[:5]})" if diff else ""))
    txt = "\n".join(L)
    print(txt)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            f.write(txt + "\n")


if __name__ == "__main__":
    main()
