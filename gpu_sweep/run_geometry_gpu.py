"""Mode-solve 0.4-2.0 µm for one or more TFLN rib geometries on GPUs (JAX + CUDA): ONE solver
process per GPU, several wavelengths in flight per GPU (as many as fit in its memory), and the same
per-geometry result files as chunked_sweep/run_geometry.py (results_io reads both).

    python run_geometry_gpu.py --out OUT                                     default geometry, all GPUs of the job
    python run_geometry_gpu.py --out OUT --geom '{"w_top": 1.8, "t_clad": 1.35}'
    python run_geometry_gpu.py --out OUT --geom-csv geometries.csv --row 3
    python run_geometry_gpu.py --out OUT --geom-csv geometries.csv --task 2 --n-tasks 4
  options
    --chunks gpu           gpu_config.GPU_CHUNKS: one 10 x 10 µm window at 6.25 nm for every λ (default)
    --chunks cpu           chunked_sweep/sweep_config.CHUNKS on the GPU: the CPU sweep's exact grids
    --test                 tiny configuration (gpu_config.TEST_CHUNKS, or sweep_config's with --chunks cpu)
    --dry-run              grids, batch sizes, work units and memory estimates; solve nothing
    --gpus N               use N GPUs (default: every GPU in CUDA_VISIBLE_DEVICES / nvidia-smi)
    --platform cpu         run the same code on the CPU backend (tests on a laptop; --gpus = processes)
    --merge-only, --keep-work, --lams 0.5,1.55, --step   as in run_geometry.py
    --units-per-gpu, --max-lanes                          scheduling (gpu_config.py, README §2.4)
    --max-slots 1          one sector solve per call (no lockstep batching; README §3.3)

E fields are never stored (2.5 million pixels x 12 modes x 161 λ would be ~100 GB per geometry);
everything else is (README §5). Per geometry, beside run_geometry.py's files:
    <name>_gpu_timing.jsonl   one line per solved λ: GPU time, batch, iterations, GPU memory
    <name>_gpu_report.txt     wall time, GPU-seconds per λ by chunk, compiles, peak GPU memory

How the work is split (README §2.4): each chunk's wavelengths are cut into contiguous runs ("lanes",
solved long -> short so each λ warm-starts from the previous one). Lanes of the same grid and nev are
grouped into work units of as many lanes as fit on one GPU at once (2 sector solves per lane); the
lanes of a unit advance in lockstep, one batched solve per step. Units go to the GPU processes
longest-first; each geometry is merged as soon as its last unit is done.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import datetime
import glob
import json
import math
import multiprocessing as mp
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures.process import BrokenProcessPool

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CPU_DIR = os.path.join(ROOT, "chunked_sweep")
PATHS = [HERE, CPU_DIR, ROOT]
sys.path[:0] = [p for p in PATHS if p not in sys.path]
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")     # SuperCloud Lustre (see run_geometry.py)

import gpu_config as G  # noqa: E402
import run_geometry as RG  # noqa: E402   (imports only sweep_config at module level: no JAX here)
import sweep_config as C  # noqa: E402

GiB = 1024 ** 3
CHUNK_SETS = {("gpu", False): G.GPU_CHUNKS, ("gpu", True): G.TEST_CHUNKS,
              ("cpu", False): C.CHUNKS, ("cpu", True): C.TEST_CHUNKS}


# --------------------------------------------------------------------------------------
# Plan
# --------------------------------------------------------------------------------------
def gpu_plan(entry, chunk_set, test, step, only=None):
    """run_geometry.geometry_plan for the GPU chunks: pixel counts from cuFFT's fast radices for
    --chunks gpu, the CPU sweep's own rule (identical grids) for --chunks cpu. Fields: none."""
    chunks = CHUNK_SETS[(chunk_set, test)]
    primes = G.SMOOTH_PRIMES if chunk_set == "gpu" else RG.SMOOTH_PRIMES
    old = RG.SMOOTH_PRIMES
    RG.SMOOTH_PRIMES = primes                                 # read by run_geometry._smooth
    try:
        plan = RG.geometry_plan(entry, chunks, step, "none", only)
    finally:
        RG.SMOOTH_PRIMES = old
    plan.update(engine="gpu (jax)", chunk_set=chunk_set + ("-test" if test else ""), smooth_primes=list(primes))
    return plan


def grid_key(g):
    return (g["Nx"], g["Ny"], g["Lx"], g["Ly"])


def slot_gb(nev, N, use_parity):
    """Planning estimate of one sector solve's device memory [GiB]: the program, its copy of ε⁻¹, and
    the eigenvector copies / warm-start block kept beside it (2 x n complex128 per band, n = 2N)."""
    bands = nev if use_parity else 2 * nev
    return ((G.BYTES_PER_PX_BAND + 64) * N * bands + 72 * N) / GiB


def lanes_for(nev, N, use_parity, max_lanes):
    """Wavelengths that fit on one GPU at once (planning estimate; the workers check the compiled
    program and run fewer at a time if needed)."""
    per_lane = (2 if use_parity else 1) * slot_gb(nev, N, use_parity) + 4 * 72 * N / GiB   # + ε, ε⁻¹, ∂ε, nng
    budget = G.GPU_MEM_GB * G.MEM_FRACTION - G.HEADROOM_GB - 150 * N / GiB   # - grid masks, ln_fill, plan
    return max(1, min(max_lanes, int(budget // per_lane)))


def make_units(plan, n_gpus, units_per_gpu, max_lanes, min_run=G.MIN_RUN):
    """Work units: lanes (contiguous λ runs, long -> short) of one grid and one nev, as many per
    unit as fit on a GPU; about n_gpus·units_per_gpu units per geometry, longest first."""
    pieces = []
    for c in plan["chunks"]:
        if not c["lams"]:
            continue
        g = c["grid"]
        lams = sorted(c["lams"], reverse=True)
        cost = [RG.lam_cost(g["Nx"], g["Ny"], c["nev"], l) for l in lams]
        pieces.append((c, lams, cost))
    total = sum(sum(p[2]) for p in pieces)
    target = total / max(1, n_gpus * units_per_gpu)
    pools = {}
    for c, lams, cost in pieces:
        g = c["grid"]
        L = lanes_for(c["nev"], g["Nx"] * g["Ny"], plan["use_parity"], max_lanes)
        k_units = max(1, round(sum(cost) / target))
        n_runs = max(1, min(len(lams) // min_run, k_units * L))
        base, extra = divmod(len(lams), n_runs)
        i0 = 0
        pool = pools.setdefault((grid_key(g), c["nev"]), dict(grid=g, nev=c["nev"], L=L, runs=[]))
        for k in range(n_runs):
            n = base + (1 if k < extra else 0)
            pool["runs"].append(dict(chunk=c["name"], lams=lams[i0:i0 + n], cost=sum(cost[i0:i0 + n])))
            i0 += n
    units = []
    for pool in pools.values():
        runs = sorted(pool["runs"], key=lambda r: -len(r["lams"]))   # similar lengths share a unit
        for i in range(0, len(runs), pool["L"]):
            lanes = runs[i:i + pool["L"]]
            units.append(dict(grid=pool["grid"], nev=pool["nev"], lanes=lanes, cost=sum(r["cost"] for r in lanes)))
    units.sort(key=lambda u: -u["cost"])                      # longest first (LPT)
    return units


def print_plan(plan, units, n_gpus):
    from geometry_gen.tfln_rib import make_geometry
    print(f"\n=== {plan['name']}")
    print(f"  {make_geometry(**plan['geometry']).describe()}")
    tot = sum(u["cost"] for u in units) or 1.0
    for c in plan["chunks"]:
        g = c["grid"]
        N = g["Nx"] * g["Ny"]
        L = lanes_for(c["nev"], N, plan["use_parity"], 10 ** 6)
        cost = sum(RG.lam_cost(g["Nx"], g["Ny"], c["nev"], l) for l in c["lams"])
        print(f"  {c['name']:15s} {len(c['lams']):3d} λ  {g['Nx']}x{g['Ny']} px = {g['Lx']:.3f} x {g['Ly']:.3f} µm "
              f"({N / 1e6:.2f} Mpx; dx {1e3 * g['dx']:.2f}, dy {1e3 * g['dy']:.2f} nm)  nev {c['nev']}/sector  "
              f"cost {cost / tot:5.1%}  ~{slot_gb(c['nev'], N, plan['use_parity']):.1f} GiB per sector solve "
              f"-> {L} λ in flight per GPU")
    for w in plan["warnings"]:
        print(f"  WARNING {w}")
    cpu_cost = sum(RG.lam_cost(g["Nx"], g["Ny"], c["nev"], l) for c in plan["chunks"] for g in [c["grid"]]
                   for l in c["lams"])
    print(f"  {len(units)} work units for {n_gpus} GPU(s): "
          + ", ".join(f"[{'+'.join(sorted({r['chunk'][0] for r in u['lanes']}))} nev {u['nev']}: "
                      f"{len(u['lanes'])} x {max(len(r['lams']) for r in u['lanes'])} λ]" for u in units))
    print(f"  cost model: {cpu_cost / 3600:.1f} desktop-core-hours at the CPU rate (README §3 of chunked_sweep; "
          f"for scale only: the GPU rate is what bench_gpu.py measures)")


# --------------------------------------------------------------------------------------
# Worker process (one per GPU)
# --------------------------------------------------------------------------------------
_W = {}


def eigh_choice():
    """gpu_config.GPU_EIGH, overridable with the environment variable GPU_SWEEP_EIGH (bench / tests)."""
    return os.environ.get("GPU_SWEEP_EIGH", G.GPU_EIGH)


def _init_worker(dev_queue, platform, mem_fraction, max_slots=None):
    dev, cpus = dev_queue.get()
    if cpus:
        RG._pin(cpus)                                          # host threads: this worker's share of the cores
    if platform == "cuda":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(dev)         # before JAX is imported: this process sees ONE GPU
        os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(mem_fraction)
    init_jax(platform, dev, max_slots)


def init_jax(platform, dev, max_slots=None, verbose=False):
    """JAX + batched_modes in this process (a worker, or bench_gpu.py's own process)."""
    sys.path[:0] = [p for p in PATHS if p not in sys.path]
    from utils.setup_jax import setup_jax
    setup_jax(platform, verbose=verbose)
    import jax
    import batched_modes as BM
    BM.set_eigh(eigh_choice())                                # before anything is compiled
    _W.update(dev=dev, platform=platform, BM=BM, jax=jax, ctx={}, ctx_geom=None, max_slots=max_slots)
    print(f"[worker {os.getpid()}] device {dev}: {jax.devices()[0]} ({jax.default_backend()}), "
          f"LOBPCG eigh: {eigh_choice()}{f', at most {max_slots} slots per call' if max_slots else ''}", flush=True)
    return jax


def _context(plan, grid):
    """The GridContext of (geometry, grid), kept while the worker stays on the same geometry."""
    BM = _W["BM"]
    if _W["ctx_geom"] != plan["name"]:
        _W["ctx"].clear()
        _W["ctx_geom"] = plan["name"]
    key = grid_key(grid)
    if key not in _W["ctx"]:
        t0 = time.perf_counter()
        _W["ctx"][key] = BM.GridContext(plan["geometry"], grid, plan["align_frac"], plan["use_parity"],
                                        plan["k_tol"], plan["eig_tol"], plan["field_dtype"],
                                        headroom_gb=G.HEADROOM_GB, max_slots=_W.get("max_slots"),
                                        log=lambda s: print(s, flush=True))
        print(f"[worker {os.getpid()}] {plan['name']}: dielectric + masks for {grid['Nx']}x{grid['Ny']} px "
              f"in {time.perf_counter() - t0:.1f} s", flush=True)
    return _W["ctx"][key]


def labels_and_missing(sw, labs, targets, use_parity):
    """run_geometry._labels_and_missing, given the node-count labels (computed on the device)."""
    import numpy as np
    te = sw.data["te_frac"][0]
    names = [f"{p}{m}{n}" for p, m, n in labs]
    guided, parity, neff = sw.data["guided"][0], sw.data["parity"][0], sw.data["neff"][0]
    found = {nm for nm, g in zip(names, guided) if g}
    missing = [t for t in targets if t not in found]
    deep_guided, collided = [], []
    for t in missing:
        sel = (parity == RG.target_sector(t)) if use_parity else np.ones(len(neff), bool)
        if not (sel.any() and guided[sel][np.argmin(neff[sel])]):
            continue                                          # sector's deepest mode is cut off
        js = [j for j in np.flatnonzero(sel) if guided[j]]
        sec_names = [names[j] for j in js]
        if len(set(sec_names)) < len(sec_names) or any(RG.HYBRID[0] < te[j] < RG.HYBRID[1] for j in js):
            collided.append(t)
        else:
            deep_guided.append(t)
    return missing, deep_guided, collided


def _peak_rss_gb():
    return RG._peak_rss_gb()


def run_unit(unit, plan, work):
    """Solve one work unit (its lanes in lockstep). Returns (n solved, notes, failures, stats)."""
    import numpy as np
    BM = _W["BM"]
    ctx = _context(plan, unit["grid"])
    chunks = {c["name"]: c for c in plan["chunks"]}
    lanes = [dict(chunk=r["chunk"], lams=list(r["lams"]), nev=chunks[r["chunk"]]["nev"], warm=None)
             for r in unit["lanes"]]
    timing = os.path.join(work, f"timing_gpu{_W['dev']}_{os.getpid()}.jsonl")
    notes, failures, n_done = [], [], 0
    stats = dict(device=str(_W["jax"].devices()[0]), dev=_W["dev"], batches=[], t_solve=0.0, t_start=time.time())
    for t in range(max(len(ln["lams"]) for ln in lanes)):
        todo = []
        for ln in lanes:
            if t >= len(ln["lams"]):
                continue
            lam = ln["lams"][t]
            c = chunks[ln["chunk"]]
            d = os.path.join(work, c["name"])
            os.makedirs(d, exist_ok=True)
            path = os.path.join(d, f"lam_{lam * 1e3:08.2f}nm.h5")
            done = RG._existing_ok(path, plan, c) if os.path.exists(path) else None
            if done:                                          # solved before a restart
                ln["warm"] = None
                ln["nev"] = max(ln["nev"], int(done.get("nev_next", done.get("nev_per_sector")) or ln["nev"]))
                continue
            todo.append((ln, lam, path, c))
        groups = {}                                           # lanes with a raised nev run apart; membership
        for x in todo:                                        # fixed now: a lane that escalates below must not
            groups.setdefault(x[0]["nev"], []).append(x)      # be solved again in this step's next group
        for nev, grp in sorted(groups.items()):
            t0 = time.perf_counter()
            try:
                lds = [ctx.lam_data(lam) for _, lam, _, _ in grp]
                slots = [BM.Slot(ld, s, (ln["warm"] or {}).get(s)) for (ln, _, _, _), ld in zip(grp, lds)
                         for s in ctx.sectors]
                res, st = ctx.solve_slots(slots, nev)
            except Exception as e:                            # noqa: BLE001  the whole batch failed
                import traceback
                for ln, lam, _, c in grp:
                    failures.append((lam, repr(e)))
                    ln["warm"] = None
                print(f"[worker {os.getpid()}] {plan['name']} λ={[x[1] for x in grp]} FAILED:\n{traceback.format_exc()}",
                      flush=True)
                continue
            t_solve = time.perf_counter() - t0
            stats["t_solve"] += t_solve
            stats["batches"].append(dict(nev=nev, lanes=len(grp), slots=len(slots), sizes=st["batches"],
                                         calls=st["newton_calls"], t=round(t_solve, 2)))
            ns = len(ctx.sectors)
            for i, (ln, lam, path, c) in enumerate(grp):
                try:
                    _finish_lambda(ctx, plan, c, ln, lds[i], path, res[i * ns:(i + 1) * ns], nev,
                                   t_solve / len(grp), len(grp), st, notes, timing)
                    n_done += 1
                except Exception as e:                        # noqa: BLE001  one λ must not stop the unit
                    import traceback
                    failures.append((lam, repr(e)))
                    ln["warm"] = None
                    print(f"[worker {os.getpid()}] {plan['name']} {c['name']} λ={lam:.3f} FAILED:\n"
                          f"{traceback.format_exc()}", flush=True)
            del lds, slots, res
    stats.update(peak_gpu_gb=ctx.peak_gb(), compiles=list(ctx.compile_log), peak_rss_gb=_peak_rss_gb(),
                 t_end=time.time())
    ctx.compile_log.clear()
    return n_done, notes, failures, stats


def _finish_lambda(ctx, plan, c, ln, ld, path, results, nev, t_share, n_lanes, st, notes, timing):
    """Post-process one solved λ, re-solve it with more modes if a target may sit below the last
    solved mode (run_geometry._solve_one's rule), save it. Updates the lane's warm state / nev."""
    import numpy as np
    BM = _W["BM"]
    nev0 = c["nev"]
    nev_max = nev0 + plan["nev_extra"]
    lam = ld.lam
    sw, labs, warm_new, t_post = ctx.build_sweep(ld, results, nev, t_share)
    missing, deep, collided = labels_and_missing(sw, labs, c["targets"], plan["use_parity"])
    if collided:
        notes.append(f"{c['name']} λ={lam:.3f}: {', '.join(collided)} not labelled; its sector has a "
                     f"duplicate or hybrid label (anticrossing?) -> not re-solved")
    escalated, nev_next = False, nev
    if deep and nev < nev_max:                                # a target may sit below the last solved mode
        escalated = True
        t1 = time.perf_counter()
        res2, _ = ctx.solve_slots([BM.Slot(ld, s, None) for s in ctx.sectors], nev_max)   # cold, as on the CPU
        sw, labs, warm_new, _ = ctx.build_sweep(ld, res2, nev_max, t_share + time.perf_counter() - t1)
        missing, _, _ = labels_and_missing(sw, labs, c["targets"], plan["use_parity"])
        still = [t for t in deep if t in missing]
        if still:                                             # more modes did not reveal it: go back
            warm_new = RG._trim_warm(warm_new, ctx.bands(nev))   # (bands per solve: 2·nev without parity)
            notes.append(f"{c['name']} λ={lam:.3f}: {', '.join(still)} still missing with nev {nev_max} "
                         f"per sector -> back to nev {nev} for the next λ")
        else:                                                 # it was below the cut: keep the extra modes
            nev_next = nev_max
            notes.append(f"{c['name']} λ={lam:.3f}: {', '.join(deep)} found with nev {nev_max} per sector "
                         f"-> kept for λ <= {lam:.3f} in this lane")
        nev = nev_max
    if not (np.isfinite(sw.data["neff"]).all() and np.isfinite(sw.data["ng"]).all()):
        warm_new = None                                       # never warm-start the next λ from NaNs
        notes.append(f"{c['name']} λ={lam:.3f}: non-finite n_eff / n_g -> the next λ of this lane starts cold")
    ln["warm"], ln["nev"] = warm_new, nev_next
    sw.data["hg_pol"] = np.array([[p for p, _, _ in labs]])
    sw.data["hg_m"] = np.array([[m for _, m, _ in labs]])
    sw.data["hg_n"] = np.array([[n for _, _, n in labs]])
    sw.data["hg_err"] = np.full((1, len(labs)), np.nan)
    gpu = dict(device=str(ctx.device), peak_gb=round(ctx.peak_gb(), 3), lanes_in_step=n_lanes,
               slot_batches=st["batches"], newton_calls=st["newton_calls"])
    sw.meta.update(geometry_name=plan["name"], chunk=c["name"], align_frac=plan["align_frac"],
                   mpbx_commit=plan.get("mpbx_commit"), nev_per_sector=nev, nev_next=nev_next,
                   nev_escalated=escalated, missing_targets=missing, collided_targets=collided,
                   label_method="node count of the dominant transverse E component (on the device)",
                   cores=[], peak_rss_gb=_peak_rss_gb(), gpu=gpu)
    tmp = f"{path}.{os.getpid()}.part"
    sw.save(tmp, compression=None)
    os.replace(tmp, path)                                     # never leave a half-written file under the final name
    d = sw.data
    with open(timing, "a") as tf:                             # one line per λ, survives a time-limit kill
        tf.write(json.dumps(dict(geometry=plan["name"], chunk=c["name"], lam=lam,
                                 t=float(sw.per_lam["solve_time"][0]), t_post=round(t_post, 3), nev=nev,
                                 escalated=escalated, lanes=n_lanes, slot_batches=st["batches"],
                                 newton_max=int(d["newton_iters"].max()), lobpcg_max=int(d["lobpcg_iters"].max()),
                                 converged=bool(d["converged"].all()), gpu=str(_W["dev"]),
                                 peak_gpu_gb=round(ctx.peak_gb(), 3), peak_rss_gb=round(_peak_rss_gb(), 3),
                                 Nx=ctx.grid.Nx, Ny=ctx.grid.Ny,
                                 time=datetime.datetime.now().isoformat(timespec="seconds"))) + "\n")
    return nev_next


# --------------------------------------------------------------------------------------
# Main process
# --------------------------------------------------------------------------------------
def visible_gpus(n_req, platform):
    """Device ids the workers get: CUDA_VISIBLE_DEVICES (set by Slurm for --gres=gpu:...), else
    nvidia-smi's list; on the CPU platform, n_req (default 2) pseudo-devices."""
    if platform != "cuda":
        return [f"cpu{i}" for i in range(n_req or 2)]
    env = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if env and env != "NoDevFiles":
        ids = [x.strip() for x in env.split(",") if x.strip()]
    else:
        try:
            out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=30).stdout
            ids = [str(i) for i, ln in enumerate(l for l in out.splitlines() if l.startswith("GPU "))]
        except Exception:                                     # noqa: BLE001
            ids = []
    if not ids:
        raise SystemExit("no GPU visible (CUDA_VISIBLE_DEVICES empty, nvidia-smi found none); "
                         "use --platform cpu to test on the CPU")
    if n_req:
        if n_req > len(ids):
            raise SystemExit(f"--gpus {n_req}: only {len(ids)} visible ({ids})")
        ids = ids[:n_req]
    return ids


def consolidate_timing(gdir, name, work):
    """work/timing_gpu*.jsonl -> <name>_gpu_timing.jsonl (work/ is deleted after a complete merge).
    A line that does not parse (a worker killed while writing it) is skipped with a warning."""
    recs = []
    for p in sorted(glob.glob(os.path.join(work, "timing_gpu*.jsonl"))):
        with open(p) as f:
            for ln in f:
                for part in ln.replace("}{", "}\n{").splitlines():   # a cut line glued to the next one
                    if not part.strip():
                        continue
                    try:
                        recs.append(json.loads(part))
                    except ValueError:
                        print(f"  [{name}] skipped an unreadable timing line in {os.path.basename(p)}", flush=True)
    if recs:
        out = os.path.join(gdir, f"{name}_gpu_timing.jsonl")
        with open(out + ".part", "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in recs)
        os.replace(out + ".part", out)
    return recs


def write_report(gdir, name, plan, recs, ustats, wall, n_gpus, platform, args_line):
    import numpy as np
    from geometry_gen.tfln_rib import make_geometry
    L = [f"{name}  (GPU run)", f"  {make_geometry(**plan['geometry']).describe()}",
         f"  written {datetime.datetime.now():%Y-%m-%d %H:%M:%S}   args: {args_line}",
         f"  wall time of this geometry in this run (first unit started -> last unit done): {wall / 3600:.3f} h "
         f"({wall:.0f} s) on {n_gpus} {'GPU' if platform == 'cuda' else 'CPU process'}(es); "
         f"chunks '{plan['chunk_set']}'; LOBPCG eigh '{eigh_choice()}'",
         "", "devices (peak device memory in use, GiB):"]
    devs = {}
    for s in ustats:
        dv = devs.setdefault(s["dev"], dict(device=s["device"], peak=0.0, t_solve=0.0))
        dv["peak"] = max(dv["peak"], s.get("peak_gpu_gb") or 0.0)
        dv["t_solve"] += s["t_solve"]
    for k, dv in sorted(devs.items()):
        L.append(f"  {k}: {dv['device']}  peak {dv['peak']:.2f} GiB  busy solving {dv['t_solve'] / 3600:.3f} h "
                 f"({100 * dv['t_solve'] / max(wall, 1e-9):.0f} % of the wall time)")
    L += ["", "chunks (λ solved; GPU-seconds per λ: the batch's time shared by its wavelengths + own post-processing):"]
    for c in plan["chunks"]:
        g = c["grid"]
        r = [x for x in recs if x["chunk"] == c["name"]]
        if not r:
            L.append(f"  {c['name']:15s} no λ solved in this run")
            continue
        t = np.array([x["t"] for x in r])
        L.append(f"  {c['name']:15s} {len(r):3d}/{len(c['lams'])} λ  {g['Nx']}x{g['Ny']} px  nev {c['nev']}  "
                 f"median {np.median(t):7.1f} s/λ  mean {t.mean():7.1f}  max {t.max():7.1f}  total {t.sum() / 3600:.3f} h  "
                 f"lanes {sorted({x['lanes'] for x in r})}  newton<= {max(x['newton_max'] for x in r)}  "
                 f"lobpcg<= {max(x['lobpcg_max'] for x in r)}  escalated {sum(x['escalated'] for x in r)}  "
                 f"not converged {sum(not x['converged'] for x in r)}")
    tot = sum(x["t"] for x in recs)
    L += ["", f"total GPU-time {tot / 3600:.3f} h over {len(recs)} λ (all runs that wrote timing lines)",
          "", "compiled programs (nev, slots per call, compile seconds, device memory XLA reported):"]
    for s in ustats:
        for cp in s.get("compiles", []):
            L.append(f"  device {s['dev']}: nev {cp['nev']}  B {cp['B']}  {cp['compile_s']} s  {cp['need_gb']} GiB")
    with open(os.path.join(gdir, f"{name}_gpu_report.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="output folder (one sub-folder per geometry)")
    ap.add_argument("--geom", default="{}", help="JSON dict of RibGeometry parameters (default geometry otherwise); "
                    "t_undercut > 0 makes it an undercut rib (RibGeometryUndercut)")
    ap.add_argument("--name", default=None)
    ap.add_argument("--geom-csv", default=None)
    ap.add_argument("--row", type=int, default=None)
    ap.add_argument("--task", type=int, default=None)
    ap.add_argument("--n-tasks", type=int, default=1)
    ap.add_argument("--chunks", choices=("gpu", "cpu"), default="gpu")
    ap.add_argument("--gpus", type=int, default=None, help="number of GPUs (default: all visible)")
    ap.add_argument("--platform", choices=("cuda", "cpu"), default="cuda")
    ap.add_argument("--units-per-gpu", type=float, default=G.UNITS_PER_GPU)
    ap.add_argument("--max-lanes", type=int, default=G.MAX_LANES)
    ap.add_argument("--max-slots", type=int, default=G.MAX_SLOTS_PER_CALL,
                    help="at most this many sector solves per batched call (1 = one at a time, no lockstep)")
    ap.add_argument("--step", type=float, default=None)
    ap.add_argument("--lams", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--merge-only", action="store_true")
    ap.add_argument("--keep-work", action="store_true")
    a = ap.parse_args()
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:                                     # noqa: BLE001
            pass

    from utils.setup_jax import setup_jax
    setup_jax("cpu", verbose=False)                           # planning only: the main process never touches a GPU
    import results_io

    step = a.step or ((G.TEST_STEP if a.chunks == "gpu" else C.TEST_STEP) if a.test
                      else (G.STEP if a.chunks == "gpu" else C.STEP))
    only = [float(x) for x in a.lams.split(",")] if a.lams else None
    entries = RG.select_geometries(a)
    if not entries:
        print(f"[run_geometry_gpu] task {a.task}: no geometries for this task")
        return
    mpbx_commit = results_io.mpbx_version()
    no_solve = a.dry_run or a.merge_only                      # planning / merging only: no GPU needed
    devs = [str(i) for i in range(a.gpus or 2)] if no_solve else visible_gpus(a.gpus, a.platform)
    n_gpus = len(devs)
    allowed = RG.available_cpus()
    per = max(1, len(allowed) // n_gpus)
    cpu_sets = [allowed[i * per:(i + 1) * per] for i in range(n_gpus)]
    print(f"[run_geometry_gpu] {datetime.datetime.now():%Y-%m-%d %H:%M:%S} host "
          f"{os.uname().nodename if hasattr(os, 'uname') else ''} | platform {a.platform} | devices {devs} "
          f"| {len(allowed)} CPUs ({per} per worker) | {len(entries)} geometr{'y' if len(entries) == 1 else 'ies'} "
          f"| chunks {a.chunks}{' (test)' if a.test else ''} | out {os.path.abspath(a.out)}", flush=True)

    jobs = []
    for e in entries:
        plan = gpu_plan(e, a.chunks, a.test, step, only)
        gdir = os.path.join(a.out, e["name"])
        work = os.path.join(gdir, "work")
        units = make_units(plan, n_gpus, a.units_per_gpu, a.max_lanes)
        if not any(c["lams"] for c in plan["chunks"]):
            raise SystemExit("no wavelengths to solve (check --lams against the chunk ranges and --step)")
        print_plan(plan, units, n_gpus)
        if a.dry_run:
            continue
        plan["mpbx_commit"] = mpbx_commit
        stored = results_io.stored_setup(gdir, e["name"])
        if stored is not None and stored != results_io.setup_signature(plan):
            raise SystemExit(f"{gdir} holds results of a different setup (chunks, grids, step, ...); use a new --out folder")
        if results_io.is_complete(gdir, e["name"]) and not a.merge_only:
            print(f"  complete already ({gdir}); skipped")
            continue
        plan_path = os.path.join(work, "plan.json")
        if a.merge_only and not os.path.exists(plan_path):
            print(f"  --merge-only: nothing in {work}; skipped")
            continue
        os.makedirs(work, exist_ok=True)
        write_plan = True
        if os.path.exists(plan_path):                         # a resubmission must not change the setup
            old = json.load(open(plan_path))
            if results_io.setup_signature(old) != results_io.setup_signature(plan):
                raise SystemExit(f"{plan_path} describes a different setup; use a new --out folder (or delete {work})")
            if old.get("mpbx_commit") and old.get("mpbx_commit") != mpbx_commit:
                w = (f"mpbx (the solver) changed since this geometry was started: {old.get('mpbx_commit')} -> "
                     f"{mpbx_commit}; the per-λ files may mix the two versions")
                print(f"  WARNING {w}")
                plan["warnings"].append(w)
            plan["warnings"] += [x for x in old.get("warnings", []) if x not in plan["warnings"]]
            if only is not None and not old.get("lams_subset"):
                write_plan = False
                plan = dict(old, warnings=plan["warnings"], mpbx_commit=mpbx_commit)
        if write_plan:
            with open(plan_path, "w") as f:
                json.dump(plan, f, indent=1)
        jobs.append((plan, gdir, work, [] if a.merge_only else units))
    if a.dry_run or not jobs:
        return

    t0 = time.perf_counter()
    failed, notes, ustats = [], {}, {}
    left = {plan["name"]: len(units) for plan, _, _, units in jobs}
    args_line = " ".join(sys.argv[1:])

    def finish(plan, gdir, work):
        name = plan["name"]
        recs = []
        try:                                                  # timing is a by-product: never blocks the merge
            recs = consolidate_timing(gdir, name, work)
        except Exception as e:                                # noqa: BLE001
            print(f"  [{name}] timing files not consolidated: {e!r}", flush=True)
        try:
            info = results_io.merge_geometry(gdir, plan, notes.get(name, []), failed=[f for f in failed if f[0] == name])
            st = ustats.get(name, [])
            if st:                                            # (not on --merge-only: keep the solving run's report)
                wall = max(s["t_end"] for s in st) - min(s["t_start"] for s in st)
                write_report(gdir, name, plan, recs, st, wall, n_gpus, a.platform, args_line)
            print(f"  [{name}] merged -> {gdir} ({info}) after {(time.perf_counter() - t0) / 3600:.3f} h", flush=True)
            if info.get("complete") and not a.keep_work:
                shutil.rmtree(work, ignore_errors=True)
        except Exception as e:                                # noqa: BLE001
            failed.append((name, "merge", repr(e)))
            print(f"  [{name}] MERGE FAILED {e!r} (the per-λ files are kept in {work})", flush=True)

    todo = [(plan, gdir, work, u) for plan, gdir, work, units in jobs for u in units]
    for plan, gdir, work, units in jobs:
        if not units:
            finish(plan, gdir, work)
    if todo:
        ctx = mp.get_context("spawn")                         # never fork a process that holds JAX
        q = ctx.Queue()
        for dv, cs in zip(devs, cpu_sets):
            q.put((dv, cs))
        ex = cf.ProcessPoolExecutor(n_gpus, mp_context=ctx, initializer=_init_worker,
                                    initargs=(q, a.platform, G.MEM_FRACTION, a.max_slots))
        try:
            futs = {ex.submit(run_unit, u, plan, work): (plan, gdir, work, u) for plan, gdir, work, u in todo}
            n_fin = 0
            for fu in cf.as_completed(futs):
                plan, gdir, work, u = futs[fu]
                n_fin += 1
                tag = (f"{plan['name']} nev {u['nev']} "
                       + ", ".join(f"{r['chunk'][:1]} {r['lams'][0]:.3f}->{r['lams'][-1]:.3f}" for r in u["lanes"]))
                try:
                    n, nt, fl, st = fu.result()
                    notes.setdefault(plan["name"], []).extend(nt)
                    ustats.setdefault(plan["name"], []).append(st)
                    for lam, err in fl:
                        failed.append((plan["name"], f"λ={lam:.3f}", err))
                    print(f"  [{n_fin}/{len(todo)}] {tag}: {n} solved on device {st['dev']}"
                          f"{f', {len(fl)} FAILED (see the worker traceback above)' if fl else ''}"
                          f" ({time.perf_counter() - t0:.0f} s, peak {st['peak_gpu_gb']:.2f} GiB)", flush=True)
                except BrokenProcessPool:
                    raise
                except Exception as e:                        # noqa: BLE001
                    failed.append((plan["name"], tag, repr(e)))
                    print(f"  [{n_fin}/{len(todo)}] {tag}: FAILED {e!r}", flush=True)
                left[plan["name"]] -= 1
                if left[plan["name"]] == 0:
                    finish(plan, gdir, work)
        except BaseException:
            ex.shutdown(wait=False, cancel_futures=True)
            raise
        ex.shutdown(wait=True)
    print(f"[run_geometry_gpu] finished in {(time.perf_counter() - t0) / 3600:.3f} h", flush=True)
    if failed:
        for f in failed:
            print("  FAILED " + " | ".join(f))
        sys.exit(1)


if __name__ == "__main__":
    main()
