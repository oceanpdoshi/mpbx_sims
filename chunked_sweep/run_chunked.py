"""Chunked, parallel wavelength sweep of ONE TFLN rib geometry on CPU (one cluster task).

    python run_chunked.py --out results/sweep_A               # every CPU this task may use
    python run_chunked.py --out ... --cores-per-worker 4       # worker size (README section 4)
    python run_chunked.py --out ... --dry-run                  # print grids / work units, solve nothing
    python run_chunked.py --out ... --test                     # tiny config (minutes) to check the pipeline
    python run_chunked.py --out ... --geom '{"w_top": 1.6, "t_clad": 0}'   # RibGeometry overrides
    python run_chunked.py --out ... --step 0.05                # coarser wavelength step (must divide each chunk)

What it does
  * 0.4-2.0 µm is split into CHUNKS. Each chunk has its own window (Lx, Ly), lateral pitch dx,
    number of bands per x-parity sector (nev) and target modes; the numbers and the evidence
    behind them are in README.md section 2 (study_chunks.py / analyze_studies.py).
  * Every chunk uses the SAME fine vertical pitch dy and the same 1/3-pixel alignment of the
    film interfaces. Every wavelength is solved twice, on the fine grid (dx, dy) and on a coarse
    grid: refine="xy" (default) uses (2dx, 2dy), refine="y" uses (dx, 2dy). combine_chunks.py
    Richardson-extrapolates n_eff and n_g, q = q_f + (q_f - q_c)/3. With "xy" this removes the
    lateral and vertical h² errors together (README §2.3); with "y" only the vertical one, and
    the lateral error is estimated from a model.
  * Both x-mirror parity sectors are solved separately (odd Nx): TE00 / TM00 never compete in
    one eigensolve, crossings between the sectors are exact, and nev counts per sector.
  * Work unit = a contiguous run of wavelengths of one chunk at one level, solved long -> short
    with warm start. Units go to a pool of worker processes, each pinned to its own
    `cores_per_worker` CPUs (so XLA's thread pool matches) before JAX is imported.
  * Every wavelength is written to <out>/<chunk>/<level>/lam_XXXXX.XXnm.h5 (tfln_modes.ModeSweep
    format). A resubmitted task skips files that exist AND were made with the same chunk,
    level, grid and geometry; a file from a different configuration stops the run.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import dataclasses
import json
import math
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures.process import BrokenProcessPool

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PATHS = [ROOT, os.path.join(ROOT, "handoff_tfln_modesolve")]
ALL_TARGETS = ("TE00", "TM00", "TE10", "TM10", "TE20", "TM20", "TE30", "TM30")


# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class Chunk:
    name: str
    lam_min: float          # µm, inclusive
    lam_max: float          # µm, inclusive (a boundary wavelength is solved by both chunks)
    Lx: float               # window [µm] (rounded up to an odd, FFT-friendly pixel count)
    Ly: float
    dx: float               # fine lateral pitch [µm]; matters for the m = 2, 3 modes (README §2.4)
    dy: float               # fine vertical pitch [µm]
    nev: int                # bands per parity sector
    targets: tuple = ALL_TARGETS
    refine: str = "xy"      # coarse level: "xy" = (2dx, 2dy) (removes the dx² and dy² errors),
                            #               "y"  = (dx, 2dy)  (dy² only; dx error stays, modelled)

    def lams(self, step):
        span = self.lam_max - self.lam_min
        n = int(round(span / step))
        if abs(n * step - span) > 1e-9:
            raise ValueError(f"{self.name}: step {step} µm does not divide {self.lam_min}-{self.lam_max} µm")
        return [round(self.lam_min + i * step, 6) for i in range(n + 1)]


# From the studies in README.md section 2 (this geometry: 400 nm film, fully etched, 1.8 µm top,
# 73°, fully clad). Windows keep the window error of every target below ~2e-5 in n_g at the
# chunk's longest λ; nev = deepest target's rank in its sector + 1 (+1 guard where cheap).
LOW = ("TE00", "TM00", "TE10", "TM10")
CHUNKS = (
    Chunk("A_400-600", 0.40, 0.60, Lx=4.2, Ly=3.0, dx=0.0125, dy=0.0125, nev=7),
    Chunk("B_600-1000", 0.60, 1.00, Lx=4.4, Ly=3.2, dx=0.0125, dy=0.0125, nev=5),
    Chunk("C_1000-1500", 1.00, 1.50, Lx=5.1, Ly=4.2, dx=0.0125, dy=0.0125, nev=2, targets=LOW),
    # dx = 25 nm with refine="y" would be ~40 % cheaper here, but at 2 µm it is not in the
    # asymptotic range (TE10/TM10 off by 1.3-1.6e-4); "xy" costs ~+10 % of the whole sweep.
    Chunk("D_1500-2000", 1.50, 2.00, Lx=7.8, Ly=6.4, dx=0.0125, dy=0.0125, nev=2, targets=LOW),
)
STEP = 0.010                # µm

TEST_CHUNKS = (             # --test: coarse and short, only to exercise the pipeline
    Chunk("A_test", 0.50, 0.60, Lx=4.2, Ly=2.4, dx=0.05, dy=0.025, nev=6),
    Chunk("C_test", 1.50, 1.60, Lx=6.2, Ly=5.0, dx=0.05, dy=0.025, nev=2, targets=LOW, refine="y"),
)
TEST_STEP = 0.050

LEVELS = (("fine", 1), ("coarse", 2))      # (name, pitch multiplier)
ALIGN_FRAC = 1 / 3
STORE_FIELDS = {"fine": True, "coarse": False}


# --------------------------------------------------------------------------------------
# Grids: odd Nx (parity), Ny divisible by 4 (so the coarse level has an even Ny), FFT-friendly
# --------------------------------------------------------------------------------------
def _smooth(n, primes=(2, 3, 5, 7, 11, 13, 17, 19)):
    for p in primes:
        while n % p == 0:
            n //= p
    return n == 1


def pick_nx(L, d):
    """Smallest odd 19-smooth pixel count with N·d >= L (no large prime factors: fast FFTs)."""
    n = int(math.ceil(L / d - 1e-9))
    n += 1 - n % 2
    while not _smooth(n):
        n += 2
    return n


def pick_ny(L, d):
    """Smallest 19-smooth multiple of 4 with N·d >= L."""
    n = int(math.ceil(L / d - 1e-9))
    n += (-n) % 4
    while not _smooth(n):
        n += 4
    return n


def chunk_grid(chunk, level_mult):
    """(Nx, Ny, dx, dy) for a chunk at a pitch level (1 = fine, 2 = coarse).
    Ly is identical on both levels (Ny_fine is a multiple of 4). With refine="xy" the coarse
    level has Nx_c odd at 2·dx and the fine level Nx_f = 2·Nx_c ± 1 or ± 3 (also odd and smooth),
    so the two windows differ by at most 3 fine pixels in x (37.5 nm at dx = 12.5 nm), far below
    what the window error resolves."""
    Ny_f = pick_ny(chunk.Ly, chunk.dy)
    Ny = Ny_f // level_mult
    dy = chunk.dy * level_mult
    if chunk.refine == "y":
        return pick_nx(chunk.Lx, chunk.dx), Ny, chunk.dx, dy
    if chunk.refine != "xy":
        raise ValueError(f"{chunk.name}: refine must be 'xy' or 'y'")
    nc = pick_nx(chunk.Lx, 2 * chunk.dx)
    while True:   # fine count within 3 fine pixels of 2·Nx_c, odd and smooth
        nf = next((n for n in (2 * nc + 1, 2 * nc - 1, 2 * nc + 3, 2 * nc - 3) if _smooth(n)), None)
        if nf is not None:
            break
        nc = pick_nx((nc + 2) * 2 * chunk.dx, 2 * chunk.dx)
    return (nf, Ny, chunk.dx, dy) if level_mult == 1 else (nc, Ny, 2 * chunk.dx, dy)


def check_geometry(geom_kwargs, chunks):
    """Richardson needs every horizontal interface at the same sub-pixel fraction on both
    levels: align_to_grid pins the LN/substrate interface, so the film, slab and a finite
    cladding thickness must be whole multiples of the COARSE pitch 2·dy."""
    sys.path[:0] = [p for p in PATHS if p not in sys.path]
    from utils.setup_jax import setup_jax
    setup_jax("cpu", verbose=False)                               # tfln_rib imports mpbjax (no solve here)
    from geometry_gen.tfln_rib import RibGeometry
    g = RibGeometry(**geom_kwargs)
    problems = []
    for c in chunks:
        d = 2 * c.dy
        for name, t in (("t_ln", g.t_ln), ("slab (t_ln - etch_depth)", g.t_slab), ("t_clad", g.t_clad)):
            if t > 0 and math.isfinite(t) and abs(t / d - round(t / d)) > 1e-6:
                problems.append(f"{c.name}: {name} = {t} µm is not a multiple of 2·dy = {d} µm")
        if c.refine == "xy" and g.etch_depth > 0:
            run = g.etch_depth / math.tan(math.radians(g.sidewall_deg)) if g.sidewall_deg < 90 else 0.0
            dxc = 2 * c.dx
            if run < 2 * dxc and abs(g.w_top / (2 * dxc) - round(g.w_top / (2 * dxc))) > 1e-6:
                print(f"[run_chunked] WARNING {c.name}: near-vertical sidewall (horizontal run {run * 1e3:.0f} nm) "
                      f"and w_top/2 not a multiple of the coarse dx {dxc * 1e3:.0f} nm: the x-Richardson may "
                      f"zig-zag; prefer w_top a multiple of {2 * dxc * 1e3:.0f} nm")
    return g, problems


# --------------------------------------------------------------------------------------
# Work units and a rough cost model (for scheduling only)
# --------------------------------------------------------------------------------------
def make_units(chunks, step, n_workers, units_per_worker=3, only=None):
    pieces = []
    for c in chunks:
        lams = sorted(c.lams(step), reverse=True)            # long -> short (warm start direction)
        if only:
            lams = [l for l in lams if any(abs(l - o) < 1e-9 for o in only)]
            if not lams:
                continue
        for level, mult in LEVELS:
            Nx, Ny, _, _ = chunk_grid(c, mult)
            cost = [Nx * Ny * (2 * c.nev) ** 2 * (1.0 + (0.6 / lam) ** 2) for lam in lams]
            pieces.append((c, level, lams, cost))
    total = sum(sum(p[3]) for p in pieces)
    target = total / (n_workers * units_per_worker) if units_per_worker > 0 else float("inf")  # 0: never split
    units = []
    for c, level, lams, cost in pieces:
        run, acc, first = [], 0.0, len(units)
        for lam, co in zip(lams, cost):
            run.append(lam)
            acc += co
            if acc >= target:
                units.append(dict(chunk=dataclasses.asdict(c), level=level, lams=run, cost=acc))
                run, acc = [], 0.0
        if run:
            if len(units) > first and acc < 0.5 * target:     # fold a short tail into this piece's last run
                units[-1]["lams"] += run
                units[-1]["cost"] += acc
            else:
                units.append(dict(chunk=dataclasses.asdict(c), level=level, lams=run, cost=acc))
    units.sort(key=lambda u: -u["cost"])                     # longest first (LPT scheduling)
    return units, total


# --------------------------------------------------------------------------------------
# Worker process
# --------------------------------------------------------------------------------------
_W = {}


def _pin(cores):
    if hasattr(os, "sched_setaffinity"):                      # Linux (cluster)
        os.sched_setaffinity(0, cores)
    else:                                                     # Windows / macOS test runs
        try:
            import psutil
            psutil.Process().cpu_affinity(list(cores))
        except Exception as e:                                # noqa: BLE001
            print(f"[worker] could not pin to {cores}: {e}", flush=True)
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[v] = str(len(cores))


def _init_worker(core_queue, geom_overrides, out_dir):
    cores = core_queue.get()
    _pin(cores)
    sys.path[:0] = PATHS
    from utils.setup_jax import setup_jax
    setup_jax("cpu", verbose=False)
    import jax
    import numpy as np
    from geometry_gen import tfln_rib as R
    import tfln_modes as M
    _W.update(cores=cores, R=R, M=M, np=np, jax=jax, geom=R.RibGeometry(**geom_overrides),
              grid_key=None, timing=os.path.join(out_dir, f"timing_{os.getpid()}.jsonl"))


def node_labels(E, te_frac):
    """(pol, m, n) per mode from sign changes of the dominant transverse E component along a
    horizontal line through the film and a vertical line through the field maximum."""
    np = _W["np"]
    out = []
    for Ej, te in zip(E, te_frac):
        F = Ej[0] if te >= 0.5 else Ej[1]
        i0 = np.unravel_index(np.argmax(np.abs(F)), F.shape)
        F = np.real(F * np.exp(-1j * np.angle(F[i0])))
        jy = int(np.argmax((F ** 2).sum(axis=0)))

        def changes(line):
            a = np.abs(line)
            s = np.sign(line[a > 0.03 * a.max()])
            return int(np.sum(s[1:] != s[:-1]))

        out.append(("TE" if te >= 0.5 else "TM", changes(F[:, jy]), changes(F[i0[0], :])))
    return out


def _existing_ok(path, chunk_dict, level, Nx, Ny, geom):
    """True if ``path`` holds a finished solve of this exact configuration; raises if it holds a
    different one; False if it is unreadable (it is then solved again)."""
    import h5py
    try:
        with h5py.File(path, "r") as f:
            meta = json.loads(f.attrs["meta"])
    except Exception:                                         # noqa: BLE001  (truncated / corrupt)
        return False
    g0 = {k: v for k, v in dataclasses.asdict(geom).items() if k != "y_bot"}
    g1 = {k: v for k, v in meta.get("geometry", {}).items() if k != "y_bot"}
    same = (meta.get("chunk") == chunk_dict and meta.get("level") == level and g0 == g1
            and meta.get("align_frac") == ALIGN_FRAC
            and meta.get("grid", {}).get("Nx") == Nx and meta.get("grid", {}).get("Ny") == Ny)
    if not same:
        raise RuntimeError(f"{path} was solved with a different configuration; use a new --out folder "
                           f"(or delete the old files)")
    return True


def run_unit(unit, out_dir):
    R, M, np, jax = _W["R"], _W["M"], _W["np"], _W["jax"]
    from mpbjax import grid_2d
    c = Chunk(**{**unit["chunk"], "targets": tuple(unit["chunk"]["targets"])})
    mult = dict(LEVELS)[unit["level"]]
    Nx, Ny, dx, dy = chunk_grid(c, mult)
    if _W["grid_key"] not in (None, (Nx, Ny, dx, dy, c.nev)):
        jax.clear_caches()                                    # drop compiled programs of the previous grid
    _W["grid_key"] = (Nx, Ny, dx, dy, c.nev)
    grid = grid_2d(Nx * dx, Ny * dy, Nx, Ny)
    geom = R.align_to_grid(_W["geom"], grid, frac=ALIGN_FRAC, warn=False)
    diel = R.RibDielectric(geom, grid)
    st = M.SolverSettings(nev=c.nev, use_parity=True, warm_start=True, label_modes=False,
                          store_fields=True, field_dtype="complex64")
    d = os.path.join(out_dir, c.name, unit["level"])
    os.makedirs(d, exist_ok=True)
    chunk_dict = dataclasses.asdict(c)
    chunk_dict["targets"] = list(c.targets)
    warm, n_done = None, 0
    for lam in unit["lams"]:
        path = os.path.join(d, f"lam_{lam * 1e3:08.2f}nm.h5")
        if os.path.exists(path) and _existing_ok(path, chunk_dict, unit["level"], Nx, Ny, _W["geom"]):
            warm = None                                       # chain broken: next solve starts cold
            continue
        t0 = time.perf_counter()
        sw, warm = M.solve_wavelength(diel, lam, st, warm=warm)
        labs = node_labels(sw.E[0], sw.data["te_frac"][0])
        sw.data["hg_pol"] = np.array([[p for p, _, _ in labs]])
        sw.data["hg_m"] = np.array([[m for _, m, _ in labs]])
        sw.data["hg_n"] = np.array([[n for _, _, n in labs]])
        sw.data["hg_err"] = np.full((1, len(labs)), np.nan)
        sw.meta.update(chunk=chunk_dict, level=unit["level"], align_frac=ALIGN_FRAC,
                       label_method="node count of the dominant transverse E component (hg_err = nan)",
                       cores=list(_W["cores"]))
        if not STORE_FIELDS[unit["level"]]:
            sw.E = None
        tmp = f"{path}.{os.getpid()}.part"
        sw.save(tmp)
        os.replace(tmp, path)                                 # never leave a half-written .h5 under the final name
        n_done += 1
        with open(_W["timing"], "a") as tf:                   # one line per λ, survives a time-limit kill
            tf.write(json.dumps(dict(chunk=c.name, level=unit["level"], lam=lam, t=time.perf_counter() - t0,
                                     Nx=Nx, Ny=Ny, nev=c.nev, converged=bool(sw.data["converged"].all()),
                                     cores=len(_W["cores"]))) + "\n")
    return n_done


# --------------------------------------------------------------------------------------
def available_cpus():
    """CPUs this task may use: the affinity mask, capped by SLURM_CPUS_PER_TASK when the
    scheduler limits CPUs by quota instead of by cpuset."""
    cpus = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else list(range(os.cpu_count()))
    n_slurm = os.environ.get("SLURM_CPUS_PER_TASK")
    if n_slurm and n_slurm.isdigit():
        cpus = cpus[:int(n_slurm)]
    return cpus


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cores-per-worker", type=int, default=4)
    ap.add_argument("--max-cores", type=int, default=None, help="default: every CPU this task may use")
    ap.add_argument("--geom", default="{}", help="JSON dict of RibGeometry overrides")
    ap.add_argument("--step", type=float, default=None, help="wavelength step [µm] (default 0.010)")
    ap.add_argument("--lams", default=None,
                    help="comma-separated subset of wavelengths [µm] to solve (each must lie on its chunk's grid)")
    ap.add_argument("--units-per-worker", type=int, default=3,
                    help="work units per worker (more = better balance, fewer = longer warm-start chains; "
                         "0 = one unit per chunk and level)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--test", action="store_true")
    a = ap.parse_args()
    for stream in (sys.stdout, sys.stderr):                   # λ / µm on a Windows console
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:                                     # noqa: BLE001
            pass

    chunks = TEST_CHUNKS if a.test else CHUNKS
    step = a.step or (TEST_STEP if a.test else STEP)
    geom_kw = json.loads(a.geom)
    geom, problems = check_geometry(geom_kw, chunks)          # also validates the --geom keys
    if problems:
        raise SystemExit("geometry incompatible with the two-level Richardson grid:\n  " + "\n  ".join(problems))
    allowed = available_cpus()
    if a.max_cores:
        allowed = allowed[:a.max_cores]
    cpw = max(1, min(a.cores_per_worker, len(allowed)))
    n_workers = len(allowed) // cpw
    core_sets = [allowed[i * cpw:(i + 1) * cpw] for i in range(n_workers)]
    only = [float(x) for x in a.lams.split(",")] if a.lams else None
    units, total = make_units(chunks, step, n_workers, a.units_per_worker, only)
    if not units:
        raise SystemExit("no wavelengths to solve (check --lams against the chunk ranges and --step)")

    plan = dict(chunks=[{**dataclasses.asdict(c), "targets": list(c.targets)} for c in chunks], step=step,
                levels=LEVELS, align_frac=ALIGN_FRAC, geometry=dataclasses.asdict(geom), n_workers=n_workers,
                cores_per_worker=cpw, grids={c.name: {lv: chunk_grid(c, m) for lv, m in LEVELS} for c in chunks},
                units=units)
    print(f"[run_chunked] {len(allowed)} CPUs -> {n_workers} workers x {cpw} cores; {len(units)} work units")
    print(f"  {geom.describe()}")
    for c in chunks:
        for lv, m in LEVELS:
            Nx, Ny, dx, dy = chunk_grid(c, m)
            print(f"  {c.name:12s} {lv:6s} {len(c.lams(step)):3d} λ  {Nx}x{Ny} px = {Nx * dx:.3f} x {Ny * dy:.3f} µm "
                  f"(dx {1e3 * dx:.2f}, dy {1e3 * dy:.2f} nm)  nev {c.nev}/sector  targets {','.join(c.targets)}")
    if a.dry_run:
        for u in units:
            print(f"    unit {u['chunk']['name']:12s} {u['level']:6s} λ {u['lams'][0]:.3f}->{u['lams'][-1]:.3f} "
                  f"({len(u['lams'])})  rel. cost {u['cost'] / total:.3f}")
        return
    os.makedirs(a.out, exist_ok=True)
    plan_path = os.path.join(a.out, "plan.json")
    if os.path.exists(plan_path):                             # a resubmission must not change the setup
        old = json.load(open(plan_path))
        if (old["chunks"] != plan["chunks"] or old["geometry"] != plan["geometry"] or old["step"] != step
                or old.get("align_frac") != ALIGN_FRAC):
            raise SystemExit(f"{plan_path} describes a different run (chunks, geometry or step); use a new --out")
    with open(plan_path, "w") as f:
        json.dump(plan, f, indent=1)

    ctx = mp.get_context("spawn")                             # never fork a process that holds JAX
    q = ctx.Queue()
    for cs in core_sets:
        q.put(cs)
    t0 = time.perf_counter()
    done, failed = 0, []
    ex = cf.ProcessPoolExecutor(n_workers, mp_context=ctx, initializer=_init_worker,
                                initargs=(q, geom_kw, a.out))
    try:
        futs = {ex.submit(run_unit, u, a.out): u for u in units}
        for fu in cf.as_completed(futs):
            u = futs[fu]
            tag = f"{u['chunk']['name']} {u['level']} {u['lams'][0]:.3f}->{u['lams'][-1]:.3f} µm"
            try:
                n = fu.result()
                done += 1
                print(f"  [{done + len(failed)}/{len(units)}] {tag}: {n} solved ({time.perf_counter() - t0:.0f} s)",
                      flush=True)
            except BrokenProcessPool:                         # a worker died (OOM kill, ...): stop
                raise
            except Exception as e:                            # noqa: BLE001  one bad unit must not stop the rest
                failed.append((tag, repr(e)))
                print(f"  [{done + len(failed)}/{len(units)}] {tag}: FAILED {e!r}", flush=True)
    except BaseException:                                     # Ctrl-C, broken pool, time-limit signal
        ex.shutdown(wait=False, cancel_futures=True)
        raise
    ex.shutdown(wait=True)
    print(f"[run_chunked] {done}/{len(units)} units done in {time.perf_counter() - t0:.0f} s "
          f"-> python combine_chunks.py {a.out}")
    if failed:
        for tag, err in failed:
            print(f"  FAILED {tag}: {err}")
        sys.exit(1)


if __name__ == "__main__":
    main()
