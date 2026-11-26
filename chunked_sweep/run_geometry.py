"""Mode-solve 0.4-2.0 µm for one or more TFLN rib geometries (= one cluster task) and write
one set of result files per geometry.

    python run_geometry.py --out OUT                                   the default geometry
    python run_geometry.py --out OUT --geom '{"w_top": 1.6, "t_clad": 0}'
    python run_geometry.py --out OUT --geom '{"w_top": 1.0, "sidewall_deg": 70, "t_clad": 0, "t_undercut": 1.0}'
                                    an undercut rib (t_undercut > 0: SiO2 etched 1 µm below the LN)
    python run_geometry.py --out OUT --geom-csv geometries.csv --row 3          one row (0-based)
    python run_geometry.py --out OUT --geom-csv geometries.csv --task 2 --n-tasks 10
                                    rows 2, 12, 22, ... (one task of a multi-geometry sweep)
  options
    --dry-run              print grids, work units, cost / memory / file-size estimates; solve nothing
    --test                 tiny configuration (sweep_config.TEST_CHUNKS), minutes on a laptop
    --fields none          do not store E fields (default: all)
    --cores-per-worker 2   CPUs per worker process (README §3)
    --max-cores N          use at most N of the CPUs this task may use
    --merge-only           only (re)write the result files from what is already solved
    --keep-work            keep the per-wavelength files after a successful merge
    --lams 0.5,1.55        only these wavelengths (tests; such a run is never marked complete)

Folder layout (README §5):
    OUT/<geometry name>/<geometry name>_modes.h5        all wavelengths, all modes, no fields
                       /<geometry name>_modes.csv       the same, one row per (λ, mode)
                       /<geometry name>_tracks.csv      TE00 ... TM30 vs λ
                       /<geometry name>_fields_<label>.h5   E fields: one file per sweep_config.FIELD_FILES
                                                        group, one HDF5 group per chunk inside
                       /<geometry name>_summary.txt     grids, flags, chunk seams, timing
                       /<geometry name>_dispersion.png
                       /work/                           per-λ files while running (deleted after
                                                        a successful merge unless --keep-work)
A resubmitted task skips geometries that are complete and wavelengths that are already solved; a
folder holding results of a different setup (chunks, grids, step, fields, ...) is refused.

How the work is split: each geometry's wavelengths are cut into contiguous runs inside one chunk
("work units"), solved from long to short λ so every solve after the first starts from the
previous one (warm start, ~2x faster). Units go to a pool of worker processes, each pinned to
its own cores before JAX is imported. Geometries are queued one after the other, so the workers
start on the next geometry while the last units of the previous one finish; each geometry is
merged into its result files as soon as all of its units are done.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import csv
import dataclasses
import datetime
import io
import json
import math
import multiprocessing as mp
import os
import shutil
import sys
import time
from concurrent.futures.process import BrokenProcessPool

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PATHS = [HERE, ROOT]                                      # ROOT: sim_core/, geometry_gen/, utils/
sys.path[:0] = [p for p in PATHS if p not in sys.path]
# SuperCloud's shared file system has no file locking; HDF5 refuses to write there without this
# (workers inherit it). Set before h5py is imported anywhere.
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import sweep_config as C  # noqa: E402

SMOOTH_PRIMES = (2, 3, 5, 7, 11, 13)
SHRINK = 0.98                    # a window may come out at most 2 % smaller than asked
COST_S = 1.0e-4                  # s per (pixel x band²) for one warm λ on one desktop core (README §3)
GEOM_KEYS = ("t_ln", "etch_depth", "w_top", "sidewall_deg", "t_clad")
# Optional geometry key / CSV column, the rib-type toggle (tfln_rib.make_geometry): missing, empty or 0
# -> regular rib (RibGeometry); > 0 -> undercut rib (RibGeometryUndercut: the SiO2 is etched this deep
# below the LN at the sidewall angle; needs t_clad = 0 and etch_depth = t_ln). A plan's geometry holds
# it only for an undercut rib, so regular-rib plans (and the runs they resume) are unchanged.
UNDERCUT_KEY = "t_undercut"


# --------------------------------------------------------------------------------------
# Geometry list
# --------------------------------------------------------------------------------------
def _fmt_nm(x):
    if math.isinf(x):
        return "inf"
    s = f"{x * 1e3:.2f}".rstrip("0").rstrip(".")
    return s


def geometry_name(g):
    clad = "SiO2" if math.isinf(g["t_clad"]) else ("air" if g["t_clad"] == 0 else f"{_fmt_nm(g['t_clad'])}nmSiO2")
    uc = g.get(UNDERCUT_KEY) or 0
    return (f"tLN{_fmt_nm(g['t_ln'])}_etch{_fmt_nm(g['etch_depth'])}_w{_fmt_nm(g['w_top'])}"
            f"_sw{g['sidewall_deg']:g}_clad{clad}" + (f"_undercut{_fmt_nm(uc)}" if uc else ""))


def plan_geometry(params):
    """The geometry dict a plan stores (and result files compare against): GEOM_KEYS, plus
    t_undercut only for an undercut rib. ``params``: e.g. a geometry's ``as_dict()``."""
    g = {k: params[k] for k in GEOM_KEYS}
    if params.get(UNDERCUT_KEY):
        g[UNDERCUT_KEY] = params[UNDERCUT_KEY]
    return g


def geometry_kind(g):
    """'rib' or 'undercut rib (...)' for a plan geometry dict (log lines, submit scripts)."""
    uc = g.get(UNDERCUT_KEY) or 0
    return f"undercut rib (SiO2 etched {uc:g} µm below the LN, no cladding)" if uc else "rib"


def _undercut_row(g, uc_cell):
    """Apply the optional t_undercut cell of a CSV row to its geometry dict ``g`` (in place), with the
    checks RibGeometryUndercut makes, here without importing JAX (submit scripts run this on the
    login node). Returns an error message or None."""
    if uc_cell in (None, ""):
        return None
    uc = float(uc_cell)
    if uc == 0:
        return None
    if not (math.isfinite(uc) and uc > 0):
        return f"t_undercut must be >= 0 (0 = regular rib), got {uc_cell}"
    if g["t_clad"] != 0:
        return f"an undercut rib (t_undercut = {uc:g} > 0) has no top cladding: set t_clad = 0 (got {g['t_clad']:g})"
    if abs(g["etch_depth"] - g["t_ln"]) > 1e-9:
        return (f"an undercut rib (t_undercut = {uc:g} > 0) must be fully etched: set etch_depth = t_ln "
                f"(got etch_depth {g['etch_depth']:g}, t_ln {g['t_ln']:g})")
    g["etch_depth"] = g["t_ln"]                  # what RibGeometryUndercut stores (it snaps float noise)
    g[UNDERCUT_KEY] = uc
    return None


def read_geometry_csv(path):
    """Rows of a geometry list: columns t_ln, etch_depth, w_top, sidewall_deg, t_clad [µm, deg]
    (t_clad: inf = fully clad, 0 = air), optional 'name', optional 't_undercut' [µm] (empty or 0 =
    regular rib; > 0 = undercut rib, see UNDERCUT_KEY). Blank lines and lines starting with
    # are ignored."""
    try:
        with open(path, newline="", encoding="utf-8-sig") as f:          # utf-8-sig: also a BOM from Excel
            raw = f.read()
    except UnicodeDecodeError:                                           # e.g. a CSV saved as cp1252 on Windows
        with open(path, newline="", encoding="cp1252", errors="replace") as f:
            raw = f.read()
    cells = [[c.replace("﻿", "").strip() for c in row] for row in csv.reader(io.StringIO(raw))]
    cells = [row for row in cells if any(row) and not row[0].startswith("#")]
    if not cells:
        raise SystemExit(f"{path}: no header / rows")
    header, rows = cells[0], []
    for i, row in enumerate(cells[1:]):
        r = {k: v for k, v in zip(header, row) if k}
        missing = [k for k in GEOM_KEYS if not r.get(k)]
        if missing:
            raise SystemExit(f"{path}, data row {i}: missing {missing} (header: {header})")
        g = {k: float(r[k]) for k in GEOM_KEYS}
        err = _undercut_row(g, r.get(UNDERCUT_KEY))
        if err:
            raise SystemExit(f"{path}, data row {i}: {err}")
        rows.append(dict(name=r.get("name") or geometry_name(g), geometry=g))
    names = [r["name"] for r in rows]
    dup = sorted({n for n in names if names.count(n) > 1})
    if dup:
        raise SystemExit(f"{path}: duplicate geometry names {dup} (give the rows a 'name' column)")
    return rows


def select_geometries(a):
    if a.geom_csv:
        rows = read_geometry_csv(a.geom_csv)
        if a.row is not None:
            if not 0 <= a.row < len(rows):
                raise SystemExit(f"--row {a.row}: {a.geom_csv} has {len(rows)} geometries (0-based rows)")
            return [rows[a.row]]
        if a.task is not None:
            if not 0 <= a.task < a.n_tasks:
                raise SystemExit(f"--task {a.task} must be in [0, --n-tasks={a.n_tasks})")
            return rows[a.task::a.n_tasks]
        return rows
    from geometry_gen.tfln_rib import make_geometry
    g = plan_geometry(make_geometry(**json.loads(a.geom)).as_dict())
    return [dict(name=a.name or geometry_name(g), geometry=g)]


# --------------------------------------------------------------------------------------
# Grids
# --------------------------------------------------------------------------------------
def _smooth(n):
    for p in SMOOTH_PRIMES:
        while n % p == 0:
            n //= p
    return n == 1


def _pick_n(L, d, odd):
    """Smallest FFT-friendly pixel count with n·d >= SHRINK·L (odd if requested)."""
    n = max(3, int(math.ceil(SHRINK * L / d - 1e-9)))
    if odd and n % 2 == 0:
        n += 1
    while not _smooth(n):
        n += 2 if odd else 1
    return n


def film_dy(dy, geom):
    """The pitch closest to ``dy`` that makes the LN film a whole number of pixels (README §2.2)."""
    return geom.t_ln / max(1, round(geom.t_ln / dy))


def chunk_grid(chunk, geom):
    """Pixel counts of a chunk for a geometry: the chunk's window rounded to FFT-friendly counts,
    enlarged if it leaves less than MIN_MARGIN·λ_max of cladding beside / above / below the guide.
    An undercut rib's SiO2 pedestal must also end MIN_UNDERCUT_MARGIN inside the window (substrate
    below it, air beside it); the window stays centered on the LN film.
    With FIT_DY_TO_FILM, dy is the chunk's dy adjusted to fit the film (``film_dy``)."""
    margin = C.MIN_MARGIN * chunk.lam_max
    need_x = geom.w_base + 2 * margin
    need_y = geom.t_ln + 2 * margin
    if geom.y_sub < geom.y0:                                  # undercut rib (RibGeometryUndercut)
        need_x = max(need_x, geom.w_footprint + 2 * C.MIN_UNDERCUT_MARGIN)
        need_y = max(need_y, geom.t_ln + 2 * (geom.y0 - geom.y_sub + C.MIN_UNDERCUT_MARGIN))
    dy = film_dy(chunk.dy, geom) if C.FIT_DY_TO_FILM else chunk.dy
    out = {}
    for ax, L, d, need, odd in (("x", chunk.Lx, chunk.dx, need_x, C.USE_PARITY),
                                ("y", chunk.Ly, dy, need_y, False)):
        n = _pick_n(L, d, odd)
        enlarged = n * d < need - 1e-9
        if enlarged:
            n = _pick_n(need / SHRINK, d, odd)
        out[f"N{ax}"], out[f"d{ax}"], out[f"L{ax}"] = n, d, n * d
        out[f"enlarged_{ax}"] = enlarged
    return out


def layer_warnings(geom, grid):
    """The film interfaces sit at the same sub-pixel position only if every horizontal layer is a
    whole number of pixels thick (README §2.2); otherwise the error is less regular (not larger
    on average). ``grid`` is a chunk_grid result. The cladding's top surface is only checked when
    it lies inside the window."""
    dy, w = grid["dy"], []
    clad_in = geom.t_clad < math.inf and 0.5 * geom.t_ln + geom.t_clad < 0.5 * grid["Ly"]
    for name, t, check in (("t_ln", geom.t_ln, True), ("slab", geom.t_slab, True),
                           ("t_clad", geom.t_clad, clad_in),
                           ("t_undercut", getattr(geom, "t_undercut", 0.0), True)):
        if check and 0 < t < math.inf and abs(t / dy - round(t / dy)) > 1e-6:
            w.append(f"{name} = {t:.6g} µm is not a multiple of dy = {1e3 * dy:.3f} nm"
                     + (" (cladding top surface: matters only where the mode reaches it)" if name == "t_clad" else ""))
    return w


def n_bands(nev):
    """(number of solves, modes per solve) of one wavelength."""
    return (2, nev) if C.USE_PARITY else (1, 2 * nev)


def lam_cost(Nx, Ny, nev, lam):
    """Seconds for one warm-started λ on one desktop core (cost ~ pixels x bands², README §3)."""
    sectors, bands = n_bands(nev)
    return COST_S * Nx * Ny * sectors * bands ** 2


def chunk_lams(chunk, step):
    span = chunk.lam_max - chunk.lam_min
    n = int(round(span / step))
    if abs(n * step - span) > 1e-9:
        raise SystemExit(f"{chunk.name}: step {step} µm does not divide {chunk.lam_min}-{chunk.lam_max} µm")
    return [round(chunk.lam_min + i * step, 6) for i in range(n + 1)]


def owner_chunk(lam, chunks):
    """The chunk whose result goes into the tables at ``lam`` (the lower one at a shared boundary)."""
    for c in chunks:
        if c.lam_min - 1e-9 <= lam <= c.lam_max + 1e-9:
            return c.name
    return None


# --------------------------------------------------------------------------------------
# Plan of one geometry
# --------------------------------------------------------------------------------------
def geometry_plan(entry, chunks, step, fields, only=None):
    from geometry_gen.tfln_rib import make_geometry
    geom = make_geometry(**entry["geometry"])
    plan = dict(name=entry["name"], geometry=entry["geometry"], step=step, fields=fields,
                use_parity=C.USE_PARITY, align_frac=C.ALIGN_FRAC, k_tol=C.K_TOL, eig_tol=C.EIG_TOL,
                nev_extra=C.NEV_EXTRA, min_margin=C.MIN_MARGIN, field_dtype=C.FIELD_DTYPE,
                lams_subset=sorted(only) if only else None, chunks=[], warnings=[])
    layer = {}                                                # message -> chunks it applies to
    for c in chunks:
        g = chunk_grid(c, geom)
        lams = chunk_lams(c, step)
        if only:
            lams = [l for l in lams if any(abs(l - o) < 1e-9 for o in only)]
        owned = [l for l in lams if owner_chunk(l, chunks) == c.name]
        plan["chunks"].append(dict(dataclasses.asdict(c), targets=list(c.targets), grid=g, lams=lams, owned=owned))
        for ax in "xy":
            if g[f"enlarged_{ax}"]:
                plan["warnings"].append(f"{c.name}: window enlarged in {ax} to {g[f'L{ax}']:.3f} µm "
                                        f"(MIN_MARGIN = {C.MIN_MARGIN} x λ_max of cladding around the guide"
                                        + (f"; undercut rib: pedestal + MIN_UNDERCUT_MARGIN = {C.MIN_UNDERCUT_MARGIN} µm"
                                           if geom.y_sub < geom.y0 else "") + ")")
        for msg in layer_warnings(geom, g):
            layer.setdefault(msg, []).append(c.name.split("_")[0])
    plan["warnings"] += [f"{'all chunks' if len(n) == len(chunks) else ', '.join(n)}: {msg}"
                         for msg, n in layer.items()]
    return plan


def plan_estimates(plan):
    """Cost [desktop-core-seconds], field bytes and a rough worker memory per chunk."""
    out = []
    for c in plan["chunks"]:
        g = c["grid"]
        cost = sum(lam_cost(g["Nx"], g["Ny"], c["nev"], l) for l in c["lams"])
        nmodes = 2 * c["nev"]
        fbytes = len(c["owned"]) * nmodes * 3 * g["Nx"] * g["Ny"] * 8 if plan["fields"] == "all" else 0
        mem = 0.5e9 + g["Nx"] * g["Ny"] * (900 + 120 * nmodes)       # rough, from the desktop runs
        out.append(dict(chunk=c["name"], n_lam=len(c["lams"]), cost=cost, field_bytes=fbytes, mem=mem))
    return out


def make_units(plan, n_workers, units_per_worker, min_run=3):
    """Contiguous λ runs (long -> short) inside each chunk, about total/(workers·units_per_worker)
    of the geometry's cost each, at least ``min_run`` wavelengths (warm start pays off)."""
    pieces = []
    for c in plan["chunks"]:
        g = c["grid"]
        lams = sorted(c["lams"], reverse=True)
        pieces.append((c, lams, [lam_cost(g["Nx"], g["Ny"], c["nev"], l) for l in lams]))
    total = sum(sum(p[2]) for p in pieces)
    target = total / max(1, n_workers * units_per_worker)
    units = []
    for c, lams, cost in pieces:
        if not lams:
            continue
        n_runs = max(1, min(len(lams) // min_run, round(sum(cost) / target)))
        base, extra = divmod(len(lams), n_runs)              # equal cost per λ inside a chunk:
        i0 = 0                                                # split into runs of equal length
        for k in range(n_runs):
            n = base + (1 if k < extra else 0)
            units.append(dict(chunk=c["name"], lams=lams[i0:i0 + n], cost=sum(cost[i0:i0 + n])))
            i0 += n
    units.sort(key=lambda u: -u["cost"])                     # longest first (LPT scheduling)
    return units


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


def _init_worker(core_queue):
    cores = core_queue.get()
    _pin(cores)
    sys.path[:0] = [p for p in PATHS if p not in sys.path]
    from utils.setup_jax import setup_jax
    setup_jax("cpu", verbose=False)
    import jax
    import numpy as np
    from geometry_gen import tfln_rib as R
    import sim_core.tfln_modes as M
    _W.update(cores=cores, R=R, M=M, np=np, jax=jax, grid_key=None, diel_key=None, diel=None)


def _peak_rss_gb():
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6      # Linux: kB
    except ImportError:
        try:
            import psutil
            return psutil.Process().memory_info().peak_wset / 1e9            # Windows
        except Exception:                                                     # noqa: BLE001
            return float("nan")


def node_labels(E, te_frac):
    """(pol, m, n) per mode from sign changes of the dominant transverse E component along a
    horizontal line through the field maximum and a vertical line through it."""
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


def target_sector(t):
    """x-parity sector (H-field convention of tfln_modes) of a TEmn / TMmn mode."""
    return -1 if (t[:2] == "TE") == (int(t[2]) % 2 == 0) else +1


HYBRID = (0.2, 0.8)     # TE fraction between these = hybrid (same thresholds as results_io's flag)


def _labels_and_missing(sw, targets):
    """Node-count labels of a solve, the targets not found among the guided modes, and of those:
    ``deep_guided`` (the sector's deepest solved mode is still guided, so the target may sit below
    it: re-solving with more modes can find it) and ``collided`` (the sector also holds two guided
    modes with the same label, or a hybrid one: an anticrossing, where the target is almost surely
    present but mislabelled, so more modes would not help)."""
    np = _W["np"]
    te = sw.data["te_frac"][0]
    labs = node_labels(sw.E[0], te)
    names = [f"{p}{m}{n}" for p, m, n in labs]
    guided = sw.data["guided"][0]
    parity = sw.data["parity"][0]
    neff = sw.data["neff"][0]
    found = {nm for nm, g in zip(names, guided) if g}
    missing = [t for t in targets if t not in found]
    deep_guided, collided = [], []
    for t in missing:
        sel = (parity == target_sector(t)) if C.USE_PARITY else np.ones(len(neff), bool)
        if not (sel.any() and guided[sel][np.argmin(neff[sel])]):
            continue                                          # sector's deepest mode is cut off
        js = [j for j in np.flatnonzero(sel) if guided[j]]
        sec_names = [names[j] for j in js]
        if len(set(sec_names)) < len(sec_names) or any(HYBRID[0] < te[j] < HYBRID[1] for j in js):
            collided.append(t)
        else:
            deep_guided.append(t)
    return labs, missing, deep_guided, collided


def _trim_warm(warm, n):
    """Warm-start state of the n highest modes per sector (when going back to fewer modes)."""
    return {p: (om, list(k)[:n], list(ev)[:n], list(ng)[:n]) for p, (om, k, ev, ng) in warm.items()}


def run_unit(unit, plan, work):
    """Solve one work unit; returns (number solved, notes)."""
    R, M, np, jax = _W["R"], _W["M"], _W["np"], _W["jax"]
    from mpbjax import grid_2d
    c = next(ch for ch in plan["chunks"] if ch["name"] == unit["chunk"])
    g = c["grid"]
    gkey = (g["Nx"], g["Ny"], g["dx"], g["dy"])
    if _W["grid_key"] not in (None, gkey):
        jax.clear_caches()                                    # drop compiled programs of the previous grid
    _W["grid_key"] = gkey
    dkey = (plan["name"], c["name"])
    if _W["diel_key"] != dkey:
        grid = grid_2d(g["Lx"], g["Ly"], g["Nx"], g["Ny"])
        geom = R.align_to_grid(R.make_geometry(**plan["geometry"]), grid, frac=plan["align_frac"], warn=False)
        _W["diel"], _W["diel_key"] = R.RibDielectric(geom, grid), dkey
    diel = _W["diel"]
    d = os.path.join(work, c["name"])
    os.makedirs(d, exist_ok=True)
    nev = c["nev"]
    nev_max = nev + plan["nev_extra"]
    store = plan["fields"] == "all"
    notes, warm, n_done = [], None, 0
    timing = os.path.join(work, f"timing_{os.getpid()}.jsonl")

    def settings(n):
        return M.SolverSettings(nev=n if plan["use_parity"] else 2 * n, use_parity=plan["use_parity"],
                                warm_start=True, label_modes=False, store_fields=True,
                                field_dtype=plan["field_dtype"], k_tol=plan["k_tol"], eig_tol=plan["eig_tol"])

    failures = []
    for lam in unit["lams"]:
        path = os.path.join(d, f"lam_{lam * 1e3:08.2f}nm.h5")
        done = _existing_ok(path, plan, c) if os.path.exists(path) else None
        if done:
            warm = None                                       # chain broken: the next solve starts cold
            # keep a raise made before a restart (nev_next: files written before it existed lack it)
            nev = max(nev, int(done.get("nev_next", done.get("nev_per_sector")) or nev))
            continue
        try:                                                  # one failing λ must not stop the rest of the unit
            nev = _solve_one(lam, path, c, g, plan, diel, settings, nev, nev_max, store, warm, notes, timing)
            warm = _W.pop("warm_out")
            n_done += 1
        except Exception as e:                                # noqa: BLE001
            import traceback
            failures.append((lam, repr(e)))
            print(f"[worker {os.getpid()}] {plan['name']} {c['name']} λ={lam:.3f} FAILED:\n{traceback.format_exc()}",
                  flush=True)
            warm = None
    return n_done, notes, failures


def _solve_one(lam, path, c, g, plan, diel, settings, nev, nev_max, store, warm, notes, timing):
    """Solve, label, (maybe re-solve with more modes), save one λ. Returns the nev to use next; the
    warm-start state is left in _W['warm_out'].

    A missing target is re-solved with nev_max modes only if it may sit below the last solved mode
    (``deep``), not if its sector shows a label collision (``collided``: an anticrossing). If the
    re-solve does not find it either, the next λ goes back to ``nev``."""
    M, np = _W["M"], _W["np"]
    if f"{lam:.4f}" in os.environ.get("RUN_GEOMETRY_TEST_FAIL", "").split(","):     # test hook only
        raise RuntimeError(f"test failure injected at λ = {lam}")
    t0 = time.perf_counter()
    sw, warm_new = M.solve_wavelength(diel, lam, settings(nev), warm=warm)
    labs, missing, deep, collided = _labels_and_missing(sw, c["targets"])
    if collided:
        notes.append(f"{c['name']} λ={lam:.3f}: {', '.join(collided)} not labelled; its sector has a "
                     f"duplicate or hybrid label (anticrossing?) -> not re-solved")
    escalated, nev_next = False, nev
    if deep and nev < nev_max:                            # a target may sit below the last solved mode
        escalated = True
        sw, warm_new = M.solve_wavelength(diel, lam, settings(nev_max), warm=None)
        labs, missing, _, _ = _labels_and_missing(sw, c["targets"])
        still = [t for t in deep if t in missing]
        if still:                                         # more modes did not reveal it: go back
            warm_new = _trim_warm(warm_new, nev)
            notes.append(f"{c['name']} λ={lam:.3f}: {', '.join(still)} still missing with nev {nev_max} "
                         f"per sector -> back to nev {nev} for the next λ")
        else:                                             # it was below the cut: keep the extra modes
            nev_next = nev_max
            notes.append(f"{c['name']} λ={lam:.3f}: {', '.join(deep)} found with nev {nev_max} per sector "
                         f"-> kept for λ <= {lam:.3f} in this unit")
        nev = nev_max                                     # what THIS λ was solved with (metadata)
    _W["warm_out"] = warm_new
    sw.data["hg_pol"] = np.array([[p for p, _, _ in labs]])
    sw.data["hg_m"] = np.array([[m for _, m, _ in labs]])
    sw.data["hg_n"] = np.array([[n for _, _, n in labs]])
    sw.data["hg_err"] = np.full((1, len(labs)), np.nan)
    sw.meta.update(geometry_name=plan["name"], chunk=c["name"], align_frac=plan["align_frac"],
                   mpbx_commit=plan.get("mpbx_commit"),
                   nev_per_sector=nev, nev_next=nev_next, nev_escalated=escalated, missing_targets=missing,
                   collided_targets=collided,
                   label_method="node count of the dominant transverse E component",
                   cores=list(_W["cores"]), peak_rss_gb=_peak_rss_gb())
    if not store:
        sw.E = None
    tmp = f"{path}.{os.getpid()}.part"
    sw.save(tmp, compression=None)
    os.replace(tmp, path)                                 # never leave a half-written file under the final name
    with open(timing, "a") as tf:                         # one line per λ, survives a time-limit kill
        tf.write(json.dumps(dict(geometry=plan["name"], chunk=c["name"], lam=lam, t=time.perf_counter() - t0,
                                 Nx=g["Nx"], Ny=g["Ny"], nev=nev, escalated=escalated,
                                 converged=bool(sw.data["converged"].all()), cores=len(_W["cores"]),
                                 peak_rss_gb=_peak_rss_gb())) + "\n")
    return nev_next


def _existing_ok(path, plan, c):
    """The file's metadata (truthy) if ``path`` holds a finished solve of this configuration; raises
    if it holds a different one; None if unreadable (then it is solved again)."""
    import h5py
    try:
        with h5py.File(path, "r") as f:
            meta = json.loads(f.attrs["meta"])
            has_E = "E" in f
    except Exception:                                         # noqa: BLE001  (truncated / corrupt)
        return None
    g0 = {k: v for k, v in plan["geometry"].items()}
    g1 = {k: meta.get("geometry", {}).get(k) for k in g0}
    grid = meta.get("grid", {})
    same = (g0 == g1 and meta.get("chunk") == c["name"] and grid.get("Nx") == c["grid"]["Nx"]
            and grid.get("Ny") == c["grid"]["Ny"] and meta.get("align_frac") == plan["align_frac"]
            and (has_E or plan["fields"] != "all"))
    if not same:
        raise RuntimeError(f"{path} was solved with a different configuration; use a new --out folder "
                           f"(or delete that geometry's work/ folder)")
    return meta


# --------------------------------------------------------------------------------------
# Main process
# --------------------------------------------------------------------------------------
def available_cpus(max_cores=None, cpu_slice=None):
    """CPUs this process may use: its affinity mask (Slurm's cpuset / srun binding). With ``max_cores``
    N: at most N of them, slice ``cpu_slice`` k = CPUs [k·N, (k+1)·N) (LLsub triples, several processes
    per node). Otherwise capped by SLURM_CPUS_PER_TASK, with a warning if the mask was larger."""
    if hasattr(os, "sched_getaffinity"):
        cpus = sorted(os.sched_getaffinity(0))
    else:                                                     # Windows / macOS test runs
        try:
            import psutil
            cpus = sorted(psutil.Process().cpu_affinity())
        except Exception:                                     # noqa: BLE001
            cpus = list(range(os.cpu_count()))
    if max_cores:
        if len(cpus) > max_cores:
            k = cpu_slice or 0
            cpus = cpus[k * max_cores:(k + 1) * max_cores] or cpus[:max_cores]
        return cpus
    n_slurm = os.environ.get("SLURM_CPUS_PER_TASK")
    if n_slurm and n_slurm.isdigit() and len(cpus) > int(n_slurm):
        print(f"[run_geometry] WARNING: this process may use {len(cpus)} CPUs but Slurm gave the task "
              f"{n_slurm}: using the first {n_slurm}; if another task shares the node it may use the same "
              f"cores (both then run ~2x slower). Launch through srun --cpu-bind=cores, or use whole nodes.",
              flush=True)
        cpus = cpus[:int(n_slurm)]
    return cpus


def _gb(x):
    return f"{x / 1e9:.2f} GB"


def print_plan(plan, units, n_workers, cpw):
    print(f"\n=== {plan['name']}")
    from geometry_gen.tfln_rib import make_geometry
    print(f"  {make_geometry(**plan['geometry']).describe()}")
    est = {e["chunk"]: e for e in plan_estimates(plan)}
    tot_cost = sum(e["cost"] for e in est.values())
    for c in plan["chunks"]:
        g, e = c["grid"], est[c["name"]]
        print(f"  {c['name']:15s} {len(c['lams']):3d} λ  {g['Nx']}x{g['Ny']} px = {g['Lx']:.3f} x {g['Ly']:.3f} µm "
              f"(dx {1e3 * g['dx']:.1f}, dy {1e3 * g['dy']:.2f} nm)  nev {c['nev']}/sector  "
              f"cost {e['cost'] / tot_cost:5.1%}  fields {_gb(e['field_bytes'])}  ~{_gb(e['mem'])}/worker")
    for w in plan["warnings"]:
        print(f"  WARNING {w}")
    print(f"  estimate: {tot_cost / 3600:.1f} desktop-core-hours "
          f"(see README §3 for the cluster conversion); {len(units)} work units for {n_workers} workers x {cpw} CPUs")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="output folder (one sub-folder per geometry)")
    ap.add_argument("--geom", default="{}", help="JSON dict of RibGeometry parameters (default geometry otherwise); "
                    "t_undercut > 0 makes it an undercut rib (RibGeometryUndercut)")
    ap.add_argument("--name", default=None, help="name of the --geom geometry (default: built from its parameters)")
    ap.add_argument("--geom-csv", default=None, help="CSV geometry list (see geometries_example.csv)")
    ap.add_argument("--row", type=int, default=None, help="0-based row of --geom-csv")
    ap.add_argument("--task", type=int, default=None, help="this task's index (rows task, task+n_tasks, ...)")
    ap.add_argument("--n-tasks", type=int, default=1)
    ap.add_argument("--fields", choices=("all", "none"), default="all")
    ap.add_argument("--cores-per-worker", type=int, default=2)
    ap.add_argument("--max-cores", type=int, default=None, help="use at most this many of the task's CPUs")
    ap.add_argument("--cpu-slice", type=int, default=None,
                    help="with --max-cores N: use CPUs [k·N, (k+1)·N) of the task's CPUs (LLsub triples)")
    ap.add_argument("--units-per-worker", type=float, default=1.5,
                    help="work units per worker and geometry (more = better balance, shorter warm-start chains)")
    ap.add_argument("--step", type=float, default=None, help="wavelength step [µm] (default sweep_config.STEP)")
    ap.add_argument("--lams", default=None, help="comma-separated subset of wavelengths [µm] (for tests)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--merge-only", action="store_true")
    ap.add_argument("--keep-work", action="store_true")
    a = ap.parse_args()
    for stream in (sys.stdout, sys.stderr):                   # λ / µm on a Windows console
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:                                     # noqa: BLE001
            pass

    from utils.setup_jax import setup_jax
    setup_jax("cpu", verbose=False)                           # tfln_rib imports mpbjax (no solve here)
    import results_io

    chunks = C.TEST_CHUNKS if a.test else C.CHUNKS
    step = a.step or (C.TEST_STEP if a.test else C.STEP)
    only = [float(x) for x in a.lams.split(",")] if a.lams else None
    entries = select_geometries(a)
    if not entries:
        print(f"[run_geometry] task {a.task}: no geometries for this task")
        return
    mpbx_commit = results_io.mpbx_version()
    allowed = available_cpus(a.max_cores, a.cpu_slice)
    cpw = max(1, min(a.cores_per_worker, len(allowed)))
    n_workers = len(allowed) // cpw
    core_sets = [allowed[i * cpw:(i + 1) * cpw] for i in range(n_workers)]
    print(f"[run_geometry] {datetime.datetime.now():%Y-%m-%d %H:%M:%S} host {os.uname().nodename if hasattr(os, 'uname') else ''}"
          f" | {len(allowed)} CPUs -> {n_workers} workers x {cpw} | {len(entries)} geometr{'y' if len(entries) == 1 else 'ies'}"
          f" | fields {a.fields} | parity {C.USE_PARITY} | out {os.path.abspath(a.out)}", flush=True)

    jobs = []                                                 # (plan, geometry dir, work dir, units)
    for e in entries:
        plan = geometry_plan(e, chunks, step, a.fields, only)
        gdir = os.path.join(a.out, e["name"])
        work = os.path.join(gdir, "work")
        units = make_units(plan, n_workers, a.units_per_worker)
        if not any(c["lams"] for c in plan["chunks"]):
            raise SystemExit("no wavelengths to solve (check --lams against the chunk ranges and --step)")
        print_plan(plan, units, n_workers, cpw)
        if a.dry_run:
            continue
        plan["mpbx_commit"] = mpbx_commit
        stored = results_io.stored_setup(gdir, e["name"])    # result files already in this folder?
        if stored is not None and stored != results_io.setup_signature(plan):
            raise SystemExit(f"{gdir} holds results of a different setup (chunks, grids, step, fields, ...); "
                             f"use a new --out folder")
        if results_io.is_complete(gdir, e["name"]) and not a.merge_only:
            print(f"  complete already ({gdir}); skipped")
            continue
        plan_path = os.path.join(work, "plan.json")
        if a.merge_only and not os.path.exists(plan_path):
            print(f"  --merge-only: no per-wavelength files in {work} "
                  f"({'removed after the successful merge' if results_io.is_complete(gdir, e['name']) else 'never solved'}); skipped")
            continue
        os.makedirs(work, exist_ok=True)
        write_plan = True
        if os.path.exists(plan_path):                         # a resubmission must not change the setup
            old = json.load(open(plan_path))
            if results_io.setup_signature(old) != results_io.setup_signature(plan):
                raise SystemExit(f"{plan_path} describes a different setup; use a new --out folder "
                                 f"(or delete {work})")
            if old.get("mpbx_commit") and old.get("mpbx_commit") != mpbx_commit:
                w = (f"mpbx (the solver) changed since this geometry was started: {old.get('mpbx_commit')} -> "
                     f"{mpbx_commit}; the per-λ files may mix the two versions")
                print(f"  WARNING {w}")
                plan["warnings"].append(w)
            plan["warnings"] += [x for x in old.get("warnings", []) if x not in plan["warnings"]]
            if only is not None and not old.get("lams_subset"):
                # a --lams test inside a full run: solve only those λ, but keep (and merge with) the
                # full plan, so the result files still cover every solved λ of the full run
                write_plan = False
                plan = dict(old, warnings=plan["warnings"], mpbx_commit=mpbx_commit)
        if write_plan:
            with open(plan_path, "w") as f:
                json.dump(plan, f, indent=1)
        jobs.append((plan, gdir, work, [] if a.merge_only else units))
    if a.dry_run or not jobs:
        return

    t0 = time.perf_counter()
    failed, notes = [], {}
    left = {plan["name"]: len(units) for plan, _, _, units in jobs}

    def finish(plan, gdir, work):
        try:
            info = results_io.merge_geometry(gdir, plan, notes.get(plan["name"], []),
                                             failed=[f for f in failed if f[0] == plan["name"]])
            print(f"  [{plan['name']}] merged -> {gdir} ({info})", flush=True)
            if info.get("complete") and not a.keep_work:
                shutil.rmtree(work, ignore_errors=True)
        except Exception as e:                                # noqa: BLE001
            failed.append((plan["name"], "merge", repr(e)))
            print(f"  [{plan['name']}] MERGE FAILED {e!r} (the per-λ files are kept in {work})", flush=True)

    todo = [(plan, gdir, work, u) for plan, gdir, work, units in jobs for u in units]
    for plan, gdir, work, units in jobs:                      # nothing to solve: merge right away
        if not units:
            finish(plan, gdir, work)
    if todo:
        ctx = mp.get_context("spawn")                         # never fork a process that holds JAX
        q = ctx.Queue()
        for cs in core_sets:
            q.put(cs)
        ex = cf.ProcessPoolExecutor(n_workers, mp_context=ctx, initializer=_init_worker, initargs=(q,))
        try:
            futs = {ex.submit(run_unit, u, plan, work): (plan, gdir, work, u) for plan, gdir, work, u in todo}
            n_fin = 0
            for fu in cf.as_completed(futs):
                plan, gdir, work, u = futs[fu]
                n_fin += 1
                tag = f"{plan['name']} {u['chunk']} {u['lams'][0]:.3f}->{u['lams'][-1]:.3f} µm"
                try:
                    n, nt, fl = fu.result()
                    notes.setdefault(plan["name"], []).extend(nt)
                    for lam, err in fl:
                        failed.append((plan["name"], f"{u['chunk']} λ={lam:.3f}", err))
                    print(f"  [{n_fin}/{len(todo)}] {tag}: {n} solved"
                          f"{f', {len(fl)} FAILED (see the worker traceback above)' if fl else ''}"
                          f" ({time.perf_counter() - t0:.0f} s)", flush=True)
                except BrokenProcessPool:                     # a worker died (out of memory, ...): stop
                    raise
                except Exception as e:                        # noqa: BLE001  one bad unit must not stop the rest
                    failed.append((plan["name"], tag, repr(e)))
                    print(f"  [{n_fin}/{len(todo)}] {tag}: FAILED {e!r}", flush=True)
                left[plan["name"]] -= 1
                if left[plan["name"]] == 0:
                    finish(plan, gdir, work)
        except BaseException:                                 # Ctrl-C, broken pool, time-limit signal
            ex.shutdown(wait=False, cancel_futures=True)
            raise
        ex.shutdown(wait=True)
    print(f"[run_geometry] finished in {(time.perf_counter() - t0) / 3600:.2f} h", flush=True)
    if failed:
        for f in failed:
            print("  FAILED " + " | ".join(f))
        sys.exit(1)


if __name__ == "__main__":
    main()
