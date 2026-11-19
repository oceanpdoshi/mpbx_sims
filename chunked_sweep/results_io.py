"""Result files of run_geometry.py: writing them (merge) and reading them (notebooks, scripts).

Reading, e.g. in a Jupyter notebook:

    import sys; sys.path.insert(0, "/path/to/mpbx_sims/chunked_sweep")
    import results_io as rio
    run = "outputs/my_sweep_2026-11-20-14-05"
    rio.list_geometries(run)                         # one row per geometry: parameters, complete?, folder
    modes, meta = rio.load_modes(f"{run}/<name>")    # one row per (λ, mode), all modes, no fields
    tracks = rio.load_tracks(f"{run}/<name>")        # TE00 ... TM30 vs λ (one row per target and λ)
    E, info = rio.load_field(f"{run}/<name>", lam=0.80, label="TE00")   # (3, Nx, Ny) complex, x, y, ...
    allt = rio.load_all_tracks(run)                  # tracks of every geometry + its parameters

Files per geometry (folder OUT/<name>/):
    <name>_modes.h5     /lam (n_lam,), /chunk (n_lam,), per-mode arrays (n_lam, n_modes) padded with
                        NaN / -1 / "" where a λ has fewer modes: neff, ng, te_frac, ln_frac, edge_frac,
                        parity, label, guided, converged, newton_iters, lobpcg_iters, flags;
                        /tracks/<target>/{neff, ng, te_frac, edge_frac, mode, flags} (n_lam,);
                        /seams (rows of the chunk-boundary comparison); attrs: meta (JSON), complete.
                        Modes at each λ are sorted by descending n_eff (both parity sectors together).
    <name>_modes.csv    the same table, one row per (λ, mode)
    <name>_tracks.csv   one row per (target, λ)
    <name>_fields_<label>.h5   one group per chunk: /<chunk>/E (n_lam, n_modes, 3, Nx, Ny) complex64,
                        /<chunk>/{lam, x, y, neff, ng, label, parity, te_frac}; the mode order at each
                        λ is the same as in _modes.h5. E = (Ex, Ey, Ez), x horizontal (crystal Z, TE),
                        y vertical (crystal X, TM), normalized to ∫Re(E*×H)_z dA = 1 (mpbx units).
    <name>_summary.txt  grids, flags, chunk seams, timing, missing wavelengths
    <name>_dispersion.png
"""
from __future__ import annotations

import csv
import datetime
import glob
import json
import math
import os
import subprocess
import sys

import numpy as np

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")      # SuperCloud: no file locking on home / groups
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for _p in (HERE, ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import sweep_config as C  # noqa: E402

MODE_FLOAT = ("neff", "ng", "te_frac", "ln_frac", "edge_frac", "edge_db")
MODE_INT = ("parity", "newton_iters", "lobpcg_iters")
MODE_BOOL = ("guided", "converged")
PER_LAM = ("n_cutoff", "n_e", "n_o", "n_sio2", "solve_time")
FORMAT = "tfln_chunked_sweep/1"


# --------------------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------------------
def _paths(gdir, name=None):
    gdir = os.path.abspath(gdir)
    name = name or os.path.basename(gdir)
    return gdir, name, os.path.join(gdir, f"{name}_modes.h5")


def field_files(chunk_names):
    """{file label: [chunk names]} for the chunks of a run (sweep_config.FIELD_FILES)."""
    out, used = {}, set()
    for label, names in C.FIELD_FILES:
        sel = [n for n in names if n in chunk_names]
        if sel:
            out[label] = sel
            used.update(sel)
    for n in chunk_names:
        if n not in used:
            out[n] = [n]
    return out


def _git(*args):
    try:
        return subprocess.run(["git", *args], capture_output=True, text=True, timeout=20).stdout.strip() or None
    except Exception:                                         # noqa: BLE001
        return None


def code_version():
    """Version of this repo's code that runs: the CODE_VERSION file written when the submit scripts
    copied the code into a run folder, else `git describe` of the repo (with -dirty if modified)."""
    p = os.path.join(ROOT, "CODE_VERSION")
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            return f.readline().strip() or None
    top = _git("-C", ROOT, "rev-parse", "--show-toplevel")
    if top and os.path.normcase(os.path.abspath(top)) == os.path.normcase(ROOT):
        return _git("-C", ROOT, "describe", "--always", "--dirty", "--abbrev=10")
    return None


def mpbx_version():
    """`git describe` of the mpbx checkout that provides the installed mpbjax (editable install)."""
    try:
        import mpbjax
        return _git("-C", os.path.dirname(os.path.abspath(mpbjax.__file__)), "describe", "--always", "--dirty",
                    "--abbrev=10")
    except Exception:                                         # noqa: BLE001
        return None


SETUP_KEYS = ("geometry", "step", "fields", "use_parity", "align_frac", "k_tol", "eig_tol", "nev_extra", "field_dtype")


def setup_signature(plan):
    """Everything that must be identical for results to be combined or resumed (not the λ subset)."""
    sig = {k: plan.get(k) for k in SETUP_KEYS}
    sig["chunks"] = [dict(name=c["name"], lam_min=c["lam_min"], lam_max=c["lam_max"], grid=c["grid"],
                          nev=c["nev"], targets=list(c["targets"])) for c in plan["chunks"]]
    return json.loads(json.dumps(sig))                        # normalized (tuples -> lists, inf round-trips)


def stored_setup(gdir, name=None):
    """setup_signature of the results in a geometry folder, or None if there are none."""
    import h5py
    gdir, name, mpath = _paths(gdir, name)
    if not os.path.exists(mpath):
        return None
    try:
        with h5py.File(mpath, "r") as f:
            return json.loads(f.attrs["meta"]).get("setup")
    except Exception:                                         # noqa: BLE001
        return None


def is_complete(gdir, name=None):
    """True if the geometry's result files exist and the merge recorded every wavelength."""
    import h5py
    gdir, name, mpath = _paths(gdir, name)
    if not os.path.exists(mpath):
        return False
    try:
        with h5py.File(mpath, "r") as f:
            if not bool(f.attrs.get("complete", False)):
                return False
            files = json.loads(f.attrs["meta"]).get("field_files", [])
    except Exception:                                         # noqa: BLE001
        return False
    return all(os.path.exists(os.path.join(gdir, p)) for p in files)


# --------------------------------------------------------------------------------------
# Merge (called by run_geometry.py when all of a geometry's work units are done)
# --------------------------------------------------------------------------------------
def _read_lam_file(path):
    import h5py
    with h5py.File(path, "r") as f:
        meta = json.loads(f.attrs["meta"])
        d = {k: f[k][0] for k in MODE_FLOAT + MODE_INT + MODE_BOOL if k in f}
        pol = f["hg_pol"][0].astype(str)
        d["label"] = np.array([f"{p}{int(m)}{int(n)}" for p, m, n in zip(pol, f["hg_m"][0], f["hg_n"][0])])
        per = {k: float(f[k][0]) for k in PER_LAM if k in f}
        lam = float(f["lam"][0])
    return lam, d, per, meta


def _mode_flags(d):
    n = len(d["neff"])
    flags = [[] for _ in range(n)]
    for j in range(n):
        if 0.2 < d["te_frac"][j] < 0.8:
            flags[j].append("hybrid polarization")
        if d["edge_frac"][j] > C.EDGE_FLAG:
            flags[j].append("field at window edge")
        if not d["converged"][j]:
            flags[j].append("NOT CONVERGED")
        same = [k for k in range(n) if k != j and d["parity"][k] == d["parity"][j]]
        if same:
            gap = min(abs(d["neff"][j] - d["neff"][k]) for k in same)
            if gap < C.NEAR_GAP:
                flags[j].append(f"close same-sector neighbour ({gap:.4f}): anticrossing?")
        if d["guided"][j] and all(d["neff"][k] > d["neff"][j] for k in same):
            flags[j].append("last solved mode of its sector: a close neighbour below would not be seen")
    return ["; ".join(f) for f in flags]


def _pick(d, target):
    """Index of the guided mode labelled ``target`` with the purest polarization (or None, count)."""
    te = target.startswith("TE")
    cand = [j for j in range(len(d["neff"])) if d["label"][j] == target and d["guided"][j]]
    if not cand:
        return None, 0
    cand.sort(key=lambda j: abs(d["te_frac"][j] - (1.0 if te else 0.0)))
    return cand[0], len(cand)


def merge_geometry(gdir, plan, notes=(), failed=()):
    """Collect work/<chunk>/lam_*.h5 into the geometry's result files. Returns a short info dict."""
    import h5py
    gdir, name, mpath = _paths(gdir, plan["name"])
    work = os.path.join(gdir, "work")
    chunks = plan["chunks"]
    recs = {}                                                 # (chunk, λ) -> (d, per, meta, path)
    for c in chunks:
        for lam in c["lams"]:
            p = os.path.join(work, c["name"], f"lam_{lam * 1e3:08.2f}nm.h5")
            if os.path.exists(p):
                try:
                    lam_f, d, per, meta = _read_lam_file(p)
                except Exception as e:                        # noqa: BLE001
                    print(f"[merge] unreadable {p}: {e!r}", flush=True)
                    continue
                d["flags"] = np.array(_mode_flags(d), dtype=object)
                recs[(c["name"], round(lam, 6))] = (d, per, meta, p)
    rows = []                                                 # main table: owner chunk per λ
    missing = []
    for c in chunks:
        for lam in c["owned"]:
            r = recs.get((c["name"], round(lam, 6)))
            if r is None:
                missing.append((c["name"], lam))
            else:
                rows.append((lam, c, *r))
    rows.sort(key=lambda r: r[0])
    if not rows:
        raise RuntimeError(f"no solved wavelengths in {work}")
    lams = np.array([r[0] for r in rows])
    n_lam = len(rows)
    n_max = max(len(r[2]["neff"]) for r in rows)
    T = {}
    for k in MODE_FLOAT:
        T[k] = np.full((n_lam, n_max), np.nan)
    for k in MODE_INT:
        T[k] = np.full((n_lam, n_max), -1 if k != "parity" else 0, dtype=int)
    for k in MODE_BOOL:
        T[k] = np.zeros((n_lam, n_max), bool)
    T["label"] = np.full((n_lam, n_max), "", dtype=object)
    T["flags"] = np.full((n_lam, n_max), "", dtype=object)
    per_lam = {k: np.full(n_lam, np.nan) for k in PER_LAM}
    n_modes = np.zeros(n_lam, int)
    for i, (lam, c, d, per, meta, p) in enumerate(rows):
        n = len(d["neff"])
        n_modes[i] = n
        for k in T:
            if k in d:
                T[k][i, :n] = d[k]
        for k in PER_LAM:
            per_lam[k][i] = per.get(k, np.nan)
    chunk_of = np.array([r[1]["name"] for r in rows], dtype=object)

    # ---- tracks ---------------------------------------------------------------------
    tracks = {}
    for t in C.ALL_TARGETS:
        tr = dict(mode=np.full(n_lam, -1), neff=np.full(n_lam, np.nan), ng=np.full(n_lam, np.nan),
                  te_frac=np.full(n_lam, np.nan), edge_frac=np.full(n_lam, np.nan),
                  flags=np.full(n_lam, "", dtype=object))
        for i, (lam, c, d, *_r) in enumerate(rows):
            j, nc = _pick(d, t)
            if j is None:
                continue
            tr["mode"][i] = j
            for k in ("neff", "ng", "te_frac", "edge_frac"):
                tr[k][i] = d[k][j]
            fl = [x for x in (d["flags"][j], f"{nc} guided modes labelled {t}" if nc > 1 else "") if x]
            tr["flags"][i] = "; ".join(fl)
        tracks[t] = tr

    # ---- chunk seams: the same mode at a λ solved in two chunks ------------------------
    seams = []
    for i_c in range(len(chunks) - 1):
        a, b = chunks[i_c], chunks[i_c + 1]
        for lam in set(map(lambda x: round(x, 6), a["lams"])) & set(map(lambda x: round(x, 6), b["lams"])):
            ra, rb = recs.get((a["name"], lam)), recs.get((b["name"], lam))
            if not (ra and rb):
                continue
            for t in C.ALL_TARGETS:
                ja, _ = _pick(ra[0], t)
                jb, _ = _pick(rb[0], t)
                if ja is None or jb is None:
                    continue
                seams.append(dict(lam=lam, target=t, chunk_a=a["name"], chunk_b=b["name"],
                                  d_neff=float(ra[0]["neff"][ja] - rb[0]["neff"][jb]),
                                  d_ng=float(ra[0]["ng"][ja] - rb[0]["ng"][jb])))

    # ---- escalations / missing targets recorded by the workers -------------------------
    esc, miss_t = [], []
    for (cn, lam), (d, per, meta, p) in sorted(recs.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        if meta.get("nev_escalated"):
            esc.append(f"{cn} λ={lam:.3f}: nev -> {meta.get('nev_per_sector')} per sector")
        if meta.get("missing_targets"):
            miss_t.append(f"{cn} λ={lam:.3f}: {', '.join(meta['missing_targets'])}")
    complete = not missing and not failed and not plan.get("lams_subset")
    any_meta = rows[0][4]
    warnings = list(plan.get("warnings", []))
    versions = sorted({str(r[2].get("mpbx_commit")) for r in recs.values() if r[2].get("mpbx_commit")})
    if len(versions) > 1:
        warnings.append(f"the per-λ files were solved with different mpbx versions: {', '.join(versions)}")

    # ---- fields --------------------------------------------------------------------------
    ff = field_files([c["name"] for c in chunks]) if plan["fields"] == "all" else {}
    written = []
    for label, cnames in ff.items():
        fpath = os.path.join(gdir, f"{name}_fields_{label}.h5")
        tmp = fpath + ".part"
        with h5py.File(tmp, "w") as f:
            f.attrs["format"] = FORMAT + "/fields"
            f.attrs["layout"] = ("one group per chunk: E (n_lam, n_modes, 3, Nx, Ny) complex = (Ex, Ey, Ez); "
                                 "modes sorted by descending neff as in _modes.h5 (zero-padded); x horizontal "
                                 "(crystal Z), y vertical (crystal X), pixel centres in µm; normalized to "
                                 "int Re(E* x H)_z dA = 1 in mpbx units (x27.45 for 1 W SI fields in V/µm)")
            f.attrs["geometry"] = json.dumps(plan["geometry"])
            for cn in cnames:
                sel = [(i, r) for i, r in enumerate(rows) if r[1]["name"] == cn]
                if not sel:
                    continue
                c = sel[0][1][1]                                  # the chunk dict of these rows
                g = c["grid"]
                nm = max(len(r[2]["neff"]) for _, r in sel)
                grp = f.create_group(cn)
                E = grp.create_dataset("E", shape=(len(sel), nm, 3, g["Nx"], g["Ny"]), dtype=plan["field_dtype"],
                                       chunks=(1, 1, 3, g["Nx"], g["Ny"]), **C.FIELD_COMPRESSION)
                xs = ys = None
                for k, (i, r) in enumerate(sel):
                    with h5py.File(r[5], "r") as src:
                        e = src["E"][0]
                        E[k, :e.shape[0]] = e
                        if xs is None:
                            xs, ys = src["x"][()], src["y"][()]
                grp["lam"] = lams[[i for i, _ in sel]]
                grp["x"], grp["y"] = xs, ys
                for key in ("neff", "ng", "te_frac", "parity"):
                    grp[key] = T[key][[i for i, _ in sel], :nm]
                grp["label"] = T["label"][[i for i, _ in sel], :nm].astype("S")
                grp.attrs["grid"] = json.dumps(g)
                grp.attrs["chunk"] = json.dumps({k: v for k, v in c.items() if k not in ("lams", "owned")})
        os.replace(tmp, fpath)
        written.append(os.path.basename(fpath))

    # ---- modes file -----------------------------------------------------------------------
    meta = dict(format=FORMAT, name=name, geometry=plan["geometry"],
                chunks=[{k: v for k, v in c.items() if k not in ("lams", "owned")} for c in chunks],
                step=plan["step"], use_parity=plan["use_parity"], align_frac=plan["align_frac"],
                k_tol=plan["k_tol"], eig_tol=plan["eig_tol"], fields=plan["fields"], field_files=written,
                setup=setup_signature(plan), lams_subset=plan.get("lams_subset"),
                material=any_meta.get("material"), x64=any_meta.get("x64"), backend=any_meta.get("backend"),
                code_version=code_version(), mpbx_version=plan.get("mpbx_commit") or mpbx_version(),
                merged=datetime.datetime.now().isoformat(timespec="seconds"),
                complete=complete, missing_lams=[[cn, l] for cn, l in missing],
                failed=[list(f) for f in failed], nev_escalations=esc, missing_targets=miss_t,
                warnings=warnings)
    tmp = mpath + ".part"
    with h5py.File(tmp, "w") as f:
        f.attrs["format"] = FORMAT
        f.attrs["layout"] = ("/lam (n_lam,), /chunk (n_lam,), /n_modes (n_lam,); per-mode arrays (n_lam, n_modes) "
                             "padded with NaN / -1 / '' where a wavelength has fewer modes; modes at each λ "
                             "sorted by descending neff; /tracks/<target>/* (n_lam,), mode = -1 where not "
                             "found; /seams = the same mode at a λ solved in two chunks; meta = JSON")
        f.attrs["meta"] = json.dumps(meta)
        f.attrs["complete"] = complete
        f["lam"] = lams
        f["chunk"] = chunk_of.astype("S")
        f["n_modes"] = n_modes
        for k, v in T.items():
            f[k] = v.astype("S") if v.dtype == object else v
        for k, v in per_lam.items():
            f[k] = v
        tg = f.create_group("tracks")
        for t, tr in tracks.items():
            g_ = tg.create_group(t)
            for k, v in tr.items():
                g_[k] = v.astype("S") if v.dtype == object else v
        if seams:
            sg = f.create_group("seams")
            for k in seams[0]:
                v = np.array([s[k] for s in seams])
                sg[k] = v.astype("S") if v.dtype.kind in "OU" else v
    os.replace(tmp, mpath)

    # ---- CSV tables ---------------------------------------------------------------------------
    with open(os.path.join(gdir, f"{name}_modes.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["lam_um", "chunk", "mode", "label", "parity", "neff", "ng", "te_frac", "ln_frac", "edge_frac",
                    "guided", "converged", "flags"])
        for i in range(n_lam):
            for j in range(n_modes[i]):
                w.writerow([f"{lams[i]:.4f}", chunk_of[i], j, T["label"][i, j], T["parity"][i, j],
                            f"{T['neff'][i, j]:.8f}", f"{T['ng'][i, j]:.8f}", f"{T['te_frac'][i, j]:.4f}",
                            f"{T['ln_frac'][i, j]:.4f}", f"{T['edge_frac'][i, j]:.3e}", int(T["guided"][i, j]),
                            int(T["converged"][i, j]), T["flags"][i, j]])
    with open(os.path.join(gdir, f"{name}_tracks.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["target", "lam_um", "chunk", "mode", "neff", "ng", "te_frac", "edge_frac", "flags"])
        for t, tr in tracks.items():
            for i in range(n_lam):
                if tr["mode"][i] >= 0:
                    w.writerow([t, f"{lams[i]:.4f}", chunk_of[i], tr["mode"][i], f"{tr['neff'][i]:.8f}",
                                f"{tr['ng'][i]:.8f}", f"{tr['te_frac'][i]:.4f}", f"{tr['edge_frac'][i]:.3e}",
                                tr["flags"][i]])

    # ---- summary + plot -------------------------------------------------------------------------
    _write_summary(gdir, name, plan, meta, rows, lams, tracks, seams, recs, notes)
    try:
        _plot(gdir, name, plan, lams, tracks)
    except Exception as e:                                    # noqa: BLE001  a plot must never fail a merge
        print(f"[merge] plot failed: {e!r}", flush=True)
    return dict(complete=complete, n_lam=n_lam, missing=len(missing), field_files=len(written),
                escalations=len(esc))


def _write_summary(gdir, name, plan, meta, rows, lams, tracks, seams, recs, notes):
    from geometry_gen.tfln_rib import RibGeometry
    L = [f"{name}", f"  {RibGeometry(**plan['geometry']).describe()}",
         f"  merged {meta['merged']}  code {meta['code_version']}  mpbx {meta['mpbx_version']}  "
         f"backend {meta['backend']}  x64 {meta['x64']}",
         f"  COMPLETE: {meta['complete']}"
         + ("" if meta["complete"] else f"  (missing λ: {meta['missing_lams'][:20]}{' ...' if len(meta['missing_lams']) > 20 else ''};"
                                        f" failed: {meta['failed']}"
                                        f"{'; a --lams subset run is never marked complete' if meta['lams_subset'] else ''})"),
         "", "chunks (window, pixels, modes per parity sector, solve time per λ, peak memory per worker):"]
    for c in plan["chunks"]:
        g = c["grid"]
        ts = [recs[(c["name"], round(l, 6))][1].get("solve_time", np.nan) for l in c["lams"]
              if (c["name"], round(l, 6)) in recs]
        mem = [recs[(c["name"], round(l, 6))][2].get("peak_rss_gb", np.nan) for l in c["lams"]
               if (c["name"], round(l, 6)) in recs]
        L.append(f"  {c['name']:15s} {c['lam_min']:.3f}-{c['lam_max']:.3f} µm  {g['Lx']:.3f} x {g['Ly']:.3f} µm  "
                 f"{g['Nx']} x {g['Ny']} px (dx {1e3 * g['dx']:.1f}, dy {1e3 * g['dy']:.1f} nm)  nev {c['nev']}  "
                 f"{len(ts)}/{len(c['lams'])} λ solved, median {np.nanmedian(ts) if ts else np.nan:.0f} s/λ, "
                 f"peak {np.nanmax(mem) if mem else np.nan:.2f} GB")
    for w in meta["warnings"]:
        L.append(f"  WARNING {w}")
    L += ["", "targets (λ found / λ total; flagged λ):"]
    for t, tr in tracks.items():
        got = tr["mode"] >= 0
        nfl = int(sum(1 for i in range(len(lams)) if got[i] and tr["flags"][i]))
        L.append(f"  {t}: {int(got.sum())}/{len(lams)}" + (f"; {nfl} flagged" if nfl else ""))
    kinds = {}
    for (lam, c, d, *_r) in rows:
        for fl in d["flags"]:
            for part in filter(None, fl.split("; ")):
                kinds[part.split(" (")[0]] = kinds.get(part.split(" (")[0], 0) + 1
    if kinds:
        L += ["", "mode flags (all modes, all λ):"] + [f"  {k}: {v}" for k, v in sorted(kinds.items())]
    if meta["nev_escalations"]:
        L += ["", "nev raised (a target was missing while the sector's deepest solved mode was guided):"]
        L += [f"  {x}" for x in meta["nev_escalations"]]
    if meta["missing_targets"]:
        L += ["", "targets not found among the guided modes (cut off, or not labelled):"]
        L += [f"  {x}" for x in meta["missing_targets"]]
    if seams:
        L += ["", "chunk seams: the same mode solved on both grids (lower chunk minus upper chunk):"]
        L += [f"  λ = {s['lam']:.3f} µm  {s['target']}  {s['chunk_a']} - {s['chunk_b']}:  "
              f"Δn_eff {s['d_neff']:+.2e}   Δn_g {s['d_ng']:+.2e}" for s in seams]
    if notes:
        L += ["", "notes from the workers:"] + [f"  {n}" for n in notes]
    L += ["", "files:"]
    for p in sorted(glob.glob(os.path.join(gdir, f"{name}_*"))):
        if not p.endswith(".part"):
            L.append(f"  {os.path.basename(p)}  {os.path.getsize(p) / 1e6:.1f} MB")
    with open(os.path.join(gdir, f"{name}_summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")


def _plot(gdir, name, plan, lams, tracks):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sim_core.tfln_plots import SERIES
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True, sharex=True)
    for t, tr in tracks.items():
        if not np.isfinite(tr["neff"]).any():
            continue
        m = int(t[2])
        style = dict(color=SERIES[m % len(SERIES)], ls="-" if t.startswith("TE") else "--", lw=1.5, label=t)
        ax[0].plot(lams, tr["neff"], **style)                 # NaN (not found) breaks the line
        ax[1].plot(lams, tr["ng"], **style)
    for a_ in ax:
        for c in plan["chunks"][1:]:
            a_.axvline(c["lam_min"], color="#8a8986", lw=0.8, ls=":")
        a_.grid(alpha=0.25)
        a_.set_xlabel("wavelength λ [µm]")
    ax[0].set_ylabel(r"$n_\mathrm{eff}$")
    ax[1].set_ylabel(r"$n_g$")
    ax[0].legend(ncol=2, frameon=False, fontsize=8)
    fig.suptitle(f"{name} (chunk boundaries dotted)", fontsize=10)
    fig.savefig(os.path.join(gdir, f"{name}_dispersion.png"), dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------------------
# Reading
# --------------------------------------------------------------------------------------
def _decode(v):
    return v.astype(str) if v.dtype.kind == "S" else v


def load_modes(gdir, name=None):
    """(DataFrame with one row per (λ, mode), meta dict) from <name>_modes.h5 (no fields)."""
    import h5py
    import pandas as pd
    gdir, name, mpath = _paths(gdir, name)
    with h5py.File(mpath, "r") as f:
        meta = json.loads(f.attrs["meta"])
        lam, chunk, nm = f["lam"][()], _decode(f["chunk"][()]), f["n_modes"][()]
        cols = {k: _decode(f[k][()]) for k in MODE_FLOAT + MODE_INT + MODE_BOOL + ("label", "flags") if k in f}
        per = {k: f[k][()] for k in PER_LAM if k in f}
    recs = []
    for i in range(len(lam)):
        for j in range(nm[i]):
            r = dict(lam=lam[i], chunk=chunk[i], mode=j)
            r.update({k: v[i, j] for k, v in cols.items()})
            r["n_cutoff"] = per["n_cutoff"][i]
            recs.append(r)
    return pd.DataFrame(recs), meta


def load_tracks(gdir, name=None, targets=None):
    """DataFrame with one row per (target, λ) where the target was found."""
    import h5py
    import pandas as pd
    gdir, name, mpath = _paths(gdir, name)
    recs = []
    with h5py.File(mpath, "r") as f:
        lam, chunk = f["lam"][()], _decode(f["chunk"][()])
        for t in (targets or list(f["tracks"])):
            g = f["tracks"][t]
            d = {k: _decode(g[k][()]) for k in g}
            for i in np.flatnonzero(d["mode"] >= 0):
                recs.append(dict(target=t, lam=lam[i], chunk=chunk[i], **{k: d[k][i] for k in d}))
    return pd.DataFrame(recs)


def load_field(gdir, lam, label=None, mode=None, name=None):
    """E field (3, Nx, Ny) of one mode at the stored λ nearest to ``lam`` (by tracked ``label`` such
    as "TE00", or by ``mode`` index in the _modes table). Returns (E, info dict with x, y, lam,
    neff, ng, label, chunk, file)."""
    import h5py
    gdir, name, mpath = _paths(gdir, name)
    with h5py.File(mpath, "r") as f:
        lams = f["lam"][()]
        i = int(np.argmin(np.abs(lams - lam)))
        chunk = _decode(f["chunk"][()])[i]
        if mode is None:
            if label is None:
                raise ValueError("give label= or mode=")
            mode = int(f["tracks"][label]["mode"][i])
            if mode < 0:
                raise KeyError(f"{label} not found at λ = {lams[i]:.4f} µm")
        files = json.loads(f.attrs["meta"]).get("field_files", [])
    for fn in files:
        with h5py.File(os.path.join(gdir, fn), "r") as ff:
            if chunk not in ff:
                continue
            g = ff[chunk]
            k = int(np.argmin(np.abs(g["lam"][()] - lams[i])))
            E = g["E"][k, mode]
            info = dict(x=g["x"][()], y=g["y"][()], lam=float(g["lam"][k]), neff=float(g["neff"][k, mode]),
                        ng=float(g["ng"][k, mode]), label=g["label"][k, mode].decode(), chunk=chunk, file=fn,
                        mode=mode)
            return E, info
    raise FileNotFoundError(f"no field file holds chunk {chunk} (run with --fields all?)")


def list_geometries(run_dir):
    """DataFrame: one row per geometry folder of a run (parameters, complete, folder)."""
    import h5py
    import pandas as pd
    recs = []
    for mpath in sorted(glob.glob(os.path.join(run_dir, "*", "*_modes.h5"))):
        gdir = os.path.dirname(mpath)
        with h5py.File(mpath, "r") as f:
            meta = json.loads(f.attrs["meta"])
        recs.append(dict(name=meta["name"], **meta["geometry"], complete=meta["complete"], folder=gdir))
    return pd.DataFrame(recs)


def load_all_tracks(run_dir, targets=None):
    """Tracks of every geometry in a run, with the geometry parameters as columns."""
    import pandas as pd
    parts = []
    for _, g in list_geometries(run_dir).iterrows():
        t = load_tracks(g["folder"], targets=targets)
        for k in ("name", "t_ln", "etch_depth", "w_top", "sidewall_deg", "t_clad", "complete"):
            t[k] = g[k]
        parts.append(t)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


if __name__ == "__main__":
    # python results_io.py OUT/<name>    re-merge a geometry from its work/ folder (same as
    #                                    run_geometry.py --merge-only, without the solver checks)
    gd = sys.argv[1]
    pl = json.load(open(os.path.join(gd, "work", "plan.json")))
    print(merge_geometry(gd, pl))
