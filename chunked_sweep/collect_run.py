"""Run-level tables over all geometries of a run folder (made by run_geometry.py):

    python collect_run.py OUT          -> OUT/run_summary.csv   one row per geometry in geometries.csv
                                          OUT/all_tracks.csv    TE00 ... TM30 vs λ of every geometry

Geometries without result files are listed with complete = False, so this also tells you what is
left to do (resubmit with --resume to finish them).
"""
import csv
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import results_io as rio  # noqa: E402
from run_geometry import GEOM_KEYS, UNDERCUT_KEY, read_geometry_csv  # noqa: E402


def main():
    run = os.path.abspath(sys.argv[1])
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:                                     # noqa: BLE001
            pass
    import h5py
    rows, parts = [], []
    entries = read_geometry_csv(os.path.join(run, "geometries.csv"))
    # geometry columns: t_undercut only in runs with undercut ribs (then 0 = a regular rib)
    gkeys = list(GEOM_KEYS) + ([UNDERCUT_KEY] if any(UNDERCUT_KEY in e["geometry"] for e in entries) else [])
    for e in entries:
        gdir = os.path.join(run, e["name"])
        mpath = os.path.join(gdir, f"{e['name']}_modes.h5")
        geo = {k: e["geometry"].get(k, 0.0) for k in gkeys}
        r = dict(name=e["name"], **geo, complete=False, n_lam=0, missing_lams=None,
                 nev_escalations=None, flagged_target_lams=None, solve_hours=None, files_GB=None, folder=gdir)
        if os.path.exists(mpath):
            with h5py.File(mpath, "r") as f:
                meta = json.loads(f.attrs["meta"])
                st = f["solve_time"][()]
                r.update(complete=bool(f.attrs["complete"]), n_lam=len(f["lam"]),
                         missing_lams=len(meta.get("missing_lams", [])),
                         nev_escalations=len(meta.get("nev_escalations", [])),
                         solve_hours=round(float(st[st == st].sum()) / 3600, 2))
            t = rio.load_tracks(gdir)
            r["flagged_target_lams"] = int((t["flags"].astype(str).str.len() > 0).sum()) if len(t) else 0
            r["files_GB"] = round(sum(os.path.getsize(os.path.join(gdir, p)) for p in os.listdir(gdir)
                                      if p.startswith(e["name"])) / 1e9, 3)
            for k in gkeys:
                t[k] = geo[k]
            t.insert(0, "name", e["name"])
            parts.append(t)
        rows.append(r)
    with open(os.path.join(run, "run_summary.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    if parts:
        import pandas as pd
        pd.concat(parts, ignore_index=True).to_csv(os.path.join(run, "all_tracks.csv"), index=False)
    n_ok = sum(r["complete"] for r in rows)
    print(f"{n_ok}/{len(rows)} geometries complete -> {run}/run_summary.csv"
          + (", all_tracks.csv" if parts else ""))
    for r in rows:
        if not r["complete"]:
            print(f"  not complete: {r['name']}")


if __name__ == "__main__":
    main()
