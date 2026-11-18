"""Combine a run_chunked.py output folder into one dispersion table with Richardson-
extrapolated n_eff / n_g, error estimates, chunk-boundary checks and per-mode tracks.

    python combine_chunks.py results/sweep_A [--targets TE00 TM00 TE10 ...] [--tol 1e-4]

Writes into the same folder:
    modes_all.csv      one row per (λ, chunk, sector, rank): fine / coarse / extrapolated values
    tracks.csv         per target label and λ: n_eff, n_g (extrapolated) + error estimates
    tracks.h5          the same as arrays (target, λ)
    dispersion.png     n_eff and n_g of the targets, and the Richardson correction vs the tolerance
    boundary_check.txt the same mode at a wavelength shared by two chunks (different grids)

Matching fine <-> coarse: within one x-parity sector, modes are paired by rank (descending
n_eff) and the pair is accepted only if the node labels agree.
Anticrossing flags (the extrapolation of one branch is unreliable there):
  * "close same-sector neighbour": a rank neighbour within NEAR_GAP in n_eff;
  * "n_g correction >> n_eff correction": |ng_corr|/|neff_corr| above 2.5x the median of the
    guided modes at the same λ and grid, AND a same-sector neighbour within WIDE_GAP. (The ratio
    is 0.6-1.4 for normal modes at 0.4-1.0 µm and 4.4 for TE30 next to TM01 at 0.4 µm; it grows to
    ~4 for every mode at 2 µm, hence the comparison with the median.) *_err is then raised to
    |ng_corr|.
Rows without a valid partner keep neff_R = ng_R = nan; the *_best columns hold the
extrapolated value if there is one, else the fine value (flag "not extrapolated").

Richardson: q = q_f + (q_f - q_c)/3 for both refine modes. With refine="xy" the coarse level
is (2dx, 2dy), so the extrapolation removes the lateral and the vertical h² error together;
with refine="y" it is (dx, 2dy) and only the vertical error is removed.

Error estimate of the *_best value (``*_err``, an ESTIMATE, see README section 2):
    0.3·|Richardson correction|   (residual of the extrapolation; 0.02-0.25x in the studies,
                                   anticrossings excluded)
  + lateral model 3.5e-6 · (m+1)² · λ[µm]^0.85 · (dx / 12.5 nm)², refine="y" chunks only
    (fitted to the 0.4 and 1.0 µm dx series of the reference geometry)
  + window model 12 · edge_frac  (n_g error / edge_frac: 0.2-3.5 for the vertical window of all
    modes, up to 12 for the lateral window of TE10 at 2 µm, 24-54 for TM10 near cutoff, which
    this model therefore underestimates)
Rows whose estimate exceeds --tol get the flag "error estimate > tol".
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [ROOT, os.path.join(ROOT, "handoff_tfln_modesolve")]

TARGETS = ("TE00", "TM00", "TE10", "TM10", "TE20", "TM20", "TE30", "TM30")
LAT_C, LAT_LAM_EXP = 3.5e-6, 0.85          # lateral-error model (README section 2.4)
WIN_FACTOR = 12.0                           # window-error model: WIN_FACTOR * edge_frac
NEAR_GAP = 0.005                            # same-sector n_eff gap below which an anticrossing is flagged
WIDE_GAP = 0.03                             # ... and below which the n_g/n_eff correction-ratio test applies
RICH_RESID = 0.3                            # residual of the extrapolation, x |correction|


def load_level(d):
    """{λ: dict(sectors={sector: [mode dicts, descending neff]}, meta)} from per-λ .h5 files."""
    import h5py
    out = {}
    if not os.path.isdir(d):
        return out
    for f in sorted(os.listdir(d)):
        if not f.endswith(".h5"):
            continue
        with h5py.File(os.path.join(d, f), "r") as h:
            lam = float(h["lam"][0])
            g = {k: h[k][0] for k in ("neff", "ng", "te_frac", "parity", "hg_m", "hg_n", "guided",
                                      "converged", "edge_frac")}
            pol = h["hg_pol"][0].astype(str)
            meta = json.loads(h.attrs["meta"])
        sec = {}
        for j in range(len(g["neff"])):
            m = dict(neff=float(g["neff"][j]), ng=float(g["ng"][j]), te_frac=float(g["te_frac"][j]),
                     label=f"{pol[j]}{int(g['hg_m'][j])}{int(g['hg_n'][j])}", m=int(g["hg_m"][j]),
                     guided=bool(g["guided"][j]), converged=bool(g["converged"][j]),
                     edge_frac=float(g["edge_frac"][j]))
            sec.setdefault(int(g["parity"][j]), []).append(m)
        for s_ in sec.values():
            s_.sort(key=lambda m: -m["neff"])
        out[round(lam, 6)] = dict(sectors=sec, meta=meta)
    return out


def _same_setup(a, b):
    """Fine and coarse files must come from the same chunk and geometry (y_bot differs by the
    alignment, the grid by design)."""
    ga = {k: v for k, v in a["geometry"].items() if k != "y_bot"}
    gb = {k: v for k, v in b["geometry"].items() if k != "y_bot"}
    return a.get("chunk") == b.get("chunk") and ga == gb


def lateral_err(m, lam, dx):
    return LAT_C * (m + 1) ** 2 * lam ** LAT_LAM_EXP * (dx / 0.0125) ** 2


def combine(run_dir, tol=1e-4):
    plan = json.load(open(os.path.join(run_dir, "plan.json")))
    rows = []
    for c in plan["chunks"]:
        fine = load_level(os.path.join(run_dir, c["name"], "fine"))
        coarse = load_level(os.path.join(run_dir, c["name"], "coarse"))
        for lam in sorted(fine):
            cl = coarse.get(lam)
            if cl is not None and not _same_setup(fine[lam]["meta"], cl["meta"]):
                raise SystemExit(f"{c['name']} λ={lam}: fine and coarse files come from different setups")
            for sec, modes in sorted(fine[lam]["sectors"].items()):
                cmodes = cl["sectors"].get(sec, []) if cl else []
                for r, mf in enumerate(modes):
                    row = dict(lam=lam, chunk=c["name"], sector=sec, rank=r, label=mf["label"],
                               target=mf["label"] in c["targets"], te_frac=mf["te_frac"], guided=mf["guided"],
                               edge_frac=mf["edge_frac"], neff_fine=mf["neff"], ng_fine=mf["ng"],
                               neff_coarse=np.nan, ng_coarse=np.nan, converged=mf["converged"])
                    mc = cmodes[r] if r < len(cmodes) else None
                    flags = []
                    if mc is None:
                        flags.append("no coarse partner")
                    elif mc["label"] != mf["label"]:
                        flags.append(f"label mismatch (coarse {mc['label']})")
                        mc = None
                    lat = lateral_err(max(mf["m"], 0), lam, c["dx"]) if c.get("refine", "y") == "y" else 0.0
                    win = WIN_FACTOR * mf["edge_frac"]
                    for q in ("neff", "ng"):
                        if mc is not None:
                            corr = (mf[q] - mc[q]) / 3.0
                            row[f"{q}_coarse"] = mc[q]
                            row[f"{q}_R"] = mf[q] + corr
                            row[f"{q}_corr"] = corr
                            row[f"{q}_best"] = row[f"{q}_R"]
                            row[f"{q}_err"] = RICH_RESID * abs(corr) + lat + win
                        else:
                            row[f"{q}_R"] = row[f"{q}_corr"] = np.nan
                            row[f"{q}_best"] = mf[q]
                            row[f"{q}_err"] = np.nan              # unknown: the dy error (~1e-4) is not removed
                    gaps = [abs(mf["neff"] - modes[k]["neff"]) for k in (r - 1, r + 1) if 0 <= k < len(modes)]
                    row["_gap"] = min(gaps) if gaps else np.inf
                    if row["_gap"] < NEAR_GAP:
                        flags.append(f"close same-sector neighbour (gap {row['_gap']:.4f}; anticrossing?)")
                    if mc is not None:
                        row["converged"] = mf["converged"] and mc["converged"]
                    else:
                        flags.append("not extrapolated")
                    row["err_lateral_model"] = lat
                    row["err_window_model"] = win
                    if 0.2 < mf["te_frac"] < 0.8:
                        flags.append("hybrid polarization")
                    if mf["edge_frac"] > 1e-5:
                        flags.append("field at window edge")
                    if not row["converged"]:
                        flags.append("NOT CONVERGED")
                    row["_flags"] = flags
                    rows.append(row)
    # Anticrossing test 2: a mode whose |ng_corr|/|neff_corr| is far above that of the other
    # guided modes at the same λ and grid (the ratio itself grows from ~1 at 0.4-1 µm to ~4 at 2 µm).
    groups = {}
    for row in rows:
        groups.setdefault((row["lam"], row["chunk"]), []).append(row)
    for grp in groups.values():
        ratios = [abs(r["ng_corr"]) / abs(r["neff_corr"]) for r in grp
                  if r["guided"] and np.isfinite(r["neff_corr"]) and abs(r["neff_corr"]) > 0]
        if len(ratios) < 3:
            continue
        med = float(np.median(ratios))
        for r in grp:
            if (r["_gap"] < WIDE_GAP and np.isfinite(r["neff_corr"]) and abs(r["neff_corr"]) > 0
                    and abs(r["ng_corr"]) / abs(r["neff_corr"]) > 2.5 * med):
                r["_flags"].append("n_g correction >> n_eff correction (anticrossing?)")
                r["ng_err"] = max(r["ng_err"], abs(r["ng_corr"]))
    for row in rows:
        flags = row.pop("_flags")
        row.pop("_gap")
        if max(row["neff_err"], row["ng_err"]) > tol:                  # nan compares False
            flags.append("error estimate > tol")
        row["flags"] = "; ".join(flags)
    return plan, rows


def tracks(rows, targets):
    """Per target and λ: among guided rows with that label in a chunk that lists it as a target,
    the one with the purest polarization; at a λ shared by two chunks the first chunk wins."""
    lams = sorted({r["lam"] for r in rows})
    by_lam = {}
    for r in rows:
        by_lam.setdefault(r["lam"], []).append(r)
    tr = {}
    for t in targets:
        recs = []
        for lam in lams:
            cand = [r for r in by_lam[lam] if r["label"] == t and r["guided"] and r["target"]]
            if not cand:
                recs.append(None)
                continue
            chunk0 = sorted({r["chunk"] for r in cand})[0]
            cand = [r for r in cand if r["chunk"] == chunk0]
            cand.sort(key=lambda r: abs(r["te_frac"] - (1.0 if t.startswith("TE") else 0.0)))
            rec = dict(cand[0], n_candidates=len(cand))
            if len(cand) > 1:
                rec["flags"] = "; ".join(x for x in (rec["flags"], f"{len(cand)} modes labelled {t}") if x)
            recs.append(rec)
        tr[t] = recs
    return lams, tr


def boundary_check(rows, targets):
    out = []
    by = {}
    for r in rows:
        if r["guided"]:
            by.setdefault((r["lam"], r["label"]), []).append(r)
    for (lam, lab), rs in sorted(by.items()):
        chunks = sorted({r["chunk"] for r in rs})
        if lab in targets and len(chunks) == 2:
            a = [r for r in rs if r["chunk"] == chunks[0]][0]
            b = [r for r in rs if r["chunk"] == chunks[1]][0]
            out.append(f"λ = {lam:.3f} µm  {lab}:  {chunks[0]} vs {chunks[1]}:  "
                       f"Δn_eff fine {a['neff_fine'] - b['neff_fine']:+.2e}, best {a['neff_best'] - b['neff_best']:+.2e};  "
                       f"Δn_g fine {a['ng_fine'] - b['ng_fine']:+.2e}, best {a['ng_best'] - b['ng_best']:+.2e}")
    return out


def plot(run_dir, plan, lams, tr, tol):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from tfln_plots import SERIES
    fig, ax = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True, sharex=True)
    for t, recs in tr.items():
        m = int(t[2])
        style = dict(color=SERIES[m % len(SERIES)], ls="-" if t.startswith("TE") else "--",
                     marker="o" if t.startswith("TE") else "^", ms=3, lw=1.5, label=t)
        x = np.array([lam for lam, r in zip(lams, recs) if r])
        if not len(x):
            continue
        gap = np.flatnonzero(np.diff(x) > 1.5 * plan["step"]) + 1      # no line across missing λ

        def g(k):
            return np.insert(np.array([r[k] for r in recs if r], float), gap, np.nan)

        xg = np.insert(x, gap, np.nan)
        ax[0, 0].plot(xg, g("neff_best"), **style)
        ax[0, 1].plot(xg, g("ng_best"), **style)
        ax[1, 0].semilogy(xg, np.maximum(g("neff_err"), 1e-9), **style)
        ax[1, 1].semilogy(xg, np.maximum(g("ng_err"), 1e-9), **style)
    for a_ in ax.flat:
        for c in plan["chunks"][1:]:
            a_.axvline(c["lam_min"], color="#8a8986", lw=0.8, ls=":")
        a_.grid(alpha=0.25)
    for a_ in ax[1]:
        a_.axhline(tol, color="#52514e", lw=1)
        a_.set_xlabel("wavelength λ [µm]")
    ax[0, 0].set_ylabel(r"$n_\mathrm{eff}$ (Richardson)")
    ax[0, 1].set_ylabel(r"$n_g$ (Richardson)")
    ax[1, 0].set_ylabel(r"error estimate of $n_\mathrm{eff}$")
    ax[1, 1].set_ylabel(r"error estimate of $n_g$")
    ax[0, 0].legend(ncol=2, frameon=False, fontsize=8)
    fig.suptitle(f"{os.path.basename(os.path.abspath(run_dir))}: chunk boundaries dotted; "
                 f"horizontal line = tolerance {tol:g}", fontsize=10)
    fig.savefig(os.path.join(run_dir, "dispersion.png"), dpi=140)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--targets", nargs="*", default=list(TARGETS))
    ap.add_argument("--tol", type=float, default=1e-4)
    a = ap.parse_args()
    for stream in (sys.stdout, sys.stderr):                   # λ on a Windows console
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:                                     # noqa: BLE001
            pass
    plan, rows = combine(a.run_dir, a.tol)
    if not rows:
        raise SystemExit("no results found")
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(os.path.join(a.run_dir, "modes_all.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    lams, tr = tracks(rows, a.targets)
    tkeys = ["target", "lam", "chunk", "sector", "rank", "neff_best", "neff_err", "ng_best", "ng_err",
             "neff_fine", "ng_fine", "neff_corr", "ng_corr", "err_lateral_model", "err_window_model",
             "te_frac", "edge_frac", "n_candidates", "flags"]
    with open(os.path.join(a.run_dir, "tracks.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=tkeys, extrasaction="ignore")
        w.writeheader()
        for t, recs in tr.items():
            for r in recs:
                if r:
                    w.writerow(dict(r, target=t))
    import h5py
    with h5py.File(os.path.join(a.run_dir, "tracks.h5"), "w") as h:
        h["lam"] = np.array(lams)
        h["targets"] = np.array(a.targets, dtype="S")
        for k in ("neff_best", "neff_err", "ng_best", "ng_err", "neff_fine", "ng_fine", "neff_corr", "ng_corr",
                  "te_frac", "edge_frac"):
            h[k] = np.array([[r[k] if r else np.nan for r in tr[t]] for t in a.targets], float)
        h["chunk"] = np.array([[r["chunk"] if r else "" for r in tr[t]] for t in a.targets], dtype="S")
        h.attrs["layout"] = ("arrays are (target, lam). *_best = Richardson-extrapolated (fine value if no "
                             "coarse partner); *_corr = (fine - coarse)/3 (nan if none); *_err = estimated error "
                             "of *_best = 0.3|corr| + lateral model (refine='y' chunks) + 12 edge_frac, "
                             "raised to |ng_corr| at suspected anticrossings "
                             "(see combine_chunks.py docstring)")
        h.attrs["plan"] = json.dumps(plan)
    bc = boundary_check(rows, a.targets)
    with open(os.path.join(a.run_dir, "boundary_check.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(bc) + "\n")
    plot(a.run_dir, plan, lams, tr, a.tol)

    print(f"{len(rows)} mode rows, {len(lams)} wavelengths -> {a.run_dir}")
    for t, recs in tr.items():
        got = [r for r in recs if r]
        if not got:
            print(f"  {t}: not found")
            continue
        e1 = np.nanmax([r["neff_err"] for r in got])
        e2 = np.nanmax([r["ng_err"] for r in got])
        nflag = sum(1 for r in got if r["flags"])
        nnot = sum(1 for r in got if "not extrapolated" in r["flags"])
        print(f"  {t}: {len(got)}/{len(lams)} λ, max error estimate n_eff {e1:.1e}, n_g {e2:.1e}"
              f"{f', {nflag} flagged' if nflag else ''}{f', {nnot} NOT extrapolated' if nnot else ''}")
    for line in bc:
        print("  " + line)


if __name__ == "__main__":
    main()
