"""Summarize the study_chunks.py results (results/*.jsonl) into results/studies_summary.txt.

    python analyze_studies.py
"""
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
TARGETS = ("TE00", "TM00", "TE10", "TM10", "TE20", "TM20", "TE30", "TM30")
out_lines = []


def out(s=""):
    out_lines.append(s)
    print(s)


def load(pattern):
    recs = []
    for f in sorted(glob.glob(os.path.join(RES, pattern))):
        for line in open(f, encoding="utf-8"):
            recs.append(json.loads(line))
    return recs


def pick(rec, label):
    """The guided mode with this node label (purest polarization if several)."""
    c = [m for m in rec["modes"] if m["label"] == label and m["guided"]]
    if not c:
        return None
    te = label.startswith("TE")
    return min(c, key=lambda m: abs(m["te_frac"] - (1.0 if te else 0.0)))


# ---------------------------------------------------------------------------------------
def census():
    recs = sorted(load("census.jsonl"), key=lambda r: r["lam"])
    if not recs:
        return
    out("=" * 100)
    out("CENSUS (6 x 6 µm, 40 nm pitch, both x-parity sectors): rank of each target inside its sector")
    out("  sector -1 holds TE00, TM10, TE20, TM30, ...   sector +1 holds TM00, TE10, TM20, TE30, ...")
    out(f"  {'λ':>5}  " + "  ".join(f"{t:>5}" for t in TARGETS) + "   bands needed (-1/+1)   modes above the deepest target")
    for r in recs:
        ranks = {}
        need = {-1: 0, 1: 0}
        for t in TARGETS:
            m = pick(r, t)
            ranks[t] = "  -  " if m is None else f"{m['sector']:+d}:{m['rank_in_sector']}"
            if m is not None:
                need[m["sector"]] = max(need[m["sector"]], m["rank_in_sector"] + 1)
        out(f"  {r['lam']:5.2f}  " + "  ".join(f"{ranks[t]:>5}" for t in TARGETS)
            + f"   {need[-1]:2d} / {need[1]:2d}   (solved {r['nev_per_sector']}/sector)")
    out("\n  in-sector order (label, n_eff) at each λ — same-sector neighbours anticross, opposite-sector ones cross:")
    for r in recs:
        for s in (-1, 1):
            ms = sorted((m for m in r["modes"] if m["sector"] == s), key=lambda m: -m["neff"])
            out(f"  {r['lam']:5.2f} {s:+d}: " + ", ".join(
                f"{m['label']}{'' if m['guided'] else '(u)'} {m['neff']:.4f}" for m in ms))


# ---------------------------------------------------------------------------------------
def window():
    for f in sorted(glob.glob(os.path.join(RES, "window_*.jsonl"))):
        recs = [json.loads(line) for line in open(f, encoding="utf-8")]
        ref = [r for r in recs if r.get("sweep") == "ref"]
        if not ref:
            continue
        ref = ref[-1]
        lam = ref["lam"]
        out("=" * 100)
        out(f"WINDOW at λ = {lam} µm (40 nm pitch): n_eff / n_g minus the {ref['Lx']:.2f} x {ref['Ly']:.2f} µm reference "
            f"[edge_frac in brackets]")
        for sweep, key, fixed in (("x", "Lx", "Ly"), ("y", "Ly", "Lx")):
            rs = sorted([r for r in recs if r.get("sweep") == sweep], key=lambda r: r[key])
            out(f"  {key} sweep at {fixed} = {ref[fixed]:.2f} µm")
            for t in TARGETS:
                m0 = pick(ref, t)
                if m0 is None:
                    continue
                cells = []
                for r in rs:
                    m = pick(r, t)
                    cells.append(f"{r[key]:.2f}: " + ("missing" if m is None else
                                 f"{m['neff'] - m0['neff']:+.1e}/{m['ng'] - m0['ng']:+.1e} [{m['edge_frac']:.0e}]"))
                out(f"    {t}: " + " | ".join(cells))


# ---------------------------------------------------------------------------------------
def pitch():
    for f in sorted(glob.glob(os.path.join(RES, "pitch_*.jsonl"))):
        recs = [json.loads(line) for line in open(f, encoding="utf-8")]
        pitch_series(sorted([r for r in recs if r.get("sweep") == "dy"], key=lambda r: -r["dy"]),
                     [r for r in recs if r.get("sweep") == "dx"])
        dxr = sorted([r for r in recs if r.get("sweep") == "dx"], key=lambda r: -r["dy"])
        if len(dxr) >= 2:
            out(f"  ... and the same dy series at dx = {1e3 * dxr[0]['dx']:.2f} nm:")
            pitch_series(dxr, [])
        # lateral series at dy = 12.5 nm: dx = 25, 12.47, 8.33 nm
        at125 = lambda r: abs(r["dy"] - 0.0125) < 1e-6
        xs = sorted([r for r in recs if at125(r) and r.get("sweep") in ("dy", "dx", "dx3")], key=lambda r: -r["dx"])
        if len(xs) >= 3:
            h = np.array([r["dx"] for r in xs])
            out(f"  LATERAL series at dy = 12.5 nm, dx = " + ", ".join(f"{1e3 * x:.2f}" for x in h) + " nm: values; "
                "slope ratio (1 = clean dx²); x-Richardson of the two finest; error of dx = 25 and 12.5 nm")
            for t in TARGETS:
                ms = [pick(r, t) for r in xs]
                if any(m is None for m in ms):
                    continue
                for q in ("neff", "ng"):
                    v = np.array([m[q] for m in ms])
                    a = (v[0] - v[1]) / (h[0] ** 2 - h[1] ** 2)
                    b = (v[1] - v[2]) / (h[1] ** 2 - h[2] ** 2)
                    r_ = h[1] / h[2]
                    Rx = v[2] + (v[2] - v[1]) / (r_ ** 2 - 1)
                    out(f"    {t} {q:4s}: " + "  ".join(f"{x:.6f}" for x in v)
                        + f" | ratio {a / b if b else np.nan:.2f} | Rx {Rx:.6f} | err(dx 25) {v[0] - Rx:+.1e}"
                        + f"  err(dx 12.5) {v[1] - Rx:+.1e}")


def pitch_series(dyr, dxr):
    if True:
        if len(dyr) < 2:
            return
        lam = dyr[0]["lam"]
        out("=" * 100)
        out(f"PITCH at λ = {lam} µm, window {dyr[0]['Lx']:.3f} x {dyr[0]['Ly']:.3f} µm, dx = 25 nm, interfaces 1/3 pixel "
            f"off-centre; dy = " + ", ".join(f"{1e3 * r['dy']:.2f}" for r in dyr) + " nm")
        out("  per target: value at each dy; Richardson estimates; error of the dy = 12.5 nm value and of R(25,12.5)"
            " vs the best (finest-pair) estimate; C = err/dy² [µm⁻²]")
        h = np.array([r["dy"] for r in dyr])
        for t in TARGETS:
            ms = [pick(r, t) for r in dyr]
            if any(m is None for m in ms):
                out(f"  {t}: missing at some pitch: " + ", ".join("ok" if m else "--" for m in ms))
                continue
            for q in ("neff", "ng"):
                v = np.array([m[q] for m in ms])
                i25 = int(np.argmin(abs(h - 0.025)))
                i125 = int(np.argmin(abs(h - 0.0125)))
                R2 = v[i125] + (v[i125] - v[i25]) / 3.0                       # R(25, 12.5)
                r_last = h[-2] / h[-1]
                Rbest = v[-1] + (v[-1] - v[-2]) / (r_last ** 2 - 1)           # finest pair
                if len(v) >= 3:
                    A = np.vstack([np.ones_like(h[-3:]), h[-3:] ** 2]).T
                    fit = np.linalg.lstsq(A, v[-3:], rcond=None)[0]
                    resid = v[-3:] - A @ fit
                else:
                    fit, resid = (np.nan, np.nan), np.array([np.nan])
                ratios = []
                for k in range(len(v) - 2):
                    a = (v[k] - v[k + 1]) / (h[k] ** 2 - h[k + 1] ** 2)
                    b = (v[k + 1] - v[k + 2]) / (h[k + 1] ** 2 - h[k + 2] ** 2)
                    ratios.append(a / b if b else np.nan)
                err125 = v[i125] - Rbest
                out(f"  {t} {q:4s}: " + "  ".join(f"{x:.6f}" for x in v)
                    + f" | R(25,12.5) {R2:.6f}  R(finest) {Rbest:.6f}  fit3 {fit[0]:.6f} (resid {np.abs(resid).max():.0e})"
                    + f" | err(12.5) {err125:+.1e} (C {err125 / 0.0125 ** 2:+.2f})  err(R25,12.5) {R2 - Rbest:+.1e}"
                    + f" | slope ratios (1 = clean h²) " + " ".join(f"{x:.2f}" for x in ratios))
        for r_dx in sorted(dxr, key=lambda r: -r["dy"]):
            same = [r for r in dyr if abs(r["dy"] - r_dx["dy"]) < 1e-9]
            if not same:
                continue
            r_ref = same[0]
            out(f"  dx 25 -> {1e3 * r_dx['dx']:.2f} nm at dy = {1e3 * r_dx['dy']:.2f} nm (Δ = fine-dx minus 25 nm):")
            for t in TARGETS:
                a, b = pick(r_ref, t), pick(r_dx, t)
                if a and b:
                    out(f"    {t}: Δn_eff {b['neff'] - a['neff']:+.1e}  Δn_g {b['ng'] - a['ng']:+.1e}")
        out("  solve times [s]: " + ", ".join(f"{1e3 * r['dy']:.2f} nm {r['Nx']}x{r['Ny']}: {r['t_solve']:.0f}"
                                            for r in dyr + dxr))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    census()
    window()
    pitch()
    with open(os.path.join(RES, "studies_summary.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(out_lines) + "\n")
