"""Convergence study for the TFLN rib mode solve: simulation window and grid pitch.

(A) Window: fixed pitch, growing square window, at the LONGEST wavelength (largest mode,
    most likely to be clipped by the periodic window). Also reports how much |E|² sits
    near the window edge, the per-mode clipping diagnostic stored by every sweep.
(B) Pitch: fixed window, dy halved twice, at short / mid / long wavelength. Every grid
    keeps the LN interfaces ON pixel centers (tfln_rib.align_to_grid, frac = 0) and the
    film an integer number of pixels thick. Otherwise the error prefactor jumps with the
    sub-pixel position of the interfaces and the sequence zig-zags instead of converging.
    With fixed alignment the error is C·dy², so Richardson extrapolation of the two
    finest grids gives the grid-converged value, and each pitch's error relative to it.
(C) Lateral pitch: at the shortest wavelength, does dx need to be as fine as dy?

Modes are matched between runs by Hermite-Gaussian label (TE00, TM00, TE10, ...), not
solver index.

Run from this folder:  python run_convergence.py           (~1 h on a 12-core CPU)
                       python run_convergence.py --quick   (coarser, ~15 min)
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from utils.setup_jax import setup_jax

setup_jax("cpu")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import geometry_gen.tfln_rib as R
import tfln_modes as M
from tfln_plots import SERIES, INK

LABELS = ("TE00", "TM00", "TE10", "TM10")


def solve_point(geom, Lx, Ly, dx, dy, lam, settings):
    grid = R.make_grid(Lx, Ly, dx, dy)
    g_al = R.align_to_grid(geom, grid, frac=0.0)
    diel = R.RibDielectric(g_al, grid)
    t = time.perf_counter()
    sw, _ = M.solve_wavelength(diel, lam, settings)
    dt = time.perf_counter() - t
    out = dict(Lx=Lx, Ly=Ly, dx=grid.Lx / grid.Nx, dy=grid.Ly / grid.Ny, Nx=grid.Nx, Ny=grid.Ny,
               lam=lam, time_s=dt, y_bot=g_al.y0, all_modes=[], modes={})
    labels, g = sw.labels[0], sw.data
    for j in range(sw.n_modes):
        rec = dict(idx=j, label=labels[j], neff=float(g["neff"][0, j]), ng=float(g["ng"][0, j]),
                   te_frac=float(g["te_frac"][0, j]), hg_err=float(g["hg_err"][0, j]),
                   edge_frac=float(g["edge_frac"][0, j]), guided=bool(g["guided"][0, j]),
                   converged=bool(g["converged"][0, j]))
        out["all_modes"].append(rec)
        lab = labels[j]
        if rec["guided"] and lab and (lab not in out["modes"] or rec["hg_err"] < out["modes"][lab]["hg_err"]):
            out["modes"][lab] = rec
    conv = all(m["converged"] for m in out["all_modes"])
    print(f"  L={Lx:g}x{Ly:g}  dx={1e3 * out['dx']:5.2f} dy={1e3 * out['dy']:5.2f} nm  "
          f"N={grid.Nx}x{grid.Ny}  λ={lam:.3f}  {dt:7.1f} s  {'' if conv else 'NOT-CONVERGED '}"
          + "  ".join(f"{k}: {v['neff']:.6f}/{v['ng']:.5f} edge {v['edge_frac']:.1e}"
                      for k, v in out["modes"].items() if k in LABELS), flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--replot", action="store_true", help="re-make report + figure from convergence.json")
    ap.add_argument("--align-check", action="store_true",
                    help="(D) after the main study: repeat the 1.0 µm pitch pair with the film 1/3 "
                         "pixel off center (the production alignment) and compare")
    ap.add_argument("--out", default=os.path.join(HERE, "results", "convergence"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "convergence.json")
    if args.replot or args.align_check:
        with open(path) as fh:
            res = json.load(fh)
        if args.align_check:
            res["align_check"] = align_check(res)
            with open(path, "w") as fh:
                json.dump(res, fh, indent=1)
        report_and_plot(res, args.out)
        return
    report_and_plot(run_study(args), args.out)


def align_check(res, lam=1.0, L=4.0, pitches=(0.025, 0.0125), frac=1 / 3):
    """Same rib, same pitches as the main study at ``lam``, but the film interfaces 1/3 of a
    pixel off a pixel center (what run_sweep.py uses)."""
    geom = R.RibGeometry(**res["geometry"])
    settings = M.SolverSettings(nev=8, store_fields=False, warm_start=False)
    rows = []
    print(f"\n(D) alignment check at λ = {lam} µm, film interfaces {frac:.3f} pixel off center")
    for d in pitches:
        grid = R.make_grid(L, L, d, d)
        g_al = R.align_to_grid(geom, grid, frac=frac)
        t = time.perf_counter()
        sw, _ = M.solve_wavelength(R.RibDielectric(g_al, grid), lam, settings)
        modes = {}
        for j in range(sw.n_modes):
            lab = sw.labels[0][j]
            if lab in LABELS and sw.guided[0, j] and lab not in modes:
                modes[lab] = dict(neff=float(sw.neff[0, j]), ng=float(sw.ng[0, j]))
        rows.append(dict(lam=lam, Lx=L, dx=d, dy=d, frac=frac, time_s=time.perf_counter() - t, modes=modes))
        print(f"  dy={1e3 * d:5.2f} nm  " + "  ".join(f"{k}: {v['neff']:.6f}/{v['ng']:.5f}" for k, v in modes.items()),
              flush=True)
    return rows


def run_study(args):
    geom = R.RibGeometry()                      # Tidy3D-notebook geometry (edit to test others)
    print(geom.describe())
    settings = M.SolverSettings(nev=8, store_fields=False, warm_start=False)
    lam_long = R.LAM_RANGE[1]

    if args.quick:
        windows, d_win = (4.0, 6.0, 8.0), 0.050
        pitch_cases = [(0.4, 4.0, (0.050, 0.025)), (2.0, 6.0, (0.050, 0.025))]
        lateral = None
    else:
        windows, d_win = (4.0, 5.2, 6.0, 7.2, 8.0), 0.040     # L / d_win even -> same pitch + alignment
        pitch_cases = [(0.4, 4.0, (0.050, 0.025, 0.0125)),
                       (1.0, 4.0, (0.050, 0.025, 0.0125)),
                       (2.0, 6.0, (0.050, 0.025, 0.0125))]
        lateral = (0.4, 4.0, 0.0125, (0.0125, 0.025))

    res = dict(geometry=geom.as_dict(), settings=settings.as_dict(), window=[], pitch={}, lateral=[])
    print(f"\n(A) window at λ = {lam_long} µm, pitch {1e3 * d_win:.1f} nm")
    for L in windows:
        res["window"].append(solve_point(geom, L, L, d_win, d_win, lam_long, settings))
    for lam, L, pitches in pitch_cases:
        print(f"\n(B) pitch at λ = {lam} µm, window {L} µm")
        res["pitch"][str(lam)] = [solve_point(geom, L, L, d, d, lam, settings) for d in pitches]
    if lateral:
        lam, L, dy, dxs = lateral
        print(f"\n(C) lateral pitch at λ = {lam} µm, dy = {1e3 * dy} nm")
        done = {(r["Lx"], round(r["dx"], 9), round(r["dy"], 9)): r for r in res["pitch"].get(str(lam), [])}
        res["lateral"] = [done.get((L, round(dx, 9), round(dy, 9))) or solve_point(geom, L, L, dx, dy, lam, settings)
                          for dx in dxs]
    with open(os.path.join(args.out, "convergence.json"), "w") as fh:   # raw results first
        json.dump(res, fh, indent=1)
    return res


def report_and_plot(res, out_dir):
    import matplotlib.ticker as mticker
    geom = R.RibGeometry(**res["geometry"])
    lam_long = res["window"][0]["lam"]
    d_win = res["window"][0]["dx"]
    report = []
    ref = res["window"][-1]["modes"]
    report.append(f"(A) window, λ = {lam_long} µm: n_eff / n_g minus the L = {res['window'][-1]['Lx']:g} µm value")
    for lab in LABELS:
        if lab in ref:
            cells = [f"L={r['Lx']:g}: " + (f"{r['modes'][lab]['neff'] - ref[lab]['neff']:+.1e}/"
                                          f"{r['modes'][lab]['ng'] - ref[lab]['ng']:+.1e} "
                                          f"(edge {r['modes'][lab]['edge_frac']:.0e})"
                                          if lab in r["modes"] else "--") for r in res["window"][:-1]]
            report.append(f"  {lab}: " + " | ".join(cells))
    res["richardson"] = {}
    for lam, rows in res["pitch"].items():
        f, c = rows[-1]["modes"], rows[-2]["modes"]
        ratio = rows[-2]["dy"] / rows[-1]["dy"]
        report.append(f"(B) λ = {lam} µm: error of each pitch vs Richardson({1e3 * rows[-2]['dy']:g}, "
                      f"{1e3 * rows[-1]['dy']:g} nm); C = error/dy² [µm⁻²]")
        res["richardson"][lam] = {}
        for lab in LABELS:
            if lab not in f or lab not in c:
                continue
            n_inf, _ = M.richardson(f[lab]["neff"], c[lab]["neff"], ratio)
            g_inf, _ = M.richardson(f[lab]["ng"], c[lab]["ng"], ratio)
            res["richardson"][lam][lab] = dict(neff=float(n_inf), ng=float(g_inf))
            cells = []
            for r in rows:
                if lab in r["modes"]:
                    e_n = r["modes"][lab]["neff"] - n_inf
                    e_g = r["modes"][lab]["ng"] - g_inf
                    cells.append(f"dy={1e3 * r['dy']:g}: {e_n:+.1e}/{e_g:+.1e} (C {e_n / r['dy'] ** 2:+.2f})")
            report.append(f"  {lab}: n_inf {n_inf:.6f}, ng_inf {g_inf:.5f} | " + " | ".join(cells))
    if res["lateral"]:
        a, b = res["lateral"]
        report.append(f"(C) λ = {a['lam']} µm, dy = {1e3 * a['dy']:g} nm: effect of dx "
                      f"{1e3 * a['dx']:g} -> {1e3 * b['dx']:g} nm, and solve time {a['time_s']:.0f} -> {b['time_s']:.0f} s")
        for lab in LABELS:
            if lab in a["modes"] and lab in b["modes"]:
                report.append(f"  {lab}: Δneff {b['modes'][lab]['neff'] - a['modes'][lab]['neff']:+.1e}  "
                              f"Δng {b['modes'][lab]['ng'] - a['modes'][lab]['ng']:+.1e}")
    for lam, rows in res["pitch"].items():               # observed convergence ratios
        if len(rows) >= 3:
            parts = []
            for lab in LABELS:
                if all(lab in r["modes"] for r in rows[-3:]):
                    for q in ("neff", "ng"):
                        a, b, c = (r["modes"][lab][q] for r in rows[-3:])
                        parts.append(f"{lab} {q} {(a - b) / (b - c) if c != b else float('nan'):.2f}")
            report.append(f"    λ = {lam} µm: successive-difference ratio (q1-q2)/(q2-q3) "
                          f"(4 = clean 2nd order): " + ", ".join(parts))
    if res.get("align_check"):
        rows = res["align_check"]
        lam = str(rows[0]["lam"])
        ref = res["richardson"].get(lam, {})
        report.append(f"(D) λ = {lam} µm, film interfaces {rows[0]['frac']:.3f} pixel off center "
                      f"(production alignment), error vs the frac-0 Richardson limit; C = error/dy²")
        for lab in LABELS:
            if lab in ref and all(lab in r["modes"] for r in rows):
                cells = [f"dy={1e3 * r['dy']:g}: {r['modes'][lab]['neff'] - ref[lab]['neff']:+.1e}/"
                         f"{r['modes'][lab]['ng'] - ref[lab]['ng']:+.1e} "
                         f"(C {(r['modes'][lab]['neff'] - ref[lab]['neff']) / r['dy'] ** 2:+.2f})" for r in rows]
                n_inf, _ = M.richardson(rows[-1]["modes"][lab]["neff"], rows[-2]["modes"][lab]["neff"],
                                        rows[-2]["dy"] / rows[-1]["dy"])
                g_inf, _ = M.richardson(rows[-1]["modes"][lab]["ng"], rows[-2]["modes"][lab]["ng"],
                                        rows[-2]["dy"] / rows[-1]["dy"])
                report.append(f"  {lab}: " + " | ".join(cells) + f" | its own Richardson limit differs from "
                              f"the frac-0 one by {n_inf - ref[lab]['neff']:+.1e} (n_eff), {g_inf - ref[lab]['ng']:+.1e} (n_g)")
    report = "\n".join(report)
    print("\n" + report)
    with open(os.path.join(out_dir, "convergence.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    with open(os.path.join(out_dir, "convergence.txt"), "w", encoding="utf-8") as fh:
        fh.write(geom.describe() + "\n\n" + report + "\n")

    # ---- figure -------------------------------------------------------------------------
    n_p = len(res["pitch"])
    fig, axes = plt.subplots(2, 1 + n_p, figsize=(5.0 * (1 + n_p), 8.6), constrained_layout=True)
    rows = res["window"]
    for c, lab in enumerate(LABELS):
        if lab not in ref:
            continue
        pts = [(r["Lx"], r["modes"][lab]) for r in rows if lab in r["modes"]]
        axes[0, 0].semilogy([p[0] for p in pts[:-1]],
                            [max(abs(p[1]["neff"] - ref[lab]["neff"]), 1e-12) for p in pts[:-1]],
                            "o-", color=SERIES[c], label=lab)
        axes[1, 0].semilogy([p[0] for p in pts], [max(p[1]["edge_frac"], 1e-16) for p in pts],
                            "o-", color=SERIES[c], label=lab)
    axes[0, 0].set(xlabel="window L [µm]", ylabel=r"$|n_\mathrm{eff}(L) - n_\mathrm{eff}(L_\mathrm{max})|$",
                   title=f"(A) window, λ = {lam_long} µm, pitch {1e3 * d_win:g} nm")
    axes[1, 0].set(xlabel="window L [µm]", ylabel="fraction of |E|² within 0.25 µm of the edge",
                   title="(A) field at the window edge")
    for k, (lam, rows) in enumerate(res["pitch"].items()):
        rr = res["richardson"][lam]
        dys = np.array([r["dy"] for r in rows])
        for c, lab in enumerate(LABELS):
            if lab not in rr:
                continue
            for row, q in ((0, "neff"), (1, "ng")):
                pts = [(1e3 * r["dy"], abs(r["modes"][lab][q] - rr[lab][q])) for r in rows if lab in r["modes"]]
                axes[row, 1 + k].loglog([p[0] for p in pts], [max(p[1], 1e-12) for p in pts], "o-",
                                        color=SERIES[c], label=lab)
        for row in (0, 1):
            ax = axes[row, 1 + k]
            ax.loglog(1e3 * dys, 1.0 * dys ** 2, color=INK, ls="--", lw=0.8, label=r"$1.0\,dy^2$")
        axes[0, 1 + k].set(xlabel="pitch dx = dy [nm]", ylabel=r"$|n_\mathrm{eff} - n_\mathrm{eff}^\infty|$",
                           title=f"(B) λ = {lam} µm vs Richardson limit")
        axes[1, 1 + k].set(xlabel="pitch dx = dy [nm]", ylabel=r"$|n_g - n_g^\infty|$",
                           title=f"(B) λ = {lam} µm, group index")
        for row in (0, 1):                          # plain tick labels at the pitches used
            ax = axes[row, 1 + k]
            ax.xaxis.set_major_locator(mticker.FixedLocator(1e3 * dys))
            ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%g"))
            ax.xaxis.set_minor_locator(mticker.NullLocator())
    for ax in axes.ravel():
        ax.grid(alpha=0.25, which="both", lw=0.5)
        if ax.has_data():
            ax.legend(fontsize=8, frameon=False)
    fig.suptitle(geom.describe() + "\nLN interfaces on pixel centers in every run (align_to_grid, frac = 0)",
                 fontsize=9)
    fig.savefig(os.path.join(out_dir, "convergence.png"), dpi=140)
    print(f"\nsaved {out_dir}")


if __name__ == "__main__":
    main()
