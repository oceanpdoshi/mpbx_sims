"""Compare an mpbx ModeSweep against a Tidy3D mode-solver CSV (ModeSolverData.to_dataframe()
written by Tidy3D_sims/local_mode_solve.ipynb) for the same geometry.

    python compare_tidy3d.py <modesweep.h5> <tidy3d.csv> [out.png]

Polarization mapping: the notebook's mode plane is y-normal with x vertical and z
horizontal, so Tidy3D's "TE (Ex) fraction" is the VERTICAL (crystal X) share = 1 - te_frac
here. Each tracked mpbx branch (TE00, TM00) is matched, per wavelength, to the Tidy3D mode
of the same polarization with the closest n_eff.
"""
import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                        # utils/, geometry_gen/ (HERE: tfln_modes, tfln_plots)
sys.path[:0] = [p for p in (ROOT, HERE) if p not in sys.path]


def load_tidy3d_csv(path):
    rows = list(csv.DictReader(open(path)))
    lam = np.array([float(r["wavelength"]) for r in rows])
    return dict(lam=lam,
                neff=np.array([float(r["n eff"]) for r in rows]),
                ng=np.array([float(r["group index"]) if r["group index"] else np.nan for r in rows]),
                te_frac=1.0 - np.array([float(r["TE (Ex) fraction"]) for r in rows]),   # -> our te_frac
                mode=np.array([int(r["mode_index"]) for r in rows]))


def compare(sweep, td, targets=("TE00", "TM00")):
    import sim_core.tfln_modes as M
    out = {}
    for tgt in targets:
        tr = M.track_mode(sweep, tgt)
        want_te = tgt.startswith("TE")
        rows = []
        for i, lam in enumerate(sweep.lam):
            if tr.idx[i] < 0:
                continue
            sel = np.isclose(td["lam"], lam, atol=1e-6) & ((td["te_frac"] > 0.5) == want_te)
            if not sel.any():
                continue
            j = np.flatnonzero(sel)[np.argmin(np.abs(td["neff"][sel] - tr.neff[i]))]
            rows.append((lam, tr.neff[i], td["neff"][j], tr.ng[i], td["ng"][j], td["mode"][j]))
        out[tgt] = np.array(rows) if rows else np.zeros((0, 6))
    return out


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    from utils.setup_jax import setup_jax
    setup_jax("cpu", verbose=False)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import sim_core.tfln_modes as M
    from sim_core.tfln_plots import SERIES

    sweep = M.ModeSweep.load(sys.argv[1], fields=False)
    td = load_tidy3d_csv(sys.argv[2])
    res = compare(sweep, td)
    for tgt, a in res.items():
        print(f"\n{tgt}:  λ [µm]   n_eff mpbx  n_eff Tidy3D  Δ(mpbx-T3D)   n_g mpbx  n_g Tidy3D  Δ      (Tidy3D mode)")
        for lam, nm, nt, gm, gt, mt in a:
            print(f"        {lam:6.3f}  {nm:.6f}   {nt:.6f}    {nm - nt:+.2e}    {gm:.5f}   {gt:.5f}  {gm - gt:+.1e}  ({int(mt)})")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for c, (tgt, a) in enumerate(res.items()):
        if len(a):
            axes[0].plot(a[:, 0], a[:, 1] - a[:, 2], "o-", color=SERIES[c], label=tgt)
            axes[1].plot(a[:, 0], a[:, 3] - a[:, 4], "o-", color=SERIES[c], label=tgt)
    for ax, q in zip(axes, (r"$n_\mathrm{eff}$", r"$n_g$")):
        ax.axhline(0, color="#52514e", lw=0.8)
        ax.set(xlabel="wavelength λ [µm]", ylabel=f"{q}: mpbx − Tidy3D", title=f"{q} difference")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
    out = sys.argv[3] if len(sys.argv) > 3 else os.path.join(os.path.dirname(sys.argv[1]), "compare_tidy3d.png")
    fig.savefig(out, dpi=140)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
