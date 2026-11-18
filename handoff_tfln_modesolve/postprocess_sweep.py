"""Post-process a saved ModeSweep without solving anything: (re)label modes, track TE00 /
TM00, write the n_eff / n_g figure, E-field figures at chosen wavelengths, and a CSV.

    python postprocess_sweep.py results/sweep_quick/modesweep.h5
    python postprocess_sweep.py <file.h5 or folder of per-λ .h5> --relabel 6 --lam 0.8 1.55 --mode 2 --save

--relabel N   recompute Hermite-Gaussian labels up to order N from the stored fields
--lam ...     wavelengths [µm] for the field figures (default: shortest and longest)
--mode ...    mode indices to plot (default: all); the tracked TE00 is always highlighted
--save        write the relabeled sweep back to the .h5
--out DIR     output folder (default: next to the input)
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from utils.setup_jax import setup_jax

setup_jax("cpu", verbose=False)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tfln_modes as M
import tfln_plots as P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--relabel", type=int, default=None)
    ap.add_argument("--lam", type=float, nargs="*", default=None)
    ap.add_argument("--mode", type=int, nargs="*", default=None)
    ap.add_argument("--save", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    is_dir = os.path.isdir(a.path)
    if is_dir and a.relabel and a.save:                 # relabel every per-λ file in place
        for f in sorted(x for x in os.listdir(a.path) if x.endswith(".h5")):
            p = os.path.join(a.path, f)
            M.ModeSweep.save(M.relabel(M.ModeSweep.load(p), max_order=a.relabel, verbose=False), p)
        print(f"relabeled (max_order={a.relabel}) and saved every .h5 in {a.path}")
    sweep = M.ModeSweep.gather(a.path) if is_dir else M.ModeSweep.load(a.path)
    out = a.out or (a.path if is_dir else os.path.dirname(os.path.abspath(a.path)))
    os.makedirs(out, exist_ok=True)
    print(f"{sweep.n_lam} wavelengths ({sweep.lam[0]:.3f}-{sweep.lam[-1]:.3f} µm) x {sweep.n_modes} modes; "
          f"fields {'yes' if sweep.E is not None else 'no'}; solved on {sweep.meta.get('backend')}")
    print(sweep.geometry.describe())

    if a.relabel and not (is_dir and a.save):
        M.relabel(sweep, max_order=a.relabel)
        if a.save:
            sweep.save(a.path)
            print(f"saved relabeled sweep to {a.path}")
    te00 = M.track_mode(sweep, "TE00")
    tm00 = M.track_mode(sweep, "TM00")
    print(te00.summary())
    print(tm00.summary())
    sweep.to_csv(os.path.join(out, "modesweep.csv"))
    P.plot_dispersion(sweep, track=te00, fname=os.path.join(out, "dispersion.png"))
    if sweep.E is not None:
        for lam in (a.lam or [sweep.lam[0], sweep.lam[-1]]):
            i = sweep.index_of(lam)
            P.plot_mode_fields(sweep, lam, modes=a.mode, highlight=te00.idx[i],
                               fname=os.path.join(out, f"fields_{1e3 * sweep.lam[i]:.0f}nm.png"))
            plt.close("all")
    print(f"figures + CSV in {out}")


if __name__ == "__main__":
    main()
