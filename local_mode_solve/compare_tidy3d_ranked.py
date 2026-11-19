"""Label-free comparison of an mpbx modes CSV against a Tidy3D mode-solver CSV.

    python compare_tidy3d_ranked.py <mpbx_modes.csv> <tidy3d.csv> [out_prefix]

No field loading or mode labelling: at each wavelength the modes of each solver are
rank-ordered by n_eff, rank 1 = highest n_eff (the fundamental), rank 2 = next, ... and
mode k of mpbx is compared with mode k of Tidy3D. The two solvers return different numbers
of modes per wavelength (Tidy3D: fixed num_modes; mpbx: nev per parity sector, varying by
chunk), so only ranks 1..min(N_mpbx, N_tidy3d) are compared at each wavelength.

Writes <out_prefix>_neff.png and <out_prefix>_ng.png (default prefix: next to the mpbx CSV).
Each has three panels: overlay vs λ, Δ = mpbx − Tidy3D, and Δ / mpbx.
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
MARKERS = ["o", "^", "s", "v", "D", "P", "X", "<", ">", "h"]


def load(path, lam_col, neff_col, ng_col):
    d = pd.read_csv(path)
    return pd.DataFrame({"lam": d[lam_col].astype(float).round(4),
                         "neff": d[neff_col].astype(float),
                         "ng": pd.to_numeric(d[ng_col], errors="coerce")})


def rank(df):
    df = df.sort_values(["lam", "neff"], ascending=[True, False]).copy()
    df["rank"] = df.groupby("lam").cumcount() + 1
    return df


def paired(mpbx, td):
    """Inner join on (λ, rank): keeps ranks 1..min(N_mpbx, N_tidy3d) at every common λ."""
    return rank(mpbx).merge(rank(td), on=["lam", "rank"], suffixes=("_m", "_t"))


def plot_quantity(p, q, qlabel, out):
    ranks = sorted(p["rank"].unique())
    fig, axes = plt.subplots(3, 1, figsize=(8, 11), sharex=True, constrained_layout=True)
    for k in ranks:
        s = p[p["rank"] == k].sort_values("lam")
        c, mk = SERIES[(k - 1) % len(SERIES)], MARKERS[(k - 1) % len(MARKERS)]
        kw = dict(color=c, marker=mk, ms=4, lw=0.8, label=f"mode {k}")
        m, t = s[f"{q}_m"].to_numpy(), s[f"{q}_t"].to_numpy()
        axes[0].plot(s["lam"], m, **kw)
        axes[0].plot(s["lam"], t, "--", color="k", lw=1.0, zorder=5,
                     label="Tidy3D" if k == ranks[0] else None)
        axes[1].plot(s["lam"], m - t, **kw)
        axes[2].plot(s["lam"], (m - t) / m, **kw)
    axes[0].set(ylabel=qlabel, title=f"{qlabel}: mpbx (markers) vs Tidy3D (dashed), rank-ordered by $n_\\mathrm{{eff}}$")
    axes[1].set(ylabel=f"Δ{qlabel} (mpbx − Tidy3D)")
    axes[2].set(ylabel=f"Δ{qlabel} / {qlabel}$_\\mathrm{{mpbx}}$", xlabel="wavelength λ [µm]")
    for ax in axes[1:]:
        ax.axhline(0, color="#52514e", lw=0.8)
    for ax in axes:
        ax.grid(alpha=0.25)
    axes[0].legend(frameon=False, ncol=2, fontsize=8)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved {out}")


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    sys.stdout.reconfigure(encoding="utf-8")         # λ, Δ on Windows consoles
    mpbx_csv, td_csv = sys.argv[1], sys.argv[2]
    if "lam_um" not in pd.read_csv(mpbx_csv, nrows=0).columns:   # accept the two CSVs in either order
        mpbx_csv, td_csv = td_csv, mpbx_csv
    prefix = sys.argv[3] if len(sys.argv) > 3 else os.path.splitext(mpbx_csv)[0].removesuffix("_modes") + "_vs_tidy3d"

    mpbx = load(mpbx_csv, "lam_um", "neff", "ng")
    td = load(td_csv, "wavelength", "n eff", "group index")
    p = paired(mpbx, td)

    n_m, n_t = mpbx.groupby("lam").size(), td.groupby("lam").size()
    common = n_m.index.intersection(n_t.index)
    print(f"λ: mpbx {len(n_m)}, Tidy3D {len(n_t)}, common {len(common)}")
    print(f"modes/λ: mpbx {n_m.min()}–{n_m.max()}, Tidy3D {n_t.min()}–{n_t.max()}; compared ranks 1–{p['rank'].max()}")
    print("\nrank   max|Δn_eff|   max|Δn_g|")
    for k, s in p.groupby("rank"):
        print(f"{k:4d}   {np.nanmax(np.abs(s.neff_m - s.neff_t)):.2e}     {np.nanmax(np.abs(s.ng_m - s.ng_t)):.2e}")

    plot_quantity(p, "neff", r"$n_\mathrm{eff}$", prefix + "_neff.png")
    plot_quantity(p, "ng", r"$n_g$", prefix + "_ng.png")


if __name__ == "__main__":
    main()
