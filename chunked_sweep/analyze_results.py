#%% [markdown]
# Look at the results of run_geometry.py (cell by cell in VS Code / Jupyter, or as a script).
# Set RUN to an output folder (one sub-folder per geometry).

#%%
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
sys.path.insert(0, HERE)
import results_io as rio  # noqa: E402

RUN = os.path.join(HERE, "..", "outputs", "EDIT_ME")       # <- your run folder
if "ipykernel" not in sys.modules and len(sys.argv) > 1:    # as a script: python analyze_results.py RUN
    RUN = sys.argv[1]
geoms = rio.list_geometries(RUN)
if geoms.empty:
    raise SystemExit(f"no geometry results (<name>/<name>_modes.h5) under {os.path.abspath(RUN)}: set RUN")
print(geoms[[c for c in ("name", "t_ln", "etch_depth", "w_top", "sidewall_deg", "t_clad", "t_undercut", "complete")
             if c in geoms]].to_string())                    # t_undercut: runs with undercut ribs only

#%% One geometry: dispersion of the tracked modes
G = geoms.iloc[0]["folder"]
tracks = rio.load_tracks(G)
fig, ax = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
for t, d in tracks.groupby("target"):
    ls = "-" if t.startswith("TE") else "--"
    ax[0].plot(d["lam"], d["neff"], ls, label=t)
    ax[1].plot(d["lam"], d["ng"], ls, label=t)
ax[0].set(xlabel="λ [µm]", ylabel="n_eff")
ax[1].set(xlabel="λ [µm]", ylabel="n_g")
ax[0].legend(ncol=2, fontsize=8)
for a in ax:
    a.grid(alpha=0.3)

#%% Every mode at every wavelength (to see crossings / anticrossings the labels may hide)
modes, meta = rio.load_modes(G)
guided = modes[modes["guided"]]
fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
for pol, mk in (("TE", "o"), ("TM", "^")):
    d = guided[guided["label"].str.startswith(pol)]
    ax.scatter(d["lam"], d["neff"], s=4, marker=mk, label=pol)
flag = guided[guided["flags"].str.len() > 0]
ax.scatter(flag["lam"], flag["neff"], s=18, facecolors="none", edgecolors="k", lw=0.5, label="flagged")
ax.set(xlabel="λ [µm]", ylabel="n_eff", title="all guided modes")
ax.legend()

#%% Field of one mode (only if the run stored fields: FIELDS=all)
if meta.get("field_files"):
    E, info = rio.load_field(G, lam=0.8, label="TE00")
    X, Y = np.meshgrid(info["x"], info["y"], indexing="ij")
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6), constrained_layout=True)
    for k, comp in enumerate(("Ex", "Ey", "Ez")):
        ax[k].pcolormesh(X, Y, np.abs(E[k]), shading="auto")
        ax[k].set(aspect="equal", title=f"|{comp}|  {info['label']}  λ = {info['lam']:.3f} µm  n_eff = {info['neff']:.5f}")
else:
    print("no field files in this run (FIELDS=none)")

#%% A geometry sweep: n_g of TE00 at one wavelength vs ridge width
allt = rio.load_all_tracks(RUN, targets=["TE00", "TM00"])
lam0 = 1.55
sel = allt[np.isclose(allt["lam"], allt["lam"].iloc[(allt["lam"] - lam0).abs().argmin()])]
for t, d in sel.groupby("target"):
    d = d.sort_values("w_top")
    plt.figure(figsize=(5, 3.5))
    plt.plot(d["w_top"], d["ng"], "o-")
    plt.xlabel("w_top [µm]")
    plt.ylabel(f"n_g of {t} at {lam0} µm")

plt.show()
