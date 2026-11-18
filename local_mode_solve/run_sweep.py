#%% [markdown]
# Forward mode solve of an X-cut TFLN rib waveguide with mpbx (CPU)
#
# 1. geometry + material check (refractive-index map with the geometry outline)
# 2. wavelength sweep: n_eff, n_g, E-fields, polarization, Hermite-Gaussian labels
# 3. TE00 tracking, n_eff / n_g plots, E-field plots at a chosen wavelength
#
# Frames: mpbx x = horizontal = crystal Z (TE), y = vertical = crystal X (TM), z = propagation.
# Run cell by cell (#%%), or as a script:
#     python run_sweep.py            quick test: 0.4-1.6 µm in 200 nm steps (~40 min on a 12-core CPU)
#     python run_sweep.py --full     0.4-2.0 µm in 10 nm steps (161 wavelengths, 12+ h on CPU: use the cluster)
#     add --batch to write figures to files without opening windows

#%%
import os
import sys


def _find_root(start):
    """Walk up from `start` to the repo root: the folder holding geometry_gen/ and utils/."""
    d = os.path.abspath(start)
    while not (os.path.isfile(os.path.join(d, "geometry_gen", "tfln_rib.py"))
               and os.path.isfile(os.path.join(d, "utils", "setup_jax.py"))):
        if os.path.dirname(d) == d:
            raise FileNotFoundError(f"no geometry_gen/ + utils/ above {os.path.abspath(start)}: set ROOT by hand")
        d = os.path.dirname(d)
    return d


try:
    START = os.path.dirname(os.path.abspath(__file__))
except NameError:                                   # interactive kernel: no __file__, start from the cwd
    START = os.getcwd()
ROOT = _find_root(START)                            # utils/, geometry_gen/
HERE = os.path.join(ROOT, "handoff_tfln_modesolve") # tfln_modes.py, tfln_plots.py; results/ go here
assert os.path.isfile(os.path.join(HERE, "tfln_modes.py")), f"tfln_modes.py not found in {HERE}"
for p in (HERE, ROOT):                              # re-running the cell does not stack duplicates
    if p not in sys.path:
        sys.path.insert(0, p)

from utils.setup_jax import setup_jax

import numpy as np
import matplotlib
BATCH = "--batch" in sys.argv or "--full" in sys.argv
if BATCH:
    matplotlib.use("Agg")                           # write files, no windows
import matplotlib.pyplot as plt

import geometry_gen.tfln_rib as R
import sim_core.tfln_modes as M
import sim_core.tfln_plots as P

FULL = "--full" in sys.argv
OUT = os.path.join(HERE, "results", "sweep_full" if FULL else "sweep_quick")
os.makedirs(OUT, exist_ok=True)

#%% Geometry: the Tidy3D-notebook waveguide (local_mode_solve.ipynb)
geom = R.RibGeometry(
    t_ln=0.400,          # LN film thickness [µm]
    etch_depth=0.400,    # = t_ln -> fully etched, no slab
    w_top=1.800,         # ridge top width [µm]
    sidewall_deg=73.0,   # from horizontal
    t_clad=np.inf,       # SiO2 everywhere (0 = air clad, finite = planar SiO2 above the ridge top)
)
# Window and pitch from run_convergence.py (README.md, results/convergence/):
#  * 6 x 6 µm: TE00/TM00 within 3e-6 (n_eff) / 8e-5 (n_g) of an 8 µm window at λ = 2 µm.
#  * the error is set by the VERTICAL pitch: n_eff error = C·dy², |C| <= 0.5 (TE) / 0.93 (TM)
#    µm⁻² with the film interfaces 1/3 pixel off a pixel center (ALIGN_FRAC). At dy = 12.5 nm
#    that is <= 8e-5 (TE) / 1.5e-4 (TM) in n_eff and ~2e-4 in n_g.
#  * the lateral pitch barely matters: dx 12.5 -> 25 nm moved n_eff by 5e-6 and halved the cost.
DX, DY = 0.025, 0.0125
ALIGN_FRAC = 1 / 3
grid = R.make_grid(Lx=6.0, Ly=6.0, dx=DX, dy=DY)
geom = R.align_to_grid(geom, grid, frac=ALIGN_FRAC)   # fixes where the film interfaces fall in their pixels
print(geom.describe())
print(f"window {grid.Lx} x {grid.Ly} µm, {grid.Nx} x {grid.Ny} px "
      f"(dx = {1e3 * grid.Lx / grid.Nx:.1f} nm, dy = {1e3 * grid.Ly / grid.Ny:.1f} nm)")
R.check_material_frame(1.55)

#%% Check the dielectric: index map + outline, sidewall zoom, line cuts
diel = R.RibDielectric(geom, grid)
for lam_chk in (0.4, 1.55):
    R.plot_index_map(geom, grid, lam_chk, diel=diel, fname=os.path.join(OUT, f"index_map_{1e3 * lam_chk:.0f}nm.png"))

#%% Sweep
settings = M.SolverSettings(
    nev=10 if FULL else 8,   # modes per wavelength (ranked by n_eff). At 0.4 µm TE00 is the 8th
                             # mode of this guide (TM00..TM60 sit above it), so 8 is the bare
                             # minimum (what the quick test used); 10 leaves margin.
    warm_start=True,
    label_modes=True,
    store_fields=True,   # complex64 E of every mode: 8 x 3 x 240 x 480 x 8 B ≈ 22 MB per λ in RAM/disk
)
if FULL:
    lams = np.round(np.arange(0.400, 2.0001, 0.010), 4)     # 161 wavelengths, as in Tidy3D
else:
    lams = np.round(np.arange(0.400, 1.6001, 0.200), 4)     # quick test: 0.4, 0.6, ..., 1.6
sweep = M.run_sweep(geom, grid, lams, settings,
                    out=os.path.join(OUT, "modesweep.h5"),
                    checkpoint_dir=os.path.join(OUT, "per_lambda"))
sweep.to_csv(os.path.join(OUT, "modesweep.csv"))
print(sweep.table())

#%% TE00 (and TM00) tracking
te00 = M.track_mode(sweep, "TE00")
tm00 = M.track_mode(sweep, "TM00")
print(te00.summary())
print(tm00.summary())

#%% n_eff and n_g vs wavelength
P.plot_dispersion(sweep, track=te00, fname=os.path.join(OUT, "dispersion.png"))

#%% E-fields of every mode at one wavelength: pick the wavelength and (optionally) a mode
LAM_PICK = 0.8           # µm, nearest stored wavelength is used
MODE_PICK = None         # None -> all modes; an int or list -> just those mode indices
i = sweep.index_of(LAM_PICK)
P.plot_mode_fields(sweep, LAM_PICK, modes=MODE_PICK, highlight=te00.idx[i],
                   fname=os.path.join(OUT, f"fields_{1e3 * sweep.lam[i]:.0f}nm.png"))
P.plot_mode_fields(sweep, sweep.lam[-1], highlight=te00.idx[-1],
                   fname=os.path.join(OUT, f"fields_{1e3 * sweep.lam[-1]:.0f}nm.png"))

#%% Reload from disk (what a cluster post-processing step would do)
sweep2 = M.ModeSweep.load(os.path.join(OUT, "modesweep.h5"), fields=False)
assert np.array_equal(sweep2.data["neff"], sweep.data["neff"])
print(f"reloaded {sweep2.n_lam} wavelengths x {sweep2.n_modes} modes from {OUT}")
if BATCH:
    plt.close("all")
else:
    plt.show()
