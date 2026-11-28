#%% [markdown]
# Normalized SHG efficiency (eta) of the Praevium 2799 QPM devices, from MPBX mode-solver data
#
# Port of Tidy3D_sims/eta_pred_2799_DOIs.ipynb (from "Load the mode solver data" on). The Tidy3D
# ModeSolverData is replaced by a chunked MPBX sweep with stored E-fields (chunked_sweep/results_io.py format):
#   1. load one waveguide width: n_eff / n_g / labels of every mode, TE00 track, E-fields on demand
#   2. n_eff of every mode over the FH and SH ranges; E-fields of every mode at one wavelength
#   3. Delta k and the first-order poling period in a FH window (lambda_SH = lambda_FH / 2)
#   4. TE00 mode areas from the true modal power, TE00 -> TE00 overlap over the LN, kappa, eta (lossless, lossy)
#   5. eta_lossy vs. (alpha_1, alpha_2)
#   6. a measured CW QPM scan -> inhomogeneous-broadening factor F -> predicted peak eta
#
# Frames. mpbx (this file): x = horizontal = crystal Z (TE, n_e), y = vertical = crystal X (TM), z = propagation.
# Tidy3D notebook: z = horizontal, x = vertical, y = propagation. So notebook Ez <-> mpbx Ex (TE),
# notebook Ex <-> mpbx Ey (TM), notebook Ey <-> mpbx Ez (longitudinal). Arrays here are E[c, ix, iy].
#
# What differs from the Tidy3D notebook (all handled below):
#   * Field normalization: mpbx stores E with int Re(E* x H)_z dA = 1 in natural units (eps0 = mu0 = c = 1).
#     The eta formulas only use ratios (A_eff = S / (n_eff max|E_TE|^2)), so this works in those units directly.
#   * Grids: the sweep is solved in wavelength chunks with different windows / pitches, so the FH and SH
#     fields usually sit on different grids (e.g. FH 1.34 um: 25 x 12.5 nm, SH 0.67 um: 12.5 x 12.5 nm).
#     The overlap integral interpolates one field onto the other's grid (linear, inside the LN only).
#   * Each chunk's geometry was shifted vertically by geometry_gen.tfln_rib.align_to_grid (film interface 1/3
#     pixel above a pixel center), i.e. by up to ~6 nm between chunks. Fields are registered in film-centered
#     coordinates (y_c = 0 at mid-film) before the overlap.
#   * Cladding: these runs have a planar 1.35 um SiO2 top cladding (air above); the Tidy3D run was fully clad.
#   * TE00 mode index: taken per wavelength from the stored Hermite-Gaussian TE00 track (it can change inside
#     a window), and cross-checked with the notebook's HG00-overlap tracking (te00_track_hg) on the fields.
#
# Run cell by cell (#%%), or as a script:
#     python eta_pred_2799_DOIs.py                    DEVICE as set in the config cell
#     python eta_pred_2799_DOIs.py DEV20_1.6um        any key of DEVICES
#     add --batch to run headless (no windows: figures are closed instead of shown)
# SAVE_FIGS (on by default, with or without --batch) writes the eta map, the broadening plots and a CSV table.

#%% Imports and paths
import json
import os
import sys
from datetime import datetime

import h5py
import numpy as np
import pandas as pd
import matplotlib as mpl

BATCH = "--batch" in sys.argv
if BATCH:
    mpl.use("Agg")                                    # headless: no windows (show() just closes figures)
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon
from matplotlib.ticker import MultipleLocator
from scipy.constants import c as C_SI, mu_0
from scipy.interpolate import CubicSpline, RegularGridInterpolator


def _find_root(start):
    """Walk up from `start` to the mpbx_sims repo root (the folder holding chunked_sweep/results_io.py)."""
    d = os.path.abspath(start)
    while not os.path.isfile(os.path.join(d, "chunked_sweep", "results_io.py")):
        if os.path.dirname(d) == d:
            raise FileNotFoundError(f"no chunked_sweep/results_io.py above {os.path.abspath(start)}: set ROOT by hand")
        d = os.path.dirname(d)
    return d


try:
    START = os.path.dirname(os.path.abspath(__file__))
except NameError:                                     # interactive kernel: no __file__, start from the cwd
    START = os.getcwd()
ROOT = _find_root(START)
HERE = os.path.join(ROOT, "local_mode_solve")
for p in (HERE, os.path.join(ROOT, "chunked_sweep")):  # re-running the cell does not stack duplicates
    if p not in sys.path:
        sys.path.insert(0, p)

import results_io as rio                              # noqa: E402  (MPBX result-file reader, no JAX)
from qpm_tools import MPBX_SIMS_DIR, ROOT_PREFIX      # noqa: E402  (Dropbox paths, macOS / Windows)


def show(fig=None):
    """plt.show() interactively; in --batch mode just free the figure(s)."""
    if BATCH:
        plt.close(fig if fig is not None else "all")
    else:
        plt.show()


Z_0 = mu_0 * C_SI                                     # [Ohm] impedance of free space
UM = 1e-6                                             # [m / um]
C_UM = C_SI / UM                                      # [um/s] speed of light (td.C_0)

#%% Configuration: device, mode-solver runs, measured scans
# Each device fixes the waveguide width (-> MPBX run), the FH window for Delta k / eta, and the measured-scan
# analysis windows. These are the alternatives that are commented in and out in the Tidy3D notebook.
#   wav_fh_sel          None -> every paired wavelength; 1.55 -> the solved FH wavelength nearest 1.55 um;
#                       (1.48, 1.60) -> every solved FH wavelength in [1.48, 1.60] um
#   int_window_nm       signal part of the scan (integration window); None as the upper end -> end of the scan
#   baseline_windows_nm signal-free parts of the scan, used for the baseline
#   fringe_nm           None -> measure the facet Fabry-Perot fringe period from the scan; a number overrides it
#   scan_min_date       None -> the first (by file name) unsaturated scan of the device; a datetime -> the first one
#                       taken on or after it (the notebook's commented-out filter "for JUST DEV20_1.6um and DEV21_1.6um":
#                       with the Tidy3D modes their May-2026 scans give F < 0 / F > 1; the Sept-2026 ones give
#                       F = 0.441 (DEV20, the value noted in cell 16) and F = 0.600 (DEV21))
#   eta_meas_peak       measured peak eta [%/W/cm^2] for the last comparison plot (None -> skip that plot)
DEVICES = {
    "DEV14_1.8um": dict(width_nm=1800, band="O", wav_fh_sel=(1.320, 1.380),
                        int_window_nm=(1330.0, 1363.0), baseline_windows_nm=[(1320.0, 1330.0), (1360.0, 1370.0)],
                        fringe_nm=None, scan_min_date=None, eta_meas_peak=948.0),
    "DEV20_1.6um": dict(width_nm=1600, band="C", wav_fh_sel=(1.48, 1.60),
                        int_window_nm=(1470.0, 1530.0), baseline_windows_nm=[(1550.0, 1570.0)],
                        fringe_nm=None, scan_min_date=datetime(2026, 9, 1), eta_meas_peak=None),
    "DEV21_1.6um": dict(width_nm=1600, band="C", wav_fh_sel=(1.48, 1.60),
                        int_window_nm=(1485.0, 1540.0), baseline_windows_nm=[(1470.0, 1482.0), (1555.0, 1580.0)],
                        fringe_nm=None, scan_min_date=datetime(2026, 9, 1), eta_meas_peak=None),
    "DEV19_1.6um": dict(width_nm=1600, band="Yb", wav_fh_sel=(1.140, 1.180),
                        int_window_nm=(1150.0, None), baseline_windows_nm=[(1150.0, 1160.0)],
                        fringe_nm=0.024, scan_min_date=None, eta_meas_peak=None),   # VCSEL: fringe period set by hand
}
DEVICE = "DEV14_1.8um"
_cli = [a for a in sys.argv[1:] if a in DEVICES]      # command-line override (ignored in a Jupyter kernel)
if "ipykernel" not in sys.modules:                    # a mistyped device name must not run the default device
    _bad = [a for a in sys.argv[1:] if not a.startswith("-") and a not in DEVICES]
    if _bad:
        raise SystemExit(f"unknown device {_bad}: choose from {list(DEVICES)}")
if _cli:
    DEVICE = _cli[0]
dev = DEVICES[DEVICE]

# MPBX runs with stored E-fields (run_geometry.py --fields all), one per waveguide top width [nm]
MODE_RUNS = {
    1600: os.path.join(MPBX_SIMS_DIR, "tLN400_etch400_w1600_sw73_clad1350nmSiO2-11-19-10-11"),
    1800: os.path.join(MPBX_SIMS_DIR, "tLN400_etch400_w1800_sw73_clad1350nmSiO2-11-19-08-51"),
}

# TE00 mode index (rank by n_eff at each wavelength). None -> the stored TE00 track at each wavelength.
# An int forces that MPBX index (an n_eff rank) at every selected wavelength. Do NOT copy the notebook's
# TE00_idx_FH / TE00_idx_SH: Tidy3D's mode_index is not an n_eff rank (e.g. the 1.8 um SH TE00 is Tidy3D index 0
# but MPBX rank 3, with TM00 above it).
TE00_IDX_FH = None
TE00_IDX_SH = None

# Measured CW QPM scans (the notebook's Windows path, built for either OS)
SCAN_BASE_DIR = os.path.join(ROOT_PREFIX, "MIT Dropbox", "Sagar Doshi", "LN-NLO", "poling-waveguide-shg", "Praevium_2799")
C_BAND_CW_SPECTRA_DIR = os.path.join(SCAN_BASE_DIR, "CW_QPM_Spectra_HP81680A")
O_BAND_CW_SPECTRA_DIR = os.path.join(SCAN_BASE_DIR, "CW_QPM_Spectra_Santec-TSL-570")
YB_BAND_CW_SPECTRA_DIR = os.path.join(SCAN_BASE_DIR, "CW_QPM_Spectra_VCSEL")
BAND_DIRS = {"C": C_BAND_CW_SPECTRA_DIR, "O": O_BAND_CW_SPECTRA_DIR, "Yb": YB_BAND_CW_SPECTRA_DIR}

SAVE_FIGS = True                                      # write the eta map / broadening figures + a CSV table
OUT_DIR = os.path.join(HERE, "results", "eta_pred_2799_DOIs")   # local_mode_solve/results/ is git-ignored
if SAVE_FIGS:
    os.makedirs(OUT_DIR, exist_ok=True)

print(f"Device {DEVICE}: top width {dev['width_nm']} nm, {dev['band']}-band, FH window {dev['wav_fh_sel']} um")

#%% Unit conversions and Miller's rule for chi2(lambda)
def alpha_dB_per_cm_to_per_m(alpha_dB_cm):
    """Power attenuation coefficient: alpha [dB/cm] -> alpha [1/m], with P(z) = P(0) e^{-alpha z}.

    Loss in dB over length L is 10 log10(e) * alpha * L, so alpha [1/m] = alpha [dB/cm] * 100 * ln(10) / 10
    (~23.03 per dB/cm). Works elementwise on arrays.
    """
    return np.asarray(alpha_dB_cm) * 100.0 * np.log(10) / 10.0


def alpha_per_m_to_dB_per_cm(alpha_per_m):
    """Inverse of alpha_dB_per_cm_to_per_m: power attenuation alpha [1/m] -> [dB/cm]."""
    return np.asarray(alpha_per_m) * 10.0 / (100.0 * np.log(10))


# Sellmeier equations for bulk MgO:LN, D. Matsuda et al. Proc. SPIE 9347, 93471S (2015), lambda in um.
# The MPBX runs use the same model (mpbjax.materials.MgO_LiNbO3_matsuda); checked against the stored n_e below.
no_sq = lambda wl: 4.8898 + 0.11738 / (wl**2 - 0.04564) - 0.02662 * wl**2          # noqa: E731
ne_sq = lambda wl: 4.54514 + 0.096471 / (wl**2 - 0.043763) - 0.021502 * wl**2      # noqa: E731

##### Miller's rule: wavelength scaling of chi2 (Ettoumi et al., Opt. Express 18, 6613 (2010), Eqs. 1, 23, 27) #####
#   chi2_zzz(w3; w1, w2)       chi1_zz(w3) * chi1_zz(w1) * chi1_zz(w2)
#   ---------------------  =  ----------------------------------------- ,   chi1_zz(w) = n_e(w)^2 - 1,   w3 = w1 + w2
#   chi2_zzz(w3'; w1', w2')    chi1_zz(w3') * chi1_zz(w1') * chi1_zz(w2')
# For SHG (w1 = w2 = w, w3 = 2w):  chi2(lam) = chi2(lam_ref) * [chi1(lam/2) chi1(lam)^2] / [chi1(lam_ref/2) chi1(lam_ref)^2]

##### Shoji et al. (1997) reference values for congruent LN (ballpark numbers) #####
REF_SHOJI_CHI2_1313nm = 2 * 19.5e-12   # [m/V]
REF_SHOJI_CHI2_1064nm = 2 * 25.2e-12   # [m/V]
REF_SHOJI_CHI2_852nm = 2 * 25.7e-12    # [m/V]


def chi1_e(lam_um):
    """Linear susceptibility seen by z-polarized light, from the Matsuda (2015) MgO:LN n_e Sellmeier."""
    return ne_sq(lam_um) - 1.0


def miller_scale(lam_1, lam_2, lam_1_ref, lam_2_ref, chi1=chi1_e):
    """chi2(w3; w1, w2) / chi2(w3_ref; w1_ref, w2_ref) for sum-frequency mixing, wavelengths in um.

    lam_3 follows from energy conservation, 1/lam_3 = 1/lam_1 + 1/lam_2. Works elementwise on arrays.
    """
    lam_3 = 1.0 / (1.0 / lam_1 + 1.0 / lam_2)
    lam_3_ref = 1.0 / (1.0 / lam_1_ref + 1.0 / lam_2_ref)
    return (chi1(lam_3) * chi1(lam_1) * chi1(lam_2)) / (chi1(lam_3_ref) * chi1(lam_1_ref) * chi1(lam_2_ref))


def chi2_shg(lam_FH, chi2_ref=2 * 24.0e-12, lam_FH_ref=1.55):
    """chi2 for SHG of FH wavelength lam_FH [um], given chi2_ref for SHG of lam_FH_ref [um]. Same units as chi2_ref."""
    return chi2_ref * miller_scale(lam_FH, lam_FH, lam_FH_ref, lam_FH_ref)


for lam in (1.0, 1.064, 1.17, 1.31, 1.55, 2.0):
    print(f"lambda_FH = {lam * 1e3:6.0f} nm -> chi2_33 = {chi2_shg(lam) * 1e12:6.2f} pm/V, d_33 = {chi2_shg(lam)/2 * 1e12:6.2f} pm/V "
          f"(x{miller_scale(lam, lam, 1.55, 1.55):.3f} vs. 1550 nm)")

#%% [markdown]
# #### Load the mode solver data

#%% Load the MPBX run: mode table (all wavelengths, no fields) and the field-file reader
def load_mode_table(gdir):
    """Per-wavelength mode arrays of one geometry from <name>_modes.h5 (no fields), and its meta dict.

    Arrays are (n_lam,) or (n_lam, n_modes_max), padded with NaN / -1 / "" where a wavelength has fewer
    modes. At each wavelength the modes are sorted by descending n_eff over both parity sectors ("mode index"
    = that rank; NOT the Tidy3D runs' mode_index, which is ordered differently). tracks[target] holds the
    merge-time Hermite-Gaussian tracks (mode = -1: absent).
    """
    name = os.path.basename(os.path.normpath(gdir))
    dec = lambda v: v.astype(str) if v.dtype.kind == "S" else v          # noqa: E731
    with h5py.File(os.path.join(gdir, f"{name}_modes.h5"), "r") as f:
        meta = json.loads(f.attrs["meta"])
        T = {k: dec(f[k][()]) for k in ("lam", "chunk", "n_modes", "neff", "ng", "te_frac", "ln_frac", "edge_frac",
                                         "guided", "label", "parity", "flags", "n_e", "n_o", "n_cutoff")}
        T["tracks"] = {t: {k: dec(f["tracks"][t][k][()]) for k in f["tracks"][t]} for t in f["tracks"]}
    return T, meta


def chunk_grid(chunk):
    """The grid dict (Nx, Ny, dx, dy, Lx, Ly) of a chunk of the loaded run."""
    return next(c["grid"] for c in meta["chunks"] if c["name"] == chunk)


def film_center_y(y_px, chunk):
    """Mid-film height [um] of the LN in a chunk's solve.

    run_geometry.py solved each chunk with geometry_gen.tfln_rib.align_to_grid(geom, grid, frac=align_frac):
    the film (centered on y = 0 before alignment) is shifted so its bottom interface sits `frac` of a pixel above
    a pixel center. Same arithmetic here (numpy only, no mpbjax import): y_bot = y[0] + (j + frac) dy.
    """
    g = chunk_grid(chunk)
    dy = g["Ly"] / g["Ny"]
    y0 = geom.get("y_bot") if geom.get("y_bot") is not None else -0.5 * t_ln_um
    j = round((y0 - y_px[0]) / dy - meta["align_frac"])
    return float(y_px[0] + (j + meta["align_frac"]) * dy) + 0.5 * t_ln_um


def read_fields(lam_um, modes=None):
    """E-fields at the solved wavelength lam_um (must be on the solve grid) from the _fields_*.h5 files.

    modes: None -> every mode solved at that wavelength; an int or a list of mode indices otherwise.
    Returns a dict: E (n, 3, Nx, Ny) complex128 = (Ex, Ey, Ez) [mpbx units, int Re(E* x H)_z dA = 1],
    x, y [um] (pixel centers of the chunk's window), yc = y - mid-film height [um], lam, chunk, modes,
    neff, te_frac (n,).
    """
    i = int(np.argmin(np.abs(T["lam"] - lam_um)))
    assert abs(T["lam"][i] - lam_um) < 1e-6, f"lambda = {lam_um} um was not solved"
    modes = np.arange(T["n_modes"][i]) if modes is None else np.atleast_1d(modes).astype(int)
    for fn in meta["field_files"]:
        with h5py.File(os.path.join(GDIR, fn), "r") as f:
            if T["chunk"][i] not in f:
                continue
            g = f[T["chunk"][i]]
            k = int(np.argmin(np.abs(g["lam"][()] - lam_um)))
            assert abs(g["lam"][k] - lam_um) < 1e-6
            E = np.stack([g["E"][k, m] for m in modes]).astype(np.complex128)
            x, y = g["x"][()], g["y"][()]
            neff, te = g["neff"][k][modes], g["te_frac"][k][modes]
            # the field file must list the modes in the same order as the mode table: its neff is a copy of the
            # table, so check E itself, through the TE fraction computed from the field just read
            assert np.allclose(neff, T["neff"][i, modes], rtol=0, atol=1e-10), "field file / mode table mismatch"
            ex2, ey2 = ((np.abs(E[:, c]) ** 2).sum(axis=(1, 2)) for c in (0, 1))
            assert np.allclose(ex2 / (ex2 + ey2), T["te_frac"][i, modes], rtol=0, atol=1e-5), \
                "stored E is not in the mode-table order"
            return dict(E=E, x=x, y=y, yc=y - film_center_y(y, T["chunk"][i]), lam=float(T["lam"][i]),
                        chunk=T["chunk"][i], modes=modes, neff=neff, te_frac=te)
    raise FileNotFoundError(f"no field file holds chunk {T['chunk'][i]} (was the run made with --fields all?)")


RUN_DIR = MODE_RUNS[dev["width_nm"]]
geoms = rio.list_geometries(RUN_DIR)
assert len(geoms) == 1, f"expected one geometry in {RUN_DIR}, found {len(geoms)}"
GDIR = geoms["folder"].iloc[0]
T, meta = load_mode_table(GDIR)
assert meta["complete"], f"{GDIR}: the run is not complete"
assert meta["field_files"] and meta["fields"] == "all", f"{GDIR}: no stored E-fields"

geom = meta["geometry"]                               # [um, deg]
t_ln_um, etch_um, w_top_um = geom["t_ln"], geom["etch_depth"], geom["w_top"]
sidewall_angle_deg, t_clad_um = geom["sidewall_deg"], geom["t_clad"]
t_slab_um = t_ln_um - etch_um
w_bot_um = w_top_um + 2 * etch_um / np.tan(np.radians(sidewall_angle_deg))   # ridge base width [um]
assert w_top_um * 1e3 == dev["width_nm"], f"{DEVICE} is {dev['width_nm']} nm wide, but {GDIR} has w_top = {w_top_um} um"
# The notebook's geometry variables [nm]
top_width, t_LN, t_slab = w_top_um * 1e3, t_ln_um * 1e3, t_slab_um * 1e3
print(f"{os.path.basename(GDIR)}: t_LN = {t_LN:.0f} nm, etch {etch_um * 1e3:.0f} nm (slab {t_slab:.0f} nm), "
      f"w_top = {top_width:.0f} nm, w_base = {w_bot_um * 1e3:.1f} nm, sidewall {sidewall_angle_deg:g} deg, "
      f"SiO2 cladding {t_clad_um * 1e3:.0f} nm above the ridge (air above), align_frac = {meta['align_frac']:.4f}")
for c in meta["chunks"]:
    g = c["grid"]
    print(f"  chunk {c['name']}: {g['Lx']:.3f} x {g['Ly']:.3f} um, dx = {g['dx'] * 1e3:.1f} nm, dy = {g['dy'] * 1e3:.2f} nm")

# The Miller-rule chi1 uses the notebook's Sellmeier: it must be the solver's LN model
assert np.allclose(np.sqrt(ne_sq(T["lam"])), T["n_e"], rtol=0, atol=1e-9), "Sellmeier n_e differs from the solver's"
assert np.allclose(np.sqrt(no_sq(T["lam"])), T["n_o"], rtol=0, atol=1e-9), "Sellmeier n_o differs from the solver's"

##### Pair every FH wavelength with its second harmonic (lambda_SH = lambda_FH / 2 exactly) #####
# On the 10 nm solve grid that's lambda_FH = 800, 820, ..., 2000 nm  <->  lambda_SH = 400, 410, ..., 1000 nm.
FH_RANGE_UM = (0.0, np.inf)                           # [um] optional window on lambda_FH, e.g. (1.50, 1.65)
lam_all = T["lam"]
pairs = [(i, int(np.flatnonzero(np.isclose(lam_all, l / 2, rtol=0, atol=1e-6))[0])) for i, l in enumerate(lam_all)
         if np.isclose(lam_all, l / 2, rtol=0, atol=1e-6).any() and FH_RANGE_UM[0] <= l <= FH_RANGE_UM[1]]
assert pairs, "No FH wavelength has its second harmonic in the solved wavelength grid"
i_FH_all, i_SH_all = (np.array(v) for v in zip(*pairs))   # indices into the mode table, increasing wavelength
print(f"{len(pairs)} FH/SH pairs: lambda_FH = {lam_all[i_FH_all].min() * 1e3:.0f}-{lam_all[i_FH_all].max() * 1e3:.0f} nm, "
      f"lambda_SH = {lam_all[i_SH_all].min() * 1e3:.0f}-{lam_all[i_SH_all].max() * 1e3:.0f} nm")

te00_track = T["tracks"]["TE00"]                      # merge-time TE00 track (HG label + guided + purest TE)

#%% [markdown]
# #### Plot all the modes at a user-supplied wavelength

#%% E-fields of the modes solved at one wavelength
target_wav = 0.750                                    # [um], snapped to the nearest solved wavelength
N_MODES_FIELD_PLOT = 6                                # plot at most this many modes (highest n_eff first)

i_t = int(np.argmin(np.abs(lam_all - target_wav)))
fd = read_fields(lam_all[i_t], modes=np.arange(min(N_MODES_FIELD_PLOT, T["n_modes"][i_t])))
x, y = fd["x"], fd["y"]
y_LN_bot = film_center_y(y, fd["chunk"]) - 0.5 * t_ln_um
y_LN_top = y_LN_bot + t_ln_um
print(f"Plotting mode fields at wavelength {fd['lam']:.3f} um (chunk {fd['chunk']}, TE00 = mode "
      f"{te00_track['mode'][i_t]}); LN film y = {y_LN_bot:+.4f} .. {y_LN_top:+.4f} um")

# LN trapezoid outline in (x, y) plot coordinates
ln_outline_xy = [(-w_bot_um / 2, y_LN_bot), (w_bot_um / 2, y_LN_bot), (w_top_um / 2, y_LN_top), (-w_top_um / 2, y_LN_top)]
x_lo, x_hi, y_lo, y_hi = x.min(), x.max(), y.min(), y.max()

num_modes = len(fd["modes"])
fig, axes = plt.subplots(num_modes, 4, figsize=(20, 3.2 * num_modes), constrained_layout=True, squeeze=False)
for fig_r, mode_idx in enumerate(fd["modes"]):
    ex, ey, ez = fd["E"][fig_r]
    ex_int, ey_int, ez_int = (np.sum(np.abs(c) ** 2) for c in (ex, ey, ez))
    total_intensity = np.abs(ex) ** 2 + np.abs(ey) ** 2 + np.abs(ez) ** 2
    tot = ex_int + ey_int + ez_int
    te_fraction = ex_int / tot if tot > 0 else 0          # TE = horizontal (crystal Z) = mpbx Ex
    tm_fraction = ey_int / tot if tot > 0 else 0          # TM = vertical (crystal X) = mpbx Ey
    print(f"Plotting mode idx {mode_idx}: {T['label'][i_t, mode_idx]:4s} TE (x-pol) = {te_fraction:.3f}, "
          f"TM (y-pol) = {tm_fraction:.3f}, n_eff = {fd['neff'][fig_r]:.5f}, guided = {bool(T['guided'][i_t, mode_idx])}")
    I_norm = total_intensity / np.max(total_intensity)
    panels = [(axes[fig_r, 0], np.real(ex), "Real Ex (TE)", None, None),
              (axes[fig_r, 1], np.real(ey), "Real Ey (TM)", None, None),
              (axes[fig_r, 2], I_norm, "|E|^2", 0.0, 1.0),
              (axes[fig_r, 3], 10 * np.log10(np.maximum(I_norm, 1e-12)), "|E|^2 [dB]", -40, 0)]   # avoid log(0)
    for ax, field, title, vmin, vmax in panels:
        if vmin is None:                                  # symmetric color scale for the real parts
            vmax = np.max(np.abs(field))
            vmin = -vmax
        im = ax.pcolormesh(x, y, field.T, shading="auto", cmap="RdBu_r" if "Real" in title else "inferno",
                           vmin=vmin, vmax=vmax, rasterized=True)
        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(y_lo, y_hi)
        ax.set_xticks(np.arange(np.ceil(x_lo), x_hi + 1e-9, 1.0))
        ax.set_yticks(np.arange(np.ceil(y_lo * 2) / 2, y_hi + 1e-9, 0.5))
        ax.set_title(f"Mode idx {mode_idx:d} ({T['label'][i_t, mode_idx]}) - {title}")
        ax.set_xlabel("x (um)")
        ax.set_ylabel("y (um)")
        ax.set_aspect("equal")
        if "|E|" in title:
            ax.add_patch(Polygon(ln_outline_xy, closed=True, fill=False, edgecolor="white", linewidth=0.5))
            if 0 < t_clad_um < np.inf:                    # SiO2 / air surface of the planar cladding
                ax.axhline(y_LN_top + t_clad_um, color="white", lw=0.5, ls="--")
        fig.colorbar(im, ax=ax, shrink=0.9)
show(fig)

#%% [markdown]
# #### Plot the waveguide dispersion for all modes, at the FH and SH - track the TE00 index

#%% n_eff vs. wavelength for every solved mode, FH and SH, with the TE00 (x-pol) track overlaid
# Style: Nature-family figure conventions, sized up for readability (same as the plot_facet*_summary.py scripts).
NATURE_RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "mathtext.default": "regular",
    "axes.labelsize": 22,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    "legend.fontsize": 17,
    "axes.linewidth": 1.2,
    "axes.edgecolor": "black",
    "axes.spines.top": True,
    "axes.spines.right": True,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.major.size": 7,
    "ytick.major.size": 7,
    "xtick.minor.size": 3.5,
    "ytick.minor.size": 3.5,
    "xtick.major.width": 1.2,
    "ytick.major.width": 1.2,
    "xtick.minor.width": 0.9,
    "ytick.minor.width": 0.9,
    "xtick.major.pad": 7,
    "ytick.major.pad": 7,
    "legend.frameon": True,
    "legend.framealpha": 1.0,
    "legend.edgecolor": "0.6",
    "legend.fancybox": False,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "svg.fonttype": "none",                 # keep text editable in the SVG
}

# Okabe-Ito colorblind-safe hues (+ gray for an 8th mode) and a distinct marker per mode index,
# so identity never relies on color alone. Black is reserved for the TE00 overlay.
MODE_COLORS = ["#D55E00", "#0072B2", "#009E73", "#E69F00", "#CC79A7", "#56B4E9", "#F0E442", "#999999"]
MODE_MARKERS = ["o", "s", "D", "^", "v", "<", ">", "p"]
MARKER_SIZE = 8
MARKER_EDGE = dict(markeredgecolor="black", markeredgewidth=0.9)
LINE_W = 2.0
TE00_STYLE = dict(color="black", lw=2.5, ls=(0, (4, 2.5)))  # dashed, drawn on top of every mode
N_MODES_DISP_PLOT = 6                                 # mode indices 0 .. N-1 (the Tidy3D run solved 6 modes)


def neff_table(idx):
    """(wavelength [nm] x mode_index) tables of n_eff and TE (x-pol) fraction at the mode-table rows idx.
    mpbx te_frac = sum|Ex|^2 / (sum|Ex|^2 + sum|Ey|^2): the notebook's z-pol fraction (1 - Tidy3D 'TE (Ex) fraction')."""
    wl = pd.Index(lam_all[idx] * 1e3, name="wavelength_nm")
    cols = pd.Index(range(N_MODES_DISP_PLOT), name="mode_index")
    neff = pd.DataFrame(T["neff"][idx, :N_MODES_DISP_PLOT], index=wl, columns=cols)
    zpol = pd.DataFrame(T["te_frac"][idx, :N_MODES_DISP_PLOT], index=wl, columns=cols)
    return neff, zpol


def te00_series(idx):
    """Stored TE00 track at the mode-table rows idx: (n_eff, mode index) Series indexed by wavelength [nm]."""
    wl = pd.Index(lam_all[idx] * 1e3, name="wavelength_nm")
    m = te00_track["mode"][idx]
    return (pd.Series(np.where(m >= 0, te00_track["neff"][idx], np.nan), index=wl, name="TE00 n_eff"),
            pd.Series(np.where(m >= 0, m, np.nan), index=wl, name="TE00 mode_index"))


def nice_step(span, target_ticks=6):
    """Round tick spacing (1, 2 or 5 x 10^k) that puts about target_ticks major ticks across span."""
    raw = span / target_ticks
    mag = 10 ** np.floor(np.log10(raw))
    return min((s * mag for s in (1, 2, 5, 10)), key=lambda s: abs(s - raw))


def plot_neff_vs_wavelength(idx, title, x_major=None, x_minor=None, y_major=0.05, y_minor=0.025):
    """n_eff of mode indices 0 .. N_MODES_DISP_PLOT-1 at the mode-table rows idx, with the TE00 track on top.
    x_major / x_minor [nm]: tick spacing; if None, chosen from the wavelength span (~6 labelled ticks)."""
    neff, _ = neff_table(idx)
    te00_neff, te00_idx = te00_series(idx)
    x = neff.index.to_numpy()

    fig, ax = plt.subplots(figsize=(8, 6.4))
    handles = []
    for m in neff.columns:
        c, mk = MODE_COLORS[m % len(MODE_COLORS)], MODE_MARKERS[m % len(MODE_MARKERS)]
        ax.plot(x, neff[m], color=c, lw=LINE_W, zorder=3)
        ax.plot(x, neff[m], ls="none", marker=mk, ms=MARKER_SIZE, markerfacecolor=c, zorder=4, clip_on=False, **MARKER_EDGE)
        handles.append(Line2D([], [], color=c, lw=LINE_W, marker=mk, ms=MARKER_SIZE * 0.85, markerfacecolor=c,
                              label=f"Mode {m}", **MARKER_EDGE))
    ax.plot(te00_neff.index.to_numpy(), te00_neff.to_numpy(), zorder=5, **TE00_STYLE)
    handles.append(Line2D([], [], label="TE$_{00}$ (x-pol.)", **TE00_STYLE))

    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Effective index, $n_{eff}$")
    ax.set_title(title, fontsize=22, pad=12)
    if x_major is None:
        x_major = nice_step(x.max() - x.min())
    if x_minor is None:
        x_minor = x_major / 2
    x_pad = 0.5 * np.median(np.diff(x)) if x.size > 1 else 0.5 * x_minor
    ax.set_xlim(x.min() - x_pad, x.max() + x_pad)
    ax.xaxis.set_major_locator(MultipleLocator(x_major))
    ax.xaxis.set_minor_locator(MultipleLocator(x_minor))
    vals = np.concatenate([neff.values.ravel(), te00_neff.values])
    ax.set_ylim(np.floor(np.nanmin(vals) / y_minor) * y_minor, np.ceil(np.nanmax(vals) / y_minor) * y_minor)
    ax.yaxis.set_major_locator(MultipleLocator(y_major))
    ax.yaxis.set_minor_locator(MultipleLocator(y_minor))
    ax.grid(True, which="major", color="0.88", lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0,
              handletextpad=0.4, borderpad=0.5)
    # Which mode index the TE00 track picks at each wavelength (it changes across mode crossings)
    print(", ".join(f"{w:.0f} nm: {int(i) if np.isfinite(i) else 'none'}" for w, i in te00_idx.items()))
    return fig, ax


with mpl.rc_context(NATURE_RC):
    # Ticks auto-scale to the span: 800-2000 nm -> every 200 nm, 400-1000 nm -> every 100 nm.
    fig_FH, ax_FH = plot_neff_vs_wavelength(i_FH_all, "FH")
    fig_SH, ax_SH = plot_neff_vs_wavelength(i_SH_all, "SH")
    show()

#%% [markdown]
# #### NL coefficient and poling period calculation

#%% FH window, TE00 indices (checked against the notebook's HG00-overlap tracking), Delta k, poling period
WAV_FH_SEL = dev["wav_fh_sel"]

lam_FH_all = lam_all[i_FH_all]                        # [um], increasing
if WAV_FH_SEL is None:
    keep = np.ones(lam_FH_all.size, dtype=bool)
elif np.isscalar(WAV_FH_SEL):
    keep = np.zeros(lam_FH_all.size, dtype=bool)
    keep[np.argmin(np.abs(lam_FH_all - WAV_FH_SEL))] = True
else:
    lo, hi = sorted(WAV_FH_SEL)
    keep = (lam_FH_all >= lo - 1e-9) & (lam_FH_all <= hi + 1e-9)
assert keep.any(), f"No solved FH wavelength matches WAV_FH_SEL = {WAV_FH_SEL}"
i_FH, i_SH = i_FH_all[keep], i_SH_all[keep]           # mode-table rows of the selected pairs, increasing wavelength
lam_FH, lam_SH = lam_all[i_FH], lam_all[i_SH]         # [um]
assert np.allclose(lam_SH, lam_FH / 2, rtol=1e-9), "FH and SH wavelength grids are not paired as lambda_SH = lambda_FH / 2"
print(f"Selected {keep.sum()} FH wavelength(s): " + ", ".join(f"{l * 1e3:.0f}" for l in lam_FH) + " nm")


def te00_index(rows, forced):
    """TE00 mode index at each mode-table row: the forced int, else the stored TE00 track."""
    if forced is not None:
        idx = np.full(len(rows), int(forced))
        te = T["te_frac"][rows, idx]
        assert np.all(te > 0.5), f"forced TE00 index {forced} is not TE-polarized (te_frac {np.round(te, 3)}) at " \
                                 f"{lam_all[rows]} um: MPBX indices are n_eff ranks, see TE00_IDX_FH"
        return idx
    idx = te00_track["mode"][rows]
    assert np.all(idx >= 0), f"no TE00 in the stored track at {lam_all[rows][idx < 0]} um: set TE00_IDX_* by hand"
    return idx


TE00_idx_FH = te00_index(i_FH, TE00_IDX_FH)
TE00_idx_SH = te00_index(i_SH, TE00_IDX_SH)


##### Hermite-Gaussian TE00 tracking (port of Tidy3D_sims/hg_mode_idx_tracking.py), on the selected wavelengths #####
def hg00_overlap(field, x, y):
    """Normalized overlap |<E|HG00>|^2 / (<E|E><HG00|HG00>) of a complex 2D field E(x, y) with a
    Hermite-Gaussian HG00 (i.e. a Gaussian) matched to the field's own intensity profile.

    The Gaussian center is the |E|^2 centroid, and its waists come from the |E|^2 second moments
    (for a Gaussian amplitude exp(-r^2/w^2), the intensity standard deviation is w/2, so w = 2*sigma).
    ~1 for a single-lobed fundamental mode, ~0 for modes with a node through the centroid (e.g. TE10, TE01).
    """
    dA = np.outer(np.gradient(x), np.gradient(y))
    I = np.abs(field) ** 2
    P = np.sum(I * dA)
    X, Y = np.meshgrid(x, y, indexing="ij")
    x0, y0 = np.sum(X * I * dA) / P, np.sum(Y * I * dA) / P
    w_x = 2.0 * np.sqrt(np.sum((X - x0) ** 2 * I * dA) / P)
    w_y = 2.0 * np.sqrt(np.sum((Y - y0) ** 2 * I * dA) / P)
    hg00 = np.exp(-((X - x0) / w_x) ** 2 - ((Y - y0) / w_y) ** 2)
    return np.abs(np.sum(field * hg00 * dA)) ** 2 / (P * np.sum(hg00 ** 2 * dA))


def te00_track_hg(rows, zpol_min=0.5, overlap_min=0.5):
    """TE00 at each mode-table row from BOTH polarization and mode shape (the notebook's te00_track_hg):
    candidates have TE (x-pol) fraction > zpol_min AND TE-component HG00 overlap > overlap_min; the TE00 is the
    candidate with the largest overlap (-1 if none). Returns (te00_idx, overlap table (wavelength x mode))."""
    nm = int(T["n_modes"][rows].max())
    overlap = np.full((len(rows), nm), np.nan)
    for r, i in enumerate(rows):
        for m in range(T["n_modes"][i]):
            fm = read_fields(lam_all[i], modes=m)
            overlap[r, m] = hg00_overlap(fm["E"][0, 0], fm["x"], fm["y"])
    zpol = T["te_frac"][rows, :nm]
    score = np.where((zpol > zpol_min) & (overlap > overlap_min), overlap, -np.inf)
    idx = np.where(np.isfinite(score).any(axis=1), np.argmax(score, axis=1), -1)
    table = pd.DataFrame(overlap, index=pd.Index(lam_all[rows] * 1e3, name="wavelength_nm"),
                         columns=pd.Index(range(nm), name="mode_index"))
    return idx, table


hg_idx_FH, overlap_FH = te00_track_hg(i_FH)
hg_idx_SH, overlap_SH = te00_track_hg(i_SH)
print("HG00 overlap of the TE component, FH:\n" + overlap_FH.round(4).to_string())
print("HG00 overlap of the TE component, SH:\n" + overlap_SH.round(4).to_string())
for name, used, hg, rows in (("FH", TE00_idx_FH, hg_idx_FH, i_FH), ("SH", TE00_idx_SH, hg_idx_SH, i_SH)):
    print(f"TE00 mode index {name}: used {used.tolist()} (HG00 tracking: {hg.tolist()})")
    if np.any(used != hg):
        print(f"WARNING: TE00_idx_{name} = {used.tolist()}, but HG00 tracking picks {hg.tolist()} in this window")
    for i, m in zip(rows, used):                      # flags the merge stored for the chosen modes
        if T["flags"][i, m] or not T["guided"][i, m]:
            print(f"  note {lam_all[i] * 1e3:.0f} nm, mode {m} ({T['label'][i, m]}): guided = {bool(T['guided'][i, m])}"
                  f"{'; ' + T['flags'][i, m] if T['flags'][i, m] else ''}")

# n_eff and n_g of the TE00 mode, increasing wavelength
neff_FH, neff_SH = T["neff"][i_FH, TE00_idx_FH], T["neff"][i_SH, TE00_idx_SH]
ng_FH, ng_SH = T["ng"][i_FH, TE00_idx_FH], T["ng"][i_SH, TE00_idx_SH]
print(f"lam_FH = {lam_FH} um\nlam_SH = {lam_SH} um")

k_FH = 2 * np.pi / lam_FH * neff_FH                   # [rad/um]
k_SH = 2 * np.pi / lam_SH * neff_SH                   # [rad/um]
delta_k = k_SH - 2 * k_FH                             # [rad/um]
poling_period = 2 * np.pi / delta_k                   # [um] first-order QPM period, Lambda = 2 pi / Delta k

for lf, nf, ns, gf, dk, L in zip(lam_FH, neff_FH, neff_SH, ng_FH, delta_k, poling_period):
    print(f"lambda_FH = {lf * 1e3:7.2f} nm | n_eff,FH = {nf:.5f} | n_eff,SH = {ns:.5f} | n_g,FH = {gf:.4f} | "
          f"Delta k = {dk:.5f} rad/um | Lambda = {L:.4f} um")

#%% Poling length, chi2_max, propagation losses
L_POL_CM = 0.4                                        # [cm]
L_POL_M = L_POL_CM * 1e-2                             # [m]

# NOTE - the factor of two for chi2_max comes from Boyd convention / description of the nonlinear susceptibility + contracted notation
# chi2_max = chi2_shg(lam_FH, chi2_ref=2.0*24.0e-12, lam_FH_ref=1.55) * (2 / np.pi) * np.sin(np.pi * dc)
# chi2_max = chi2_shg(lam_FH, chi2_ref=2.0*25.2e-12, lam_FH_ref=1.064)  * (2 / np.pi) * np.sin(np.pi * dc)
# F_inhomo = 0.441    # add it in here (np.sqrt(F_inhomo)), just for the alpha_1, alpha_2 contour plots
dc = 0.50                                             # poling duty cycle - ideal
chi2_max = chi2_shg(lam_FH, chi2_ref=2.0 * 25.2e-12, lam_FH_ref=1.064) * (2 / np.pi) * 0.95 * 0.96 * 0.98  # * np.sqrt(F_inhomo)

alpha_1 = alpha_dB_per_cm_to_per_m(2.0)               # FH power loss [1/m]
alpha_2 = alpha_dB_per_cm_to_per_m(6.0)               # SH power loss [1/m]

print("d_33_max (pm/V):", np.round(chi2_max / 2 * 1e12, 8))
print(f"alpha_1: {alpha_1:.1f} 1/m, alpha_2: {alpha_2:.1f} 1/m")


def L_eff_lossy(L, alpha_1, alpha_2):
    """Phase-matched effective length [m], eq. (29): (e^{-a1 L} - e^{-a2 L/2}) / (a2/2 - a1). Vectorized.

    Same as the cosh form, but stays finite on the alpha_2 = 2 alpha_1 line, where the cosh form is 0/0 -> NaN
    (that line runs straight through the alpha map). Limit there: L e^{-alpha_1 L}.
    """
    alpha_1, alpha_2 = np.asarray(alpha_1, float), np.asarray(alpha_2, float)
    del_alpha = alpha_2 / 2.0 - alpha_1
    x = del_alpha * L
    small = np.abs(x) < 1e-12
    safe = np.where(small, 1.0, del_alpha)            # avoid 0/0 where the limit is used instead
    return np.where(small, L, -np.expm1(-x) / safe) * np.exp(-alpha_1 * L)

#%% TE00 mode areas from the true modal power, LN overlap, kappa, eta
# Mode area (reference index n_i = n_eff):  A_i = S_i / (n_eff,i max|E_TE,i|^2)  in mpbx natural units
# (= Z0 S_i / (n_eff,i max|E_TE,i|^2) in SI), with S_i = int Re(E* x H)_z dA the real Poynting flux.
# mpbx stores E with S = 1 exactly (its own H, spectral curl). S is also rebuilt from E alone here, as in the
# notebook, as a check of the stored normalization and of the frame / phase conventions.
# The overlap integral needs the FH and SH fields on one grid: the field on the finer grid stays on its own
# pixels (cropped to the LN), the other one is interpolated (linear) onto them, both in film-centered y.

def poynting_flux_from_E(E, x_um, y_um, lam_um, n_eff):
    """Modal power int Re(E* x H)_z dA from E alone, in mpbx natural units (eps0 = mu0 = c = 1, lengths in um).

    H = curl(E) / (i w) with d/dz -> i beta (fields ~ exp(i(beta z - w t)), as in mpbx: H = (k+G) x E / w), so
        S = (1 / w) int [ beta (|Ex|^2 + |Ey|^2) - Im(Ex* dEz/dx + Ey* dEz/dy) ] dA,   w = 2 pi / lam, beta = w n_eff.
    The notebook's function with its frame mapped (Tidy3D Ez, Ex, Ey -> mpbx Ex, Ey, Ez). Ez is tangential to every
    interface, so its finite-difference gradient is well behaved.
    """
    Ex, Ey, Ez = E
    dA = np.outer(np.gradient(x_um), np.gradient(y_um))                     # [um^2]
    w = 2 * np.pi / lam_um                                                    # [rad/um]
    beta = w * n_eff                                                          # [rad/um]
    dEz_dx = np.gradient(Ez, x_um, axis=0)
    dEz_dy = np.gradient(Ez, y_um, axis=1)
    integrand = beta * (np.abs(Ex) ** 2 + np.abs(Ey) ** 2) - np.imag(np.conj(Ex) * dEz_dx + np.conj(Ey) * dEz_dy)
    return np.sum(integrand * dA) / w


def ln_fill_fraction(x, y, y_bot, y_top, w_bot, w_top, t_slab=0.0, n_sub=8):
    """Fraction of each grid cell (centered on the field sample points) that lies inside LN, shape (Nx, Ny).

    LN = a trapezoidal ridge centered on x = 0 (bottom width w_bot at its base, top width w_top at y_top),
    plus an optional full-width slab of thickness t_slab from y_bot up to the ridge base. All lengths in um.
    Each cell is sampled on an n_sub x n_sub sub-grid, so field points on or near an LN boundary get a
    partial weight instead of an all-or-nothing 0/1. (The notebook's function, with mpbx axes.)
    """
    offsets = (np.arange(n_sub) + 0.5) / n_sub - 0.5
    xs = (x[:, None] + offsets[None, :] * np.gradient(x)[:, None]).ravel()
    ys = (y[:, None] + offsets[None, :] * np.gradient(y)[:, None]).ravel()
    XS, YS = np.meshgrid(xs, ys, indexing="ij")
    y_ridge_bot = y_bot + t_slab
    half_w = 0.5 * (w_bot + (w_top - w_bot) * (YS - y_ridge_bot) / (y_top - y_ridge_bot))   # linear sidewalls
    in_ridge = (YS >= y_ridge_bot) & (YS <= y_top) & (np.abs(XS) <= half_w)
    in_slab = (YS >= y_bot) & (YS < y_ridge_bot)                                           # empty if t_slab = 0
    return (in_ridge | in_slab).reshape(len(x), n_sub, len(y), n_sub).mean(axis=(1, 3))


def finer_grid(fh, sh):
    """"FH" or "SH": whose field grid has the smaller pixel area (ties -> "SH")."""
    px = lambda d: np.diff(d["x"]).mean() * np.diff(d["yc"]).mean()        # noqa: E731
    return "FH" if px(fh) < px(sh) else "SH"


def ln_overlap_integral(fh, sh, ref, margin_px=3):
    """int E_SH conj(E_FH)^2 dA over the LN [um^2] of the peak-normalized TE fields, on one grid.

    fh, sh: dicts with Ehat (Nx, Ny) complex, x, yc [um] (film-centered). ref = "FH" | "SH": whose pixels to
    integrate on (cropped to the LN + margin_px of the coarser pitch); the other field is linearly interpolated
    onto them. Returns (integral, mask area [um^2]).
    """
    assert t_slab_um == 0, "the crop below assumes a fully etched ridge: widen it to the window for a slab"
    a, b = (fh, sh) if ref == "FH" else (sh, fh)
    pad_x = margin_px * max(np.gradient(fh["x"]).max(), np.gradient(sh["x"]).max())
    pad_y = margin_px * max(np.gradient(fh["yc"]).max(), np.gradient(sh["yc"]).max())
    sx = np.abs(a["x"]) <= 0.5 * max(w_bot_um, w_top_um) + pad_x
    sy = (a["yc"] >= -0.5 * t_ln_um - pad_y) & (a["yc"] <= 0.5 * t_ln_um + pad_y)
    x, yc = a["x"][sx], a["yc"][sy]
    Ea = a["Ehat"][np.ix_(sx, sy)]
    Eb = RegularGridInterpolator((b["x"], b["yc"]), b["Ehat"], method="linear", bounds_error=True)(
        np.stack(np.meshgrid(x, yc, indexing="ij"), axis=-1))
    E_FH_c, E_SH_c = (Ea, Eb) if ref == "FH" else (Eb, Ea)
    mask = ln_fill_fraction(x, yc, -0.5 * t_ln_um, 0.5 * t_ln_um, w_bot_um, w_top_um, t_slab=t_slab_um)
    dA = np.outer(np.gradient(x), np.gradient(yc))                          # [um^2]
    return np.sum(E_SH_c * np.conj(E_FH_c) ** 2 * mask * dA), np.sum(mask * dA)


# TE00 fields of every selected pair (FH and SH usually come from different chunks / grids)
n_eff_FH, n_eff_SH = neff_FH, neff_SH
freqs_FH, freqs_SH = C_UM / lam_FH, C_UM / lam_SH    # [Hz]
FH = [read_fields(l, modes=m) for l, m in zip(lam_FH, TE00_idx_FH)]
SH = [read_fields(l, modes=m) for l, m in zip(lam_SH, TE00_idx_SH)]

S_FH, S_SH = np.zeros(lam_FH.size), np.zeros(lam_FH.size)                  # rebuilt modal power (mpbx units)
E2max_FH, E2max_SH = np.zeros(lam_FH.size), np.zeros(lam_FH.size)          # max |E_TE|^2 (mpbx units)
A_old_FH, A_old_SH = np.zeros(lam_FH.size), np.zeros(lam_FH.size)          # Ez-only (here Ex-only) areas [um^2]
I_LN, I_LN_alt, A_LN_mask = (np.zeros(lam_FH.size, complex), np.zeros(lam_FH.size, complex), np.zeros(lam_FH.size))
for i, (fh, sh) in enumerate(zip(FH, SH)):
    for d, lam_i, n_i, S, E2, A_old in ((fh, lam_FH[i], n_eff_FH[i], S_FH, E2max_FH, A_old_FH),
                                        (sh, lam_SH[i], n_eff_SH[i], S_SH, E2max_SH, A_old_SH)):
        E = d["E"][0]
        assert np.isclose(d["neff"][0], n_i)
        S[i] = poynting_flux_from_E(E, d["x"], d["y"], lam_i, n_i)
        E2[i] = np.max(np.abs(E[0]) ** 2)
        A_old[i] = np.sum(np.abs(E[0]) ** 2 * np.outer(np.gradient(d["x"]), np.gradient(d["y"]))) / E2[i]
        d["Ehat"] = E[0] / np.sqrt(E2[i])                                   # TE component, unitless, peak |E_TE| = 1
    ref = finer_grid(fh, sh)                                                # integrate on the finer pixels
    I_LN[i], A_LN_mask[i] = ln_overlap_integral(fh, sh, ref=ref)
    I_LN_alt[i], _ = ln_overlap_integral(fh, sh, ref="SH" if ref == "FH" else "FH")   # check: the coarser ones

assert np.allclose(S_FH, 1.0, atol=5e-3) and np.allclose(S_SH, 1.0, atol=5e-3), \
    f"power rebuilt from E is not the stored normalization (1): FH {S_FH}, SH {S_SH}"
S_FH_used, S_SH_used = np.ones(lam_FH.size), np.ones(lam_FH.size)          # the solver's exact normalization

A_eff_FH = S_FH_used / (n_eff_FH * E2max_FH) * UM**2                       # [m^2]
A_eff_SH = S_SH_used / (n_eff_SH * E2max_SH) * UM**2                       # [m^2]
for lf, cf, cs, af, as_, rf, rs, sf, ss in zip(lam_FH, [d["chunk"] for d in FH], [d["chunk"] for d in SH],
                                               A_eff_FH, A_eff_SH, A_eff_FH / (A_old_FH * UM**2),
                                               A_eff_SH / (A_old_SH * UM**2), S_FH, S_SH):
    print(f"lambda_FH = {lf * 1e3:7.2f} nm | A_eff,FH = {af / UM**2:.4f} um^2 (x{rf:.4f} vs Ex-only) | "
          f"A_eff,SH = {as_ / UM**2:.4f} um^2 (x{rs:.4f}) | S rebuilt from E: FH {sf:.5f}, SH {ss:.5f} | "
          f"grids {cf} / {cs}")
print("---------------------------------------------------------------------------------------------------------------")

##### LN mask, overlap, kappa, eta #####
A_LN_exact = 0.5 * (w_top_um + w_bot_um) * t_ln_um                         # trapezoid (t_slab = 0)
print(f"LN area: mask = {A_LN_mask.mean():.4f} um^2, exact = {A_LN_exact:.4f} um^2")
assert np.allclose(A_LN_mask, A_LN_exact, rtol=1e-2), "LN mask doesn't match the trapezoid geometry"

A_bar = (A_eff_FH**2 * A_eff_SH) ** (1 / 3)                                # [m^2]
overlap = I_LN * UM**2 / A_bar                                              # unitless
kappa = np.sqrt(Z_0 / (2 * C_SI**2 * n_eff_SH * n_eff_FH**2 * A_bar)) * chi2_max * overlap

omega_FH = 2 * np.pi * freqs_FH                                             # [rad/s]
eta_SI = np.abs(omega_FH * kappa) ** 2                                      # [1/(W m^2)]
eta_pct_W_cm2 = eta_SI * 1e-4 * 100                                         # [%/(W cm^2)]
eta_lossy_pct_W_cm2 = eta_pct_W_cm2 * (L_eff_lossy(L_POL_M, alpha_1, alpha_2) / L_POL_M) ** 2   # phase matched, input-referenced
eta_grid_spread = np.abs(I_LN_alt) ** 2 / np.abs(I_LN) ** 2 - 1            # eta change if integrated on the other grid

for lf, ab, ov, eta, etl, gs in zip(lam_FH, A_bar, overlap, eta_pct_W_cm2, eta_lossy_pct_W_cm2, eta_grid_spread):
    print(f"lambda_FH = {lf * 1e3:7.2f} nm | A_bar = {ab / UM**2:.4f} um^2 | |overlap| = {np.abs(ov):.4f} | "
          f"eta = {eta:7.1f} %/W/cm^2 | eta_lossy = {etl:7.1f} %/W/cm^2 | other-grid integration: {gs * 100:+.2f}% eta")

#%% [markdown]
# #### Plot $\eta(\alpha_1, \alpha_2; \Delta k = 0, L_{pol}=4mm, \lambda_{FH} \rightarrow \lambda_{SH})$

#%% eta_lossy vs. (alpha_1, alpha_2) map: viridis fill + labelled constant-eta contours
LAM_PLOT_UM = lam_FH[min(1, lam_FH.size - 1)]                  # [um] e.g. 1.55; nearest selected wavelength is used
i_lam = np.argmin(np.abs(lam_FH - LAM_PLOT_UM))
eta_0_pct = eta_SI[i_lam] * 1e-4 * 100                         # [%/(W cm^2)] lossless

# alpha grid [dB/cm] -> [1/m]
a1_dB = np.linspace(0.1, 10.0, 400)                            # x axis
a2_dB = np.linspace(0.1, 20.0, 400)                            # y axis
A1_dB, A2_dB = np.meshgrid(a1_dB, a2_dB)                       # (N_a2, N_a1), rows = alpha_2
eta_map = eta_0_pct * (L_eff_lossy(L_POL_M, alpha_dB_per_cm_to_per_m(A1_dB),
                                   alpha_dB_per_cm_to_per_m(A2_dB)) / L_POL_M) ** 2   # [%/(W cm^2)]

# Contour levels: fixed round values, keeping only those inside the map's range
levels = np.array([0, 500, 1000, 1500, 2000])                 # [%/(W cm^2)]
levels = levels[(levels > eta_map.min()) & (levels < eta_map.max())]

# Fixed color range: values outside it saturate, flagged by the colorbar's end arrows
ETA_CLIM = (0, 2500)                                           # [%/(W cm^2)]
extend = {(False, False): "neither", (True, False): "min",
          (False, True): "max", (True, True): "both"}[(eta_map.min() < ETA_CLIM[0], eta_map.max() > ETA_CLIM[1])]

# Device label: offset from the star [dB/cm], left-aligned and vertically centered on that point
ANNOT_OFFSET = (1.2, 0.3)
ETA_HIGHLIGHT = 800     # [%/(W cm^2)] drawn as a dashed white contour, e.g. a measured efficiency


def centered_label_positions(cs, xlim, ylim, avoid=(), min_dist=0.10):
    """One clabel position per contour level: the point halfway along that level's longest segment.

    Arc length is measured in axes-fraction coordinates, so "halfway" matches what you see even though the
    x and y spans differ (10 vs. 20 dB/cm). If the midpoint is within min_dist (axes fraction) of any point in
    avoid, the label slides along the contour to the nearest vertex that is clear.
    """
    lo = np.array([xlim[0], ylim[0]])
    span = np.array([xlim[1] - xlim[0], ylim[1] - ylim[0]])
    avoid_u = [(np.asarray(p, float) - lo) / span for p in avoid]
    positions = []
    for segs in cs.allsegs:
        if not segs:
            continue
        seg = max(segs, key=len)                                            # longest piece of this level
        u = (seg - lo) / span                                               # axes-fraction coordinates
        s = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(u, axis=0).T))])
        order = np.argsort(np.abs(s - s[-1] / 2))                           # vertices, from the middle outward
        clear = [i for i in order if all(np.hypot(*(u[i] - a)) >= min_dist for a in avoid_u)]
        positions.append(tuple(seg[clear[0] if clear else order[0]]))
    return positions


with mpl.rc_context(NATURE_RC):
    fig, ax = plt.subplots(figsize=(8.5, 6.8))
    xlim, ylim = (a1_dB.min(), a1_dB.max()), (a2_dB.min(), a2_dB.max())

    im = ax.pcolormesh(a1_dB, a2_dB, eta_map, cmap="viridis", shading="auto", rasterized=True,
                       vmin=ETA_CLIM[0], vmax=ETA_CLIM[1])
    cbar = fig.colorbar(im, ax=ax, pad=0.03, extend=extend)
    cbar.set_label(r"$\eta_{lossy}$ (% W$^{-1}$ cm$^{-2}$)")
    cbar.set_ticks(np.arange(ETA_CLIM[0], ETA_CLIM[1] + 1, 1000))

    # Current device losses from the parameter cell (needed first, so contour labels can steer around them)
    a1_dev, a2_dev = alpha_per_m_to_dB_per_cm(alpha_1), alpha_per_m_to_dB_per_cm(alpha_2)
    eta_dev = eta_0_pct * (L_eff_lossy(L_POL_M, alpha_1, alpha_2) / L_POL_M) ** 2
    box_xy = (a1_dev + ANNOT_OFFSET[0], a2_dev + ANNOT_OFFSET[1])

    # Constant-eta contours, each label centered along its line (thin black lines, white-haloed labels)
    cs = ax.contour(a1_dB, a2_dB, eta_map, levels=levels, colors="black", linewidths=1.2)
    keep_clear = [(a1_dev, a2_dev), box_xy, ((a1_dev + box_xy[0]) / 2, (a2_dev + box_xy[1]) / 2),
                  (box_xy[0] + 0.5, box_xy[1])]                             # star, leader, label box
    label_pos = centered_label_positions(cs, xlim, ylim, avoid=keep_clear)
    labels = ax.clabel(cs, fmt="%.0f", fontsize=15, inline=True, inline_spacing=6, manual=label_pos)
    for t in labels:
        t.set_path_effects([pe.withStroke(linewidth=3, foreground="white")])

    # Highlighted eta contour: dashed white, so it stands apart from the black constant-eta lines
    if eta_map.min() < ETA_HIGHLIGHT < eta_map.max():
        cs_hl = ax.contour(a1_dB, a2_dB, eta_map, levels=[ETA_HIGHLIGHT], colors="white",
                           linewidths=2.2, linestyles=[(0, (5, 3))], zorder=4)
        # Center its label too, steering clear of the star/box and the neighboring contours' labels
        hl_labels = ax.clabel(cs_hl, fmt="%.0f", fontsize=15, inline=True, inline_spacing=6,
                              manual=centered_label_positions(cs_hl, xlim, ylim, avoid=keep_clear + label_pos))
        for t in hl_labels:
            t.set_color("white")
            t.set_fontweight("bold")
            t.set_path_effects([pe.withStroke(linewidth=3, foreground="black")])
        cbar.ax.axhline(ETA_HIGHLIGHT, color="white", lw=2.2, ls=(0, (5, 3)))   # same marker on the colorbar

    # Star at the current device losses
    ax.plot(a1_dev, a2_dev, marker="*", ms=18, color="white", markeredgecolor="black", markeredgewidth=1.2,
            ls="none", zorder=6)

    ax.set_xlabel(r"FH loss, $\alpha_1$ (dB/cm)")
    ax.set_ylabel(r"SH loss, $\alpha_2$ (dB/cm)")
    ax.set_title(rf"$\lambda_{{FH}}$ = {lam_FH[i_lam] * 1e3:.0f} nm, L = {L_POL_CM:g} cm," + "\n"
                 rf"$\eta_0$ = {eta_0_pct:.0f} % W$^{{-1}}$ cm$^{{-2}}$ (MPBX)", fontsize=18, pad=12)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.xaxis.set_major_locator(MultipleLocator(2))
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.yaxis.set_major_locator(MultipleLocator(4))
    ax.yaxis.set_minor_locator(MultipleLocator(2))
    ax.tick_params(which="both", color="white")        # in-facing ticks stay visible on the dark end of viridis

    if SAVE_FIGS:
        fig.savefig(os.path.join(OUT_DIR, f"Praevium2799_{DEVICE}_MPBX_eta-lossy_vs_alpha_L{L_POL_CM:g}cm_"
                                          f"{lam_FH[i_lam] * 1e3:.0f}nm.svg"))
    show(fig)

#%% [markdown]
# #### Add in inhomogeneous broadening for a given device of interest

#%% Measured CW QPM scans: file list of the device's band
device_of_interest = DEVICE
data_dir = BAND_DIRS[dev["band"]]
for d in (C_BAND_CW_SPECTRA_DIR, O_BAND_CW_SPECTRA_DIR, YB_BAND_CW_SPECTRA_DIR):
    print(f"{os.path.isdir(d)}  {d}")


def parse_fname(data_dir, fname):
    """Metadata from a scan file name like SH-CONST-PP_DEV14_1.8um_05-29-2026-16-26-43_TSL-570_1240-1380nm_23C_..."""
    s_list = fname.removesuffix(".h5").split("_")
    return {
        "DevType": s_list[0],
        "DevNum": s_list[1],
        "WGWidth": s_list[2],
        "datetime": s_list[3],
        "laser": s_list[4],          # might be incorrect, as it wasn't changed for some of the file names
        "wav_range": s_list[5],
        "temp": s_list[6],
        "fpath": os.path.join(data_dir, fname),
    }


# Sorted by name first, so the "first file of the device" below is the same on every OS (Windows lists by name)
fnames = sorted(f for f in os.listdir(data_dir) if f.endswith(".h5"))
fname_metadata = pd.DataFrame([parse_fname(data_dir, f) for f in fnames])
fname_metadata = fname_metadata.sort_values(by=["DevType", "DevNum", "WGWidth"], kind="stable")

title_fs = 20
label_fs = 20
tick_fs = 20
legend_fs = 12

if data_dir == C_BAND_CW_SPECTRA_DIR:
    SIG_COLOR = "maroon"
    xmin_plot, xmax_plot, xstep_plot = 1470.0, 1580.0, 10.0
elif data_dir == O_BAND_CW_SPECTRA_DIR:
    SIG_COLOR = "tab:red"
    xmin_plot, xmax_plot, xstep_plot = 1240.0, 1380.0, 10.0
elif data_dir == YB_BAND_CW_SPECTRA_DIR:
    SIG_COLOR = "tab:green"
    xmin_plot, xmax_plot, xstep_plot = 1050.0, 1180.0, 10.0
else:
    SIG_COLOR = "black"
TAP_COLOR = "mediumorchid"
WAV_COLOR = "black"                                   # trigger converted to wavelength

##### Special processing for the VCSEL #####
VCSEL_PASS = "up"            # which full sweep pass to analyze: "up" or "down" (up/down passes are offset by ~0.6 nm)
VCSEL_GRID_NM = 0.020        # uniform resampling step [nm]; the VCSEL wavelength readout is quantized at ~16 pm


def split_sweep_passes(wl, smooth_pts=201, jump_nm=1.0, min_pts=50):
    """Index arrays of the monotonic passes of a back-and-forth sweep, in time order.
    Breaks where the wavelength jumps by more than jump_nm or the (smoothed) sweep direction flips."""
    wl_s = np.convolve(wl, np.ones(smooth_pts) / smooth_pts, mode="same")
    direction = np.sign(np.gradient(wl_s))
    breaks = np.flatnonzero((np.abs(np.diff(wl)) > jump_nm) | (np.diff(direction) != 0)) + 1
    return [s for s in np.split(np.arange(wl.size), breaks) if s.size >= min_pts]


def resample_uniform(wl, y, lo, hi, dx):
    """Bin-average y onto a uniform grid [lo, hi] with step dx, then linearly interpolate across empty bins.
    Returns (grid, y_on_grid, fraction of bins that held data)."""
    edges = np.arange(lo, hi + dx / 2, dx)
    c = 0.5 * (edges[1:] + edges[:-1])
    idx = np.digitize(wl, edges) - 1
    ok = (idx >= 0) & (idx < c.size)
    n = np.bincount(idx[ok], minlength=c.size)
    yb = np.bincount(idx[ok], weights=y[ok], minlength=c.size) / np.maximum(n, 1)
    filled = n > 0
    return c, np.interp(c, c[filled], yb[filled]), filled.mean()

#%% Load and plot the device's scan (the first matching file, saturated scans skipped)
plt.close("all")
scan_row = None
for idx, row in fname_metadata.iterrows():
    if (row["DevNum"] + "_" + row["WGWidth"]) != device_of_interest or "saturated" in row["fpath"]:
        continue
    if dev["scan_min_date"] is not None and \
            datetime.strptime(row["datetime"], "%m-%d-%Y-%H-%M-%S") < dev["scan_min_date"]:
        continue
    scan_row = row

    loaded_dict = dict()
    with h5py.File(row["fpath"], "r") as f:
        for key in f.keys():
            dataset = f[key]
            loaded_dict[key] = dataset[()] if dataset.ndim == 0 else dataset[:]   # scalar or array

    t = loaded_dict["t"]
    sig_processed_W = loaded_dict["sig_processed_W"]
    tap_processed_W = loaded_dict["tap_processed_W"]
    wl = loaded_dict["trig_processed_nm"]

    if data_dir != YB_BAND_CW_SPECTRA_DIR:
        sort_indices = np.argsort(wl)
        wl_s, sig_s, tap_s = wl[sort_indices], sig_processed_W[sort_indices], tap_processed_W[sort_indices]
        wavs_truncated = np.linspace(np.min(wl_s), np.max(wl_s), len(wl_s))
        tap_to_norm_to = np.interp(wavs_truncated, wl_s, tap_s)
        sig_to_norm = np.interp(wavs_truncated, wl_s, sig_s)
    else:
        # The VCSEL is current-swept back and forth (several passes over the same wavelengths) and its wavelength
        # readout is quantized, so a global sort gives runs of identical wavelengths (median step 0) and interleaves
        # passes whose wavelength calibrations differ. Keep ONE full pass, then put it on a uniform grid.
        passes = split_sweep_passes(wl)
        span = lambda s: wl[s].max() - wl[s].min()                          # noqa: E731
        full = [s for s in passes if span(s) > 0.8 * max(span(p) for p in passes if wl[p].max() > xmax_plot - 5)
                and wl[s].max() > xmax_plot - 5]                            # passes reaching the long-wl end
        want_up = VCSEL_PASS == "up"
        pass_idx = next(p for p in full if (wl[p[-1]] > wl[p[0]]) == want_up)
        print(f"VCSEL pass ({VCSEL_PASS}): {wl[pass_idx].min():.1f}-{wl[pass_idx].max():.1f} nm, {pass_idx.size} samples")
        lo, hi = wl[pass_idx].min(), min(wl[pass_idx].max(), xmax_plot)
        wavs_truncated, sig_to_norm, fill = resample_uniform(wl[pass_idx], sig_processed_W[pass_idx], lo, hi, VCSEL_GRID_NM)
        _, tap_to_norm_to, _ = resample_uniform(wl[pass_idx], tap_processed_W[pass_idx], lo, hi, VCSEL_GRID_NM)
        print(f"  resampled to a uniform {VCSEL_GRID_NM * 1e3:.0f} pm grid ({fill:.0%} of bins held data)")

    sig_norm_to_tap = sig_to_norm / (tap_to_norm_to**2)
    sig_norm_to_tap_norm = (sig_norm_to_tap - np.min(sig_norm_to_tap)) / (np.max(sig_norm_to_tap) - np.min(sig_norm_to_tap))

    fig, ax = plt.subplots(1, 1, figsize=(14, 9))
    ax.plot(wavs_truncated, sig_norm_to_tap_norm, color=SIG_COLOR, linewidth=1.0)
    ax.set_ylim(0, 1.0)
    ax.set_yticks(np.arange(0.0, 1.05, 0.1))
    ax.set_xlim(xmin_plot, xmax_plot)
    ax.set_xticks(np.arange(xmin_plot, xmax_plot + 1, xstep_plot))
    ax.set_xlabel("Wavelength (nm)", fontsize=label_fs)
    ax.set_ylabel("Normalized $P_{SH}/P_{FUND}^{2}$ (a.u.)", fontsize=label_fs)
    ax.tick_params(axis="both", which="major", labelsize=tick_fs)
    fig.suptitle(f"Device: {row['DevType']}_{row['DevNum']}_WGWidth={row['WGWidth']}", fontsize=title_fs)
    plt.tight_layout()
    show(fig)
    break
assert scan_row is not None, f"no (unsaturated) scan of {device_of_interest} in {data_dir}"
print(f"Scan: {os.path.basename(scan_row['fpath'])}")

#%% Inhomogeneous-broadening correction factor F for the scan loaded above
# Helmfrid, Arvidsson & Webjorn, JOSA B 10, 222 (1993), App. A: the area under P_SH vs. Delta k is unchanged by any
# variation of Delta k along the waveguide, so an ideal (uniform) device of the same length has the SAME area with a
# sinc^2 shape. With both scans scaled to unit peak, the peak-efficiency correction factor is
#     F = P_peak / P_peak,ideal = A_ideal / A_meas   (<= 1),     eta_peak,pred = eta_lossy * F.
# Uses wavs_truncated [nm] and sig_norm_to_tap (= P_SH / P_tap^2, NOT min/max normalized) from the plotting cell.
L_WG_CM = 1.3                                         # [cm] chip length (facet-to-facet), for the fringe-period check

# NOTE - make sure the FH window (wav_fh_sel) spans the measured peak!
# Phase mismatch of TE00 -> TE00 vs. FH wavelength, from the Delta k cell (lam_FH [um], delta_k [rad/um]).
# Only its slope d(Delta k)/d(lambda) enters: the poling period shifts the peak, not its width.
order = np.argsort(lam_FH)                            # spline needs increasing x
LAM_DK_NM = np.asarray(lam_FH)[order] * 1e3
DK_RAD_PER_UM = np.asarray(delta_k)[order]
NG_FH = np.asarray(ng_FH)[order]                      # TE00 FH group index, same order as LAM_DK_NM
assert LAM_DK_NM.size >= 2, "WAV_FH_SEL selects a single wavelength: a range is needed to get d(Delta k)/d(lambda)"
dk_of_lam = CubicSpline(LAM_DK_NM, DK_RAD_PER_UM * 1e6)                     # [rad/m] vs. lambda_FH [nm]
print(f"LAM_DK_NM = {LAM_DK_NM}\nDK_RAD_PER_UM = {DK_RAD_PER_UM}")

INT_WINDOW_NM = (dev["int_window_nm"][0],
                 np.max(wavs_truncated) if dev["int_window_nm"][1] is None else dev["int_window_nm"][1])
BASELINE_WINDOWS_NM = dev["baseline_windows_nm"]
N_FRINGES = 5                                         # boxcar (moving average) width, in facet Fabry-Perot fringe periods


def _plot_stride(n_points, max_points=200_000):
    """Stride that keeps at most ~max_points samples for display (plotting 1e7 points is slow and unreadable)."""
    return max(1, n_points // max_points)


def moving_average_uniform(x, y, width, plot=True):
    """Centered boxcar average of y over a window of `width` (same units as x); x must be uniformly spaced.
    Uses a cumulative sum, so it is O(N) even for the full 11e6-sample sweep. Edges use the available samples.

    plot=True shows the input vs. the smoothed output: the full range, and a zoom of +/- 5 windows around the
    smoothed maximum where the averaged-out ripple is visible.
    """
    dx = np.median(np.diff(x))
    assert np.allclose(np.diff(x), dx, rtol=1e-2), "x must be uniformly spaced"   # 1e-2: float rounding of x
    n = max(1, int(round(width / dx)))
    c = np.concatenate([[0.0], np.cumsum(y)])
    i = np.arange(y.size)
    lo = np.clip(i - n // 2, 0, y.size)
    hi = np.clip(i - n // 2 + n, 0, y.size)
    y_s = (c[hi] - c[lo]) / (hi - lo)

    if plot:
        x_pk = x[np.argmax(y_s)]
        zoom = np.abs(x - x_pk) <= 5 * width
        fig, axes = plt.subplots(1, 2, figsize=(14, 4.5), constrained_layout=True)
        for ax, sel, title in ((axes[0], slice(None), "Full range"),
                               (axes[1], zoom, "Zoom: $\\pm$5 windows around the smoothed peak")):
            xs, ys, yss = x[sel], y[sel], y_s[sel]
            k = _plot_stride(xs.size)
            ax.plot(xs[::k], ys[::k], color="0.7", lw=0.6, label="Input")
            ax.plot(xs[::k], yss[::k], color="maroon", lw=1.6, label=f"Boxcar, width = {width:.4g} ({n} samples)")
            ax.set_title(title)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.grid(True, alpha=0.3)
        axes[1].axvspan(x_pk - width / 2, x_pk + width / 2, color="maroon", alpha=0.12, lw=0, label="One window")
        axes[0].legend(*axes[1].get_legend_handles_labels(), loc="upper right", fontsize=10)   # empty corner
        show(fig)
    return y_s


def fringe_period(x, y, x_lo, x_hi, p_min, p_max, plot=True):
    """Dominant modulation period of y in [x_lo, x_hi], searched between p_min and p_max (same units as x).

    plot=True shows (left) the selected data with the cubic envelope that gets subtracted, and (right) the
    power spectrum of the detrended, Hann-windowed residual with the search band shaded and the found peak marked.
    """
    m = (x >= x_lo) & (x <= x_hi)
    xs, ys = x[m], y[m]
    u = xs - xs.mean()                                                           # centered: well-conditioned fit
    envelope = np.polyval(np.polyfit(u, ys, 3), u)
    resid = ys - envelope                                                        # drop the slow envelope
    P = np.abs(np.fft.rfft(resid * np.hanning(resid.size))) ** 2
    f = np.fft.rfftfreq(resid.size, d=np.median(np.diff(xs)))
    band = (f > 1 / p_max) & (f < 1 / p_min)
    i_pk = np.flatnonzero(band)[np.argmax(P[band])]
    period = 1 / f[i_pk]

    if plot:
        fig, axes = plt.subplots(1, 2, figsize=(14, 4.5), constrained_layout=True)
        k = _plot_stride(xs.size)
        axes[0].plot(xs[::k], ys[::k], color="0.6", lw=0.6, label="Data in window")
        axes[0].plot(xs, envelope, color="maroon", lw=2.0, label="Cubic envelope (subtracted)")
        axes[0].set_xlabel("x")
        axes[0].set_ylabel("y")
        axes[0].set_title(f"Selected window [{x_lo:g}, {x_hi:g}]")
        axes[0].legend(loc="upper right", fontsize=10)
        axes[0].grid(True, alpha=0.3)

        ax = axes[1]
        pos = f > 0                                                              # log axis: drop the DC bin
        ax.loglog(f[pos], P[pos], color="0.35", lw=0.6, label="Power spectrum of residual")
        ax.axvspan(1 / p_max, 1 / p_min, color="tab:blue", alpha=0.12, lw=0,
                   label=f"Search band: period {p_min:g}-{p_max:g}")
        ax.plot(f[i_pk], P[i_pk], "o", ms=9, mfc="none", mec="maroon", mew=2,
                label=f"Peak: period = {period:.4g} (f = {f[i_pk]:.4g})")
        ax.set_xlabel("Frequency (cycles per unit x)")
        ax.set_ylabel("Power (a.u.)")
        ax.set_title("FFT of detrended, Hann-windowed residual")
        ax.legend(loc="lower left", fontsize=10)
        ax.grid(True, which="major", alpha=0.3)
        show(fig)
    return period


def fwhm_about_peak(x, y, i_pk, level=0.5):
    """Full width at `level` x peak of the lobe that contains index i_pk: the nearest crossings on each side of the
    peak, linearly interpolated between samples. Returns nan if the curve never drops below the level on one side."""
    thr = level * y[i_pk]
    left = np.flatnonzero(y[:i_pk] < thr)
    right = np.flatnonzero(y[i_pk:] < thr)
    if left.size == 0 or right.size == 0:
        return np.nan
    il, ir = left[-1], i_pk + right[0]          # y[il] < thr <= y[il+1],  y[ir] < thr <= y[ir-1]
    x_l = np.interp(thr, [y[il], y[il + 1]], [x[il], x[il + 1]])
    x_r = np.interp(thr, [y[ir], y[ir - 1]], [x[ir], x[ir - 1]])
    return x_r - x_l


def inhomogeneous_correction_factor(lam_nm, y, L_m, dk_of_lam, int_window_nm, baseline_windows_nm,
                                    fringe_nm, n_fringes=5):
    """Peak-efficiency correction factor F = A_ideal / A_meas for one phase-matching scan (see cell header).

        A_ideal = int sinc^2[(Delta k(lam) - Delta k(lam_pk)) L/2] d lam   (-> 2 pi / (L |d Delta k / d lam|)),
        A_meas  = int y_s(lam) / y_s(lam_pk) d lam,
    with y_s the scan after fringe averaging and baseline subtraction, both integrated over int_window_nm.
    Any wavelength-independent scaling of y (coupling, detector gain, propagation loss) cancels.
    """
    # 1) average out the facet Fabry-Perot fringes over a whole number of periods (scales peak and area equally)
    y_s = moving_average_uniform(lam_nm, y, n_fringes * fringe_nm)
    # 2) baseline = median of signal-free regions (not the minimum, which is a negative noise excursion)
    base = np.median(np.concatenate([y_s[(lam_nm >= a) & (lam_nm <= b)] for a, b in baseline_windows_nm]))
    y_s = y_s - base
    # 3) peak and area of the measured order
    win = (lam_nm >= int_window_nm[0]) & (lam_nm <= int_window_nm[1])
    lam_w, y_w = lam_nm[win], y_s[win]
    i_pk = np.argmax(y_w)
    lam_pk, y_pk = lam_w[i_pk], y_w[i_pk]
    A_meas = np.trapezoid(y_w / y_pk, lam_w)                                     # [nm]
    fwhm_meas = fwhm_about_peak(lam_w, y_w / y_pk, i_pk)                         # [nm], main lobe of the measured order
    # 4) ideal uniform device of the same length: unit-peak sinc^2 on the measured peak, same window
    x = (dk_of_lam(lam_w) - dk_of_lam(lam_pk)) * L_m / 2
    A_ideal = np.trapezoid(np.sinc(x / np.pi) ** 2, lam_w)                        # np.sinc(u) = sin(pi u) / (pi u)
    slope = dk_of_lam(lam_pk, 1) * 1e9                                           # [rad/m per m]
    return dict(F=A_ideal / A_meas, lam_pk=lam_pk, A_meas=A_meas, A_ideal=A_ideal, baseline=base, y_pk=y_pk,
                A_ideal_analytic=2 * np.pi / (L_m * abs(slope)) * 1e9,         # [nm], infinite window
                fwhm_ideal=4 * 1.39156 / (L_m * abs(slope)) * 1e9,             # [nm]
                fwhm_meas=fwhm_meas,                                           # [nm]
                lam=lam_w, y_norm=y_w / y_pk, sinc2=np.sinc(x / np.pi) ** 2, slope=slope)


# Fringe period measured from the scan, unless the device preset sets it (DEV19_1.6um: VCSEL scan)
fringe_nm = dev["fringe_nm"] if dev["fringe_nm"] is not None else \
    fringe_period(wavs_truncated, sig_norm_to_tap, INT_WINDOW_NM[0], INT_WINDOW_NM[1], 0.005, 0.5)

res = inhomogeneous_correction_factor(wavs_truncated, sig_norm_to_tap, L_POL_M, dk_of_lam,
                                      INT_WINDOW_NM, BASELINE_WINDOWS_NM, fringe_nm, N_FRINGES)
F_inhom = res["F"]
assert 0 < F_inhom <= 1.02, \
    f"F = {F_inhom:.3f}: F > 1 is unphysical for uniform poling and F < 0 means a negative measured area: " \
    "check the scan file (scan_min_date), L_POL_M, the Delta k slope, the windows or the fringe period"

print(f"Facet fringe period: {fringe_nm * 1e3:.1f} pm -> boxcar {N_FRINGES} periods = {N_FRINGES * fringe_nm:.3f} nm")

# Expected facet Fabry-Perot fringe period: FSR = lambda^2 / (2 n_g L_wg). The cavity is the FH mode between the chip
# facets; SH ~ P_FH^2 inside the guide, so the SH scan ripples with the same period.
fsr_pm = (LAM_DK_NM * 1e-9) ** 2 / (2 * NG_FH * L_WG_CM * 1e-2) * 1e12          # [pm]
print(f"Expected fringe period (L_wg = {L_WG_CM:g} cm):\n " +
      ", ".join(f"{l:.0f} nm: {p:.2f} pm (n_g = {g:.4f})\n" for l, p, g in zip(LAM_DK_NM, fsr_pm, NG_FH)))
fsr_pk_pm = np.interp(res["lam_pk"], LAM_DK_NM, fsr_pm)
ng_pk = np.interp(res["lam_pk"], LAM_DK_NM, NG_FH)
L_implied_cm = (res["lam_pk"] * 1e-9) ** 2 / (2 * ng_pk * fringe_nm * 1e-9) * 1e2
print(f"At the measured peak ({res['lam_pk']:.1f} nm): expected {fsr_pk_pm:.2f} pm vs. measured {fringe_nm * 1e3:.2f} pm "
      f"-> implied cavity length {L_implied_cm:.3f} cm")
print("----------------------------------------------------------------------------------------------------------------------")

print(f"Peak at {res['lam_pk']:.3f} nm | d(Delta k)/d(lambda) = {res['slope'] * 1e-9:.1f} rad/m per nm | "
      f"ideal FWHM = {res['fwhm_ideal']:.2f} nm\n")
print(f"Unit-peak areas: measured {res['A_meas']:.3f} nm, ideal {res['A_ideal']:.3f} nm "
      f"(infinite window {res['A_ideal_analytic']:.3f} nm) | baseline = {res['baseline'] / res['y_pk']:+.4f} x peak")
print(f"FWHM: measured {res['fwhm_meas']:.2f} nm vs. ideal {res['fwhm_ideal']:.2f} nm "
      f"(x{res['fwhm_meas'] / res['fwhm_ideal']:.2f}) | equivalent width (unit-peak area): "
      f"measured {res['A_meas']:.2f} nm vs. ideal {res['A_ideal']:.2f} nm (x{1 / F_inhom:.2f})\n")
print(f"Inhomogeneous-broadening correction factor F = {F_inhom:.3f}")
i_eta = np.argmin(np.abs(lam_FH * 1e3 - res["lam_pk"]))                       # nearest simulated wavelength
eta_pred_peak = eta_lossy_pct_W_cm2[i_eta] * F_inhom
if not LAM_DK_NM[0] <= res["lam_pk"] <= LAM_DK_NM[-1]:   # Delta k spline extrapolated, eta taken at the window edge
    print(f"WARNING: the measured peak ({res['lam_pk']:.1f} nm) is outside the FH window "
          f"{LAM_DK_NM[0]:.0f}-{LAM_DK_NM[-1]:.0f} nm: widen wav_fh_sel")
print(f"eta_lossy ({lam_FH[i_eta] * 1e3:.0f} nm) = {eta_lossy_pct_W_cm2[i_eta]:.0f} %/W/cm^2 -> "
      f"predicted peak eta = {eta_pred_peak:.0f} %/W/cm^2")

# Measured order vs. the ideal uniform device with the SAME area (peak 1/F): same integral, different shape
fig, ax = plt.subplots(figsize=(8, 6), tight_layout=True)
ax.plot(res["lam"], res["sinc2"] / F_inhom, color="0.45", lw=1.5, ls="--", label=f"Ideal, same area (peak = 1/F = {1 / F_inhom:.2f})")
ax.plot(res["lam"], res["y_norm"], color=SIG_COLOR, lw=1.5, label="Measured")
ax.set_xlim(*INT_WINDOW_NM)
ax.set_ylim(bottom=0)
ax.set_xlabel("Wavelength (nm)", fontsize=label_fs)
ax.set_ylabel("($P_{SH}/P_{FUND}^{2})_{norm}$", fontsize=label_fs)
ax.tick_params(axis="both", which="major", labelsize=tick_fs)
ax.legend(fontsize=legend_fs, loc="upper right")
if SAVE_FIGS:
    fig.savefig(os.path.join(OUT_DIR, f"inhomogeneous-broadening_{device_of_interest}_MPBX.svg"), dpi=300)
show(fig)

# Measured order vs. the ideal device, both on the eta axis (measured scaled to its measured peak efficiency)
if dev["eta_meas_peak"] is not None:
    fig, ax = plt.subplots(figsize=(8, 6), tight_layout=True)
    ax.plot(res["lam"], eta_lossy_pct_W_cm2[i_eta] * (res["sinc2"] / np.max(res["sinc2"])), color="0.45", lw=1.5,
            ls="--", label="Adapted poling + improved facets")
    ax.plot(res["lam"], dev["eta_meas_peak"] * res["y_norm"], color=SIG_COLOR, lw=1.5, label="Measured")
    ax.set_xlim(*INT_WINDOW_NM)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Wavelength (nm)", fontsize=label_fs)
    ax.set_ylabel(r"$\eta \  (\%/W/cm^2)$", fontsize=label_fs)
    ax.tick_params(axis="both", which="major", labelsize=tick_fs)
    ax.legend(fontsize=legend_fs, loc="upper right")
    if SAVE_FIGS:
        fig.savefig(os.path.join(OUT_DIR, f"inhomogeneous-broadening_{device_of_interest}_MPBX_eta.svg"), dpi=300)
    show(fig)

#%% Summary table (and CSV)
summary = pd.DataFrame({
    "lam_FH_nm": lam_FH * 1e3, "TE00_idx_FH": TE00_idx_FH, "TE00_idx_SH": TE00_idx_SH,
    "neff_FH": neff_FH, "neff_SH": neff_SH, "ng_FH": ng_FH, "delta_k_rad_per_um": delta_k,
    "poling_period_um": poling_period, "d33_max_pm_per_V": chi2_max / 2 * 1e12,
    "A_eff_FH_um2": A_eff_FH / UM**2, "A_eff_SH_um2": A_eff_SH / UM**2, "A_bar_um2": A_bar / UM**2,
    "overlap_abs": np.abs(overlap), "eta_pct_W_cm2": eta_pct_W_cm2, "eta_lossy_pct_W_cm2": eta_lossy_pct_W_cm2,
})
print(summary.round(5).to_string(index=False))
print(f"\n{DEVICE}: F = {F_inhom:.3f}, peak at {res['lam_pk']:.2f} nm -> predicted peak eta = {eta_pred_peak:.0f} %/W/cm^2 "
      f"(eta_lossy {eta_lossy_pct_W_cm2[i_eta]:.0f} at {lam_FH[i_eta] * 1e3:.0f} nm)")
if SAVE_FIGS:
    summary.to_csv(os.path.join(OUT_DIR, f"eta_table_{DEVICE}_MPBX.csv"), index=False)
