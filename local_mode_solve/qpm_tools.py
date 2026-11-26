"""Shared helpers for the TFLN SHG / QPM design and analysis notebooks.

Used by chirped_doubler_fourier_design.ipynb and dispersion_error_plot_2799.ipynb (import from the
notebook's folder: `from qpm_tools import ...`).

Units: lengths in um unless the name says _nm or _m. Wavenumbers / Delta_k in rad/um.
Convention: Delta_k = k_SH - 2 k_FH (TE00 -> TE00), first-order QPM when Delta_k = 2 pi / Lambda.

Sections:
    Paths               ROOT_PREFIX, sweep folders, mode-file lookup
    Dispersion          TE00 n_eff from a *_modes.csv, Delta_k(lambda_FH), inverse lambda_FH(Delta_k)
    QPM grating         linearly chirped +-1 domain pattern (layout function and an exact lookup)
    Grating spectrum    Fourier amplitude via FFT or exactly from the domain edges, and FWHM
"""
import os
import platform
import re

import numpy as np
import pandas as pd
from natsort import natsorted
from scipy.fft import rfft, rfftfreq
from scipy.interpolate import CubicSpline

# ============================================================================ #
# Paths
# ============================================================================ #
USER = os.getlogin()
if platform.system() == "Darwin":  # macOS
    ROOT_PREFIX = f"/Users/{USER}/"
elif platform.system() == "Windows":
    ROOT_PREFIX = f"C:/Users/{USER}/"
else:
    raise OSError(f"Unsupported operating system: {platform.system()}")

MPBX_SIMS_DIR = os.path.join(ROOT_PREFIX, "MIT Dropbox/Sagar Doshi/MPBX_SIMS/")
# t_LN = 390:2:410nm x W_WG = 1.5:0.05:1.9um sweeps (etch = full film, 70 deg sidewall, 1350nm SiO2 clad)
SWEEP_MATSUDA = os.path.join(MPBX_SIMS_DIR, "tfln_sweep_2026-11-19-16-58/")  # Matsuda MgO:LN
# SWEEP_NANOLN  = os.path.join(MPBX_SIMS_DIR, "tfln_sweep_2026-11-20-23-55/")  # Perturbed Matsuda - NanoLN sample (Die 2_1 VASE)
SWEEP_NANOLN  = os.path.join(MPBX_SIMS_DIR, "tfln_sweep_2026-11-22-00-31/")

_GEOM_RE = re.compile(r"tLN(\d+)_etch\d+_w(\d+)_")


def list_modes_fpaths(sweep_dir):
    """Every <geometry>/<geometry>_modes.csv in a sweep folder, naturally sorted."""
    return natsorted([os.path.join(sweep_dir, f"{x}/{x}_modes.csv") for x in os.listdir(sweep_dir)
                      if os.path.isdir(os.path.join(sweep_dir, x)) and x not in ['code', 'logs']])


def parse_geom(fpath):
    """(t_ln_nm, w_top_nm) from a sweep file name like tLN400_etch400_w1600_sw70_..._modes.csv."""
    m = _GEOM_RE.search(os.path.basename(fpath))
    if m is None:
        raise ValueError(f"Can't parse t_LN / W from {fpath}")
    return int(m.group(1)), int(m.group(2))


def get_modes_fpath(t_ln_nm, w_top_nm, sweep_dir):
    """The _modes.csv path in sweep_dir for a given LN thickness and top width (both in nm)."""
    matches = [f for f in list_modes_fpaths(sweep_dir) if parse_geom(f) == (t_ln_nm, w_top_nm)]
    if len(matches) != 1:
        raise ValueError(f"Expected 1 modes file for tLN={t_ln_nm}nm, w={w_top_nm}nm in {sweep_dir}, found {len(matches)}: {matches}")
    return matches[0]


# ============================================================================ #
# Dispersion: n_eff -> Delta_k(lambda)
# ============================================================================ #
def get_neff_TE00(df_modes, lam_range_nm):
    """
    Guided TE00 n_eff vs. wavelength [nm] within lam_range_nm (inclusive).
    guided==1 drops the spurious, unguided window-edge "TE00" rows the labeller produces at ~1930-2000nm.
    """
    df_te00 = df_modes[(df_modes['label'] == 'TE00') & (df_modes['guided'] == 1)].copy()
    df_te00['lam_nm'] = np.round(df_te00['lam_um'] * 1000, 6)
    df_te00 = df_te00[(df_te00['lam_nm'] >= lam_range_nm[0]) & (df_te00['lam_nm'] <= lam_range_nm[1])]
    # If a wavelength still has >1 guided TE00 candidate, keep the highest-index one (the fundamental)
    df_te00 = df_te00.sort_values('neff', ascending=False).drop_duplicates('lam_nm').sort_values('lam_nm')
    return df_te00['lam_nm'].values, df_te00['neff'].values


def delta_k_TE00(df_modes, lam_FH_nm):
    """
    Delta_k = k_SH - 2 k_FH [rad/um] (TE00 -> TE00) on the FH wavelength grid lam_FH_nm [nm].

    The solver's n_eff (typically 10nm spaced) is cubic-splined onto lam_FH_nm (FH) and lam_FH_nm/2 (SH),
    using solver points in [min, max] of each grid.
    """
    lam_FH_nm = np.asarray(lam_FH_nm, dtype=float)
    lam_SH_nm = lam_FH_nm / 2.0
    fh_range  = (lam_FH_nm.min(), lam_FH_nm.max())
    sh_range  = (fh_range[0] / 2.0, fh_range[1] / 2.0)

    lam_FH_solved, neff_FH = get_neff_TE00(df_modes, fh_range)
    lam_SH_solved, neff_SH = get_neff_TE00(df_modes, sh_range)

    k_FH = 2*np.pi*CubicSpline(lam_FH_solved, neff_FH)(lam_FH_nm) / (lam_FH_nm/1000)  # [rad/um]
    k_SH = 2*np.pi*CubicSpline(lam_SH_solved, neff_SH)(lam_SH_nm) / (lam_SH_nm/1000)  # [rad/um]
    return k_SH - 2*k_FH


def delta_k_for_geom(t_ln_nm, w_top_nm, sweep_dir, lam_FH_nm):
    """Delta_k(lam_FH_nm) [rad/um] for one (t_LN, W) geometry of a sweep. Returns (delta_k, modes_fpath)."""
    modes_fpath = get_modes_fpath(t_ln_nm, w_top_nm, sweep_dir)
    return delta_k_TE00(pd.read_csv(modes_fpath), lam_FH_nm), modes_fpath


def lam_FH_at_delta_k(delta_k, lam_FH_nm, K):
    """
    FH wavelength [nm] where delta_k(lam_FH_nm) = K, for scalar or array K (linear interpolation on the grid).
    Needs delta_k strictly decreasing in wavelength (true for TE00 -> TE00 SHG in TFLN; checked) so the crossing is unique.
    NaN where K is outside delta_k's range, i.e. the crossing is outside the lam_FH_nm window.
    """
    delta_k, lam_FH_nm = np.asarray(delta_k), np.asarray(lam_FH_nm)
    if np.any(np.diff(delta_k) >= 0):
        raise ValueError("delta_k is not strictly decreasing in wavelength; the crossing may not be unique.")
    return np.interp(K, delta_k[::-1], lam_FH_nm[::-1], left=np.nan, right=np.nan)  # np.interp needs increasing x -> reverse


# ============================================================================ #
# QPM grating: linearly chirped poling period
# ============================================================================ #
def chirped_domain_edges(L_pol_um, chirp_pp_start_um, chirp_pp_end_um):
    """
    Domain boundaries of the chirped +-1 pattern, by replaying the layout's stepped algorithm
    (generate_chirped_chi2_norm): the local period is interp'd at the start of each period, 50% duty cycle.

    Returns (edges_um, signs): signs[j] is valid on [edges_um[j], edges_um[j+1]); the last domain runs to L_pol_um.
    """
    edges = [0.0]   # z-positions where the sign changes
    signs = [1]     # sign valid on [edges[k], edges[k+1])

    current_z = 0.0
    while current_z < L_pol_um:
        period = np.interp(current_z, [0.0, L_pol_um], [chirp_pp_start_um, chirp_pp_end_um])
        half = period / 2.0

        edges.append(current_z + half)   # start of the -1 sub-domain
        signs.append(-1)

        current_z += period              # start of next full period
        if current_z < L_pol_um:
            edges.append(current_z)      # start of the next +1 sub-domain
            signs.append(1)

    return np.asarray(edges), np.asarray(signs)


def make_chirped_chi2_lookup(L_pol_um, chirp_pp_start_um, chirp_pp_end_um):
    """
    Returns a callable chi2(z_m) (z in meters) that exactly reproduces generate_chirped_chi2_norm,
    using precomputed domain boundaries + searchsorted (grid-resolution independent).
    """
    edges, signs = chirped_domain_edges(L_pol_um, chirp_pp_start_um, chirp_pp_end_um)
    n = signs.size

    def chi2(z_m):
        z = np.asarray(z_m, dtype=float) * 1e6  # convert to um
        idx = np.searchsorted(edges, z, side='right') - 1
        idx = np.clip(idx, 0, n - 1)
        return signs[idx]

    return chi2


# This is the function that was used to layout the chip. Kept verbatim as the reference the lookup is checked against.
def generate_chirped_chi2_norm(z_array_um, L_pol_um, chirp_pp_start_um, chirp_pp_end_um):
    """
    Efficiently generates a chi2_norm array for a linear chirp without integration.

    It iteratively calculates the domain boundaries based on the local poling period.

    Args:
        z_array_um (np.ndarray): Array of z-positions [um] to calculate chi2 for.
        L_pol_um (float): Total length of the poled section [um].
        chirp_pp_start_um (float): Poling period at the start [um].
        chirp_pp_end_um (float): Poling period at the end [um].

    Returns:
        np.ndarray: An array of +1/-1 values representing the chirped square wave.
    """
    chi2_norm = -1*np.ones_like(z_array_um)
    current_z_um = 0.0
    sign = 1

    while current_z_um < L_pol_um:
        # Determine the local period at the start of the current domain
        local_period_um = np.interp(current_z_um, [0, L_pol_um], [chirp_pp_start_um, chirp_pp_end_um])

        # A full period consists of a positive and a negative domain (50% duty cycle assumed for the base pattern)
        domain_width_um = local_period_um / 2.0

        start_z = current_z_um
        end_z = current_z_um + domain_width_um

        # Find the indices in the z_array that fall within this domain
        # This is more efficient than a direct loop over the z_array
        indices = np.where((z_array_um >= start_z) & (z_array_um < end_z))

        if indices[0].size > 0:
            chi2_norm[indices] = sign

        # Move to the start of the next domain
        current_z_um = current_z_um + local_period_um

    # Ensure the first value is +1 if it was missed (e.g., if first domain is very small)
    chi2_norm[0] = 1

    return chi2_norm


# ============================================================================ #
# Grating spectrum
# ============================================================================ #
# Normalization (both functions): G(dk) = (1/L) int_0^L g(z) exp(-i dk z) dz, so |G| = d_eff/d33 at that Delta_k.
# A uniform 50%-duty grating gives |G|^2 = (2/pi)^2 = 0.405 at its first-order peak; a linear chirp spreads that
# over its K band, giving a plateau of ~(2/pi)^2 * 2pi / (L * (K_max - K_min)).
def qpm_spectrum_fft(g, dz_um, pad_factor=4):
    """
    |G(dk)|^2 of a sampled grating g (on a uniform z grid, step dz_um) via a zero-padded real FFT.
    Returns (dk [rad/um], |G|^2). Padding only interpolates the spectrum (finer dk sampling), it does not add resolution.
    """
    g = np.asarray(g, dtype=float)
    N = g.size
    N_fft = int(2 ** np.ceil(np.log2(N * pad_factor)))
    G = rfft(g, n=N_fft) / N  # Normalize by the unpadded length so padding doesn't shrink the amplitude
    dk = 2*np.pi*rfftfreq(N_fft, d=dz_um)  # Delta_k = 2 pi f
    return dk, np.abs(G)**2


def qpm_spectrum_exact(edges_um, signs, L_um, dk, chunk=200):
    """
    |G(dk)|^2 of a piecewise-constant +-1 grating, computed exactly from its domain edges (no z grid), at any dk [rad/um]:
        G(dk) = (1/L) sum_j s_j (exp(-i dk a_j) - exp(-i dk b_j)) / (i dk),   domain j = [a_j, b_j)
    Cost ~ len(dk) * len(edges); ~2s for 20k dk points x 7k domains.
    """
    a = np.asarray(edges_um, dtype=float)
    b = np.minimum(np.append(a[1:], L_um), L_um)
    dk = np.atleast_1d(np.asarray(dk, dtype=float))
    G = np.empty(dk.size, dtype=complex)
    for i in range(0, dk.size, chunk):
        k = dk[i:i + chunk, None]
        G[i:i + chunk] = (signs * (np.exp(-1j*k*a) - np.exp(-1j*k*b))).sum(axis=1) / (1j*dk[i:i + chunk])
    return np.abs(G / L_um)**2


def fwhm(x, y, window=None, level_from='plateau', percentile=90):
    """
    Full width at half max of a single (possibly flat-topped, rippled) peak, with linearly interpolated crossings.

    level_from='peak'    -> half of the global max (sensitive to ripple tips)
    level_from='plateau' -> half of the median of the top-(100-percentile)% samples in the window.
                            Note: this depends somewhat on the window width (the band's share of the window).
    Crossings are the OUTERMOST half-level crossings in the window, so ripple dips inside a flat top can't
    truncate the width. Keep the window tight enough to exclude side lobes that reach half the level.

    Returns (width, xL, xR, top, half).
    """
    x, y = np.asarray(x), np.asarray(y)
    if window is not None:
        m = (x >= window[0]) & (x <= window[1])
        x, y = x[m], y[m]

    if level_from == 'peak':
        top = y.max()
    elif level_from == 'plateau':
        top = np.median(y[y >= np.percentile(y, percentile)])
    else:
        raise ValueError(f"level_from must be 'peak' or 'plateau', got {level_from!r}")
    half = top / 2.0

    i_above = np.flatnonzero(y >= half)
    iL, iR = i_above[0], i_above[-1]
    if iL == 0 or iR == y.size - 1:
        raise ValueError("Half-max crossing is at the window edge; widen the window.")

    lerp = lambda i, j: x[i] + (half - y[i]) * (x[j] - x[i]) / (y[j] - y[i])
    xL, xR = lerp(iL - 1, iL), lerp(iR, iR + 1)
    return xR - xL, xL, xR, top, half
