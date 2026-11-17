r"""X-cut MgO:LN (Matsuda) rib waveguide for mpbx: geometry, materials, smoothed
dielectric, and a refractive-index plot with the geometry outline drawn over it.

Frames and conventions
----------------------
mpbx lab frame for 2-D waveguide cross-sections:
    x : horizontal, in the film plane (waveguide width)
    y : vertical, film normal (the SiO2 substrate is below the LN film)
    z : propagation direction
X-cut, Y-propagating TFLN, crystal axes in that lab frame:
    lab x = crystal Z  -> sees n_e   -> "TE"  = Ex-dominant
    lab y = crystal X  -> sees n_o   -> "TM"  = Ey-dominant
    lab z = crystal Y  -> sees n_o   (propagation)
The Tidy3D notebook (local_mode_solve.ipynb) uses x = vertical, z = horizontal,
y = propagation, so:   mpbx Ex <-> Tidy3D Ez (TE),   mpbx Ey <-> Tidy3D Ex (TM).

Units: µm everywhere; ω = 1/λ [µm⁻¹] (no 2π); n_eff = k/ω.

Rib cross-section (all heights measured from the LN/substrate interface):

             |<-- w_top -->|
              _____________          ___ y_top  = y_bot + t_ln
             /             \            |  etch_depth
    ________/               \________  _|_ y_slab = y_bot + (t_ln - etch_depth)
   |              LN slab            |  |  slab thickness = t_ln - etch_depth
   |_________________________________| _|_ y_bot  (LN / SiO2 substrate interface)
             SiO2 substrate (BOX)
   sidewall angle = interior angle at the ridge base, measured from HORIZONTAL
   (90° = vertical walls), the same convention as the Tidy3D notebook (73°).
"""
from __future__ import annotations

import dataclasses
import math
import matplotlib.pyplot as plt
import warnings

import jax.numpy as jnp
import numpy as np

from mpbjax import grid_2d, shapes as Shp
from mpbjax.bend import slab_neff
from mpbjax.helmholtz import sliceinv_3x3
from mpbjax.materials import MgO_LiNbO3_matsuda, SiO2, Vacuum, eps_dispersion, mat_vals, rotate
from mpbjax.smoothing_plan import apply_plan, smoothing_plan

# --------------------------------------------------------------------------------------
# Materials
# --------------------------------------------------------------------------------------
# mpbx's rotate(mat, R) gives eps_lab = R^T · eps_crystal · R, so lab axis i is the
# crystal direction R[:, i]. This R puts crystal Z on lab x, crystal X on lab y
# (film normal) and crystal Y on lab z (propagation): a true X-cut, Y-propagating film.
R_XCUT = np.array([[0.0, 1.0, 0.0],
                   [0.0, 0.0, 1.0],
                   [1.0, 0.0, 0.0]])
# The mpbx example (inverse_design_waveguide_dispersion.py) and tutorial_LN_waveguide_modes.py
# use this Ry instead. It gives the SAME permittivity tensor (diag(n_e², n_o², n_o²) in
# the lab frame) because n_X = n_Y = n_o, but it puts crystal Y on the film normal and
# crystal X along propagation. That only matters for chi(2)/electro-optic tensors.
R_Y_MPBX_EXAMPLE = np.array([[0.0, 0.0, 1.0],
                             [0.0, 1.0, 0.0],
                             [-1.0, 0.0, 0.0]])

LN_XCUT = rotate(MgO_LiNbO3_matsuda, R_XCUT, name="MgO_LiNbO3_matsuda_xcut")
MATS = [LN_XCUT, SiO2, Vacuum]          # material columns used by every geometry below
I_LN, I_SIO2, I_AIR = 0, 1, 2
LAM_RANGE = (0.4, 2.0)                  # Matsuda 2015 Sellmeier validity (MgO_LiNbO3_matsuda.lam_range)


def ln_indices(lam):
    """(n_e, n_o) of the Matsuda MgO:LN model at wavelength ``lam`` [µm] (20 °C)."""
    eps = np.asarray(eps_dispersion(LN_XCUT, 1.0 / lam)[0])
    return float(np.sqrt(eps[0, 0])), float(np.sqrt(eps[1, 1]))


def sio2_index(lam):
    """n of the Malitson SiO2 model at ``lam`` [µm]."""
    return float(np.sqrt(np.asarray(eps_dispersion(SiO2, 1.0 / lam)[0])[0, 0]))


def check_material_frame(lam=1.55, verbose=True):
    """Check that lab x sees n_e and lab y, z see n_o, and that the mpbx-example Ry
    rotation gives the same permittivity tensor. Returns the lab-frame index diagonal."""
    eps = np.asarray(eps_dispersion(LN_XCUT, 1.0 / lam)[0])
    eps_ry = np.asarray(eps_dispersion(rotate(MgO_LiNbO3_matsuda, R_Y_MPBX_EXAMPLE), 1.0 / lam)[0])
    n_diag = np.sqrt(np.diag(eps))
    assert np.allclose(eps, np.diag(np.diag(eps))), "lab-frame LN tensor should be diagonal"
    assert n_diag[0] < n_diag[1] and np.isclose(n_diag[1], n_diag[2]), "expected (n_e, n_o, n_o)"
    assert np.allclose(eps, eps_ry, atol=1e-14), "R_XCUT and the mpbx-example Ry should agree"
    if verbose:
        print(f"[material] λ={lam:.3f} µm  lab-frame n = (x: {n_diag[0]:.6f}, y: {n_diag[1]:.6f}, "
              f"z: {n_diag[2]:.6f}) = (n_e, n_o, n_o);  n_SiO2 = {sio2_index(lam):.6f}")
    return n_diag


# --------------------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class RibGeometry:
    """X-cut TFLN rib (trapezoid ridge on an optional LN slab) on a SiO2 substrate.

    t_ln         : total LN film thickness [µm]
    etch_depth   : ridge height = etch depth [µm]. etch_depth == t_ln is a fully etched
                   ridge (no slab); 0 < etch_depth < t_ln leaves a slab of thickness
                   t_ln - etch_depth; etch_depth == 0 is an unetched film.
    w_top        : ridge top width [µm]
    sidewall_deg : sidewall angle at the ridge base, from HORIZONTAL [deg] (90 = vertical)
    t_clad       : SiO2 top cladding. 0 -> no cladding (air); 0 < t_clad < inf -> planar
                   SiO2 that fills the etched trenches with its flat top surface t_clad
                   above the ridge top; inf -> SiO2 everywhere above (fully clad, as in
                   the Tidy3D notebook).
    y_bot        : y of the LN/substrate interface [µm]. None centers the LN film on y = 0.
    """
    t_ln: float = 0.400
    etch_depth: float = 0.400
    w_top: float = 1.800
    sidewall_deg: float = 73.0
    t_clad: float = math.inf
    y_bot: float | None = None

    def __post_init__(self):
        if not self.t_ln > 0:
            raise ValueError(f"t_ln must be > 0, got {self.t_ln}")
        if not 0 <= self.etch_depth <= self.t_ln:
            raise ValueError(f"etch_depth must be in [0, t_ln={self.t_ln}], got {self.etch_depth}")
        if not 0 < self.sidewall_deg <= 90:
            raise ValueError(f"sidewall_deg must be in (0, 90], got {self.sidewall_deg}")
        if self.etch_depth > 0 and not self.w_top > 0:
            raise ValueError(f"w_top must be > 0, got {self.w_top}")
        if not self.t_clad >= 0:
            raise ValueError(f"t_clad must be >= 0 (0 = air, inf = fully clad), got {self.t_clad}")

    # ---- derived heights / widths ---------------------------------------------------
    @property
    def t_slab(self):
        """Remaining LN slab thickness after the etch [µm]."""
        return self.t_ln - self.etch_depth

    @property
    def w_base(self):
        """Ridge width at its base (slab top / substrate) [µm]."""
        if self.sidewall_deg >= 90:
            return self.w_top
        return self.w_top + 2.0 * self.etch_depth / math.tan(math.radians(self.sidewall_deg))

    @property
    def y0(self):
        """LN / substrate interface height [µm]."""
        return -0.5 * self.t_ln if self.y_bot is None else float(self.y_bot)

    @property
    def y_slab(self):
        """Top of the slab = base of the ridge [µm]."""
        return self.y0 + self.t_slab

    @property
    def y_top(self):
        """Top of the ridge [µm]."""
        return self.y0 + self.t_ln

    @property
    def y_clad_top(self):
        """Top surface of the SiO2 cladding [µm] (inf if fully clad, y0 if no cladding)."""
        if math.isinf(self.t_clad):
            return math.inf
        return self.y_top + self.t_clad if self.t_clad > 0 else self.y0

    @property
    def fully_clad(self):
        return math.isinf(self.t_clad)

    def replace(self, **kw):
        """Copy with some parameters changed, e.g. ``geom.replace(w_top=1.6)``."""
        return dataclasses.replace(self, **kw)

    def as_dict(self):
        return {f.name: getattr(self, f.name) for f in dataclasses.fields(self)}

    def describe(self):
        clad = ("SiO2 everywhere above (fully clad)" if self.fully_clad else
                "none (air)" if self.t_clad == 0 else
                f"planar SiO2, top surface {self.t_clad:.3f} µm above the ridge top")
        return (f"LN film t_ln = {self.t_ln:.3f} µm, etch depth = {self.etch_depth:.3f} µm "
                f"(slab {self.t_slab:.3f} µm), w_top = {self.w_top:.3f} µm, "
                f"w_base = {self.w_base:.3f} µm, sidewall = {self.sidewall_deg:.1f}° from horizontal; "
                f"cladding: {clad}; LN/SiO2 interface at y = {self.y0:+.3f} µm")

    # ---- mpbx shapes ----------------------------------------------------------------
    def shapes(self, far=200.0):
        """(shapes, minds) for ``kottke.smooth_eps`` / ``smoothing_plan``.

        mpbx resolves overlaps FIRST-shape-wins (the opposite of Tidy3D), so the LN ridge
        and slab come before the SiO2 box. The SiO2 substrate and the cladding are one
        box spanning y in [y0 - far/2, y_clad_top]. ``minds`` has one entry per shape plus
        the background (last), each an index into ``MATS``. ``far`` [µm] only needs to
        exceed the simulation window.
        """
        shapes, minds = [], []
        if self.etch_depth > 0:          # a zero-height Trapezoid raises in mpbx: omit it
            shapes.append(Shp.Trapezoid([0.0, self.y_slab], self.w_base, self.etch_depth,
                                        math.radians(self.sidewall_deg)))
            minds.append(I_LN)
        if self.t_slab > 0:
            shapes.append(Shp.Cuboid.make([0.0, self.y0 + 0.5 * self.t_slab], [far, self.t_slab]))
            minds.append(I_LN)
        if self.fully_clad:
            minds.append(I_SIO2)         # background SiO2 = substrate + cladding
        else:
            y_lo = self.y0 - 0.5 * far
            y_hi = self.y_clad_top
            shapes.append(Shp.Cuboid.make([0.0, 0.5 * (y_lo + y_hi)], [far, y_hi - y_lo]))
            minds.append(I_SIO2)
            minds.append(I_AIR)          # background = air above the oxide
        return tuple(shapes), tuple(minds)

    # ---- analytic masks / outline (for diagnostics and plots) -----------------------
    def ln_mask(self, X, Y):
        """Boolean mask of points (X, Y) inside the LN (ridge or slab); half-open in y so a
        point exactly on an interface is assigned to one side only."""
        X, Y = np.asarray(X), np.asarray(Y)
        in_slab = (self.t_slab > 0) & (Y >= self.y0) & (Y < self.y_slab)
        frac = np.clip((Y - self.y_slab) / max(self.etch_depth, 1e-30), 0.0, 1.0)
        half_w = 0.5 * (self.w_base + (self.w_top - self.w_base) * frac)
        in_ridge = (self.etch_depth > 0) & (Y >= self.y_slab) & (Y < self.y_top) & (np.abs(X) <= half_w)
        return in_slab | in_ridge

    def ln_fill(self, grid, ss=4):
        """(Nx, Ny) LN area fraction of every pixel (ss x ss sub-samples per pixel)."""
        xs, ys = grid_coords(grid)
        dx, dy = grid.Lx / grid.Nx, grid.Ly / grid.Ny
        off = (np.arange(ss) + 0.5) / ss - 0.5
        fill = np.zeros((len(xs), len(ys)))
        for ox in off:
            for oy in off:
                X, Y = np.meshgrid(xs + ox * dx, ys + oy * dy, indexing="ij")
                fill += self.ln_mask(X, Y)
        return fill / ss ** 2

    def outline(self, x_extent):
        """Polylines [(x, y), ...] tracing the material boundaries inside ``x_extent``."""
        xa, xb = x_extent
        lines = []
        if self.etch_depth > 0:          # ridge (trapezoid; closed when fully etched)
            xs = [-self.w_base / 2, -self.w_top / 2, self.w_top / 2, self.w_base / 2]
            ys = [self.y_slab, self.y_top, self.y_top, self.y_slab]
            if self.t_slab == 0:
                xs.append(xs[0]), ys.append(ys[0])
            lines.append((np.array(xs), np.array(ys)))
        if self.t_slab > 0:              # slab top on both sides of the ridge
            xl = -self.w_base / 2 if self.etch_depth > 0 else xb
            xr = self.w_base / 2 if self.etch_depth > 0 else xa
            lines.append((np.array([xa, xl]), np.array([self.y_slab] * 2)))
            lines.append((np.array([xr, xb]), np.array([self.y_slab] * 2)))
        lines.append((np.array([xa, xb]), np.array([self.y0] * 2)))    # LN / substrate
        if not self.fully_clad and self.t_clad > 0:
            lines.append((np.array([xa, xb]), np.array([self.y_clad_top] * 2)))
        return lines

    def cutoff_index(self, lam):
        """Guidance cutoff n_eff at ``lam``: a ridge mode is guided if n_eff is above the
        cladding/substrate indices AND, for a partial etch, above the slab-mode indices.
        (An unetched film has no ridge, so only the cladding/substrate cutoff applies.)"""
        n_sub = sio2_index(lam)
        n_top = n_sub if (self.fully_clad or self.t_clad > 0) else 1.0
        n_cut = max(n_sub, n_top)
        if self.t_slab > 0 and self.etch_depth > 0:
            n_e, n_o = ln_indices(lam)
            om = 1.0 / lam
            n_cut = max(n_cut,
                        float(slab_neff(n_e, n_sub, n_top, self.t_slab, om, pol="TE")),
                        float(slab_neff(n_o, n_sub, n_top, self.t_slab, om, pol="TM")))
        return n_cut


# --------------------------------------------------------------------------------------
# Simulation window and smoothed dielectric
# --------------------------------------------------------------------------------------
def make_grid(Lx=6.0, Ly=6.0, dx=0.030, dy=None, odd_x=False):
    """Periodic mpbx window of Lx x Ly µm centered on the origin with pitch ~dx, dy [µm].
    Pixel counts are even integers, except Nx is odd with ``odd_x=True``, which the
    x-mirror parity solve needs: on an even-N axis mpbx's mirror projector zeroes the
    Nyquist plane and the sector solve does not converge (mpbx GOTCHAS #33).
    Pixel centers sit at -L/2 + i·d (the first one ON the left edge).
    So use odd_x=True for x-mirror parity solves, but note that you might lose FFT speed because of the odd Nx, Ny"""
    dy = dx if dy is None else dy
    Nx = 2 * max(1, int(round(Lx / dx / 2)))
    if odd_x:
        Nx = max(1, int(round(Lx / dx)))
        Nx += 1 - Nx % 2
    Ny = 2 * max(1, int(round(Ly / dy / 2)))
    for name, L, N, d in (("x", Lx, Nx, dx), ("y", Ly, Ny, dy)):
        if abs(L / N - d) > 1e-3 * d:
            warnings.warn(f"make_grid: L{name} = {L} µm is not an even multiple of d{name} = {d} µm; "
                          f"using d{name} = {L / N:.6f} µm ({N} px)")
    return grid_2d(float(Lx), float(Ly), Nx, Ny)


def grid_coords(grid):
    """(xs, ys) pixel-center coordinates [µm] as numpy arrays."""
    return np.asarray(grid.xs(), dtype=float), np.asarray(grid.ys(), dtype=float)


def align_to_grid(geom: RibGeometry, grid, frac=0.0, warn=True):
    """Copy of ``geom`` shifted vertically so the LN/substrate interface sits ``frac`` of a
    pixel above a pixel center (frac = 0: ON a pixel center).

    Why: with sub-pixel smoothing the n_eff error is C·dy², but C depends on where the
    horizontal interfaces fall inside their pixels (measured for a 0.4 µm film at
    λ = 1 µm: C ≈ -1.5 (TE) / -2.7 (TM) at frac 0, -0.5 / -0.9 at frac 1/3, +0.7 / +1.3 at
    frac 1/2, µm⁻²). Pitches that move the interfaces to a different frac therefore give
    a zig-zag "convergence". Keep frac fixed, and make the film (and slab) an integer
    number of pixels thick, so the error is one smooth C·dy² term (Richardson-friendly:
    halving dy keeps frac = 0). The same applies to the slab top and to the top surface of
    a finite cladding; a warning is issued for any layer that is not a whole number of pixels."""
    _, ys = grid_coords(grid)
    dy = grid.Ly / grid.Ny
    j = round((geom.y0 - ys[0]) / dy - frac)
    out = geom.replace(y_bot=float(ys[0] + (j + frac) * dy))
    if warn:
        layers = [("t_ln", geom.t_ln), ("t_slab", geom.t_slab)]
        if 0 < geom.t_clad < math.inf:                  # SiO2/air surface above the ridge
            layers.append(("t_clad", geom.t_clad))
        for name, t in layers:
            if t > 0 and abs(t / dy - round(t / dy)) > 1e-6:
                warnings.warn(f"{name} = {t} µm is not an integer number of pixels (dy = {dy:.5f} µm): "
                              "its top interface sits at a different sub-pixel position than the bottom one")
    return out


class RibDielectric:
    """Kottke-smoothed permittivity of a ``RibGeometry`` on a ``grid``, evaluated per λ.

    The geometry pass (which pixels straddle which interfaces, normals, fill fractions)
    is computed once (``smoothing_plan``); each wavelength only re-evaluates the material
    dispersion. ``at(lam)`` returns a dict with (numpy arrays):
        eps   (3,3,Nx,Ny)  smoothed ε tensor           einv  (3,3,Nx,Ny) its pointwise inverse
        deps  (3,3,Nx,Ny)  ∂ε/∂ω  [µm]                 d2eps (3,3,Nx,Ny) ∂²ε/∂ω²  [µm²]
        nng   (3,3,Nx,Ny)  ∂(ωε)/∂ω = ε + ω ∂ε/∂ω (energy-density tensor)
    """

    def __init__(self, geom: RibGeometry, grid):
        self.geom, self.grid = geom, grid
        self.shapes, self.minds = geom.shapes()
        self.plan = smoothing_plan(self.shapes, self.minds, grid)
        dy = grid.Ly / grid.Ny
        for name, t in (("slab", geom.t_slab), ("cladding", geom.t_clad)):
            if 0 < t < dy:
                warnings.warn(f"{name} thickness {t:.4f} µm is below the vertical pitch {dy:.4f} µm: "
                              "a layer that holds no pixel corner is invisible to the smoothing "
                              "(mpbx GOTCHAS #26). Refine the grid.")

    def at(self, lam):
        lo, hi = LAM_RANGE
        if not lo - 1e-9 <= lam <= hi + 1e-9:
            warnings.warn(f"λ = {lam} µm is outside the Matsuda model range {LAM_RANGE} µm (extrapolating)")
        om = 1.0 / lam
        mv = np.asarray(mat_vals(MATS, om))
        sm = apply_plan(self.plan, mv)                       # (3,3,3,Nx,Ny): [.., 0]=ε, 1=∂ε/∂ω, 2=∂²ε/∂ω²
        eps = sm[:, :, 0]
        einv = np.asarray(sliceinv_3x3(eps))
        eps, deps, d2eps = (np.asarray(sm[:, :, j]) for j in range(3))
        return dict(lam=lam, omega=om, eps=eps, einv=einv, deps=deps, d2eps=d2eps,
                    nng=eps + om * deps)


# --------------------------------------------------------------------------------------
# Plot: refractive index map with the geometry outline drawn over it
# --------------------------------------------------------------------------------------
_OUTLINE = "#eb6834"      # orange, contrasts with the grey index map


def _pixel_edges(grid):
    xs, ys = grid_coords(grid)
    dx, dy = grid.Lx / grid.Nx, grid.Ly / grid.Ny
    return np.append(xs - dx / 2, xs[-1] + dx / 2), np.append(ys - dy / 2, ys[-1] + dy / 2)


def draw_outline(ax, geom, grid, color=_OUTLINE, lw=1.0, **kw):
    xs, _ = grid_coords(grid)
    for x, y in geom.outline((xs[0], xs[-1])):
        ax.plot(x, y, color=color, lw=lw, **kw)


def plot_index_map(geom: RibGeometry, grid, lam=1.55, diel: RibDielectric | None = None,
                   zoom=None, fname=None):
    """2x2 check figure of the smoothed dielectric at ``lam`` with the geometry outline.

    (a) n_xx = sqrt(ε_xx), the index seen by TE (crystal Z), full window
    (b) n_yy = sqrt(ε_yy), the index seen by TM (crystal X), zoomed on the core region
    (c) zoom on the right sidewall, pixel by pixel: shows the sub-pixel (Kottke) smoothing
    (d) line cuts through the ridge: horizontal at mid-ridge height and vertical at x = 0
    """

    diel = diel or RibDielectric(geom, grid)
    d = diel.at(lam)
    n_xx = np.sqrt(d["eps"][0, 0])
    n_yy = np.sqrt(d["eps"][1, 1])
    xs, ys = grid_coords(grid)
    xe, ye = _pixel_edges(grid)
    n_e, n_o = ln_indices(lam)
    vmin, vmax = 1.0 if not geom.fully_clad else sio2_index(lam) - 0.05, n_o + 0.02

    fig = plt.figure(figsize=(12, 10.5), layout="constrained")
    gs = fig.add_gridspec(2, 2)
    # (a) full window: where the waveguide sits in the window
    # (b) core region: TM index map with the outline
    half_x = 0.5 * max(geom.w_base, geom.w_top) + 0.6
    core = ((-half_x, half_x), (geom.y0 - 0.5, geom.y_top + 0.5))
    for cell, n, title, lims in (
            (gs[0, 0], n_xx, f"(a) $n_{{xx}}$ (TE, crystal Z), full window: LN n_e = {n_e:.4f}", None),
            (gs[0, 1], n_yy, f"(b) $n_{{yy}}$ (TM, crystal X), core: LN n_o = {n_o:.4f}", core)):
        ax = fig.add_subplot(cell)
        im = ax.pcolormesh(xe, ye, n.T, cmap="Greys", vmin=vmin, vmax=vmax, shading="flat")
        draw_outline(ax, geom, grid)
        ax.set(aspect="equal", xlabel="x [µm]", ylabel="y [µm]", title=title)
        if lims:
            ax.set(xlim=lims[0], ylim=lims[1])
        fig.colorbar(im, ax=ax, shrink=0.85, label="index")

    # (c) zoom on the right sidewall, pixel edges visible
    ax = fig.add_subplot(gs[1, 0])
    if zoom is None:
        xc = 0.25 * (geom.w_top + geom.w_base) if geom.etch_depth > 0 else 0.0
        yc = 0.5 * (geom.y0 + geom.y_top)
        half = max(0.6 * geom.t_ln, 8 * grid.Lx / grid.Nx)
        zoom = ((xc - half, xc + half), (yc - half, yc + half))
    im = ax.pcolormesh(xe, ye, n_xx.T, cmap="Greys", vmin=vmin, vmax=vmax, shading="flat",
                       edgecolors="#c3c2b7", linewidth=0.2)
    draw_outline(ax, geom, grid, lw=1.5)
    ax.set(xlim=zoom[0], ylim=zoom[1], aspect="equal", xlabel="x [µm]", ylabel="y [µm]",
           title=f"(c) sidewall zoom, $n_{{xx}}$ per pixel (dx = {1e3 * grid.Lx / grid.Nx:.1f}, "
                 f"dy = {1e3 * grid.Ly / grid.Ny:.1f} nm)")
    fig.colorbar(im, ax=ax, shrink=0.85, label="index")

    # (d) line cuts, horizontal and vertical, stacked (one x-axis each)
    sub = gs[1, 1].subgridspec(2, 1)
    ax_h, ax_v = fig.add_subplot(sub[0]), fig.add_subplot(sub[1])
    y_mid = 0.5 * (geom.y_slab + geom.y_top) if geom.etch_depth > 0 else 0.5 * (geom.y0 + geom.y_top)
    jy = int(np.argmin(np.abs(ys - y_mid)))
    ix = int(np.argmin(np.abs(xs)))
    for a, coord, cut_xx, cut_yy, lab, bounds in (
            (ax_h, xs, n_xx[:, jy], n_yy[:, jy], f"x [µm]  (cut at y = {ys[jy]:+.3f} µm)",
             [-0.5 * geom.w_top, 0.5 * geom.w_top, -0.5 * geom.w_base, 0.5 * geom.w_base]
             if geom.etch_depth > 0 else []),
            (ax_v, ys, n_xx[ix, :], n_yy[ix, :], f"y [µm]  (cut at x = {xs[ix]:+.3f} µm)",
             [geom.y0, geom.y_slab, geom.y_top] +
             ([geom.y_clad_top] if not geom.fully_clad and geom.t_clad > 0 else []))):
        a.plot(coord, cut_xx, color="#2a78d6", lw=2, label="$n_{xx}$ (TE)")
        a.plot(coord, cut_yy, color="#eb6834", lw=2, ls="--", label="$n_{yy}$ (TM)")
        for b in bounds:
            a.axvline(b, color="#52514e", lw=0.8, ls=":")
        a.set(xlabel=lab, ylabel="index")
        a.grid(alpha=0.25)
    ax_h.set_title("(d) line cuts; dotted = geometry boundaries")
    ax_h.legend(loc="upper right", fontsize=8, frameon=False)

    import textwrap
    fig.suptitle(f"λ = {lam:.3f} µm   window {grid.Lx:g} × {grid.Ly:g} µm, {grid.Nx} × {grid.Ny} px\n"
                 + textwrap.fill(geom.describe(), 150), fontsize=9)
    if fname:
        fig.savefig(fname, dpi=150)
    return fig
