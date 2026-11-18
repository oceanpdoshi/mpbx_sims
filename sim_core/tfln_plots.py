"""Plots for a ModeSweep: n_eff / n_g vs wavelength and the E-fields of every mode at a
chosen wavelength."""
from __future__ import annotations

import textwrap

import numpy as np
import matplotlib.pyplot as plt

from geometry_gen.tfln_rib import draw_outline

# Categorical slots in a fixed order (validated light-mode palette); index 8+ folds to grey.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
OTHER = "#8a8984"
INK = "#52514e"


def _style_axes(ax, ylabel):
    ax.set_xlabel("wavelength λ [µm]")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25, lw=0.6)


def _quantity(sweep, quantity):
    names = {"neff": r"$n_\mathrm{eff}$", "ng": r"$n_g$"}
    return np.asarray(sweep.data[quantity], float), names.get(quantity, quantity)


EDGE_MAX = 1e-3    # modes with > 0.1 % of |E|² near the window edge are treated as clipped


def _reliable(sweep, edge_max=EDGE_MAX):
    """Guided AND not clipped by the window."""
    return sweep.data["guided"].astype(bool) & (sweep.data["edge_frac"] <= edge_max)


def plot_by_index(sweep, quantity="neff", ax=None, show_cutoff=True, edge_max=EDGE_MAX):
    """Tidy3D-style: one line per solver mode index (modes ranked by n_eff at each λ).
    Points that are unguided (n_eff below the cutoff) or clipped by the window
    (edge_frac > edge_max) are drawn hollow."""
    ax = ax or plt.gca()
    Q, ylabel = _quantity(sweep, quantity)
    guided = _reliable(sweep, edge_max)
    for j in range(sweep.n_modes):
        c = SERIES[j] if j < len(SERIES) else OTHER
        ax.plot(sweep.lam, Q[:, j], color=c, lw=1.5, label=f"mode {j}")
        ax.plot(sweep.lam[guided[:, j]], Q[guided[:, j], j], "o", ms=5, color=c)
        ax.plot(sweep.lam[~guided[:, j]], Q[~guided[:, j], j], "o", ms=5, mfc="none", color=c)
    if show_cutoff and quantity == "neff":
        ax.plot(sweep.lam, sweep.per_lam["n_cutoff"], color=INK, lw=1, ls=":", label="cutoff")
    _style_axes(ax, ylabel)
    ax.set_title(f"{ylabel} by mode index (hollow = not guided or clipped by the window)")
    ax.legend(fontsize=8, loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
    return ax


def plot_by_label(sweep, quantity="neff", track=None, ax=None, err_max=0.35, show_cutoff=True,
                  edge_max=EDGE_MAX):
    """One series per Hermite-Gaussian label (TE00, TM00, TE10, ...). TE = circles/solid,
    TM = triangles/dashed; color = spatial order (TE00 and TM00 share the first color).
    Unguided or window-clipped modes, and modes with a poor HG fit (hg_err > err_max), are
    grey crosses. ``track`` (a ModeTrack) is a thick grey band under its branch."""
    ax = ax or plt.gca()
    Q, ylabel = _quantity(sweep, quantity)
    labels = sweep.labels
    ok = _reliable(sweep, edge_max) & (np.nan_to_num(sweep.data["hg_err"], nan=0.0) <= err_max) \
        & (labels != "")
    # Color = spatial order (m, n), shared by TE and TM (marker / line style give the
    # polarization), assigned lowest order first: TE00 and TM00 always get the first color.
    # Orders beyond the 8 categorical slots are drawn grey but keep their own legend entry.
    uniq = sorted({l for l in labels[ok]}, key=lambda l: (int(l[2]) + int(l[3]), int(l[3]), l[:2]))
    orders = []
    for l in uniq:
        if l[2:] not in orders:
            orders.append(l[2:])
    for lab in uniq:
        k = orders.index(lab[2:])
        c = SERIES[k] if k < len(SERIES) else OTHER
        te = lab.startswith("TE")
        mk, ls = ("o", "-") if te else ("^", "--")
        sel = ok & (labels == lab)
        lam_l, q_l, lam_d, q_d = [], [], [], []
        for i in range(sweep.n_lam):
            js = np.flatnonzero(sel[i])
            if js.size:                      # if a label repeats at one λ, line through the best fit
                j = js[np.argmin(np.nan_to_num(sweep.data["hg_err"][i, js], nan=0.0))]
                lam_l.append(sweep.lam[i])
                q_l.append(Q[i, j])
                for jj in js[js != j]:       # ...and draw the other same-label modes hollow
                    lam_d.append(sweep.lam[i])
                    q_d.append(Q[i, jj])
        ax.plot(lam_l, q_l, ls=ls, marker=mk, ms=5, lw=1.3, color=c, label=lab)
        if lam_d:
            ax.plot(lam_d, q_d, ls="none", marker=mk, ms=7, mfc="none", color=c,
                    label=f"{lab} (repeated label)")
    bad = ~ok
    if bad.any():
        ax.plot(np.broadcast_to(sweep.lam[:, None], Q.shape)[bad], Q[bad], "x", ms=4, color=OTHER,
                label="unguided / clipped / unlabeled")
    if track is not None:
        sel = track.idx >= 0
        q_t = track.neff if quantity == "neff" else track.ng if quantity == "ng" else None
        if q_t is not None:
            ax.plot(track.lam[sel], q_t[sel], color="#0b0b0b", lw=3.0, alpha=0.45, zorder=0,
                    label=f"{track.target} tracked")
    if show_cutoff and quantity == "neff":
        ax.plot(sweep.lam, sweep.per_lam["n_cutoff"], color=INK, lw=1, ls=":", label="cutoff")
    _style_axes(ax, ylabel)
    ax.set_title(f"{ylabel} by Hermite-Gaussian label")
    ax.legend(fontsize=8, loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
    return ax


def plot_dispersion(sweep, track=None, fname=None, quantities=("neff", "ng")):
    """Rows = quantities (n_eff, n_g); columns = by mode index | by HG label (+ track)."""
    fig, axes = plt.subplots(len(quantities), 2, figsize=(14, 4.6 * len(quantities)),
                             constrained_layout=True, squeeze=False)
    for r, q in enumerate(quantities):
        plot_by_index(sweep, q, ax=axes[r, 0])
        plot_by_label(sweep, q, track=track, ax=axes[r, 1])
    g = sweep.meta.get("grid", {})
    fig.suptitle(textwrap.fill(sweep.geometry.describe(), 150) + f"\nwindow {g.get('Lx')} x {g.get('Ly')} µm, "
                 f"{g.get('Nx')} x {g.get('Ny')} px", fontsize=9)
    if fname:
        fig.savefig(fname, dpi=150)
    return fig


def plot_mode_fields(sweep, lam, modes=None, highlight=None, fname=None, xlim="core", ylim="core",
                     db_floor=-40.0):
    """E-fields of the modes stored at the wavelength nearest ``lam``.

    modes     : None -> every mode solved at that wavelength; an int or list -> just those.
    highlight : mode index (or list) whose row titles are marked with ">>".
    xlim/ylim : "core" (default) zooms the three linear panels to the waveguide ± 1.5 µm;
                None = full window; or (lo, hi). The dB panel always shows the full window,
                so you can see how far the field reaches toward the edges (clipping).
    Columns: Re(Ex) [TE, crystal Z], Re(Ey) [TM, crystal X], |E|²/max, |E|² [dB].
    Re(Ex) and Re(Ey) share one color scale per mode, so the minor polarization looks faint.
    """
    if sweep.E is None:
        raise ValueError("this sweep has no stored fields (SolverSettings(store_fields=True))")
    i = sweep.index_of(lam)
    modes = list(range(sweep.n_modes)) if modes is None else list(np.atleast_1d(modes))
    highlight = set(np.atleast_1d(highlight).tolist()) if highlight is not None else set()
    geom, grid = sweep.geometry, sweep.grid
    xs, ys = sweep.xs, sweep.ys
    dx, dy = xs[1] - xs[0], ys[1] - ys[0]
    xe = np.append(xs - dx / 2, xs[-1] + dx / 2)
    ye = np.append(ys - dy / 2, ys[-1] + dy / 2)
    labels = sweep.labels
    g = sweep.data
    half_w = 0.5 * max(geom.w_base, geom.w_top) + 1.5
    if isinstance(xlim, str):
        xlim = (max(-half_w, xe[0]), min(half_w, xe[-1]))
    if isinstance(ylim, str):
        ylim = (max(geom.y0 - 1.5, ye[0]), min(geom.y_top + 1.5, ye[-1]))

    nr = len(modes)
    fig = plt.figure(figsize=(17, 3.0 * nr), layout="constrained")
    rows = np.atleast_1d(fig.subfigures(nr, 1))          # one sub-figure (with its own title) per mode
    axes = np.array([sf.subplots(1, 4) for sf in rows]).reshape(nr, 4)
    for r, j in enumerate(modes):
        E = sweep.E[i, j].astype(complex)
        I2 = np.sum(np.abs(E) ** 2, axis=0)
        I2n = I2 / I2.max()
        lim = max(np.abs(E[0]).max(), np.abs(E[1]).max())
        panels = (
            (np.real(E[0]) / lim, "RdBu_r", (-1, 1), "Re $E_x$ (TE, crystal Z)", INK),
            (np.real(E[1]) / lim, "RdBu_r", (-1, 1), "Re $E_y$ (TM, crystal X)", INK),
            (I2n, "Blues", (0, 1), r"$|E|^2$ / max", "#eb6834"),
            (10 * np.log10(I2n + 10 ** (db_floor / 10)), "Blues", (db_floor, 0), r"$|E|^2$ [dB]", "#eb6834"),
        )
        mark = ">> " if j in highlight else ""
        head = (f"{mark}λ = {sweep.lam[i]:.4f} µm, mode {j}: {labels[i, j] or '?'}  "
                f"n_eff = {g['neff'][i, j]:.5f}, n_g = {g['ng'][i, j]:.4f}, "
                f"TE frac = {g['te_frac'][i, j]:.3f}, "
                f"{'guided' if g['guided'][i, j] else 'NOT guided'}, edge {100 * g['edge_frac'][i, j]:.1e} %")
        for c, (Z, cmap, (vmin, vmax), title, oc) in enumerate(panels):
            ax = axes[r, c]
            im = ax.pcolormesh(xe, ye, Z.T, cmap=cmap, vmin=vmin, vmax=vmax, shading="flat",
                               rasterized=True)
            draw_outline(ax, geom, grid, color=oc, lw=0.8)
            ax.set(aspect="equal", title=title + (" (full window)" if c == 3 else ""), xlabel="x [µm]")
            if c < 3 and xlim:
                ax.set_xlim(xlim)
            if c < 3 and ylim:
                ax.set_ylim(ylim)
            rows[r].colorbar(im, ax=ax, shrink=0.8)
        axes[r, 0].set_ylabel("y [µm]")
        rows[r].suptitle(head, fontsize=9.5, fontweight="bold" if mark else "normal", x=0.01, ha="left")
    if fname:
        fig.savefig(fname, dpi=130)
    return fig
