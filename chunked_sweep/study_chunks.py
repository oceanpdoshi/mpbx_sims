"""Measurements behind the chunked-sweep recommendation (README.md in this folder).
Nothing in sim_core/ is modified; its solver (sim_core/tfln_modes) and the geometry
(geometry_gen/tfln_rib) are reused as they are.

    python study_chunks.py census            mode ordering / nev needed vs λ (coarse grid)
    python study_chunks.py window LAM ...    Lx and Ly window convergence at LAM [µm]
    python study_chunks.py pitch LAM ...     dy (and dx) convergence at LAM, 1/3-pixel alignment

Every solve uses the x-mirror parity sectors (odd Nx), so each target mode's rank WITHIN
its sector is reported:  sector -1 = TE00, TM10, TE20, TM30, ...
                         sector +1 = TM00, TE10, TM20, TE30, ...
Results are appended (one JSON line per solve) to results/<study>_<lam>.jsonl.
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [ROOT]
from utils.setup_jax import setup_jax  # noqa: E402

setup_jax("cpu")

import numpy as np  # noqa: E402

from geometry_gen import tfln_rib as R  # noqa: E402
import sim_core.tfln_modes as M  # noqa: E402

GEOM = R.RibGeometry(t_ln=0.400, etch_depth=0.400, w_top=1.800, sidewall_deg=73.0, t_clad=np.inf)
TARGETS = ("TE00", "TE10", "TE20", "TE30", "TM00", "TM10", "TM20", "TM30")
ALIGN = 1 / 3
OUT = os.path.join(HERE, "results")
os.makedirs(OUT, exist_ok=True)


from mpbjax import grid_2d  # noqa: E402


def make(Nx, Ny, dx, dy):
    assert Nx % 2 == 1 and Ny % 2 == 0, (Nx, Ny)
    return grid_2d(float(Nx * dx), float(Ny * dy), int(Nx), int(Ny))


# ---------------------------------------------------------------------------------------
# fast mode label: polarization from te_frac, (m, n) from sign changes of the dominant
# transverse component along a horizontal and a vertical line through the mode
# ---------------------------------------------------------------------------------------
def _sign_changes(line, thr=0.03):
    a = np.abs(line)
    keep = line[a > thr * a.max()]
    s = np.sign(keep)
    return int(np.sum(s[1:] != s[:-1]))


def node_label(E, te):
    F = np.asarray(E[0] if te >= 0.5 else E[1])
    i0 = np.unravel_index(np.argmax(np.abs(F)), F.shape)
    F = np.real(F * np.exp(-1j * np.angle(F[i0])))
    I = F ** 2
    jy = int(np.argmax(I.sum(axis=0)))            # row through the film (max of ∫|F|² dx)
    m = _sign_changes(F[:, jy])
    n = _sign_changes(F[i0[0], :])
    pol = "TE" if te >= 0.5 else "TM"
    return f"{pol}{m}{n}"


def solve(geom, grid, lam, nev, tag, extra=None, fname=None):
    """Both parity sectors with ``nev`` bands each. Returns a JSON-able record. A point already
    in ``fname`` (same λ and pixel counts) is not solved again (restartable studies)."""
    if fname and os.path.exists(os.path.join(OUT, fname)):
        for line in open(os.path.join(OUT, fname), encoding="utf-8"):
            r = json.loads(line)
            if abs(r["lam"] - lam) < 1e-9 and r["Nx"] == grid.Nx and r["Ny"] == grid.Ny:
                print(f"[{tag}] λ={lam:.3f} {grid.Nx}x{grid.Ny}: already done", flush=True)
                return r
    geom = R.align_to_grid(geom, grid, frac=ALIGN, warn=False)
    t0 = time.perf_counter()
    diel = R.RibDielectric(geom, grid)
    t_plan = time.perf_counter() - t0
    st = M.SolverSettings(nev=nev, use_parity=True, label_modes=False, store_fields=True,
                          warm_start=False, field_dtype="complex64")
    t1 = time.perf_counter()
    sw, _ = M.solve_wavelength(diel, lam, st)
    t_solve = time.perf_counter() - t1
    g = sw.data
    modes = []
    rank_in_sector = {-1: 0, +1: 0}
    for j in range(sw.n_modes):
        te = float(g["te_frac"][0, j])
        par = int(g["parity"][0, j])
        lab = node_label(sw.E[0, j], te)
        modes.append(dict(rank=j, sector=par, rank_in_sector=rank_in_sector[par], label=lab,
                          neff=float(g["neff"][0, j]), ng=float(g["ng"][0, j]), te_frac=te,
                          edge_frac=float(g["edge_frac"][0, j]), edge_db=float(g["edge_db"][0, j]),
                          guided=bool(g["guided"][0, j]), converged=bool(g["converged"][0, j]),
                          lobpcg_iters=int(g["lobpcg_iters"][0, j]), newton=int(g["newton_iters"][0, j])))
        rank_in_sector[par] += 1
    rec = dict(tag=tag, lam=lam, nev_per_sector=nev, Lx=grid.Lx, Ly=grid.Ly, Nx=grid.Nx, Ny=grid.Ny,
               dx=grid.Lx / grid.Nx, dy=grid.Ly / grid.Ny, n_cutoff=float(sw.per_lam["n_cutoff"][0]),
               t_plan=t_plan, t_solve=t_solve, modes=modes, **(extra or {}))
    found = {t: [m["rank_in_sector"] for m in modes if m["label"] == t and m["guided"]]
             for t in TARGETS}
    rec["found"] = found
    if fname:
        with open(os.path.join(OUT, fname), "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
    print(f"[{tag}] λ={lam:.3f} {grid.Nx}x{grid.Ny} ({grid.Lx:.3f}x{grid.Ly:.3f} µm, dx {1e3 * rec['dx']:.2f} "
          f"dy {1e3 * rec['dy']:.2f} nm) nev/sector={nev}: plan {t_plan:.1f} s, solve {t_solve:.0f} s; "
          + " ".join(f"{t}:{v[0] if v else '-'}" for t, v in found.items()), flush=True)
    return rec


# ---------------------------------------------------------------------------------------
def census():
    """Coarse 40 nm grid, 6 x 6 µm, many bands: which rank each target has in its sector."""
    grid = make(151, 150, 0.04, 0.04)
    # at 0.4 µm the deepest target (TE30) is rank 5 in its sector: 8/sector leaves margin,
    # and the mode count only shrinks with λ
    plan = [(0.40, 16), (0.50, 8), (0.60, 7), (0.70, 7), (0.80, 6), (0.90, 6), (1.00, 6),
            (1.25, 5), (1.50, 5), (2.00, 4)]
    for lam, nev in plan:
        solve(GEOM, grid, lam, nev, "census", fname="census.jsonl")


WINDOW_NEV = {0.6: 6, 1.0: 5, 2.0: 3}


def window(lam):
    """Separate Lx and Ly sweeps around a 9 x 9 µm reference, 40 nm pitch (window error
    does not depend on the pitch to leading order)."""
    nev = WINDOW_NEV.get(lam, 8)
    d = 0.04
    NX_REF, NY_REF = 225, 224
    fname = f"window_{lam:.2f}.jsonl"
    solve(GEOM, make(NX_REF, NY_REF, d, d), lam, nev, "window-ref", dict(sweep="ref"), fname)
    for Nx in (101, 125, 151, 175, 201):
        solve(GEOM, make(Nx, NY_REF, d, d), lam, nev, "window-x", dict(sweep="x"), fname)
    for Ny in (50, 76, 100, 126, 150, 200):
        solve(GEOM, make(NX_REF, Ny, d, d), lam, nev, "window-y", dict(sweep="y"), fname)


# window (Nx at dx = 25 nm, Ly) and nev per sector for the pitch study at each λ
PITCH_SETUP = {0.4: (165, 3.0, 8, (0.025, 0.0125, 0.025 / 3, 0.00625), True),
               0.6: (165, 3.0, 7, (0.025, 0.0125, 0.025 / 3, 0.00625), True),
               1.0: (225, 4.0, 7, (0.025, 0.0125, 0.025 / 3), False),
               2.0: (243, 6.0, 3, (0.025, 0.0125, 0.025 / 3), False)}


def pitch(lam):
    Nx, Ly, nev, dys, dx_check = PITCH_SETUP[lam]
    fname = f"pitch_{lam:.2f}.jsonl"
    for dy in dys:
        Ny = int(round(Ly / dy))
        Ny += Ny % 2
        solve(GEOM, make(Nx, Ny, 0.025, Ly / Ny), lam, nev, "pitch", dict(sweep="dy"), fname)
    if dx_check:                      # dx 25 -> 12.47 nm at dy = 12.5 nm
        Ny = int(round(Ly / 0.0125))
        solve(GEOM, make(2 * Nx + 1, Ny, Nx * 0.025 / (2 * Nx + 1), Ly / Ny), lam, nev, "pitch-dx",
              dict(sweep="dx"), fname)


def pitchx(lam):
    """The dy series again at dx ≈ 12.5 nm (same window): does the irregular convergence of
    the m = 2, 3 modes come from the coarse lateral sampling of the slanted sidewalls?"""
    Nx, Ly, nev, dys, _ = PITCH_SETUP[lam]
    Nx2 = 2 * Nx + 1
    dx2 = Nx * 0.025 / Nx2
    fname = f"pitch_{lam:.2f}.jsonl"
    for dy in dys:
        Ny = int(round(Ly / dy))
        Ny += Ny % 2
        solve(GEOM, make(Nx2, Ny, dx2, Ly / Ny), lam, nev, "pitch-dx", dict(sweep="dx"), fname)


def dx3(lam):
    """Third lateral pitch (dx = 25/3 nm) at dy = 12.5 nm: with the 25 and 12.47 nm runs this is
    a dx series at fixed dy, i.e. the order and size of the lateral (sidewall) error."""
    Nx, Ly, nev, _, _ = PITCH_SETUP[lam]
    Ny = int(round(Ly / 0.0125))
    solve(GEOM, make(3 * Nx, Ny, Nx * 0.025 / (3 * Nx), Ly / Ny), lam, nev, "pitch-dx3",
          dict(sweep="dx3"), f"pitch_{lam:.2f}.jsonl")


if __name__ == "__main__":
    what = sys.argv[1]
    lams = [float(x) for x in sys.argv[2:]]
    if what == "census":
        census()
    elif what == "window":
        for lam in lams:
            window(lam)
    elif what == "pitch":
        for lam in lams:
            pitch(lam)
    elif what == "pitchx":
        for lam in lams:
            pitchx(lam)
    elif what == "dx3":
        for lam in lams:
            dx3(lam)
    else:
        raise SystemExit(__doc__)
