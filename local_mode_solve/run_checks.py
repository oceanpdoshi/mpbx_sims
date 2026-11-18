"""Fast self-checks of the TFLN mode-solve code (coarse grids, ~2-3 min on CPU).

    python run_checks.py

1. material frame: lab x = n_e, lab y/z = n_o; R_XCUT gives the same ε as the mpbx-example Ry
2. geometry: LN area carried by the smoothed ε matches the analytic trapezoid (to < 1 pixel)
3. the jit-compiled Newton driver reproduces mpbjax.modes.solve_k
4. group index vs finite difference  n_g = n_eff - λ dn_eff/dλ  (ε re-evaluated at λ ± δ)
5. fields are normalized to unit power, ∫ Re(E* × H)_z dA = 1
6. ModeSweep HDF5 save / load round trip
7. an unetched film (exact answer: 3-layer slab) at the production pitch/alignment
"""
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                        # utils/, geometry_gen/ (HERE: tfln_modes, tfln_plots)
sys.path[:0] = [p for p in (ROOT, HERE) if p not in sys.path]
from utils.setup_jax import setup_jax

setup_jax("cpu")

import numpy as np
from mpbjax import modes as Mo

import geometry_gen.tfln_rib as R
import sim_core.tfln_modes as M

ok_all = True


def check(name, ok, detail=""):
    global ok_all
    ok_all &= bool(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}  {detail}")


# 1 -------------------------------------------------------------------------------------
for lam in (0.4, 1.55, 2.0):
    n = R.check_material_frame(lam, verbose=False)
    n_e, n_o = R.ln_indices(lam)
    check(f"material frame at {lam} µm", np.allclose(n, [n_e, n_o, n_o]), f"n_lab = {np.round(n, 6)}")
# Matsuda 2015 Sellmeier written out (the Tidy3D notebook's formula)
lam = 1.55
no = np.sqrt(4.8898 + 0.11738 / (lam ** 2 - 0.04564) - 0.02662 * lam ** 2)
ne = np.sqrt(4.54514 + 0.096471 / (lam ** 2 - 0.043763) - 0.021502 * lam ** 2)
check("Matsuda Sellmeier matches mpbjax MgO_LiNbO3_matsuda", np.allclose(R.ln_indices(lam), (ne, no), atol=1e-12),
      f"(n_e, n_o) = ({ne:.6f}, {no:.6f})")

# 2 -------------------------------------------------------------------------------------
for geom in (R.RibGeometry(), R.RibGeometry(t_ln=0.6, etch_depth=0.3, w_top=1.2, sidewall_deg=65, t_clad=0.0)):
    grid = R.make_grid(6.0, 6.0, 0.025)
    d = R.RibDielectric(geom, grid).at(1.0)
    n_e, n_o = R.ln_indices(1.0)
    eps_bg = R.sio2_index(1.0) ** 2
    # ε_zz is tangential to every interface, so it is the arithmetic (volume) average
    X, Y = np.meshgrid(*R.grid_coords(grid), indexing="ij")
    below = Y < geom.y_slab + 1e-9 if geom.t_slab > 0 else np.ones_like(X, bool)
    win_slab_area = geom.t_slab * grid.Lx
    if geom.fully_clad:
        frac = (d["eps"][2, 2] - eps_bg) / (n_o ** 2 - eps_bg)
        area = frac.sum() * grid.dV
        exact = 0.5 * (geom.w_top + geom.w_base) * geom.etch_depth + win_slab_area
        check("LN area of the smoothed ε (fully clad rib)", abs(area - exact) < 2 * grid.dV,
              f"{area:.6f} vs {exact:.6f} µm² (pixel = {grid.dV:.6f})")
    else:
        # air above, oxide below: count LN as the excess over the local background
        bg = np.where(Y > geom.y0 + 1e-9, 1.0, eps_bg)
        frac = np.clip((d["eps"][2, 2] - bg) / (n_o ** 2 - bg), 0, 1)
        area = frac.sum() * grid.dV
        exact = 0.5 * (geom.w_top + geom.w_base) * geom.etch_depth + win_slab_area
        check("LN area of the smoothed ε (partial etch, air clad)", abs(area - exact) < 0.01 * exact,
              f"{area:.6f} vs {exact:.6f} µm² (interface pixels mix 3 materials, 1% tolerance)")

# 3 -------------------------------------------------------------------------------------
geom = R.RibGeometry()
grid = R.make_grid(6.0, 6.0, 0.075)                     # 80 x 80: coarse, only for these checks
diel = R.RibDielectric(geom, grid)
lam = 1.3
d = diel.at(lam)
om = 1 / lam
k_ref, ev_ref = Mo.solve_k(om, d["einv"], grid, nev=4)
k_new, ev_new, diag = M.solve_k_checked(om, d["einv"], grid, 4)
check("jit Newton driver == mpbjax.modes.solve_k", np.allclose(k_new, k_ref, rtol=0, atol=1e-10),
      f"max |Δneff| = {np.abs(k_new - k_ref).max() / om:.1e}; converged = {all(x['lobpcg_converged'] for x in diag)}")

# 4 -------------------------------------------------------------------------------------
settings = M.SolverSettings(nev=4, label_modes=False, store_fields=True, warm_start=False)
sw0, _ = M.solve_wavelength(diel, lam, settings)
dl = 2e-3
sw_p, _ = M.solve_wavelength(diel, lam + dl, settings)
sw_m, _ = M.solve_wavelength(diel, lam - dl, settings)
for j in range(2):
    ng_fd = sw0.neff[0, j] - lam * (sw_p.neff[0, j] - sw_m.neff[0, j]) / (2 * dl)
    check(f"group index vs finite difference (mode {j})", abs(ng_fd - sw0.ng[0, j]) < 2e-4,
          f"ng = {sw0.ng[0, j]:.6f}, FD = {ng_fd:.6f}")

# 5 -------------------------------------------------------------------------------------
for j in range(2):
    E = sw0.E[0, j].astype(complex)
    k = sw0.neff[0, j] * om
    ev = ev_new[j]
    H = np.asarray(Mo.h_field(k, ev, om, d["einv"], d["nng"], grid))
    E_raw = np.asarray(Mo.e_field(k, ev, d["einv"], d["nng"], grid))
    s = np.vdot(E_raw.ravel(), E.ravel()) / np.vdot(E_raw.ravel(), E_raw.ravel())   # E = s * E_raw
    Hs = H * s
    P = np.real(np.conj(E[0]) * Hs[1] - np.conj(E[1]) * Hs[0]).sum() * grid.dV
    check(f"unit-power normalization (mode {j})", abs(P - 1) < 1e-5, f"∫Sz dA = {P:.7f}")

# 7 -------------------------------------------------------------------------------------
# An unetched film is a 1-D slab with an exact answer (mpbjax.bend.slab_neff): the n_eff
# error at the production vertical pitch (12.5 nm) and alignment (1/3 pixel) should be C·dy²
# with |C| <~ 1 µm⁻² (see README section 3).
from mpbjax.bend import slab_neff
film = R.RibGeometry(etch_depth=0.0)
for lam in (0.4, 1.0, 2.0):
    g_film = R.make_grid(0.5, 6.0, 0.125, 0.0125)          # thin periodic strip: x-invariant
    f_al = R.align_to_grid(film, g_film, frac=1 / 3)
    d_f = R.RibDielectric(f_al, g_film).at(lam)
    km_f, ev_f, _ = M.solve_k_checked(1 / lam, d_f["einv"], g_film, 2, k_tol=1e-10, eig_tol=1e-10)
    n_e, n_o = R.ln_indices(lam)
    n_s = R.sio2_index(lam)
    exact = {"TE": float(slab_neff(n_e, n_s, n_s, film.t_ln, 1 / lam, pol="TE")),
             "TM": float(slab_neff(n_o, n_s, n_s, film.t_ln, 1 / lam, pol="TM"))}
    dyf = g_film.Ly / g_film.Ny
    for k, ev in zip(km_f, ev_f):
        Ef = np.asarray(Mo.e_field(k, ev, d_f["einv"], d_f["nng"], g_film))
        pol = "TE" if (abs(Ef[0]) ** 2).sum() > (abs(Ef[1]) ** 2).sum() else "TM"
        err = k * lam - exact[pol]
        check(f"flat film vs analytic slab, {pol} at {lam} µm", abs(err) < 1.2 * dyf ** 2,
              f"n_eff {k * lam:.6f} vs {exact[pol]:.6f}: error {err:+.1e} = {err / dyf ** 2:+.2f}·dy²")

# 6 -------------------------------------------------------------------------------------
with tempfile.TemporaryDirectory() as tmp:
    path = sw0.save(os.path.join(tmp, "t.h5"))
    back = M.ModeSweep.load(path)
    same = all(np.array_equal(back.data[k], sw0.data[k], equal_nan=True) if back.data[k].dtype.kind == "f"
               else np.array_equal(back.data[k], sw0.data[k]) for k in sw0.data)
    check("HDF5 round trip", same and np.array_equal(back.E, sw0.E) and back.geometry == geom
          and back.grid == grid, f"{os.path.getsize(path) / 1e6:.2f} MB")

print("\nALL CHECKS PASSED" if ok_all else "\nSOME CHECKS FAILED")
sys.exit(0 if ok_all else 1)
