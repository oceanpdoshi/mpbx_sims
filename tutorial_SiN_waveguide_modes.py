#%%
import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp
from mpbjax import grid_2d, kottke as K, modes as Mo, shapes as Shp, analysis as An
from mpbjax.helmholtz import sliceinv_3x3
from mpbjax.materials import Si3N4, SiO2, mat_vals

lam = 1.55
omega = 1 / lam
grid = grid_2d(4.0, 3.0, 96, 72)                 # 4 x 3 um window

core = Shp.Cuboid.make([0.0, 0.0], [1.60, 0.80])  # 1.6 x 0.8 um Si3N4 core
mv   = np.asarray(mat_vals([Si3N4, SiO2], omega))
sm   = np.asarray(K.smooth_eps((core,), mv, (0, 1), grid))

einv = np.asarray(sliceinv_3x3(jnp.asarray(sm[:, :, 0])))
deps, d2eps = sm[:, :, 1], sm[:, :, 2]

print(f"n(Si3N4) = {np.sqrt(mv[0, 0]):.5f},  n(SiO2) = {np.sqrt(mv[0, 1]):.5f}")

#%%
kmags, evecs = Mo.solve_k(omega, einv, grid, nev=3)
for i, k in enumerate(kmags):
    print(f"mode {i}:  n_eff = {float(k) / omega:.6f}")

#%%
k0 = float(kmags[0])
ng = float(An.group_index(k0, evecs[0], omega, einv, deps, grid))
E = Mo.e_field(k0, evecs[0], einv, deps, grid)
aeff = float(An.effective_area(k0 / omega, ng, E))

print(f"n_eff = {k0 / omega:.6f}")
print(f"n_g   = {ng:.6f}")
print(f"A_eff = {aeff:.4f} um^2")

#%%
x = (np.arange(grid.Nx) - grid.Nx / 2) * (grid.Lx / grid.Nx)
y = (np.arange(grid.Ny) - grid.Ny / 2) * (grid.Ly / grid.Ny)
extent = (x[0], x[-1], y[0], y[-1])

E = np.asarray(E)
mag = np.sqrt(np.sum(np.abs(E) ** 2, axis=0))
dominant = int(np.argmax([np.max(np.abs(E[c])) for c in range(3)]))
comp = np.real(E[dominant])

fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.4))
for ax, data, cmap, title in (
        (axes[0], mag, "magma", "|E|"),
        (axes[1], comp, "RdBu_r", f"Re E_{'xyz'[dominant]}")):
    lim = np.max(np.abs(data))
    kw = dict(vmin=-lim, vmax=lim) if cmap == "RdBu_r" else {}
    im = ax.imshow(data.T, origin="lower", extent=extent, cmap=cmap, **kw)
    ax.add_patch(plt.Rectangle((-0.8, -0.4), 1.6, 0.8, fill=False,
                               ec="w", lw=1.1, ls="--"))
    ax.set_xlabel("x [um]"); ax.set_title(title)
    fig.colorbar(im, ax=ax, shrink=0.85)
axes[0].set_ylabel("y [um]")
plt.tight_layout(); plt.show()

#%%
widths = np.linspace(0.9, 2.1, 9)
rows = []
for w in widths:
    smw = np.asarray(K.smooth_eps((Shp.Cuboid.make([0.0, 0.0], [float(w), 0.80]),),
                                  mv, (0, 1), grid))
    ei = np.asarray(sliceinv_3x3(jnp.asarray(smw[:, :, 0])))
    dw = smw[:, :, 1]
    km, ev = Mo.solve_k(omega, ei, grid, nev=1)
    k = float(km[0])
    rows.append((w, k / omega, float(An.group_index(k, ev[0], omega, ei, dw, grid))))

rows = np.array(rows)
fig, ax = plt.subplots(figsize=(5.2, 3.4))
ax.plot(rows[:, 0], rows[:, 1], "o-", color="#1f4e79", label="n_eff")
ax2 = ax.twinx()
ax2.plot(rows[:, 0], rows[:, 2], "s--", color="#a2364f", label="n_g")
ax.set_xlabel("core width [um]"); ax.set_ylabel("n_eff", color="#1f4e79")
ax2.set_ylabel("n_g", color="#a2364f")
ax.set_title("Si3N4 strip, 0.8 um thick, lambda = 1.55 um")
plt.tight_layout(); plt.show()

for w, neff, g in rows:
    print(f"  w = {w:.2f} um:  n_eff = {neff:.5f}   n_g = {g:.5f}")