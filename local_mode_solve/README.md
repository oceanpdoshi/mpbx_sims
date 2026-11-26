# X-cut TFLN rib waveguide: forward mode solves with mpbx (CPU)

Staging folder for proofreading. Nothing outside this folder was changed: your
`tutorial_*.py` files and the mpbx repo are untouched. Copy in what you want.

| file | what it is |
|---|---|
| `tfln_setup.py` | `setup_jax("cpu")`: pins JAX to CPU, turns on the on-disk compile cache, checks float64. Call it **before** anything imports mpbjax. |
| `tfln_rib.py` | `RibGeometry` (LN thickness, etch depth / slab, top width, sidewall angle, SiO2 cladding thickness incl. none / fully clad), `RibGeometryUndercut` (fully etched, air-clad ridge on a SiO2 pedestal: the oxide etched `t_undercut` deeper at the sidewall angle), `make_geometry` (picks the class from `t_undercut`), Matsuda MgO:LN in the x-cut frame, Kottke-smoothed dielectric (`RibDielectric`), `make_grid`, `align_to_grid`, and `plot_index_map` (index map + geometry outline + pixel zoom + line cuts). |
| `tfln_modes.py` | Solver driver, per-mode diagnostics, `run_sweep`, `ModeSweep` (HDF5 / CSV / DataFrame), `track_mode` (TE00 tracking), `richardson`, `to_si_fields`. |
| `tfln_plots.py` | n_eff / n_g vs λ (by solver index, and by Hermite-Gaussian label with the TE00 track) and E-fields of every mode at a chosen λ. |
| `run_checks.py` | ~5 min self-test (material frame, LN area, driver vs `Mo.solve_k`, n_g vs finite difference, power normalization, HDF5 round trip, unetched film vs the exact slab answer). All pass: `results/run_checks.log`. |
| `run_convergence.py` | Window-size and grid-pitch convergence study (`results/convergence/`). `--align-check` adds the production-alignment check (D); `--replot` remakes report + figure from the JSON. |
| `run_sweep.py` | The main `#%%` script: geometry check, sweep, tracking, plots. Default = quick test 0.4–1.6 µm / 200 nm (~40 min on CPU); `--full` = 0.4–2.0 µm / 10 nm; `--batch` = files only, no windows. |
| `postprocess_sweep.py` | Load a saved sweep (one `.h5`, or a folder of per-λ files from a cluster run), optionally re-label, track, and make all figures + CSV. No solving. |
| `compare_tidy3d.py` | n_eff / n_g of the tracked TE00 / TM00 vs your Tidy3D CSV. |
| `results/` | Outputs of my runs (figures, HDF5, CSV, logs). |

Run everything from this folder with the project venv, e.g.
`..\.venv\Scripts\python.exe run_sweep.py --batch`. `results/` is ~280 MB (mostly field HDF5);
don't commit it (e.g. add `*.h5` to `.gitignore`). Nothing here has been committed.

---

## 1. Conventions (read this first)

| | mpbx lab frame (this code) | crystal | Tidy3D notebook |
|---|---|---|---|
| horizontal, in-plane (width) | **x** | **Z** (n_e) | z |
| vertical, film normal | **y** | **X** (n_o) | x |
| propagation | **z** | **Y** (n_o) | y |
| **TE** = crystal-Z polarized | E_x dominant | | E_z dominant |
| **TM** = crystal-X polarized | E_y dominant | | E_x dominant |

* Units: µm; ω = 1/λ [µm⁻¹] (no 2π); n_eff = k/ω. n_g = c/v_g **includes material
  dispersion** (verified against a finite-difference n_eff − λ·dn_eff/dλ to 1e-6).
* Material: `mpbjax.materials.MgO_LiNbO3_matsuda` (Matsuda 2015, 20 °C; identical to the
  Sellmeier in your Tidy3D notebook, checked to 1e-12), rotated with
  `R_XCUT = [[0,1,0],[0,0,1],[1,0,0]]`. mpbx's `rotate` applies ε_lab = Rᵀ ε R, so lab axis
  *i* is crystal direction R[:, i]. The `Ry` in the mpbx example and in your
  `tutorial_LN_waveguide_modes.py` gives **the identical ε tensor** (checked), but puts crystal Y
  on the film normal and crystal X along propagation. That only matters once χ(2)/EO
  tensors are used, so I used the true x-cut rotation.
* Sidewall angle: from **horizontal** (90° = vertical), as in your Tidy3D notebook (73°).
  Note that Tidy3D's own `PolySlab.sidewall_angle` is measured from vertical.
* Cladding: `t_clad = 0` means air, `t_clad = np.inf` means SiO2 everywhere (your Tidy3D
  setup), and a finite value means a **planar** SiO2 layer that fills the trenches with its
  flat top `t_clad` above the ridge top. The substrate is always SiO2.
* `te_frac` = Σ|Ex|²/(Σ|Ex|²+Σ|Ey|²). That is **1 − Tidy3D's "TE (Ex) fraction"** for your
  notebook's plane (Tidy3D's "TE" there is the vertical = TM component), i.e. your `zpol`.
* Stored E: Ex, Ey, Ez on the pixel centers, normalized to ∫Re(E*×H)_z dA = 1 in mpbx
  natural units. Tidy3D's 1 W SI fields are `to_si_fields(E)` = √(2·Z0)·E ≈ 27.45·E [V/µm].
* "Mode index" = rank by n_eff at that wavelength (0 = highest). It is **not** a mode identity:
  use the HG label or `track_mode`.

## 2. Geometry generator and the check plot

```python
geom = R.RibGeometry(t_ln=0.6, etch_depth=0.3, w_top=1.2, sidewall_deg=65, t_clad=0.8)
grid = R.make_grid(6.0, 6.0, dx=0.025, dy=0.0125)
geom = R.align_to_grid(geom, grid, frac=1/3)          # see section 3
R.plot_index_map(geom, grid, lam=1.55, fname="check.png")
```

Undercut rib (`RibGeometryUndercut`): the etch continues `t_undercut` into the SiO2 at the same
sidewall angle, leaving the fully etched LN ridge on a SiO2 pedestal with air around it (no
cladding; SiO2 fills the window below the pedestal). It requires `t_clad = 0` and `etch_depth =
t_ln` (the default). `make_geometry(**params)` returns it for `t_undercut > 0` and a `RibGeometry`
otherwise; result files (`ModeSweep.geometry`) and the cluster sweeps use that toggle.

```python
geom = R.RibGeometryUndercut(t_ln=0.4, w_top=1.0, sidewall_deg=70, t_undercut=1.0)
geom = R.align_to_grid(geom, grid, frac=1/3)          # also checks t_undercut is whole pixels
R.plot_index_map(geom, grid, lam=1.55, fname="check_undercut.png")   # (c) zooms on the LN/pedestal corner
```

Edge cases handled: `etch_depth == t_ln` (no slab), `etch_depth == 0` (unetched film; an mpbx
zero-height Trapezoid would raise, so the ridge is omitted), `t_clad = 0` / finite / `inf`.
Shapes are listed ridge → slab → oxide because mpbx resolves overlaps **first-shape-wins**
(the opposite of Tidy3D). A slab or cladding thinner than one pixel triggers a warning, because
mpbx cannot see a layer that holds no pixel corner (GOTCHAS #26). Examples:
`results/sweep_quick/index_map_400nm.png` and `index_map_1550nm.png` (Tidy3D geometry).

## 3. Convergence (`results/convergence/`, Tidy3D geometry: 0.4 µm film, 1.8 µm top, 73°, fully clad)

**Window, λ = 2.0 µm** (largest modes), 40 nm pitch, n_eff / n_g relative to an 8 µm window:

| mode | 5.2 µm | **6 µm** | 7.2 µm | edge fraction at 6 µm |
|---|---|---|---|---|
| TE00 | 1e-6 / 2e-5 | **1e-7 / 2e-6** | 1e-8 | 1e-6 |
| TM00 | 2e-5 / 4e-4 | **3e-6 / 8e-5** | 2e-7 / 6e-6 | 3e-5 |
| TE10 | 8e-5 / 2e-3 | **2e-5 / 5e-4** | 1e-6 / 5e-5 | 6e-5 |
| TM10 (near cutoff) | 1e-3 / 2e-2 | **4e-4 / 9e-3** | 6e-5 / 2e-3 | 6e-4 |

→ **6 × 6 µm** is converged for the fundamentals. The stored per-mode `edge_frac` (|E|² within
0.25 µm of the window edge) tracks the window error. As a rule of thumb, `edge_frac < 1e-4`
means a window error below ~2e-5 in n_eff. Near-cutoff higher-order modes at long λ need a
bigger window (or just treat them as unreliable).

**Pitch.** This is the part that needed care:

1. **The error depends on where the horizontal LN interfaces fall inside their pixels.** For
   the same pitch, an interface on a pixel center gives an n_eff error ~3× larger than one 1/3
   of a pixel off, and of opposite sign to one on a pixel edge. Pitches like 30 nm vs 40 nm
   move the interfaces to different sub-pixel positions, so naive "is it the same at a finer
   step?" tests **zig-zag**. My first, unaligned quick study did exactly that (exploratory run,
   not shipped): at 2 µm, 60 → 40 → 30 nm went −3e-4, then +1e-3. An unetched film has an exact
   answer, and mpbx converges to it as C·dy² with C fixed by the alignment
   (`run_checks.py` check 7 reproduces this at the production pitch). `align_to_grid(geom, grid,
   frac)` pins the alignment, so every pitch is comparable.
2. With the alignment pinned, n_eff converges at second order: successive-difference ratios
   (50→25)/(25→12.5 nm) are 3.3–4.3 for the fundamental modes (4.7 / 6.0 for TE10 / TM10 at 2 µm,
   where the 50 nm point is not yet asymptotic). Error = C·dy², interfaces on pixel centers
   (frac 0), C in µm⁻². Here C is measured against Richardson(25, 12.5 nm), so C at 25 and at
   12.5 nm are equal by construction; the 50 nm point is the independent test.

| λ | TE00 | TM00 | TE10 | TM10 | n_eff error at dy = 25 nm → 12.5 nm (TM00) |
|---|---|---|---|---|---|
| 0.4 µm | −1.35 | −2.02 | | −2.06 | 1.3e-3 → 3.2e-4 |
| 1.0 µm | −1.48 | −2.71 | −1.53 | −2.74 | 1.7e-3 → 4.2e-4 |
| 2.0 µm | −0.74 | −0.69 | −0.50 | −0.31 | 4.3e-4 → 1.1e-4 |

   **n_g:** at 1.0 and 2.0 µm n_g converges like n_eff (errors 3–6e-4 at 12.5 nm). At 0.4 µm
   it is not yet in the asymptotic regime (error ±3e-4 and changes sign between 50 and 25 nm),
   so treat 0.4 µm n_g as good to a few e-4 only.

   **Production alignment** (film 1/3 pixel off a center; `run_convergence.py --align-check`,
   section (D) of `convergence.txt`): the rib at 1.0 µm gives **C = −0.49 (TE00), −0.92 (TM00),
   −0.55 (TE10), −0.96 (TM10)**, ~3× smaller than with interfaces on pixel centers. Its own
   Richardson limit agrees with the pixel-center one to 3–4e-6 in n_eff (≤ 1.5e-5 in n_g). Two
   independent extrapolations agreeing makes those limits good references. For the flat film at
   this alignment and dy = 12.5 nm, `run_checks.py` gives C = −0.27 to −0.92 over 0.4–2.0 µm.
   The 1/3-pixel C was measured at 1.0 µm only for the rib; at the other wavelengths the
   pixel-center C was smaller or similar.
3. **Only the vertical pitch matters.** At λ = 0.4 µm, dy = 12.5 nm, going from dx = 12.5 to
   25 nm changed n_eff by 5e-6 for TE00/TM00 (2e-5 for TM10) and halved the solve time.

**Chosen grid (`run_sweep.py`):** 6 × 6 µm, **dx = 25 nm, dy = 12.5 nm** (240 × 480 px),
film interfaces 1/3 pixel off center. Expected discretization error: n_eff ≤ ~8e-5 (TE) /
~1.5e-4 (TM), n_g ~2e-4. It is smooth in λ (fixed grid), so it barely affects dispersion
curves. If you need ~1e-5 absolute, run a second sweep at dy = 25 nm with the same
`align_to_grid` frac and combine with `M.richardson(q_fine, q_coarse)` on the tracked branches.
It costs roughly +45% (at 0.4 µm the dy = 25 nm grid solved in 66 s vs 145 s for dy = 12.5 nm).

## 4. Test sweep (`results/sweep_quick/`)

`python run_sweep.py --batch`: 0.4–1.6 µm in 200 nm steps, 8 modes, production grid, CPU.
Then `python postprocess_sweep.py results/sweep_quick/modesweep.h5 --relabel 6 --lam 0.4 0.8 1.6 --save`
(and the same on `per_lambda/`). The sweep ran with HG order 3, which labels TM40–TM60 at 0.4 µm
as "TM30". Order 6, with the degenerate-fit guard, is now the default, and the shipped files are relabeled.
`sweep_quick_run.log` still shows the original order-3 labels.

* All 56 mode solves converged (LOBPCG + Newton). Wall time per λ (12-core i7-12700, CPU):
  1.6 µm 462 s, 1.4 µm 349 s, 1.2 µm 191 s, 1.0 µm 161 s, 0.8 µm 214 s, 0.6 µm 335 s, 0.4 µm 594 s.
  Long λ is slow because of the weakly guided / cladding modes in the 8-mode set. 0.4 µm is
  slow because LOBPCG needs many more iterations for the tightly spaced TM00–TM60 cluster (and
  nev must reach the 8th band to include TE00). The warm start (on by default) matters for the real 10 nm
  sweep: from 1.55 to 1.56 µm it cut the Newton steps from 3–4 to 2 per band and the solve from
  443 s to 350 s, with identical results (Δn_eff 2e-15). **Full 161-λ sweep on this CPU:
  roughly 12–15 h at nev = 8**, and more with the nev = 10 that `--full` uses for margin. Run it on
  the cluster (section 8), giving each array task a contiguous chunk of λ to keep the warm start.
* **TE00 track** (`track_mode`): found at every λ. Its solver index moves 7 → 3 → 2 → 1 → 0 → 0 → 0 as
  the TM modes (n_o > n_e) cross it, and the field overlap with the previous λ is ≥ 0.99
  throughout. TM00 is index 0 up to 1.0 µm and index 1 above. The TE00/TM00 crossing is
  between 1.0 and 1.2 µm, ≈ 1.02 µm by linear interpolation.

| λ (µm) | TE00 n_eff | TE00 n_g | TM00 n_eff | TM00 n_g |
|---|---|---|---|---|
| 0.4 | 2.276382 | 2.84675 | 2.383125 | 3.07460 |
| 0.6 | 2.117955 | 2.41206 | 2.186299 | 2.57156 |
| 0.8 | 2.037779 | 2.31403 | 2.075189 | 2.48049 |
| 1.0 | 1.973953 | 2.27447 | 1.977404 | 2.45338 |
| 1.2 | 1.916198 | 2.25077 | 1.884581 | 2.42513 |
| 1.4 | 1.862037 | 2.23065 | 1.798250 | 2.37246 |
| 1.6 | 1.810773 | 2.20991 | 1.721392 | 2.29250 |

  Cross-check: at 1.0 µm these sit 9e-5 (TE00) / 1.5e-4 (TM00) below the Richardson-converged
  values from the convergence study, as predicted for this grid.
* Figures: `dispersion.png` (n_eff and n_g, by solver index | by HG label with the TE00 track;
  hollow / grey = not guided, clipped by the window, or poorly labeled), `fields_400nm.png`,
  `fields_800nm.png`, `fields_1600nm.png` (all 8 modes; TE00 row marked ">>").
  At 1.6 µm you can see what the flags catch: TM20 is guided but clipped (0.8 % of |E|² at the
  window edge), and modes 6–7 are SiO2 cladding modes (not guided).
* Data: `modesweep.h5` (all of the above incl. fields), `modesweep.csv` (one row per
  λ × mode, like Tidy3D's `to_dataframe()`), `per_lambda/*.h5` (one file per λ).
  `results/sweep_quick_run.log` has the full per-λ tables.

## 5. Comparison with your Tidy3D run (same geometry, 400 × 400 on 6 × 6 µm)

`python compare_tidy3d.py results/sweep_quick/modesweep.h5 <...WGWidth1800nm...csv>`
→ `results/sweep_quick/compare_tidy3d.png`. Each tracked mpbx branch is matched to the same-polarization
Tidy3D mode with the closest n_eff.

| λ (µm) | TE00 Δn_eff (mpbx − Tidy3D) | TE00 Δn_g | TM00 Δn_eff | TM00 Δn_g |
|---|---|---|---|---|
| 0.4 | (no TE mode in the 6 Tidy3D modes) | | −1.2e-4 | −4.8e-4 |
| 0.6 | −1.07e-3 | +1.5e-4 | +0.9e-4 | −6.9e-4 |
| 1.0 | −1.72e-3 | −4.4e-4 | +8.3e-4 | −1.5e-3 |
| 1.6 | −2.18e-3 | −1.7e-3 | +2.11e-3 | −1.6e-4 |

What this does and does not tell us:

* The mode sets, orderings, polarizations and crossings agree (e.g. TM above TE below ~1 µm,
  the TE00/TM00 crossing near 1 µm, all six Tidy3D modes at 0.4 µm being TM).
* At 0.4 µm TM00/TM10 agree with the **grid-converged** mpbx values to 1e-5.
* The n_eff difference grows smoothly with λ to ~2e-3 at 1.6 µm, with **opposite sign for TE and
  TM**. **I could not explain it, and it is not settled which side is off.** What I ruled out:
  - mpbx discretization: ≤ 1.5e-4 on this grid (section 3), ~10× smaller.
  - mpbx planar-interface handling: an unetched film converges to the exact slab answer for
    both TE and TM (`run_checks.py` check 7).
  - Material: the pole-residue fit your Tidy3D run used
    (`Matsuda_MgO-LN_anisotropic_dispersive_uniaxial_pole-residue.hdf5`) equals the Matsuda
    Sellmeier to ~1e-12 for both n_o and n_e over 0.4–2.0 µm.
  - Geometry: same thickness, top/base width and SiO2 model. The LN area in mpbx's ε matches
    the analytic trapezoid to < 1 pixel.

  Tidy3D grid error is **not** an obvious culprit either. Your earlier Tidy3D 256²→512² test
  (10 × 10 µm plane, 39 → 19.5 nm, a different waveguide) moved n_eff by up to 2.0e-3, with
  TE **rising** and TM **falling** on refinement. That is the direction that would *widen* this
  gap, although sub-pixel alignment effects like those in section 3 can make such two-point
  trends misleading.

  Decisive next tests, cheapest first:
  1. Check whether `use_local_subpixel=True` actually took effect in that run. The notebook
     comment says it falls back to plain averaging when the tidy3d_extras build is missing.
  2. An unetched-film Tidy3D solve vs the exact slab answer, which tests Tidy3D's anisotropic
     interface treatment alone.
  3. A finer local Tidy3D solve at 1.0 µm (e.g. 800 × 800 on 6 × 6 µm), compared with the
     grid-converged mpbx values in `results/convergence/convergence.txt` (TE00 n_eff 1.974039 /
     n_g 2.27457, TM00 1.977556 / 2.45359).
* **At long λ, large differences in the higher-order modes are expected.** The Tidy3D solve used
  a fixed 6 × 6 µm window at every λ. That is enough for the fundamentals, but not for
  higher-order modes as they approach cutoff and spread into the cladding. The mpbx window study
  (§3) shows this at 2.0 µm: a 6 µm window leaves n_g errors of ~5e-4 for TE10 and ~9e-3 for
  TM10. Near cutoff (TE20 / TM20 above ~1.5–1.6 µm for this geometry), the differences grow to
  ~0.01 in n_eff and ~0.1 in n_g. Past cutoff, Tidy3D's n_eff drops below the SiO2 index, so it
  is no longer a guided mode and the comparison means nothing. The production sweep
  (`chunked_sweep/`) uses windows that grow with λ, up to 7.9 × 7.9 µm. When comparing the two,
  trust only modes with a small `edge_frac`, and treat points flagged "field at window edge" as
  unreliable. Tidy3D's boundary conditions differ from mpbx's periodic window, so its window
  error will not equal these numbers, but the same modes are affected. This does **not** explain
  the smooth ±2e-3 offset of the 00 modes above: a 6 µm window moves those by ≲ 2e-6 in n_eff.
* Your 6-mode Tidy3D solve misses TE00 at 0.4 µm (TE00 is the 8th mode there). Use ≥ 8 modes,
  better 10.

## 6. CPU vs GPU

* **This run is CPU only, confirmed:** `jax.devices() == [CpuDevice(id=0)]`, backend `cpu`,
  float64 on. `setup_jax()` prints this in run_sweep / run_convergence / run_checks, and the
  backend is stored in every result file's metadata (new runs also store the device list and the
  x64 flag). The venv
  has the CPU-only jax 0.12 wheel and no CUDA plugin. JAX's CUDA wheels are Linux-only, so on
  this Windows box the RTX 3060 is not usable from the native venv (WSL2 would be needed).
* **Single GPU later** (from `mpbx/docs/installation.md`): install `jax[cuda12]` (mpbjax has a
  `[cuda]` extra) on Linux/WSL2, call `setup_jax("cuda")`, keep float64; mpbx needs no other
  switch. Single-GPU is marked "ready" in the mpbx README but STATUS.md says it has not been
  run on hardware. mpbx requires FP64, and GeForce cards (e.g. the RTX 3060 here) run FP64 at
  ~1/64 rate. mpbx's own scaling notes found small 2-D grids dispatch-bound with the stock
  `solve_k`. With the jitted driver here the CPU solve is compute-bound (time grows with the
  pixel count), so a GPU could help, but the FP64 rate limits it. Measure one wavelength first.
* `setup_jax()` must run **before** `import mpbjax`: importing mpbjax starts the JAX backend,
  and a platform chosen afterwards is silently ignored.

## 7. Solver notes, speed, and known limitations

* `tfln_modes.solve_k_checked` is mpbx's `Mo.solve_k` Newton iteration (same result to
  ≤ 4e-15 in n_eff, `run_checks.py`) with two changes. (1) The LOBPCG eigensolve is `jax.jit`-compiled once, where
  stock `solve_k` recompiles it on every Newton step (GOTCHAS #21). (2) It reports LOBPCG /
  Newton convergence; stock `solve_k` silently ignores non-convergence. The per-mode
  `converged` flag is stored.
* **Parity sectors are off by default.** Splitting the x-mirror sectors (`use_parity=True`,
  i.e. `parities={0: ±1}`) needs an **odd** number of x pixels (`make_grid(..., odd_x=True)`).
  On an even Nx, mpbx's mirror projector zeroes the Nyquist plane (mpbx GOTCHAS #33), and the
  sector solve never converged on this rib (1000 iterations, n_eff off by 2e-5, ~40× slower).
  That was my grid choice, not an mpbx bug. With Nx = 121 it matched the full solve to 2e-15 but
  was only ~15 % faster in one test, so it is left off. `solve_wavelength` refuses parity on an
  even Nx.
* **nev = 8 is the bare minimum here.** At λ = 0.4 µm TE00 is the **8th** mode by n_eff (index 7):
  seven TM modes, TM00–TM60, sit above it because n_o > n_e. Use nev = 10 for the full sweep or
  when changing the geometry. `run_sweep` prints a warning when TE00 is the last solved band.
* Mode labels come from mpbx's Hermite-Gaussian fits (`hg_fit_residuals`, the engine of
  `hg_mode_label`), with two added guards. (1) Numerically degenerate fits are discarded:
  sub-pixel waist, center outside the crop, or absurd amplitude. At order 6 one such fit had
  labeled a guided TE30 "TE06". (2) Among fits within 0.03 of the best, the lowest order wins.
  Without that, a weakly confined TE00 at 2 µm was labeled "TE31" (exploratory run, not shipped).
  Labels of unguided / clipped modes are meaningless; the plots grey them out.
  `track_mode` picks TE00 with your Tidy3D-era rule (TE-polarized,
  guided, best moment-matched HG00 overlap) and cross-checks the HG label and the field overlap
  with the previous wavelength.
* The window is **periodic, not PML**, and the operator is Hermitian. Leaky / substrate-
  radiating behavior is not modeled; weakly guided modes show up as large `edge_frac`. In
  partially etched ribs, lateral leakage of TM-like modes into the slab continuum cannot
  appear (the `guided` flag uses max(slab TE, slab TM) n_eff as the cutoff).
* `ln_frac` (electric energy in LN), `hg00`, `edge_db`, Newton / LOBPCG iteration counts and
  per-λ solve times are also stored.

## 8. Output data and scaling to a cluster

* `ModeSweep.save(path)` writes HDF5: per-mode datasets `(n_lam, n_modes)` (`neff`, `ng`,
  `te_frac`, `hg_pol/m/n`, `hg_err`, `guided`, `edge_frac`, ...), `lam`, `x`, `y`,
  `E (n_lam, n_modes, 3, Nx, Ny)` complex64, and JSON metadata (geometry, grid, solver settings,
  material + rotation, backend). `ModeSweep.load(path, fields=False)` skips the fields.
  `to_csv` / `to_dataframe()` give a Tidy3D-like `(wavelength, mode_index)` table (pandas is
  not installed in this venv, CSV works without it). mpbx's own `WgSimulation` serializer
  refuses rotated custom materials, which is why this file format exists.
* Size: E is nev × 3 × 240 × 480 × 8 B ≈ 22 MB per wavelength at nev = 8 (28 MB at nev = 10,
  ≈ 4.5 GB for the full 161-λ sweep). Field data barely compresses. If that is too much, set
  `store_fields=False` (labels and tracking overlaps then need a re-solve), or keep only the
  tracked modes.
* `run_sweep(..., checkpoint_dir=...)` also writes one file per wavelength. That is exactly
  what one cluster array task would produce: give each task one λ (or a chunk), then
  `ModeSweep.gather(dir)` (it refuses mixed geometries / grids, duplicate λ, and a mix of
  files with and without fields), or simply `python postprocess_sweep.py <dir>`. Point
  `JAX_COMPILATION_CACHE_DIR` at shared scratch so tasks reuse compiled solvers. Each λ is
  independent, so this is embarrassingly parallel. One process already uses all cores, so on a
  node run one task per process with `XLA_FLAGS=--xla_cpu_multi_thread_eigen=false` if you
  pack several.

## 9. Suggested proofreading order

1. `tfln_rib.py`: `RibGeometry.shapes()` (heights, sidewall, cladding), `R_XCUT`.
2. `results/sweep_quick/index_map_*.png`: does the geometry look right?
3. `tfln_modes.py`: `solve_wavelength` (what is stored), `track_mode`.
4. `results/sweep_quick/dispersion.png`, `fields_*.png`, the TE00 track printed in
   `results/sweep_quick_run.log`.
5. `results/convergence/convergence.txt` / `.png`.
