# Chunked TFLN mode-solve sweep on MIT SuperCloud

One **task** = one waveguide geometry, 400–2000 nm in 10 nm steps, split into five wavelength
**chunks** with their own window, pitch and number of modes. Inside a task, a pool of pinned worker
processes solves the wavelengths in parallel, and each geometry ends up as a handful of result files
(no per-wavelength file clutter). Multi-geometry sweeps are one Slurm job array (or one LLsub
"triples" job) with one or more geometries per array task.

`handoff_tfln_modesolve/` is used as it is: its `tfln_modes.py` solver and `geometry_gen/tfln_rib.py`
geometry are imported by this folder's code. Numbers below are for the reference geometry (400 nm
film, fully etched, 1.8 µm top width, 73° sidewalls, fully SiO2-clad) on the i7-12700 desktop unless
stated otherwise.

**Quick start on SuperCloud** (details in §4):

```bash
cd ~/github && git clone git@github.com:oceanpdoshi/mpbx.git && git clone <this repo> mpbx_sims
cd mpbx_sims && bash chunked_sweep/supercloud/setup_env.sh          # once, on a login node
bash chunked_sweep/supercloud/submit_single_geometry.sh --test        # ~10 min check on a compute node
bash chunked_sweep/supercloud/submit_single_geometry.sh               # one geometry, one full node
bash chunked_sweep/supercloud/submit_geometry_sweep.sh my_geoms.csv   # many geometries
```

---------------------------------------------------------------------------------------------------

## 1. Files

| file | what it is |
|---|---|
| `sweep_config.py` | **the settings**: chunks (λ range, window, dx, dy, modes), step, solver, outputs. Edit this. |
| `run_geometry.py` | runs one or more geometries (one task): plans grids, runs the worker pool, merges each geometry |
| `results_io.py` | writes the per-geometry result files and **reads them** (`load_modes`, `load_tracks`, `load_field`, `list_geometries`, `load_all_tracks`) |
| `collect_run.py` | run-level tables over all geometries of a run (`run_summary.csv`, `all_tracks.csv`) |
| `make_geometries.py` | writes a geometry list (CSV) from lists of widths, thicknesses, ... |
| `geometries_example.csv` | the CSV format |
| `analyze_results.py` | notebook-style (#%%) plotting examples |
| `supercloud/setup_env.sh` | one-time environment setup on a login node |
| `supercloud/submit_single_geometry.sh` | submit one geometry (also `--test`, `--dry-run`, `--resume`) |
| `supercloud/submit_geometry_sweep.sh` | submit a geometry list as a Slurm job array (+ a collect job) |
| `supercloud/submit_geometry_sweep_triples.sh` | the same with LLsub triples (what SuperCloud recommends for many tasks) |
| `supercloud/submit_benchmark.sh` | one-time node benchmark to choose cores per worker (§3) |
| `supercloud/job_*.sbatch`, `common_env.sh`, `lib_submit.sh` | used by the submit scripts |
| `bench_parity.py`, `bench_cpu.py` | the parity / FFT-size and CPU-scaling benchmarks behind §2.3 and §3 |
| `study_chunks.py`, `analyze_studies.py` | the convergence / window / mode-count studies behind §2 (`results/studies_summary.txt`) |

The earlier two-grid (Richardson) driver `run_chunked.py` / `combine_chunks.py` is replaced by
`run_geometry.py` / `results_io.py` (it is still in git history, commit 8c3109d).

## 2. The chunks

### 2.1 What is implemented, and what changed from your proposal

| chunk | λ [µm] | window asked → used [µm] (px) | dx / dy [nm] | modes per parity sector (total) | must-find targets |
|---|---|---|---|---|---|
| A | 0.40–0.60 | 4 × 4 → 3.94 × 3.94 (315 × 315) | 12.5 / 12.5 | 6 (12) | all 8 |
| B | 0.60–1.00 | 5 × 5 → 5.06 × 4.90 (405 × 392) | 12.5 / 12.5 | 5 (10) | all 8 |
| C | 1.00–1.30 | 6 × 6 → 5.94 × 6.00 (297 × 480) | 20 / **12.5** (you: 20) | 4 (8) | TE/TM 00, 10 |
| D | 1.30–1.70 | 7 × 7 → 6.88 × 6.88 (275 × 550) | 25 / **12.5** (you: 25) | 3 (6) | TE/TM 00, 10 |
| E | 1.70–2.00 | 8 × 8 → 7.88 × 7.88 (315 × 630) | 25 / **12.5** (you: 25) | 3 (6) | TE/TM 00, 10 |

* **Windows**: yours, rounded to FFT-friendly pixel counts (at most 2 % smaller; odd Nx for the parity
  solve). For a geometry that needs more room (a wider ridge, a thicker film) the window is enlarged
  automatically to leave ≥ 1.4·λ_max of cladding around the guide (§2.4); the dry-run and the summary
  file say so.
* **Modes**: your totals, split equally into the two parity sectors (§2.3). If a target mode is
  missing at some λ while the deepest solved mode of its sector is still guided, that λ (and the rest
  of its work unit) is re-solved with 2 more modes per sector (§2.5).
* **One change: dy = 12.5 nm in every chunk** (§2.2). Your dx is kept. This costs ~25 % more than
  dy = dx (23 vs 19 desktop-core-hours per geometry) and buys a 2–4× smaller error above 1 µm and
  near-invisible steps at the chunk boundaries. Your exact proposal is one comment away in
  `sweep_config.py`.
* **Single grid** (no Richardson), parity on, 1/3-pixel interface alignment in every chunk, as you
  asked. Boundary wavelengths (0.6, 1.0, 1.3, 1.7 µm) are solved by both neighbouring chunks: the
  lower chunk's result goes in the tables, the comparison goes in the summary ("chunk seams").
* **Measured seams** with these chunks (`results/check_prod/`, all five chunks solved at their
  boundary λ): TE/TM00 ≤ 7e-6 in n_eff and ≤ 4e-6 in n_g; TE/TM10 ≤ 2.6e-5 / ≤ 2.5e-5; TE/TM20–30
  ≤ 7e-5 / ≤ 9e-5 (TE20 at 1.7 µm, near its cutoff, differs by 2e-3 in n_g between the two windows).
  With dy = 20 / 25 nm above 1 µm the 00-mode steps would be ~2e-4.

### 2.2 Discretization: why dy matters and dx much less

With sub-pixel smoothing the error is close to C·h² in each direction (`results/studies_summary.txt`;
a "slope ratio" of 1 = clean h²): 0.95–1.05 for the TE modes and for every mode at dx = 12.5 nm at
1 µm, 1.1–1.4 for TM n_g at 0.4–0.6 µm, and 0.4–1.3 at 2 µm, where 12.5–25 nm is not yet fully
asymptotic. So C·h² scalings (including the dy = 20 nm column below) are good to tens of percent,
and the 2 µm numbers carry a ±30 % spread.

* **Vertical (dy)**: n_eff error C·dy² with C ≈ −0.5 µm⁻² (TE) and −0.9 µm⁻² (TM) at 0.4–1.0 µm,
  ≈ −0.35 µm⁻² at 2 µm; for n_g C ≈ −0.6 (TE) / −1.3 (TM) µm⁻² at 1 µm and **stays ≈ −1.3 (TM) at
  2 µm**. It barely depends on the lateral mode order. This is what the "fine grid" errors you found
  acceptable are made of.
* **Lateral (dx)**: much smaller for the 00 modes, growing ≈ (m+1)² with the lateral order and with λ.

Single-grid error (value minus converged reference), worst TE / TM target:

| λ | dy = 12.5 nm (dx 12.5–25) | dy = 20 nm | dy = 25 nm | dx 25 vs 12.5 nm, m = 0 / 1 / 3 |
|---|---|---|---|---|
| 0.4 µm | n_eff −8e-5 / −1.3e-4, n_g −9e-5 | (−2e-4 / −3.3e-4) | n_eff −3e-4 / −5e-4 | −6e-6 / −2.5e-5 / −1.0e-4 |
| 1.0 µm | n_eff −8e-5 / −1.5e-4, n_g −1.1e-4 / −2.3e-4 | (−2e-4 / −3.8e-4, n_g −6e-4) | n_eff −3.1e-4 / −5.9e-4, n_g −8.4e-4 | −1.5e-5 / −5.5e-5 / −1.9e-4 |
| 2.0 µm | n_eff −5e-5 (00) / −0.6 … −0.9e-4 (TE10), n_g −1.0e-4 (TE00) / −1.9e-4 (TM00) / −2.4 … −2.9e-4 (TM10) | | n_eff −1.7e-4, n_g −4.3e-4 (TE00) / −7.6e-4 (TM00) / −8.7e-4 (TM10) | m = 0: ~−2.5e-5; m = 1: ~−1.4e-4 |

The exception is a mode close to a same-sector anticrossing: TE30 at 0.4 µm (0.0075 above TM01) has
an n_g error of −7e-4 at dy = 12.5 nm (§2.6).

(dy = 20 nm values in brackets are C·dy² from the 12.5 / 25 nm data, not solved.)

So your dy = 20 / 25 nm above 1 µm would make the TM n_g error 3–4× what you accepted at 12.5 nm,
and every chunk boundary would show a step of ~2e-4 (n_eff) / ~3e-4 (n_g) in the TM curves. Coarsening
only dx (12.5 → 20 → 25 nm) changes the 00 modes by ≤ 2.5e-5 and the 10 modes by ≤ ~1.4e-4 at 2 µm.
The errors are systematic (all modes too low, nearly the same for neighbouring λ), so differences
such as Δn between two modes at one λ are more accurate than the absolute values.

Interface alignment: every chunk puts the LN/SiO2 interface 1/3 pixel above a pixel centre. With
dy = 12.5 nm everywhere, a film or slab whose thickness is a multiple of 12.5 nm has both interfaces
at the same sub-pixel position in every chunk, which keeps C the same everywhere (the dry-run warns
otherwise; the error then varies a bit from geometry to geometry, it does not blow up).

### 2.3 Parity solve vs FFT-friendly sizes (`bench_parity.py`, `results/bench_parity.json`)

Chunk A's grid, λ = 0.5 µm, 8 P-cores:

| | time per LOBPCG iteration (6 modes) | full solve of the same 12 modes |
|---|---|---|
| Nx = 320 (2⁶·5), no parity | 174 ms | 387 s (nev 12) |
| Nx = 321 (3·107, "320 + 1"), no parity / parity | 177 / 184 ms | |
| Nx = 325 (5²·13), no parity / parity | 176 / 181 ms | **239 s** (6 + 6 per sector) |
| Nx = 319 (11·29), no parity / parity | 207 / 177 ms | |

* **FFT sizing hardly matters** for this solver: 320, 321 and 325 pixels are within 3 %; the worst
  case measured was +19 %. The per-iteration cost is dominated by the 3 × 3 tensor products and the
  block orthogonalization, not the FFTs. The code still picks sizes with prime factors ≤ 13 (it
  costs nothing).
* **Parity is the bigger lever: 1.6× faster** for the same 12 modes (239 vs 387 s), because each
  sector solves a block of 6 instead of 12 (the cost grows faster than linearly with the block). It
  needs an odd Nx; the two are compatible (315, 325, 405, ... are odd and FFT-friendly).
* **Parity also matters for correctness at crossings**: modes of opposite x-parity (e.g. TE00 and
  TM20/TM40, TE00 and TM00 near 1.02 µm) cross exactly when solved separately. In one solve they are
  near-degenerate there, LOBPCG slows down and can return mixtures, which breaks the TE/TM labels and
  n_g in the few nm around each crossing; this sweep is about crossings.
* At 0.4 µm the two sectors need 5 and 6 modes (§2.5): one solve without parity needs 11.

### 2.4 Windows (`results/studies_summary.txt`, window study)

Window error of n_g at the chunk's longest λ (40 nm pitch, Lx and Ly varied separately):

| λ, mode | Lx = 4 | 5 | 6 | 7 | 8 µm | | Ly = 3 | 4 | 5 | 6 | 8 µm |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1.0 µm, worst of 8 | 2.4e-6 | 1.3e-7 | | | | | 1.5e-6 | 8e-7 | | | |
| 2.0 µm, TE00 | 4.1e-4 | 2.9e-5 | 1.6e-6 | 1e-7 | | | 1.5e-3 | 1.1e-4 | 6.2e-6 | 4.2e-7 | |
| 2.0 µm, TM00 | 3.1e-3 | 4.6e-4 | 5.3e-5 | 7.1e-6 | 6.9e-7 | | 9.2e-3 | 1.4e-3 | 1.8e-4 | 2.4e-5 | 4.0e-7 |
| 2.0 µm, TE10 | 1.4e-2 | 2.8e-3 | 4.6e-4 | 8.3e-5 | 1.1e-5 | | 7.0e-3 | 1.2e-3 | 1.6e-4 | 2.5e-5 | 4.4e-7 |
| 2.0 µm, TM10 | 7.1e-2 | 2.9e-2 | 1.1e-2 | 3.9e-3 | 1.0e-3 | | 3.3e-2 | 1.3e-2 | 4.4e-3 | 1.5e-3 | 1.1e-4 |

* Your windows are fine for this geometry: they leave ≈ 1.4–1.5·λ_max of cladding on each side of
  the ridge base, which the study shows keeps the window error of TE00/TM00/TE10 n_g ≲ 1e-5 at every
  chunk's longest λ. The error falls as exp(−γL), γ = k0·sqrt(n_eff² − n_SiO2²), so it is tiny at the
  short end of each chunk.
* **B (0.6–1.0 µm) is the generous one**: 4.4 × 3.2 µm would do (window error < 3e-6) and would cut
  B's cost (39 % of the total) by ~40 %. Kept at your 5 × 5 µm, the safer choice for wider ridges.
* **Wider / thicker geometries** get a bigger window automatically: `MIN_MARGIN = 1.4` (× λ_max of
  cladding beyond the ridge base / film). A partially etched rib with a thin slab has slab-guided
  tails that need more room than this rule gives; watch the "field at window edge" flags there.
* **TM10 above ~1.8 µm is near cutoff** (n_eff 1.476 vs 1.438 at 2 µm): its n_g misses 1e-4 even in
  8 × 9 µm. It is reported and flagged ("field at window edge").
* Every mode carries `edge_frac`, the fraction of |E|² within 0.25 µm of the window edge; n_g error ≈
  0.2–12 × edge_frac (24–54 × for TM10 near cutoff). Modes with edge_frac > 1e-5 are flagged.

### 2.5 Number of modes (census, 6 × 6 µm, both sectors)

Rank of each target inside its sector (0 = highest n_eff); "needed" = deepest target + 1:

| λ [µm] | TE00 | TM00 | TE10 | TM10 | TE20 | TM20 | TE30 | TM30 | needed (−1 / +1) |
|---|---|---|---|---|---|---|---|---|---|
| 0.40 | −1:3 | +1:0 | +1:4 | −1:0 | −1:4 | +1:1 | +1:5 | −1:1 | 5 / 6 |
| 0.50 | −1:2 | +1:0 | +1:3 | −1:0 | −1:3 | +1:1 | +1:4 | −1:1 | 4 / 5 |
| 0.60 | −1:1 | +1:0 | +1:2 | −1:0 | −1:3 | +1:1 | +1:4 | −1:2 | 4 / 5 |
| 0.70–1.25 | | | | | | | | | 4 / 4 |
| 1.50 | | | | | | | cut off | cut off | 3 / 3 |
| 2.00 | −1:0 | +1:0 | +1:1 | −1:1 | cut off | cut off | | | 2 / 2 |

Sector −1 holds TE00, TM10, TE20, TM30; sector +1 holds TM00, TE10, TM20, TE30. Your totals fit this
geometry exactly or with one spare (A: 6/6 vs 5/6 needed at 0.4 µm; B: 5/5 vs 4/5 at 0.6 µm). A wider
ridge has more lateral TM modes above TE30, so the automatic `NEV_EXTRA` re-solve (+2 per sector) is
the safety net; the summary lists every λ where it fired, and every target that stayed missing (cut
off, or not recognized by the node-count labels).

Above 1 µm only TE/TM 00 and 10 are required, but C still finds TE20/TM20/TE30/TM30 at 1.0–1.25 µm
with 4 per sector, and the tables keep whatever is found.

### 2.6 Accuracy you can expect, and the known limits

| region | n_eff | n_g |
|---|---|---|
| A, B (all 8 targets, away from anticrossings) | −0.8 … −1.5e-4 (TE / TM) | −0.6 … −2.3e-4 |
| A at 0.40–0.42 µm, TE30 (next to TM01) | ~−1e-4 | **~−7e-4** |
| C–E, 00 modes | −0.5 … −1.5e-4 | −1 … −2e-4 |
| C–E, TE10 | ≲ −2.5e-4 (lateral part at dx = 25 nm) | ≲ −3e-4 |
| E, TM10 above ~1.8 µm | window-limited | **up to ~+1e-3** (window error, positive) |

* The **discretization** errors are systematic and negative (the solver sits slightly low) and smooth
  in λ. **Window** errors have the other sign for the 10 modes (n_g too high): negligible except for
  TM10 near cutoff above ~1.8 µm, where they dominate (§2.4).
* **Same-sector anticrossings** (TE00–TM30 at 0.5–0.6 µm, TE10–TM20 at 0.7–0.8 µm, TE00–TM10 at
  0.8–0.9 µm, TE30–TM01 near 0.4 µm, ...; full list in `results/studies_summary.txt`): n_g of both
  branches changes fast there and is only good to ~1e-4–1e-3 within a few nm of the centre; the
  node-count labels can also swap. Flags in the tables:
  * "close same-sector neighbour (gap): anticrossing?" when a same-sector neighbour is closer than
    `NEAR_GAP` = 0.008 in n_eff;
  * "last solved mode of its sector": the neighbour below was not solved, so the gap test cannot see
    it. TE30 at 0.4 µm is the last of its 6 sector modes in chunk A, so its anticrossing with TM01
    shows up only as this flag. `nev = 7` in chunk A (+6 % total cost) would solve TM01 too.
  `analyze_results.py` plots all guided modes for checking by eye.
* TM10 above ~1.8 µm: window-limited (§2.4).
* Labels come from counting sign changes of the dominant field component; hybrid modes
  (0.2 < TE fraction < 0.8) are flagged and may carry a wrong label.

## 3. Run time, cores and memory

* **Cost per geometry ≈ 23 desktop-core-hours** (the dry-run prints it per geometry): A 18 %,
  B 39 %, C 17 %, D 13 %, E 13 %. The model is (pixels × modes² per sector), calibrated on the desktop;
  it predicted chunk A at 0.5 µm within 15 % (273 vs 239 s on 8 cores).
* **The solver is limited by memory bandwidth**: on the desktop 8 P-cores give only ~2.7× one core,
  whether as 1 × 8, 2 × 4 or 8 × 1 processes (`results/log_bench.txt`). A xeon-p8 node has two sockets
  with six memory channels each, so it should do several times better than the desktop, but by how
  much is unknown until measured: **run `supercloud/submit_benchmark.sh` once** (~1 h) and read the
  "all processes: N× one core" column. Until then my estimate is ~1.5–3 h per geometry on a full node
  and ~3–6 h at 24 cores.
* **Cores per worker**: default 2 (24 workers on a full node). Warm start (each λ starts from the
  previous one) gave 0–50 % in my tests, so short work units are cheap; more workers is better as long
  as the node's throughput keeps rising in the benchmark. Change `CORES_PER_WORKER` in the submit
  scripts after the benchmark.
* **Memory**: measured 0.9–1.4 GB peak per worker on the production grids (`results/check_prod/`;
  peak RSS is in every summary file), so 24–48 workers need ≤ ~70 GB of a node's 192 GB. Memory is
  not the constraint; Slurm gives 4 GB per requested core.
* **Measured per-λ times** on the desktop, 5 workers at once on 5 P-cores (the production grids):
  A 1100–1500 s, B 940–990 s, C 460–610 s, D 340–550 s, E 490–950 s; the first λ of each work unit
  (cold start) is the slower one. That matches the cost model for a bandwidth-shared core.
* **Disk** with `FIELDS=all`: the per-λ work files are uncompressed (~4.7 GB per geometry) and the
  merge writes ~2.9 GB of compressed field files before deleting them, so plan for ~8 GB per geometry
  that is running at the same time, ~3 GB per finished geometry.
* **Many geometries**: the per-user limit is 8 xeon-p8 nodes (384 cores) on the standard allocation
  (2 nodes on the starting one). Throughput is about the same whether you run 1 geometry per node
  (48 cores) or 2 (24 cores each) or 3 (16 each); fewer cores per geometry means less idle time at the
  end of each geometry, more means each finishes sooner. Default: 24 cores (2 per node), so 16
  geometries at a time on 8 nodes, roughly 3–5 geometries per hour (estimate), with fields off.
* **Time limits**: 4 days maximum per job; the submit scripts use 24 h. A task that runs out of time
  is resumed with `--resume` (solved wavelengths are kept). SuperCloud is down on the **second
  Tuesday of every month** (draining starts Monday evening): a job that would still be running then
  is not started, so use realistic time limits around that week.

## 4. Running on SuperCloud

### 4.1 One-time setup (login node)

```bash
cd ~/github
git clone git@github.com:oceanpdoshi/mpbx.git          # the solver: must sit next to mpbx_sims
git clone <url of this repo> mpbx_sims
cd mpbx_sims
bash chunked_sweep/supercloud/setup_env.sh
```

`setup_env.sh` installs uv (into `~/.local/bin`) if needed, gets a standalone Python 3.12 (installed
by uv on the login node's local disk, then copied to `~/.local/share/mpbx-python/` where the compute
nodes see it), and creates `mpbx_sims/.venv` with exactly the versions in `uv.lock` (jax CPU, h5py,
...) and mpbx in editable mode. uv uses file locks, which SuperCloud's Lustre does not support, so
its cache and Python install run on local disk; if `uv sync` still fails, the script falls back to
`python -m venv` + pip with the same pinned versions. It then checks the imports. Re-run it after a
`git pull` that changes `uv.lock`.

What the SuperCloud docs require, and where the scripts do it:

| docs | done in |
|---|---|
| compute nodes have no internet: install on a login node | `setup_env.sh`; jobs only read the venv in your home directory |
| home/group directories (Lustre) have **no file locking**: HDF5 needs `HDF5_USE_FILE_LOCKING=FALSE` | `common_env.sh`, and `run_geometry.py` / `results_io.py` set it too (so notebooks work) |
| `/tmp` is tiny: point `TMPDIR` to `/state/partition1/user/$USER` for installs | `setup_env.sh` |
| `$TMPDIR` in a job is node-local and deleted at the end | used only for caches (JAX compile cache, matplotlib); results go to home |
| `source /etc/profile` before `module` in batch scripts | job scripts |
| Python buffers output in batch jobs (`python -u`) | job scripts (`-u`, `PYTHONUNBUFFERED`) |
| `PYTHONNOUSERSITE` so `pip --user` packages do not shadow an environment | `common_env.sh` |
| < ~1000 files per directory, files ≥ 1 MB | ≤ ~45 per-λ files per folder while running (deleted after the merge); 5–8 result files per geometry |
| job arrays: prefer LLsub triples; no huge arrays | array size capped (`MAX_TASKS`); `submit_geometry_sweep_triples.sh` |
| memory = 4 GB per requested core | ~1 GB per worker needed |
| don't run work on login nodes | only `--dry-run` (seconds) runs there |

Windows line endings break shell scripts on Linux; `.gitattributes` keeps `*.sh` / `*.sbatch` as LF.

### 4.2 Check the pipeline on a compute node (do this first)

```bash
bash chunked_sweep/supercloud/submit_single_geometry.sh --test
```

8 cores, ≤ 1 h limit (takes ~10 min): two tiny chunks, 6 wavelengths, the full path from solving to
the merged files. Look at `outputs/tfln_single_TEST_<date>/logs/*.out` and the geometry's
`*_summary.txt` ("COMPLETE: True").

### 4.3 One geometry

Edit the settings block at the top of `supercloud/submit_single_geometry.sh` (geometry, `FIELDS`,
`CPUS`, `TIME`, `OUT_ROOT`), then:

```bash
bash chunked_sweep/supercloud/submit_single_geometry.sh --dry-run   # grids, cost, sizes; nothing submitted
bash chunked_sweep/supercloud/submit_single_geometry.sh
```

It creates `outputs/<RUN_NAME>_<date-time>/` with `geometries.csv` (one row), `run_info.txt` (git
commits, settings, job id), `logs/`, a **copy of this repo's code** in `code/` (so editing
`sweep_config.py` or pulling while the job waits in the queue cannot change the run; a resume uses the
same copy; `code/CODE_VERSION` records its git version), and the geometry's folder (§5). One Slurm
task, `-c 48`, `--exclusive=user`, launched through `srun --cpu-bind=cores` so the solver stays on
the task's own cores. The solver itself (`../mpbx`, installed in the venv) is **not** copied: its git
version is stored with every geometry, and a resume after an mpbx update prints a warning.

### 4.4 Many geometries

```bash
python chunked_sweep/make_geometries.py my_geoms.csv     # edit its lists first (or write the CSV by hand)
bash chunked_sweep/supercloud/submit_geometry_sweep.sh my_geoms.csv --dry-run
bash chunked_sweep/supercloud/submit_geometry_sweep.sh my_geoms.csv
```

* **Job array** (`submit_geometry_sweep.sh`): one array task per geometry (up to `MAX_TASKS`; beyond
  that a task solves rows k, k + N, ...), `CPUS = 24` cores each by default (two geometries per
  node), `FIELDS = none` by default (~3 GB per geometry otherwise; check `LLquota`). Slurm runs as many
  tasks at once as your node limit allows and queues the rest. A collect job runs afterwards
  (`--dependency=afterany`) and writes `run_summary.csv` and `all_tracks.csv` for the whole run.
* **LLsub triples** (`submit_geometry_sweep_triples.sh`): the SuperCloud docs' recommended way to run
  many independent tasks, and what staff ask for before granting more than 8 nodes. `[NODES, NPPN,
  NTPP]` = `[8, 2, 24]`: 8 whole nodes, 2 processes per node, 24 cores each; process r solves rows r,
  r + 16, ... one after the other. Caveats: LLsub ignores `#SBATCH` lines' command-line options, so the
  generated job script has none, and the docs do not say how to set its time limit (check `LLsub -h`);
  logs go to `<run>/logs/LLSUB.<n>/`; run `collect_run.py` yourself at the end. I could not test LLsub
  here; the job array is the safer first choice.

### 4.5 Watching, resuming, failures

```bash
LLstat                                   # your jobs (SuperCloud); or: squeue -u $USER
tail -f outputs/<run>/logs/<name>_<jobid>.out
sacct -j <jobid> -X --format=JobID%20,State%15,ExitCode,Elapsed,NodeList%20          # per array task
sacct -j <jobid> --format=JobID%25,State%15,Elapsed,AllocCPUS,MaxRSS --units=G       # memory: the <id>.0 (srun step) rows
sacct -S now-7days -s F,TO,OOM,NF,CA -X --format=JobID%20,JobName,State%25,ExitCode,Elapsed,NodeList
LLkill <jobid>                           # or: scancel <jobid>
```

* **Resume** a run that timed out, was cancelled or hit the monthly downtime (only after its jobs have
  ended): `bash chunked_sweep/supercloud/submit_<...>.sh --resume outputs/<run>`. Complete geometries
  are skipped in seconds; partly solved ones keep their solved wavelengths (`<geometry>/work/`).
  (`--resume` cannot be combined with `--dry-run`.)
* A wavelength that raises an error is logged with its traceback and skipped; the rest of its work
  unit and all other units carry on. The geometry is then merged as incomplete (`COMPLETE: False` and
  the failed λ in its summary, work files kept), the task exits with status 1, and a resume retries
  exactly the missing wavelengths. A worker killed by the system (e.g. out of memory) stops the task;
  resume after fixing the cause.
* A run folder only takes results of one setup: running other chunks / grids / step / fields into a
  folder that already holds results is refused. `--lams` (a few test wavelengths) runs are never
  marked complete and never delete the work files.
* Don't run two jobs on the same run folder at the same time.

### 4.6 Choosing cores per worker (once)

```bash
bash chunked_sweep/supercloud/submit_benchmark.sh
```

One exclusive node, ~1 h. In the log, the rows "N procs x c cores" give the node throughput in
"one-core equivalents"; set `CORES_PER_WORKER` in the submit scripts to the `c` with the highest value
(and `CPUS` per geometry to a multiple of it).

## 5. Output files

Per geometry, in `<run>/<geometry name>/` (name e.g. `tLN400_etch400_w1800_sw73_cladSiO2`):

| file | content | size (default chunks) |
|---|---|---|
| `<name>_modes.h5` | every mode at every λ, no fields: n_eff, n_g, TE fraction, LN fraction, edge_frac, parity, label, guided, converged, flags; per-target tracks; chunk seams; all settings (JSON) | ~1 MB |
| `<name>_modes.csv` | the same table, one row per (λ, mode) | ~1 MB |
| `<name>_tracks.csv` | TE00 … TM30 vs λ (n_eff, n_g, TE fraction, edge_frac, flags) | < 1 MB |
| `<name>_fields_0400-1000nm.h5` | E fields of chunks A, B (one HDF5 group per chunk: different grids) | ~1.4 GB |
| `<name>_fields_1000-2000nm.h5` | E fields of chunks C, D, E | ~1.5 GB |
| `<name>_summary.txt` | grids, timing, memory, flags, nev re-solves, missing targets, chunk seams, file sizes | |
| `<name>_dispersion.png` | n_eff and n_g of the tracked modes | |

Fields are complex64 (Ex, Ey, Ez), gzip-compressed (~60 % of raw), normalized to ∫Re(E*×H)_z dA = 1 in
mpbx units (×27.45 for Tidy3D's 1 W SI fields in V/µm); the grouping into files is `FIELD_FILES` in
`sweep_config.py`. At each λ the modes are sorted by descending n_eff (both sectors together), and the
mode index is the same in the modes and fields files.

Reading (also in `analyze_results.py`):

```python
import sys; sys.path.insert(0, "/home/gridsan/<user>/github/mpbx_sims/chunked_sweep")
import results_io as rio
run = "outputs/tfln_sweep_2026-11-20-14-05"
geoms = rio.list_geometries(run)                        # parameters, complete?, folder
modes, meta = rio.load_modes(geoms.folder[0])            # DataFrame: one row per (λ, mode)
tracks = rio.load_tracks(geoms.folder[0])                # DataFrame: target, lam, neff, ng, te_frac, flags
E, info = rio.load_field(geoms.folder[0], lam=0.8, label="TE00")   # (3, Nx, Ny), info["x"], info["y"]
allt = rio.load_all_tracks(run)                          # every geometry, with its parameters as columns
```

Download a run to your desktop (from the desktop): `rsync -rlughv <user>@txe1-login.mit.edu:/home/gridsan/<user>/github/mpbx_sims/outputs/<run> .`
(add `--exclude '*_fields_*'` to skip the fields). Nothing on SuperCloud is backed up.

## 6. Caveats: what was and was not tested

* Tested on the desktop:
  * the whole Python pipeline with `--test`: one and two geometries per task, resume after a partial
    run, a wavelength that fails (skipped, retried on resume), `--lams` inside a partial run, refusal of
    a different setup in the same folder, `--merge-only`, `collect_run.py`, the loaders, CSVs saved as
    cp1252 or with a BOM, `--dry-run` for a wide geometry that triggers window enlargement;
  * all five production chunks at their boundary wavelengths (`results/check_prod/`): grids, memory,
    timing, seams, field files;
  * the parity / FFT-size benchmark, and the cost model against those solves.
* The shell scripts were reviewed and syntax-checked (`bash -n`) but **never run**: nothing was run on
  SuperCloud (`setup_env.sh`, sbatch / srun / LLsub, Linux CPU pinning under Slurm, node throughput).
  Run `--test` first; the benchmark second.
* A full 0.4–2.0 µm production sweep of the new chunks has not been run, so the per-geometry run time
  is an estimate from the cost model.
* Accuracy numbers are for the reference geometry; other geometries were not studied (the flags and
  the automatic window / mode-count safety nets are there for them).
