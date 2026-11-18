# Chunked wavelength sweep: evaluation and recommendation

**Question.** Per waveguide geometry, how should 400–2000 nm be swept to keep n_eff and n_g errors
below 1e-4? The targets are TE00/TM00 everywhere and TE10–TE30 / TM10–TM30 mainly below 1 µm. The
sweep is split into wavelength chunks, each with its own window, pitch and number of modes, and
runs as a 48-core / 192 GB cluster task.

Everything here is new; `handoff_tfln_modesolve/` is unchanged (its `tfln_modes.py` solver and
`geometry_gen/tfln_rib.py` are imported as they are). Numbers are for the reference geometry:
400 nm film, fully etched, 1.8 µm top width, 73° sidewalls, fully SiO2-clad. They were measured on
the i7-12700 desktop.

## 1. Recommendation

| chunk | λ [µm] | window [µm] (fine grid, px) | fine grid dx / dy | coarse grid | bands per parity sector | targets |
|---|---|---|---|---|---|---|
| A | 0.40–0.60 | 4.29 × 3.0 (343 × 240) | 12.5 / 12.5 nm | 25 / 25 nm | 7 | all 8 |
| B | 0.60–1.00 | 4.69 × 3.2 (375 × 256) | 12.5 / 12.5 nm | 25 / 25 nm | 5 | all 8 |
| C | 1.00–1.50 | 5.51 × 4.2 (441 × 336) | 12.5 / 12.5 nm | 25 / 25 nm | 2 | TE00, TM00, TE10, TM10 |
| D | 1.50–2.00 | 7.84 × 6.4 (627 × 512) | 12.5 / 12.5 nm | 25 / 25 nm | 2 | TE00, TM00, TE10, TM10 |

1. **Solve the two x-mirror parity sectors separately** (odd Nx).
   * Sector −1 holds TE00, TM10, TE20 and TM30; sector +1 holds TM00, TE10, TM20 and TE30.
   * Each sector needs about half the bands, and the cost grows ~quadratically with bands.
   * TE↔TM crossings between sectors are exact.
2. **Solve every wavelength on two grids and Richardson-extrapolate**: q = q_f + (q_f − q_c)/3, with
   the coarse grid 2× coarser in **both** x and y.
   * No affordable single grid meets 1e-4. At 12.5/12.5 nm the TM modes are off by 1.3–2e-4 in
     n_eff and up to 2.8e-4 in n_g (§2.3).
   * The two-grid value lands within ≤ 1.3e-5 (n_eff) and ≤ 3.1e-5 (n_g) of the best reference at
     0.4 and 1.0 µm for all eight targets. The one exception is TE30 next to the TE30–TM01
     anticrossing at 0.4 µm.
   * The coarse grid has a quarter of the pixels, so the pair costs ≈ 1.3× the fine solve alone.
3. **Keep the fine dy = 12.5 nm and the 1/3-pixel interface alignment in every chunk.** Share the
   boundary wavelengths: 0.6, 1.0 and 1.5 µm are solved by both neighbours.
4. **Known limits**, flagged automatically by `combine_chunks.py`:
   * Inside a same-sector anticrossing, n_g of the two branches is unreliable at the 1e-4 level and
     can be ~1e-3 off at the one or two wavelengths nearest the centre (§2.4).
   * TM10 above ~1.8 µm is near cutoff and window-limited.
   * TE20/TM20 (guided to ~1.6 µm) and TE30/TM30 (to ~1.3 µm) are not solved above 1 µm. To include
     them, use 4 bands in C: about 4× C's cost, +25–30 % in total.

`run_chunked.py` implements this as one cluster task per geometry with a pool of pinned worker
processes, and is restartable. `combine_chunks.py` merges, extrapolates, estimates errors and
flags. Cost and cluster layout are in §4.

## 2. Evidence

All of it comes from `study_chunks.py`. Raw data is in `results/*.jsonl`; the tables are in
`results/studies_summary.txt`, made by `analyze_studies.py`. "Error" means value minus the best
reference, which is the finest Richardson pair plus the lateral extrapolation where it was measured.

### 2.1 How many bands (census, 6 × 6 µm, 40 nm, both sectors)

Rank of each target inside its sector (0 = highest n_eff). Bands needed = deepest target + 1.

| λ [µm] | TE00 | TM00 | TE10 | TM10 | TE20 | TM20 | TE30 | TM30 | needed (−1 / +1) |
|---|---|---|---|---|---|---|---|---|---|
| 0.40 | −1:3 | +1:0 | +1:4 | −1:0 | −1:4 | +1:1 | +1:5 | −1:1 | 5 / 6 |
| 0.50 | −1:2 | +1:0 | +1:3 | −1:0 | −1:3 | +1:1 | +1:4 | −1:1 | 4 / 5 |
| 0.60 | −1:1 | +1:0 | +1:2 | −1:0 | −1:3 | +1:1 | +1:4 | −1:2 | 4 / 5 |
| 0.70–1.25 | | | | | | | | | 4 / 4 |
| 1.50 | | | | | | | cut off | cut off | 3 / 3 |
| 2.00 | −1:0 | +1:0 | +1:1 | −1:1 | cut off | cut off | | | 2 / 2 |

* **Without parity** the same targets need 11 bands at 0.4 µm, because TM00…TM60 and TE00…TE20
  sit above TE30.
* **Extra bands are not free.** At long λ the bands below the last target are cladding/box modes
  that converge slowly.
* **Guard band in A.** A uses 7 bands per sector, so at 0.4 µm there is one guard band. TE30's
  neighbour TM01 is only Δn_eff = 0.003 below it there, which is why the guard is needed.
* **A at 0.5–0.6 µm is over-provisioned** by 2–3 bands. Splitting A into 0.40–0.50 (7 bands) and
  0.50–0.60 (6 bands) would save ~5 % in total.

### 2.2 Window (40 nm pitch, Lx and Ly varied separately around a 9 × 9 µm reference)

Window error of n_g at the chunk's longest wavelength. The 1.0 µm row is the worst of all 8 targets.

| λ, mode | Lx = 4 | 5 | 6 | 7 | 8 | | Ly = 2 | 3 | 4 | 5 | 6 | 8 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1.0 µm, all 8 | 2.4e-6 | 1.3e-7 | 6e-8 | | | | 3.1e-5 | 1.5e-6 | 8e-7 | | | |
| 2.0 µm, TE00 | 4.1e-4 | 2.9e-5 | 1.6e-6 | 1e-7 | | | 2.0e-2 | 1.5e-3 | 1.1e-4 | 6.2e-6 | 4.2e-7 | |
| 2.0 µm, TM00 | 3.1e-3 | 4.6e-4 | 5.3e-5 | 7.1e-6 | 6.9e-7 | | 5.1e-2 | 9.2e-3 | 1.4e-3 | 1.8e-4 | 2.4e-5 | 4.0e-7 |
| 2.0 µm, TE10 | 1.4e-2 | 2.8e-3 | 4.6e-4 | 8.3e-5 | 1.1e-5 | | 4.1e-2 | 7.0e-3 | 1.2e-3 | 1.6e-4 | 2.5e-5 | 4.4e-7 |
| 2.0 µm, TM10 | 7.1e-2 | 2.9e-2 | 1.1e-2 | 3.9e-3 | 1.0e-3 | | 6.3e-2 | 3.3e-2 | 1.3e-2 | 4.4e-3 | 1.5e-3 | 1.1e-4 |

* **The window error falls as exp(−γL)**, with γ = k0·sqrt(n_eff² − n_SiO2²). For TM00 at 2 µm the
  fitted rates are 2.1 and 2.0 µm⁻¹ against γ = 2.2 µm⁻¹.
* **So the margin around the core scales with 1/γ.** It is tiny at short λ, large near 2 µm, and
  largest for the weakest mode.
* **Short wavelengths (chunk A):** A's window has edge fractions ≤ 2e-7 at 0.4–0.6 µm, i.e. a
  window error of ~1e-6.
* **Around 1.5 µm (chunk C):** C's window is the 2 µm requirement scaled by γ, with ~20 % margin.
* **At 2 µm both directions need 6.5–8 µm.** That is about 7 µm across for TM00 and ~7.8 µm for
  TE10, and ~6.4 µm vertically for both, because the thin film's mode spreads into the oxide. The
  old 6 × 6 µm window is marginal for TM00 n_g (~8e-5 combined) and fails for TE10 (~5e-4).
* **TM10 above ~1.8 µm is near cutoff** (n_eff 1.476 vs 1.438). It misses 1e-4 in n_g even at
  8 × 9 µm. It is reported, and `combine_chunks.py` flags it.
* **Per-mode `edge_frac`** (the fraction of |E|² within 0.25 µm of the edge) tracks the window error:
  * n_g error / edge_frac is 0.2–3.5 for the vertical window of every mode;
  * it reaches ~12 for the lateral window of TE10 at 2 µm, and 24–54 for TM10 near cutoff;
  * `combine_chunks.py` uses 12·edge_frac as its window estimate and flags edge_frac > 1e-5.

### 2.3 Pitch: single grid vs two-grid Richardson (interfaces 1/3 pixel off-centre)

Worst error over the 8 targets, excluding TE30 at the 0.4 µm anticrossing (§2.4):

| λ | single 25/12.5 nm (dx/dy) | single 12.5/12.5 nm | Richardson in dy only, dx = 25 nm | **Richardson in x and y (25/25 → 12.5/12.5)** |
|---|---|---|---|---|
| 0.4 µm | n_eff 2.2e-4, n_g 1.2e-4 | n_eff 1.5e-4, n_g 9.7e-5 | n_eff 1.0e-4, n_g 3.0e-5 | **n_eff 1.3e-5, n_g 3.1e-5** |
| 1.0 µm | n_eff 3.4e-4, n_g 4.3e-4 | n_eff 2.0e-4, n_g 2.8e-4 | n_eff 1.9e-4, n_g 2.0e-4 | **n_eff 4.9e-6, n_g 8.9e-6** |
| 2.0 µm | n_eff 9e-5, n_g 2.9e-4 | — | 00 modes: n_eff 1.5e-5, n_g 1.7e-5; 10 modes: n_eff 5.6e-5, n_g 8.2e-5 (both plus a lateral ≈ 3e-5 / 1e-4) | not measured |

References:
* **0.4 µm:** a dy series 25 → 6.25 nm at dx = 25 nm, plus an x-Richardson from a dx series
  25 → 8.33 nm.
* **1.0 µm:** a dy series 25 → 8.33 nm at dx = 12.5 nm, plus the lateral shift
  (q(12.5, 12.5) − q(25, 12.5))/3. That shift assumes a clean dx² lateral error, which was verified
  with three points at 0.4 µm only.
* **2.0 µm:** a dy series 25 → 8.33 nm at dx = 25 nm. Lateral error comes from the model below.

What the table shows:

* **Vertical error, C·dy².** For n_eff, C ≈ −0.5 µm⁻² (TE) and −0.9 µm⁻² (TM) at 0.4–1.0 µm,
  dropping to ≈ −0.35 at 2 µm. C barely depends on the lateral order. So every TM mode misses 1e-4
  in n_eff at dy = 12.5 nm below ~1.5 µm. Meeting it without extrapolation needs dy ≈ 9 nm for
  n_eff and ≈ 7 nm for n_g at 2 µm, at 1.6–3.5× the cost.
* **Lateral error.** It is a clean dx² term (slope ratio 0.97–1.01) that grows ≈ as (m+1)² and
  with λ. At dx = 25 nm and 0.4 µm it is −6e-6 / −2.5e-5 / −5.7e-5 / −1.0e-4 for m = 0…3, and about
  twice that at 1 µm. The model, used only by refine "y" chunks, is 3.5e-6·(m+1)²·λ^0.85 at
  dx = 12.5 nm.
* **Why refine x and y together.** At dx = 25 nm the slanted sidewall is sampled so coarsely that
  the dy convergence turns irregular, through a dx·dy cross term. That happens for the m = 2, 3
  modes at 1 µm (slope ratios 0.4–1.8) and for all modes at 2 µm (n_eff ratios 0.4–0.8). At
  dx = 12.5 nm all eight modes converge cleanly in dy at 1 µm (ratios 0.96–1.04).
* **The extrapolation floor.** The coarse 25 nm level is just inside the asymptotic range: at
  0.4–0.6 µm the extrapolated TM n_g sits ≈ 2–3e-5 high (slope ratios 1.1–1.4).
* **2 µm is extrapolated, not measured.** The x-and-y scheme was validated at 0.4 and 1.0 µm. At
  2 µm (chunk D) it rests on the same mechanism. A functional check at 1.99–2.0 µm gave
  corrections of 4–5e-5 (n_eff) and 1–2e-4 (n_g) for TE00/TM00/TE10, in line with the 1 µm
  behaviour (`results/check_D/`).

### 2.4 Crossings and anticrossings

* **Opposite-sector pairs cross freely**, and exactly with parity: TE00/TM00 near 1.02 µm, TE00
  passing TM20/TM40, and so on.
* **Same-sector pairs anticross**, because they hybridize through the slanted sidewalls. The
  census shows these order changes between targets and same-sector modes:

  | λ [µm] | pairs |
  |---|---|
  | 0.4–0.5 | TE00–TM50, TE20–TM50, TE10–TM60, TE30–TM60 |
  | ~0.4 | TE30–TM01 |
  | 0.5–0.6 | TE00–TM30, TE10–TM40 |
  | 0.6–0.7 | TE30–TM40 |
  | 0.7–0.8 | TE10–TM20, TE20–TM30 |
  | 0.8–0.9 | TE00–TM10 |
  | 1.5–1.7 | TE10 and TM00 approach (gap ≈ 0.02) |
* **Near an anticrossing n_g changes fast with λ.** It also depends on the detuning, which the
  discretization shifts. Example: TE30 at 0.4 µm, 0.0036 from TM01, whose dy error is ~6× larger:
  * single 25/12.5 nm grid: n_g error −6.0e-4;
  * two-grid scheme: −2.2e-4;
  * dy-only Richardson (12.5, 6.25) at dx = 25 nm: +9.6e-5. Its dy part is only −7e-6; the rest is
    TE30's lateral error at dx = 25 nm.

  That point sits on the wing (gap ≈ 0.004). Within a few nm of a centre, n_g can be off by ~1e-3.
* **Flags.** `combine_chunks.py` flags:
  * same-sector neighbours closer than 0.005 in n_eff;
  * modes whose n_g correction is unusually large compared with their n_eff correction (> 2.5× the
    median ratio at that λ, with a neighbour within 0.03). In the test runs this caught TE30
    (0.4 µm), TE10 (0.79 µm, next to TM20) and TE20 (0.49 µm, next to TM50);
  * hybrid polarizations.
* **If an anticrossing itself matters**, re-solve that λ window with dy = 6.25 nm at dx = 12.5 nm,
  using refine "y" (fine 12.5/6.25, coarse 12.5/12.5). Expected n_g error is ~3e-5, an estimate:
  12.5/6.25 was not solved. Also use a finer λ step there. n_g is computed analytically per λ
  (Hellmann–Feynman), so the step does not limit n_g, but it does limit how well the anticrossing's
  position and gap are resolved.

### 2.5 What the recommended scheme should deliver

`combine_chunks.py` writes, per mode and λ, `*_best` (extrapolated) and an estimate:

`*_err` = 0.3·|correction| + 12·edge_frac + (lateral model, refine "y" only)

The estimate is raised to |n_g correction| at flagged anticrossings, and rows above the tolerance
are flagged. Expected for this geometry:

| region | n_eff | n_g |
|---|---|---|
| all targets, chunks A–C (away from anticrossings) | ≲ 1.5e-5 | ≲ 3e-5 |
| TE00 / TM00 / TE10, chunk D (estimate at 2 µm) | ≲ 6e-5 | ≲ 1e-4 |
| exceptions | TM10 above 1.8 µm window-limited; same-sector anticrossings | see §2.4 |

The `*_err` column is deliberately conservative, typically 2–3× the true error. Near 2 µm it
approaches the tolerance for TM00/TE10 n_g, so a borderline "error estimate > tol" flag there does
not by itself mean the 1e-4 goal failed.

## 3. Your plan, point by point

* **"Re-run the smoothing only N_CHUNKS times."** True, but it is not where the time goes.
  `smoothing_plan` takes 0.2–20 s per grid, mostly a one-time JIT compile, against minutes per
  solve. The real gains from chunking are:
  * a window per chunk: A's fine grid has 82k pixels against 115k for the old 6 × 6 µm grid, even
    at half the lateral pitch (230k at equal pitch);
  * a band count per chunk: cost ∝ bands², with 7 per sector at 0.4 µm but 2 above 1 µm;
  * the parity split itself.
* **Warm starts within chunks.** Yes. They need a fixed grid and they serialize the λ chain. The
  measured saving is ~50 % per solve (§4.1). Keep long chains: a few contiguous runs per worker,
  each costing one cold start. `run_chunked.py` does this.
* **Chunk boundaries.**
  * Keep dy and the alignment fraction identical in all chunks; otherwise a single-grid value would
    jump by Δ(C·dy²) ≈ 1e-4 at a seam.
  * Overlap one wavelength between neighbours.
  * The window, bands and refinement can change between chunks. The extrapolated values should
    then agree across a seam within the error estimates.
  * That agreement has not yet been shown on a full run. `combine_chunks.py` writes
    `boundary_check.txt` for every shared wavelength, so check it on the first real geometry.
* **Chunk edges.** 400–600 / 600–1000 / 1000–2000 is right below 1 µm. Split the long end once
  more, at 1.0–1.5 / 1.5–2.0, because the required window grows ~1.5–2× between 1.5 and 2.0 µm.
  Each extra chunk only costs one cold start and one shared wavelength.
* **Number of eigenvectors.** Count per sector, from §2.1. Don't over-provision: cost goes as
  bands², and the extras are slow cladding/box modes at long λ.
* **Alternatives considered:**
  * One finer grid with no extrapolation: 2–4× the cost, and it still misses n_g at 2 µm.
  * Per-λ windows with no warm start: about 2× the cost, plus small grid-to-grid jitter in n_g.
  * Richardson in dy only: cheaper, but it fails the m = 2, 3 modes at dx = 25 nm and leaves the
    lateral error at dx = 12.5 nm.
  * Adaptive λ refinement around flagged anticrossings is a natural next step; it is not
    implemented.

## 4. Cost and cluster layout

### 4.1 Measured on the desktop (i7-12700, 8 P-cores; `results/timing_slice/`, `results/check_D/`, `results/log_bench.txt`)

`run_chunked.py` ran the chunk table on two neighbouring wavelengths per chunk: 4 workers, each on
2 P-cores (4 logical CPUs), all running at once. Wall time per solve, per worker:

| chunk | fine, cold | fine, warm (10 nm later) | coarse, cold | coarse, warm | warm pair per λ |
|---|---|---|---|---|---|
| A (0.49–0.50) | 1041 s | 514 s | 295 s | 253 s | 767 s |
| B (0.79–0.80) | 480 s | 248 s | 123 s | 72 s | 320 s |
| C (1.24–1.25) | 146 s | 65 s | 38 s | 11 s | 76 s |
| D (1.99–2.00)* | 163 s | 81 s | 39 s | 20 s | 101 s (≈ 150 s with 4 busy workers) |

\* D was timed with only 2 workers busy, so with less memory-bandwidth contention than the other
rows.

* **The warm start halves a solve.** That is much more than the 20 % measured in the handoff,
  because parity sectors and small band counts make the previous λ's eigenvectors a far better
  guess. So keep warm-start chains long: a few units per worker, not one λ per job.
* **Full sweep at 10 nm** (161 λ, both grids, plus ~10 % for cold chain starts):
  * ≈ 46,000 worker-seconds, i.e. **≈ 3.2 h per geometry on the 8 P-cores**;
  * chunk A is ~40 % of that, B ~30 %, D ~20 %;
  * for comparison, the old 6 × 6 µm single-grid sweep (10 bands, no parity) was estimated at
    12–15 h, and it misses 1e-4.
* **CPU scaling of one solve** (fixed LOBPCG iterations, chunk-B grid, one sector):

  | layout | speed-up vs one core |
  |---|---|
  | 1 process × 2 cores | 1.55× |
  | 1 process × 4 cores | 1.87× |
  | 1 process × 8 cores | 2.74× |
  | 2 × 4 / 4 × 2 / 8 × 1 processes × cores | 2.88× / 2.76× / 2.54× |

* **The solver is memory-bandwidth bound.** Every way of dividing the 8 P-cores gives the same total
  throughput, ≈ 2.5–2.9× one core.
* **Memory is small.** Peak use is ≈ 0.7 GB per worker on this grid, and ≲ 1.5–2 GB on the largest
  (D fine, 321k px).

### 4.2 On the cluster

* **Measure once per node type** (5–10 min):

  ```bash
  srun --exclusive -c 48 python bench_cpu.py
  ```

  It prints the node throughput for 1 process × {1…48} cores and for packed layouts. Server nodes
  have 4–8× the desktop's memory bandwidth but less per core, so the per-node throughput should
  keep rising up to somewhere between 16 and 48 busy cores. Use the layout with the highest
  "all processes: N× one core".
* **Then time a slice of the real sweep** and scale it by the number of wavelengths:

  ```bash
  python run_chunked.py --out t --lams 0.50,0.49,0.80,0.79,1.25,1.24,1.75,1.74 --units-per-worker 0
  ```
* **Task size.**
  * Memory bandwidth sets the throughput per node, so one 48-core task per geometry is fine but not
    necessary.
  * Tasks of 12–16 cores per geometry (3–4 workers × 4 cores, the default `--cores-per-worker 4`)
    pack several geometries per node. They also avoid idle cores while a 48-core task waits for its
    last long unit; chunk A's fine-grid chain dominates the tail.
  * Memory is not a constraint: 4 GB/core against ≲ 2 GB per worker.
* **Parallelism inside a task** comes from contiguous λ runs, one chain per worker
  (`--units-per-worker`, default 3). More threads per solve don't help beyond what the node's
  bandwidth allows, and each extra chain costs a cold start (≈ 2× that solve).
* **Output size.** Fine-grid fields (complex64) are ≈ 15–30 MB per λ, ≈ 3 GB per geometry. For a
  large geometry sweep, set `STORE_FIELDS["fine"] = False`, or keep fields only at chosen
  wavelengths. Labels are still computed from the fields before saving.
* **Restarts.** Resubmit the same command. Finished wavelengths are skipped once their metadata
  (chunk, level, grid, geometry, alignment) checks out. A different configuration in the same
  `--out` folder stops the run.

## 5. Files

| file | what |
|---|---|
| `run_chunked.py` | the sweep for one geometry: chunks, pinned worker pool, restartable per-λ HDF5 files |
| `combine_chunks.py` | Richardson + error estimate + flags → `modes_all.csv`, `tracks.csv/.h5`, `dispersion.png`, `boundary_check.txt` |
| `study_chunks.py` | the census / window / pitch / lateral studies (§2) |
| `analyze_studies.py` | tables from the study results → `results/studies_summary.txt` |
| `bench_cpu.py` | cores-per-process benchmark (run it once on a cluster node) |
| `results/` | study data (`*.jsonl`, `studies_summary.txt`), benchmark, `timing_slice/`, `check_D/`, `test_run/` |

```bash
python run_chunked.py --out runs/geom001 --dry-run
python run_chunked.py --out runs/geom001 --geom '{"w_top": 1.6}'
python combine_chunks.py runs/geom001
```

## 6. Caveats

* **The chunk table is for the reference geometry.** A narrower or thinner guide, or air cladding,
  changes the mode census, the window and the lateral error. For a geometry sweep:
  * re-run the census at the chunk edges for the extremes of the sweep; it is cheap;
  * keep the edge_frac window rule;
  * let the flags in `combine_chunks.py` catch outliers.
* **Layer thicknesses must be multiples of the coarse dy (25 nm).** This is checked at start-up;
  otherwise pick dy to suit, e.g. 10 nm. With near-vertical sidewalls the x-Richardson also wants
  w_top/2 to be a multiple of the coarse dx (a warning is printed).
* **Errors are relative to the discretized model** (Kottke-smoothed, periodic window). They say
  nothing about the ~1e-3 mpbx–Tidy3D difference noted in the handoff README.
* **Study timings come from one desktop running 3–4 studies at once.** §4 has the clean numbers.
