# TFLN mode-solve sweep on GPUs (JAX on SuperCloud's V100s)

The GPU counterpart of `chunked_sweep/`. It solves the same modes of the same waveguide with the
same solver (mpbx's LOBPCG and Newton iteration, the same tolerances, parity sectors, mode labels,
nev escalation and result files). The differences:

* **One grid for every wavelength:** 10 × 10 µm at dx = dy = 6.25 nm, which comes out as
  1575 × 1568 px = 2.47 million pixels. The CPU sweep uses 0.1–0.3 Mpx per chunk.
* **One solver process per GPU.** Each GPU solves several sector solves at a time in one batched,
  compiled call: both parity sectors of a wavelength, and several wavelengths where they fit in its
  32 GB. At this grid one wavelength fills a GPU (§2.3).
* **No E fields are stored** (they would be about 100 GB per geometry); everything else is.

The question it is built to answer: **is one GPU node (2 × V100) at least 6× faster than one CPU
node (48 cores)?** That is the break-even of 4 GPU nodes against 24 CPU nodes. `bench_gpu.py`
answers it in 1–3 hours before you commit to full runs (§3).

**Quick start on SuperCloud** (login node, from `~/github/mpbx_sims`, after `git pull`):

```bash
bash gpu_sweep/supercloud/setup_env_gpu.sh                    # once: .venv-gpu (~3 GB)
bash gpu_sweep/supercloud/submit_single_geometry_gpu.sh --check   # ~10 min: driver + GPU vs CPU numbers
bash gpu_sweep/supercloud/submit_single_geometry_gpu.sh --test    # ~15 min: whole pipeline, tiny grids
bash gpu_sweep/supercloud/submit_bench_gpu.sh outputs/tfln_single_2026-11-19-08-51/tLN400_etch400_w1800_sw73_clad1350nmSiO2
bash gpu_sweep/supercloud/submit_single_geometry_gpu.sh           # the full geometry (if the bench says so)
python gpu_sweep/compare_runs.py outputs/tfln_single_2026-11-19-08-51/<geometry> outputs/tfln_gpu_single_<date>/<geometry>
```

---------------------------------------------------------------------------------------------------

## 1. Files

| file | what it is |
|---|---|
| `gpu_config.py` | **the settings**: the window and pitch, chunks (nev and targets per λ range), GPU memory and batching. Everything else (targets, tolerances, parity, NEV_EXTRA, flags) comes from `chunked_sweep/sweep_config.py`. |
| `batched_modes.py` | the batched solver: Newton/LOBPCG for several (wavelength, parity sector) "slots" in one vmapped call; per-mode diagnostics and node-count labels on the device |
| `run_geometry_gpu.py` | the driver: plans grids and work units, one worker process per GPU, writes per-λ files, merges each geometry (with `chunked_sweep/results_io.py`) |
| `bench_gpu.py` | times a few wavelengths per chunk on one GPU and projects the full geometry; compares with a CPU run; nev and batch-size scans (§3) |
| `compare_runs.py` | a CPU run against a GPU run of the same geometry: wall-time speed-up, busy-time speed-up, n_eff / n_g differences per mode |
| `check_gpu_local.py` | the batched GPU path against `sim_core/tfln_modes.solve_wavelength` on a small grid (CPU backend, or GPU vs CPU) |
| `supercloud/setup_env_gpu.sh` | one-time GPU venv (`.venv-gpu`) on a login node |
| `supercloud/submit_single_geometry_gpu.sh` | one geometry on one GPU node (`--check`, `--diag`, `--test`, `--dry-run`, `--resume`) |
| `diag_lobpcg_gpu.py` | why LOBPCG does not converge on the GPU: building blocks, small eigensolves, residual histories (`--diag`) |
| `supercloud/submit_bench_gpu.sh` | the benchmark job (§3) |
| `supercloud/submit_geometry_sweep_gpu.sh` | many geometries: a job array of whole GPU nodes, at most `MAX_NODES` at a time |
| `supercloud/job_gpu.sbatch`, `common_env_gpu.sh`, `lib_submit_gpu.sh` | used by the submit scripts (they reuse `chunked_sweep/supercloud/`'s helpers) |

Nothing in `chunked_sweep/`, `sim_core/`, `geometry_gen/` or `utils/` was changed. The GPU code
imports their planning, geometry, solver, labelling and file-merging code, so a fix there applies
to both pipelines.

## 2. How it works

### 2.1 Grid and chunks

`WINDOW = 10 × 10 µm, 6.25 nm` is rounded to pixel counts made only of cuFFT's fast radices
(2, 3, 5, 7), at most 2 % below the requested window (the CPU sweep's rule), with an odd Nx for
the parity solve: **1575 × 1568 px = 9.844 × 9.800 µm**. The window leaves ≥ 3.8 µm of cladding
around the ridge, ≥ 1.9 λ even at 2 µm, so the window error of the CPU sweep's chunks D and E
(TM10 near cutoff) is much smaller here.

The chunks keep the CPU sweep's modes per parity sector and target modes:

| chunk | λ [µm] | nev / sector | targets |
|---|---|---|---|
| A | 0.40–0.60 | 6 | all 8 |
| B | 0.61–1.00 | 5 | all 8 |
| C | 1.01–1.30 | 4 | TE/TM 00, 10 |
| D | 1.31–1.70 | 3 | TE/TM 00, 10 |
| E | 1.71–2.00 | 3 | TE/TM 00, 10 |

The ranges do not overlap: on a single grid, a boundary wavelength solved by two chunks would be the
same solve twice, so there are no chunk seams. Each boundary wavelength stays with the lower chunk,
as in the CPU tables.

**Work:** 2.47 Mpx for all 161 λ is about **17× the CPU baseline's work** by the CPU sweep's cost
model (pixels × modes², 393 vs 23 desktop-core-hours). Discretization error falls as dy²
(chunked_sweep README §2.2), so n_eff / n_g errors should come out ~4× smaller than on the CPU
sweep's 12.5 nm grids (≈ 2e-5 instead of ≈ 8e-5 in n_eff).

### 2.2 The batched solve

A **slot** is one x-parity sector of one wavelength. `batched_modes.newton_step` runs mpbx's
`solve_omega2` (LOBPCG) vmapped over slots. Each slot has its own k, ε⁻¹ (its wavelength), LOBPCG
block and parity sign. The sign is a traced value, so one compiled program serves both sectors and
every wavelength of the grid.

The Newton iteration on k (`tfln_modes.solve_k_checked`) runs for all slots in lockstep:

* For each band, one batched eigensolve per Newton step.
* A slot that has converged (|Δk| < k_tol) is frozen: its result is recorded at that step, exactly
  where the CPU code stops. It then rides along until the slowest slot of the batch converges.
* A cold slot starts from the same random block LOBPCG draws for the CPU's cold solve, so a cold
  solve here repeats the CPU computation exactly.

After the solve, the per-mode quantities are computed **on the GPU**: power normalization, TE / LN
fraction, edge fraction, HG00 overlap, and the two field lines whose sign changes give the
node-count label. No field ever leaves the device.

**One GPU-only setting: the small dense eigensolves inside LOBPCG** (Gram and Rayleigh-Ritz
matrices, ≤ 3·nev square). mpbjax's Rayleigh-Ritz step pushes rank-deficient ("null") trial
directions to the top of the spectrum with a diagonal penalty. **This needs mpbx with the scaled
penalty** (`eigensolver._null_penalty`, 2026-11-20). The old penalty was a 1e12 wall: it made the
small matrix's norm ~1e12× its Ritz values, and cuSOLVER's stopping / deflation tests are relative
to that norm, so it ignored the small couplings LOBPCG needs near convergence; on a V100 every
solve stalled at a residual of ~4e-5 (§6). The new penalty, 2·‖M‖_F, sits just above every kept
Ritz value; on the CPU it changes nothing measurable.

`gpu_config.GPU_EIGH` picks the small eigensolver:

* `"default"` (production): what JAX picks, cuSOLVER's batched Jacobi solver for these sizes.
* `"qr"`: cuSOLVER's `syevd`. `"jacobi"`: `syevj` explicitly.
* `"host"`: LAPACK on the CPU through `jax.pure_callback`; each iteration copies the small
  matrices (a few kB) off the GPU and back. The only choice that works with an old mpbx.
* On the V100 all four converge in the CPU's iteration count (`--diag`); the benchmark times
  chunk A with `"qr"` and `"host"` against `"default"` (§3.3).
* With an old mpbx, a cuSOLVER choice on a GPU stops the worker with an error (update mpbx, or
  set `GPU_SWEEP_EIGH=host`).
* Only `mpbjax.eigensolver` inside the GPU processes is switched; on the CPU backend every choice
  is the same LAPACK routine.

**Checked on the desktop (CPU backend, `check_gpu_local.py`):** against
`tfln_modes.solve_wavelength`, n_eff agrees to 3e-15, n_g to 2e-9, with the same labels, the same
mode count, the same Newton step counts and LOBPCG iteration counts within 1 (the vmapped program
orders some sums differently). Two wavelengths in one call (4 slots) give bit-identical results to
each wavelength alone.

**Checked end to end:** `run_geometry_gpu.py --test --chunks cpu --platform cpu --gpus 2` reproduces
`run_geometry.py --test` to 3e-15 in n_eff and 4e-10 in n_g. It also has the same nev escalation
at 0.60 µm and the same labels at every wavelength.

One known difference: `power_raw` (the raw ∫S_z before normalization) can have the opposite sign
from the CPU file for some odd modes. In the CPU path, the phase fixing in mpbx's `e_field` and
`h_field` picks its reference pixel from two mirror-symmetric pixels by rounding. Only |P| is used.

### 2.3 GPU memory: how many wavelengths fit

One sector solve needs about **680–830 bytes per pixel per band**. This is XLA's own memory
analysis of the compiled program: 825 B at nev 3, 707 at nev 6, 683 at nev 8. Add cuFFT workspace,
plus the eigenvector copies kept beside the program (64 B per pixel per band). LOBPCG keeps about
17 blocks of (2·Nx·Ny × nev) complex128 alive. At 2.47 Mpx (planning estimate,
`BYTES_PER_PX_BAND` = 850 + 64):

| nev / sector | one sector solve | wavelengths in flight per 32 GB GPU (2 sectors each) |
|---|---|---|
| 6 (A) | ~12.8 GiB | 1 |
| 5 (B) | ~10.7 GiB | 1 |
| 4 (C) | ~8.6 GiB | 1 |
| 3 (D, E) | ~6.5 GiB | 1 (2 would need ~27 GiB beside what is resident) |
| 8 (A's escalation, nev + 2) | ~17 GiB | 1, the sectors one after the other |

So at this grid, "half of the wavelengths × all modes" does not fit: one wavelength's two sectors
already take about half to all of the usable GPU memory. Two consequences:

* **The batch is the two parity sectors of one wavelength**, in one call.
* **At 2.5 Mpx the GPU is busy with a single solve anyway.** Each LOBPCG iteration is 12·nev FFTs
  of 2.5 M points. Batching several wavelengths pays off at small grids, where one solve leaves the
  GPU idle: at the CPU sweep's 0.1–0.3 Mpx grids, 8 wavelengths (16 slots) per GPU fit.

The same code batches as many wavelengths as fit when the grid is smaller or nev lower. The
benchmark's lanes scan measures whether batching helps at this grid (§3.3).

The planner uses the estimate above (`BYTES_PER_PX_BAND`) to decide how many wavelengths a work
unit carries. Each worker then compiles the batched program of the size it is about to run, and
reads its exact buffer sizes from XLA. "Fits" means: within `MEM_FRACTION` = 92 % of the GPU,
next to what is already on it, minus `HEADROOM_GB`, and within the allocator's largest free block.
If it does not fit, the worker splits the slots into the fewest equal calls that do, and remembers
that limit for the rest of the geometry. If the GPU still runs out of memory, it halves the batch
and repeats the step.

### 2.4 Scheduling

* **Lanes:** each chunk's wavelengths are cut into contiguous runs, solved long → short so that
  every wavelength after the first is warm-started from its neighbour (fewer Newton and LOBPCG
  steps).
* **Work units:** lanes of the same nev (all chunks share the grid) are grouped, as many per unit
  as fit on a GPU. A unit's lanes advance in lockstep, one batched solve per step.
* **Scheduling:** about 5 units per GPU, sorted by estimated cost and handed out to whichever GPU
  process is free. A geometry is merged as soon as its last unit finishes.

`--dry-run` prints the units. For the default geometry there are 10 units of one lane each:
D 40 λ, B 14 + 13 + 13 λ, E 30 λ, A 3 × 7 λ, C 2 × 15 λ.

A nev escalation (a target found only with nev + 2) is kept for the rest of its lane, as on the
CPU, where a work unit is 3–5 λ. GPU lanes are longer (7–40 λ), so an escalation costs more here:
the summary lists every one of them.

**Hardware use:** the job takes one whole GPU node (2 GPUs, 40 cores, 384 GB). Each GPU process
gets 20 cores for its host work (dielectric per λ, Newton bookkeeping, file writing). The cores do
no solving: this is a clean GPU vs CPU comparison. Hybrid use of the cores could be added later.

**More modes at long wavelengths:** lanes of chunks with the same nev share batches. So a common,
larger nev for C–E (e.g. 4) gives bigger batches and fewer compiled programs, but only where
several wavelengths fit on a GPU, which at 2.47 Mpx they don't. The cost side: the extra modes
above ~1.1 µm are unguided cladding modes, which come in dense, near-degenerate clusters that can
slow LOBPCG. The bench's `--nev-scan` measures that cost (§3.3).

## 3. Is it faster? The 6× question

### 3.1 The CPU baseline

Run `tfln_single_2026-11-19-08-51`, job 6281540: one xeon-p8 node, 48 workers × 1 core, fields
on, code a7003e8. Note that this commit used the older windows (4 × 4 … 8 × 8 µm,
0.10–0.20 Mpx per chunk).

| | CPU node | needed from the GPU node for 6× |
|---|---|---|
| one geometry, wall time | **2.44 h** | ≤ 24 min |
| busy time per geometry, node kept full (Σ per-λ worker time ÷ 48 = 44.8 core-h ÷ 48) | **0.93 h** | ≤ 9.3 min |

The 2.44 h is mostly a scheduling tail: 46 of 49 work units were done by 1.3 h. The last unit
held the 0.71 µm nev escalation and finished at 2.4 h. In a sweep of many geometries the next
geometry fills that tail. So **the busy-time number is the fair one for the 24 vs 4 node
decision**. `compare_runs.py` and `bench_gpu.py` report both.

### 3.2 What to expect (estimates, not measurements)

Nothing in mpbx has run on a GPU before. The mpbx docs contain only roofline models for
H100/A100, and the "10–100×" figure there is against one CPU core, not a node. The solver is
memory-bandwidth bound (FFTs and thin matrix products on complex128):

* 2 × V100 have 2 × 900 GB/s of memory bandwidth.
* A 2-socket Xeon 8260 node has ~280 GB/s.
* So **for equal work, expect ≲ 6× node vs node**, possibly less: the CPU node was measured at only
  ~17× one core, i.e. it is not at its bandwidth limit either.
* **At the GPU grid (17× the work), 6× over the CPU's wall time would need ~100× per unit of work.**
  That is very unlikely by this argument.

The likely outcome is that the GPU node's value lies in **accuracy per hour**, not in raw speed at
equal accuracy: the 6.25 nm / 10 µm grid is out of reach of the CPU sweep. This is exactly what the
benchmark is for: measure it before deciding.

### 3.3 The benchmark (`submit_bench_gpu.sh`, ~1–3 h, one GPU node)

Both GPUs work at the same time. All timings go through the production code (`run_unit`). Each
timed unit follows an untimed warm-up of the same batch shape, so compiles and first calls are
measured apart and added to the projection once.

* **GPU 1, the GPU grids:** 3 wavelengths per lane in every chunk (the first cold, the rest
  warm-started). The script projects the full geometry on 2 GPUs: the production units, in the
  order the driver hands them out, plus compile time. It prints the projected speed-up over the
  CPU run's wall time against the threshold.
* **GPU 2:**
  * **The CPU run's own grids (equal work).** Per chunk, node vs node: the CPU's measured per-λ
    times are taken with 48 workers busy, and the GPU's with its production batch size.
  * **nev scan at 1.5 µm (3, 4, 6):** what more modes per sector cost at long wavelengths.
  * **Lanes scan (1, 2, 3 wavelengths per unit) at 1.5 µm:** whether batching wavelengths helps
    at this grid. Where they don't fit in one call, the log's "slots per call" shows the split.
  * **Chunk D with `--max-slots 1`:** one sector solve per call, i.e. no lockstep. Compare its
    s/λ with GPU 1's D. Lockstep runs every slot of a call until the slowest converges; if
    one-at-a-time is faster, set `MAX_SLOTS_PER_CALL = 1` in `gpu_config.py`.
  * **Chunk A with `--eigh qr` and with `--eigh host`:** the other small eigensolvers (§2.2),
    against GPU 1's A with `"default"`. If one is clearly faster, set `GPU_EIGH` to it.

Read `logs/bench_gpu_grids.log` and `logs/bench_cpu_grids_and_scans.log`; the numbers are also in
`bench*/*.json`.

The comparison at **equal grids** is also available as a full run:

```bash
CHUNKS=cpu   # in submit_single_geometry_gpu.sh: the CPU sweep's chunks and grids, on the GPU
```

These are the grids of the current `sweep_config.py`. The baseline used older windows, so for an
exactly equal-work full run, check out the baseline's `sweep_config.py` (commit a7003e8) first.
The benchmark's `--grids cpu-run` mode needs no checkout: it reads the baseline's grids from its
`_modes.h5`.

## 4. Running on SuperCloud

### 4.1 One-time setup (login node)

```bash
cd ~/github/mpbx_sims && git pull
bash chunked_sweep/supercloud/setup_env.sh     # only if not done yet (uv + Python 3.12 + CPU venv)
bash gpu_sweep/supercloud/setup_env_gpu.sh
```

This creates `.venv-gpu` with exactly the CPU venv's versions (from `uv.lock`), plus JAX's CUDA 12
plugin and NVIDIA's CUDA 12 libraries as pip wheels (~3 GB in your home directory). Why these
exact pins:

* **V100 needs the CUDA 12 wheels.** The V100 is compute capability 7.0. JAX's CUDA 13 wheels need
  ≥ 7.5, because CUDA 13 dropped Volta, so it has to be `jax[cuda12]`.
* **Pin JAX exactly (`==0.12.0.1`).** JAX's maintainers announced that CUDA 12 support ends with
  0.12.2 (15 Dec 2026), so 0.12.1 will be the last release with CUDA 12 wheels. Never let pip pick
  "the latest jax".
* **cuDNN 9.10.2:** the last cuDNN that lists Volta, and the one JAX's CUDA 12 build is compiled
  against. The mode solver itself uses no cuDNN routine.
* **Same jax/jaxlib as the CPU venv:** a CPU run and a GPU run differ only in XLA's backend.
* **NVIDIA driver ≥ 525** is needed on the GPU nodes. The SuperCloud docs don't say which driver
  they have. `--check` prints it, and every job stops with a message if it is too old.
* **Do not `module load cuda`.** The job scripts drop CUDA paths from `LD_LIBRARY_PATH` so that JAX
  uses its own libraries.
* **Re-running the setup rebuilds `.venv-gpu`.** It refuses while you have jobs queued or running,
  since they use that venv; `FORCE=1` overrides.

### 4.2 Check, test

```bash
bash gpu_sweep/supercloud/submit_single_geometry_gpu.sh --check   # 1 GPU, ~10 min
bash gpu_sweep/supercloud/submit_single_geometry_gpu.sh --diag    # 1 GPU, ~10-20 min, if --check does not converge
bash gpu_sweep/supercloud/submit_single_geometry_gpu.sh --test    # 2 GPUs, ~15 min
```

* **`--check`** prints the GPUs and driver. It then solves a small grid with the CPU reference
  (`tfln_modes`, on the node's CPU) and with the batched path on the GPU, and compares every
  quantity. Look for `ALL CHECKS PASSED` in `outputs/gpu_check_<date>/logs/`. LOBPCG iteration
  counts may differ by a few: sums are ordered differently on the GPU.
  The production eigensolver (`GPU_EIGH`) runs first and decides the verdict; a wavelength in
  which any LOBPCG solve runs to its 1000-iteration cap stops the check with `NOT CONVERGED`.
* **`--diag`** (if `--check` fails to converge): `diag_lobpcg_gpu.py` takes one eigensolve apart:
  FFT / operator / preconditioner on the GPU vs the CPU, cuSOLVER vs LAPACK on the real small
  matrices of a GPU LOBPCG run, and the LOBPCG residual history with each small eigensolver,
  including `"host"` (LAPACK through a host callback). Its `VERDICT` says which variant converges.
* **`--test`** runs the whole pipeline (3 small chunks, 9 λ, both GPUs, merge) into
  `outputs/tfln_gpu_single_TEST_<date>/`.

### 4.3 Benchmark (§3.3), then one geometry

```bash
bash gpu_sweep/supercloud/submit_bench_gpu.sh outputs/tfln_single_2026-11-19-08-51/tLN400_etch400_w1800_sw73_clad1350nmSiO2
bash gpu_sweep/supercloud/submit_single_geometry_gpu.sh --dry-run
bash gpu_sweep/supercloud/submit_single_geometry_gpu.sh
```

The settings block at the top of `submit_single_geometry_gpu.sh` holds:

* the geometry (preset to the baseline's: w_top 1.8, t_clad 1.35), including the rib type
  `T_UNDERCUT` (0 = regular rib, > 0 = undercut rib; chunked_sweep/README.md §4.3);
* `CHUNKS` (`gpu` or `cpu`);
* `GPUS`;
* `TIME`: set it from the benchmark's projection plus margin; the default is 12 h.

The job is one Slurm task with `--gres=gpu:volta:2 -c 40`. `run_geometry_gpu.py` starts one
worker process per GPU (Slurm's `CUDA_VISIBLE_DEVICES`), each pinned to 20 cores. Results go to
`outputs/tfln_gpu_single_<date>/<geometry>/` (§5).

### 4.4 Many geometries

```bash
bash gpu_sweep/supercloud/submit_geometry_sweep_gpu.sh my_geoms.csv --dry-run
bash gpu_sweep/supercloud/submit_geometry_sweep_gpu.sh my_geoms.csv
```

The sweep is one job array of whole GPU nodes, at most `MAX_NODES` = 4 at a time (your
allocation). With more geometries than `MAX_TASKS`, a task solves several geometries one after the
other. A collect job then writes `run_summary.csv` and `all_tracks.csv`, the same as the CPU sweep.

### 4.5 Watching, resuming, failures

* **Watching:** use `LLstat` and `tail -f` on the log. `LLload -g --detail` shows how busy the
  GPUs are; it should be close to 2.0 for a 2-GPU job.
* **Resume:** `--resume outputs/<run>` works as on the CPU. Solved wavelengths are kept, and each
  lane continues cold at the next missing wavelength.
* **A batch that fails** marks its wavelengths failed and the unit carries on. The geometry is then
  merged as incomplete, and a resume retries exactly the failed wavelengths.
* **Re-merging:** `run_geometry_gpu.py ... --merge-only` re-merges from `work/` and needs no GPU,
  so it can run on a login node.
* **Out of GPU memory:** the batch is halved and retried automatically. If it still fails with one
  slot, lower `WINDOW` or the pitch (or nev) in `gpu_config.py`. The worker logs show the memory
  each compiled program needs, and the report shows each GPU's peak.

## 5. Output files

Per geometry, `<run>/<geometry name>/` holds the same files as the CPU sweep (chunked_sweep
README §5), all read by `chunked_sweep/results_io.py` (`load_modes`, `load_tracks`, ...):

* `<name>_modes.h5`, `_modes.csv`, `_tracks.csv`, `_summary.txt`, `_dispersion.png`;
* no `_fields_*.h5`;
* no chunk seams (one grid).

There are two extra files:

| file | content |
|---|---|
| `<name>_gpu_timing.jsonl` | one line per solved λ: GPU-seconds (the batch's time shared by its wavelengths + own post-processing), batch sizes, Newton / LOBPCG maxima, escalation, peak GPU memory |
| `<name>_gpu_report.txt` | wall time; per GPU: device, peak memory, time spent solving; per chunk: median / mean / max GPU-s per λ, iterations, escalations; compiled programs and their memory |

In `_modes.h5` / `_summary.txt`, `solve_time` and "median s/λ" are those GPU-seconds per λ, and
"peak" is the host memory. The GPU memory is in the report.

## 6. What was and was not tested

* **Tested on the desktop (CPU backend, JAX 0.12.0.1):**
  * `check_gpu_local.py`;
  * `run_geometry_gpu.py --test` with both chunk sets and 2 worker processes, including lockstep
    lanes across chunks, nev escalation, merge and report;
  * agreement with `run_geometry.py --test` to 1e-14 / 1e-9;
  * `--merge-only` with no GPU visible;
  * a stub test of a lane escalating while another lane already runs at the raised nev (each
    wavelength solved once);
  * `bench_gpu.py` with `--grids cpu-run`, `--nev-scan`, `--lanes-scan`, `--max-slots 1` and
    `--eigh`;
  * `compare_runs.py`;
  * `--dry-run` of the production plan;
  * `bash -n` of every shell script and of the generated benchmark job.
* **Independent review:** the code was reviewed adversarially in four parts (numerics vs the CPU
  path, orchestration, JAX/GPU specifics, scripts and benchmark logic). The confirmed defects are
  fixed. Among them were a batch-size cap that only ever shrank and a double solve after a nev
  escalation; the JAX/GPU part also prompted the eigensolver choice (§2.2).
* **First SuperCloud run (2026-11-20, `--check` on one V100, driver 580):** `setup_env_gpu.sh`, the
  `--gres` request and the driver check worked. **LOBPCG did not converge on the GPU:** with
  `eigh "default"` every eigensolve ran to its 1000-iteration cap (CPU: ~50 per solve), n_eff came
  out 2.5e-5 off, and the `"qr"` chain did not finish two wavelengths in 23 min (the CPU needs ~30 s);
  the 30-min job hit its time limit. (The log's `nproc 1` was not a binding problem: GNU `nproc`
  reports `OMP_NUM_THREADS`, which `common_env.sh` sets to 1. The job log now prints the real
  count. `job_gpu.sbatch` starts the script with `srun --cpu-bind=cores` anyway, as the CPU sweep does.)
* **`--diag` on a V100 (2026-11-20):** FFT, operator, preconditioner and Gram products agree with
  the CPU to ≤ 2e-15. The small eigensolves do not: on the recorded Rayleigh-Ritz matrices,
  cuSOLVER's eigenvalues differ from LAPACK's by 2e-3 (`default`, `jacobi`) and 5e-3 (`qr`)
  relative, while Gram matrices agree to 5e-14. LOBPCG with any cuSOLVER variant stalls at a
  residual of 4e-5 (`qr`: 5e-4, then diverging); with `"host"` it converges in 72 iterations, as on
  the CPU, with ω² equal to 2e-15. Production switched to `GPU_EIGH = "host"` until the fix below.
* **Root cause and mpbjax fix (2026-11-20, CPU only so far):** the 1e12 penalty on null directions
  in mpbjax's Rayleigh-Ritz step (§2.2). Emulating cuSOLVER's norm-relative deflation on the CPU
  (off-diagonal entries below 1e-14·‖M‖_F ignored) reproduces the V100 stall exactly (residual
  4.2e-5 for 1000 iterations); with the penalty scaled to the kept block (`_null_penalty`) the same
  solve converges in 72 iterations with ω² unchanged. mpbjax has a regression test for it.
* **`--diag` on a V100 with the fixed mpbjax (2026-11-20): confirmed.** The recorded Rayleigh-Ritz
  matrices now agree with LAPACK to 8e-15 (was 2e-3); `default`, `qr`, `jacobi` and `host` all
  converge in 72 iterations, as on the CPU, with ω² within 4e-15. Production is back to
  `GPU_EIGH = "default"`. The job step had 40 usable CPUs for the 20 requested (hyper-threads or
  no core binding on SuperCloud; harmless). Next: `--check`, then `--test` and the benchmark.
  **Do not use the GPU sweep for results until `--check` passes.**
* **Not run anywhere yet:**
  * a converging GPU solve: memory figures on the V100, compile times, speed;
  * the array scripts on SuperCloud.

  Run `--check`, `--test` and the benchmark, in that order, before a production run.
* mpbx itself has never been validated on a GPU (its own docs say so). `--check` compares GPU and
  CPU numbers directly.
