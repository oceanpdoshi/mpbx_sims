"""Settings for the GPU (JAX on NVIDIA V100) TFLN mode-solve sweep. Edit THIS file;
run_geometry_gpu.py and bench_gpu.py read it.

Everything not set here is shared with the CPU sweep and comes from
../chunked_sweep/sweep_config.py: the target modes, parity on/off, ALIGN_FRAC, NEV_EXTRA,
MIN_MARGIN, the Newton / LOBPCG tolerances and the table flags. A GPU run and a CPU run therefore
differ only in the grid (and in nothing else unless you change it there).
Units: µm.
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_CPU = os.path.join(os.path.dirname(_HERE), "chunked_sweep")
if _CPU not in sys.path:
    sys.path.insert(0, _CPU)

from sweep_config import ALL_TARGETS, LOW_TARGETS, Chunk  # noqa: E402,F401

# ---------------------------------------------------------------------------------------
# The grid: ONE window and ONE pitch for every wavelength (README §2). The window is rounded to a
# pixel count whose prime factors are cuFFT's fast radices (SMOOTH_PRIMES), at most 2 % smaller
# than asked (the CPU sweep's rule), with an odd Nx for the parity solve:
# 10 x 10 µm at 6.25 nm -> 1575 x 1568 px = 9.844 x 9.800 µm = 2.47 million pixels.
WINDOW = dict(Lx=10.0, Ly=10.0, dx=0.00625, dy=0.00625)

# The wavelength "chunks" no longer change the grid; they set the modes per parity sector (nev) and
# the modes that must be found (targets), exactly as in the CPU sweep. The ranges do not overlap:
# on one grid a boundary wavelength solved twice would be the same solve twice. Each boundary
# wavelength stays with the LOWER chunk, as in the CPU tables (0.60 µm -> A, 1.00 µm -> B, ...).
# Chunks with the same nev share batches on the GPU (D and E here), so giving the long-wavelength
# chunks one common nev makes bigger batches; whether that pays off is what
# `bench_gpu.py --nev-scan` measures (README §3.3).
GPU_CHUNKS = (
    Chunk("A_0400-0600nm", 0.40, 0.60, nev=6, **WINDOW),
    Chunk("B_0610-1000nm", 0.61, 1.00, nev=5, **WINDOW),
    Chunk("C_1010-1300nm", 1.01, 1.30, nev=4, targets=LOW_TARGETS, **WINDOW),
    Chunk("D_1310-1700nm", 1.31, 1.70, nev=3, targets=LOW_TARGETS, **WINDOW),
    Chunk("E_1710-2000nm", 1.71, 2.00, nev=3, targets=LOW_TARGETS, **WINDOW),
)
STEP = 0.010            # wavelength step [µm]; must divide every chunk's range

# cuFFT is fastest for sizes 2^a 3^b 5^c 7^d (other primes fall back to slower kernels).
# Used for GPU_CHUNKS only; `--chunks cpu` keeps the CPU sweep's grids exactly (README §3.2).
SMOOTH_PRIMES = (2, 3, 5, 7)

# ---------------------------------------------------------------------------------------
# GPU memory and batching (README §2.3)
GPU_MEM_GB = 32.0           # memory of one GPU (V100 32 GB). Only used for planning / --dry-run;
                            # the workers read the real size from the device.
MEM_FRACTION = 0.92         # share of the GPU memory JAX may use (XLA_PYTHON_CLIENT_MEM_FRACTION;
                            # JAX's default is 0.75 = 24 GB of 32 GB)
HEADROOM_GB = 2.5           # kept free beside the batched eigensolve: the dielectric tensors of the
                            # wavelengths in flight, one mode's fields during post-processing, cuFFT
                            # plans and workspace (not in XLA's memory analysis), fragmentation
BYTES_PER_PX_BAND = 850     # device memory of one sector solve per (pixel x band): XLA's memory
                            # analysis gives ~825 (nev 3), ~707 (nev 6), ~683 (nev 8) on the CPU backend,
                            # + cuFFT workspace. Planning only: the workers check the real program size.
MAX_LANES = 8               # at most this many wavelengths in flight per GPU (each = 2 sector solves)
MIN_RUN = 3                 # at least this many wavelengths per warm-start chain ("lane")
UNITS_PER_GPU = 5           # work units per GPU and geometry (more = better balance at the end, but
                            # every unit starts cold, which costs ~1.5-2x a warm-started wavelength)
MAX_SLOTS_PER_CALL = None   # cap on sector solves per batched call (None = as many as fit; 1 = one at a
                            # time, no lockstep). bench_gpu.py --max-slots measures which is faster.

# The small dense eigensolves inside LOBPCG (README §2.2): "default" = what JAX picks (cuSOLVER's
# batched Jacobi syevj for these small matrices), "qr" = cuSOLVER syevd, "host" = LAPACK on the CPU
# through a host callback (copies the small matrices off the GPU and back every iteration).
# cuSOLVER needs mpbjax's scaled null-direction penalty (eigensolver._null_penalty, mpbx 2026-11-20):
# with the old 1e12 penalty every solve stalled; with the new one "default", "qr" and "jacobi"
# converge in the CPU's iteration count, ω² within 4e-15 (--diag on a V100, 2026-11-20; README §6).
# batched_modes.set_eigh refuses a cuSOLVER choice on a GPU if mpbjax is the old one.
# No effect on the CPU backend. Override with GPU_SWEEP_EIGH; the benchmark compares "qr" and "host".
GPU_EIGH = "default"

# ---------------------------------------------------------------------------------------
# --test: tiny configuration (minutes, also on a laptop CPU) to check the whole pipeline. One grid
# for all chunks (5.25 x 3 µm, wide enough for 0.9 µm); A and B have the same nev, so their lanes
# share a work unit (lockstep across chunks), C makes a second unit for the second GPU.
TEST_WINDOW = dict(Lx=5.25, Ly=3.0, dx=0.05, dy=0.025)
TEST_CHUNKS = (
    Chunk("A_test", 0.50, 0.60, nev=4, **TEST_WINDOW),
    Chunk("B_test", 0.65, 0.75, nev=4, targets=LOW_TARGETS, **TEST_WINDOW),
    Chunk("C_test", 0.80, 0.90, nev=3, targets=LOW_TARGETS, **TEST_WINDOW),
)
TEST_STEP = 0.050
