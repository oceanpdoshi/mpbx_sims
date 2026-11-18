# Environment for every SuperCloud job of this sweep. SOURCED by the job scripts (not run).
# Each line follows the SuperCloud docs (README §4.1):

# HDF5 (h5py) refuses to write on the Lustre file system (home and group directories) unless file
# locking is switched off: "file locking disabled on this file system ... errno = 38".
export HDF5_USE_FILE_LOCKING=FALSE

# Packages installed with `pip install --user` (~/.local) must not shadow the project venv.
export PYTHONNOUSERSITE=1

# Python buffers stdout in batch jobs; without this the log file stays empty for a long time.
export PYTHONUNBUFFERED=1

# Caches go to the node's local disk ($TMPDIR is set by Slurm to a per-job folder on the local disk
# and deleted at the end of the job; the shared file system has no file locking).
LOCAL_DIR="${TMPDIR:-/state/partition1/user/$USER}"
mkdir -p "$LOCAL_DIR"
export JAX_COMPILATION_CACHE_DIR="$LOCAL_DIR/jax_cache"     # shared by the workers of this job
export MPLCONFIGDIR="$LOCAL_DIR/matplotlib"
export XDG_CACHE_HOME="$LOCAL_DIR/cache"

# The worker processes pin themselves to their own cores; keep stray thread pools at one thread.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
