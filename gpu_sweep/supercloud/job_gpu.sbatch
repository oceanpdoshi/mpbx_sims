#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
# CHILD script of the gpu_sweep submit scripts. Do not run it directly. The submit scripts set the job
# name, GPUs (--gres), CPUs, time limit, array and log file on the sbatch command line and pass:
#   $1 = run folder, $2 = python, $3 = script under <run>/code/, $4... = its arguments.
# In a job array, task $SLURM_ARRAY_TASK_ID (0-based) handles rows task, task + n_tasks, ... of the
# run's geometries.csv (--n-tasks is among the arguments; only for run_geometry_gpu.py).
# The script runs as ONE process; run_geometry_gpu.py starts one worker process per GPU of the job
# (CUDA_VISIBLE_DEVICES, set by Slurm for --gres=gpu:volta:N, lists them).

RUN_DIR="$1"
PY="$2"
SCRIPT="$3"
shift 3

source /etc/profile                       # SuperCloud: initializes `module` etc. in batch scripts
source "$RUN_DIR/code/gpu_sweep/supercloud/common_env_gpu.sh"

echo "=== $(date '+%F %T')  job ${SLURM_JOB_ID:-?}  array task ${SLURM_ARRAY_TASK_ID:-none}  node $(hostname)"
# CPUs this process may run on. Not plain `nproc`: GNU nproc reports OMP_NUM_THREADS, which
# common_env.sh sets to 1 (one-thread BLAS pools), so it printed "nproc 1" whatever the binding.
usable_cpus() { env -u OMP_NUM_THREADS -u OMP_THREAD_LIMIT nproc; }
echo "    SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-?}  usable CPUs $(usable_cpus)  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "    TMPDIR=${TMPDIR:-unset}  XLA_PYTHON_CLIENT_MEM_FRACTION=$XLA_PYTHON_CLIENT_MEM_FRACTION"
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index,name,memory.total,driver_version,compute_cap --format=csv 2>/dev/null \
        || nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
    DRV="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -n 1 | cut -d. -f1)"
    if [ -n "$DRV" ] && [ "$DRV" -lt 525 ] 2>/dev/null; then
        echo "ERROR: NVIDIA driver $DRV < 525: JAX's CUDA 12 wheels cannot run here (README §4.1)"
        exit 2
    fi
else
    echo "    WARNING: nvidia-smi not found"
fi
echo "    python $PY"

TASK_ARGS=()
if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
    TASK_ARGS=(--task "$SLURM_ARRAY_TASK_ID")
fi

cd "$RUN_DIR"
# srun runs the script as a job step bound to the job's CPUs, as in the CPU sweep's job script. A step
# gets all of the job's GPUs unless it asks for fewer (the --diag job of 2026-11-20 confirmed the GPU).
# The line printed inside the step shows both; usable CPUs should equal the CPUs requested.
LAUNCH=()
STEP="no srun"
if command -v srun >/dev/null 2>&1; then
    LAUNCH=(srun --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-1}" --cpu-bind=cores)
    STEP="srun step"
fi
"${LAUNCH[@]}" bash -c 'echo "    in the job ($0): usable CPUs $(env -u OMP_NUM_THREADS -u OMP_THREAD_LIMIT nproc)  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"; exec "$@"' \
    "$STEP" "$PY" -u "$RUN_DIR/code/$SCRIPT" "${TASK_ARGS[@]}" "$@"
status=$?
echo "=== $(date '+%F %T')  exit status $status"
exit $status
