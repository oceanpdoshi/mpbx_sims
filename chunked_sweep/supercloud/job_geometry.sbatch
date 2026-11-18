#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive=user
# CHILD script of submit_single_geometry.sh / submit_geometry_sweep.sh. Do not run it directly.
# The submit scripts set the job name, cores (-c), time limit, array and log file on the sbatch
# command line and pass:   $1 = run folder, $2 = python, $3... = arguments of run_geometry.py.
# In a job array, task $SLURM_ARRAY_TASK_ID (0-based) handles rows task, task + n_tasks, ... of the
# run's geometries.csv (--n-tasks is among the arguments).

RUN_DIR="$1"
PY="$2"
shift 2

source /etc/profile                       # SuperCloud: initializes `module` etc. in batch scripts
source "$RUN_DIR/code/chunked_sweep/supercloud/common_env.sh"

echo "=== $(date '+%F %T')  job ${SLURM_JOB_ID:-?}  array task ${SLURM_ARRAY_TASK_ID:-none}  node $(hostname)"
echo "    SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-?}  usable CPUs (nproc) $(nproc)  TMPDIR=${TMPDIR:-unset}"
lscpu 2>/dev/null | grep -E '^(Model name|Socket\(s\)|Thread\(s\) per core|Core\(s\) per socket):' | sed 's/^/    /'
echo "    python $PY"

TASK_ARGS=()
if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
    TASK_ARGS=(--task "$SLURM_ARRAY_TASK_ID")
fi

cd "$RUN_DIR"
# srun binds the solver to this task's own CPUs. Without it, two tasks sharing a node (e.g. -c 24)
# could both see all 48 CPUs and pin their workers to the same ones (run_geometry.py warns if so).
LAUNCH=()
if command -v srun >/dev/null 2>&1; then
    LAUNCH=(srun --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-1}" --cpu-bind=cores)
fi
"${LAUNCH[@]}" "$PY" -u "$RUN_DIR/code/chunked_sweep/run_geometry.py" --out "$RUN_DIR" "${TASK_ARGS[@]}" "$@"
status=$?
echo "=== $(date '+%F %T')  exit status $status"
exit $status
