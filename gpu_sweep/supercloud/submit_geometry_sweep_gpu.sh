#!/bin/bash
# Submit MANY geometries to the GPU nodes as a Slurm job array: one array task = one GPU node
# (2 x V100), at most MAX_NODES of them at a time; task k solves rows k, k + N_TASKS, ... of the
# geometry list. A small CPU job (afterany) writes the run-level tables over all geometries.
# Run on a SuperCloud LOGIN node:
#
#   python chunked_sweep/make_geometries.py my_geoms.csv                  (or write the CSV by hand)
#   bash gpu_sweep/supercloud/submit_geometry_sweep_gpu.sh my_geoms.csv --dry-run
#   bash gpu_sweep/supercloud/submit_geometry_sweep_gpu.sh my_geoms.csv
#   bash gpu_sweep/supercloud/submit_geometry_sweep_gpu.sh --resume outputs/<run folder>

# ============================== settings ==============================
RUN_NAME="tfln_gpu_sweep"
CHUNKS=gpu              # gpu = gpu_config.py | cpu = chunked_sweep/sweep_config.py's grids
GPUS=2                  # GPUs per array task (= one whole node)
MAX_NODES=4             # array tasks running at the same time (your GPU-node allocation)
TIME=24:00:00           # time limit PER ARRAY TASK (several geometries per task if N_GEOM > MAX_TASKS)
MAX_TASKS=64            # at most this many array tasks; with more geometries a task solves several
PARTITION=""
OUT_ROOT=""
# ======================================================================

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib_submit_gpu.sh"
OUT_ROOT="${OUT_ROOT:-$REPO/outputs}"

MODE=submit
RESUME=""
CSV=""
while [ $# -gt 0 ]; do
    case "$1" in
        --dry-run) MODE=dry; shift ;;
        --resume) RESUME="$2"; shift 2 ;;
        -*) echo "unknown option $1"; exit 1 ;;
        *) CSV="$1"; shift ;;
    esac
done
if [ -n "$RESUME" ] && [ "$MODE" != submit ]; then
    echo "ERROR: --resume cannot be combined with --dry-run"
    exit 1
fi
check_env_gpu

TMP_RUN=0
if [ -n "$RESUME" ]; then
    RUN_DIR="$(cd "$RESUME" && pwd)"
    [ -f "$RUN_DIR/run_args.txt" ] || { echo "ERROR: $RUN_DIR/run_args.txt not found (not a run folder?)"; exit 1; }
    read -r -a ARGS < "$RUN_DIR/run_args.txt"
    echo "resuming $RUN_DIR"
else
    [ -n "$CSV" ] && [ -f "$CSV" ] || { echo "usage: $0 GEOMETRIES.csv [--dry-run]   or   $0 --resume RUN_DIR"; exit 1; }
    if [ "$MODE" = dry ]; then
        RUN_DIR="$(mktemp -d)"
        TMP_RUN=1
    else
        RUN_DIR="$(new_run_dir "$OUT_ROOT" "$RUN_NAME")"
    fi
    snapshot_code_gpu "$RUN_DIR"
    cp "$CSV" "$RUN_DIR/geometries.csv"
    ARGS=(--chunks "$CHUNKS" --gpus "$GPUS")
    echo "${ARGS[*]}" > "$RUN_DIR/run_args.txt"
fi

N_GEOM=$(HDF5_USE_FILE_LOCKING=FALSE PYTHONNOUSERSITE=1 "$PY" -c "
import sys; sys.path.insert(0, '$RUN_DIR/code/chunked_sweep')
from run_geometry import read_geometry_csv
print(len(read_geometry_csv('$RUN_DIR/geometries.csv')))")
N_TASKS=$(( N_GEOM < MAX_TASKS ? N_GEOM : MAX_TASKS ))
ARGS+=(--geom-csv "$RUN_DIR/geometries.csv" --n-tasks "$N_TASKS" --out "$RUN_DIR")
echo "$N_GEOM geometries -> $N_TASKS array tasks x $GPUS GPUs, at most $MAX_NODES at a time"

if [ "$MODE" = dry ]; then
    HDF5_USE_FILE_LOCKING=FALSE PYTHONNOUSERSITE=1 JAX_PLATFORMS=cpu "$PY" "$RUN_DIR/code/gpu_sweep/run_geometry_gpu.py" \
        "${ARGS[@]}" --dry-run
    [ "$TMP_RUN" = 1 ] && rm -rf "$RUN_DIR"
    exit 0
fi
write_run_info "$RUN_DIR" "submit    submit_geometry_sweep_gpu.sh: $N_GEOM geometries, $N_TASKS array tasks x $GPUS GPUs (max $MAX_NODES at a time), $TIME" \
                          "args      ${ARGS[*]}"

JOBID=$(sbatch --parsable -J "$RUN_NAME" $(gpu_request "$GPUS") --time="$TIME" --array=0-$((N_TASKS - 1))%"$MAX_NODES" \
        -o "$RUN_DIR/logs/%x_%A_%a.out" \
        "$RUN_DIR/code/gpu_sweep/supercloud/job_gpu.sbatch" "$RUN_DIR" "$PY" gpu_sweep/run_geometry_gpu.py "${ARGS[@]}")
CJOB=$(sbatch --parsable -J "${RUN_NAME}_collect" -c 2 --time=02:00:00 --dependency=afterany:"$JOBID" \
       -o "$RUN_DIR/logs/%x_%j.out" "$RUN_DIR/code/chunked_sweep/supercloud/job_collect.sbatch" "$RUN_DIR" "$PY")
echo "array job $JOBID, collect job $CJOB" >> "$RUN_DIR/run_info.txt"
echo "submitted array job $JOBID ($N_TASKS tasks, $MAX_NODES at a time) and collect job $CJOB -> $RUN_DIR"
echo "  status:   LLstat    (or: squeue -u $USER);  GPU load: LLload -g --detail"
echo "  per task: sacct -j $JOBID -X --format=JobID%20,State%15,ExitCode,Elapsed,NodeList%20"
echo "  resume:   bash $GPU_SC_DIR/submit_geometry_sweep_gpu.sh --resume $RUN_DIR"
