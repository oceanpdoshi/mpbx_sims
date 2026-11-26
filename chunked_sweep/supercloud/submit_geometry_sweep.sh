#!/bin/bash
# Submit MANY waveguide geometries as a Slurm job array: array task k solves rows k, k + N_TASKS, ...
# of the geometry list, each geometry with CPUS cores (README §3: 24 cores = two geometries per node).
# A small follow-up job (afterany) writes run-level tables over all geometries.
# Run on a SuperCloud LOGIN node:
#
#   python chunked_sweep/make_geometries.py my_geoms.csv            (or write the CSV by hand)
#   bash chunked_sweep/supercloud/submit_geometry_sweep.sh my_geoms.csv
#   bash chunked_sweep/supercloud/submit_geometry_sweep.sh my_geoms.csv --dry-run
#   bash chunked_sweep/supercloud/submit_geometry_sweep.sh --resume outputs/<run folder>
#
# The SuperCloud docs prefer LLsub "triples" over large job arrays (fewer log files, whole-node
# scheduling); submit_geometry_sweep_triples.sh does the same sweep that way. With up to a few
# hundred geometries either works; staff ask for triples before granting more than 8 nodes.

# ============================== settings ==============================
RUN_NAME="tfln_sweep"
FIELDS=none             # all | none   (all: ~3 GB per geometry; check `LLquota` first)
CPUS=24                 # cores per geometry (48 = full node; 24 = two per node; 16 = three per node)
CORES_PER_WORKER=1      # cores per mode-solver process (README §3)
TIME=24:00:00           # time limit PER ARRAY TASK (one geometry at 24 cores: ~4-8 h estimated)
MAX_TASKS=240           # at most this many array tasks; with more geometries a task solves several
OUT_ROOT=""             # empty = <repo>/outputs ; e.g. /home/gridsan/groups/<group>/tfln_outputs
# ======================================================================

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib_submit.sh"
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
check_env

TMP_RUN=0
if [ -n "$RESUME" ]; then
    RUN_DIR="$(cd "$RESUME" && pwd)"
    [ -f "$RUN_DIR/run_args.txt" ] || { echo "ERROR: $RUN_DIR/run_args.txt not found (not a run folder?)"; exit 1; }
    read -r -a ARGS < "$RUN_DIR/run_args.txt"
    echo "resuming $RUN_DIR"
else
    [ -n "$CSV" ] && [ -f "$CSV" ] || { echo "usage: $0 GEOMETRIES.csv [--dry-run]   or   $0 --resume RUN_DIR"; exit 1; }
    check_geometry_csv "$CSV"                              # rib types; a bad row stops here
    if [ "$MODE" = dry ]; then
        RUN_DIR="$(mktemp -d)"
        TMP_RUN=1
    else
        RUN_DIR="$(new_run_dir "$OUT_ROOT" "$RUN_NAME")"
    fi
    snapshot_code "$RUN_DIR"
    cp "$CSV" "$RUN_DIR/geometries.csv"
    ARGS=(--fields "$FIELDS" --cores-per-worker "$CORES_PER_WORKER")
    echo "${ARGS[*]}" > "$RUN_DIR/run_args.txt"            # reused by --resume
fi

# number of geometries = data rows of the CSV (validated by run_geometry.py)
N_GEOM=$(HDF5_USE_FILE_LOCKING=FALSE PYTHONNOUSERSITE=1 "$PY" -c "
import sys; sys.path.insert(0, '$RUN_DIR/code/chunked_sweep')
from run_geometry import read_geometry_csv
print(len(read_geometry_csv('$RUN_DIR/geometries.csv')))")
N_TASKS=$(( N_GEOM < MAX_TASKS ? N_GEOM : MAX_TASKS ))
ARGS+=(--geom-csv "$RUN_DIR/geometries.csv" --n-tasks "$N_TASKS")
echo "$N_GEOM geometries -> $N_TASKS array tasks x $CPUS CPUs ($(( (N_GEOM + N_TASKS - 1) / N_TASKS )) geometries per task at most)"

if [ "$MODE" = dry ]; then
    HDF5_USE_FILE_LOCKING=FALSE PYTHONNOUSERSITE=1 "$PY" "$RUN_DIR/code/chunked_sweep/run_geometry.py" \
        --out "$RUN_DIR" "${ARGS[@]}" --max-cores "$CPUS" --dry-run
    [ "$TMP_RUN" = 1 ] && rm -rf "$RUN_DIR"                # only the temporary folder made above
    exit 0
fi
write_run_info "$RUN_DIR" "submit    submit_geometry_sweep.sh: $N_GEOM geometries, $N_TASKS array tasks x $CPUS CPUs, $TIME" \
                          "args      ${ARGS[*]}"

JOBID=$(sbatch --parsable -J "$RUN_NAME" -c "$CPUS" --time="$TIME" --array=0-$((N_TASKS - 1)) \
        -o "$RUN_DIR/logs/%x_%A_%a.out" \
        "$RUN_DIR/code/chunked_sweep/supercloud/job_geometry.sbatch" "$RUN_DIR" "$PY" "${ARGS[@]}")
CJOB=$(sbatch --parsable -J "${RUN_NAME}_collect" -c 2 --time=02:00:00 --dependency=afterany:"$JOBID" \
       -o "$RUN_DIR/logs/%x_%j.out" "$RUN_DIR/code/chunked_sweep/supercloud/job_collect.sbatch" "$RUN_DIR" "$PY")
echo "array job $JOBID, collect job $CJOB" >> "$RUN_DIR/run_info.txt"
echo "submitted array job $JOBID ($N_TASKS tasks) and collect job $CJOB -> $RUN_DIR"
echo "  status:   LLstat    (or: squeue -u $USER)"
echo "  logs:     ls $RUN_DIR/logs/"
echo "  per task: sacct -j $JOBID -X --format=JobID%20,State%15,ExitCode,Elapsed,NodeList%20"
echo "  memory:   sacct -j $JOBID --format=JobID%25,State%15,Elapsed,AllocCPUS,MaxRSS --units=G   (rows ending in .0 = the srun step)"
echo "  resume:   bash $SC_DIR/submit_geometry_sweep.sh --resume $RUN_DIR"
