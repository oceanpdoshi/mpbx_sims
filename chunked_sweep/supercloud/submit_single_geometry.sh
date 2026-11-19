#!/bin/bash
# Submit ONE waveguide geometry: one Slurm task on one full xeon-p8 node (48 cores, 192 GB).
# Run from anywhere on a SuperCloud LOGIN node:
#
#   bash chunked_sweep/supercloud/submit_single_geometry.sh              submit (edit the settings below first)
#   bash chunked_sweep/supercloud/submit_single_geometry.sh --dry-run    print grids, cost, file sizes; submit nothing
#   bash chunked_sweep/supercloud/submit_single_geometry.sh --test       ~10-min pipeline check on 8 cores (tiny grids)
#   bash chunked_sweep/supercloud/submit_single_geometry.sh --resume outputs/<run folder>
#                                    continue a run that hit its time limit or crashed (same code, same settings)
#
# Results: OUT_ROOT/<RUN_NAME>_<date-time>/<geometry name>/  (README §5). Logs: .../logs/.

# ============================== settings ==============================
RUN_NAME="tfln_single"
T_LN=0.400              # LN film thickness [µm]
ETCH_DEPTH=0.400        # etch depth [µm] (= T_LN: fully etched)
W_TOP=1.600             # ridge top width [µm]
SIDEWALL_DEG=73         # sidewall angle from horizontal [deg]
T_CLAD=1.350            # inf = SiO2 everywhere above, 0 = air, else planar SiO2 this far above the ridge top [µm]
GEOM_NAME=""            # empty = built from the parameters (tLN400_etch400_w1800_sw73_cladSiO2)
FIELDS=all              # all | none   (E fields: ~3 GB per geometry, README §5)
CPUS=48                 # cores for this task (48 = one full node)
CORES_PER_WORKER=1      # cores per mode-solver process (README §3; check with submit_benchmark.sh)
TIME=24:00:00           # time limit (estimate ~2-4 h for the default chunks; max 96:00:00)
OUT_ROOT=""             # empty = <repo>/outputs ; e.g. /home/gridsan/groups/<group>/tfln_outputs
# ======================================================================

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib_submit.sh"
OUT_ROOT="${OUT_ROOT:-$REPO/outputs}"

MODE=submit
RESUME=""
while [ $# -gt 0 ]; do
    case "$1" in
        --dry-run) MODE=dry; shift ;;
        --test) MODE=test; shift ;;
        --resume) RESUME="$2"; shift 2 ;;
        *) echo "unknown option $1"; exit 1 ;;
    esac
done
if [ -n "$RESUME" ] && [ "$MODE" != submit ]; then
    echo "ERROR: --resume cannot be combined with --dry-run or --test"
    exit 1
fi
check_env

TMP_RUN=0
if [ -n "$RESUME" ]; then
    RUN_DIR="$(cd "$RESUME" && pwd)"
    [ -f "$RUN_DIR/run_args.txt" ] || { echo "ERROR: $RUN_DIR/run_args.txt not found (not a run folder?)"; exit 1; }
    read -r -a ARGS < "$RUN_DIR/run_args.txt"
    case " ${ARGS[*]} " in *" --test "*) CPUS=8; TIME=01:00:00 ;; esac
    echo "resuming $RUN_DIR"
    write_run_info "$RUN_DIR" "resubmitted $(date): $CPUS CPUs, $TIME"
else
    if [ "$MODE" = dry ]; then
        RUN_DIR="$(mktemp -d)"
        TMP_RUN=1
    else
        RUN_DIR="$(new_run_dir "$OUT_ROOT" "$RUN_NAME$([ "$MODE" = test ] && echo _TEST)")"
    fi
    snapshot_code "$RUN_DIR"
    {
        echo "# geometry of this run (µm, degrees)"
        echo "name,t_ln,etch_depth,w_top,sidewall_deg,t_clad"
        echo "$GEOM_NAME,$T_LN,$ETCH_DEPTH,$W_TOP,$SIDEWALL_DEG,$T_CLAD"
    } > "$RUN_DIR/geometries.csv"
    ARGS=(--row 0 --fields "$FIELDS" --cores-per-worker "$CORES_PER_WORKER")
    if [ "$MODE" = test ]; then
        ARGS+=(--test)
        CPUS=8
        TIME=01:00:00
    fi
    echo "${ARGS[*]}" > "$RUN_DIR/run_args.txt"            # reused by --resume
fi
ARGS+=(--geom-csv "$RUN_DIR/geometries.csv")

if [ "$MODE" = dry ]; then
    HDF5_USE_FILE_LOCKING=FALSE PYTHONNOUSERSITE=1 "$PY" "$RUN_DIR/code/chunked_sweep/run_geometry.py" \
        --out "$RUN_DIR" "${ARGS[@]}" --max-cores "$CPUS" --dry-run
    [ "$TMP_RUN" = 1 ] && rm -rf "$RUN_DIR"                # only the temporary folder made above
    exit 0
fi
[ -n "$RESUME" ] || write_run_info "$RUN_DIR" "submit    submit_single_geometry.sh $MODE: $CPUS CPUs, $TIME" \
                                   "geometry  $(tail -n 1 "$RUN_DIR/geometries.csv")" "args      ${ARGS[*]}"

JOBID=$(sbatch --parsable -J "$RUN_NAME" -c "$CPUS" --time="$TIME" -o "$RUN_DIR/logs/%x_%j.out" \
        "$RUN_DIR/code/chunked_sweep/supercloud/job_geometry.sbatch" "$RUN_DIR" "$PY" "${ARGS[@]}")
echo "job $JOBID" >> "$RUN_DIR/run_info.txt"
echo "submitted job $JOBID -> $RUN_DIR"
echo "  status:   LLstat    (or: squeue -u $USER)"
echo "  log:      tail -f $RUN_DIR/logs/${RUN_NAME}_$JOBID.out"
echo "  after it: sacct -j $JOBID --format=JobID%20,State%15,ExitCode,Elapsed,AllocCPUS,MaxRSS --units=G   (memory: the .0 row)"
echo "  resume:   bash $SC_DIR/submit_single_geometry.sh --resume $RUN_DIR"
