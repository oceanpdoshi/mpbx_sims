#!/bin/bash
# Submit ONE waveguide geometry to one GPU node (xeon-g6-volta: 2 x V100 32 GB, 40 cores, 384 GB):
# one solver process per GPU. Run from anywhere on a SuperCloud LOGIN node:
#
#   bash gpu_sweep/supercloud/submit_single_geometry_gpu.sh --check      ~10 min: driver + GPU vs CPU numbers (do this first)
#   bash gpu_sweep/supercloud/submit_single_geometry_gpu.sh --test       ~10-20 min: the whole pipeline on tiny grids
#   bash gpu_sweep/supercloud/submit_single_geometry_gpu.sh --dry-run    grids, batch sizes, work units; submits nothing
#   bash gpu_sweep/supercloud/submit_single_geometry_gpu.sh              submit (edit the settings below first)
#   bash gpu_sweep/supercloud/submit_single_geometry_gpu.sh --resume outputs/<run folder>
#
# Results: OUT_ROOT/<RUN_NAME>_<date-time>/<geometry name>/ (README §5), logs in .../logs/.
# Compare with a CPU run of the same geometry:
#   python gpu_sweep/compare_runs.py <cpu run>/<geometry> <gpu run>/<geometry>

# ============================== settings ==============================
RUN_NAME="tfln_gpu_single"
T_LN=0.400              # LN film thickness [µm]
ETCH_DEPTH=0.400        # etch depth [µm] (= T_LN: fully etched)
W_TOP=1.800             # ridge top width [µm]
SIDEWALL_DEG=73         # sidewall angle from horizontal [deg]
T_CLAD=1.350            # inf = SiO2 everywhere above, 0 = air, else planar SiO2 this far above the ridge top [µm]
GEOM_NAME=""            # empty = built from the parameters (tLN400_etch400_w1800_sw73_clad1350nmSiO2)
CHUNKS=gpu              # gpu = gpu_config.py (10 x 10 µm at 6.25 nm) | cpu = chunked_sweep/sweep_config.py's grids
GPUS=2                  # GPUs (one node has 2); 20 CPU cores are requested per GPU
TIME=12:00:00           # time limit (bench_gpu.py projects the real time; max 96:00:00)
PARTITION=""            # empty = let --gres pick the GPU partition (xeon-g6-volta)
OUT_ROOT=""             # empty = <repo>/outputs ; e.g. /home/gridsan/groups/<group>/tfln_outputs
# ======================================================================

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib_submit_gpu.sh"
OUT_ROOT="${OUT_ROOT:-$REPO/outputs}"

MODE=submit
RESUME=""
while [ $# -gt 0 ]; do
    case "$1" in
        --dry-run) MODE=dry; shift ;;
        --test) MODE=test; shift ;;
        --check) MODE=check; shift ;;
        --resume) RESUME="$2"; shift 2 ;;
        *) echo "unknown option $1"; exit 1 ;;
    esac
done
if [ -n "$RESUME" ] && [ "$MODE" != submit ]; then
    echo "ERROR: --resume cannot be combined with --dry-run, --test or --check"
    exit 1
fi
check_env_gpu

if [ "$MODE" = check ]; then                              # one GPU, no run folder of results
    RUN_DIR="$(new_run_dir "$OUT_ROOT" "gpu_check")"
    snapshot_code_gpu "$RUN_DIR"
    write_run_info "$RUN_DIR" "submit    submit_single_geometry_gpu.sh --check"
    JOBID=$(sbatch --parsable -J gpu_check $(gpu_request 1) --time=00:30:00 -o "$RUN_DIR/logs/%x_%j.out" \
            "$RUN_DIR/code/gpu_sweep/supercloud/job_gpu.sbatch" "$RUN_DIR" "$PY" gpu_sweep/check_gpu_local.py --platform cuda)
    echo "job $JOBID" >> "$RUN_DIR/run_info.txt"
    echo "submitted check job $JOBID -> $RUN_DIR/logs/gpu_check_$JOBID.out  (look for 'ALL CHECKS PASSED')"
    exit 0
fi

TMP_RUN=0
if [ -n "$RESUME" ]; then
    RUN_DIR="$(cd "$RESUME" && pwd)"
    [ -f "$RUN_DIR/run_args.txt" ] || { echo "ERROR: $RUN_DIR/run_args.txt not found (not a run folder?)"; exit 1; }
    read -r -a ARGS < "$RUN_DIR/run_args.txt"
    case " ${ARGS[*]} " in *" --test "*) TIME=01:00:00 ;; esac
    echo "resuming $RUN_DIR"
    write_run_info "$RUN_DIR" "resubmitted $(date): $GPUS GPUs, $TIME"
else
    if [ "$MODE" = dry ]; then
        RUN_DIR="$(mktemp -d)"
        TMP_RUN=1
    else
        RUN_DIR="$(new_run_dir "$OUT_ROOT" "$RUN_NAME$([ "$MODE" = test ] && echo _TEST)")"
    fi
    snapshot_code_gpu "$RUN_DIR"
    {
        echo "# geometry of this run (µm, degrees)"
        echo "name,t_ln,etch_depth,w_top,sidewall_deg,t_clad"
        echo "$GEOM_NAME,$T_LN,$ETCH_DEPTH,$W_TOP,$SIDEWALL_DEG,$T_CLAD"
    } > "$RUN_DIR/geometries.csv"
    ARGS=(--row 0 --chunks "$CHUNKS" --gpus "$GPUS")
    if [ "$MODE" = test ]; then
        ARGS+=(--test)
        TIME=01:00:00
    fi
    echo "${ARGS[*]}" > "$RUN_DIR/run_args.txt"            # reused by --resume
fi
ARGS+=(--geom-csv "$RUN_DIR/geometries.csv" --out "$RUN_DIR")

if [ "$MODE" = dry ]; then
    HDF5_USE_FILE_LOCKING=FALSE PYTHONNOUSERSITE=1 JAX_PLATFORMS=cpu "$PY" "$RUN_DIR/code/gpu_sweep/run_geometry_gpu.py" \
        "${ARGS[@]}" --dry-run
    [ "$TMP_RUN" = 1 ] && rm -rf "$RUN_DIR"
    exit 0
fi
[ -n "$RESUME" ] || write_run_info "$RUN_DIR" "submit    submit_single_geometry_gpu.sh $MODE: $GPUS GPUs, $TIME" \
                                   "geometry  $(tail -n 1 "$RUN_DIR/geometries.csv")" "args      ${ARGS[*]}"

JOBID=$(sbatch --parsable -J "$RUN_NAME" $(gpu_request "$GPUS") --time="$TIME" -o "$RUN_DIR/logs/%x_%j.out" \
        "$RUN_DIR/code/gpu_sweep/supercloud/job_gpu.sbatch" "$RUN_DIR" "$PY" gpu_sweep/run_geometry_gpu.py "${ARGS[@]}")
echo "job $JOBID" >> "$RUN_DIR/run_info.txt"
echo "submitted job $JOBID -> $RUN_DIR"
echo "  status:   LLstat    (or: squeue -u $USER);  GPU load: LLload -g --detail"
echo "  log:      tail -f $RUN_DIR/logs/${RUN_NAME}_$JOBID.out"
echo "  after it: sacct -j $JOBID --format=JobID%20,State%15,ExitCode,Elapsed,AllocCPUS,MaxRSS --units=G"
echo "  resume:   bash $GPU_SC_DIR/submit_single_geometry_gpu.sh --resume $RUN_DIR"
