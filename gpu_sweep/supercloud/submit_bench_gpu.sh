#!/bin/bash
# GPU benchmark (README §3.3): one GPU node, ~1-3 h. Answers "is a GPU node >= 6x faster than a CPU
# node?" before a full run. Run on a login node, after the --check job passed:
#
#   bash gpu_sweep/supercloud/submit_bench_gpu.sh outputs/tfln_single_2026-11-19-08-51/tLN400_etch400_w1800_sw73_clad1350nmSiO2
#
# The argument is the geometry folder of a finished CPU run (chunked_sweep, e.g. the 2.44 h baseline);
# its geometry is benchmarked. Without it, the baseline geometry and no CPU comparison. Both GPUs work
# at the same time:
#   GPU 1: the GPU grids (gpu_config.py), every chunk -> projected wall time of the full geometry on
#          2 GPUs, against the CPU run's wall time                    -> logs/bench_gpu_grids.log
#   GPU 2: the CPU run's own grids (equal work, node vs node); nev scan at 1.5 µm (3, 4, 6 modes per
#          sector); lanes scan (1, 2, 3 wavelengths per batch); chunk D one sector solve per call
#          (no lockstep); chunk A with JAX's default small eigensolver instead of QR
#                                                                     -> logs/bench_cpu_grids_and_scans.log
# Results: <run>/bench*/bench_*.json. The job fails (sacct: FAILED) if any part failed.

# ============================== settings ==============================
N_LAM=3                 # wavelengths per lane (1 cold + warm-started ones)
TIME=06:00:00
PARTITION=""
OUT_ROOT=""
# ======================================================================

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib_submit_gpu.sh"
OUT_ROOT="${OUT_ROOT:-$REPO/outputs}"
check_env_gpu
CPU_RUN=""
if [ $# -ge 1 ]; then
    CPU_RUN="$(cd "$1" && pwd)"
    ls "$CPU_RUN"/*_modes.h5 >/dev/null 2>&1 || { echo "ERROR: no *_modes.h5 in $CPU_RUN (give the GEOMETRY folder of a CPU run)"; exit 1; }
fi
RUN_DIR="$(new_run_dir "$OUT_ROOT" bench_gpu)"
snapshot_code_gpu "$RUN_DIR"
write_run_info "$RUN_DIR" "submit    submit_bench_gpu.sh  cpu run: ${CPU_RUN:-none}"
CPU_OPT=""                                              # quoted for the generated script
[ -n "$CPU_RUN" ] && CPU_OPT="--cpu-run \"$CPU_RUN\""
B="$RUN_DIR/code/gpu_sweep/bench_gpu.py"
cat > "$RUN_DIR/job_bench_gpu.sbatch" <<EOF
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
source /etc/profile
source "$RUN_DIR/code/gpu_sweep/supercloud/common_env_gpu.sh"
echo "=== \$(date '+%F %T')  job \$SLURM_JOB_ID  node \$(hostname)  CUDA_VISIBLE_DEVICES=\$CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
IFS=, read -r G0 G1 <<< "\$CUDA_VISIBLE_DEVICES"
G1="\${G1:-\$G0}"
cd "$RUN_DIR"
BENCH() { "$PY" -u "$B" "\$@"; }
(
    export CUDA_VISIBLE_DEVICES=\$G0
    BENCH --out "$RUN_DIR/bench" --grids gpu --n-lam $N_LAM $CPU_OPT
) > "$RUN_DIR/logs/bench_gpu_grids.log" 2>&1 &
P1=\$!
(
    export CUDA_VISIBLE_DEVICES=\$G1
    rc=0
    if [ -n "$CPU_RUN" ]; then
        BENCH --out "$RUN_DIR/bench" --grids cpu-run --n-lam 2 $CPU_OPT || rc=1
    fi
    BENCH --out "$RUN_DIR/bench" --grids gpu --nev-scan 3,4,6 --scan-lam 1.5 --n-lam $N_LAM $CPU_OPT || rc=1
    BENCH --out "$RUN_DIR/bench_lanes" --grids gpu --lanes-scan 1,2,3 --scan-lam 1.5 --n-lam $N_LAM $CPU_OPT || rc=1
    BENCH --out "$RUN_DIR/bench_slots1" --grids gpu --chunks D --max-slots 1 --n-lam $N_LAM $CPU_OPT || rc=1
    BENCH --out "$RUN_DIR/bench_eigh_default" --grids gpu --chunks A --eigh default --n-lam $N_LAM $CPU_OPT || rc=1
    exit \$rc
) > "$RUN_DIR/logs/bench_cpu_grids_and_scans.log" 2>&1 &
P2=\$!
s1=0; wait \$P1 || s1=\$?
s2=0; wait \$P2 || s2=\$?
echo "=== \$(date '+%F %T')  done: GPU-grid bench exit \$s1, CPU-grid bench + scans exit \$s2"
[ \$s1 -eq 0 ] && [ \$s2 -eq 0 ]
EOF
JOBID=$(sbatch --parsable -J bench_gpu $(gpu_request 2) --time="$TIME" -o "$RUN_DIR/logs/%x_%j.out" "$RUN_DIR/job_bench_gpu.sbatch")
echo "job $JOBID" >> "$RUN_DIR/run_info.txt"
echo "submitted GPU benchmark job $JOBID -> $RUN_DIR"
echo "  logs: $RUN_DIR/logs/bench_gpu_grids.log (projection vs the CPU run), bench_cpu_grids_and_scans.log"
