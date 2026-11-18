#!/bin/bash
# ONE-TIME: measure how the mode solver scales on a xeon-p8 node, to choose CORES_PER_WORKER and
# CPUS per geometry (README §3). Takes one exclusive node for ~1 h. Run on a login node:
#
#   bash chunked_sweep/supercloud/submit_benchmark.sh
#
# Output (log): for 1 process on 1, 2, 4, ... 48 cores, and for 48 x 1, 24 x 2, 12 x 4 ... processes
# at once, the LOBPCG iterations per second of the whole node. Pick the CORES_PER_WORKER with the
# highest "all processes: N x one core".
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib_submit.sh"
check_env
RUN_DIR="$(new_run_dir "$REPO/outputs" benchmark)"
snapshot_code "$RUN_DIR"
cat > "$RUN_DIR/job_benchmark.sbatch" <<EOF
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive
source /etc/profile
source "$RUN_DIR/code/chunked_sweep/supercloud/common_env.sh"
echo "node \$(hostname), nproc \$(nproc)"
lscpu
"$PY" -u "$RUN_DIR/code/chunked_sweep/bench_cpu.py" --iters 30
EOF
JOBID=$(sbatch --parsable -J bench_mpbx -c 48 --time=03:00:00 -o "$RUN_DIR/logs/%x_%j.out" "$RUN_DIR/job_benchmark.sbatch")
echo "submitted benchmark job $JOBID -> $RUN_DIR/logs/bench_mpbx_$JOBID.out"
