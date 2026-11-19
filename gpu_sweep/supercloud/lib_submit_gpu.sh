# Helpers for the GPU submit scripts (SOURCED, not run). Reuses the CPU sweep's helpers
# (new_run_dir, write_run_info, snapshot_code) and points them at the GPU venv.

source "$(dirname "${BASH_SOURCE[0]}")/../../chunked_sweep/supercloud/lib_submit.sh"   # sets REPO, SC_DIR, PY
GPU_SC_DIR="$REPO/gpu_sweep/supercloud"
PY="$REPO/.venv-gpu/bin/python"

check_env_gpu() {
    if [ ! -x "$PY" ]; then
        echo "ERROR: $PY not found. Run once on a login node:  bash gpu_sweep/supercloud/setup_env_gpu.sh"
        exit 1
    fi
    if ! command -v sbatch >/dev/null 2>&1; then
        echo "ERROR: sbatch not found. Run this on a SuperCloud login node."
        exit 1
    fi
}

# snapshot_code_gpu RUN_DIR: the CPU sweep's code snapshot (sim_core, geometry_gen, utils, chunked_sweep)
# plus gpu_sweep, so editing gpu_config.py while a job waits cannot change the run.
snapshot_code_gpu() {
    snapshot_code "$1"
    mkdir -p "$1/code/gpu_sweep/supercloud"
    cp "$REPO"/gpu_sweep/*.py "$1/code/gpu_sweep/"
    cp "$GPU_SC_DIR"/*.sh "$GPU_SC_DIR"/*.sbatch "$1/code/gpu_sweep/supercloud/"
}

# gpu_request NGPUS: the sbatch options for NGPUS V100s of one node (SuperCloud docs: --gres picks the
# xeon-g6-volta partition; ask for 20 CPUs per GPU = memory 9 GB per CPU). PARTITION, if set, is added.
gpu_request() {
    local n="$1"
    echo "--gres=gpu:volta:$n -c $((20 * n))${PARTITION:+ -p $PARTITION}"
}
