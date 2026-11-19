# Environment for every GPU job of this sweep. SOURCED by the job scripts (not run).

# Everything the CPU jobs set (HDF5 file locking off on Lustre, no ~/.local shadowing, unbuffered
# output, JAX compile cache / matplotlib on the node's local disk, one-thread BLAS / OpenMP pools).
source "$(dirname "${BASH_SOURCE[0]}")/../../chunked_sweep/supercloud/common_env.sh"

# JAX: float64 everywhere (mpbjax also switches it on at import; a job must never run in float32).
export JAX_ENABLE_X64=1
# Share of each GPU's memory JAX may take (default 0.75 = 24 of 32 GB). Each worker sets it too
# (gpu_config.MEM_FRACTION); exported here so that bench / check processes get it as well.
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.92}"
# JAX finds its CUDA 12 libraries in the venv (pip wheels). CUDA libraries of a loaded module would
# be found first through LD_LIBRARY_PATH (JAX's install docs warn about this): drop such entries.
if [ -n "${LD_LIBRARY_PATH:-}" ]; then
    LD_LIBRARY_PATH="$(echo "$LD_LIBRARY_PATH" | tr ':' '\n' | grep -vi 'cuda\|cudnn' | paste -sd: -)"
    export LD_LIBRARY_PATH
fi
# TensorFlow-style C++ log level used by XLA: 1 = no INFO spam, warnings and errors stay visible.
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-1}"
