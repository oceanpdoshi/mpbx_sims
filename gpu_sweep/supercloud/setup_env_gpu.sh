#!/bin/bash
# ONE-TIME setup of the GPU environment on a SuperCloud LOGIN node (compute nodes have no internet).
# Run the CPU setup first (it installs the standalone Python 3.12 this script reuses):
#
#   bash chunked_sweep/supercloud/setup_env.sh          # once (CPU venv .venv + Python 3.12)
#   bash gpu_sweep/supercloud/setup_env_gpu.sh          # once (GPU venv .venv-gpu)
#
# Creates mpbx_sims/.venv-gpu: the SAME package versions as uv.lock (jax/jaxlib 0.12.0.1, numpy, h5py,
# ...) plus JAX's CUDA-12 plugin of the same version and the NVIDIA CUDA 12 libraries as pip wheels
# (~3 GB in your home directory: check `LLquota`). Why exactly these (README §4.1):
#   * The V100 is compute capability 7.0. JAX's CUDA 13 wheels need >= 7.5 (CUDA 13 dropped Volta), so
#     it must be the CUDA 12 wheels. JAX 0.12.1 is the last release announced to ship them (CUDA 12
#     support is dropped in 0.12.2): the versions are pinned exactly, never `>=`.
#   * cuDNN 9.10.2 is the last cuDNN listing Volta and the version JAX's CUDA 12 build is compiled
#     against; a newer one would be pulled in otherwise. (The mode solver uses no cuDNN routine.)
#   * Same jax/jaxlib as the CPU venv: a CPU run and a GPU run differ only in the XLA backend.
# The NVIDIA driver on the GPU nodes must be >= 525 (CUDA 12); `submit_single_geometry_gpu.sh --check`
# prints it. Do NOT `module load cuda` in GPU jobs: it would put other CUDA libraries first.
set -euo pipefail

JAX_VERSION="0.12.0.1"
CUDNN_VERSION="9.10.2.21"

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MPBX="$(cd "$REPO/.." && pwd)/mpbx"
VENV="$REPO/.venv-gpu"
echo "repo: $REPO"
[ -f "$MPBX/python/pyproject.toml" ] || { echo "ERROR: $MPBX/python not found (clone mpbx next to this repo)"; exit 1; }
echo "mpbx: $MPBX ($(git -C "$MPBX" log --oneline -1 2>/dev/null || echo 'not a git checkout'))"

LOCAL="/state/partition1/user/$USER"                   # login node's local disk (file locks work there)
mkdir -p "$LOCAL/tmp" "$LOCAL/uv-cache"
export TMPDIR="$LOCAL/tmp"
export UV_CACHE_DIR="$LOCAL/uv-cache"
export UV_LINK_MODE=copy
export PYTHONNOUSERSITE=1
export HDF5_USE_FILE_LOCKING=FALSE
export PATH="$HOME/.local/bin:$PATH"
command -v uv >/dev/null 2>&1 || { echo "ERROR: uv not found: run chunked_sweep/supercloud/setup_env.sh first"; exit 1; }

# Rebuilding deletes the venv that queued / running GPU jobs run with: refuse while any job is queued.
if [ -d "$VENV" ] && [ "${FORCE:-0}" != 1 ] && command -v squeue >/dev/null 2>&1; then
    NJOBS=$(squeue -u "$USER" -h 2>/dev/null | wc -l)
    if [ "$NJOBS" -gt 0 ]; then
        echo "ERROR: $VENV exists and you have $NJOBS queued/running job(s) that may use it (squeue -u $USER)."
        echo "       Wait until they end, or rebuild anyway with:  FORCE=1 bash $0"
        exit 1
    fi
fi

PY_HOME="$HOME/.local/share/mpbx-python"
PYDIR="$(find "$PY_HOME" -maxdepth 1 -type d -name 'cpython-3.12.*' 2>/dev/null | sort -V | tail -n 1 || true)"
BASE_PY="$PYDIR/bin/python3.12"
[ -n "$PYDIR" ] && [ -x "$BASE_PY" ] || { echo "ERROR: no Python 3.12 in $PY_HOME: run chunked_sweep/supercloud/setup_env.sh first"; exit 1; }
"$BASE_PY" --version

# ---- requirements: uv.lock's pins (the two editable mpbx packages are added with absolute paths) ----
cd "$REPO"
REQ="$LOCAL/tmp/requirements-gpu.txt"
uv export --frozen --no-hashes --no-emit-project -o "$REQ.all" >/dev/null
grep -v '^-e ' "$REQ.all" > "$REQ"
grep -q "^jax==$JAX_VERSION" "$REQ" || echo "WARNING: uv.lock does not pin jax==$JAX_VERSION; the GPU venv uses $JAX_VERSION anyway"

rm -rf "$VENV"
uv venv "$VENV" --python "$BASE_PY"
PY="$VENV/bin/python"
GPU_PKGS=("jax[cuda12]==$JAX_VERSION" "jaxlib==$JAX_VERSION" "jax-cuda12-plugin[with-cuda]==$JAX_VERSION"
          "jax-cuda12-pjrt==$JAX_VERSION" "nvidia-cudnn-cu12==$CUDNN_VERSION")
if uv pip install --python "$PY" -r "$REQ" "${GPU_PKGS[@]}" -e "$MPBX/python" -e "$MPBX/python/meepjax"; then
    echo "uv pip install: ok"
else
    echo "uv pip install failed; falling back to python -m venv + pip"
    rm -rf "$VENV"
    "$BASE_PY" -m venv "$VENV"
    "$PY" -m pip install --no-cache-dir --upgrade pip
    "$PY" -m pip install --no-cache-dir -r "$REQ" "${GPU_PKGS[@]}" -e "$MPBX/python" -e "$MPBX/python/meepjax"
fi

# ---- check (imports only: the login node has no GPU; the --check job tests the GPU) ---------------
echo "checking the imports ..."
export JAX_PLATFORMS=cpu
export JAX_COMPILATION_CACHE_DIR="$LOCAL/jax_cache"
export MPLCONFIGDIR="$LOCAL/matplotlib"
"$PY" - <<'EOF'
import importlib.metadata as md, os, sys
root = os.getcwd()
sys.path[:0] = [root, os.path.join(root, "chunked_sweep"), os.path.join(root, "gpu_sweep")]
from utils.setup_jax import setup_jax
setup_jax("cpu", verbose=False)
import jax, numpy, h5py
import batched_modes, run_geometry_gpu, bench_gpu, compare_runs, results_io
v = {p: md.version(p) for p in ("jax", "jaxlib", "jax-cuda12-plugin", "jax-cuda12-pjrt", "nvidia-cudnn-cu12",
                                "nvidia-cufft-cu12", "nvidia-cublas-cu12", "numpy", "h5py")}
print(" | ".join(f"{k} {x}" for k, x in v.items()))
assert v["jax"] == v["jax-cuda12-plugin"] == v["jaxlib"], "jax / jaxlib / jax-cuda12-plugin versions differ"
EOF
rm -rf "$LOCAL/tmp"
echo
echo "GPU environment ready: $PY  ($(du -sh "$VENV" 2>/dev/null | cut -f1))"
echo "Next, on a GPU node (one short job each):"
echo "    bash gpu_sweep/supercloud/submit_single_geometry_gpu.sh --check    # GPU vs CPU numbers, driver, ~10 min"
echo "    bash gpu_sweep/supercloud/submit_single_geometry_gpu.sh --test     # the whole pipeline on tiny grids"
