#!/bin/bash
# ONE-TIME setup on a SuperCloud LOGIN node (the compute nodes have no internet).
#
#   cd ~/github && git clone git@github.com:oceanpdoshi/mpbx.git          # the solver, next to ...
#   git clone <this repo> mpbx_sims && cd mpbx_sims                      # ... this repo
#   bash chunked_sweep/supercloud/setup_env.sh
#
# Creates mpbx_sims/.venv with exactly the package versions of uv.lock (Python 3.12, jax CPU, h5py,
# ...) and mpbx installed in editable mode from ../mpbx/python, then checks it with a tiny solve.
# Re-run it after `git pull` changes uv.lock or after updating mpbx's dependencies.
#
# Why not the anaconda module + `pip install --user` that the SuperCloud docs suggest first: this
# project needs Python >= 3.12 and the exact jax version it was tested with; a project venv keeps
# them identical to your desktop. It lives in your home directory (shared file system), so every
# compute node sees it.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MPBX="$(cd "$REPO/.." && pwd)/mpbx"
echo "repo: $REPO"
if [ ! -f "$MPBX/python/pyproject.toml" ]; then
    echo "ERROR: $MPBX/python not found. Clone mpbx next to this repo first:"
    echo "       cd $(dirname "$REPO") && git clone git@github.com:oceanpdoshi/mpbx.git"
    exit 1
fi
echo "mpbx: $MPBX ($(git -C "$MPBX" log --oneline -1 2>/dev/null || echo 'not a git checkout'))"

# pip / uv temporary files must not go to /tmp (small quota on SuperCloud); the node's local disk
# also supports file locking, which uv's cache needs and the shared file system does not have.
LOCAL="/state/partition1/user/$USER"
mkdir -p "$LOCAL/tmp" "$LOCAL/uv-cache"
export TMPDIR="$LOCAL/tmp"
export UV_CACHE_DIR="$LOCAL/uv-cache"
export UV_LINK_MODE=copy                 # the cache (local disk) and the venv (home) are different file systems
export UV_PYTHON_INSTALL_DIR="$LOCAL/uv-python"   # uv installs Python here (local disk: file locks work) ...
PY_HOME="$HOME/.local/share/mpbx-python"          # ... and we copy it here, where the compute nodes see it
export PYTHONNOUSERSITE=1
export HDF5_USE_FILE_LOCKING=FALSE

# ---- 1. uv ------------------------------------------------------------------------------------
export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
    echo "installing uv into ~/.local/bin ..."
    curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$HOME/.local/bin" UV_NO_MODIFY_PATH=1 sh
fi
uv --version

# ---- 2. Python 3.12 in your home directory ------------------------------------------------------
# real directories named cpython-3.12.<patch>-... only (uv also makes a cpython-3.12-... symlink)
find_py() { find "$1" -maxdepth 1 -type d -name 'cpython-3.12.*' 2>/dev/null | sort -V | tail -n 1; }
PYDIR="$(find_py "$PY_HOME" || true)"
if [ -z "$PYDIR" ] || [ ! -x "$PYDIR/bin/python3.12" ]; then
    uv python install 3.12                                  # into $LOCAL/uv-python (login node's local disk)
    SRC="$(find_py "$UV_PYTHON_INSTALL_DIR")"
    [ -n "$SRC" ] || { echo "ERROR: no cpython-3.12.* folder in $UV_PYTHON_INSTALL_DIR"; exit 1; }
    mkdir -p "$PY_HOME"
    cp -a "$SRC" "$PY_HOME/"                                # standalone build: works from any path
    PYDIR="$PY_HOME/$(basename "$SRC")"
    # uv may also have linked ~/.local/bin/python3.12 to the login node's local disk: remove that link
    for l in "$HOME"/.local/bin/python3.12 "$HOME"/.local/bin/python3; do
        if [ -L "$l" ] && readlink "$l" | grep -q "^$LOCAL"; then rm -f "$l"; fi
    done
fi
BASE_PY="$PYDIR/bin/python3.12"
[ -L "$BASE_PY" ] && [ ! -e "$BASE_PY" ] && { echo "ERROR: $BASE_PY is a dangling link"; exit 1; }
"$BASE_PY" --version

# ---- 3. the venv ----------------------------------------------------------------------------------
cd "$REPO"
PY="$REPO/.venv/bin/python"
if uv sync --frozen --python "$BASE_PY"; then
    echo "uv sync: ok"
else
    # Fallback, e.g. if uv cannot lock files on the shared file system: plain venv + pip with the
    # same pinned versions (exported from uv.lock; uv export only reads the lock file).
    echo "uv sync failed; falling back to python -m venv + pip with the versions pinned in uv.lock"
    uv export --frozen --no-hashes --no-emit-project -o "$LOCAL/tmp/requirements.txt"
    rm -rf "$REPO/.venv"
    "$BASE_PY" -m venv "$REPO/.venv"
    "$PY" -m pip install --no-cache-dir --upgrade pip
    "$PY" -m pip install --no-cache-dir -r "$LOCAL/tmp/requirements.txt"
fi

# ---- 4. check (imports only: no solving on a login node; the --test job below solves) ---------------
echo "checking the imports ..."
export JAX_COMPILATION_CACHE_DIR="$LOCAL/jax_cache"
export MPLCONFIGDIR="$LOCAL/matplotlib"
"$PY" - <<'EOF'
import os, sys
root = os.getcwd()
sys.path[:0] = [root, os.path.join(root, "handoff_tfln_modesolve"), os.path.join(root, "chunked_sweep")]
from utils.setup_jax import setup_jax
setup_jax("cpu")
import h5py, numpy as np, jax, pandas, matplotlib
from geometry_gen import tfln_rib
import tfln_modes, run_geometry, results_io
print("python", sys.version.split()[0], "| jax", jax.__version__, "| numpy", np.__version__, "| h5py", h5py.__version__,
      "| pandas", pandas.__version__)
EOF
rm -rf "$LOCAL/tmp"
echo
echo "Environment ready: $PY"
echo "Next, check it on a compute node (one small job, ~10 min):"
echo "    bash chunked_sweep/supercloud/submit_single_geometry.sh --test"
