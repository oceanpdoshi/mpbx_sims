# Helpers for the submit scripts (SOURCED, not run).

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SC_DIR="$REPO/chunked_sweep/supercloud"
PY="$REPO/.venv/bin/python"

check_env() {
    if [ ! -x "$PY" ]; then
        echo "ERROR: $PY not found. Run once on a login node:  bash chunked_sweep/supercloud/setup_env.sh"
        exit 1
    fi
    if ! command -v sbatch >/dev/null 2>&1; then
        echo "ERROR: sbatch not found. Run this on a SuperCloud login node."
        exit 1
    fi
}

# check_geometry_csv CSV: read and check a geometry list exactly as the jobs will (run_geometry.py's
# read_geometry_csv; no JAX, seconds) and print each geometry's rib type (regular / undercut), so a bad
# row (e.g. t_undercut > 0 with a cladding or a slab) stops the script before anything is submitted.
check_geometry_csv() {
    HDF5_USE_FILE_LOCKING=FALSE PYTHONNOUSERSITE=1 "$PY" - "$REPO/chunked_sweep" "$1" <<'EOF'
import sys
sys.path.insert(0, sys.argv[1])
from run_geometry import geometry_kind, read_geometry_csv
rows = read_geometry_csv(sys.argv[2])
kinds = [geometry_kind(r["geometry"]) for r in rows]
n_uc = sum(k != "rib" for k in kinds)
print(f"geometries: {len(rows)} ({len(rows) - n_uc} regular rib, {n_uc} undercut rib)")
for r, k in list(zip(rows, kinds))[:10]:
    print(f"  {r['name']}: {k}")
if len(rows) > 10:
    print(f"  ... and {len(rows) - 10} more")
EOF
}

# new_run_dir OUT_ROOT RUN_NAME -> prints the new folder OUT_ROOT/RUN_NAME_<date-time>[_n]
new_run_dir() {
    local base="$1/${2}_$(date +%F-%H-%M)" d n=2
    d="$base"
    while [ -e "$d" ]; do d="${base}_$n"; n=$((n + 1)); done
    mkdir -p "$d/logs"
    echo "$(cd "$d" && pwd)"
}

# snapshot_code RUN_DIR: copy the code the jobs run into RUN_DIR/code, so editing sweep_config.py or a
# `git pull` while a job waits in the queue (or before a resubmission) cannot change the run.
snapshot_code() {
    local d="$1/code"
    mkdir -p "$d/chunked_sweep/supercloud" "$d/sim_core" "$d/geometry_gen" "$d/utils"
    cp "$REPO"/chunked_sweep/*.py "$d/chunked_sweep/"
    cp "$SC_DIR"/*.sh "$SC_DIR"/*.sbatch "$d/chunked_sweep/supercloud/"
    cp "$REPO"/sim_core/*.py "$d/sim_core/"
    cp "$REPO"/geometry_gen/*.py "$d/geometry_gen/"
    cp "$REPO"/utils/*.py "$d/utils/"
    # version of the copied code (results_io.code_version reads the first line)
    {
        echo "$(git -C "$REPO" describe --always --dirty --abbrev=10 2>/dev/null || echo unknown) (copied $(date '+%F %T'))"
        git -C "$REPO" status --short 2>/dev/null || true
    } > "$d/CODE_VERSION"
}

# write_run_info RUN_DIR "line" "line" ...
write_run_info() {
    local d="$1"
    shift
    {
        echo "created   $(date)"
        echo "user      $USER on $(hostname)"
        echo "repo      $REPO   commit $(git -C "$REPO" log --oneline -1 2>/dev/null || true)"
        { git -C "$REPO" status --short 2>/dev/null || true; } | sed 's/^/          uncommitted: /'
        echo "mpbx      $(cd "$REPO/.." && pwd)/mpbx   commit $(git -C "$REPO/../mpbx" describe --always --dirty 2>/dev/null || true)"
        echo "python    $PY"
        local l
        for l in "$@"; do echo "$l"; done
    } >> "$d/run_info.txt"
}
