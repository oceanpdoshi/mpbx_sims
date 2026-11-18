#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
# CHILD script of submit_geometry_sweep.sh: runs after the array (afterany) and writes the run-level
# tables (run_summary.csv, all_tracks.csv). Arguments: $1 = run folder, $2 = python.
RUN_DIR="$1"
PY="$2"
source /etc/profile
source "$RUN_DIR/code/chunked_sweep/supercloud/common_env.sh"
"$PY" -u "$RUN_DIR/code/chunked_sweep/collect_run.py" "$RUN_DIR"
