#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
unset PYTHONPATH PYTHONHOME PYTHONEXE LD_LIBRARY_PATH CUDA_HOME CONDA_PREFIX
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="$project_dir/outputs/.matplotlib"
mkdir -p "$MPLCONFIGDIR"
exec "$project_dir/.venv-rl/bin/python" "$@"
