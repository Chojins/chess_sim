#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
isaac_dir="${ISAAC_SIM_PATH:-$HOME/isaacsim/_build/linux-x86_64/release}"
cd -- "$project_dir"
unset PYTHONPATH PYTHONHOME PYTHONEXE CONDA_PREFIX AMENT_PREFIX_PATH COLCON_PREFIX_PATH
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
exec "$isaac_dir/python.sh" -m teacher.server "$@"
