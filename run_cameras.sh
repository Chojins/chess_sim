#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
isaac_dir="${ISAAC_SIM_PATH:-$HOME/isaacsim/_build/linux-x86_64/release}"
if [[ ! -x "$isaac_dir/python.sh" ]]; then
    echo "Set ISAAC_SIM_PATH to an Isaac Sim installation containing python.sh" >&2
    exit 1
fi
# Keep ROS/conda/user Python packages out of Isaac's bundled interpreter.
unset PYTHONPATH PYTHONHOME PYTHONEXE CONDA_PREFIX AMENT_PREFIX_PATH COLCON_PREFIX_PATH
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
exec "$isaac_dir/python.sh" "$project_dir/camera_server.py" "$@"
