#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
isaac_dir="${ISAAC_SIM_PATH:-$HOME/isaacsim/_build/linux-x86_64/release}"
shopt -s nullglob
usd_candidates=()
for usd_candidate in "$isaac_dir"/extscache/omni.usd.libs-*; do
    if [[ -d "$usd_candidate/pxr" ]]; then
        usd_candidates+=("$usd_candidate")
    fi
done
if [[ ${#usd_candidates[@]} -ne 1 || ! -d "${usd_candidates[0]}/pxr" ]]; then
    echo "Expected one Isaac 5.0 Python 3.11 USD library; set ISAAC_SIM_PATH or use a Python environment with pxr and numpy." >&2
    exit 1
fi
unset PYTHONHOME PYTHONEXE CONDA_PREFIX AMENT_PREFIX_PATH COLCON_PREFIX_PATH
export PYTHONNOUSERSITE=1
export PYTHONPATH="${usd_candidates[0]}"
export LD_LIBRARY_PATH="${usd_candidates[0]}/bin${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
cd "$project_dir"
exec "$isaac_dir/python.sh" "$@"
