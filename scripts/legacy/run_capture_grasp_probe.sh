#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
run_output="${1:-outputs/capture-grasp-physics-v8}"
trap 'run_code=$?; printf "exit_code=%s\n" "$run_code" > "${run_output}.status"' EXIT
./run_rl_python.sh capture_grasp_probe.py --output "$run_output"
