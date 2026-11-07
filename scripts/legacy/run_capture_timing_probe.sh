#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
trap 'run_code=$?; printf "exit_code=%s\n" "$run_code" > outputs/capture-timing-probe-v11.status' EXIT
./run_rl_python.sh capture_timing_probe.py --output outputs/capture-timing-probe-v11
