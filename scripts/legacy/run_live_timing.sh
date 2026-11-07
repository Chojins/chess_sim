#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
trap 'run_code=$?; printf "exit_code=%s\n" "$run_code" > outputs/capture-live-timing-v12.status' EXIT
./run_rl_python.sh evaluate_live_timing.py --output outputs/capture-live-timing-v12 \
    --seeds 1801 1802 1803 1804 1805 --repeats 2
