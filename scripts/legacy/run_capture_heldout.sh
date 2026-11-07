#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
run_output="${1:-outputs/capture-heldout-v9}"
if [[ -e "$run_output" ]]; then
    echo "Output already exists: $run_output" >&2
    exit 1
fi
mkdir -p -- "$(dirname -- "$run_output")"
trap 'run_code=$?; printf "exit_code=%s\n" "$run_code" > "${run_output}.status"' EXIT
./run_rl_python.sh evaluate_openings.py --output "$run_output" \
    --pilot outputs/capture-alignment-rl-v9 --ids scandinavian_capture \
    --baseline-checkpoint residual_1131.zip --learned-checkpoint residual_2031.zip \
    --baseline-label 'v9 checkpoint 1131' --learned-label 'v9 checkpoint 2031' \
    --seeds 1501 1502 1503 1504 1505 --repeats 2 --record-video \
    --comparison-note 'Fresh-seed pickup comparison of two intermediate v9 actors; no training or IK assistance; same training bounds/filter/reward'
