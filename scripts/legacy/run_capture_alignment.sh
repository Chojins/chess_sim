#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
run_output="${1:-outputs/capture-alignment-rl-v9}"
if [[ -e "$run_output" ]]; then
    echo "Output already exists: $run_output" >&2
    exit 1
fi
mkdir -p -- "$(dirname -- "$run_output")"
trap 'run_code=$?; printf "exit_code=%s\n" "$run_code" > "${run_output}.status"' EXIT
./run_rl_python.sh train_openings.py --output "$run_output" \
    --ids scandinavian_capture --pickup-only --capture-alignment --record-evaluations \
    --steps 3000 --eval-every 1000 --seed 1400 --eval-seeds 1401 1402 1403
./run_rl_python.sh verify_opening_rl.py "$run_output"
