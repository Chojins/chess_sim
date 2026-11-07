#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
run_output="${1:-outputs/opening-rl-capture-pickup-v6}"
if [[ -e "$run_output" ]]; then
    echo "Output already exists: $run_output" >&2
    exit 1
fi
mkdir -p -- "$(dirname -- "$run_output")"
trap 'run_code=$?; printf "exit_code=%s\n" "$run_code" > "${run_output}.status"' EXIT
./run_rl_python.sh train_openings.py --output "$run_output" \
    --steps 12000 --eval-every 3000 --capture-repeats 3 \
    --seed 1000 --eval-seeds 1001 1002 \
    --initialize-from outputs/opening-rl-placement-v5/residual_best.zip
./run_rl_python.sh verify_opening_rl.py "$run_output"
