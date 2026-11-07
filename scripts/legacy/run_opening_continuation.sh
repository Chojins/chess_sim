#!/usr/bin/env bash
# Run after starting the headless control server; safe to detach with nohup.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
run_output="${1:-outputs/opening-rl-square-v2}"
if [[ -e "$run_output" ]]; then
    echo "Output already exists: $run_output" >&2
    exit 1
fi
mkdir -p -- "$(dirname -- "$run_output")"
trap 'run_code=$?; printf "exit_code=%s\n" "$run_code" > "${run_output}.status"' EXIT
./run_rl_python.sh train_openings.py \
    --output "$run_output" --steps 12000 --eval-every 3000 --seed 701 --eval-seeds 801 802 \
    --initialize-from outputs/opening-rl-pilot/residual_final.zip
./run_rl_python.sh verify_opening_rl.py "$run_output"
