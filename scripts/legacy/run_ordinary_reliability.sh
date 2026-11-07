#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
output=${1:-outputs/opening-ordinary-reliability-v15}
preceding_status=${2:-outputs/capture-complete-v15.status}
seed_start=${3:-2601}
[[ ! -e "$output" ]] || { echo "Output already exists: $output" >&2; exit 1; }
trap 'run_code=$?; printf "exit_code=%s\n" "$run_code" > "${output}.status"' EXIT
echo 'Waiting for the full-capture pipeline to release the simulator.'
while [[ ! -f "$preceding_status" ]]; do
    sleep 5
done
[[ "$(cat "$preceding_status")" == 'exit_code=0' ]] || {
    echo 'Capture pipeline failed; leaving the controller untouched for diagnosis.' >&2
    exit 1
}
seeds=()
for ((index=0; index<10; index++)); do seeds+=("$((seed_start+index))"); done
./run_rl_python.sh evaluate_openings.py --output "$output" \
    --pilot outputs/opening-rl-capture-pickup-v6 --only-policy learned \
    --ids initial_e2e4 initial_d2d4 open_game_knight \
    --seeds "${seeds[@]}" --repeats 2 \
    --record-video --learned-label 'retained ordinary-move expert (v6)' \
    --comparison-note 'Twenty fresh complete-move trials per ordinary task, frozen actor, current square-based scoring; no capture training changes'
