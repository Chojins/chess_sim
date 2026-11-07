#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
run_output=outputs/capture-replay-rl-v14
heldout_output=outputs/capture-replay-heldout-v14
for path in "$run_output" "$heldout_output"; do
    if [[ -e "$path" ]]; then
        echo "Output already exists: $path" >&2
        exit 1
    fi
done
trap 'run_code=$?; printf "exit_code=%s\n" "$run_code" > "outputs/capture-replay-v14.status"' EXIT
./run_rl_python.sh train_openings.py --output "$run_output" \
    --ids scandinavian_capture --pickup-only --capture-alignment \
    --live-timing --adaptive-alignment --record-evaluations \
    --initialize-from outputs/capture-adaptive-rl-v13/residual_best.zip \
    --replay-seeds 2104 2105 2106 2107 2108 2109 2110 2111 2112 2113 \
    --bc-weight .5 --exploration-log-std -3 \
    --steps 6000 --eval-every 2000 --seed 2100 --eval-seeds 2101 2102 2103
./run_rl_python.sh verify_opening_rl.py "$run_output"
./run_rl_python.sh evaluate_openings.py --output "$heldout_output" \
    --pilot "$run_output" --ids scandinavian_capture \
    --baseline-checkpoint residual_initial.zip --learned-checkpoint residual_best.zip \
    --baseline-label 'retained starting actor with adaptive timing' --learned-label 'selected actor after successful replay SAC' \
    --seeds 2201 2202 2203 2204 2205 --repeats 2 --record-video \
    --comparison-note 'Fresh-seed live pickup comparison after matched successful replay, lower exploration and teacher retention; not full capture success'
