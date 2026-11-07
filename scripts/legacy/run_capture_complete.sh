#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
run_output=outputs/capture-complete-rl-v15
heldout_output=outputs/capture-complete-heldout-v15
for path in "$run_output" "$heldout_output"; do
    if [[ -e "$path" ]]; then
        echo "Output already exists: $path" >&2
        exit 1
    fi
done
trap 'run_code=$?; printf "exit_code=%s\n" "$run_code" > "outputs/capture-complete-v15.status"' EXIT
echo 'Waiting for the v14 controller client to finish before starting v15.'
while [[ ! -f outputs/capture-replay-v14.status ]]; do
    sleep 5
done
if [[ "$(cat outputs/capture-replay-v14.status)" != 'exit_code=0' ]]; then
    echo 'Previous pipeline failed; refusing to overlap or train from an incomplete run.' >&2
    exit 1
fi
# Complete-move rewards use fresh critics/replay. Pickup success rewards are
# not reused. Only the verified teacher's residual actions are retained by BC.
./run_rl_python.sh train_openings.py --output "$run_output" \
    --ids scandinavian_capture --capture-alignment \
    --live-timing --adaptive-alignment --latch-alignment --record-evaluations \
    --initialize-from outputs/capture-replay-rl-v14/residual_best.zip \
    --demonstrations outputs/capture-replay-rl-v14/teacher_replay/demonstrations.npz \
    --bc-weight .5 --exploration-log-std -3 \
    --steps 6000 --eval-every 2000 --seed 2400 --eval-seeds 2401 2402 2403
./run_rl_python.sh verify_opening_rl.py "$run_output"
./run_rl_python.sh evaluate_openings.py --output "$heldout_output" \
    --pilot "$run_output" --ids scandinavian_capture \
    --baseline-checkpoint residual_initial.zip --learned-checkpoint residual_best.zip \
    --baseline-label 'transferred pickup actor with latched timing' --learned-label 'selected complete-capture actor' \
    --seeds 2501 2502 2503 2504 2505 --repeats 2 --record-video \
    --comparison-note 'Fresh complete-capture evaluation: victim removal, attacker placement and home return; distinct from pickup diagnostics'
