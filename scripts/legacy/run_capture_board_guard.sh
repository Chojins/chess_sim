#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
run_output=outputs/capture-board-guard-rl-v16
heldout_output=outputs/capture-board-guard-heldout-v16
for path in "$run_output" "$heldout_output"; do
    [[ ! -e "$path" ]] || { echo "Output already exists: $path" >&2; exit 1; }
done
trap 'run_code=$?; printf "exit_code=%s\n" "$run_code" > "outputs/capture-board-guard-v16.status"' EXIT
./run_rl_python.sh train_openings.py --output "$run_output" \
    --ids scandinavian_capture --capture-alignment \
    --live-timing --adaptive-alignment --latch-alignment --board-grasps-only \
    --grasp-interventions 4 --record-evaluations \
    --initialize-from outputs/capture-replay-rl-v14/residual_best.zip \
    --demonstrations outputs/capture-replay-rl-v14/teacher_replay/demonstrations.npz \
    --bc-weight .5 --exploration-log-std -3 \
    --steps 6000 --eval-every 2000 --seed 2700 --eval-seeds 2701 2702 2703
./run_rl_python.sh verify_opening_rl.py "$run_output"
./run_rl_python.sh evaluate_openings.py --output "$heldout_output" \
    --pilot "$run_output" --ids scandinavian_capture \
    --baseline-checkpoint residual_initial.zip --learned-checkpoint residual_best.zip \
    --baseline-label 'transferred pickup actor with board-only timing' --learned-label 'selected complete-capture actor' \
    --seeds 2801 2802 2803 2804 2805 --repeats 2 --record-video \
    --comparison-note 'Full captures with timing guards restricted to the static square footprint; tabletop release is never guarded as a pickup'
