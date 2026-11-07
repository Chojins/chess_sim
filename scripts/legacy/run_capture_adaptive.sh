#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
run_output=outputs/capture-adaptive-rl-v13
heldout_output=outputs/capture-adaptive-heldout-v13
for path in "$run_output" "$heldout_output"; do
    if [[ -e "$path" ]]; then
        echo "Output already exists: $path" >&2
        exit 1
    fi
done
trap 'run_code=$?; printf "exit_code=%s\n" "$run_code" > "outputs/capture-adaptive-v13.status"' EXIT
./run_rl_python.sh train_openings.py --output "$run_output" \
    --ids scandinavian_capture --pickup-only --capture-alignment \
    --live-timing --adaptive-alignment --record-evaluations \
    --initialize-from outputs/capture-supervised-rl-v10/residual_best.zip \
    --demonstrations outputs/capture-demo-v10/demonstrations.npz --bc-weight .1 \
    --steps 6000 --eval-every 2000 --seed 1900 --eval-seeds 1901 1902 1903
./run_rl_python.sh verify_opening_rl.py "$run_output"
./run_rl_python.sh evaluate_openings.py --output "$heldout_output" \
    --pilot "$run_output" --ids scandinavian_capture \
    --baseline-checkpoint residual_initial.zip --learned-checkpoint residual_best.zip \
    --baseline-label 'initial residual with adaptive timing' --learned-label 'selected adaptive timing residual' \
    --seeds 2001 2002 2003 2004 2005 --repeats 2 --record-video \
    --comparison-note 'Fresh-seed live pickup evaluation with open-jaw residual alignment and close-before-lift control; not full capture success'
