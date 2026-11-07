#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
demo_output=outputs/capture-demo-v10
run_output=outputs/capture-supervised-rl-v10
heldout_output=outputs/capture-supervised-heldout-v10
for path in "$demo_output" "$run_output" "$heldout_output"; do
    if [[ -e "$path" ]]; then
        echo "Output already exists: $path" >&2
        exit 1
    fi
done
trap 'run_code=$?; printf "exit_code=%s\n" "$run_code" > "outputs/capture-supervised-v10.status"' EXIT
./run_usd_python.sh export_capture_kinematics.py --output outputs/capture-supervised-kinematics-v10.json
./run_rl_python.sh prepare_capture_demonstrations.py --output "$demo_output" --kinematics outputs/capture-supervised-kinematics-v10.json
./run_rl_python.sh pretrain_capture_residual.py --demonstrations "$demo_output/demonstrations.npz" --updates 1200
./run_rl_python.sh train_openings.py --output "$run_output" \
    --ids scandinavian_capture --pickup-only --capture-alignment --record-evaluations \
    --initialize-from "$demo_output/pretraining/demo_pretrained.zip" \
    --demonstrations "$demo_output/demonstrations.npz" --bc-weight .5 \
    --steps 6000 --eval-every 2000 --seed 1600 --eval-seeds 1601 1602 1603
./run_rl_python.sh verify_opening_rl.py "$run_output"
./run_rl_python.sh evaluate_openings.py --output "$heldout_output" \
    --pilot "$run_output" --ids scandinavian_capture \
    --baseline-checkpoint residual_initial.zip --learned-checkpoint residual_best.zip \
    --baseline-label 'supervised initial actor' --learned-label 'supervised plus RL selected actor' \
    --seeds 1701 1702 1703 1704 1705 --repeats 2 --record-video \
    --comparison-note 'Fresh-seed pickup comparison after supervised warm-start and anchored RL; no online IK assistance'
