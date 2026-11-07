#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
run_output=outputs/capture-complete-replay-rl-v17
capture_output=outputs/capture-complete-reliability-v17
ordinary_output=outputs/opening-ordinary-reliability-v17
for path in "$run_output" "$capture_output" "$ordinary_output"; do
    [[ ! -e "$path" ]] || { echo "Output already exists: $path" >&2; exit 1; }
done
trap 'run_code=$?; printf "exit_code=%s\n" "$run_code" > outputs/complete-replay-v17.status' EXIT
# Exactly zero initial correction reproduces the successful timing-only ACT
# teacher. Complete-move rewards and terminal flags enter replay; pickup-only
# actor weights/rewards are not reused.
./run_rl_python.sh train_openings.py --output "$run_output" \
    --ids scandinavian_capture --capture-alignment --capture-home-retry \
    --live-timing --adaptive-alignment --latch-alignment --board-grasps-only \
    --grasp-interventions 4 --record-evaluations \
    --replay-seeds 3104 3105 3106 3107 3108 3109 3110 3111 3112 3113 \
    --bc-weight .5 --exploration-log-std -4 \
    --steps 4000 --eval-every 2000 --seed 3100 --eval-seeds 3101 3102 3103
./run_rl_python.sh verify_opening_rl.py "$run_output"
./run_rl_python.sh evaluate_openings.py --output "$capture_output" \
    --pilot "$run_output" --ids scandinavian_capture --only-policy learned \
    --seeds 3201 3202 3203 3204 3205 3206 3207 3208 3209 3210 --repeats 2 \
    --record-video --learned-label 'selected complete-move residual with board-only timing' \
    --comparison-note 'Twenty fresh full captures; selected actor may retain the zero-correction teacher if SAC does not improve selection scores'
# Capture-only cannot pass the overall gate: proceed to ordinary-move tests
# only once its own per-task complete-move threshold is met.
./run_rl_python.sh - "$capture_output" <<'PY'
import json, sys
from pathlib import Path
from verify_policy_reliability import assess
root = Path(sys.argv[1])
report = assess(json.loads((root/'manifest.json').read_text()), json.loads((root/'results.json').read_text()))
(root/'capture_readiness.json').write_text(json.dumps(report, indent=2))
if not all(row['passed'] for row in report['per_task'].values()):
    raise SystemExit('Capture reliability below threshold: stop for analysis, without collecting ACT data')
(root/'pipeline_ready.status').write_text('exit_code=0\n')
PY
bash scripts/legacy/run_ordinary_reliability.sh "$ordinary_output" "$capture_output/pipeline_ready.status" 3301
./run_rl_python.sh verify_policy_reliability.py "$capture_output" "$ordinary_output" \
    --output outputs/complete-policy-readiness-v17.json
./run_rl_python.sh collect_opening_episodes.py --pilot "$run_output" \
    --output outputs/act-capture-episodes-v17 --episodes 20 --seed-start 3401 --record-video
./run_rl_python.sh collect_opening_episodes.py --pilot outputs/opening-rl-capture-pickup-v6 \
    --ids initial_e2e4 initial_d2d4 open_game_knight \
    --output outputs/act-ordinary-episodes-v17 --episodes 20 --seed-start 3501 --record-video
for kind in capture ordinary; do
    bash run_dataset_python.sh export_act_episodes.py \
        --index "outputs/act-${kind}-episodes-v17/accepted_episodes.json" \
        --output "outputs/act-${kind}-dataset-v17" --repo-id "local/chess-sim-${kind}-v17"
done
