# ACT evaluation and residual RL bring-up

The first milestone is implemented: a Gymnasium environment, a versioned task
reward, a bounded residual wrapper, and an ACT execution-window evaluator.
No RL learner has been trained yet. Existing ACT weights and real datasets are
unchanged. The validated task is **blue e2e4 with red held at home**.

## Running the simulator and checks

Start one camera/control server:

```bash
./run_cameras.sh --replay-episode outputs/replay/episode_000000.npz \
  --joint-map calibration/legacy_blue_joint_map_physics.json \
  --replay-mode physics --contacts --magnets --control --fps 30 \
  --output outputs/control-rl
```

Use `/home/jacob/miniconda3/envs/lerobot312/bin/python` for the client scripts.
It already has Gymnasium, NumPy, OpenCV and the ACT dependencies installed.

```bash
python -m unittest test_chess_task test_chess_env test_board_magnets
python verify_chess_env.py
python verify_control.py
python verify_reward_replay.py --output outputs/reward-replay
python evaluate_act.py --windows 100 25 10 --seeds 11 23 37 \
  --pawn-jitter-mm 0.5 --output outputs/act-window-pilot
python verify_evaluation.py outputs/act-window-pilot
```

Only one client may control the server at a time. Run checks sequentially.
Output directories for replay/evaluation must be new. The server remains
running and paused between commands when the client finishes.

## Environment contract

`ChessEnv` in `chess_env.py` follows the
[Gymnasium reset/step API](https://gymnasium.farama.org/api/env/):

```python
from chess_env import ChessEnv

env = ChessEnv(max_steps=400, pawn_jitter_mm=0.5)
obs, info = env.reset(seed=11)
# action_legacy is a six-element absolute target, such as an ACT prediction.
obs, reward, terminated, truncated, info = env.step(action_legacy)
env.close()
```

Actor observations are `state` (six legacy joint readings), `laptop_rgb` (red
camera with the e2e4 overlay), and `phone_rgb` (raw blue camera). Images are
uint8 HWC RGB at the existing 640x480 calibration. Exact simulator object
positions and reward diagnostics are in `info`, not actor observations.
Do not feed all of `info` into a controller intended to operate on hardware.

Actions are absolute targets in the existing legacy convention: five arm
angles and the old gripper percentage. Bounds are derived from the calibrated
map and USD joint limits. The plain environment rejects invalid actions before
sending them. Each accepted step advances 1/30 simulated second and verifies
the frame chain and timestamps. Lost replies or mismatched captures raise an
exception and require reset; they never produce a fabricated transition.

`terminated` indicates task success or failure; `truncated` indicates the
episode time limit. Learners should preserve this distinction for value
bootstrapping. Reset uses a seeded RNG for requested pawn offsets within
the configured range (maximum +/-2 mm per board axis). The server applies
offsets after initial settling and reports them. An explicit
`options={'pawn_offset_mm': [x, y]}` overrides sampling. The source baseline
and CAD destination square remain fixed. Seeding does not guarantee bitwise
determinism of PhysX or camera rendering.

## Task reward: e2e4_v1

The reward uses simulator geometry and history:

| Event | Reward / condition |
| --- | --- |
| Time | -0.001 per step |
| First sustained lift | +1; pawn above 5 mm for three consecutive steps |
| First arrival over target while lifted | +1; within 5 mm of CAD e4 centre |
| Completed move | +10; stable upright placement, released, and arm home |
| Other piece displaced over 3 mm | -2 and terminate as failure |

Lift and transport bonuses can be earned only once per episode. Completed
placement requires height within 3 mm of the initial resting height, upright
cosine above 0.95, and movement below 0.5 mm per step for ten steps. A gripper
tool-point proxy must be more than 35 mm from the pawn. Return home requires
departure by over 15 degrees first, then ten steps within 5 degrees of the
initial arm pose. The final home condition provides additional separation
from the piece. The tool point is a kinematic proxy, not a contact sensor.

Returning home without completing the move is classified as `failed_grasp`,
`wrong_placement`, or `unstable_or_unreleased`. Other outcomes are `success`,
`disturbed_other_piece`, and `timeout`. The evaluator separately records
invalid policy actions and infrastructure errors. Scoring is stricter than
the earlier two-run ACT check: it requires sustained lift and stable release,
and uses the CAD square centre rather than initial pawn position plus 45 mm.
Do not compare its success percentages to earlier reports as identical tests.

The reward is a manipulation proxy to validate before long training runs;
it does not prove a mechanically secure grasp. The magnet strength, friction,
piece masses and motor dynamics still need real-world calibration. The current
joint map has only been validated for e2e4 and exceeds limits in a knight demo.

## Evaluation outputs

`evaluate_act.py` preserves the checkpoint's predicted chunk size, changing
only how many actions are executed before replanning. It resets the ACT queue
between trials and rotates window order across seeds. All windows receive the
same requested pawn perturbation for each seed; initial actual states and
physics settings are saved for audit.

Outputs include `manifest.json` (checkpoint SHA256, task parameters, seeds),
`summary.json`, `results.json`, and per-trial initial/final observations, camera
snapshots and step traces. Traces contain state/action/frame chaining, rewards,
termination flags and timing. These are diagnostic evaluations, not complete
camera transition datasets or expert demonstrations. A few seeds establish
pipeline behaviour, not a reliable success-rate estimate.

## Residual controller interface

`ResidualACTEnv(env, policy)` wraps a frozen `LegacyACT`. Its actions are six
normalized values in [-1, 1]. Default correction bounds are 2 degrees for each
arm joint and 3 degrees for the gripper joint. Corrections are converted using
the map's signs and scale; the gripper correction is not incorrectly treated
as an arm angle in the legacy input. Joint-limit projection is explicit and
reported, and correction magnitude incurs a 0.01 * mean(action²) penalty.

Observations additionally include ACT's next base action and the chunk phase.
The actor remains partially observed: chunk phase does not expose the whole
queued trajectory, and camera history may be useful. Time-limit observations
include a valid next base action for bootstrapping. `close()` does not terminate
the externally managed simulator.

## Next training decisions

Choose the baseline execution window from a larger held-out evaluation after
the pilot, then attach a small residual learner and validate its reward use.
Keep policy inputs available on the real robot; privileged simulator state
can be used by a separate critic or a teacher intended for later distillation.

The current policy environment has PyTorch 2.11.0 built for CUDA 13.0; its
CUDA initialization fails with the installed driver. These evaluations use
CPU inference. Select a compatible PyTorch/CUDA environment and test GPU
compute before a substantial training run. The current HTTP/image transport
and one rendered scene also limit throughput; measure before scaling to
multiple environments. No driver or package changes are made by these scripts.

## Workstation verification, 2026-09-15

Fourteen numerical/transport unit tests passed, as did live checks of spaces,
action rejection, seeded reset offsets, fixed targets, 30 Hz stepping, timeout
semantics and zero residual equivalence. The existing paired camera/control
integration checks also passed.

The positive control (`outputs/rl-baseline-positive/report.json`) replayed the
recorded move at half speed through the Gym environment. It passed after 642
control steps, with 22.59 mm maximum lift and 0.40 mm CAD target error. The
reward identified lift, transport, release and return home.

The ACT pilot used three paired seeds with +/-0.5 mm requested pawn jitter:

| Actions executed before replanning | Successes | Outcomes | Control steps / wall second |
| --- | --- | --- | --- |
| 100 | 0/3 | Failed grasp, returned home | 7.12 |
| 25 | 0/3 | Timeout, no sustained lift | 6.84 |
| 10 | 0/3 | Timeout, no sustained lift | 6.47 |

All 3,266 transitions passed the evaluation integrity checks. Reports are in
`outputs/act-window-pilot/summary.json` and `results.json`, with each trial's
initial conditions, physics settings, traces and camera snapshots alongside.
This small pilot gives no evidence to adopt a shorter execution window. Keep
100 as the reference, and investigate grasp robustness before a substantial
RL run. Unlike the earlier nominal two-trial check, this experiment includes
small reset perturbations and stricter scoring; it does not establish that
ACT can never succeed. No learned residual weights were produced.
