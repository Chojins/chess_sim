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

## Training scope

Keep the 100-action baseline for the initial residual pilot, then choose the
execution window from a larger held-out evaluation before scaling training.
Keep policy inputs available on the real robot; privileged simulator state
can be used by a separate critic or a teacher intended for later distillation.

The original policy environment has PyTorch 2.11.0 built for CUDA 13.0;
the September 15 evaluations used CPU inference. The isolated learner
environment described below uses PyTorch 2.7.1 with CUDA 12.8. The HTTP/image transport
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

## Residual SAC pilot, prepared 2026-09-20

`train_residual.py` implements a small SAC learner over frozen ACT. Its actor
receives 4,096 frozen ResNet18 camera features from the existing checkpoint,
measured joints, ACT's proposed action, previous correction, chunk phase,
episode phase and a shoulder-angle gate. Camera normalization remains the
checkpoint's original normalization. Object positions are used for reward
only. The learner starts with a zero deterministic correction and uses gSDE
exploration, refreshed every 12 control steps.

Corrections remain bounded to +/-2 degrees per arm joint and +/-3 degrees at
the gripper, followed by explicit joint-limit projection. A smooth gate is
zero at home, rises between shoulder angles 60 and 80 degrees, stays at one
through 120 degrees, and falls to zero at 140 degrees. This is a heuristic for
the validated e2e4 motion, not a general chess controller.

The task reward and success criteria above remain in force. Additional
potential shaping encourages approach: `Phi = gate * clip(1 - clearance/0.03,
0, 1)` and `reward += 2 * (0.99 * Phi(next) - Phi(current))`. Clearance is the
tool-point distance to pawn centre. True terminations use zero next potential;
timeouts retain it for bootstrapping. Shaping, residual cost, task reward and
success are logged separately so shaped return cannot substitute for success.

The grasp comparison in [calibration/grasp_comparison.md](calibration/grasp_comparison.md)
shows about 10 mm tool-position error at ACT closure versus the successful
recording. An offline kinematic fit within the proposed arm correction bounds
reduces that error to 1.7–2.0 mm. It does not demonstrate a physical grasp.

### Isolated environment

The existing `.venv-rl` is prepared. To recreate it, from this repository:

```bash
./setup_rl_env.sh
./run_rl_python.sh -m pip check
./run_rl_python.sh verify_rl_runtime.py
./run_rl_python.sh verify_rl_runtime.py --cuda --output outputs/rl-runtime-gpu.json
```

Setup uses Python 3.12 from the existing `lerobot312` environment and imports
the checkout at `/home/jacob/lerobot/src` through a `.pth` file. Override
`RL_PYTHON` and `LEROBOT_SRC` if these paths change. Packages are installed
only in `.venv-rl`; Isaac, the old policy environment and the LeRobot checkout
are not modified. The launcher clears inherited ROS/CUDA Python/library paths.
`requirements-rl.txt` defines the install constraints; `requirements-rl-lock.txt`
records the tested environment. Its `lerobot==0.5.2` entry identifies the local
checkout metadata, not an instruction to replace that checkout with PyPI code.
The runtime parity check needs the saved `outputs/act-window-pilot` fixtures.

CPU verification passed: the largest first-action difference from the previous
runtime across three saved observations was 0.000016 legacy units. Frozen
camera features were finite and correctly shaped. Seventeen offline tests
passed, including bounded/gated control, termination versus timeout shaping,
synthetic SAC updates and save/load equivalence. The synthetic test is not
chess training. Dependency checks also passed.

### Run the pilot

The initial September 20 GPU check failed before model inference. The NVIDIA
kernel log reported `uvm encountered global fatal error 0x60` and
`Node Reboot Required`. Rebooting restored CUDA: the GPU runtime check passed
with maximum first-action difference of 0.00269 legacy units from the saved CPU
reference. Live camera/control and Gym integration checks also passed.
Require the CUDA runtime check above to pass before a new run.

Start a single simulator in one terminal and wait for `FIRST_FRAMES`:

```bash
./run_cameras.sh --replay-episode outputs/replay/episode_000000.npz \
  --joint-map calibration/legacy_blue_joint_map_physics.json \
  --replay-mode physics --contacts --magnets --control --fps 30 \
  --output outputs/control-residual
```

In another terminal, from this repository:

```bash
./run_rl_python.sh train_residual.py --output outputs/residual-pilot-new \
  --steps 3000 --device cuda --seed 123 --eval-seeds 101 102 103
```

The output directory must be new. The script evaluates zero residual on three
seeds, trains for 3,000 transitions with +/-0.5 mm pawn jitter, then evaluates
deterministically on the same three seeds. Training uses a separate seed.
Requested offsets match across evaluations; rendering and physics are not
guaranteed bitwise identical. Keep other control clients disconnected.

Outputs contain the configuration/checkpoint fingerprint, initial/latest/final
SAC checkpoints, training traces and episode outcomes, evaluation traces and
initial/final camera snapshots, and a comparison summary. ACT's weight hash is
checked again at completion. This is a bring-up pilot: three evaluation seeds
cannot establish reliability, and improvement must mean more completed moves,
not merely higher shaping reward. The replay buffer is not saved for resume.

Validate a completed run with:

```bash
./run_rl_python.sh verify_residual_pilot.py outputs/residual-pilot-new
```

The audit checks rewards, correction bounds, physical joint limits, frame/time
continuity, paired reset offsets, terminal snapshots, finite saved weights,
actual residual actor updates and the unchanged ACT checkpoint hash.

The first live attempt (`outputs/residual-pilot`) stopped after 40 training
transitions because nearest float32 rounding put a clipped gripper command
0.000000015 degrees beyond its strict limit. `ChessEnv` now rounds action-space
endpoints inward. Eighteen offline tests and the live physical-boundary check
passed after this fix. That incomplete attempt remains separate from the
restarted experiment at `outputs/residual-pilot-v2`.

### Completed pilot results, 2026-09-20

The restarted pilot completed 3,000 training transitions and 2,744 SAC gradient
updates in 522.6 seconds (about 8.7 minutes). The ACT checkpoint stayed unchanged.

| Evaluation seed | ACT alone | ACT + learned residual | Final learned placement error | Maximum learned lift |
| --- | --- | --- | --- | --- |
| 101 | Failed grasp | Success | 0.61 mm | 23.50 mm |
| 102 | Failed grasp | Success | 0.62 mm | 25.19 mm |
| 103 | Failed grasp | Success | 0.44 mm | 23.40 mm |

All three learned evaluations returned home and passed stable release and
neighbouring-piece checks. Exploration during training produced two successes
among fifteen completed episodes; other outcomes were three failed grasps,
nine neighbouring-piece disturbances and one wrong placement. Exploration
outcomes should not be treated as the deterministic policy's success rate.

`outputs/residual-pilot-v2/summary.json` contains the comparison, and
`residual_final.zip` is the learned SAC checkpoint (used together with frozen
ACT and the same feature/control wrappers). `integrity.json` records the
passing audit of all 4,721 transitions: 3,000 training and 1,721 evaluation.
The saved residual actor changed, its parameters are finite, both camera
snapshots are valid, and the original ACT weight hash matches.

This is encouraging preliminary evidence, not a reliability estimate or a
sim-to-real result. In the interrupted first attempt, ACT succeeded on seed
101; it failed on that same seed in the restarted baseline. Initial logged
joints and pawn centre matched exactly, but initial camera images differed
by about 0.59/0.80 intensity levels on average for laptop/phone. This establishes
that resets do not reproduce identical image inputs; it does not isolate the
cause of the different outcome. The diagnostic is saved alongside the results.

Next, evaluate the frozen final checkpoint against ACT on a larger set of
previously unused seeds, with repeated resets and interleaved controller order,
before increasing training duration or widening the perturbations.
