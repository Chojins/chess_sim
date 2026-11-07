# ACT evaluation and residual RL bring-up

The workspace runs frozen ACT with bounded residual SAC, versioned chess-task
scoring, camera recording and audited episode export. Existing ACT weights and
real datasets remain unchanged. The v29 hybrid baseline passed the tested
blue-robot opening cases, with red held at home. Its scripted capture transport
still adds pauses and an intermediate home return; it is not yet the desired
motion-quality standard for large-scale ACT data collection. See the
[baseline record](docs/baselines/v29.md) for results and remaining limitations.

The dated sections below retain earlier experiment results and their limits.
Superseded shell launchers now live in `scripts/legacy/`; reusable Python tools
remain at the project root. Historical readiness statements refer to the
completion/export checks at that stage, not approval of trajectory quality.

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

### Held-out repeated evaluation, 2026-09-20

The larger comparison is complete in `outputs/residual-heldout`. No further
training was performed. It used seeds 201–210, two resets per seed, and both
controllers on each reset: 40 episodes total. Controller order alternated
between seeds and reversed on the second repeat. Pawn jitter stayed at
+/-0.5 mm, with the same e2e4 task, magnets, contacts and correction bounds.

| Controller | Successful moves | Other outcomes |
| --- | --- | --- |
| Frozen ACT alone | 1/20 (5%) | 17 missed grasps, 1 neighbouring-piece disturbance, 1 wrong placement |
| Frozen ACT + saved residual | 19/20 (95%) | 1 missed grasp |

Successful learned placements ended 0.43–0.63 mm from e4's centre. All passed
stable release, return home and neighbouring-piece checks. The learned policy
succeeded 10/10 when evaluated first within its pair and 9/10 when second.
There were 19 pairs where only the residual succeeded and one where only ACT
succeeded. The one residual failure was seed 209 on repeat 0; it barely lifted
the pawn (0.094 mm maximum) and returned home. It succeeded on repeat 1.
ACT had the opposite outcome on that seed, confirming residual reset/render
variability rather than a deterministic success/failure label for each seed.
The precise cause of the remaining missed grasp has not been isolated.

The full run took 1,929.5 seconds (32.2 minutes). The offline audit passed all
11,324 transitions, checking rewards, limits, frame/time continuity, camera
snapshots, identical requested offsets, closely matching actual paired reset
states, constant simulator settings, and unchanged ACT and SAC checkpoint
files. `summary.json`, `results.json`, and `integrity.json` contain the results;
each trial directory contains traces and initial/final camera observations.

Reproduce this evaluation in a new output directory, with the simulator already
running in the same control configuration:

```bash
./run_rl_python.sh evaluate_residual.py --output outputs/residual-heldout-new \
  --seeds 201 202 203 204 205 206 207 208 209 210 --repeats 2 --device cuda
./run_rl_python.sh verify_residual_evaluation.py outputs/residual-heldout-new
```

The evaluator refuses seeds used in training or previous pilot evaluations,
freezes the loaded residual parameters, validates task/action compatibility,
and saves its complete schedule before moving. Infrastructure errors stop the
run and are recorded separately; they are not silently counted as policy
failures. Two offline tests cover counterbalancing and paired outcome counts.

These results support an improvement in this simulated e2e4 setup, not a 95%
real-world reliability claim. Repeats share only ten distinct offset seeds and
the physical parameters remain provisional. Keep this checkpoint fixed for
the next test: wider pawn offsets (for example +/-1 mm, then +/-2 mm), followed
by plausible magnet/friction variation before further training or hardware use.

## Legal opening positions and captures: implementation status

The next stage is implemented and passed live reset and recording checks on
September 22 after a reboot restored CUDA. The September 20 launch had hit a
fatal NVIDIA UVM error (`Node Reboot Required`). Twenty-five offline tests
also pass, including legal move histories, capture identities, en passant,
capture sequencing, and the existing task/control/residual regressions.

`opening_curriculum.py` reuses the existing `/usr/games/stockfish` workflow
with 0.1-second searches. `outputs/openings/curriculum.json` contains 28 legal
tasks: four fixed probes plus three eight-ply lines beginning e4, d4 and c4.
Fifteen are executable with the currently validated blue calibration. The
others remain explicitly marked unavailable; red control is not silently
assigned blue's calibration. Generated games are saved, so replay does not
depend on Stockfish choosing the same continuation on a later run.

`BoardLayout` tracks physical piece identities from the initial position
through each legal UCI history. `/reset` accepts `history` and `move`, restores
the resulting board, and sets a move-specific overlay and reward. Previously
captured pieces are parked off-board in their captor's own right-side area.
The authored USD has the red king and queen swapped; legal resets correct
their positions by piece type in memory. Legacy resets preserve their previous
behaviour, and no source USD is saved or modified.

Capture areas follow [the robot-relative convention](calibration/capture_zones.md):
120 mm outward from the board-frame edge, 150 mm along the board, and on the
lower tabletop (scene Z=0). Blue's right is world +X; red's is world -X. The
150 mm span is provisionally centred alongside the board. These dimensions
are approximate, not a measured calibration.

The `legal_move_v1` capture task enforces the demonstrated single-gripper
sequence: lift the correct victim, release it upright and stable in the acting
robot's right-side area, then move the attacker. Moving the attacker more than
3 mm before that release fails the sequence. The intended victim is exempt
from the unrelated-piece disturbance penalty; other pieces remain protected.
Success requires the attacker at its target, both releases valid, and the arm
home. En passant uses the victim's actual square. Castling and promotion
execution are rejected until their multi-piece goals/assets are implemented.

Start the updated camera server with the usual physics,
contacts, magnets and control options, then run:

```bash
./run_rl_python.sh verify_board_resets.py
./run_rl_python.sh benchmark_openings.py --output outputs/opening-baseline --record
./run_rl_python.sh verify_opening_recordings.py outputs/opening-baseline
```

The baseline probes e2e4, d2d4, g1f3 after e4/e5, and e4xd5 after e4/d5.
It uses frozen ACT with zero residual and explicit joint-limit projection;
the learned e2e4 residual is not assumed to generalize. Captures allow 800
steps; ordinary moves allow 400. Each position is an independent reset, so
failure does not advance a logical game or prevent exercising later positions.

With `--record`, every observation includes synchronized highlighted fixed-
camera RGB, wrist RGB and measured legacy joints in NPZ files. Transition
logs contain requested and executed actions, limit projection, frame IDs,
simulation time, rewards and terminal outcomes. This is a diagnostic episode
format, not yet a LeRobot dataset export. Failed attempts are not expert
demonstrations; review successful episodes before adding them to ACT training.

### Opening baseline results, 2026-09-22

Live reset verification passed for the standard starting position, the position
after e4/d5, and a position with a previously captured pawn parked on the
tabletop. Maximum piece-centre placement error after settling was 0.0022 mm.
Illegal moves and unvalidated red control were rejected without advancing the
published frame, and legacy e2e4 resets still worked. Evidence is in
`outputs/openings/reset-verification/report.json`.

The four ACT-only probes completed without transport or GPU errors:

| Probe | Outcome | Steps | Limited commands |
| --- | --- | --- | --- |
| e2e4, standard position | Missed grasp, returned home | 289 | 0 |
| d2d4, standard position | Missed grasp, returned home | 296 | 0 |
| g1f3, after e4/e5 | Disturbed another piece | 104 | 4 |
| e4xd5, after e4/d5 | Victim removed correctly; attacker grasp missed, returned home | 598 | 0 |

For the capture, ACT lifted the correct red pawn and released it upright and
stable in blue's right-side tabletop area. The scorer advanced from
`remove_captured` to `move_attacker`. The blue pawn never reached the sustained
lift threshold, so the full move correctly remained a failure despite the
successful disposal. This supports retaining the demonstrated capture sequence
and focusing correction on grasp execution rather than introducing a new
capture procedure.

The knight trial's four limited commands exceeded the wrist-flex limit by
0.056–0.246 degrees. This does not establish why the neighbouring piece moved;
inspect its recorded approach before interpreting that outcome as a pure
policy-learning failure. The joint map still has only e2e4 validation.

`outputs/opening-baseline` contains 1,287 transitions and 1,291 paired RGB/state
observations (about 650 MiB). The full recording audit passed camera shapes,
state/action/frame/time alignment, physical command limits, reward totals,
terminal outcomes and the unchanged ACT checkpoint hash. None of these four
complete episodes is a successful demonstration. No ACT retraining occurred.
These are single probes per task, not success-rate estimates, and used no
learned residual; they are not directly comparable to the earlier 19/20
ACT-plus-residual e2e4 result.

Next, use the recorded approaches and existing demonstrations to validate
calibration/contact behaviour for d2d4, g1f3 and the second grasp in e4xd5.
Then test or train move-conditioned corrections with capture-phase-aware
rewards before generating a successful imitation-training dataset.

## Multi-move RL without further calibration fitting

Following Jacob's preference, the next pilot trains against the current
simulator calibration rather than fitting each failed move first.
`train_openings.py` cycles through the four opening probes with a fresh SAC
residual. ACT remains frozen, and the previously successful single-move
residual checkpoint remains separate. Corrections retain the +/-2 degree arm
and +/-3 degree gripper bounds, shoulder gate and explicit joint-limit
projection. Episodes alternate by task, with 400-step ordinary-move and
800-step capture limits. Pawn jitter is +/-0.5 mm.

The actor receives the existing 4,117 camera/proprioception features plus five
game-command features: source file/rank, destination file/rank and capture
flag. These commands are available from the game coordinator in real use.
Simulator piece positions and the privileged capture-phase state are not
actor inputs. The approach potential targets the victim during removal and
the attacker after a verified victim release; the task reward still enforces
capture order, placement, release, return home and protected pieces.

```bash
./run_rl_python.sh train_openings.py --output outputs/opening-rl-pilot-new --steps 6000
./run_rl_python.sh verify_opening_rl.py outputs/opening-rl-pilot-new
```

The pilot uses training seed 501 and evaluates seeds 601/602 on each task
before and after learning. Two trials per task are only a bring-up check.
The saved multi-move SAC policy needs `OpeningLearningEnv` (4,122 features);
it is not interchangeable with the old single-move wrapper. Twenty-seven
offline tests passed before the first live run, including victim-to-attacker
reward switching and separation of training/evaluation task selection.

Calibration randomization is intentionally not enabled in this first pilot.
The proposed next stage is bounded per-episode physical joint-zero offsets,
then small scale variation, while keeping the controller's nominal mapping
fixed. Randomized physical calibration must not leak into actor features or
silently weaken joint limits. Evaluate on unseen calibration draws, including
separate blue/red setup profiles once red has a usable nominal mapping.
This targets robustness to calibration variation; performance in a fixed
simulator alone does not establish that robustness or real-world transfer.

### Completed multi-move pilot (audited 2026-09-24)

`outputs/opening-rl-pilot` completed 6,000 training transitions and 5,744
gradient updates in 1,020.5 seconds (17 minutes of training). ACT's weights
remained unchanged. Calibration offsets/scales were not randomized.

| Task | ACT alone | ACT + final multi-move residual |
| --- | --- | --- |
| e2e4 | 2/2 | 0/2 |
| d2d4 | 0/2 | 1/2 |
| g1f3 after e4/e5 | 0/2 | 2/2 |
| e4xd5 after e4/d5 | 0/2 | 1/2 |
| Total | 2/8 | 4/8 |

The successful capture completed the full demonstrated sequence: correct victim
lifted and released in blue's right-side tabletop area, attacker placed on d5,
and arm returned home. Final attacker placement error was 0.42 mm. The other
capture evaluation missed the initial victim grasp. Both learned knight trials
succeeded, with about 0.53 mm final placement error. The learned e2e4 trials
both missed the grasp, so the broader policy is not an across-the-board upgrade.

The audit passed all 11,086 transitions (6,000 training and 5,086 evaluation),
including paired reset requests, reward totals, frame/time continuity, command
limits, camera snapshots, changed finite residual weights and the unchanged ACT
checkpoint. Results are in `summary.json`, `baseline_results.json`,
`learned_results.json` and `integrity.json`. The previous single-move checkpoint
is preserved and has not been replaced by this experiment.

This supports proceeding with RL despite imperfect calibration, but two trials
per move cannot establish reliable performance. The next experiment should
address the e2e4 regression while checking the new skills on more seeds, before
expanding calibration randomization. Successful evaluation traces here contain
initial/final images only; they must be replayed with the full episode recorder
before serving as complete ACT training demonstrations. No ACT retraining has
been performed.


### Next run: square-based disturbance reward

`legal_move_v2` allows unrelated on-board pieces to move while their centres
remain inside their original CAD square. Crossing a square edge (including an
outer board edge), or tilting more than 30 degrees, fails the episode. This is
a centre-based occupancy test, not a whole-footprint test. Previously captured
pieces may shift within their original capture zone. The intended victim and
attacker remain exempt from this unrelated-piece check; capture ordering still
requires removing and releasing the victim before moving the attacker more
than 3 mm. Legacy single-move `e2e4_v1` scoring is unchanged.

The next four-task run uses 12,000 steps, training seed 701 and evaluation
seeds 801/802. It warm-starts the actor from
`outputs/opening-rl-pilot/residual_final.zip`, with fresh critics, optimizers
and replay because the reward changed. ACT-only and the inherited residual
are evaluated before training; the learned residual is evaluated afterwards.
The runner audits the saved results automatically and writes its final status.
Calibration remains fixed. These small evaluations are a progress check.


### Grasp-centre follow-up

The square-v2 run completed 12,000 steps and passed its integrity audit,
but scored 3/8 versus 5/8 for its inherited actor (ACT 2/8). The next run
uses the earlier actor with a revised approach reference based on the user's
132 mm measurement from joint 4; see `calibration/grasp_reference.md`.
Release criteria and physical parameters are unchanged.


### User-specified piece grasp heights

The current approach reward now uses a 7 mm finger-front setback (at zero
hinge angle), with body-relative targets at pawn/knight 17 mm, rook 19 mm,
bishop 27 mm, king 32 mm and queen 30 mm above each piece's bottom. See
`calibration/grasp_reference.md`. This supersedes the preceding 132 mm
joint-4 reference and bounding-box-centre target. No training run has been
launched with these settings yet.


### Grasp heights run with checkpoint selection

`outputs/opening-rl-heights-v4` uses 12,000 training steps and evaluates at
the first completed episode at/after each 3,000-step interval, plus the final
checkpoint. Intermediate evaluation is excluded from replay; training resumes
with a fresh observation and the next scheduled task. Checkpoints are retained
and `residual_best.zip` selects the most successes across the four tasks and
two evaluation seeds, retaining the earlier checkpoint on ties. The initial
warm-start actor is also eligible. These reused selection seeds are validation
data, not an independent reliability estimate. Final integrity checks verify
the selected actor matches the saved checkpoint and its reported scores.

Run settings: fixed calibration, user-specified 7 mm finger setback and six
piece heights, earlier opening-rl-pilot actor, fresh critics/replay, training
seed 701 and selection seeds 801/802. No automatic extension beyond 12,000
steps. Intermediate evaluations add runtime.


### Placement overshoot diagnosis and reward v4

Correction to earlier interpretation: heights-v4 e2e4 terminal images show
e5, not e3. At steps 191–193 of learned_801 the piece is approximately
20 mm past e4 while the commanded fingers open; it subsequently settles
about 22.7 mm from e4. The shoulder is approximately 116.8 degrees, so the
existing correction gate is fully active. It has not been widened.

`diagnose_release.py` writes `outputs/placement-audit/release.json`. An
offline FK fit at the release commands can bring the configured grasp point
to the e4 grasp location within the existing +/-2 degree arm bounds. This
is a point-position feasibility test, not proof of collision-free motion,
correct orientation or contact stability. No robot calibration was changed.

Reward version `approach_plus_placement_potential_v4` adds twice a placement
potential to the existing approach potential before applying the existing
2*(0.99*Phi_next-Phi_previous) shaping. After a confirmed lift, placement
potential is exp(-XY_error_squared/0.0225^2 - Z_error_squared/0.020^2),
using the actual piece centre and its destination resting centre. For a
capture victim it uses the nearest XY point inside the correct capture zone
and that piece's tabletop resting height until release; then it switches
to the attacker. No placement potential is awarded before a confirmed lift.
True-terminal potential remains zero; timeouts retain potential. Success,
release, collision handling, magnets, action bounds and actor inputs are
unchanged. The reward does not force an early release or override ACT.

New transition records include piece positions, measured joints and both
gripper reference points, allowing release direction and timing to be
checked directly rather than inferred from unsigned distance. No new
training run was launched as part of this diagnosis/change.


### Capture-order tolerance correction

`legal_move_v3` replaces the capture attacker's 3 mm displacement failure
with leaving its original square (piece-centre XY test). All six learned
capture-order failures in the fresh-seed v5 evaluation were in-square
nudges at termination. Their subsequent outcomes are unknown; no historical
results are relabelled. Capture release still requires the existing stable
tabletop placement in the correct RHS area. See calibration/capture_zones.md.
New evaluation manifests record the current task version separately from
the training checkpoint's original task version. No evaluation or training
run was launched as part of this rule change.


### Capture removal quality, legal_move_v4

Settled non-upright captures in the RHS zone may complete, with +0.25
removal credit. Successful moves receive an additional +0.75 if the victim
is still upright and stably parked at completion. Captures on their sides
therefore remain acceptable but less rewarded. On-board upright placement
requirements are unchanged. Details and offline trace checks are documented
in calibration/capture_zones.md; no new training/evaluation run started.


### Victim pickup learning, legal_move_v5

Capture victims previously had approach shaping and a release milestone,
but no confirmed-lift milestone like the attacker. The scorer now pays
`capture_lift` +1 exactly once when the victim rises over 5 mm above its
reset height for three consecutive steps. Brief bumps reset the streak;
repeated lifting cannot collect additional bonuses. This is the same pose
evidence used for attacker lifting, not a contact-sensor proof of grasp.
Release, square-boundary ordering and upright-finish rules are unchanged.
Capture telemetry adds height_above_start_m and lift_streak.

`scripts/legacy/run_capture_pickup.sh` prepares 12,000 steps warm-started from the v5 best
actor, with fresh critics/replay. Three capture occurrences per cycle plus
one each of the other three tasks make captures half of scheduled training
episodes. Evaluations stay balanced across all four tasks at approximately
3,000-step boundaries and retain the best checkpoint, including the initial
actor. Training seed 1000 and selection seeds 1001/1002 are new. Existing
checkpoints are preserved. Success has not yet been validated with this
change; the initial CUDA preflight failed, so no run was launched.


### Recorded validation and timing, v6

`outputs/opening-recorded-v6` compares v6's selected best actor with its
initial actor (the previous v5 best) on seeds 1101–1105, two counterbalanced
repeats and all four tasks: 80 episodes, frozen weights. Current legal_move_v5
scoring and grasp geometry apply to both.

Every episode saves browser-compatible H.264 `cameras.mp4`, fixed highlighted
camera on the left and blue wrist camera on the right. There is one reset
frame and one frame for every physics/control transition, including the
terminal frame; playback is 30 frames/second of simulation time. The encoder
and ffprobe verify codec, dimensions and exact frame count before accepting
the result. `videos.html` indexes completed episodes by policy/task/outcome.

Timing instrumentation measures server physics/control advance, observe,
render, readback, preview JPEGs and pair packaging, plus client command,
paired transfer/decode, ACT action preparation, frozen vision features,
residual prediction and video writing. Summaries include per-step means
and 95th percentiles. Nested timings overlap and must not be summed twice.
These evaluation measurements exclude SAC gradient updates; video writing
adds overhead. The timed calls naturally include synchronization where
outputs are copied back to CPU; this is wall timing, not a CUDA kernel
profile. Physics settings and camera capture frequency are unchanged.


### Recorded pickup review and isolated diagnostic v7

The recorded v6 comparison's failed learned_1102 repeat-0 pickup closes
with the configured grasp point about 12 mm laterally from the victim at
step 120 (plus about 4 mm height offset), then rises without it. The
baseline_1101 repeat-0 successful clip carries the victim as the arm rises.
See outputs/capture-pickup-review.jpg and the source H.264 recordings.
These examples support a closure/alignment issue; they do not establish
the cause of all failed captures.

`scripts/legacy/run_pickup_only.sh` launches a 3,000-step diagnostic from the previous
v5 best actor, which was stronger on capture in the recorded comparison.
Task `capture_pickup_v1` requires the victim centre at least 20 mm above
reset, upright cosine >.95, and its configured grasp target within 15 mm
of the tool reference for ten consecutive frames. It terminates on pickup
success and times out at 240 control steps; full capture-order and other
piece protections remain. The existing 800-step actor time normalization
is retained so warm-start observation semantics stay consistent.

Shaping is approach potential plus twice clipped victim lift progress
(height/20 mm) times proximity (1-distance/20 mm). Rewards use simulator
truth, while actor features/action bounds remain unchanged. Captured-piece
transport potential is replaced for this experiment. A held-piece test
is geometric evidence, not a force/contact sensor. Brief lifts, tilted or
far-away pieces cannot meet pickup success.

Training seed 1200 and evaluation seeds 1201/1202/1203; evaluations at
approximately 1,000-step episode boundaries and final, with saved H.264
videos and best-checkpoint retention. Goal is to isolate whether bounded
RL can improve pickup, not to establish complete capture reliability.
Checkpoints under outputs/capture-pickup-only-v7 must not be substituted
for full-move policies without full capture evaluation. Existing policies
and camera/physics/calibration settings are preserved.

### Centred-closing capture pickup diagnostic v9

The v8 privileged physics probe completed six selected recorded-trajectory
episodes: original commands lifted/held the victim in 1/3 cases; corrected
approaches did so in 3/3. Both previous high/backward misses were corrected.
Measured grasp error before closure was about 1.8 mm, with both fingers open.
Victims lifted about 30 mm and stayed upright; other pieces stayed on their
squares. This validates the assisted approach on those cases, not a learned
policy, complete captures, or generalization. See
`outputs/capture-grasp-physics-v8/videos.html` and `review.json`.

`scripts/legacy/run_capture_alignment.sh` launches the next 3,000-step pickup-only SAC
diagnostic with frozen ACT, seed 1400, selection seeds 1401/1402/1403 and
recorded evaluations at roughly 1,000-step episode boundaries. It starts
with a zero deterministic residual actor: copying old normalized outputs
into the wider physical range would triple its arm corrections. Fresh
critics, optimizers and replay accompany the new reward/action semantics.
ACT-only and the initial zero-correction actor are evaluated, and the best
checkpoint is retained, including the initial actor if learning regresses.

The opt-in `--capture-alignment` flag requires `--pickup-only`, capture
task IDs and zero initialization. Arm bounds are ±6°, jaw bounds remain
±3°. A filter applies alpha=.25 and limits correction changes per control
step to .4° for arms and .6° for the jaw. The shoulder gate sets the desired
correction; applied corrections fade smoothly as it drops. Only the residual
is smoothed, not the underlying ACT motion. Existing runs keep their old
unfiltered behavior. The actor already observes its previous applied
residual, allowing it to account for this filter; no new oracle features
are added.

The existing pickup potential gains one centred-grasp Gaussian, with world
XYZ scales (8,8,5) mm and zero quality for a tipped victim. Its target is
the existing piece-relative grasp point (17 mm above a pawn bottom).
Potential differences retain gamma=.99 and correct terminal/truncation
handling. The reward also charges commanded jaw closure while the tool
is within 40 mm of the victim and before sustained victim lift: a fully
misaligned 12.5° closure costs 2, scaled by shoulder gate and (1-quality).
Opening/holding commands, distant motion and post-lift motion incur no
closing cost. This uses commanded physical jaw progress to avoid treating
contact-induced finger chatter as an actor closing decision.

This run has no IK assistance or inserted pauses: the learner must control
its approach and closure through its bounded residual. Success still
requires an actual upright victim lift of at least 20 mm and near-tool
holding for ten frames, with capture-order and other-piece checks active.
The 240-step pickup limit remains. V9 checkpoints require full-capture
evaluation before they can be used as full-move policies. Original USD,
calibration, grasp reference and all previous checkpoints are preserved.

Logs/videos go to `outputs/capture-alignment-rl-v9`; the shell wrapper runs
the integrity audit on completion. That audit now checks the stateful
filter's outputs and rate bounds at each episode boundary, in addition to
reward decomposition (including closing cost), frame/time continuity,
physical joint limits, checkpoint selection and unchanged ACT weights.

V9 finished with ACT and the initial zero-correction actor at 0/3 on the
selection seeds. Both intermediate actors (steps 1131 and 2031) reached
3/3 upright pickups, while the final actor regressed to 0/3. The best-file
tie rule retained the 1131 actor; the 2031 actor had closer vertical alignment
on these three cases. All integrity checks passed. These selection trials
do not establish fresh-seed reliability or complete capture performance.

`scripts/legacy/run_capture_heldout.sh` compares the frozen intermediate 1131 and 2031
actors on fresh seeds 1501–1505 with two counterbalanced repeats: 20 recorded
pickup episodes total. No training, IK assistance or inserted pauses are
used. The evaluation restores the source pickup objective, ±6° arm/±3° jaw
bounds, filter (alpha=.25; .4° arm/.6° jaw change per step), shaping and
ACT window from the training manifest. Actor and ACT file hashes must remain
unchanged. Every episode is checked for reward/filter consistency, limits,
frame/time continuity and matching paired reset options. Videos go to
`outputs/capture-heldout-v9/videos.html` as episodes finish. These are repeated
trials on the same board/move, with the existing ±.5mm attacker reset jitter;
they do not introduce varied board states or new victim square positions.
### Supervised capture pickup experiment v10

`scripts/legacy/run_capture_supervised.sh` prepares supervision from six successful ordinary
ACT-plus-residual pickup recordings made by v9 checkpoint 2031. The paused v8
geometry probe is evidence for the grasp reference, but its inserted alignment
and closing pauses are not actions the live residual can reproduce.

The offline exporter reads USD joint frames without modifying the stage file.
Across 948 source frames, the NumPy forward kinematics reproduced the logged
grasp point within 2.7e-16 m. For 236 upright, not-yet-lifted frames within 25 mm
of the victim, arm labels move halfway toward a centred target inside the
one-step action/filter bounds. Their mean predicted one-step improvement is
about 1 mm. This is an offline geometry estimate, not a physical success claim.
The demonstrated jaw closure and subsequent lift are retained. There is no
online IK, object position input, or pause added to the actor.

Images after reset come from the existing H264 recordings, so compression is a
limitation of this initial supervision set. Camera features use the frozen ACT
backbone; proprioception, proposed ACT action, previous applied residual, chunk
phase, gate, time, and move features match live inference. Seeds 1401, 1402 and
1502 form the training split; complete episodes at 1403 and 1503 form the
validation split. All five source seeds are retired from reliability testing.

The actor starts from checkpoint 2031, receives up to 1,200 supervised updates,
and retains the best validation fit, including the untouched initial actor as
a candidate. If supervision does not improve validation fit, the pipeline stops.
The next SAC learner copies only that actor into fresh critics, optimizers and
replay, then runs 6,000 simulator steps with an additional weighted supervised
actor loss after every SAC update. Existing checkpoints and the ACT weights
remain immutable. The physical correction bounds stay ±6° for the arm and ±3°
for the jaw, with alpha 0.25 and limits of 0.4°/0.6° per control step.

RL seed 1600 and checkpoint-selection seeds 1601–1603 are separate from the
demonstration seeds. Selection includes the supervised initial actor. After
the training integrity audit, the pipeline compares that initial actor with
the selected actor on seeds 1701–1705, twice each, with recorded videos. This
experiment still measures victim pickup and stable upright hold, not complete
capture transport or attacker placement. Validation fit alone is not evidence
of improved pickup reliability.

Outputs: `outputs/capture-demo-v10`, `outputs/capture-supervised-rl-v10`, and
`outputs/capture-supervised-heldout-v10`. The detached pipeline log and exit
status are `outputs/capture-supervised-v10.log` and `.status`. Budget about
45–60 minutes after launch at the observed v9 simulator throughput, allowing
extra time if reset/rendering or supervised updates slow this run.

### Close-before-lift timing diagnostic v11

The user's review of all twenty v10 held-out videos matched the recorded
success/failure outcomes exactly. In eleven of the thirteen failures, commanded
tool lift begins before the closing command reaches 1° or less, with closure
arriving 0.3–0.8 simulated seconds later. Most approach minima coincide with the
first command after the 100-action ACT chunk refresh (step 101). The jaw is still
commanded open at roughly 9–12° during the start of these lifts. This is a
command-order problem as well as a contact-timing problem; the existing ±3° jaw
residual cannot force this open command into a closed grasp. Some approaches
also remain 10–40 mm too high, which a timing change alone cannot repair.

`diagnose_capture_timing.py` writes a numbered command/encoder audit and plot,
then prepares paired recorded-command tests for videos 2, 4, 9, 13 and 20. They
include a successful control, the high grasp noted in video 4, and three near
approaches that lift early. The control replays the original executed commands.
The timing variant holds the lowest recorded open-jaw approach pose, plays the
subsequent recorded finger closing commands, waits at least twelve frames and
five consecutive encoder-settled frames, then resumes the same arm path while
keeping the jaw closed until its original closing phase catches up.

The approach landmark uses only commanded joint angles and robot forward
kinematics; closing/settling feedback uses arm and both-finger angles. Piece
poses remain scoring-only. Finger settling is not treated as proof that a piece
was picked up. A bounded 45-frame settling timeout prevents indefinite waits.
Grasp coordinates, calibration, USD geometry, ACT and saved SAC weights are
unchanged. Every arm command remains an exact recorded pose, with deliberate
separation of arm and jaw timing. This is a diagnostic replay controller,
not a learned policy or a fresh-seed reliability evaluation.

`scripts/legacy/run_capture_timing_probe.sh` runs ten episodes with alternating pair order.
Artifacts and numbered videos are in `outputs/capture-timing-probe-v11`; log
and exit status are `outputs/capture-timing-probe-v11.log` and `.status`.
There is no new RL training in this experiment. A positive result would justify
adding explicit close-before-lift timing support or training ACT on corrected
sequences; more bounded residual training alone does not provide that timing
capability.

### Live timing and adaptive approach v12–v13

The v11 replay intervention recovered three early-lift failures, but the live
v12 comparison did not show an overall improvement: the same frozen residual
actor achieved 8/10 pickups without timing control and 7/10 with it. The
remaining intervention failures froze an approach 10–20 mm too high. These
are diagnostic pickups on one capture position, not complete capture outcomes.

`CloseBeforeLiftACT` now has an opt-in adaptive approach stage. It pauses ACT's
arm-command cursor while the existing bounded, smoothed residual can adjust
the arm for at least 18 frames with the jaw open at 10 degrees. Once encoder
tracking and arm targets settle, it locks the combined arm target, consumes
the cached jaw closing commands, waits for both fingers to settle, then
resumes the original ACT arm sequence. The controller uses cached predictions,
robot FK and encoders; piece poses remain reward/scoring inputs only. It is
a required deployment component for a checkpoint trained with it.

`scripts/legacy/run_capture_adaptive.sh` runs 6,000 SAC steps from the frozen v10 actor with
demonstration anchoring reduced from 0.5 to 0.1. Training seed 1900, selection
seeds 1901–1903 and fresh comparison seeds 2001–2005 are separate. The latter
are repeated twice for each actor. Logs and status are
`outputs/capture-adaptive-v13.log` and `.status`. This still diagnoses victim
pickup before complete-move training. Offline timing audits verify residual
construction, native ACT cursor continuity, arm locking through closure and
minimum settle durations.

The working reliability target is at least 90% complete moves on fresh repeated
trials, plus an ordinary-move regression check. It is a bring-up threshold for
the tested opening curriculum, not a general chess or sim-to-real guarantee.
Selection results and pickup-only success cannot satisfy that target.

### Successful episode collection

Complete-move evaluation can enable `--collect-episodes`; training selection
evaluations can enable `--collect-evaluations`. Each lossless bundle saves the
before-action RGB camera pair, before-action legacy joint state and the target
actually executed after residual corrections, timing and physical clipping.
The legacy calibration remains unchanged so existing ACT normalization retains
its meaning. Provenance identifies ACT weights, the residual actor state,
calibration, timing configuration and deterministic execution mode.

Only complete moves with stable correct placement, home return and no protected
piece disturbance are accepted. Captures additionally require an upright victim
released stably on the robot's RHS tabletop; toppled captures remain valid chess
moves but are excluded from ACT training. Pickup diagnostics, interrupted runs
and failed moves remain explicitly rejected. `verify_collected_episodes.py`
independently recomputes the move outcome from physics traces, verifies every
camera/state/action frame and produces a hashed accepted episode index. No data
is uploaded automatically, and a lossless bundle is not yet a LeRobot dataset.

`export_act_episodes.py` converts that audited index into a new local LeRobot
dataset with `observation.images.laptop`, `observation.images.phone`,
`observation.state` and `action`. It rechecks the indexed hashes and physics
outcomes before writing, retains six legacy joint units, and uses lossless PNG
images. `bash run_dataset_python.sh` runs this optional export in a separate
dataset environment; the active RL environment is unchanged. The exporter
never calls a Hugging Face upload method. A synthetic temporary episode is
used to verify writer/reader round trips; it is not a collected robot episode.

`collect_opening_episodes.py` runs one frozen complete-move teacher across its
trained task IDs with exploration disabled. Each attempted episode is recorded,
but only independently audited successes enter `accepted_episodes.json`.
Use a new output directory and seed range. Its checkpoint requires the matching
training manifest because bounds, residual filtering, ACT action window and
optional timing controller are part of deployment. Collection from pickup-only
checkpoints is explicitly rejected. Example commands, after a complete-move
policy has passed its separate reliability test:

```bash
./run_rl_python.sh collect_opening_episodes.py --pilot outputs/COMPLETE_MOVE_PILOT \
  --output outputs/verified-act-episodes --episodes 40 --seed-start 3001 --record-video
bash run_dataset_python.sh export_act_episodes.py \
  --index outputs/verified-act-episodes/accepted_episodes.json \
  --output outputs/act-sim-dataset
```

### Successful replay and lower exploration v14

The v13 adaptive controller allowed the initial actor to achieve 3/3 selection
pickups. SAC then regressed: every trained candidate scored 0/3. All 34 training
attempts failed, with 27 arm-settling timeouts. The training/reward/weights audit
passed, so the earlier actor was retained. In fresh repeated trials, the two
identical retained actor copies scored 9/10 and 4/10. That difference is rollout
variation, not a learned-policy improvement, and pickup reliability is still
insufficient. Some failed closures were 8–27 mm above the requested grasp height.

`scripts/legacy/run_capture_replay.sh` first runs the deterministic retained teacher on seeds
2104–2113. It stores the exact learner feature vectors, raw bounded residuals,
rewards, next observations and terminal flags. Only successful, undisturbed
episodes can seed SAC's replay buffer. The largest successful seed is held out
from both replay and supervised training as retention validation. All attempted
teacher seeds are retired from later reliability testing. A small supervised
loss retains the same teacher's actual actions; these are residual-training
transitions, not complete ACT episodes.

Only SDE exploration log standard deviation resets from approximately -1 to -3,
reducing initial noise by about sevenfold while preserving the deterministic
actor mean. Mean weights still copy the selected source exactly; the integrity
audit explicitly checks that exception. This change addresses a training run
that had no successful transitions rather than simply adding more blind steps.

The learner runs 6,000 new environment steps with BC weight 0.5, training seed
2100 and selection seeds 2101–2103. Successful replay transitions are counted
separately from those environment steps. Fresh seeds 2201–2205, twice per actor,
follow the training audit. Bounds, filter, grasp geometry, calibration and ACT
weights are unchanged. This experiment still evaluates pickup, not completed
captures. Outputs are `outputs/capture-replay-rl-v14`,
`outputs/capture-replay-heldout-v14`, and `outputs/capture-replay-v14.log`/`.status`.

### Latched approach and complete captures v15

The v14 teacher supplied seven successful pickups from ten training attempts,
with 1,151 exact successful transitions in replay after validation exclusion.
Nevertheless all 31 online training attempts failed: 30 arm-settling timeouts
and one pickup timeout. The selected actor remained the starting actor.
Lower noise and successful replay alone did not resolve the execution barrier.

The opt-in `--latch-alignment` protocol now allows 18 frames of open-jaw arm
correction, then freezes that combined target for encoder settling and closure.
Continuing exploratory requests cannot continually reset the settling target.
The maximum timeout, open-finger requirement and encoder tolerance remain.
The default disabled protocol preserves older deployment manifests and behaviour.
A unit test exercises continuing one-degree request changes through adaptation,
settling, closure and resumption; the offline timing audit checks the lock too.

`scripts/legacy/run_capture_complete.sh` waits for the v14 client to finish before training v15.
Its objective is `legal_move_v5`, with the full 800-step capture budget and up to
two guarded closures. It warm-starts the retained pickup actor, keeps verified
teacher actions through a 0.5 supervision weight, and starts fresh critics and
replay. Pickup-only terminal rewards are never reused for complete-move training.
Six thousand steps use training seed 2400 and selection seeds 2401–2403; a fresh
complete-capture comparison uses 2501–2505 twice per actor. Selection and fresh
tests record videos. The ordinary-move expert remains separate from this run.
This experiment has not yet passed the complete-move reliability gate.

`scripts/legacy/run_ordinary_reliability.sh` queues behind that complete-capture pipeline and
tests the retained v6 ordinary-move expert on e2e4, d2d4 and g1f3. Ten new seeds
(2601–2610), repeated twice, provide twenty complete trials per task. The
`--only-policy learned` evaluation mode tests one frozen teacher without
spending simulator time on a duplicate paired comparison. It preserves all
requested repeats and integrity checks; paired mode remains the default.
The combined reliability verifier can audit distinct capture/ordinary task
profiles together, recording each profile's checkpoint, bounds and controller.

### Tabletop release guard correction v16

The first complete-capture evaluation of v15 exposed a controller error. On
seed 2403 the victim reached the RHS tabletop, but a guard at native ACT index
200 held the gripper around `(0.1324, -0.1869, 0.0166)` metres and consumed a
later closing sequence. That location is off the board: the guard interpreted
release followed by closure as a new pickup and returned the victim toward the
board. The learner was interrupted gracefully; `stop_reason.json` preserves
why this incomplete experiment stopped. No v15 training success is claimed.

`--board-grasps-only` restricts forecast hold and closing points to bounds
computed from the static calibrated 64 square centres. It does not inspect
piece poses or use a simulator success flag. Tabletop opening, retreat and
empty-jaw closure pass through unchanged. Tests cover an entirely off-board
sequence and a sequence crossing from the board into the release area, and
the timing audit verifies guarded forecast points against the saved bounds.
Disabled settings serialize exactly as before for older manifests.

`scripts/legacy/run_capture_board_guard.sh` runs v16 with latched alignment, board-only guards
and up to four guarded pickups so a retry does not consume the guard needed
for the attacker. It retains the verified v14 teacher's residual actions,
starts fresh complete-move critics/replay, and trains 6,000 steps on seed 2700.
Selection seeds are 2701–2703 and fresh comparison seeds 2801–2805 (twice per
actor). The ordinary v6 teacher is queued afterward on 2901–2910, twice per
task, through `scripts/legacy/run_ordinary_reliability.sh`'s explicit output/status/seed arguments.
These are pending experiments, not a certified complete-move policy.

### Resume on 2026-10-25: complete-move replay v17

V16 was stopped at the user's request before training completed. Its saved
initial evaluations showed **2/3 complete captures with zero residual** and
**0/3 with the inherited pickup-trained actor**. The board-only timing fix
therefore demonstrated complete captures, but the pickup actor was a poor
starting point for the longer task. These small selection trials do not
establish reliability or a learned improvement.

`scripts/legacy/run_complete_replay.sh` now starts from a fresh actor whose deterministic
residual is exactly zero. It retains board-only guarded grasps and latched
settling, enables intermediate-home retries within the 800-step capture
budget, and sets initial exploration log standard deviation to -4. No
pickup-trained actor weights or pickup terminal rewards enter this run.

The first ten teacher attempts use seeds 3104–3113 and full capture rewards.
At least two successful seeds are required before SAC training proceeds.
The largest successful seed is excluded from replay and supervision. The
exact frozen teacher checkpoint is saved independently and hashed; its
predictions and complete-move physics outcomes are recomputed by the audit.
Generated demonstration metadata describes the current environment, including
the home-retry and timing settings, rather than inheriting an old pickup label.

Training uses 4,000 steps, seed 3100, selection seeds 3101–3103, BC weight 0.5,
fresh critics/replay, and successful complete-move replay from the teacher.
Selection retains the earlier checkpoint on ties, so a selected zero-correction
teacher is explicitly possible and must not be described as an RL improvement.
Twenty fresh capture trials use 3201–3210 twice each. If fewer than 18 succeed,
the pipeline stops for analysis without collecting ACT data.

If the capture threshold passes, the retained ordinary-move v6 expert is tested
twenty times per task on 3301–3310. The combined verifier independently audits
physics and requires at least 90% complete-move success on every tested task.
Only after that gate passes does collection attempt twenty complete captures
and twenty ordinary moves on separate new seeds (3401–3420 and 3501–3520).
Only independently verified, upright complete episodes enter two local LeRobot
datasets. Failed and toppled episodes remain labelled and excluded. Nothing is
uploaded and the existing ACT checkpoint is not retrained automatically.

Log/status: `outputs/complete-replay-v17.log` and `.status`. Training, capture
reliability and ordinary reliability outputs are
`outputs/capture-complete-replay-rl-v17`,
`outputs/capture-complete-reliability-v17`, and
`outputs/opening-ordinary-reliability-v17`. Dataset outputs, if the gates pass,
are `outputs/act-capture-dataset-v17` and `outputs/act-ordinary-dataset-v17`.

### Approved grasp geometry and teacher collection v18–v23

The user approved the v18 open-gripper pose in both the simulated cameras and
the CAD views. Its target is the midpoint of opposing material points, each
7 mm behind the front of its finger, at the requested piece-specific height.
The midpoint follows both measured jaw angles; it is not a fixed point on one
finger. The saved pose was within 0.015 mm of the pawn's 17 mm grasp target.
Sources, measurements and camera views are in `outputs/grasp-pose-review-v18`.

The controlled v19 probe closed at that pose, lifted the pawn 34.9 mm and held
it for two seconds with 0.010 mm maximum relative slip. The bounded geometric
teacher then succeeded in 3/3 live ACT pickups in `outputs/geometric-pickup-v20`.
Its 613 transitions passed action, feature, timing and frame audits. These
are pickup results, not complete-capture reliability or RL improvement.

Full v21 captures exposed a separate timing error. All three victims reached
the RHS tabletop upright, but later ACT opening/retreat/empty-jaw closure over
the board could trigger another guarded pickup. Two attackers reached within
one millimetre of their target before being picked up again. No v21 episode
completed successfully; all 2,157 transitions and their full-move rewards were
independently audited. Source snapshots and videos are retained with the run.

The opt-in release-aware controller uses only calibrated robot FK, commanded
jaw angles and measured opposing fingers. A grasp requires both fingers to
remain obstructed against a closing command; subsequent lift and opening
identify a release. Forecast placement followed by empty-jaw closure is not
paused as another pickup. Capture profiles require an off-board release before
an on-board release completes the sequence, allowing retries after a victim is
incorrectly dropped on the board. It does not use simulator piece positions.
V22 is the initial release-history probe; v23 will test the capture-aware
variant. V22 completed 1/3 captures with an upright victim; its other two trials
released the victim on the board. All 1,696 transitions, teacher actions and
full-move rewards passed the independent audit. V23 then completed **5/6 full
captures**, all upright. Its 3,451 transitions passed the same independent
action, observable-feature, timing, frame, outcome and reward audits.

`collect_geometric_grasps.py` stores actual live teacher corrections and exact
learner transitions, with 32 open-jaw adaptation frames, latched closure and
the same six-degree arm/three-degree jaw residual limits. Only the teacher may
read simulator piece targets. `verify_geometric_teacher.py` recomputes every
teacher action, observable feature tail, full-move outcome and full-move reward.
Actor observations remain 4,122 camera/proprioception/command features.

`prepare_geometric_demonstrations.py` accepts only complete, upright, centred
teacher episodes, reserving an entire successful seed for validation. Pickup
diagnostics, failed moves and toppled captures cannot enter this dataset.
`geometric_replay.py` verifies matching controller, task, calibration, bounds,
filter and reward semantics before seeding SAC replay with these real simulator
transitions. Validation is excluded from replay and supervised actor updates.
The supervised actor must first demonstrate physical success on fresh trials;
validation fit alone is not a reliability result. The existing 20-trial/90%
full-move gate still applies before ACT episode gathering.

`run_geometric_pipeline.py --output outputs/geometric-pipeline-v24` continues
sequentially after v23 releases the simulator. It can collect up to eighteen
fresh teacher attempts across v23/v23b/v23c, requiring four centred complete
episodes before supervised initialization. One whole seed is validation.
The v24 actor is physically screened on five additional seeds before SAC:
at least two complete captures and four stable RHS victim removals are required
to justify training. Uprightness is reported separately, rewarded and required
for ACT exports; it is not a hard capture failure. This screening threshold is
not a reliability threshold.

If screening passes, v25 runs 4,000 SAC steps with successful complete-move
replay, BC retention, low exploration, matching 32/90-frame timing and fresh
selection seeds. Twenty fresh capture tests must then reach 90%; ordinary
regression tests independently check twenty trials per task with the retained
v6 expert. The combined physics verifier must pass before the pipeline gathers
and locally exports accepted ACT episodes. Source ACT weights stay frozen.

Pipeline progress and failures are persisted in
`outputs/geometric-pipeline-v24/status.json` and
`outputs/geometric-pipeline-v24.log`. A failed gate records `needs_analysis`
and prevents later training/collection stages. No run is described as reliable
merely because supervised validation improves or a training process finishes.

V24 supervision used four successful complete episodes and reserved seed 3806
as validation. Validation MSE fell from 0.05213 to 0.007673 (selected update
1,950 of 2,000). The first fresh physical screening trial, seed 3901, completed
successfully with an upright RHS victim using the learned actor only, without
privileged teacher corrections. The completed screen achieved 2/5 full captures
and 4/5 stable RHS removals, with only 1/5 victims remaining upright. The first
screening gate mistakenly required four upright removals and stopped the
pipeline before RL. That was stricter than the user's explicit preference that
toppling should reduce quality without becoming a capture failure.

Screening criterion v2 retains the two-completion/four-removal training gate
and treats uprightness as the existing reward preference. The original failed
screen is preserved as `screening_upright_gate_v1.json`; recorded outcomes and
the actor are unchanged. `--resume-rl` rechecks the checkpoint hash, transition
integrity and independently recomputes all five outcomes before starting SAC.
The final 90%/20-trial reliability gate and upright ACT-export filter remain.
No reliability or RL-improvement claim follows from these five screening trials.
The active resumed pipeline PID is recorded in
`outputs/geometric-pipeline-v24.pid`; its log preserves the resolved scheduling
error before screening started. Single-policy screening permits one repeat;
counterbalanced paired comparisons still require at least two.

### V25 result: RL regression; retained actor at 12/20

The 4,000-step SAC run completed with 3,744 gradient and demonstration updates.
Its initial 2,134 replay transitions excluded validation seed 3806. The saved
weights, bounds, filtering, timing and ACT-weight checks passed. On the two
selection seeds, the supervised starting actor completed 1/2 captures; both
the intermediate and final RL actors completed 0/2. Selection therefore kept
`before`. The evaluated best checkpoint is the pre-RL supervised actor, not an
RL improvement.

That retained actor completed **12/20 fresh full captures (60%)**, and all
twelve successes left the victim upright. Six failures lifted the victim but
released it back onto the board before moving the attacker off its original
square; these are genuine ordering failures, not minor same-square nudges.
One further trial timed out and one hit the arm-settling timeout. Independent
physics rescoring of all twenty traces passed (`outcome_audit.json`). The 90%
gate failed, so the pipeline stopped before ordinary regression tests or ACT
episode collection/export. Source ACT weights remain unchanged.

The offline actor-retention audit found weighted validation action MSE rose
from 0.007673 before RL to 0.019483 after RL, despite the BC retention loss.
This supports tightening actor retention before another training run; simply
extending the same optimization is not justified. Victim transfer to the RHS
is the principal remaining complete-move failure.

Four seeds changed outcome between repeats. Reset joint positions were exactly
equal; maximum piece-centre differences were below 0.0011 mm. Initial camera
images differed by roughly 0.6–2.5 intensity levels per channel on average,
and initial ACT commands differed too. `reset_repeatability_review.json`
records these observations; it does not establish rendering as the cause.
Recordings are indexed at `outputs/geometric-capture-reliability-v25/videos.html`.

### Capture supervisor and qualification v26–v28 (2026-10-26)

V25's dominant error was a victim being lifted and released onto a different
board square. The next profile retains the v24 learned residual but adds an
explicit capture supervisor. This is a hybrid execution policy, not evidence
that RL learned the complete victim transfer.

`capture_transport.py` consumes the chess command, static board/RHS calibration,
measured arm/finger joints and encoder-based grasp history. It does not read
piece poses, task-success flags or the reward's capture phase. After confirmed
contact and lift at the commanded victim square, five-axis IK preserves the
payload's initial vertical direction, raises it above the board, transfers it
to the RHS tabletop, lowers it near the surface and slowly opens the gripper.
The tabletop grasp height is inferred from the closed encoder pose over the
calibrated board surface. Motion uses bounded joint increments, authored
limits and measured settling feedback. Every physical step is recorded.

The v26 probe removed the victim upright and placed the attacker correctly,
but timed out before returning home. V28 corrects the integration: residual
history and the actor's original 800-step progress input pause during transport;
the physical capture budget is 1,600 steps. The supervisor returns empty to the
measured home pose, clears stale ACT predictions and replans from fresh images.
Native command indices remain monotonic, with an explicit cache origin for the
new chunk. Encoder release history survives this replan. Offline audits rebuild
the route, verify every supervised command, and check the paused step counter.

`outputs/capture-transport-probe-v28` completed **3/3 full upright captures**
(seeds 5201–5203; 1,066, 1,037 and 1,075 physical steps). They passed frame,
command, timing and checkpoint-integrity checks. These are bring-up results,
not the 20-trial reliability gate. The profile is limited to the tested blue
first-capture opening and an unoccupied RHS area; other capture histories and
red control remain unqualified.

`run_hybrid_readiness.py --output outputs/hybrid-readiness-v28` then runs twenty
fresh complete captures (5301–5310, twice). If needed, a new SAC run collects
replay under this exact supervisor; old-controller rewards/transitions are
not reused. Critics warm up for 1,000 updates, actor learning rate is reduced
tenfold, BC weight is 5, and actor updates exceeding a training-only retention
budget are rolled back together with their optimizer state. Validation never
enters that budget. The nonzero minimum budget permits learning when the frozen
teacher exactly matches its own labels. Targeted tests cover rollback, warmup,
validation exclusion, actor-clock/history preservation, cache resync, upright
IK routes, and source-calibration-only controller construction.

The selected capture candidate must still pass >=90% on twenty fresh trials.
The retained v6 ordinary expert then runs twenty trials each for e2e4, d2d4 and
g1f3 on 5801–5810. Only a combined independent physics-verifier pass allows two
local proof-episode attempts and an ACT dataset export. Uprightness remains a
quality preference during control and a strict filter on exported episodes.
The pipeline stops at `ready_to_collect`, with its verified report, proof
dataset and exact larger-collection command. ACT weights are not retrained;
nothing is uploaded. Progress: `outputs/hybrid-readiness-v28/status.json` and
`outputs/hybrid-readiness-v28.log`.

### Commanded pickup-square guard v29

V28 qualification completed 17/20 moves, with **20/20 upright victim removals**.
All three failures were attacker placement failures. Trace inspection showed
the timing controller scheduling another pickup at the destination while the
attacker was already being placed. The scheduled RL fallback was stopped
before training began; the completed evaluation and all its seeds remain
preserved under `outputs/hybrid-readiness-v28`.

After victim removal, v29 limits forecast hold and closing points for guarded
pickups to within 18 mm of the commanded attacker's original square. This
uses the public chess command and calibrated square centres, not a piece pose
or simulator phase. Destination release followed by closure passes through
without a new pickup guard. Rejected plans are logged and independently
checked; immediate rejected guards also restore the intervention budget.

The corrected profile completed 3/3 fresh probe captures (6101–6103), all upright,
in 901, 901 and 859 physical steps. A saved camera sequence review is at
`outputs/capture-transport-probe-v29/repeat_0/scandinavian_capture/learned_6101/sequence_review.jpg`.
The first five fresh qualification trials also succeeded; the complete gate
is still pending and must not be inferred from these partial results.

The active pipeline is `outputs/hybrid-readiness-v29.log`/`status.json`, with
profile `outputs/capture-transport-profile-v29`. Its disjoint seed base is 7000:
7001–7010 for initial capture qualification, 7300 for optional conservative RL,
7401–7410 for post-training qualification if needed, 7501–7510 for ordinary
regressions, and 7601–7602 for the upright writer/export proof. It stops at
`ready_to_collect` only after the unchanged combined reliability gate and the
local dataset round trip pass. No ACT retraining or remote upload is performed.

V29 capture qualification subsequently passed **20/20 complete captures** on
7001–7010 repeated twice, all with upright victims. The complete controller
traces passed independent command, frame, timing and physics-outcome checks.
No additional RL training was needed; the gain came from the explicit capture
supervisor and command-based pickup restriction. Ordinary-move regression and
the lossless ACT-export proof remain pending before overall readiness.

The v29 ordinary regressions then passed **20/20 each** for e2e4, d2d4 and g1f3.
The combined independent verifier passed all four task profiles: **80/80**
complete moves. `outputs/hybrid-readiness-v29/policy_readiness.json` records
the exact model/controller profiles and the tested scope. The pipeline is now
performing its two-episode lossless capture/export proof.

### Ready to collect: v29 completed

The combined verifier passed **80/80 complete moves**, twenty per tested task.
The first writer proof exposed an extra collector requirement on `transported`,
which is a transient reward event (within 5 mm of the destination while still
lifted), not the complete-move success condition. Both recordings were genuine
upright successes. Collector acceptance v2 uses the verified complete-move,
lift, stable placement, release, home, disturbance and upright-capture checks;
the physical task rules and all qualification thresholds are unchanged. Tests
cover valid complete moves without the bonus and continued rejection of failed,
unlifted, incomplete, disturbed or non-upright episodes. Old rejected bundles
are preserved unchanged.

`finish_hybrid_readiness.py` rechecked all eighty qualification traces, then
recorded two fresh successful, upright captures (7901–7902). Both were accepted:
**1,793 lossless frames at 30 Hz**. Export succeeded through the real LeRobot
writer. `verify_act_dataset.py` read six sampled frames back, comparing exact
RGB pixels, legacy states, executed actions, frame/episode indices and timestamps
against the independently audited source bundles.

Final status: **`ready_to_collect`** in `outputs/hybrid-readiness-v29/status.json`.
The proof dataset is `outputs/hybrid-readiness-v29/sample_act_dataset`; its
`roundtrip_integrity.json` passed. `READY_TO_COLLECT.md` contains the exact
commands for forty capture attempts and sixty balanced ordinary-move attempts.
No larger collection, ACT retraining, hardware execution or remote upload has
been started. The existing ACT weights remain unchanged. Readiness covers the
tested blue opening states only, using the learned residual plus the explicit
capture supervisor; no new RL improvement is claimed.
