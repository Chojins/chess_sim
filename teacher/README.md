# State teacher and RGB distillation experiments

See [PROGRESS.md](PROGRESS.md) for current results, negative experiments and
the qualification gate. Launching a run does not establish collection readiness.

The current **RGB-only ACT pilot** passed 20/20 held-out simulated starts with
**50 executed actions per chunk**, no robot contacts and at least 7.919 mm
sampled camera clearance. Use the exact checkpoint and runtime profile in
[`v116-rgb-act-student.json`](../docs/baselines/v116-rgb-act-student.json).
Its source checkpoint config still says ten actions; `GoalACT` and the student
training/review CLIs now default to fifty, or pass `--n-action-steps 50`
explicitly. Scope remains the four initial-board blue pawn moves below.
Captures, other positions, red control and real hardware remain unqualified.

The current collision-aware four-goal teacher is
[`v109-collision-aware-neural-teacher.json`](../docs/baselines/v109-collision-aware-neural-teacher.json):
the frozen v108 final trained actor passed 20/20 fresh starts and 4/4 matched
camera reviews, with no robot contacts and at least 7.66 mm sampled camera
clearance. Scope: initial-board blue e2e4, e2e3, d2d4 and c2c4 only. Pilot ACT
collection prerequisites pass; each newly collected episode still needs its
independent acceptance audit. No ACT retraining or hardware transfer is claimed.

The current local RGB pilot is queued through
`outputs/clearance-rgb-pipeline-v112.log`: twelve new episodes → three-view
LeRobot export → a separate original-ACT copy fine-tuned with equal-weight
sim/real replay → checkpoint reload verification → four RGB-only simulated
tests. See the stage JSON next to that log for the active phase. One whole sim
episode per move and four whole real episodes are held out. Original ACT
weights remain immutable; a fitted checkpoint still needs closed-loop checks.

The historical standalone e2–e4 teacher has a pinned [qualified RL checkpoint](../docs/baselines/v65-neural-teacher.json):
20/20 fresh home starts, plus matched-camera review. This qualification is for
the blue robot's ordinary e2–e4 with self-collision disabled. It establishes no
camera/servo clearance. The original ACT is unchanged.
The separately versioned four-move curriculum and RGB student pilot below are
in development; their results do not extend that pinned qualification.

## Four-move teacher and RGB ACT pilot

`ordinary_move_v1` supports explicit legal histories and requested blue pawn or
knight moves. Current experiments train the initial-board pawn goals e2e4,
e2e3, d2d4 and c2c4. Support in the task code does not establish knight or capture
reliability. `build_ordinary_bank.py` creates disjoint physical home starts;
`probe_moves.py` distinguishes frozen learned evaluations from analytic reference
runs. The original e2e4 task, calibration and qualified actors stay separate.

`ordinary-relational-bootstrap-v76` passed the first four recorded camera trials,
then 7/8 development validation starts. Fresh qualification stopped after an
e2e3 miss made its predeclared per-move threshold unreachable. It has **not**
qualified for balanced four-move ACT collection. Corrective fitting uses only
training rollouts, never the validation or qualification failures as labels.

The current collision-aware candidate is `clearance-return-arm-v99`: 8/8
development moves and 4/4 filmed moves, with no self/inter-robot contacts.
Sampled exact camera/servo CAD gaps in the filmed trials exceeded 7.41 mm;
`clearance-final-qualification-v102` separately tests twenty fresh starts under
a predeclared 5 mm gate. Check its final status. This actor was fitted with
supervision, not improved by RL. See [collision geometry and audit scope](../calibration/robot_collisions.md).
Launch corrected worlds with `--robot-self-collisions --robot-collision-geometry
decomposed`, using a new reset bank. Old self-disabled replay is incompatible.

For measured clearance rewards, use `run_dataset_python.sh` (its `python-fcl`
dependency is separate from the locked RL environment). `train_move` accepts
`--camera-clearance-reward-weight .04 --camera-clearance-buffer-mm 8` with the
existing mesh/margin flags. The 5 mm acceptance gate stays unchanged. Rebuild
physical replay with the matching `reconstruct --camera-clearance-reward-meshes`
contract before training; mismatched replay rewards are rejected. Raw reward
audits recompute CAD costs from measured joints.

`placement_corrections` makes supervised arm labels from audited training-only
state exports. `train_move --placement-corrections <pool.npz>` adds these labels
to the actor loss, without inserting pose variants into physical replay.
`--arm-head-only` preserves the shared actor, jaw and exploration parameters
while refining its five arm output rows. Saved inference remains the ordinary
neural actor, with no analytic action override. Current comparison:
`clearance-placement-rl-v108`; its launch is not a success/qualification result.

Once an actor passes `qualify_move` and matched camera review for every case,
collect at least two accepted episodes per goal, then export the separate desired
board image with `teacher.export_episodes --goal-input`. For an initial student:

```bash
bash run_dataset_python.sh -m teacher.train_student \
  --index outputs/ordinary-episodes/accepted_episodes.json \
  --output outputs/ordinary-rgb-act --updates 1000 --batch-size 1
bash run_dataset_python.sh -m teacher.review_student \
  --checkpoint outputs/ordinary-rgb-act/checkpoint_001000 \
  --teacher-run outputs/ordinary-qualified-actor \
  --output outputs/ordinary-rgb-review --url http://127.0.0.1:8082
```

The student strictly copies the original ACT model and normalization, adding a
third RGB view for the desired board. Shared vision/model weights also receive
equal-weight two-view replay from local blue real demonstrations, where the
desired board was not recorded. Only the known `main_` joint namespace is aliased;
values retain their original order and legacy units. Training chunks stay inside
one episode and mask padding; the last whole episode per goal is validation.
Both domains contribute gradients to one AdamW update. Source checkpoints and
datasets are read-only; all student checkpoints are new local copies.

Student inference receives only measured joints, highlighted fixed RGB, wrist
RGB and the requested desired-board diagram. Simulator geometry scores the
rollout, but supplies no action proposals. Absolute ACT targets pass through a
generic uniform joint slew limit; there is no grasp or capture phase controller.
Offline action loss and an optimizer smoke test do not establish closed-loop
success or hardware readiness. Follow the checkpoint named by `selection.json`
and inspect real camera rollouts before treating it as a usable student.

```bash
# One controlling client per port. Output directories must be new.
bash run_rl_python.sh -m teacher.collect_episodes \
  --qualification outputs/rl-home-qualification-v62 \
  --motion-review outputs/rl-neural-home-review-v63 \
  --output outputs/rl-act-pilot-new --seeds 113001 113002 \
  --url http://127.0.0.1:8082
bash run_dataset_python.sh -m teacher.export_episodes \
  --index outputs/rl-act-pilot-new/accepted_episodes.json \
  --output outputs/rl-act-dataset-new --repo-id local/chess-rl-pilot
```

These commands retain executed legacy joint targets and lossless before-action
camera pairs. They write local data only. Keep the qualification, frozen actor
and source recordings with the dataset; output artifacts are excluded from Git.

This directory starts the new approach while preserving the existing ACT,
residual controllers and [v29 baseline](../docs/baselines/v29.md). It currently
implements a **single-environment state teacher**, not a reliable learned
capture policy, parallel SGS training, or an RGB student.

## Implemented

- A dedicated localhost server that reuses the existing chess physics,
  modified gripper, contacts and magnets. It defaults to state-only operation;
  `--render-cameras` enables paired authored-camera recording with a guard that
  rendering leaves physics unchanged. Original USD/calibration files are never saved.
- A goal-conditioned, privileged-state Gymnasium environment. Its 424 features
  include joints, measured motion, previous commands, piece centres, finger
  opening, desired piece locations and task progress. The exact layout is saved
  in each experiment's manifest. Simulator state is intentional teacher input;
  these features cannot be sent directly to the real robot.
- A standalone policy that commands all six physical joints. There is no
  ACT action proposal, capture route, phase-dependent command override or
  intermediate home-return instruction. A generic target-speed limit applies
  to every action. It bounds target speed, not physical acceleration or jerk.
  Action contracts are versioned in each manifest: the first capture experiment
  used absolute targets; ordinary-move comparisons also use increments or arm
  increments with a learned gripper-position target.
- Existing complete-move scoring, potential-based shaping, and penalties for
  command velocity and acceleration. Square-based disturbance and capture-order
  rules remain unchanged. Upright captured-piece placement remains desirable.
- Optional BC initialization from independently audited v29 episodes. The last
  whole source episode is validation only. These recordings contain known pauses
  and the home detour; this prior is temporary, not improved demonstration data.
  Its old transitions/rewards are never imported into SAC replay because the new
  control/reward contracts differ. The manifest counts old commands that exceed
  the new speed limit. Training collects fresh transitions under the new limit.
- Separate frozen before/after evaluation, optimizer logs, checkpoints, replay
  buffer and progress/error files. Evaluation does not add samples to replay.

The server uses 240 Hz physics and 30 Hz control, matching the existing baseline.
It binds to a separate port (8081) and has no video preview. Only one controlling
client may use a server at a time. Existing Python dependencies are reused through
`run_rl_python.sh`; no new package installation is required.

## Unified move teacher and adaptive sampling

The v38 placement run completed 12,000 steps. Its selected policy passed 4/6
fresh tests: both near-target and both near-release starts, but neither
held-source carry. Both carry failures displaced another pawn off its square;
they were not merely harmless contacts. The selected actor remains in
`outputs/transport-place-rl-v38/place_best.zip` and is preserved for comparison.

The next runner, `teacher.train_move`, trains **one policy with one objective**
from every start: move the e2 pawn to e4, release it upright on the intended
square, and return the blue arm to the original camera-viewing pose for ten
stable control frames. Grasping, carrying and release do not terminate separate
policies, and no scripted handoff or intermediate home command runs during RL.
The original action contract remains six joint increments at 30 Hz, with arm
targets limited to 15 degrees/s and jaw targets to 45 degrees/s.

The 71 actor features contain actual joint/finger feedback, previous action,
pawn position/velocity/upright direction, nearby pieces, desired-square error
and desired-home error. Lift/release history is used for scoring, not supplied
as a phase flag to the actor. Exact geometric observations are still privileged
teacher inputs; RGB imitation and robustness remain future work.

The new reset set has distinct train seeds 81001–81003, checkpoint-validation
seed 82001 and final-test seed 83001. Physical starts cover home, approach,
open-gripper grasp, the qualified frozen grasp's actual endpoint, lift clearance,
mid-carry, near-release, retreat, near-home and final home settling. All starts
share the same terminal objective. Original board baselines and physically
verified lift/release history survive reset. Nearly complete starts still need
a fresh stability window; they do not receive success at reset. Each reset has
a time budget based on its remaining feasibility continuation, capped at 800
control steps. Only evaluations beginning at home measure the entire motion
from home; late-start completions must be reported separately.

`teacher.success_sampler` adapts training reset probabilities from a rolling
window of 100 Boolean training outcomes. Its SGS-style beta kernel uses mode
0.5, concentration 1 and temperature 2. Unknown configurations start at the
mode; an additional 20% uniform mixture preserves coverage in this small-data
experiment. This mixture is our adaptation, not an exact reproduction of the
paper. Validation and final test use fixed resets and never update the sampler.
Tests compare complete real-SAC replay and sampler histories against an otherwise
identical run without checkpoint evaluations. Runtime evaluation also checks
that sampler history, actor weights and learner replay remain unchanged.

The sampler is saved in `sampler.json`, including recent outcomes and sampling
probabilities. Selected actors use `move_best.zip`; the final training actor is
saved separately. Actor, critic and replay start fresh. The frozen grasp and IK
references generate physical reset states and check feasibility; they never
override the unified learner's actions or supply demonstration gradients.

This remains one CPU physics environment with SAC. It adopts intermediate
resets and success-guided selection from [the SGS paper](https://arxiv.org/html/2610.12465v1),
but does not reproduce its parallel PPO training scale. The sampler is a
testable efficiency improvement, not a guarantee that a short run learns the
whole move. The current test sets are bring-up diagnostics, not a reliability
qualification or authorization to export ACT training episodes.

```bash
# A fresh build now includes open-grasp and final-home boundaries directly.
bash run_rl_python.sh -m teacher.build_move_bank \
  --output outputs/unified-move-reset-set-new --name-prefix move_new
bash run_rl_python.sh -m teacher.train_move \
  --bank outputs/unified-move-reset-set-new/manifest.json \
  --output outputs/unified-move-feasibility-new --reference-only
bash run_rl_python.sh -m teacher.train_move \
  --bank outputs/unified-move-reset-set-new/manifest.json \
  --output outputs/unified-move-rl-new --steps 16000 \
  --reference-proof outputs/unified-move-feasibility-new
```

`add_home_boundary.py` and `add_open_grasp.py` upgrade the initially built v39
reset index while preserving its immutable physical checkpoints. They are
one-time utilities; a new build generates those families directly.

## Held-pawn transport and placement

The next experiment is `teacher.train_transport`. Its 69 privileged features
include the desired e4 displacement, actual joint/finger feedback, pawn pose and
velocity, nearby pieces, prior action and release progress. It learns all six
joint increments at 30 Hz, with the same 15 degrees/s arm and 45 degrees/s jaw
target limits. The qualified grasp actor is neither trained nor overwritten.

`outputs/transport-reset-set-v38/manifest.json` contains 21 physical checkpoints:
three families per seed, with train 72001–72003, validation 73001–73002 and final
test 74001–74002. `held_source` comes from the frozen learned grasp's actual
endpoint. `near_target` and `near_release` are held states reached by replaying
the audited complete e2–e4 reference. These starts are not created by moving or
attaching the pawn. All 32 pieces, both robots, velocities and servo targets are
restored, with the original board baseline and verified prior lift history.
As before, PhysX contact caches rebuild after restoration.

Placement requires opening over e4 no more than 8 mm above the original pawn
centre height, followed by an upright pawn within 5 mm of e4, near board height,
and at least 35 mm tool clearance for ten stable frames. Opening intent is
recorded from the physical frame before the opening step, so subsequent magnetic centring cannot turn a
wrong-square release into success. Neighbours may shift within their own square;
leaving it or tipping beyond 30 degrees remains a failure. A placement-stage
success does not include approach from home or final home return.

Three audited feasibility probes in `outputs/transport-feasibility-v38c/` passed,
with final errors of 0.30–0.53 mm and no protected-piece disturbance. These use
an explicit IK reference **only to check physical feasibility**. The reference
is never called during RL action selection, and its success is not a learned
policy result. The first attempted probe (`v38`) overconstrained a slightly
tilted held pawn to be exactly vertical; the revised probe prioritizes position
and lets actual board contact establish final upright placement.

The learner uses fresh SAC actor, critics and replay, fixed entropy 0.002,
actor/critic LR 0.0001/0.0003 and 512 critic-only warmup updates. Episode-boundary
checkpoints are named `place_*.zip`; the best validation actor is copied to
`place_best.zip`, with ties retaining the earlier checkpoint. Each validation
trace is re-scored from raw physics, joint commands and rewards before selection.
The separate final-test states are evaluated only after checkpoint selection.

The corrected integration run `outputs/transport-rl-smoke-v38b/` completed 640
steps and 128 gradient updates. After selection it passed both near-target and
near-release test starts, but not the held-source carry (2/3 stage successes).
This smoke partition uses only seeds 72001–72003; the main run's final-test seeds
74001–74002 remain unused by this placement actor. The main 12,000-step run is
`outputs/transport-place-rl-v38/`, launched detached with its PID in
`outputs/transport-place-rl-v38.pid`. It starts fresh rather than reusing smoke
weights or replay. Neither result is a complete-move qualification.

```bash
bash run_rl_python.sh -m teacher.build_transport_bank \
  --output outputs/transport-reset-set-v38
bash run_rl_python.sh -m teacher.train_transport \
  --bank outputs/transport-reset-set-v38/manifest.json \
  --output outputs/transport-place-rl-v38 \
  --reference-proof outputs/transport-feasibility-v38c \
  --steps 12000 --checkpoint-every 2000
```

The reset bank already exists locally; the build command requires a fresh output
and checkpoint names if rebuilding. Only one client controls port 8081 at a time.
The server must use the original `outputs/state-teacher-server-v35/reset_bank`
after a reboot. `status.json` and `outputs/transport-place-rl-v38.log` record run
progress. This state-only run has no live camera videos and never exports ACT
training episodes. Qualification of the complete motion and camera review remain
required before collection.

## Checkpoint-selected near-grasp RL

`outputs/near-grasp-qualification-v37/` passed **20/20** frozen-policy trials
(ten fresh seeds, with open/contact prefixes). This test did not train or choose
between checkpoints. V36 achieved 256 training lifts in 355 episodes,
then lost all successes over its final 30 episodes and failed 0/4 final tests.
Only its initial/final weights were saved, so the earlier training behaviour
cannot be recovered as a checkpoint. Its entropy coefficient rose from roughly
0.001 to 0.01; this is an observed correlation, not an established cause.

`teacher.conservative_grasp` starts fresh with the same physical task/rewards,
fixes entropy at 0.002, uses actor LR 0.0001 versus critic LR 0.0003, and warms the
critics for 512 gradient updates before changing the actor. Checkpoints are saved
and tested about every 1,000 collected training steps, at the next episode end.
The final checkpoint is also tested. A later failure cannot overwrite an earlier
checkpoint with stronger validation results.

The live integration run `outputs/near-grasp-checkpoints-smoke-v37/` unexpectedly
produced a promising actor after 625 collected steps: four of four deterministic
validation lifts, followed by two of two separate test lifts. The selected actor
is `grasp_best.zip`, SHA-256
`34c4e3f4f53c072dfac364d1f3eaf0b8f3ba58f65cf2c0736a59791c6819491e`.
Its horizon is 32 steps; that horizon is part of its observation/control contract.
All 22 recorded validation/test trials (including the failed earlier candidates)
passed physics/action/reward re-scoring. The source actor and replay are preserved.

The qualification job froze that exact actor and evaluated twenty newly
reached states at the same 32-step horizon. The retained actor remains frozen
while the next curriculum stage learns transport and release. Passing
this test only establishes the near-grasp stage, not approach from home,
transport, placement, complete chess moves or RGB transfer.

The longer-run runner supports these separate data roles:

| Role | Source reset seeds | States |
| --- | --- | --- |
| Training | New 61001–61004 | Eight open/contact states |
| Checkpoint validation | Previously examined 56001–56002 | Four open/contact states |
| Final test | New 62001–62002 | Four open/contact states |

Validation maximizes the lower of the two family success fractions, then total
successes; ties keep the earlier actor. Selection is frozen before the final
test. These remain small diagnostic sets, not a complete reliability gate.
All models are evaluated without exploration. Validation physics traces and
rewards are re-scored before a checkpoint can be selected.

Evaluation runs only after the terminal training transition is stored. It uses
the raw environment rather than the learner's Monitor/replay path. Afterwards,
the unused automatic training reset and Python/NumPy/Torch RNG states are
restored. Runtime checks require unchanged actor weights and replay position
throughout validation. Tests compare complete replay observations, actions,
next observations and terminal flags against an otherwise identical SAC run
without evaluation. The selected model is a byte-identical copy of its frozen
checkpoint, retained as `grasp_best.zip`; the final training actor is separately
saved as `grasp_final.zip`. Replay at the selected episode boundary is saved when
a training checkpoint improves on the initial actor.

```bash
bash run_rl_python.sh -m teacher.conservative_grasp \
  --bank outputs/near-grasp-reset-bank-v37/manifest.json \
  --validation-bank outputs/near-grasp-reset-bank-v36/manifest.json \
  --output outputs/near-grasp-rl-v37 --steps 12000 --seed 63001
```

For training runs, progress is in `status.json`; checkpoint results and model hashes are in
`checkpoint_selection.json`. Source files are copied into the run before
training. The runner automatically validates checkpoints and tests the selected
actor, so it can be left running without manual monitoring. Grasp/lift success
still does not authorize complete-move ACT collection or real-hardware use.

For the current qualification, see
`outputs/near-grasp-qualification-v37/status.json`. Its actor is copied before
reset preparation and hash-checked after evaluation. There are no optimizer
updates, and success thresholds are at least 90% in each family with ten trials
per family. The two prefixes share each seed; fractions describe this narrow
bring-up test rather than independent-sample confidence guarantees.

## Near-grasp RL curriculum

The first curriculum RL run was `outputs/near-grasp-rl-v36/`, with log
`outputs/near-grasp-rl-v36.log`. Its explicit command remains **e2–e4**, but its
current objective is a **grasp/lift diagnostic**, not the complete chess move:
raise the actual pawn at least 10 mm and maintain an upright, stable grasp for
ten control frames. The opening midpoint must stay within 12 mm of the pawn's
specified grasp point; both fingers must be obstructed under a closing command
and settled, and relative slip during the hold must stay below 3 mm. Non-target
pieces must remain in their own squares and upright. The 12 mm grasp allowance
accepts the existing reference grasp; centring more accurately is also rewarded.

`teacher.reset_bank` stores both robots' joint positions, velocities and servo
targets plus all 32 pieces' poses and velocities. Original piece baselines and
chess goals remain separate from the intermediate robot state. Checkpoints are
reached by physically replaying a checked approach; pieces are not attached or
repositioned during policy execution. Pose/velocity restoration is checked before
the bank is accepted. PhysX contact solver caches are not serialized, so a reset
is not a bitwise replay guarantee. Reference continuation tests check that the
restored physical states can still achieve the lift objective.

`outputs/near-grasp-reset-bank-v36/manifest.json` selects two reset families:
an open approach (prefix step 140) and naturally closed finger contact before
lifting (step 155). Four source seeds provide eight training states; two disjoint
seeds provide four evaluation states. Starting with contact does not count as
success: the learner must cause a fresh lift and stable hold. Results are reported
separately by family so a lift from closed fingers is not presented as a newly
learned complete grasp or chess move.

`teacher.near_grasp` exposes 65 compact state/goal features: robot position and
velocity, opposing-finger midpoint relative to the pawn's 17 mm grasp target,
piece motion, neighbouring pieces, the desired e4 position, explicit move
coordinates, and the requested lift height. No highlighted image is needed by
the physics teacher. Actions are six bounded joint-target increments; zero holds
the previous command. Arm targets are limited to 15 degrees/s and the jaw to
45 degrees/s. No ACT, IK grasp alignment, closure guard or capture macro replaces
the learned actions. The actor and critics start fresh.

```bash
# One reset-enabled state server, normally already running on port 8081.
# After a reboot, reuse the bank while giving the server a fresh output folder:
bash teacher/run_server.sh --output outputs/state-teacher-server-restart \
  --reset-bank outputs/state-teacher-server-v35/reset_bank

bash run_rl_python.sh -m teacher.train_grasp \
  --bank outputs/near-grasp-reset-bank-v36/manifest.json \
  --output outputs/near-grasp-rl-v36 --steps 12000 --seed 58001 \
  --reference-proof outputs/near-grasp-feasibility-v36

# After completion, recompute saved evaluation outcomes and rewards:
bash run_rl_python.sh -m teacher.verify_grasp outputs/near-grasp-rl-v36
```

Output folders must be new. Only one client may control this server. The runner
checks reference feasibility, evaluates a frozen initial actor, trains only on
training checkpoints, then freezes and evaluates the final actor on the separate
reset set. The evaluation transitions never enter replay. Training and evaluation
successes remain explicitly labelled `grasp_lift_only`; they cannot be exported
as complete ACT demonstrations. Goal-specific reward terms and motion costs are
saved for audit, with success dominating the small bounded per-step preferences.

Initial checks: all 18 open checkpoints in the v35 bank restored their recorded
poses/velocities, and continuations from three prefixes achieved stable lifts.
A 640-step integration smoke test completed 128 SAC updates, saved checkpoints,
and ran its held-out evaluation. Its 32-step horizon was only an integration
check; it achieved no successful lifts. The v36 bank adds genuine closed-contact
starts to make early lift successes easier to discover. Current run outcomes
must be read from that run's status/results, not inferred from these checks.

## Ordinary imitation experiments before the reset curriculum

The current experiment targets **initial e2–e4**, before returning to captures.
The capture pilot (`state-teacher-pilot-v30`) completed 3,000 SAC steps but never
picked up the victim. No further capture RL was launched.

`teacher.reference` independently re-scores an existing successful e2–e4 trace,
then stretches whole six-joint segments together wherever needed to meet the
teacher's speed limit. This preserves arm/finger timing. It records new physical
state/action transitions; old source timings and rewards are not reused as RL
replay. Four references completed the move, in 306, 300, 300 and 307 steps, under
`outputs/e2e4-teacher-bootstrap-v31/reference_*/`. These are recorded references,
not learned-policy successes.

`teacher.ordinary` fits a standalone actor to those new trajectories. Three whole
episodes train it; the fourth is validation. Training-only normalization and a
small setpoint perturbation are saved with the model. The first incremental
actor picked up the pawn in 2/2 fresh rollouts but failed both complete moves,
retaining its grip and disturbing other pieces. The absolute-target comparison
picked up in 0/2; the mixed arm-increment/gripper-position comparison picked up in
1/2 and completed 0/2. These candidates are not ready for ACT-data collection.

The next experiment is `outputs/e2e4-teacher-robust-v34/`, log
`outputs/e2e4-teacher-robust-v34.log`. It uses the same audited reference data,
incremental arm commands and a learned gripper-position target. The clock input
is capped at the training demonstration range rather than extrapolated after
their end. During imitation fitting, 1.5-degree arm-pose perturbations recompute
the opposing-finger midpoint and every correlated piece-relative feature using
batched kinematics. Those are supervised robustness examples, not fabricated
simulator transitions or changed calibration. All validation frames remain
unperturbed and excluded from gradient updates. Two new simulation seeds,
43001–43002, test the frozen result after fitting. **No RL is started by this run.**

`teacher.refit` compares action interfaces on exactly the same physical targets;
source traces and scores are preserved, and converted labels never enter an RL
buffer. `teacher.evaluate` can re-evaluate a saved checkpoint through either the
state-only server or the original camera/control server. Camera recordings use
`--record-video --url http://127.0.0.1:8080`; they show actual closed-loop teacher
execution, not an offline animation. The actor still uses privileged state and
is not an RGB student.

Before restarting RL, require reproducible learned grasps and placements and
review actual motion. A low imitation loss or a pickup alone does not certify
the full move. Preserve a successful imitation baseline when comparing RL.

## Original capture bring-up commands

Run from the repository root. Output directories must be new.

```bash
bash teacher/run_server.sh --output outputs/state-teacher-server-v30

# In another terminal, after TEACHER_STATE_READY:
bash run_rl_python.sh -m teacher.benchmark \
  --output outputs/state-teacher-benchmark-v30 --steps 120

bash run_rl_python.sh -m teacher.train \
  --output outputs/state-teacher-pilot-v30 --steps 3000 \
  --seed 30001 --eval-seeds 31001 31002 \
  --bootstrap \
  outputs/hybrid-readiness-v29/sample_episodes_v2/scandinavian_capture/teacher_7901 \
  outputs/hybrid-readiness-v29/sample_episodes_v2/scandinavian_capture/teacher_7902
```

The local source reset episode, `outputs/replay/episode_000000.npz`, is still
required by the physics driver. Bootstrap recordings are also local artifacts
excluded from Git. Committed copies of the exported kinematics and approved
finger references are in `assets/`, with their original paths and hashes.

The new teacher's physical action scale is normalized to authored joint limits;
the transport converts back to the existing legacy units at the server boundary.
The frozen ACT calibration and model weights are unchanged.

`status.json` reports stages, completed training steps and final evaluation
counts. `pilot_complete` only means the experiment finished; it does not mean
the policy is ready to collect ACT data. This short run is intended to expose
learning/control issues before choosing a larger training budget.

## Next stages

1. Assess capture success, target/actual motion, unnecessary travel and pauses.
   Compare with v29 and the original real demonstrations. Review camera
   recordings before accepting any new policy as an ACT-data teacher.
2. Qualify the unified teacher from home, including visual motion-quality
   review. It now covers approach, grasp, carry, release and final home under
   one objective, but learned full-move reliability is not yet established.
   Capture reset history remains pending.
3. Benchmark cloned, parallel physics environments before large RL runs. The
   current server retains the original CPU backend and one environment; removing
   rendering alone does not make it a massively parallel GPU training stack.
4. Expand the implemented success-guided sampler beyond the small e2–e4 reset
   set to more moves and board configurations. Reserve complete configurations
   and episodes for evaluation rather than splitting frames.
5. Distill a satisfactory teacher into a camera student through DAgger. The
   intended student inputs are live fixed/wrist RGB, joint feedback and a
   separate desired-board render with move highlights. Compare separate versus
   transparent goal overlays, keeping live piece placement visible. Simulator
   poses and internal task progress remain teacher-only inputs; a student needs
   visual/history evidence to infer progress. Do not expect the existing ACT to
   understand a new goal-image input without training.
6. Add realistic calibration, camera, appearance, dynamics and magnet variation,
   then independently qualify unseen moves and real-hardware transfer. Recovery
   and board reset require examples of those tasks and explicit capture-area goals.

The proposed sampler/distillation direction is inspired by
[A Balanced Data Diet](https://arxiv.org/abs/2610.12465). Its scale and transfer
results have not been reproduced here.

## Validation

```bash
bash run_rl_python.sh -m unittest teacher.test_teacher teacher.test_imitation teacher.test_camera_evaluation teacher.test_resets teacher.test_grasp_checkpoints teacher.test_transport_place teacher.test_move_teacher
```

Tests cover unit conversion, target-speed and joint limits, acceleration costs,
uncertain transport, observation order, timeout bootstrapping, validation-prior
exclusion, and actual SAC updates through the Gymnasium Monitor wrapper. A real
simulator smoke run additionally checks the full local server/learner path.

Initial bring-up on this machine measured 120 state-only hold steps in 3.59 wall
seconds: 33.4 control steps/s, or 1.11 times real time. This excludes learner
updates and resets and is not a moving/grasping throughput measurement. The
320-step live SAC smoke completed 64 optimizer updates, saved/reloaded a working
checkpoint, and had zero successful captures under its short 60-step horizon.
It verifies the pipeline only. Local evidence:
`outputs/state-teacher-setup-v30.json` and `outputs/state-teacher-smoke-v30b/`.

The first full-horizon pilot is `outputs/state-teacher-pilot-v30/`, with log
`outputs/state-teacher-pilot-v30.log`. It runs 3,000 fresh training steps, optional
BC initialization from the two proof episodes above, and before/after evaluation
on 31001–31002. Results must be read from that run's status/evaluation files;
launching the run does not establish an improvement.
