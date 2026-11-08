# Current experiment record

Updated 2026-11-08. Local output artifacts are excluded from Git. Results here
are scoped per experiment below; the pinned baseline covers blue e2–e4 only.

## Latest collision-aware candidate

**RGB-only pilot passed:** frozen ACT checkpoint 750 with **50 executed actions
per chunk** completed 20/20 held-out starts (`clearance-rgb-heldout50-v116`),
5/5 per initial blue pawn goal. No runtime state-teacher action or analytic
intervention was used. Robot contacts: zero; minimum sampled camera gap:
7.919 mm; maximum final placement error: 0.589 mm. Exact weights, runtime
override, scope and reports are pinned in
[`v116-rgb-act-student.json`](../docs/baselines/v116-rgb-act-student.json).
This is limited simulator validation, not real-robot readiness. Historical
ten-action failures below remain retained. New GoalACT/training/review defaults
now use 50; saved historical checkpoint configs are unchanged and must receive
the explicit override when launched through another adapter.

**Qualified trained candidate:** v108 `move_final.zip`, pinned in
[`v109-collision-aware-neural-teacher.json`](../docs/baselines/v109-collision-aware-neural-teacher.json).
Fresh qualification v109 passed 20/20 (5/5 per listed pawn move), 11,337 audited
transitions, zero robot contacts and minimum sampled camera gap 7.662 mm.
Final placement error was at most 0.595 mm. Matched camera review passed 4/4
with the same frozen actor. These satisfy the existing pilot-collection
prerequisites; no new ACT episodes have been collected from this checkpoint yet.
Captures, other positions, red control and real hardware remain unqualified.

**RGB distillation pilot completed:** `clearance-rgb-episodes-v110` accepted
12/12 episodes, 6,725 frames, and the three-view local LeRobot export passed
its image/state/action and goal-image loader round trips. Mixed ACT fitting
completed 1,000 updates and selected checkpoint 750. Held-out sim L1 fell from
0.41384 to 0.05887; real replay L1 fell from 0.14935 to 0.12474. Selected
checkpoint reload was bit-exact on 16 sim/real held-out samples, and original
ACT weights/config remained unchanged.

The RGB-only closed-loop test passed **1/4**, d2d4 (683 steps, 0.591 mm final
placement error). e2e4/e2e3 timed out after only 4.66/5.09 mm maximum lift and
never established the sustained-lift condition. c2c4 stopped on reported
jaw_2/wrist self-contact (two samples, zero reported impulse); it reached only
4.48 mm lift. All four sampled camera/shoulder/upper-arm margin audits passed,
so the c2 failure is a different configured collision pair. These results do
not establish a reliable RGB policy. Gallery: `clearance-rgb-review-v113/videos.html`.
The qualified state teacher is unchanged. The next useful step is pickup/lift
diagnosis and training-only corrective distillation on states the student
actually visits, with separate held-out evaluation retained.

**Lift-stall execution diagnostic v114:** the preserved original ACT config
executes 100 actions per chunk, while the pilot adapter overrode execution to
10 (training still predicts all 100). Offline reconstruction from H.264 review
frames shows the stalled e2 policies predicting roughly 26–34 mm tool rise in
their full chunks, but usually only 0.4–2.2 mm in the first ten targets. A target
rise above 15 mm first appears around indices 19–21. Replanning after ten steps
may repeatedly defer that lift. These are approximate-image FK diagnostics,
not physical rollout outcomes or proof of causation; no weights changed.
Artifact: `rgb-lift-chunk-diagnostic-v114.json`. `clearance-rgb-chunk50-v115`
tests the same selected weights with 50-action execution on development starts,
retaining speed limits, robot-contact termination and the camera margin audit.
Resolve this execution mismatch before assuming broader move coverage is needed.

`clearance-rgb-chunk50-v115` completed **4/4 development moves** with the exact
same checkpoint-750 weights, no additional training. e2e4/e2e3 lifted 29.1/28.1 mm
and completed placement/release/home. Every trial had no self/inter-robot
contacts, at least 7.919 mm sampled camera clearance and final placement error
below 0.552 mm. This supports the chunk-replanning explanation, but the earlier
10-action test used different physical reset seeds, so this is not a matched-seed
causal comparison or broad reliability qualification.

Freeze execution at 50 actions per chunk and the same model SHA256
`7e89ad2566a2c21be8f9d79f1bfefa5731f7e8a8f0457c37fe3cfc4ba44c47e7`.
`clearance-rgb-heldout50-v116` now evaluates all twenty v109 reset starts
(140001–5 per goal), held out from RGB fitting and execution-window tuning.
Model inputs remain measured joints and the three RGB inputs, with privileged
physics used only for task scoring and the unchanged contact/clearance audits.
No training, runtime teacher action or geometric recovery intervention is used.
The source checkpoint config still records ten executed actions; evaluation
explicitly overrides this with `--n-action-steps 50`. Do not use the earlier
ten-action wrapper setting when interpreting the new result. The twenty-trial
run is unattended; its outcomes remain pending.

The pilot collection requested
three fresh starts per goal (seeds 141001–3), twelve attempts total. Frames
contain matched before-action fixed/wrist RGB, measured legacy joints,
executed legacy targets and the separate desired-board rendering. Every
accepted episode must independently reproduce physics, rewards, clearance,
frame/action correspondence and the frozen actor's actions.

The automated local handoff is `outputs/run-clearance-rgb-pipeline-v112.py`,
log `outputs/clearance-rgb-pipeline-v112.log`, stage file
`outputs/clearance-rgb-pipeline-v112-status.json`. It waits for the collector
to release port 8082, requires three accepted episodes per goal, exports
`clearance-rgb-dataset-v111` with a third goal image, then fine-tunes a separate
copy of the preserved original ACT as `clearance-rgb-act-v112`. Eight complete
sim episodes train and four (one per goal) validate. Replay uses 32 real
episodes spread across the recording sequence and four disjoint whole real
episodes. Each AdamW update gives equal weight to sim and real loss. Real
replay supplies its actual two views; simulation supplies three. Legacy units,
ordering and original camera/state/action normalization remain unchanged.

The planned 1,000-update fit uses LR 1e-5, per-domain batch size one and ten
executed actions per predicted 100-action chunk. Checkpoint selection minimizes
held-out sim L1 subject to real validation L1 no greater than
`initial_real_l1 * 1.10 + 0.01`; initial weights are the fallback. This gate
does not prove retained real-hardware performance. The pipeline checks exact
selected-checkpoint reload, then records four RGB-only home-start tests as
`clearance-rgb-review-v113`, with the teacher used solely for the unchanged
task/physics contract and scoring. Safe successes require robot-contact and
sampled camera-clearance audits. No source ACT overwrite or remote upload.

`clearance-rgb-preflight-v111` passed two mixed-domain GPU updates and bit-exact
three-view saved inference, with original ACT/config hashes unchanged. Its
discarded weights did not initialize the actual fit. Latest tests: 116 teacher
tests pass. The unattended handoff completed; its completion flag describes
the pipeline, not RGB-policy qualification.

Both active worlds use non-adjacent self-collision and decomposed printed
housings with ten duplicate imported collision meshes disabled. The real camera
housing/joint-2 servo contact remains detectable; the approach reference now
reaches through joint space to avoid it. Details and reproducible mesh-export
commands are in [robot collisions](../calibration/robot_collisions.md).

`clearance-return-arm-v99/move_bc.zip` is the current fitted candidate, SHA256
`f1393f1cbe3d751e94d37741d8238d2acc95046c5ceb36decd084c6508cf6409`.
It passed 8/8 development moves and four actual paired-camera reviews
(`clearance-arm-camera-v101`). Independent reconstruction reproduced every one
of the 2,176 filmed actions exactly. Sampled exact camera/CAD separation was at
least 7.41 mm, with zero self/inter-robot contacts. These results cover initial
blue e2e4, e2e3, d2d4 and c2c4 only.

The preceding v92 actor passed 19/20 fresh starts, failing c2c4 after release:
its arm dipped into the placed piece while the jaw command remained open.
Separate training-only C2 rollouts supplied the correction. A full-network
refit v97 regressed to 5/8 and was rejected. v99 instead fits only five arm
output rows, preserving the v92 shared trunk and jaw parameters exactly.
Fresh qualification is `clearance-final-qualification-v102` (seeds 136001–5,
five starts per goal, no gradients or checkpoint selection). Its final status,
including the predeclared 5 mm CAD gate, determines qualification.

The next SAC comparison has an optimizer/replay preflight
(`clearance-optimizer-preflight-v101.json`): 2,130 real reference transitions,
zero synthetic replay, unchanged actor during critic warmup, and active actor
learning rate 1e-6. Supervised observations anchor the warm neural predictions;
they add no invented rewards or physics transitions. Contact penalties,
immediate unsafe termination and a camera-margin checkpoint gate are optional
new-run settings. No RL improvement or new ACT collection is claimed yet.

v102 found an e2e3 placement timeout: the pawn remained gripped roughly 6 mm
above the board and 10 mm from target. This makes its per-case reliability gate
unreachable, even if the remaining starts pass. The failure is retained; its
states/actions are not training data. `launch-clearance-rl-v104.py` queues a
6,000-step comparison after v102 finishes and releases port 8081. It requires
the independent contact/CAD audits to pass, while allowing the reported task
failure because improving reliability is the comparison's purpose. The
baseline remains ineligible for ACT collection. Log: `clearance-rl-v104.log`.

v104 completed 6,000 SAC updates. Initial validation passed 8/8 including the
camera margin. The checkpoint at step 3,323 passed only 1/8; six moves placed
the pawn but timed out during completion, and one disturbed another piece.
The final trained actor physically completed 8/8 validation moves, but its two
e2e3 trajectories had sampled camera gaps of 4.994 and 4.888 mm, below the
predeclared 5 mm gate. Thus final accepted validation was 6/8. No validation
self/inter-robot contacts were recorded. Selection correctly retained the
initial fitted actor, which passed 3/4 separate final tests; e2e3 again stalled
with the pawn held roughly 6.8 mm above the board. This run did not establish an
improvement over the starting actor and does not permit ACT collection. The v102 fresh baseline
qualification finished 19/20, zero robot contacts, minimum sampled gap 7.197 mm;
its e2e3 per-case gate failed. A future comparison should give learning direct
clearance feedback before approaching the hard acceptance boundary, alongside
training-only placement corrections. Qualification/test failures remain excluded
from gradients and replay.

## Clearance reward and placement supervision v108

`placement-training-states-v105` collected eight new frozen v99 rollouts from
the existing training starts (physical reset seeds 128031–2), all successful,
4,482 physical transitions, no analytic interventions. These are training-state
data, not accepted ACT episodes. `placement-physical-replay-v106.npz` re-audits
the original action/physics traces, then recomputes contact and camera-buffer
costs from recorded measured joints. Qualification and final-test data remain
excluded.

The optional `quadratic_camera_buffer_v1` cost begins inside 8 mm. It is zero
outside the buffer and reaches 0.04 per control step at the unchanged 5 mm
acceptance boundary. It uses the same hashed four CAD surface pairs as the
acceptance audit. Default rewards and historical scores stay unchanged.
`clearance-reward-live-preflight-v107` completed a training-start e2e3 with the
new reward; its independent raw action/reward/clearance audit passed.

`placement-arm-corrections-v106.npz` contains 1,555 supervised held-piece arm
labels from 528 measured training states and 1,027 consistent pose variants.
The labels use the existing configuration reference, balance the four goals,
and correct transport/lowering only; they never enter physical replay and do
not override executed actions or supervise the jaw. `clearance-optimizer-preflight-v107`
verified physical-only replay, unchanged actor during critic warmup, arm updates
after warmup, exact preservation of the shared actor/jaw/exploration parameters,
and bit-exact saved neural inference. All 114 teacher tests pass.

`clearance-placement-rl-v108` starts from the preserved fitted v99 actor with
fresh critics/replay: 6,000 SAC steps, actor LR 1e-6, 2,048 critic-warmup updates,
warm-policy anchor weight 50, placement supervision weight 5, bounded exploration,
contact costs/unsafe termination and the unchanged 5 mm checkpoint gate.
Only the five arm mean-output rows train; the shared actor, jaw and exploration
parameters are frozen. This is a normal neural actor at inference. The run is
launched without ongoing monitoring; inspect its final status and selection
before claiming any improvement or collecting ACT episodes. Log:
`outputs/clearance-placement-rl-v108.log`.

v108 completed all 6,000 updates. The **final trained actor** passed 8/8
development validation trials with zero self/inter-robot contacts and a minimum
sampled camera gap of 7.919 mm. The grasp preservation audit passed exactly.
Initial validation also passed 8/8; mean completion was 551 steps initially
versus 557.25 finally. Both occupy the same predeclared 19-second efficiency
bucket, so selection correctly retained the initial actor on the tie. The
reported 3/4 final-test score therefore belongs to the **initial fitted actor**,
not the newly trained final actor. No unseen-start success improvement is yet
established for the trained actor.

Freeze the trained candidate explicitly as `move_final.zip`, SHA256
`ede018b3113ed9367b0b362afc28c44c3223797513f18d4257f9a8a3e889ace5`.
`clearance-trained-qualification-v109` tests twenty fresh starts (seeds
140001–5, five per goal) with the unchanged contact and 5 mm margin gates.
`clearance-trained-camera-v109` records four validation moves in the independent
camera world. Both use the frozen final actor without gradients, selection or
analytic overrides. These checks are running; their final reports determine
whether this trained candidate can progress toward ACT collection.

Current ownership: state-only port 8081 `clearance-state-v85b`, camera port 8082
`clearance-cameras-v89`, shared immutable reset bank under v85b. Historical port
records below describe previous experiments. Latest verification: 110 teacher
tests and eight selected root geometry/magnet/video tests passed.

## Established

- Original ACT, authored geometry and legacy calibration remain preserved.
- `near-grasp-qualification-v37`: frozen learned pickup primitive passed 20/20
  fresh near-grasp/contact starts. This does not establish a move from home.
- `unified-move-feasibility-v39b`: physical reference completed ten reset
  families, including home, with 15°/s arm and 45°/s jaw command limits.
- `task-space-feasibility-v42`: the same ten reference continuations completed
  through generic differential IK and a binary open/close jaw.
- `unified-bootstrap-validation-v40d`: one neural actor completed grasp, carry,
  placement and return from an open near-grasp start. Home-start reliability
  was not established. This was imitation, not an RL improvement.
- `reactive-oracle-pilot-v45`: analytic geometric feedback completed home →
  grasp → carry → release → home in 567 steps, final error 0.24 mm, with no
  neighbouring square violations. This is a training aid/calibration check,
  **not a learned-policy result**.

## Negative results

- Scratch joint SAC v39 did not learn a full move from home. The selected
  checkpoint was the initial actor; its one test goal was `home_settle` only.
- Reference-assisted SAC v41 did not improve on its initial fitted actor.
  Selected test successes were retreat/home continuations, not home starts.
  Increasing discount to .997 addressed the long horizon but was insufficient.
- Low offline action error did not imply closed-loop success. Task-space
  imitation v42–v44 failed the home-start camera check.
- Some reference waits paired almost identical settled states with different
  next actions. An optional filter removes 39 idle imitation labels from 2,780
  reference transitions while preserving gripper transitions and all replay.

## Current direction

Train a continuous neural policy with geometric corrective supervision, then
compare conservative RL against that frozen baseline. Learned-policy evaluation
must run the saved actor without an oracle, phase switch or waypoint override.

`fit_oracle.py` labels training states and geometry-consistent supervised pose
variants using the analytic oracle. `collect_corrections.py` gathers training-only
on-policy states; oracle interventions are counted and mixture successes cannot
qualify a learned actor. Actual physical transitions remain independently audited.
Synthetic observations never enter physical RL replay.

First fit: `oracle-distilled-bootstrap-v47`; independent home camera check:
`oracle-distilled-home-review-v47`. The broader analytic probe is
`reactive-oracle-training-v46`. Some legacy held/carry starts are unsafe for this
field; inspect per-trial outcomes before treating it as positive demonstration
data. Those starts use a different grasp pose from the oracle's home trajectory.

The compact geometry actor v51 completed the held-out home camera trial in 783
steps. The v53 corrective fit completed it in 572 steps (19.1 simulated seconds),
0.35 mm placement error, no square violations. RMS measured arm acceleration
fell from 45.2 to 16.4 °/s²; this diagnostic alone is not a motion-quality certificate.
`neural-home-qualification-v54` froze v53 and passed **20/20 new home-start
trials**, 0.5 mm pawn jitter, 11,402 independently audited physical transitions.
Frozen actor SHA256:
`1b2073129b40ddd6c0df2431e15274bdef8a2d2d1a3b8bfb5f494a31a75f37f9`.
This establishes the implemented e2–e4 completion goal on this setup only.

PhysX instantaneous jaw velocities were noisy under stationary contact. The v2
oracle uses a full-close command and obstructed narrow aperture to recognize
a grip, rather than blocking lift on those solver velocity spikes. All v2 labels
and oracle version/source hashes are recorded in the new fit manifests.

`flow-reset-set-v52b` contains 50 physically reached checkpoints across five
disjoint seed groups and ten families, all using one consistent geometric grasp.
Five analytic home references and a 10-family restored-state feasibility check
(`flow-feasibility-v55`) passed. The v56 neural fit completed all ten training
families independently (`flow-neural-check-v57`, no oracle interventions).
`flow-neural-rl-v58` compares conservative SAC against that frozen fitted actor.
Its checkpoints are selected by held-out family success, retaining the earlier
actor on ties; only final evaluation can establish an RL improvement.

v58 was stopped at 4,030 training steps after the first six completed checkpoint
validation trials all failed (initial actor: 10/10). Raw partial evaluation and
the initial checkpoint are retained; this was a regression, not an improvement.
The new `flow-regularized-rl-v59` combines SAC and supervision in **one** Adam
actor update, rather than two momentum-coupled steps. Reference weight is 20,
online analytic corrective supervision weight 5, with no decay in this run.
The oracle labels replay observations only; it never changes replay rewards,
executed actions or runtime inference. A live optimizer preflight verified actor
freezing during critic warmup, one Adam step per update, and exact predictions
after reloading a checkpoint without the oracle object.

v59's first trained checkpoint (2,324 steps) passed 10/10 validation families,
including home, in 570 home steps versus the fitted initial actor's 566. That
shows retained behavior, not improved speed or broad reliability. Optional
efficiency selection only applies after perfect family success and compares
home completion time in whole simulated-second buckets; other ties preserve
the earlier checkpoint. Its fresh-start qualification is a separate next gate.

`neural-act-pilot-v60` collected two new home starts from qualified v53: both
accepted, 1,145 frames. `neural-act-dataset-v60` passed the real LeRobot loader
round trip (six sampled frames, exact RGB/state/action values). This is a local
two-episode pipeline pilot, not a diverse ACT training corpus. A localhost-only
gallery serves the review artifacts on port 8090 and does not control physics.

The explicitly frozen **trained** v59 checkpoint at step 2,324 passed a second
20/20 fresh-home qualification (`rl-home-qualification-v62`), 11,431 audited
transitions, and actual camera review (`rl-neural-home-review-v63`, 570 steps).
Its neural weights differ from the fitted initialization; reloaded inference
has no analytic oracle. This establishes a usable RL-fine-tuned e2–e4 teacher,
not an improvement over the fitted actor's already perfect tested success.
The fixed candidate was chosen as the first trained checkpoint matching all
ten validation families, then frozen before these new qualification trials.
The exact pinned checkpoint is documented in
`docs/baselines/v65-neural-teacher.json`.

v59's mean policy stayed successful at 4,770 steps (10/10 validation), but learned
Gaussian exploration standard deviations reached >4 on lift/jaw outputs. The
run was stopped for excessive stochastic noise; deterministic performance was
retained. `flow-bounded-rl-v61` tests the same guided SAC with Gaussian log std
bounded to [-5, -2.5], about 0.0067–0.082 normalized pre-squash standard deviation.
Warm-switch/save/reload tests prove the deterministic neural mean is unchanged
by the distribution bound. Its 6,000-step run completed **22/22 training
episodes**, including three home starts. Initial, checkpoint and final validation
all passed 10/10 families. The selected final test passed 10/10 (one home start,
nine continuations). Selection retained the fitted initial actor because trained
checkpoints did not improve the predeclared success/time ranking. This establishes
stable training and retained performance, not an RL improvement. Collection uses
the separately qualified **trained** v59 checkpoint pinned above; the selected
initial actor is not represented as RL-trained.

`rl-act-pilot-v65` accepted 2/2 episodes from the qualified RL checkpoint,
**1,149 frames**. `rl-act-dataset-v65` passed the real LeRobot loader roundtrip
(six sampled frames, exact RGB/state/action values). Together with the fitted
baseline pilot there are four source episodes; the two teachers stay in separate
datasets with explicit checkpoint provenance.

The final teacher validation suite passed 78 tests. Original ACT weights,
authored scene, physics calibration and root residual/hybrid code are preserved.

`collect_episodes.py` and `export_episodes.py` prepare a local ACT pilot after a
passing >=20-trial qualification and matched-camera review of the exact actor.
They preserve before-action RGB, measured legacy state and executed legacy
joint targets. `episode_data.py` independently replays scoring/controller math,
checks frame/action correspondence, reproduces every neural action from the
frozen checkpoint, and verifies the move overlay from the raw red view. This
is separate from the legacy hybrid collector; no scores are renamed to make
an episode pass its acceptance gate. No remote upload or ACT retraining occurs.
Episodes also retain the exact initial/desired FEN and a desired-board SVG with
the move highlighted, following the separate board-diagram idea in
`chess_vision/dual_highlight_move.py`. The current export keeps the original two
camera inputs; the goal diagram is recorded for a later student input change.

## Qualification and collection gate

1. Neural actor completes from home without interventions.
2. Freeze it and audit at least 20 fresh home resets. `qualify_move.py` defaults
   to a 90% threshold and 0.5 mm pawn jitter. Scope remains this e2–e4 setup.
3. Review matched camera recordings for smoothness, travel and release quality.
4. Export only a small explicitly accepted ACT pilot, using executed joint
   targets in legacy units; raw Cartesian actions are not ACT training targets.
5. Expand to other moves, both robots, board states and captures. RGB
   distillation and real hardware transfer remain subsequent stages.

## Live simulator ownership

- 8081: state-only server, `state-teacher-server-v35`.
- 8082: paired-camera server, `state-teacher-camera-review-v39`.
- Independent physical worlds share immutable reset files. Only one controlling
  client per port. Verify a worker has stopped before starting another.
- Rendering checks that time, both robots and every piece remain unchanged.
  Videos contain matched actual camera pairs. Teacher inputs remain state.


## Multi-move curriculum (2026-11-08)

The original e2–e4 task/qualification stays pinned. A separately versioned
`ordinary_move_v1` task validates legal move histories, source/destination and
blue pawn/knight control. Observation move coordinates now reflect the requested
UCI move. Existing e2–e4 inputs reproduced all 1,149 recorded RL-pilot actions
with zero numerical error (`e2e4-compatibility-v66.json`).

`multi-goal-frozen-v66` tested the original frozen teacher on two new seeds per
move: e2e4 2/2, e2e3 0/2, d2d4 0/2, c2c4 0/2. These evaluation seeds (120001–2)
are excluded from corrective training. The failures show the original learned
teacher cannot yet be treated as a general goal-conditioned chess policy.

The analytic training aid completed e2e3/c2c4, but its original 65° pitch/90° roll
pose contacted the back-rank king during d2d4 approach. Tightening XY centring
alone did not resolve that physical interference. Recorded successful legacy
ACT d2d4 traces use a different wrist orientation. Separate orientation probes
are labelled analytic reference results, never learned-policy improvements.
An 85° grasp pitch passed d2d4 and e2e4 with actual camera recordings; the wider
four-goal reference and new multi-goal training bank are being checked.

Home-state bank: `ordinary-curriculum-v73`; training references:
`ordinary-references-v73`. Training seeds 121012–14, validation 122001–2 and final
hold-out 123001. The held-out probes and original qualified policies are preserved.
Warm imitation now copies only actor weights into a new model, with fresh critics,
optimizer and replay, including when the parent actor was RL-fine-tuned.

The 85° pitch/90° roll analytic aid completed all twelve training home references
(three per goal). Their 7,461 physical transitions stay training-only. The first
multi-goal fits v74/v75 had low offline loss but stalled in closed loop. v76 adds
three continuous geometric distance features and retains unperturbed physical
examples in half its batches. Its four camera validation trials passed 4/4;
the full development validation passed 7/8 (one e2e3 miss). These are imitation
results, not measured RL improvements.

`ordinary-rl-v77` compares bounded, reference-regularized SAC with the frozen v76
baseline. `ordinary-qualification-v78` was stopped after five e2e4 successes and
one e2e3 miss: with five planned trials per goal, e2e3 could no longer reach the
predeclared 90% threshold. Qualification traces are retained and excluded from
training. `ordinary-corrective-transitions-v80.npz` instead reconstructs twelve
**training-only** v75 rollouts (8,891 physical transitions) for corrective labels.

The RGB ACT path now supports a separate desired-board image, balanced sim goals,
whole-episode validation and two-view real replay. A two-update GPU preflight
(`rgb-student-preflight-v79c`) uses the existing qualified e2e4 pilot only; it is
an optimizer/checkpoint plumbing test, not a four-goal or camera-policy success.
The original ACT and old calibration units remain preserved. A save/reload test
is required before using student checkpoints in simulation.

`student-roundtrip-cpu-v79c.json` verifies exact copied model parameters, a
2.4e-6 legacy-unit difference from original two-view inference, and bit-exact
three-view save/reload outputs. The e2e4 RGB pilot v81 completed 1,000 updates
with equal real replay. Selected checkpoint 750 reduced held-out episode L1
from 0.508 to 0.052, but its first camera-only home rollout timed out after a
partial 7.3 mm lift. Low offline loss does not establish a usable RGB policy.

The full v77 comparison completed 4,000 SAC updates: initial validation 7/8,
first trained checkpoint 4/8, final trained actor 5/8. Selection retained the
**initial fitted** actor, which passed 3/4 final tests. This is a regression,
not an RL improvement. Corrective fit v80 also failed its first recorded e2e4
pickup and was rejected. No new four-move ACT episodes were accepted.

## Camera housing and robot self-contact

The user identified a camera-housing/joint-2 servo collision on the real robot.
The historical simulation explicitly disabled self-collisions. An optional
non-adjacent self-contact mode and robot-pair contact reports now provide a
separate physics audit; see [robot collisions](../calibration/robot_collisions.md).
The first frozen e2e4 probe recorded gripper/shoulder contact in 243 control
windows and gripper/upper-arm contact in 30 windows, then failed the pickup.
Coarse hulls require surface-level review before training against this geometry.
No further policy expansion or collection readiness is claimed from those runs.

Latest checks: 96 teacher tests passed; 23 selected root/contact tests passed.
The USD-only launcher now recognizes the installed extension's actual name.
Live world 8081 retains historical physics (`multi-goal-state-v66`); 8082 is
the separate self-contact audit (`self-collision-camera-v83`), with its own bank.
Detailed probe `camera-servo-contact-v83b` identifies the modified camera/gripper
mesh against `Rotation_Pitch_Motor`, with contacts near local Z=70–73 mm at the
top of the camera housing. It records self-contact in 51/60 approach windows;
the collision acceptance audit rejects it. All experiment workers have finished;
the simulator worlds remain paused between commands. No further RL run has
been launched against the unreviewed self-contact geometry.
