# Current experiment record

Updated 2026-11-07. Local output artifacts are excluded from Git. Results here
apply to the blue robot's ordinary e2–e4 move on the current simulated setup.

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
