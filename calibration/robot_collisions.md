# Robot collision audit — 2026-11-08

The trained v108 final actor subsequently passed fresh qualification v109:
20/20 moves (five per initial blue pawn goal), zero robot contacts, minimum
sampled camera/CAD gap 7.662 mm, plus 4/4 matched-camera reviews. The exact
checkpoint and scope are pinned in
[`v109-collision-aware-neural-teacher.json`](../docs/baselines/v109-collision-aware-neural-teacher.json).
The earlier v99 qualification and corrections below remain historical results.

The active audit worlds now use decomposed printed housings, ten duplicate
imported hulls removed, and non-adjacent self-collision enabled. The new
joint-space reference clears the camera/servo interference. The current fitted
neural actor (`clearance-return-arm-v99`) passed 8/8 development moves and 4/4
camera reviews. All 2,176 filmed actions reproduced exactly from the frozen
checkpoint; their sampled camera/CAD gap stayed above 7.41 mm. Fresh qualification
is recorded separately in `clearance-final-qualification-v102`; check its final
status before using this candidate for collection.

Its predecessor passed 19/20 fresh qualification moves but failed the per-move
gate: after releasing in c2c4, its arm dipped and disturbed a neighbour. The jaw
command stayed open; contact drove the actual fingers closed. Correction used
separate training rollouts and changed only the arm output rows, preserving the
parent's jaw and shared network parameters exactly. A full-network refit
regressed to 5/8 development successes and was rejected. Neither fit is an RL
improvement.

The original contact configuration enables external contacts on both arms
(14 visual collision meshes per arm) but explicitly disables articulation
self-contact in `chess_physics.configure_contacts`. The authored USD has
self-contact enabled; the session override disables it. No authored cross-robot
pair/group filters were found. Robot-to-robot contacts are configured, but a
deliberate positive inter-robot contact test remains outstanding.

The user identified a real collision between the wrist camera housing and the
joint-2 servo corner during descent. The CAD repository contains
`gripper/gripper body with camera.stl`. Its aligned outer dimensions match
the imported `Fixed_Jaw_mod_m` mesh: approximately 60.2 × 39.0 × 93.38 mm.
The optical camera centre is inside its upper extension at gripper-local
(0, -25, 50) mm. Matching bounds does not prove every printed/contact surface
is identical; the camera is part of this assembly rather than a separate
named camera-housing collider.

## Separate experimental mode

```bash
bash teacher/run_server.sh --output outputs/self-contact-new --port 8082 \
  --robot-self-collisions --robot-collision-geometry decomposed --render-cameras
```

Use a new reset bank. The physics identity includes the self-contact mode and
pair exclusions, so old reset files cannot silently enter this configuration.
Only one client may control the server. The default retains the historical
physics for reproducibility; enabling this mode does not modify the USD.

Non-adjacent links collide. Connected neighbours and the two meshing jaw bodies
are excluded, preserving the intentional joints/gearing. In particular,
gripper/camera against shoulder or upper arm is **not** excluded. Housing hulls
are still coarse and may cover empty spaces; this is an audit configuration,
not a calibrated replacement for dedicated CAD collision meshes. Whole-body
jaw-pair exclusion also excludes finger-to-finger contacts.

`robot_contacts` records self/inter-robot body pairs, substep contact samples,
impulses and penetration in each control window. Detailed reports also identify
collider pairs and the deepest contact position. Piece contacts do not count
as robot collisions. Qualification and episode audits reject robot contact in
this new configuration; old self-disabled qualifications establish no
self-clearance guarantee. These contact checks do not certify a positive
clearance margin. Optional RL contact costs and immediate unsafe termination
are now available; checkpoint selection can also require the sampled CAD gap.

## First measured result

The frozen four-move fitted actor was checked on e2e4 with self-contact enabled
(`outputs/self-collision-e2e4-v82`). The 800-step motion failed to lift the pawn.
The report recorded gripper/shoulder contact in 243 control windows (steps
19–261), reaching about 0.93 mm solver penetration, and gripper/upper-arm contact
in 30 windows. These are contacts between configured convex collision hulls;
body-level records alone do not identify the exact printed camera surface.

The detailed 60-step probe `outputs/camera-servo-contact-v83b` identifies the
pair as `gripper/visuals/Fixed_Jaw_mod_m/mesh` against
`shoulder/visuals/Rotation_Pitch_Motor/mesh`. Reported contact points transform
to approximately gripper-local Z=69.8–73.0 mm, near the top of the camera
extension (mesh maximum Z=72.98 mm). Those local coordinates use end-of-frame
measured joints for substep contact points, so they are approximate. The probe
records self-contact in 51/60 control windows and correctly fails the contact
acceptance audit. It is deliberately time-limited, not a full move attempt.
The saved `contact_pose.png` and `videos.html` show the inspected geometry and
the actual synchronized camera views.

Before further policy expansion, inspect those specific contact surfaces,
replace false coarse-hull overlaps where necessary, then produce an approach
with camera/servo clearance and include that clearance in teacher observations
and learning/acceptance criteria. A successful piece move under the old
self-disabled physics is insufficient for this purpose.

## Surface refinement and corrected approach

An exact point-to-triangle audit (`collision-surface-audit-v84`) placed the
median camera and joint-2 motor contact points within 0.04 mm of the actual
camera surface and essentially on the motor surface. Some upper-arm contact
points were 3.7 mm from its real surface, showing the coarse hull filling a
void. The imported arm also contained duplicate collision meshes alongside
the newly added visual colliders. The decomposed mode expands those instanced
groups, disables ten duplicates, decomposes printed housings, and retains
servo hulls and both gripper fingers. Defaults preserve historical experiments.

Changing wrist pitch or hovering higher did not reliably clear the obstacle.
The old straight Cartesian approach initially extended the elbow through the
camera/servo interference. `ConfigurationReference` instead solves the desired
grasp configuration, approaches through joint space, retains physical finger
feedback for grasp/transport, and blends back to the home configuration after
release. This is a supervised training aid only. The fitted actor has continuous
joint/posture features and uses no reference planner or action override at runtime.

Twelve training references across e2e4, e2e3, d2d4 and c2c4 passed with no robot
contacts. Exact CAD surface gaps at recorded control frames remained at least
7.9 mm for the four checked camera/shoulder/upper-arm pairs. One e4e5 reference
from a later board state also passed without self/inter-robot contact. These
results do not establish arbitrary-move or real-hardware reliability.

The first neural actor (`clearance-neural-bootstrap-v92`) reproduced all 4,349
held-out recorded actions exactly during independent observation reconstruction.
Those eight trials passed the 5 mm sampled camera-clearance gate, with a smallest
gap of 7.75 mm. A separate test confirmed the old unsafe actor still generates
camera/servo contact under the refined geometry; the real hazard was not deleted.

Mesh-gap validation uses `python-fcl` in the separate dataset environment.
Generate the source geometry without launching physics:

```bash
bash run_usd_python.sh -m teacher.export_collision_surfaces \
  --output outputs/robot-surface-export-new
bash run_dataset_python.sh -m teacher.robot_clearance \
  --meshes outputs/robot-surface-export-new/meshes.npz \
  --trial outputs/your-review/your-trial --output outputs/camera-gap-new.json
```

The metric checks four named CAD surface pairs at recorded control frames.
It does not certify between-frame separation, every possible self-contact,
or positive clearance to the other robot. PhysX contact records cover those
other robot-body contacts. Future RL runs can penalize robot contact and reject
checkpoints violating the camera-margin gate; historical scores stay unchanged.

The v108 comparison additionally gives learning a continuous measured CAD-gap
cost inside an 8 mm buffer, while retaining the 5 mm acceptance threshold.
Costs are independently reconstructed from measured joints and the hashed mesh
archive. Physical replay is re-exported under that exact reward contract; pose
variants remain supervised labels only. The warm jaw/shared network is frozen
while five arm output rows learn clearance and placement corrections. This is
an experimental comparison, not evidence of improved or hardware-safe motion.
