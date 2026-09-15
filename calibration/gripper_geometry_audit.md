# Modified gripper audit — 2026-09-15

The stage references `SO100/so100_modified/so100_modified.usd` and its
configuration layers. It uses the custom centre-closing two-jaw gripper:
`Fixed_Jaw_mod_m`, `Moving_Jaw_1`, and `Moving_Jaw_2`, rather than the standard
fixed/moving-jaw pair in `SO100/so100.urdf`.

## Visual mesh checks

USD jaw vertices match the corresponding `SO100/assets/Moving_Jaw_1.stl`
(25,134 vertices) and `Moving_Jaw_2.stl` (34,764 vertices) exactly, in metres.
The separate CAD exports in `../LeRobot-S0-100-Models/gripper/` use millimetres
and a different coordinate frame/naming. Their approximate alignment is:

```text
imported metres = [CAD x, CAD z, -CAD y] / 1000 + [0, 0, -0.029]
imported jaw 1 corresponds to CAD moving jaw 2
imported jaw 2 corresponds to CAD moving jaw 1
```

Bidirectional vertex-to-triangle surface checks find at most 0.298 mm
difference at the fingertips (local Y below -40 mm), with tessellation
differences. One imported jaw has up to 3 mm difference around the servo hub
(near local Y=0 and Z=±17.5 mm), away from the contact tips. The complete
meshes are therefore not identical to those CAD exports. The comparison is
saved in `outputs/replay/gripper_mesh_comparison.json`; which CAD revision
matches the printed hardware still needs confirmation.

## Contact setup bug and correction

`/World/so100_blue/jaw_1/visuals` is an instance. The original contact setup
used ordinary USD traversal, which skipped that instance's mesh. Blue jaw 2
was not instanced, so only one blue finger received a collider. This was a
bug in the added physics setup, not evidence that the custom visual model
was a standard gripper or that the hardware grasp height was wrong.

`chess_physics.configure_contacts` now expands robot visual instances in the
temporary session layer before adding colliders. It asserts that both jaws
on both robots have collision meshes and records their paths in the replay
audit. The first `outputs/replay/both-jaws/` audit confirms all four. No
original USD or STL was saved or replaced.

All earlier contact trials and ACT runs used the incomplete collision set.
Their failed grasps do not establish correct/incorrect friction, aperture or
height. Treat the later height/pan/gripper-offset maps as experiments, not
validated calibration. Transport, model loading and recorded-transition
checks from those runs remain valid.

## Remaining physical assumptions

Finger colliders are convex decompositions, so visual-mesh agreement does
not prove contact-surface fidelity. The modified URDF specifies motor
reduction 1.0 and an opposite-angle follower jaw. The USD mimic relationship
also drives the jaws oppositely, but the old gripper percentage's physical
zero and opening still require validation. USD jaw-1 limits are -3° to 45°;
the source URDF instead specifies ±0.6 radians, so limits have been edited
since import. Neither should be treated as a measured hardware limit.

The model repository also describes magnets in the pieces and board.
Current contact physics includes gravity and friction, but no magnetic
attraction. Mass, friction and drive gains remain assumed simulation values.

## Corrected contact result

With both jaw colliders present, the original board-fit replay moved the pawn
but tipped it. The higher contact candidate then completed episode 0's physical
move at half speed: maximum pawn lift 22.82 mm, final board-plane error 1.35 mm,
upright cosine 0.9999999, and no other pieces displaced over 3 mm. The robot
returned to the recorded final pose. See
`outputs/replay/both-jaws-neck/physics_report.json` and the saved camera frames.

`legacy_blue_joint_map_physics.json` preserves that candidate separately. This
is one successful recorded state replay, not an ACT success or proof of general
calibration. The knight episode still exceeds the wrist limits with this map;
the original board-fit map remains available for comparison.

The subsequent ACT tests (`outputs/act-both-jaws/results.json`) also lifted the
pawn, placed it upright on e4, and returned home. Placement errors were 4.39
and 5.36 mm; one of the two passed the predefined 5 mm tolerance. Both left
the other pieces undisturbed. These are limited bring-up checks, not evidence
of reliability across the board or confirmation of the printed CAD revision.
