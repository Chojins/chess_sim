# Legacy blue home alignment check

Measured from local episode 0 and the saved Sim_Stage.usd on 2026-09-14. Bases and board held fixed. No calibration correction applied.

The recorded home and the approximate USD home are not independently registered. This comparison measures differences under the existing provisional map; it cannot determine the true physical joint zero offsets.

| Joint | Legacy home preset | Recorded start | Recorded end | Saved USD home | Mapped start | Mapped end |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| shoulder_pan (degrees) | 0.000 | -0.088 | 0.264 | -2.895 | -2.812 | -3.164 |
| shoulder_lift (degrees) | 180.000 | 180.967 | 180.967 | 22.947 | 21.733 | 21.733 |
| elbow_flex (degrees) | 130.800 | 134.648 | 132.627 | -48.698 | -44.852 | -46.873 |
| wrist_flex (degrees) | 86.600 | 84.902 | 86.045 | 57.701 | 56.002 | 57.145 |
| wrist_roll (degrees) | -90.000 | -92.812 | -92.109 | 89.999 | 92.812 | 92.109 |
| gripper (legacy %; sim degrees) | 0.000 | 6.803 | 0.227 | -0.400 | 7.510 | -0.136 |

Start and end values are also the medians of their respective first/last 10 frames. The gripper is partly open at the start, so it should not be forced to the closed USD home.

## What endpoint matching changes

The board rank pitch is 22.5 mm. Matching the recorded start arm pose to the exact authored USD home would shift the gripper link origin about 8.6 mm forward near closing (frames 180–190), and 7.9 mm near release (frames 240–250). Matching the final home instead gives about 3.7 mm and 3.4 mm. These probes leave the gripper conversion unchanged.

Those shifts are smaller than one square, but do not exclude a home/zero error: they only test the difference between the old preset and recorded endpoints, while assuming the approximately posed USD arm is correct.

## Where to calibrate next

Keep the fixed bases and board fixed. Resolve shoulder-lift, elbow-flex and wrist-flex zero offsets against multiple known board contact poses, including the pawn pickup and placement. The current FK predicts that one simulated elbow degree moves the gripper link origin roughly 3 mm along the board near both contact phases; a modest zero error can therefore account for a square. An elbow-only change also changes height and tool orientation, so it is not a validated correction.

The mapping also stores shoulder home as 22.700 degrees whereas the saved USD contains 22.947 degrees (a 0.247-degree discrepancy). This small anchor approximation is separate from the unknown physical alignment.

The matching pickup and placement displacement reported by the user is consistent with a calibration bias, but these endpoints alone cannot uniquely determine the joint offsets. Use multiple poses to check a correction before applying it to ACT.

Machine-readable joint values, hypothetical offset corrections and FK sensitivities are in `../outputs/replay/alignment/home_comparison.json`. FK values describe the gripper link origin, not a measured fingertip tool center.

## Subsequent candidate fit

`legacy_blue_joint_map_board_fit.json` now provides a separate correction fitted
to the episode 0 pawn pickup/release, keeping bases and board fixed. It changes
shoulder lift by +0.885 degrees, elbow by -1.810 degrees and wrist flex by
-6.910 degrees. The geometric fingertip proxy's board-plane errors fall from
22.8/22.5 mm to 1.54/1.65 mm at the two fit frames, and remain below 2.8 mm at
four neighbouring check frames. All measured states and recorded actions stay
within USD limits. Absolute grasp height, aperture and physical contact remain
unvalidated; this is a same-episode fit, not an independent calibration.

The candidate completed full kinematic replay loops with both cameras: a sample
after 60 seconds recorded 880 complete pairs and zero missing pairs at 14.55
pairs/s. Pickup and release images were inspected, and the existing vision
client received ten distinct pairs with the move overlay on red only. Fit
inputs are in `episode_000000_contacts.json`; rerun with `../run_joint_fit.sh`.
