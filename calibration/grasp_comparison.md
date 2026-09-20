# Recorded versus ACT grasp, 2026-09-20

Run `./run_usd_python.sh compare_grasps.py` to reproduce
`outputs/grasp-comparison/comparison.json`. This reads the successful
half-speed physics replay and the three window-100 ACT failures, then evaluates
forward kinematics using their measured joints and the existing joint map.
It does not move robot bases or change the map, USD assets or checkpoint.

The comparison uses the gripper tool proxy at local `[0, -0.08112725, 0]` m.
Closure is the first command below zero gripper degrees after opening above
four degrees, while the tool is within 30 mm of the pawn. Actual jaw position
alone is unsuitable: contact can stop the jaw before the commanded position.

| Trace | Tool offset from pawn at closure (world X, Y, Z), mm | Bounded fit error, mm |
| --- | --- | --- |
| Successful recorded replay | 0.30, -0.35, 3.32 | Reference |
| ACT seed 11 | 5.28, 8.64, 1.23 | 1.65 |
| ACT seed 23 | See generated report; similar 5–6 mm X, 8–9 mm Y offset | 1.98 |
| ACT seed 37 | See generated report; similar 5–6 mm X, 8–9 mm Y offset | 2.02 |

Recorded replay offsets use the measured moving pawn centre. ACT trace offsets
use the fixed source-square centre because those logs did not preserve the
per-step pawn centre; these failed attempts moved the pawn only slightly.
The offsets are therefore approximate, not contact measurements. The replay
also has a different wrist-roll angle; this alone does not establish a map
error, particularly for a round pawn.

The fit changes the first five joint angles by at most two degrees to match
the successful closure's relative tool position. It is a kinematic feasibility
check only: it does not model collisions, contact, approach orientation or
motor tracking, and its corrections have not been executed. It supports
testing small residual corrections before changing the calibration or bases.
