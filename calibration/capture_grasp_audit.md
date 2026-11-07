# Capture grasp audit — 2026-10-04

Recorded misses show two different approach errors: closing too high above
the pawn and closing toward the robot base instead of around its centre.
The requested pawn grasp target remains 17 mm above its bottom, and the
fixed tool reference remains 7 mm behind the zero-angle finger front.

The finger surface midpoint is already close to that fixed reference at
closure. Changing the reference alone would not explain the large misses.

| Recorded trial | Closing sample | Fixed reference height error | Front/back error | Best fixed-point fit within ±2° | Within ±4° | Within ±6° |
|---|---:|---:|---:|---:|---:|---:|
| Successful e2e4, v6 comparison 1102 | 126 | +2.86 mm | +4.24 mm | <0.01 mm | <0.01 mm | <0.01 mm |
| Successful victim pickup, v7 before 1202 | 130 | +4.64 mm | −4.71 mm | 0.49 mm | <0.01 mm | <0.01 mm |
| High victim miss, v7 before 1203 | 138 | +15.63 mm | −1.46 mm | 12.63 mm | 4.98 mm | 0.18 mm |
| High victim miss, v7 final 1201 | 134 | +12.23 mm | −2.36 mm | 2.10 mm | <0.01 mm | <0.01 mm |
| Backward victim miss, v6 learned 1102 | 111 | +4.23 mm | −11.67 mm | 2.06 mm | <0.01 mm | <0.01 mm |

Negative front/back means toward the blue robot base. Fit errors are 3D
distances, not only height errors. Fits use the frozen ACT command as the
centre of each residual range, clipped to existing USD joint limits.

For the final 1201 miss, measured shoulder/elbow lag the commanded pose
by about 1.76°/1.21° at the selected sample. The measured jaw is already
at −1.78° while its command is +2.25°. That makes a fit to commanded
arm positions look better than the actual observed grasp. It is not proof
that changing the joint gains will fix the miss.

The worst high miss has small arm tracking error (~0.16° maximum), so
its large height error exists in the approach command itself. The ±2°
residual has insufficient static position authority at that recorded pose.
This does not prove the entire changing ACT trajectory is unreachable:
closing earlier/later or changing the preceding observations can change it.

## Finger opening and orientation

Opposing surface points were derived from the actual instantiated USD jaw
meshes by intersecting each with the user-requested axial plane and local
Z=0 at zero hinge angle. These points follow the rotating fingers. They
are not an independently measured centre of the printed V groove.

Their midpoint differs from the fixed reference by about 0.96–1.27 mm at
the selected ordinary closing samples. The final miss has a 0.51 mm
offset of the opposite sign because its measured jaw is past zero. A sweep
from −3° to 30° finds at most 2.34 mm of axial midpoint shift; at the full
45° opening it is 3.70 mm. Symmetry keeps the lateral midpoint centred.

The opening plane is close to horizontal with the finger lengths directed
downward. The successful e2e4 approach tilts 1.67° from vertical; the
selected capture poses tilt 4.30–8.78°. Fits that also favour the successful
approach tilt still leave errors in the restricted ±2° range. A position
fit therefore must not be reported as a correct finger orientation.

The geometry plot shows why a tool centre 12–16 mm too high can leave
the finger fronts at the pawn's head, and why a tool centre 12 mm toward
the base can leave the pawn outside the fingers' front/back contact region.
This is consistent with the user's video observations; it is not a measured
contact-force or collision-hull diagnosis.

## Next physical experiment

Use a short, recorded capture approach test before another long SAC run:

1. Allow a separately configured, smooth correction range during the victim
   approach; the static audit suggests testing up to ±6° for the arm joints.
   Keep existing USD limits and the current production checkpoints intact.
2. Centre the opening on the victim at its existing 17 mm target before
   closing, and check measured convergence rather than only commanded FK.
   Record both physical finger angles and the approach/lift on video.
3. Check clearance around neighbouring pieces and successful lifting.
   The larger static fits do not establish collision-free trajectories.
4. If that test passes, train with the validated correction range and a
   closing-alignment reward, while retaining actual lift/hold success.
   Merely extending the present ±2° training run would not resolve the
   observed command error at the worst recorded closing pose.

## Reproduce and inspect

Run `./run_usd_python.sh diagnose_capture_grasp.py`, then
`MPLCONFIGDIR=/tmp/chess-grasp-matplotlib python3 plot_capture_grasp_audit.py`.
Neither command starts Isaac rendering or advances physics. Source USD,
joint maps, reward configuration, ACT and residual checkpoints are untouched.

Artifacts:

- [Closing geometry comparison](../outputs/capture-grasp-audit-v7/closing_poses.png)
- [Fit results and surface anchors](../outputs/capture-grasp-audit-v7/report.json)
- Per-trial approach/closure/lift samples in that output directory.

FK reproduced all 250 recorded tool reference samples within 1 micrometre
(asserted by the audit). The two scripts compile and `git diff --check`
passes. Fits use SciPy local least squares, so the listed best fits are
solutions found, not certified global optima.

Jaw 2 is assumed to be the opposite of the logged jaw-1 angle, matching the
configured mimic constraint; its independent measured angle was not logged.
Full piece quaternions were also not logged, so the plot translates the
authored upright pawn shape to each measured centre. All plotted samples
have upright cosine above 0.999. PhysX convex decompositions and real contact
forces are outside this offline audit.
