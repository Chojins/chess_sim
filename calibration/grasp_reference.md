# Current grasp reference

Configuration is in `grasp_geometry.py`, version
`finger_setback7mm_piece_heights_v1`.

The finger front at zero hinge angle is local Y=-82.12725 mm. The requested
7 mm setback places the grasp reference at (0, -75.12725, 0) mm in the
gripper frame, centred on its axis. This is a fixed axial point referenced
to the zero-angle finger front, not a moving midpoint of V contact surfaces;
opening the fingers can change its distance to the actual front faces.

Requested heights above each piece's bottom surface:

| Piece | Height (mm) |
|---|---:|
| Pawn | 17 |
| Knight | 17 |
| Rook | 19 |
| Bishop | 27 |
| King | 32 |
| Queen | 30 |

Targets start on the XY centre of each upright piece's world bounds, at
bottom Z plus its specified height. They are converted to rigid-body local
coordinates with import scale accounted for, and subsequently transformed
with the piece pose. Thus lifting, rotating and parking a piece on the
lower tabletop do not change its body-relative grasp height.

Only approach reward uses these targets: victim first during a capture,
then the attacker. Placement and release scoring still use the existing
piece centres and tip-clearance reference. Actor inputs, ACT weights and
physical collision meshes are unchanged. Opening training rejects a stale
server reference or absent piece grasp targets; restart the server before
the next run. No new run was launched for this change.

`grasp_reference_audit.json` checks all 32 authored pieces, target heights
within mesh bounds and world/local coordinate round trips. Unit tests
cover the 7 mm offset, six heights on board/table surfaces and capture-phase
target selection. These checks validate the requested reference coordinates,
not collision-free approaches or successful grasps.

Previous experiment: `wrist4_axis_132mm_v1` used local Y=-71.9 mm and
piece bounding-box centres. That configuration and its saved results in
`outputs/opening-rl-grasp-v3` remain historical.
