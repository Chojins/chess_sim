# Robot-relative capture areas

Jacob's teleop convention is to remove the opponent's piece first, release it
on the tabletop to the acting robot's own right, then pick and place the
attacking piece. The fixed camera's image-right direction is not the reference.

The provisional areas extend 120 mm outward from the corresponding board-frame
edge and 150 mm along the board, centred alongside it. Blue's area is world +X;
red's is world -X. These are opposite sides for the facing robots. The tabletop
uses the scene's existing ground plane at Z=0; the square tops are about 14.2 mm
higher. Dimensions are Jacob's approximate measurements, not a calibrated fit.

`BoardLayout` computes both rectangles from the actual CAD board-frame bounds;
the resolved world coordinates and the acting robot's selected zone are saved
in every legal-task reset's `task_geometry`. Captured pieces already absent
from a position are parked in their captor's zone on a spaced grid.

For a new capture, the correct victim must be lifted for at least three frames,
released upright within that rectangle at tabletop resting height, remain
stable for ten frames, and be clear of the tool proxy. This earns a one-time
capture-release bonus. Moving the attacker more than 3 mm before that release
is a capture-order failure. Completion additionally requires the attacker at
its target and the robot home, while the captured piece remains released in
the correct area. Moving an unrelated piece is still a disturbance failure.

Live validation passed on 2026-09-22 after GPU recovery. Blue's resolved zone is
X=[0.09955, 0.21955] m, Y=[-0.26299, -0.11299] m, with tabletop Z=0. ACT correctly
removed and released the red pawn there during the e4xd5 probe; it then missed
the blue pawn's grasp, so the complete capture was not successful. The episode
is recorded in `outputs/opening-baseline/scandinavian_capture`.

Red's zone is defined, but red robot control still needs its own validated
joint map. Castling and promotion execution are explicitly rejected until
their multi-piece goals and reserve-piece handling are implemented.

Under `legal_move_v2`, unrelated board pieces may be nudged within their
original square (piece centre test); crossing its edge or tipping beyond
30 degrees fails. Parked pieces may shift within their original capture zone.
The capture-order requirement above is unchanged.


## Current capture ordering: legal_move_v3

This supersedes the historical 3 mm attacker-motion rule above. Before
the captured piece has been released stably in the acting robot's RHS
zone, the attacker's centre must stay inside its original CAD square in
the board plane. In-square nudges (and small vertical bumps) are allowed.
The boundary is fixed to the square centre, not the jittered reset pose;
half the square pitch is approximately 11.25 mm. This is a geometric
approximation to remaining in the original magnet's area, not a magnetic
force-basin measurement or a whole-piece-footprint check. Victim release
and unrelated-piece protection criteria are unchanged.

All six learned capture-order failures in opening-heldout-v5 retained
the attacker inside its square (about 3–5 mm lateral movement). See
outputs/capture-order-audit.json. The trials stopped at that point, so
we cannot relabel them as successful captures without running them again.


## Acceptable removal versus preferred storage: legal_move_v4

User preference: fallen captured pieces make board reset harder, but should
not invalidate an otherwise completed move. Victim removal now permits a
non-upright piece in the capture zone if it has been lifted, its centre is
above the tabletop and below upright rest height + 3 mm, it moves less than
0.5 mm per frame, and the tool is over 35 mm away for ten consecutive frames.
This height band is a near-table proxy, not a measured contact-force test.
Pieces outside the zone, below the tabletop, held high, moving or too close
to the tool remain incomplete. On-board placement still requires uprightness.

A one-time +0.25 rewards accepted removal. A further +0.75 is awarded only
on successful completion of the entire move if the captured piece is still
upright, at its upright rest height, and stable for ten frames. Thus upright
capture retains the former total +1 quality reward, while tipped removal
earns less. Delaying the upright bonus prevents rewarding an initially
upright release that subsequently falls over. Existing placement shaping
still targets the upright resting centre.

Capture telemetry reports removal_quality (upright/non_upright/incomplete)
and upright_stable separately from released_stable. Offline rescoring of
both seed-903 failures recognises non-upright release at step 222, before
the former failure points (297/298). It does not establish completed-move
success, since the saved trajectories stop early. See
outputs/capture-release-rescore-v4.json. No new run has been launched.
