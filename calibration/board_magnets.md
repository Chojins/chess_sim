# Magnetic board approximation

`--magnets` enables forces for all 32 pieces and 64 square centres in the
session layer. The original USD is not saved. It works in physics replay and
synchronous ACT control; it requires `--contacts`.

The source CAD README at `../LeRobot-S0-100-Models/README.md` specifies magnets
5.04 mm in diameter and 1.6 mm high. The imported pawn and board square contain
approximately 2.6 mm radius recesses. Magnet seating depth and strength have
not been measured on the assembled real board.

`board_magnets.py` uses an effective attraction potential for each square:

```
U = -E * exp(-r² / (2 * width²)) * exp(-clearance / height_decay)
```

All square contributions are summed. Their negative spatial gradients give
horizontal centring and downward attraction. A local viscous damping term
reduces oscillation. Forces are applied at each piece's circular foot centre
on every physics substep, producing contact dynamics and torque naturally.
The sites follow the rigid body poses, including after reset. Pieces are never
teleported, welded, or constrained to a chosen destination by the magnet model.
At a boundary the neighbouring squares compete; this does not guarantee the
intended square if the robot releases across the boundary.

These are effective sites at the board surface and piece foot, not literal
magnet centre coordinates. The height decay absorbs unknown seating depth.
This model does not solve magnetic fields, represent polarity, add dipole
alignment torque, or model attraction between pieces. Damping, contact friction
and the existing 5 g piece masses are provisional simulation parameters.

Defaults are E = 0.00056 J, width = 4 mm, height decay = 4 mm, damping =
0.3 N·s/m. An isolated centred piece experiences 0.14 N downward attraction at
the board surface. At 20 mm clearance the attraction falls to about 0.67%.
`--magnet-strength` scales E. Width, decay and damping are in `MagnetSettings`.

With magnets enabled, `--board-friction` defaults to 0.12 while
`--contact-friction` remains 1.0 for finger/piece contact. The board material uses
PhysX's `min` friction combine mode so its lower coefficient applies to
piece/board contact. See [NVIDIA's material combine rules](https://nvidia-omniverse.github.io/PhysX/physx/5.1.0/_build/physx/latest/struct_px_combine_mode.html).
Runs without `--magnets` retain the previous friction behaviour.

The settings and effective magnet positions are saved in `replay_audit.json`
and the control observation's `physics_settings.magnets`, which is included in
ACT recording manifests. Recordings label the model as uncalibrated.

The initial board settles for one simulated second under magnetic forces before
piece reset poses and episode baselines are recorded. Some authored pieces are
about 3 mm off centre; their physical startup movements are reported separately
under `magnets.initial_settling`. Subsequent episode disturbance measurements
compare against this settled board. Reset restores those settled starting poses.

To calibrate against hardware, measure centring distance and settling time for
a piece released a few millimetres off centre, plus the approximate force
needed to lift it. Fit lateral width, energy, height decay and board friction
to those measurements before using simulated grasp outcomes as real-world
success labels.
