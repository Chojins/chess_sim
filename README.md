# Chess simulation cameras

The [RL bring-up guide](RL_APPROACH.md) covers the Gymnasium environment,
ACT execution-window evaluation, reward checks and the bounded residual SAC pilot.
The learner has its own environment (`setup_rl_env.sh`, `run_rl_python.sh`);
see the guide for validation results and launch commands.

## Current baseline and next direction

The v29 hybrid controller completed 80/80 trials across four tested blue-robot
opening moves. Its capture supervisor adds waypoint pauses and an intermediate
home return. This establishes task completion for those cases; motion quality
still needs improvement before large-scale collection for ACT training.

See the [v29 baseline record](docs/baselines/v29.md) for the exact qualification
report, recordings, model locations and scope. The next direction is a smoother,
goal-conditioned teacher followed by RGB policy distillation. That training
pipeline has not been implemented yet.

Reusable simulation, training, evaluation and export tools remain at the project
root. Superseded experiment launchers are in [scripts/legacy](scripts/legacy/README.md).
They preserve the old experiments and are not the recommended next training run.
Local recordings, checkpoints and Python environments are excluded from Git;
committing this repository does not back them up.

## Launch the camera server

Launch the existing chess USD and stream both SO-100 wrist cameras:

```bash
./run_cameras.sh
```

Open <http://127.0.0.1:8080/> for a side-by-side preview. The Isaac window
also opens. For capture without the Isaac window:

```bash
./run_cameras.sh --headless
```

Close Isaac or press Ctrl+C in the launching terminal to stop. First startup
may take a few minutes while shaders compile. Wait for `FIRST_FRAMES` in the
terminal before connecting a client. Subsequent starts use the shader cache.

The default is 640×480 at a target of 15 paired captures/second. Physics is
**paused**: this first milestone verifies rendering and image transport, not
robot dynamics or ACT execution. Both cameras are captured in one render step.
The authored camera pose/lens settings are used unchanged.

## Camera outputs

| URL | Output |
| --- | --- |
| `/blue.mjpg` | Blue wrist MJPEG stream, readable by OpenCV |
| `/red.mjpg` | Red wrist MJPEG stream, readable by OpenCV |
| `/blue.jpg`, `/red.jpg` | Latest JPEG preview images |
| `/frames.npz` | Lossless RGB arrays `blue`, `red`, plus `frame_id` and `simulation_time` from one capture |
| `/stats` | Capture count, missing pairs, measured rate and image statistics |

HTTP binds to localhost. The JPEG streams are for preview/compatibility;
prefer `/frames.npz` for policy inputs to avoid JPEG artifacts and independent
stream buffering. It is a simple local bring-up transport, not yet a
high-throughput RL interface. `simulation_time` stays at zero while paused.
`frame_id` increments per render attempt. Identical scene content is expected
with paused physics; the missing-pair counter validates buffers, not motion.

Read paired frames in the existing chess vision environment:

```bash
/home/jacob/isaac_venvs/chess_vision/bin/python camera_client.py
```

Or use the adapter in Python (put this repo on that interpreter's import path):

```python
from camera_client import PairedCameraClient

client = PairedCameraClient()
metadata, frames = client.read(color_order="bgr")
blue_bgr = frames["blue"]   # input to chess_vision/OpenCV
red_bgr = frames["red"]
# client.read(color_order="rgb") supplies RGB for policy preprocessing.
```

Existing code using `cv2.VideoCapture` can consume
`cv2.VideoCapture("http://127.0.0.1:8080/blue.mjpg")` and the corresponding red
URL. OpenCV decodes these as BGR. The old highlighter's hard-coded physical
device numbers and virtual-camera setup are not needed for this transport.

The existing vision module maps blue to White and red to Black. Confirm this
against the trained checkpoint before running ACT. Do not reuse real-camera
board transforms blindly: simulated camera calibration and board projection
need validation, especially once the wrists move. No move overlay is applied
by the camera server.

## Empty-board calibration and move overlays

The saved `calibration/sim_board.json` maps both current camera views to the
same board coordinates (blue = White, red = Black). It is a **2-D homography
for these fixed poses**, not a camera-intrinsics or robot-pose calibration.
Recalibrate after moving either wrist, moving the board, or changing lenses
or resolution. Automatic ordering assumes the current upright wrist views.

To regenerate, stop the existing server, then launch:

```bash
./run_cameras.sh --capture-empty-board --output outputs/calibration
```

The server temporarily hides `/World/chess_pieces_blue` and
`/World/chess_pieces_red` in its unsaved session layer, captures `blue-empty.png`
and `red-empty.png`, and restores their previous visibility before streaming.
It also captures the populated views as `blue.png` and `red.png`. Wait for
`FIRST_FRAMES`, then run in another terminal:

```bash
/home/jacob/isaac_venvs/chess_vision/bin/python calibrate_board.py
/home/jacob/isaac_venvs/chess_vision/bin/python camera_client.py --move e2e4
```

Calibration fits all 49 inner corners and saves labelled grid images and
populated-board `e2e4` previews under `outputs/calibration/`. The client uses
the existing `chess_vision.highlight_chess_move` function with real-camera
undistortion disabled and explicit White/Black orientation. The raw server
feeds stay raw. Real-camera transforms in the vision repo are untouched.

## Diagnostics

```bash
# Short test of one camera.
./run_cameras.sh --headless --camera blue=/World/so100_blue/gripper/Camera --frames 150 --port 0

# About five minutes at the default 15 FPS, if the machine keeps up.
./run_cameras.sh --headless --frames 4500 --output outputs/stability

# Check the lossless transport from a separate terminal.
/home/jacob/isaac_venvs/chess_vision/bin/python camera_client.py --headless --frames 100 --output outputs/client
```

Every run saves first-frame PNGs, `stage_inventory.json` and `stats.json` in
`outputs/<timestamp>/` (or `--output`). The inventory records camera lenses,
articulation roots and loaded asset layers. Statistics update every ten
seconds; measured rate includes rendering, image encoding and pacing.

`--resolution WIDTH HEIGHT`, `--fps FPS`, `--port PORT`, and repeated
`--camera NAME=/prim/path` options are available. Only run one server per port.
`./run_cameras.sh --help` works without starting Isaac.

## Replay an old recorded episode

The local `Chojins/chess_game_001_blue_stereo` episode 0 contains 368 frames
at 30 Hz. Its videos confirm `laptop` = red's stationary camera (highlighted)
and `phone` = blue's moving wrist camera (raw). The active robot returns home
so that its camera can become the fixed view on the other robot's turn.

Export only the joint data using the existing LeRobot environment:

```bash
/home/jacob/miniconda3/envs/lerobot/bin/python prepare_episode.py
```

After stopping any existing server, replay the measured poses visually:

```bash
./run_cameras.sh --replay-episode outputs/replay/episode_000000.npz \
  --fps 30 --replay-loop --output outputs/replay/live
```

This default **kinematic replay** moves blue's links using the USD joint
frames, leaves red fixed, and does not move pieces or simulate grasping.
Without `--replay-loop`, it plays once and holds the final pose. Frames are
never skipped to match wall time; if rendering runs at 15 FPS, the 30 Hz
recording plays at roughly half speed. Dataset time is in `/stats` under
`replay.episode_time`; physics time remains zero for this mode.

The provisional mapping in `calibration/legacy_blue_joint_map.json` uses:

```text
sim_degrees = sim_home_degrees + scale * (legacy_values - legacy_home)
```

The user's old home preset is aligned to the approximate saved USD home.
Direction signs were visually checked against the recorded camera motion.
The gripper's old linear range converts using its 1323-tick calibration span,
but the assumed 1:1 motor/jaw ratio still requires physical measurement.
This is **not a verified hardware calibration**. Do not apply it blindly to
red, other recordings, or current LeRobot joint conventions.

The [home alignment comparison](calibration/legacy_blue_alignment.md) lists
episode 0's recorded start/end angles alongside the exact saved USD pose,
and measures hypothetical endpoint-based corrections with the bases fixed.
The current mapping remains provisional; endpoint matching alone does not
establish the physical joint zeros.

### Candidate correction for the one-square reach error

`calibration/legacy_blue_joint_map_board_fit.json` is a separate candidate
fitted to episode 0's pawn pickup and release, with both bases and the board
fixed. Relative to the original map, it changes shoulder lift by +0.885 degrees,
elbow flex by -1.810 degrees, and wrist flex by -6.910 degrees. Pan, roll,
gripper, signs and scales are unchanged. To inspect it:

```bash
./run_cameras.sh --replay-episode outputs/replay/episode_000000.npz \
  --joint-map calibration/legacy_blue_joint_map_board_fit.json \
  --replay-loop --fps 30 --output outputs/replay/board-fit-live
```

The geometric fingertip proxy's board-plane error decreases from 22.8/22.5 mm
at pickup/release to 1.54/1.65 mm. Nearby frames excluded from fitting are
within 2.8 mm; these are checks within the same episode, not independent
validation. All 368 measured states and commanded actions satisfy the USD
joint limits with this candidate, including the previously excessive wrist
targets. No trajectory values are clipped.

The fit uses the centre between the distal closed-jaw tips as an approximate
tool point. It targets the pawn's board coordinates and preserves the original
replay's estimated tip height at frames 185 and 240; absolute grasp height and
gripper aperture are still uncalibrated. Tool pitch also changes by about 7.8
degrees. This candidate needs more poses/episodes and contact validation before
ACT. The kinematic replay does not pick up pieces. Red's fixed-camera board
homography remains applicable; blue's original home homography does not apply
to the changed home pose.

Reproduce the offline fit with `./run_joint_fit.sh`. This launcher uses the
existing Isaac 5.0 Python 3.11 USD library without launching the simulator.
Fit constraints and selected frames are in
`calibration/episode_000000_contacts.json`; the full report is in
`outputs/replay/board-fit/fit_report.json`. The original map remains the default
unless the candidate is explicitly selected with `--joint-map`.

The mapper is reversible. It rejects joint-limit violations without clipping
or wrapping, and uses radians only at the PhysX articulation API boundary.
Every replay writes `replay_audit.json` and camera snapshots at selected
episode frames. The audit also compares forward kinematics against the saved
USD link transforms before applying any replay edits.

For a bounded motor-drive tracking test:

```bash
./run_cameras.sh --headless --replay-episode outputs/replay/episode_000000.npz \
  --replay-mode physics --frames 368 --fps 30 --output outputs/replay/physics
```

This uses PhysX at 120 Hz and position targets at 30 Hz, starting at the first
recorded pose. Blue uses temporary stiffness/damping 20/1; red uses 1000/20
to hold the fixed camera. These session-only gains are bring-up settings,
not identified servo dynamics. The original USD velocity and effort limits
are retained. `physics_report.json` reports target tracking and red-arm drift;
`simulation_time` uses the physics world's clock. No successful chess move or
real-to-sim calibration is implied by a completed tracking test.

`--replay-source states` is the default: it follows the recorded measured
trajectory. `--replay-source actions` tests the commanded trajectory instead.
With the original home-alignment map, episode 0's actions exceed the wrist-flex
upper limit by up to 1.05 degrees in five frames and are rejected. The separate
board-fit candidate satisfies those limits. Geometric and contact validation
are still required before connecting ACT; do not silently clip training data.

To overlay the move on the stationary red view only:

```bash
/home/jacob/isaac_venvs/chess_vision/bin/python camera_client.py \
  --move e2e4 --overlay-camera red
```

The saved fixed-pose board homography is invalid for the moving blue wrist.
Revalidate red's board calibration if its pose drifts under physics.

## Contact physics, synchronous control and ACT

The original pieces have no rigid-body/collision setup and the robot jaw
collision groups are empty. `--contacts` supplies temporary physical geometry:
5 g pieces with convex hulls, decomposed jaw meshes, hulls for robot housings,
a flat board collider, friction 1.0, and 0.5 mm contact offsets. Robot self
collision is disabled because the housing hulls and jaw gears overlap; dedicated
collision meshes are needed before relying on self-contact. These are assumed
simulation parameters, not measured hardware dynamics. The USD files are not saved.

With contacts enabled, arm gains are 100/4, the fixed red arm uses 1000/20,
and the gripper uses 2/0.2 (stiffness/damping), configurable with
`--gripper-stiffness` (damping is one tenth of stiffness). `--contact-friction`
sets the shared material friction. Physics runs at
240 Hz. Pieces are never attached to the gripper by a scripted constraint.
`physics_samples.jsonl` logs measured trajectories and piece movement during
physics replay. A camera replay completing does not establish a successful grasp.

The modified gripper visuals come from `Moving_Jaw_1.stl`, `Moving_Jaw_2.stl`
and `Fixed_Jaw_mod_m.stl` in `SO100/assets`, not the standard single-moving-jaw
gripper. A contact-setup bug was found on 2026-09-15: blue jaw 1's visual group
is a USD instance, and ordinary traversal skipped its mesh. Contact setup now
expands robot visual instances in the session layer, checks that both jaws on
both robots have collision meshes, and records every collider path in the audit.
Earlier `contact-test*`, `contact-height*`, `contact-centered`,
`control-replay-compliant` and `act-integration` results used an incomplete
collider set; they remain useful for transport/recording checks but cannot
validate gripper dynamics. Their parameter trials should not be promoted as
calibration results. Runs beginning `both-jaws` use the corrected colliders.

With corrected colliders, `outputs/replay/both-jaws-neck/` completed the recorded
e2e4 move at half speed: 22.82 mm lift, 1.35 mm final position error, upright
placement and no other pieces displaced over 3 mm. The successful candidate is
saved separately as `calibration/legacy_blue_joint_map_physics.json`. Its higher
grasp pose is specific to this initial validation: the knight episode still
exceeds its wrist limits, so it is not a general calibration. See the
[gripper audit](calibration/gripper_geometry_audit.md) for mesh comparisons,
collision corrections. A provisional magnetic board model is now available below.

Start the synchronous control server using this e2e4 physics candidate:

```bash
./run_cameras.sh --replay-episode outputs/replay/episode_000000.npz \
  --joint-map calibration/legacy_blue_joint_map_physics.json \
  --replay-mode physics --contacts --control --fps 30 --output outputs/control
```

Add `--magnets` to reproduce the board's self-centring attraction:

```bash
./run_cameras.sh --replay-episode outputs/replay/episode_000000.npz \
  --joint-map calibration/legacy_blue_joint_map_physics.json \
  --replay-mode physics --contacts --magnets --control --fps 30 \
  --output outputs/control-magnets
```

The model applies attraction at each piece's base toward all 64 square centres,
with downward retention that fades as a piece is lifted. `--magnet-strength 1.0`
scales attraction; `--board-friction 0.12` controls sliding separately from
finger contact friction. A one-second physical settling phase centres initially
misaligned pieces before defining reset poses and episode measurements; the
startup displacements are recorded in the audit. There is no scripted snapping.
These settings are provisional, pending comparison with the real magnets.
See [magnetic model and calibration](calibration/board_magnets.md).

Run the numerical checks with `python -m unittest test_board_magnets` in a
NumPy environment. Run the headless centring and grasp checks with Isaac:

```bash
/home/jacob/isaacsim/_build/linux-x86_64/release/python.sh verify_magnets.py
```

Check `outputs/magnets/verification/magnet_report.json` for `passed: true`;
this Isaac build can report exit code zero even after a Python exception.
Stop the camera server before starting another Isaac instance on this GPU.

`--control` uses the episode's first pose for reset and waits for requests.
Physics stays paused between commands. `POST /step` accepts JSON
`{"frame_id": 0, "action": [six legacy values]}`; `POST /reset` accepts
`{"frame_id": 0}`. Use the actual latest frame ID. Each step advances exactly
1/30 second, then publishes measured state and both cameras together.
`/frames.npz` additionally contains `observation_json`; `PairedCameraClient`
decodes it into `metadata["observation"]`. Invalid and out-of-limit actions,
stale frame IDs and concurrent queued commands are rejected; actions are not
silently clipped. Reset teleports the simulated robots and pieces to their
initial poses, zeros velocities, settles them, and resets the episode clock
without rebuilding camera graphs. This is a simulation reset, not a learned
return-home action.

Run the checkpoint in the existing Python 3.12 environment:

```bash
/home/jacob/miniconda3/envs/lerobot312/bin/python act_client.py \
  --device cpu --steps 400 --episodes 2 --move e2e4 --output outputs/act-two-episodes
```

The adapter loads the local `checkpoints/last/pretrained_model`, checks all
network weight keys strictly, and uses its embedded normalization statistics.
It supplies legacy measured joint state, red's highlighted image as `laptop`,
and blue's raw wrist image as `phone`. RGB images are converted to CHW floats
in [0,1] before applying the saved mean/std. ACT's original 100-action queue
is preserved unless `--n-action-steps` is explicitly set. It is reset between
episodes. The lightweight overlay was verified pixel-for-pixel against
`chess_vision` for three moves; it avoids that module's unused 3-D dependencies.
After the arm departs home by more than 15 degrees, ten consecutive steps
within 5 degrees of its initial arm pose end the rollout. This prevents a
continuously highlighted move from starting another attempt after returning
home. `--steps` remains an upper bound. Returning home is distinct from grasp
success; the result records the termination reason and task measurements.

CPU inference is intentional here: this machine's Python 3.12 PyTorch build
uses CUDA 13, which the installed driver does not support. A first CPU action
chunk took about 0.21 seconds in the smoke test. No driver, environment,
checkpoint or dataset migration was needed. Rendering/transport need not run
in real time: physics advances only when a command is accepted.

Each run writes diagnostic simulated episodes: input images (including exactly
the highlighted policy image), observed legacy state, commanded actions,
next state, frame IDs and simulation timestamps, plus terminal images and a
manifest with checkpoint hash and mapping. This is `chess_sim_transitions_v1`,
not a LeRobot training dataset. Failed and stopped runs retain their failure
reason and must not be treated as expert demonstrations. The current reset
board, task measurement and ACT runner support blue `e2e4` only.

Useful checks, with the control server running:

```bash
/home/jacob/miniconda3/envs/lerobot312/bin/python verify_control.py
/home/jacob/miniconda3/envs/lerobot312/bin/python replay_control.py \
  --repeat 2 --output outputs/control-replay
```

The first verifies paired observations, invalid/stale rejection, fixed stepping,
idle pause and reset repeatability. The second drives the recorded episode via
the same interface as ACT; `--repeat 2` gives half-speed physical motion.
Both checks reset the simulation, so run one controller at a time.

Independent recorded moves were checked without refitting (`g1f3`, episode 1;
`d2d4`, episode 2). Board-fit estimated pickup/release errors were 8.11/8.00 mm
and 1.49/7.51 mm, compared with 30.95/30.05 mm and 21.60/23.03 mm for the original
home map. Both board-fit trajectories satisfy state and action limits. These
are geometric checks against video-selected contact frames, not successful
physical replays with reconstructed game positions. Reproduce with
`./run_usd_python.sh validate_joint_map.py` after exporting episodes 1 and 2.
The contact-height trial violates limits in episode 1 and is not a general
replacement for the board-fit map.

## Runtime and assets

The launcher defaults to the existing installation at
`~/isaacsim/_build/linux-x86_64/release` (5.0.0-rc.23 on this workstation).
To select another installation:

```bash
ISAAC_SIM_PATH=/path/to/isaacsim ./run_cameras.sh --headless
```

The launcher isolates Isaac from user Python/ROS packages. Run the client in
the vision/policy environment with NumPy and OpenCV installed. No additional
Isaac packages or driver changes are required by these scripts.

The original USD files are never saved. Capture-specific edits go into the
session layer, including disabling four old saved Replicator render products
and creating fresh products. This prevents reusing old capture settings.

The stage currently references pieces and pedestals in the sibling
`../LeRobot-S0-100-Models/` directory. Keep that directory available; this repo
is not yet a self-contained asset bundle.

## Verified on this workstation

On 2026-09-14, the existing 5.0.0-rc.23 build and NVIDIA driver 570.169 on an
RTX 5070 Ti captured both 640×480 cameras for over six minutes. The last
periodic sample before manually stopping the test recorded 5,702 complete
pairs, zero missing pairs and 14.56 paired captures/second. Logs and sample
images are in `outputs/dual-camera/` and `outputs/dual-camera.log`.

The existing `chess_vision` Python environment successfully received 100
lossless pairs, and OpenCV decoded both MJPEG streams concurrently. The
existing 7×7 board-corner detector did not find the full pattern in either
populated-board view. Temporarily hiding the pieces subsequently enabled all
49 corners to be detected in both views. The fitted 2-D homographies have
0.131 px (blue) and 0.099 px (red) RMS reprojection error; grid alignment and
`e2e4` overlays were visually checked with pieces restored. Physics, joint
control and ACT execution were not part of that initial camera check.

Episode 0 subsequently completed kinematic and PhysX state replay with both
cameras. The kinematic audit found no joint-limit violations and reproduced
the authored USD transforms to a maximum matrix-element error of 6.1e-7.
The final physics run captured all 368 pairs with zero missing pairs at
13.74 pairs/s. Joint tracking RMS was 0.22–2.95 degrees, with an 11.37-degree
shoulder peak; the stationary red arm's maximum joint drift was 0.0255 degrees.
Recorded shoulder/elbow speeds reach about 90 degrees/s, while the USD limits
them to 57.3 degrees/s; this contributes to tracking lag. Rendering currently
runs slower than the recorded 30 Hz, although physics advances at its configured
time step. The report is under `outputs/replay/physics-final/`; kinematic
comparison images are under `outputs/replay/run1/`.

On 2026-09-15, after fixing the skipped instanced jaw collider, a half-speed
recorded replay lifted the pawn 22.82 mm and placed it upright 1.35 mm from e4's
target centre. Both camera streams completed the run. See
`outputs/replay/both-jaws-neck/physics_report.json`.

Two subsequent ACT runs using the corrected contacts and
`legacy_blue_joint_map_physics.json` terminated on return home after 282 and
284 steps. They lifted the pawn 22.39/22.97 mm and placed it upright on e4,
with 4.39/5.36 mm centre error and no other pieces displaced over 3 mm.
Return-home maximum arm errors were 3.27/3.14 degrees. The first passed the
predefined success test (5 mm placement, upright, lift over 5 mm, no disturbed
pieces, home within 5 degrees); the second narrowly failed placement tolerance.
This is a two-run bring-up result, not a reliability estimate. Results and
566 recorded transitions are in `outputs/act-both-jaws/`.

The synchronous control integration check passed with corrected colliders:
paired observations, finite/limit/stale action rejection, 1/30-second stepping,
paused physics while idle, and repeatable robot/piece reset. Recordings can be
checked with `python verify_recordings.py outputs/act-both-jaws` in the Python
3.12 environment; it verifies frame/state chaining, timestamps, image shapes
and terminal state consistency.

Magnetic board verification (`outputs/magnets/verification-final/`) passed four
off-centre releases: +8/-8 mm along one board axis, +10 mm along the other, and
a 6 mm + 6 mm diagonal. Final distances from the CAD square centre were
0.11–0.71 mm; the disabled-magnet comparison remained 8 mm off centre. The
recorded half-speed grasp lifted the pawn 22.60 mm and placed it upright about
0.44 mm from the CAD e4 centre, with no other pieces disturbed after initial
settling. The older replay target metric uses the initial pawn centre plus
45 mm and reports a slightly different error. This tests the provisional
simulation model, not the strength of the real magnets.

With magnets enabled, the camera/control checks passed again, including paused
idle physics and repeatable resets. Two ACT runs were recorded under
`outputs/act-magnets/`: the first lifted the pawn 22.80 mm, placed it about
0.48 mm from the CAD e4 centre and returned home after 286 steps. The second
returned home after 288 steps but failed to lift the pawn (maximum lift
0.38 mm); it remained on e2. Thus 1/2 runs passed the unchanged task criteria.
Both left other pieces undisturbed. All 574 recorded transitions passed frame,
state, timestamp and image consistency checks. The magnetic centring feature
works in these tests; ACT grasp reliability and real magnet calibration remain
separate outstanding work.
