# Chess simulation cameras

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
