"""Capture paired wrist-camera frames from a USD stage using Isaac Sim.

Run with ./run_cameras.sh. Only the main thread accesses Isaac/USD; HTTP
threads read completed frame pairs. The source USD is never saved.
"""

import argparse
import io
import json
import math
from pathlib import Path
import threading
import queue
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class FrameStore:
    def __init__(self):
        self.condition = threading.Condition()
        self.frame_id = -1
        self.images = {}
        self.pair = b""
        self.stats = {}
        self.closed = False
        self.control_enabled = False
        self.commands = queue.Queue(maxsize=1)

    def publish(self, frame_id, images, pair, stats):
        with self.condition:
            self.frame_id, self.images, self.pair, self.stats = frame_id, images, pair, stats
            self.condition.notify_all()


def make_handler(store):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_POST(self):
            if not store.control_enabled or self.path not in ('/step', '/reset'):
                self.send_error(404)
                return
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length <= 4096 or self.headers.get('Content-Type') != 'application/json':
                    raise ValueError('Expected a small JSON body')
                payload = json.loads(self.rfile.read(length))
                if not isinstance(payload, dict) or type(payload.get('frame_id')) is not int:
                    raise ValueError('frame_id is required')
                command = {'path': self.path, 'payload': payload, 'done': threading.Event(),
                           'expires': time.monotonic() + 30}
                store.commands.put_nowait(command)
            except (ValueError, queue.Full) as error:
                self.respond(json.dumps({'error': str(error)}).encode(), 'application/json', 400)
                return
            if not command['done'].wait(30):
                self.respond(b'{"error":"Control timed out; read current state before retrying"}', 'application/json', 504)
                return
            self.respond(json.dumps(command['result']).encode(), 'application/json', command.get('status', 200))

        def do_GET(self):
            path = self.path.split("?", 1)[0]
            if path == "/":
                body = ("<!doctype html><title>Chess sim cameras</title>"
                        "<style>body{background:#15171b;color:#eee;font:18px sans-serif}"
                        "section{display:flex;flex-wrap:wrap;gap:20px}img{max-width:100%}</style>"
                        '<h1>Chess sim wrist cameras</h1><p id="mode">Camera checks</p>'
                        '<p id="status">Waiting for camera frames…</p><section>'
                        + "".join(f'<div><h2>{name}</h2><img src="/{name}.mjpg"></div>'
                                  for name in store.stats.get("cameras", {}))
                        + '</section><script>async function status(){try{'
                        'const s=await(await fetch("/stats")).json();'
                        'document.getElementById("mode").textContent=s.replay?.mode==="policy_control"?'
                        '"Policy control · episode "+s.replay.episode_id+" · step "+s.replay.control_step:'
                        's.replay?'
                        's.replay.mode+" replay: frame "+s.replay.episode_frame+" / "+(s.replay.total_frames-1)'
                        '+(s.replay.complete?" (complete)":""):'
                        '"Physics paused for camera checks";'
                        'document.getElementById("status").textContent=s.capture_fps===undefined?'
                        '"Cameras warming up…":s.capture_fps.toFixed(1)+" FPS · "+'
                        's.complete_pairs+" frame pairs · "+s.missing_pairs+" missing";'
                        '}catch(e){document.getElementById("status").textContent="Camera server disconnected";}}'
                        'status();setInterval(status,1000);</script>').encode()
                self.respond(body, "text/html")
                return
            with store.condition:
                if path == "/stats":
                    body, mime = json.dumps(store.stats).encode(), "application/json"
                elif path == "/frames.npz" and store.pair:
                    body, mime = store.pair, "application/octet-stream"
                elif path.endswith(".jpg") and path[1:-4] in store.images:
                    body, mime = store.images[path[1:-4]], "image/jpeg"
                else:
                    body = None
            if body is not None:
                self.respond(body, mime)
                return
            name = path[1:-5] if path.endswith(".mjpg") else ""
            if name not in store.stats.get("cameras", {}):
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            last_id = -1
            try:
                while True:
                    with store.condition:
                        store.condition.wait_for(lambda: store.closed or store.frame_id != last_id, timeout=5)
                        if store.closed:
                            return
                        if store.frame_id == last_id or name not in store.images:
                            continue
                        last_id, jpeg = store.frame_id, store.images[name]
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                     + str(len(jpeg)).encode() + b"\r\n\r\n" + jpeg + b"\r\n")
            except (BrokenPipeError, ConnectionResetError):
                pass

        def respond(self, body, mime, status=200):
            self.send_response(status)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

    return Handler


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--usd", type=Path, default=Path(__file__).parent / "Sim_Stage.usd")
    parser.add_argument("--camera", action="append", metavar="NAME=PRIM_PATH",
                        help="Repeat for selected cameras; defaults to blue and red wrists")
    parser.add_argument("--resolution", nargs=2, type=int, default=[640, 480], metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--fps", type=float, default=15)
    parser.add_argument("--frames", type=int, default=0, help="Stop after N capture attempts; 0 runs until closed")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--capture-empty-board", action="store_true",
                        help="Save empty-board calibration images, then restore pieces before streaming")
    parser.add_argument("--replay-episode", type=Path, help="Replay exported joint data; default mode is kinematic")
    parser.add_argument("--joint-map", type=Path, default=Path(__file__).parent / "calibration/legacy_blue_joint_map.json")
    parser.add_argument("--replay-source", choices=("states", "actions"), default="states")
    parser.add_argument("--replay-mode", choices=("kinematic", "physics"), default="kinematic")
    parser.add_argument("--replay-loop", action="store_true", help="Repeat kinematic replay for visual inspection")
    parser.add_argument("--contacts", action="store_true", help="Add session-only piece and robot contact physics")
    parser.add_argument("--control", action="store_true", help="Wait for synchronous local /step and /reset commands")
    parser.add_argument('--gripper-stiffness', type=float, default=2.0)
    parser.add_argument('--contact-friction', type=float, default=1.0)
    parser.add_argument('--magnets', action='store_true', help='Add provisional board/piece magnetic attraction')
    parser.add_argument('--magnet-strength', type=float, default=1.0, help='Scale magnetic potential energy')
    parser.add_argument('--board-friction', type=float, default=0.12, help='Board sliding friction with --magnets')
    parser.add_argument("--replay-speed", type=float, default=1.0, help="Physics trajectory speed; 0.5 is half speed")
    parser.add_argument("--port", type=int, default=8080, help="Local HTTP port; 0 disables HTTP")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "outputs" / time.strftime("%Y%m%d-%H%M%S"))
    args = parser.parse_args()
    if not args.usd.is_file():
        parser.error(f"USD does not exist: {args.usd}")
    if args.replay_loop and (not args.replay_episode or args.replay_mode != "kinematic"):
        parser.error("--replay-loop requires a kinematic replay")
    if args.contacts and args.replay_mode != 'physics':
        parser.error('--contacts requires --replay-mode physics')
    if args.magnets and (not args.contacts or not args.replay_episode):
        parser.error('--magnets requires --contacts and --replay-episode')
    if args.control and (not args.replay_episode or args.replay_mode != 'physics' or not args.port or args.replay_loop):
        parser.error('--control requires --replay-episode for the reset pose, --replay-mode physics, and an HTTP port')
    if args.control and args.replay_speed != 1:
        parser.error('Control steps use the dataset frequency; replay-speed must be 1')
    if not math.isfinite(args.replay_speed) or not 0.1 <= args.replay_speed <= 2:
        parser.error('replay-speed must be between 0.1 and 2')
    if not all(math.isfinite(v) and v > 0 for v in (args.gripper_stiffness, args.contact_friction,
                                                  args.magnet_strength, args.board_friction)):
        parser.error('Contact stiffness, friction and magnet strength must be finite and positive')
    if args.capture_empty_board and args.replay_episode:
        parser.error("Capture board calibration in a separate static run")
    if not math.isfinite(args.fps) or args.fps <= 0 or min(args.resolution) <= 0 or args.frames < 0:
        parser.error("fps and resolution must be positive; frames must be nonnegative")
    if not 0 <= args.port <= 65535:
        parser.error("port must be between 0 and 65535")
    args.cameras = {}
    for entry in args.camera or ["blue=/World/so100_blue/gripper/Camera", "red=/World/so100_red/gripper/Camera"]:
        name, separator, path = entry.partition("=")
        if (not separator or not name.isidentifier() or not path.startswith("/") or name in args.cameras
                or name in ("frame_id", "simulation_time")):
            parser.error(f"Expected unique NAME=/absolute/prim/path, got {entry!r}")
        args.cameras[name] = path
    return args


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": args.headless, "renderer": "RaytracedLighting",
                         "multi_gpu": False, "anti_aliasing": 2,
                         "width": args.resolution[0], "height": args.resolution[1]})
    server = None
    store = FrameStore()
    annotators, products = {}, []
    try:
        import numpy as np
        from PIL import Image
        import omni.usd
        import omni.timeline
        import omni.replicator.core as rep
        from pxr import UsdGeom, UsdRender

        context = omni.usd.get_context()
        if not context.open_stage(str(args.usd.resolve())):
            raise RuntimeError(f"Could not open {args.usd}")
        while context.get_stage_loading_status()[2] > 0:
            app.update()
        stage = context.get_stage()
        # All capture edits live in an unsaved session layer. Saved Replicator
        # products otherwise get reused, including old post-processing graphs.
        stage.SetEditTarget(stage.GetSessionLayer())
        disabled_products = []
        for prim in list(stage.Traverse()):
            if prim.IsA(UsdRender.Product) and prim.GetName().startswith("Replicator"):
                disabled_products.append(str(prim.GetPath()))
                prim.SetActive(False)
        for prim in stage.Traverse():
            if prim.IsA(UsdRender.Settings):
                relationship = UsdRender.Settings(prim).GetProductsRel()
                relationship.SetTargets([p for p in relationship.GetTargets() if str(p) not in disabled_products])
        timeline = omni.timeline.get_timeline_interface()
        timeline.stop()
        inventory = {"usd": str(args.usd.resolve()), "meters_per_unit": UsdGeom.GetStageMetersPerUnit(stage),
                     "up_axis": str(UsdGeom.GetStageUpAxis(stage)), "cameras": [], "articulations": [],
                     "disabled_saved_render_products": disabled_products,
                     "layers": sorted(layer.identifier for layer in stage.GetUsedLayers() if not layer.anonymous)}
        for prim in stage.Traverse():
            if prim.IsA(UsdGeom.Camera):
                camera = UsdGeom.Camera(prim)
                inventory["cameras"].append({"path": str(prim.GetPath()),
                    "focal_length": camera.GetFocalLengthAttr().Get(),
                    "horizontal_aperture": camera.GetHorizontalApertureAttr().Get(),
                    "vertical_aperture": camera.GetVerticalApertureAttr().Get()})
            if "PhysicsArticulationRootAPI" in prim.GetAppliedSchemas():
                inventory["articulations"].append(str(prim.GetPath()))
        (args.output / "stage_inventory.json").write_text(json.dumps(inventory, indent=2))
        print("STAGE", json.dumps(inventory), flush=True)
        # World.reset() invalidates existing camera graphs in this Kit build.
        # Initialize physics before attaching any capture annotators.
        replay = control = None
        magnets = None
        if args.magnets:
            from board_magnets import MagnetSettings
            magnets = MagnetSettings(energy_j=0.00056 * args.magnet_strength, board_friction=args.board_friction)
        if args.control:
            from sim_control import SimControl
            control = SimControl(stage, args.replay_episode, args.joint_map, args.output, contacts=args.contacts,
                                 gripper_stiffness=args.gripper_stiffness, friction=args.contact_friction, magnets=magnets)
            store.control_enabled = True
        elif args.replay_episode:
            from episode_replay import EpisodeReplay, PhysicsEpisodeReplay
            replay_class = PhysicsEpisodeReplay if args.replay_mode == "physics" else EpisodeReplay
            kwargs = {'contacts': args.contacts, 'speed': args.replay_speed, 'gripper_stiffness': args.gripper_stiffness,
                      'friction': args.contact_friction, 'magnets': magnets} if args.replay_mode == 'physics' else {}
            replay = replay_class(stage, args.replay_episode, args.joint_map, args.output, args.replay_source, **kwargs)
            print("REPLAY_READY", len(replay.targets), "frames; mode:", args.replay_mode, flush=True)
        rep.orchestrator.set_capture_on_play(False)
        for name, path in args.cameras.items():
            prim = stage.GetPrimAtPath(path)
            if not prim or not prim.IsA(UsdGeom.Camera):
                raise RuntimeError(f"Missing camera {path}; see stage_inventory.json")
            product = rep.create.render_product(path, tuple(args.resolution), force_new=True, name=f"chess_{name}")
            products.append(product)
            annotator = rep.AnnotatorRegistry.get_annotator("rgb")
            annotator.attach(product)
            annotators[name] = annotator
        store.stats = {"cameras": args.cameras, "resolution": args.resolution, "status": "warming_up"}
        if args.port:
            server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(store))
            threading.Thread(target=server.serve_forever, daemon=True).start()
            print(f"PREVIEW http://127.0.0.1:{args.port}/", flush=True)
        for _ in range(10):
            rep.orchestrator.step(rt_subframes=1, delta_time=0.0, pause_timeline=True)
        if args.capture_empty_board:
            visibility = []
            try:
                for path in ("/World/chess_pieces_blue", "/World/chess_pieces_red"):
                    prim = stage.GetPrimAtPath(path)
                    if not prim or not prim.IsA(UsdGeom.Imageable):
                        raise RuntimeError(f"Cannot hide missing piece group: {path}")
                    attr = UsdGeom.Imageable(prim).GetVisibilityAttr()
                    visibility.append((attr, attr.Get()))
                    attr.Set(UsdGeom.Tokens.invisible)
                # Let temporal lighting history settle with the pieces hidden.
                for _ in range(30):
                    rep.orchestrator.step(rt_subframes=1, delta_time=0.0, pause_timeline=True)
                for name, annotator in annotators.items():
                    data = annotator.get_data()
                    if isinstance(data, dict):
                        data = data.get("data")
                    rgba = np.asarray(data)
                    if rgba.shape != (args.resolution[1], args.resolution[0], 4):
                        raise RuntimeError(f"Invalid empty-board image for {name}: {rgba.shape}")
                    Image.fromarray(rgba[..., :3]).save(args.output / f"{name}-empty.png")
                print("EMPTY_BOARD_CAPTURED", str(args.output), flush=True)
            finally:
                for attr, value in visibility:
                    attr.Set(value)
            for _ in range(30):
                rep.orchestrator.step(rt_subframes=1, delta_time=0.0, pause_timeline=True)
            print("PIECES_RESTORED", flush=True)
        start = time.monotonic()
        count = valid = missing = consecutive_missing = 0
        next_report = start
        while app.is_running() and (not args.frames or count < args.frames):
            frame_start = time.monotonic()
            command = None
            timing_ms = {}
            if control and count:
                try:
                    command = store.commands.get(timeout=0.05)
                except queue.Empty:
                    app.update()  # timeline is paused between control steps
                    continue
                try:
                    if time.monotonic() > command['expires']:
                        raise ValueError('Expired command')
                    if command['payload']['frame_id'] != store.frame_id:
                        raise ValueError('Stale frame_id; action was not applied')
                    controlled = time.perf_counter()
                    if command['path'] == '/reset':
                        control.reset(command['payload'].get('options'))
                    else:
                        control.step(command['payload'].get('action'))
                    timing_ms['server_physics_ms'] = (time.perf_counter()-controlled)*1000
                except ValueError as error:
                    command.update(result={'error': str(error)}, status=409)
                    command['done'].set()
                    continue
            if replay and args.replay_loop and replay.index == len(replay.targets) - 1:
                replay.index = -1
            observed = time.perf_counter()
            replay_status = control.observe() if control else (replay.step() if replay else None)
            timing_ms['server_observe_ms'] = (time.perf_counter()-observed)*1000
            rendered = time.perf_counter()
            # Capture both products without advancing beyond the pose just set
            # by the replay (or the unchanged scene during camera bring-up).
            rep.orchestrator.step(rt_subframes=1, delta_time=0.0, pause_timeline=not (replay and args.replay_mode == "physics"))
            timing_ms['server_render_ms'] = (time.perf_counter()-rendered)*1000
            timing_ms['server_readback_ms'] = 0.
            timing_ms['server_preview_jpeg_ms'] = 0.
            images, arrays = {}, {}
            for name, annotator in annotators.items():
                readback = time.perf_counter()
                raw = annotator.get_data()
                if isinstance(raw, dict):
                    raw = raw.get("data")
                if raw is None:
                    continue
                rgba = np.asarray(raw)
                if rgba.shape != (args.resolution[1], args.resolution[0], 4) or rgba.dtype != np.uint8:
                    continue
                rgb = np.ascontiguousarray(rgba[..., :3])
                arrays[name] = rgb
                timing_ms['server_readback_ms'] += (time.perf_counter()-readback)*1000
                encoded = time.perf_counter()
                buffer = io.BytesIO()
                Image.fromarray(rgb).save(buffer, format="JPEG", quality=90)
                images[name] = buffer.getvalue()
                timing_ms['server_preview_jpeg_ms'] += (time.perf_counter()-encoded)*1000
            count += 1
            if len(arrays) != len(annotators):
                missing += 1
                consecutive_missing += 1
                if command:
                    command.update(result={'error': 'Action applied, but paired capture failed; reset required'}, status=500)
                    command['done'].set()
                    raise RuntimeError('Control capture failed after action')
                if consecutive_missing >= 30:
                    raise RuntimeError("No complete camera frames for 30 consecutive attempts")
                continue
            consecutive_missing = 0
            valid += 1
            elapsed = time.monotonic() - start
            simulation_time = replay_status["physics_time"] if replay_status and args.replay_mode == "physics" else timeline.get_current_time()
            stats = {"status": "running", "frame_id": count - 1, "complete_pairs": valid,
                     "missing_pairs": missing, "elapsed_seconds": elapsed,
                     "capture_fps": valid / max(elapsed, 1e-6), "target_fps": args.fps,
                     "simulation_time": simulation_time, "physics_running": bool((replay or control) and args.replay_mode == "physics"),
                     "cameras": args.cameras, "resolution": args.resolution,
                     "pixel_std": {name: float(rgb.std()) for name, rgb in arrays.items()}}
            if replay_status:
                stats["replay"] = replay_status
            packaged = time.perf_counter()
            buffer = io.BytesIO()
            extras = {'observation_json': np.array(json.dumps(replay_status))} if control else {}
            np.savez(buffer, **arrays, frame_id=count - 1, simulation_time=simulation_time, **extras)
            timing_ms['server_pair_pack_ms'] = (time.perf_counter()-packaged)*1000
            stats['timing_ms'] = timing_ms
            store.publish(count - 1, images, buffer.getvalue(), stats)
            if command:
                command['result'] = {'frame_id': count - 1, 'observation': replay_status, 'timing_ms': timing_ms}
                command['done'].set()
            if valid == 1:
                for name, rgb in arrays.items():
                    Image.fromarray(rgb).save(args.output / f"{name}.png")
                print("FIRST_FRAMES", str(args.output), flush=True)
            if replay and replay.index in (0, 100, 150, 185, 200, 240, 250, len(replay.targets) - 1):
                for name, rgb in arrays.items():
                    destination = args.output / f"{name}-frame-{replay.index:04d}.png"
                    if not destination.exists():
                        Image.fromarray(rgb).save(destination)
            if time.monotonic() >= next_report:
                (args.output / "stats.json").write_text(json.dumps(stats, indent=2))
                print("CAPTURE", json.dumps(stats), flush=True)
                next_report = time.monotonic() + 10
            time.sleep(max(0, 1 / args.fps - (time.monotonic() - frame_start)))
        if not valid:
            raise RuntimeError("Run ended without a complete camera frame")
        store.stats["status"] = "stopped"
    except KeyboardInterrupt:
        store.stats["status"] = "interrupted"
    except Exception as error:
        store.stats.update(status="error", error=str(error))
        import traceback
        traceback.print_exc()
        raise
    finally:
        (args.output / "stats.json").write_text(json.dumps(store.stats, indent=2))
        with store.condition:
            store.closed = True
            store.condition.notify_all()
        if server:
            server.shutdown()
            server.server_close()
        try:
            for annotator, product in zip(annotators.values(), products):
                annotator.detach(product)
            for product in products:
                product.destroy()
        finally:
            app.close()


if __name__ == "__main__":
    main()
