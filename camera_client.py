"""Receive a complete, lossless pair of camera frames outside Isaac Sim.

Uses NumPy and OpenCV from your vision/policy environment. The HTTP MJPEG
URLs are also available for cv2.VideoCapture, but independent streams do
not guarantee that the two reads belong to the same capture.
"""

import argparse
import io
import json
from pathlib import Path
import time
from urllib.request import urlopen

import numpy as np


class PairedCameraClient:
    def __init__(self, url="http://127.0.0.1:8080", timeout=10):
        self.url = url.rstrip("/")
        self.timeout = timeout

    def read(self, color_order="bgr"):
        """Return (metadata, images) with uint8 HWC images from one capture.

        Use BGR for chess_vision/OpenCV or RGB for policy preprocessing.
        Repeated reads may return the same frame_id; no frames are queued.
        Network failures raise exceptions rather than returning stale frames.
        """
        if color_order not in ("rgb", "bgr"):
            raise ValueError("color_order must be rgb or bgr")
        with urlopen(f"{self.url}/frames.npz", timeout=self.timeout) as response:
            body = response.read()
        with np.load(io.BytesIO(body), allow_pickle=False) as pair:
            metadata = {"frame_id": int(pair["frame_id"]), "simulation_time": float(pair["simulation_time"])}
            if 'observation_json' in pair.files:
                metadata['observation'] = json.loads(str(pair['observation_json']))
            images = {}
            for name in pair.files:
                if name in metadata or name == 'observation_json':
                    continue
                rgb = pair[name]
                if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
                    raise ValueError(f"Invalid camera image: {name}: {rgb.shape} {rgb.dtype}")
                images[name] = np.ascontiguousarray(rgb[..., ::-1] if color_order == "bgr" else rgb)
        return metadata, images


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8080")
    parser.add_argument("--headless", action="store_true", help="Check frames without displaying a window")
    parser.add_argument("--frames", type=int, default=0, help="Stop after this many distinct pairs; 0 runs continuously")
    parser.add_argument("--output", type=Path, help="Save the first pair as PNG files")
    parser.add_argument("--move", help="Show a UCI move using the saved fixed-pose sim board calibration")
    parser.add_argument("--overlay-camera", action="append", choices=("blue", "red"),
                        help="Apply the overlay only to this stationary camera (repeat to select both)")
    parser.add_argument("--calibration", type=Path, default=Path(__file__).parent / "calibration/sim_board.json")
    parser.add_argument("--vision-repo", type=Path, default=Path(__file__).resolve().parent.parent / "chess_vision")
    args = parser.parse_args()
    if args.frames < 0:
        parser.error("frames must be nonnegative")
    import cv2

    if args.move:
        import chess
        from calibrate_board import highlight, load_highlighter
        chess.Move.from_uci(args.move)
        calibration = json.loads(args.calibration.read_text())["cameras"]
        highlighter = load_highlighter(args.vision_repo)

    client = PairedCameraClient(args.url)
    last_id, count = -1, 0
    started = time.monotonic()
    last_new_frame = started
    try:
        while not args.frames or count < args.frames:
            metadata, images = client.read()
            if metadata["frame_id"] == last_id:
                if time.monotonic() - last_new_frame > 10:
                    raise TimeoutError("Server has not produced a new frame for 10 seconds")
                time.sleep(0.01)
                if not args.headless and cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue
            last_id = metadata["frame_id"]
            last_new_frame = time.monotonic()
            count += 1
            if args.move:
                images = {name: (highlight(bgr, calibration[name], args.move, highlighter)
                                 if args.overlay_camera is None or name in args.overlay_camera else bgr)
                          for name, bgr in images.items()}
            if count == 1 and args.output:
                args.output.mkdir(parents=True, exist_ok=True)
                for name, bgr in images.items():
                    if not cv2.imwrite(str(args.output / f"{name}.png"), bgr):
                        raise RuntimeError(f"Could not save {name}")
            if not args.headless:
                for name, bgr in images.items():
                    cv2.imshow(f"Chess sim: {name} (q to quit)", bgr)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            if count == 1 or count % 100 == 0:
                print(json.dumps({**metadata, "received_pairs": count,
                                  "receive_fps": count / max(time.monotonic() - started, 1e-6),
                                  "shapes": {name: list(img.shape) for name, img in images.items()}}), flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        if not args.headless:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
