"""Fit image-to-board homographies from empty-board camera captures.

This calibrates a 2-D board overlay at fixed camera poses, not camera
intrinsics or 3-D robot poses. No real-camera calibration files are changed.
"""

import argparse
import importlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np


def detect_board(image, white_side):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCorners(
        cv2.bitwise_not(gray), (7, 7), cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)
    if found:
        corners = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1),
                                 (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001))
    else:
        found, corners = cv2.findChessboardCornersSB(gray, (7, 7), cv2.CALIB_CB_NORMALIZE_IMAGE)
    if not found:
        raise ValueError("Could not detect all 49 inner board corners")
    grid = corners.reshape(7, 7, 2)
    outer = np.array([grid[0, 0], grid[0, -1], grid[-1, -1], grid[-1, 0]])
    # These wrist views are upright: establish image TL, TR, BR, BL, then
    # rotate Black's view by 180 degrees to share White's board coordinates.
    sums, differences = outer.sum(axis=1), outer[:, 0] - outer[:, 1]
    order = [sums.argmin(), differences.argmax(), sums.argmax(), differences.argmin()]
    if len(set(order)) != 4:
        raise ValueError("View is too rotated for automatic upright-board ordering")
    target = outer[order]
    candidates = [np.rot90(g, k) for g in (grid, grid[:, ::-1]) for k in range(4)]
    grid = min(candidates, key=lambda g: np.linalg.norm(
        np.array([g[0, 0], g[0, -1], g[-1, -1], g[-1, 0]]) - target))
    if not white_side:
        grid = np.rot90(grid, 2)
    board_points = np.array([(x * 100, y * 100) for y in range(1, 8) for x in range(1, 8)], np.float32)
    image_points = np.ascontiguousarray(grid.reshape(-1, 2))
    homography, _ = cv2.findHomography(board_points, image_points, method=0)
    if homography is None or not np.isfinite(homography).all():
        raise ValueError("Failed to fit board homography")
    reprojected = cv2.perspectiveTransform(board_points[:, None], homography)[:, 0]
    errors = np.linalg.norm(reprojected - image_points, axis=1)
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    if rmse > 2:
        raise ValueError(f"Board fit is inaccurate: {rmse:.2f} pixels RMS")
    inner = cv2.perspectiveTransform(np.float32([[[100, 100], [700, 100], [700, 700], [100, 700]]]), homography)[0]
    return {"inner_corners": inner.tolist(), "board_size": 800, "square_size": 100,
            "resolution": [image.shape[1], image.shape[0]], "white_side": white_side,
            "board_to_image": homography.tolist(), "image_to_board": np.linalg.inv(homography).tolist(),
            "detected_corners": 49, "reprojection_rmse_px": rmse, "max_reprojection_error_px": float(errors.max())}


def load_highlighter(vision_repo):
    sys.path.insert(0, str(Path(vision_repo).resolve()))
    return importlib.import_module("chess_vision").highlight_chess_move


def highlight(frame, calibration, move, highlighter):
    if [frame.shape[1], frame.shape[0]] != calibration["resolution"]:
        raise ValueError("Frame resolution differs from board calibration; recalibrate")
    # The existing 2-D highlighter only uses pose for optional axes/3-D drawing.
    # Pass placeholders with those features off; do not invent a calibrated pose.
    return highlighter(frame, move, np.float32(calibration["inner_corners"]),
                       calibration["board_size"], calibration["square_size"],
                       (np.zeros((3, 1)), np.zeros((3, 1))), undistort=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--captures", type=Path, default=Path("outputs/calibration"))
    parser.add_argument("--output", type=Path, default=Path("calibration/sim_board.json"))
    parser.add_argument("--white-camera", choices=("blue", "red"), default="blue", help="Camera viewing the board from White's end")
    parser.add_argument("--move", default="e2e4")
    parser.add_argument("--vision-repo", type=Path, default=Path(__file__).resolve().parent.parent / "chess_vision")
    args = parser.parse_args()
    import chess
    chess.Move.from_uci(args.move)
    inventory = json.loads((args.captures / "stage_inventory.json").read_text())
    highlighter = load_highlighter(args.vision_repo)
    result = {"schema_version": 1, "kind": "fixed_pose_2d_board_homography", "usd": inventory["usd"],
              "resolution_note": "Recalibrate after camera/board motion or lens/resolution changes.", "cameras": {}}
    for name in ("blue", "red"):
        empty = cv2.imread(str(args.captures / f"{name}-empty.png"))
        populated = cv2.imread(str(args.captures / f"{name}.png"))
        if empty is None or populated is None:
            raise ValueError(f"Missing {name} capture; run --capture-empty-board first")
        entry = detect_board(empty, name == args.white_camera)
        entry["prim_path"] = f"/World/so100_{name}/gripper/Camera"
        result["cameras"][name] = entry
        preview = highlight(populated, entry, args.move, highlighter)
        cv2.imwrite(str(args.captures / f"{name}-{args.move}.png"), preview)
        grid_image = empty.copy()
        matrix = np.array(entry["board_to_image"])
        for value in range(9):
            for line in ([[value * 100, 0], [value * 100, 800]], [[0, value * 100], [800, value * 100]]):
                pixels = cv2.perspectiveTransform(np.float32([line]), matrix)[0].round().astype(int)
                cv2.line(grid_image, tuple(pixels[0]), tuple(pixels[1]), (0, 200, 0), 1)
        for label, point in {"a8": (50, 50), "h8": (750, 50), "a1": (50, 750), "h1": (750, 750)}.items():
            pixel = cv2.perspectiveTransform(np.float32([[point]]), matrix)[0, 0].round().astype(int)
            cv2.putText(grid_image, label, tuple(pixel), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
        cv2.imwrite(str(args.captures / f"{name}-grid.png"), grid_image)
        print(f"{name}: 49 corners, {entry['reprojection_rmse_px']:.3f} px RMS; saved {args.move} preview")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
