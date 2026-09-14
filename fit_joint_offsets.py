"""Fit a provisional blue e2-e4 joint-zero correction with fixed scene geometry.

Run through run_joint_fit.sh for the existing Isaac USD Python environment.
This fits a fingertip proxy's board position, not a measured grasp or hardware
calibration. Height is preserved from the original replay, not fitted to data.
"""
import argparse
import copy
import json
from pathlib import Path

import numpy as np
from pxr import Gf, Usd, UsdGeom

from episode_replay import EpisodeReplay, JointMap


def fit(args):
    if args.output_map.resolve() == args.joint_map.resolve():
        raise ValueError("Use a separate candidate map; preserve the original")
    spec = json.loads(args.constraints.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.Open(str(args.usd))
    replay = EpisodeReplay(stage, args.episode, args.joint_map, args.output, kinematic=False)
    if replay.root != spec["robot_prim"]:
        raise ValueError("Constraints and mapping refer to different robots")
    with np.load(args.episode, allow_pickle=False) as data:
        if int(data["episode_index"]) != spec["episode_index"]:
            raise ValueError("Constraints refer to another episode")
        actions = replay.mapping.to_sim_deg(data["actions"])
    active = [replay.mapping.sim_names.index(n) for n in spec["fit_joints"]]
    tool = Gf.Vec3d(*spec["tool_point_in_gripper_m"])
    units_mm = UsdGeom.GetStageMetersPerUnit(stage) * 1000
    bounds = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])
    pawn = stage.GetPrimAtPath(spec["pawn_prim"])
    if not pawn:
        raise ValueError("Missing pawn prim")
    pawn_center = np.array(bounds.ComputeWorldBound(pawn).ComputeAlignedRange().GetMidpoint()) * units_mm
    # The current board is aligned with world X/Y. Recover its rank spacing
    # from square geometry rather than assuming a physical board dimension.
    squares = [np.array(bounds.ComputeWorldBound(p).ComputeAlignedRange().GetMidpoint())
               for p in Usd.PrimRange(stage.GetPrimAtPath("/World/board_assembly"))
               if p.GetName().startswith("tn__boardsquare_")]
    ranks = np.unique(np.round(np.array(squares)[:, 1] * units_mm, 3))
    if len(squares) != 64 or len(ranks) != 8 or np.ptp(np.diff(ranks)) > 0.01:
        raise ValueError("Expected the current 8x8 board aligned with world axes")
    pitch = float(np.median(np.diff(ranks)))

    def tip(frame, delta):
        q = replay.targets[frame].copy()
        q[active] += delta
        angles = dict(zip(replay.mapping.sim_names, q))
        angles["jaw2_hinge"] = -angles["jaw1_hinge"]
        matrix = replay.forward(angles)[replay.root + "/gripper"]
        return np.array(matrix.Transform(tool)) * units_mm

    anchors = spec["anchors"]
    zero = np.zeros(len(active))
    targets = []
    for anchor in anchors:
        target = tip(anchor["frame"], zero)
        target[:2] = pawn_center[:2]
        target[1] += anchor["rank_advance"] * pitch
        targets.append(target)
    targets = np.array(targets)

    def residual(delta):
        # X is audited, not fitted; shoulder pan and wrist roll stay fixed.
        return (np.array([tip(a["frame"], delta) for a in anchors])[:, 1:] - targets[:, 1:]).ravel()

    limits = np.array([[j[1].GetAttribute("physics:lowerLimit").Get(),
                        j[1].GetAttribute("physics:upperLimit").Get()] for j in replay.joints[:6]])
    all_targets = np.concatenate((replay.targets, actions))
    lower = np.maximum(limits[active, 0] - all_targets[:, active].min(0), -spec["max_offset_deg"])
    upper = np.minimum(limits[active, 1] - all_targets[:, active].max(0), spec["max_offset_deg"])
    if np.any(lower > upper):
        raise ValueError("No offset can satisfy the configured limits")
    # Project fit parameters into their feasible interval, never clip replay
    # samples. Every sample retains the same affine joint mapping.
    delta = np.clip(zero, lower, upper)
    for _ in range(60):
        r = residual(delta)
        jac = np.column_stack([(residual(delta + np.eye(len(active))[k] * 0.001) - r) / 0.001
                               for k in range(len(active))])
        step = np.linalg.lstsq(jac, -r, rcond=None)[0]
        for scale in (1, 0.5, 0.25, 0.125, 0.0625):
            trial = np.clip(delta + scale * step, lower, upper)
            if np.linalg.norm(residual(trial)) < np.linalg.norm(r):
                break
        else:
            break
        change = np.linalg.norm(trial - delta)
        delta = trial
        if change < 1e-7:
            break
    if np.sqrt(np.mean(residual(delta) ** 2)) > spec["max_fit_rmse_mm"]:
        raise ValueError("Could not fit the contacts within the configured tolerance")
    correction = np.zeros(6)
    correction[active] = delta
    candidate = copy.deepcopy(replay.mapping.config)
    candidate["status"] = "provisional_episode_0_board_fit_not_hardware_calibrated"
    candidate["sim_home_deg"] = (replay.mapping.sim_home + correction).tolist()
    candidate["fit"] = {"constraints": str(args.constraints), "report": str(args.output / "fit_report.json"),
                        "offset_from_original_deg": correction.tolist()}
    candidate["notes"] += [
        "Candidate fitted to episode 0 e2-e4 with bases, board, signs and scales fixed.",
        "Only shoulder_lift, elbow_flex and wrist_flex offsets change; the home is now a fitted reference.",
        "Tool point approximates the centre of the distal closed-jaw tips. Absolute contact height is uncalibrated.",
        "Preserves baseline tool height at the two fit frames; does not establish a successful physical grasp.",
        "Validate other episodes and poses before treating this as a general legacy-to-sim calibration."]
    validation = {}
    for name, values in [("states", replay.targets), ("actions", actions)]:
        shifted = values + correction
        violation = (shifted < limits[:, 0]) | (shifted > limits[:, 1])
        validation[name] = {"limit_violations": int(violation.sum()), "min_deg": shifted.min(0).tolist(),
                            "max_deg": shifted.max(0).tolist()}
        if violation.any():
            raise ValueError(f"Candidate {name} exceed limits")
    checks = []
    for anchor in anchors:
        for frame in [anchor["frame"], *anchor["check_frames"]]:
            before, after = tip(frame, zero), tip(frame, delta)
            xy = pawn_center[:2].copy()
            xy[1] += anchor["rank_advance"] * pitch
            checks.append({"phase": anchor["phase"], "frame": frame, "used_for_fit": frame == anchor["frame"],
                           "target_xy_mm": xy.tolist(), "before_tip_mm": before.tolist(), "after_tip_mm": after.tolist(),
                           "before_xy_error_mm": float(np.linalg.norm(before[:2] - xy)),
                           "after_xy_error_mm": float(np.linalg.norm(after[:2] - xy))})
    report = {"constraints": spec, "source_map": replay.mapping.config, "board_rank_pitch_mm": pitch,
              "offset_deg": dict(zip(replay.mapping.sim_names, correction.tolist())),
              "fit_yz_rmse_mm": float(np.sqrt(np.mean(residual(delta) ** 2))),
              "contact_checks": checks, "trajectory_validation": validation,
              "limitations": "Only two same-episode contacts; adjacent checks are not independent episode validation. "
                              "Tool point and absolute grasp height are approximate. Pieces stay fixed in kinematic replay."}
    (args.output / "fit_report.json").write_text(json.dumps(report, indent=2) + "\n")
    args.output_map.parent.mkdir(parents=True, exist_ok=True)
    args.output_map.write_text(json.dumps(candidate, indent=2) + "\n")
    # Verify serialization and reversibility before handing it to the replay.
    loaded = JointMap(args.output_map)
    np.testing.assert_allclose(loaded.to_legacy(loaded.to_sim_deg(replay.legacy)), replay.legacy, atol=1e-10)
    print(json.dumps({"candidate": str(args.output_map), "offset_deg": report["offset_deg"],
                      "fit_yz_rmse_mm": report["fit_yz_rmse_mm"], "contact_checks": checks}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--usd", type=Path, default=Path("Sim_Stage.usd"))
    parser.add_argument("--episode", type=Path, default=Path("outputs/replay/episode_000000.npz"))
    parser.add_argument("--joint-map", type=Path, default=Path("calibration/legacy_blue_joint_map.json"))
    parser.add_argument("--constraints", type=Path, default=Path("calibration/episode_000000_contacts.json"))
    parser.add_argument("--output-map", type=Path, default=Path("calibration/legacy_blue_joint_map_board_fit.json"))
    parser.add_argument("--output", type=Path, default=Path("outputs/replay/board-fit"))
    fit(parser.parse_args())
