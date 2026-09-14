"""Kinematic pose replay for checking a legacy joint map before physics/ACT.

USD joint frames define forward kinematics. No rigid-body physics is stepped,
and no physical robot interfaces are imported. Piece poses remain unchanged.
"""
import json
from pathlib import Path

import numpy as np


class JointMap:
    def __init__(self, path):
        self.config = json.loads(Path(path).read_text())
        self.names = self.config["legacy_names"]
        self.sim_names = self.config["sim_names"]
        self.home = np.array(self.config["legacy_home"], dtype=float)
        self.sim_home = np.array(self.config["sim_home_deg"], dtype=float)
        self.scale = np.array(self.config["scale"], dtype=float)
        if not (len(self.names) == len(self.sim_names) == len(self.home) == len(self.sim_home) == len(self.scale)):
            raise ValueError("Joint mapping lengths differ")
        if np.any(self.scale == 0) or not all(np.isfinite(x).all() for x in (self.home, self.sim_home, self.scale)):
            raise ValueError("Invalid joint mapping")

    def to_sim_deg(self, values):
        return self.sim_home + self.scale * (np.asarray(values) - self.home)

    def to_legacy(self, values_deg):
        return self.home + (np.asarray(values_deg) - self.sim_home) / self.scale


class EpisodeReplay:
    def __init__(self, stage, episode, mapping, output, source="states", kinematic=True):
        from pxr import Gf, UsdGeom
        self.Gf, self.UsdGeom, self.stage = Gf, UsdGeom, stage
        self.mapping = JointMap(mapping)
        self.output = Path(output)
        with np.load(episode, allow_pickle=False) as data:
            if data["joint_names"].tolist() != self.mapping.names:
                raise ValueError("Episode joint order does not match mapping")
            self.legacy = data[source].copy()
            self.timestamps = data["timestamps"].copy()
            self.fps = float(data["fps"])
        self.targets = self.mapping.to_sim_deg(self.legacy)
        self.root = self.mapping.config["robot_prim"]
        self.joints, self.ops = [], {}
        names = self.mapping.sim_names + ["jaw2_hinge"]
        for name in names:
            prim = stage.GetPrimAtPath(f"{self.root}/joints/{name}")
            if not prim:
                raise ValueError(f"Missing USD joint {name}")
            parent = prim.GetRelationship("physics:body0").GetTargets()[0]
            child = prim.GetRelationship("physics:body1").GetTargets()[0]
            f0, f1 = Gf.Matrix4d(1), Gf.Matrix4d(1)
            for frame, suffix in [(f0, "0"), (f1, "1")]:
                frame.SetRotate(Gf.Quatd(prim.GetAttribute("physics:localRot" + suffix).Get()))
                frame.SetTranslateOnly(Gf.Vec3d(prim.GetAttribute("physics:localPos" + suffix).Get()))
            axis = {"X": Gf.Vec3d(1, 0, 0), "Y": Gf.Vec3d(0, 1, 0), "Z": Gf.Vec3d(0, 0, 1)}[prim.GetAttribute("physics:axis").Get()]
            self.joints.append((name, prim, parent, child, f0, f1, axis))
        limits = np.array([[j[1].GetAttribute("physics:lowerLimit").Get(), j[1].GetAttribute("physics:upperLimit").Get()] for j in self.joints[:6]])
        violation = (self.targets < limits[:, 0]) | (self.targets > limits[:, 1])
        audit = {"mode": "kinematic", "source": source, "frames": len(self.targets), "dataset_fps": self.fps,
                 "mapping": self.mapping.config, "limits_deg": limits.tolist(),
                 "target_min_deg": self.targets.min(0).tolist(), "target_max_deg": self.targets.max(0).tolist(),
                 "limit_violations": {name: int(violation[:, i].sum()) for i, name in enumerate(self.mapping.sim_names)},
                 "roundtrip_max_error": float(np.abs(self.mapping.to_legacy(self.targets) - self.legacy).max())}
        (self.output / "replay_audit.json").write_text(json.dumps(audit, indent=2))
        if violation.any():
            raise ValueError("Mapped episode exceeds USD joint limits; see replay_audit.json. No clipping applied.")
        # Check the FK implementation against authored link transforms before
        # replacing them. This validates joint-frame math, not real calibration.
        cache = UsdGeom.XformCache()
        authored = {str(j[3]): cache.GetLocalToWorldTransform(stage.GetPrimAtPath(j[3])) for j in self.joints}
        q = {j[0]: j[1].GetAttribute("state:angular:physics:position").Get() for j in self.joints}
        calculated = self.forward(q)
        error = max(np.abs(np.array(calculated[p]) - np.array(m)).max() for p, m in authored.items())
        audit["authored_fk_matrix_max_error"] = float(error)
        (self.output / "replay_audit.json").write_text(json.dumps(audit, indent=2))
        if error > 0.002:
            raise ValueError(f"FK does not reproduce authored pose: {error}")
        if kinematic:
            for _, _, _, child, *_ in self.joints:
                self.ops[str(child)] = UsdGeom.Xformable(stage.GetPrimAtPath(child)).MakeMatrixXform()
        self.index = -1
        self.source = source

    def forward(self, angles):
        Gf = self.Gf
        cache = self.UsdGeom.XformCache()
        result = {}
        for name, _, parent, child, f0, f1, axis in self.joints:
            parent_world = result.get(str(parent))
            if parent_world is None:
                parent_world = cache.GetLocalToWorldTransform(self.stage.GetPrimAtPath(parent))
            rotation = Gf.Matrix4d(1).SetRotate(Gf.Rotation(axis, float(angles[name])))
            result[str(child)] = f1.GetInverse() * rotation * f0 * parent_world
        return result

    def step(self):
        self.index = min(self.index + 1, len(self.targets) - 1)
        angles = dict(zip(self.mapping.sim_names, self.targets[self.index]))
        # Authored PhysX mimic gearing=1 means q2 = -q1.
        angles["jaw2_hinge"] = -angles["jaw1_hinge"]
        matrices = self.forward(angles)
        cache = self.UsdGeom.XformCache()
        for name, prim, _, child, *_ in self.joints:
            parent = self.stage.GetPrimAtPath(child).GetParent()
            parent_world = cache.GetLocalToWorldTransform(parent)
            self.ops[str(child)].Set(matrices[str(child)] * parent_world.GetInverse())
            prim.GetAttribute("state:angular:physics:position").Set(float(angles[name]))
        return {"mode": "kinematic", "source": self.source, "episode_frame": self.index,
                "episode_time": float(self.timestamps[self.index]), "total_frames": len(self.targets),
                "complete": self.index == len(self.targets) - 1,
                "legacy_state": self.legacy[self.index].tolist(), "sim_joint_deg": self.targets[self.index].tolist()}


class PhysicsEpisodeReplay(EpisodeReplay):
    """Drive joint position targets with PhysX; gains are bring-up settings."""
    def __init__(self, stage, episode, mapping, output, source="states"):
        super().__init__(stage, episode, mapping, output, source, kinematic=False)
        from isaacsim.core.api import World
        from isaacsim.core.prims import Articulation
        self.world = World(physics_dt=1 / (self.fps * 4), rendering_dt=1 / self.fps,
                           stage_units_in_meters=1.0, backend="numpy", device="cpu")
        self.robots = {}
        initial = {}
        for color in ("blue", "red"):
            root = f"/World/so100_{color}"
            initial[color] = {name: stage.GetPrimAtPath(f"{root}/joints/{name}").GetAttribute(
                "state:angular:physics:position").Get() for name in self.mapping.sim_names + ["jaw2_hinge"]}
            self.robots[color] = self.world.scene.add(Articulation(prim_paths_expr=root, name=f"replay_{color}"))
        self.world.reset()
        self.tracking = []
        for color, robot in self.robots.items():
            values = initial[color]
            if color == "blue":
                values.update(dict(zip(self.mapping.sim_names, self.targets[0])))
                values["jaw2_hinge"] = -values["jaw1_hinge"]
            pose = np.deg2rad([[values[name] for name in robot.dof_names]])
            robot.set_joint_positions(pose)
            robot.set_joint_velocities(np.zeros_like(pose))
            robot.set_joint_position_targets(pose)
            # Temporary gains for position tracking, not identified real-servo parameters.
            kp = np.full_like(pose, 20.0 if color == "blue" else 1000.0)
            kd = np.full_like(pose, 1.0 if color == "blue" else 20.0)
            for i, name in enumerate(robot.dof_names):
                if name == "jaw2_hinge":
                    kp[0, i] = kd[0, i] = 0
            robot.set_gains(kps=kp, kds=kd)
        self.indices = np.array([self.robots["blue"].get_dof_index(name) for name in self.mapping.sim_names])
        self.red_initial = self.robots["red"].get_joint_positions().copy()
        audit = json.loads((self.output / "replay_audit.json").read_text())
        audit.update(mode="physics", physics_dt=1 / (self.fps * 4), drive_stiffness=20, drive_damping=1,
                     fixed_arm_stiffness=1000, fixed_arm_damping=20,
                     note="Session-only bring-up gains; real motor dynamics are not identified")
        (self.output / "replay_audit.json").write_text(json.dumps(audit, indent=2))
        self.reported = False

    def step(self):
        if not self.world.is_playing():
            self.world.play()
        self.index = min(self.index + 1, len(self.targets) - 1)
        self.robots["blue"].set_joint_position_targets(np.deg2rad(self.targets[self.index])[None], joint_indices=self.indices)
        for _ in range(4):
            self.world.step(render=False)
        measured = np.rad2deg(self.robots["blue"].get_joint_positions(joint_indices=self.indices)[0])
        if not np.isfinite(measured).all():
            raise RuntimeError("Non-finite physics joint state")
        error = measured - self.targets[self.index]
        if len(self.tracking) < len(self.targets):
            self.tracking.append(error)
        report = {"mode": "physics", "source": self.source, "episode_frame": self.index,
                  "physics_time": self.world.current_time,
                  "episode_time": float(self.timestamps[self.index]), "total_frames": len(self.targets),
                  "complete": self.index == len(self.targets) - 1,
                  "legacy_state": self.legacy[self.index].tolist(), "sim_joint_deg": self.targets[self.index].tolist(),
                  "measured_joint_deg": measured.tolist(), "tracking_error_deg": error.tolist(),
                  "red_drift_max_deg": float(np.rad2deg(np.abs(self.robots['red'].get_joint_positions() - self.red_initial)).max())}
        if self.index == len(self.targets) - 1 and not self.reported:
            errors = np.array(self.tracking)
            report["tracking_rmse_deg"] = np.sqrt(np.mean(errors ** 2, axis=0)).tolist()
            report["tracking_max_abs_deg"] = np.abs(errors).max(axis=0).tolist()
            (self.output / "physics_report.json").write_text(json.dumps(report, indent=2))
            self.reported = True
        return report
