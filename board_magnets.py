"""Effective board/piece attraction, not a calibrated magnetic field solver.

Each square contributes U = -E exp(-r²/(2 sigma²)) exp(-h / decay).
The spatial gradient supplies lateral centring and downward retention. Forces
act at the piece base every physics substep; nothing snaps or attaches pieces.
"""
from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class MagnetSettings:
    energy_j: float = 0.00056
    lateral_width_m: float = 0.004
    height_decay_m: float = 0.004
    damping_ns_m: float = 0.3
    board_friction: float = 0.12

    def __post_init__(self):
        if not all(np.isfinite(v) and v > 0 for v in asdict(self).values()):
            raise ValueError('Magnet settings must be finite and positive')


def attraction(points, velocities, squares, settings):
    """World-space forces in N; points are effective magnet sites at piece bases.

    Height is clearance above the square top. The potential gradient is used
    above the board, with penetration clamped to avoid excessive forces.
    Local viscous damping is a separate dissipative approximation.
    """
    delta = squares[None, :, :] - points[:, None, :]
    height = np.maximum(-delta[:, :, 2], 0)
    weight = np.exp(-np.sum(delta[:, :, :2] ** 2, axis=2) /
                    (2 * settings.lateral_width_m ** 2) - height / settings.height_decay_m)
    force = np.zeros_like(points, dtype=float)
    force[:, :2] = settings.energy_j / settings.lateral_width_m ** 2 * np.sum(
        weight[:, :, None] * delta[:, :, :2], axis=1)
    force[:, 2] = -settings.energy_j / settings.height_decay_m * weight.sum(axis=1)
    force -= settings.damping_ns_m * weight.sum(axis=1)[:, None] * velocities
    return force


def rotate(quaternions, vectors):
    """Rotate N vectors by N scalar-first unit quaternions."""
    xyz = quaternions[:, 1:]
    t = 2 * np.cross(xyz, vectors)
    return vectors + quaternions[:, :1] * t + np.cross(xyz, t)


class BoardMagnets:
    def __init__(self, stage, view, paths, settings):
        from pxr import Gf, Usd, UsdGeom
        self.view, self.settings = view, settings
        bbox = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ['default', 'render'])
        cache = UsdGeom.XformCache()
        self.squares = []
        for prim in Usd.PrimRange(stage.GetPrimAtPath('/World/board_assembly')):
            if prim.GetName().startswith('tn__boardsquare_'):
                bounds = bbox.ComputeWorldBound(prim).ComputeAlignedRange()
                center = np.array(bounds.GetMidpoint())
                center[2] = bounds.GetMax()[2]
                self.squares.append(center)
        self.squares = np.asarray(self.squares)
        if self.squares.shape != (64, 3):
            raise ValueError('Expected 64 board magnet sites')
        # Find the centre of each circular foot, rather than using the rigid
        # prim origin (the imported STL origins are substantially offset).
        bases = []
        for path in paths:
            prim = stage.GetPrimAtPath(path)
            vertices = []
            for mesh in Usd.PrimRange(prim):
                if mesh.IsA(UsdGeom.Mesh):
                    points = np.asarray(UsdGeom.Mesh(mesh).GetPointsAttr().Get())
                    matrix = np.asarray(cache.GetLocalToWorldTransform(mesh))
                    vertices.append(points @ matrix[:3, :3] + matrix[3, :3])
            points = np.concatenate(vertices)
            bottom = points[:, 2].min()
            foot = points[points[:, 2] < bottom + 0.0012]
            base = (foot.min(axis=0) + foot.max(axis=0)) / 2
            base[2] = bottom
            matrix = cache.GetLocalToWorldTransform(prim)
            local = np.array(matrix.GetInverse().Transform(Gf.Vec3d(*base)))
            bases.append(local * np.array(Gf.Transform(matrix).GetScale()))
        self.local_bases = np.array(bases)
        self.local_coms = np.asarray(view.get_coms()[0]).reshape(len(paths), 3)
        self.last_forces = np.zeros_like(self.local_bases)

    def base_positions(self):
        positions, rotations = self.view.get_world_poses()
        return positions + rotate(rotations, self.local_bases)

    def apply(self):
        positions, rotations = self.view.get_world_poses()
        bases = positions + rotate(rotations, self.local_bases)
        lever = rotate(rotations, self.local_bases - self.local_coms)
        velocities = self.view.get_velocities()
        point_velocities = velocities[:, :3] + np.cross(velocities[:, 3:], lever)
        forces = attraction(bases, point_velocities, self.squares, self.settings)
        if not np.isfinite(forces).all():
            raise RuntimeError('Non-finite magnetic force')
        self.view.apply_forces_and_torques_at_pos(
            forces=forces.astype(np.float32), positions=bases.astype(np.float32), is_global=True)
        self.last_forces = forces

    def describe(self):
        return {'model': 'gaussian_potential_v1', 'calibrated': False,
                **asdict(self.settings),
                'centre_downward_force_n': self.settings.energy_j / self.settings.height_decay_m,
                'square_count': len(self.squares), 'piece_count': len(self.local_bases),
                'effective_sites': 'square top centres and circular piece foot centres',
                'square_sites_m': self.squares.tolist(), 'piece_local_sites_m': self.local_bases.tolist(),
                'limitations': 'Provisional attraction and viscous damping; no polarity, dipole alignment torque or piece-to-piece magnetism'}
