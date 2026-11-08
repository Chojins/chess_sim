"""Synchronous simulated robot steps; no hardware interfaces."""
import numpy as np
import json
from pathlib import Path
from episode_replay import PhysicsEpisodeReplay
from grasp_geometry import GRASP_LOCAL_M, GRASP_REFERENCE


class SimControl:
    def __init__(self, stage, episode, mapping, output, contacts=True, gripper_stiffness=2.0, friction=1.0, magnets=None, robot_self_collisions=False, robot_collision_geometry='hulls'):
        self.driver = PhysicsEpisodeReplay(stage, episode, mapping, output, contacts=contacts,
                                          gripper_stiffness=gripper_stiffness, friction=friction, magnets=magnets,
                                          robot_self_collisions=robot_self_collisions,robot_collision_geometry=robot_collision_geometry)
        self.driver.external_control = True
        self.physics_settings = json.loads((Path(output) / 'replay_audit.json').read_text())
        self.driver.index = 0
        self.mapping = self.driver.mapping
        self.limits = np.array([[j[1].GetAttribute('physics:lowerLimit').Get(), j[1].GetAttribute('physics:upperLimit').Get()]
                                for j in self.driver.joints[:6]])
        self.episode_id = -1
        self.task_geometry = None
        if self.driver.pieces:
            pieces = self.driver.pieces
            self.original_piece_poses = tuple(x.copy() for x in pieces.initial_poses)
            self.original_piece_centers = pieces.initial_centers.copy()
            from board_tasks import BoardLayout
            self.layout = BoardLayout(pieces.paths, pieces.initial_centers,
                                      self.physics_settings['contacts']['square_centers_m'],
                                      self.physics_settings['contacts']['board_top_m'],
                                      self.physics_settings['contacts']['board_frame_bounds_m'])
        self.reset()

    def reset(self, options=None):
        options = {} if options is None else options
        if not isinstance(options, dict) or set(options) - {'pawn_offset_mm', 'history', 'move'}:
            raise ValueError('Reset options support pawn_offset_mm, history and move')
        try:
            offset = np.asarray(options.get('pawn_offset_mm', [0, 0]), dtype=float)
        except (TypeError, ValueError) as error:
            raise ValueError('pawn_offset_mm must be a numeric pair') from error
        if offset.shape != (2,) or not np.isfinite(offset).all() or np.max(np.abs(offset)) > 2:
            raise ValueError('pawn_offset_mm must contain two finite values within +/-2 mm')
        if np.any(offset) and not self.driver.pieces:
            raise ValueError('Pawn offsets require contacts')
        plan = None
        if 'history' in options or 'move' in options:
            if not self.driver.pieces or 'move' not in options:
                raise ValueError('Legal resets require contacts and a move')
            plan = self.layout.plan(options.get('history', []), options['move'])
            if plan[1]['acting_robot'] != 'blue':
                raise ValueError('Red control needs its own validated joint map; only blue executes moves currently')
        for color, robot in self.driver.robots.items():
            q = self.driver.initial_robot_poses[color]
            robot.set_joint_positions(q)
            robot.set_joint_velocities(np.zeros_like(q))
            robot.set_joint_position_targets(q)
        if self.driver.pieces:
            self.driver.pieces.initial_poses = tuple(x.copy() for x in self.original_piece_poses)
            self.driver.pieces.initial_centers = self.original_piece_centers.copy()
            self.driver.pieces.active_index = self.driver.pieces.paths.index('/World/chess_pieces_blue/pawn_03')
            self.driver.pieces.reset()
        self.task_geometry = None
        self.driver.tracking.clear()
        self.episode_id += 1
        self.step_id = 0
        self.last_target = self.driver.targets[0].copy()
        # Settle the initial scene without invalidating camera graphs.
        for _ in range(10):
            self.driver.advance(self.last_target)
        if plan is not None:
            pieces = self.driver.pieces
            centers, geometry = plan
            positions, rotations = pieces.view.get_world_poses()
            positions += centers - pieces.centers()
            pieces.view.set_world_poses(positions, rotations)
            pieces.view.set_velocities(np.zeros((len(pieces.paths), 6)))
            for _ in range(10):
                self.driver.advance(self.last_target)
            pieces.initial_poses = tuple(x.copy() for x in pieces.view.get_world_poses())
            pieces.initial_centers = pieces.centers().copy()
            pieces.maximum_lift[:] = 0
            pieces.active_index = geometry['source_index']
            self.task_geometry = geometry
        # Apply the requested small test perturbation after settling. It is
        # part of reset, not a force or policy action; square targets stay fixed.
        if self.driver.pieces and np.any(offset):
            pieces = self.driver.pieces
            index = pieces.active_index
            positions, rotations = pieces.view.get_world_poses()
            positions[index, :2] += offset / 1000
            pieces.view.set_world_poses(positions, rotations)
            pieces.view.set_velocities(np.zeros((len(pieces.paths), 6)))
        self.reset_options = {'pawn_offset_mm': offset.tolist()}
        if plan is not None:
            self.reset_options.update(history=options.get('history', []), move=options['move'])
        self.start_time = self.driver.world.current_time
        self.driver.world.pause()
        return self.observe()

    def step(self, legacy_action):
        try:
            values = np.asarray(legacy_action, dtype=float)
        except (TypeError, ValueError) as error:
            raise ValueError('action must contain six finite legacy joint values') from error
        if values.shape != (6,) or not np.isfinite(values).all():
            raise ValueError('action must contain six finite legacy joint values')
        target = self.mapping.to_sim_deg(values)
        invalid = (target < self.limits[:, 0]) | (target > self.limits[:, 1])
        if invalid.any():
            raise ValueError('Out-of-limit action: ' + ', '.join(np.array(self.mapping.sim_names)[invalid]))
        self.driver.advance(target)
        self.last_target = target
        self.step_id += 1
        self.driver.world.pause()
        return self.observe()

    def observe(self):
        robot = self.driver.robots['blue']
        all_q = np.rad2deg(robot.get_joint_positions()[0])
        q = all_q[self.driver.indices]
        result = {'mode': 'policy_control', 'episode_id': self.episode_id, 'control_step': self.step_id,
                  'physics_time': self.driver.world.current_time - self.start_time,
                  'measured_joint_deg': q.tolist(), 'observation_state': self.mapping.to_legacy(q).tolist(),
                  'target_legacy_action': self.mapping.to_legacy(self.last_target).tolist(),
                  'joint_limits_deg': self.limits.tolist(), 'joint_map': self.mapping.config}
        result['measured_finger_joint_deg'] = {
            name: float(all_q[robot.get_dof_index(name)])
            for name in ('jaw1_hinge', 'jaw2_hinge')}
        result['physics_settings'] = {key: self.physics_settings[key] for key in
            ('physics_dt', 'substeps', 'drive_stiffness', 'drive_damping', 'fixed_arm_stiffness', 'fixed_arm_damping', 'contacts', 'magnets')
            if key in self.physics_settings}
        if self.driver.pieces:
            result['pieces'] = self.driver.pieces.sample()
            pieces = self.driver.pieces
            index = pieces.paths.index('/World/chess_pieces_blue/pawn_03')
            source = pieces.initial_centers[index].copy()
            approximate_target = source + [0, 0.045, 0]
            squares = np.asarray(self.physics_settings['contacts']['square_centers_m'])
            target = squares[np.argmin(np.linalg.norm(squares[:, :2] - approximate_target[:2], axis=1))].copy()
            target[2] = source[2]
            result['task_geometry'] = {'move': 'e2e4', 'source_center_m': source.tolist(),
                                       'target_center_m': target.tolist()}
            if self.task_geometry is not None:
                result['task_geometry'] = self.task_geometry
            angles = dict(zip(self.mapping.sim_names, q))
            angles['jaw2_hinge'] = -angles['jaw1_hinge']
            matrix = self.driver.forward(angles)['/World/so100_blue/gripper']
            result['gripper_grasp_center_m'] = list(matrix.Transform(self.driver.Gf.Vec3d(*GRASP_LOCAL_M)))
            result['grasp_reference'] = GRASP_REFERENCE
            result['gripper_tip_proxy_m'] = list(matrix.Transform(self.driver.Gf.Vec3d(0, -0.08112725, 0)))
        result['reset_options'] = self.reset_options
        return result
