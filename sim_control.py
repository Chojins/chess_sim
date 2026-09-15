"""Synchronous simulated robot steps; no hardware interfaces."""
import numpy as np
import json
from pathlib import Path
from episode_replay import PhysicsEpisodeReplay


class SimControl:
    def __init__(self, stage, episode, mapping, output, contacts=True, gripper_stiffness=2.0, friction=1.0, magnets=None):
        self.driver = PhysicsEpisodeReplay(stage, episode, mapping, output, contacts=contacts,
                                          gripper_stiffness=gripper_stiffness, friction=friction, magnets=magnets)
        self.driver.external_control = True
        self.physics_settings = json.loads((Path(output) / 'replay_audit.json').read_text())
        self.driver.index = 0
        self.mapping = self.driver.mapping
        self.limits = np.array([[j[1].GetAttribute('physics:lowerLimit').Get(), j[1].GetAttribute('physics:upperLimit').Get()]
                                for j in self.driver.joints[:6]])
        self.episode_id = -1
        self.reset()

    def reset(self):
        for color, robot in self.driver.robots.items():
            q = self.driver.initial_robot_poses[color]
            robot.set_joint_positions(q)
            robot.set_joint_velocities(np.zeros_like(q))
            robot.set_joint_position_targets(q)
        if self.driver.pieces:
            self.driver.pieces.reset()
        self.driver.tracking.clear()
        self.episode_id += 1
        self.step_id = 0
        self.last_target = self.driver.targets[0].copy()
        # Settle the initial scene without invalidating camera graphs.
        for _ in range(10):
            self.driver.advance(self.last_target)
        self.start_time = self.driver.world.current_time
        self.driver.world.pause()
        return self.observe()

    def step(self, legacy_action):
        values = np.asarray(legacy_action, dtype=float)
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
        q = np.rad2deg(robot.get_joint_positions(joint_indices=self.driver.indices)[0])
        result = {'mode': 'policy_control', 'episode_id': self.episode_id, 'control_step': self.step_id,
                  'physics_time': self.driver.world.current_time - self.start_time,
                  'measured_joint_deg': q.tolist(), 'observation_state': self.mapping.to_legacy(q).tolist(),
                  'target_legacy_action': self.mapping.to_legacy(self.last_target).tolist(),
                  'joint_limits_deg': self.limits.tolist(), 'joint_map': self.mapping.config}
        result['physics_settings'] = {key: self.physics_settings[key] for key in
            ('physics_dt', 'substeps', 'drive_stiffness', 'drive_damping', 'fixed_arm_stiffness', 'fixed_arm_damping', 'contacts', 'magnets')
            if key in self.physics_settings}
        if self.driver.pieces:
            result['pieces'] = self.driver.pieces.sample()
        return result
