"""Versioned e2e4 reward and outcome logic, independent of Isaac and Gym."""
from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class TaskConfig:
    max_steps: int = 400
    lift_m: float = .005
    lift_steps: int = 3
    placement_m: float = .005
    height_tolerance_m: float = .003
    upright_cosine: float = .95
    stable_steps: int = 10
    home_deg: float = 5
    departure_deg: float = 15
    release_clearance_m: float = .035

    def __post_init__(self):
        if not all(np.isfinite(v) and v > 0 for v in asdict(self).values()):
            raise ValueError('Task parameters must be finite and positive')
        if any(type(v) is not int for v in (self.max_steps, self.lift_steps, self.stable_steps)):
            raise ValueError('Step counts must be integers')


class ChessTask:
    version = 'e2e4_v1'

    def __init__(self, initial, config=None):
        self.config = config or TaskConfig()
        if initial['task_geometry']['move'] != 'e2e4':
            raise ValueError('Only e2e4 is validated')
        self.source = np.array(initial['task_geometry']['source_center_m'])
        self.target = np.array(initial['task_geometry']['target_center_m'])
        self.home = np.array(initial['measured_joint_deg'])[:5]
        self.step_count = self.lift_streak = self.place_streak = self.home_streak = 0
        self.lifted = self.transported = self.departed = self.done = False
        self.max_lift = 0.
        self.min_target_error = float('inf')
        self.previous_center = np.array(initial['pieces']['pawn_center_m'])

    def update(self, observation):
        if self.done:
            raise RuntimeError('Task finished; reset before stepping')
        c = self.config
        piece = observation['pieces']
        center = np.array(piece['pawn_center_m'])
        home_error = float(np.abs(np.array(observation['measured_joint_deg'])[:5] - self.home).max())
        error = float(np.linalg.norm(center[:2] - self.target[:2]))
        height = float(center[2] - self.source[2])
        clearance = float(np.linalg.norm(np.array(observation['gripper_tip_proxy_m']) - center))
        if not np.isfinite([*center, home_error, error, height, clearance]).all():
            raise ValueError('Non-finite task telemetry')
        self.step_count += 1
        self.max_lift = max(self.max_lift, height)
        self.min_target_error = min(self.min_target_error, error)
        self.departed |= home_error > c.departure_deg
        self.home_streak = self.home_streak + 1 if self.departed and home_error < c.home_deg else 0
        self.lift_streak = self.lift_streak + 1 if height > c.lift_m else 0
        terms = {'time': -.001}
        if not self.lifted and self.lift_streak >= c.lift_steps:
            self.lifted = True
            terms['lift'] = 1.
        if self.lifted and not self.transported and height > c.lift_m and error < c.placement_m:
            self.transported = True
            terms['transport'] = 1.
        placement = (self.lifted and error < c.placement_m and abs(height) < c.height_tolerance_m and
                     piece['pawn_upright_cosine'] > c.upright_cosine and clearance > c.release_clearance_m and
                     np.linalg.norm(center - self.previous_center) < .0005)
        self.previous_center = center
        self.place_streak = self.place_streak + 1 if placement else 0
        success = self.place_streak >= c.stable_steps and self.home_streak >= c.stable_steps
        reason = None
        if piece['other_pieces_displaced_over_3mm'] > 0:
            reason, success = 'disturbed_other_piece', False
            terms['disturbance'] = -2.
        elif success:
            reason = 'success'
            terms['success'] = 10.
        elif self.home_streak >= c.stable_steps:
            # Allow a full release stability window after reaching home.
            if not placement or self.home_streak >= 2 * c.stable_steps:
                reason = 'failed_grasp' if not self.lifted else (
                    'wrong_placement' if error >= c.placement_m else 'unstable_or_unreleased')
        terminated = reason is not None
        truncated = not terminated and self.step_count >= c.max_steps
        if truncated:
            reason = 'timeout'
        self.done = terminated or truncated
        return float(sum(terms.values())), terminated, truncated, {
            'success': bool(success), 'reason': reason, 'reward_terms': terms,
            'target_error_m': error, 'height_m': height, 'max_lift_m': self.max_lift,
            'min_target_error_m': self.min_target_error, 'home_error_deg': home_error,
            'clearance_m': clearance, 'lifted': self.lifted, 'transported': self.transported,
            'placement_streak': self.place_streak, 'home_streak': self.home_streak,
            'upright_cosine': piece['pawn_upright_cosine'],
            'other_pieces_displaced_over_3mm': piece['other_pieces_displaced_over_3mm']}
