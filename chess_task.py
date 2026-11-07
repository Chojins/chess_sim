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
    capture_home_retry: bool = False

    def __post_init__(self):
        if type(self.capture_home_retry) is not bool:
            raise ValueError('Capture home retry must be boolean')
        if not all(np.isfinite(v) and v > 0 for key, v in asdict(self).items() if key != 'capture_home_retry'):
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


class LegalMoveTask(ChessTask):
    """Single ordinary move or capture; captures belong in the actor's RHS zone."""
    version = 'legal_move_v5'

    def __init__(self, initial, config=None):
        self.geometry = initial['task_geometry']
        self.source_index = self.geometry['source_index']
        self.capture_index = self.geometry['capture_index']
        self.initial_centers = np.array(initial['pieces']['piece_centers_m'])
        squares = initial.get('physics_settings', {}).get('contacts', {}).get('square_centers_m', self.geometry.get('square_centers_m'))
        if np.asarray(squares).shape != (64, 3):
            raise ValueError('Square-based disturbance scoring requires 64 CAD square centres')
        sites = sorted(np.asarray(squares, dtype=float), key=lambda p: p[1])
        if len(sites) != 64:
            raise ValueError('Square-based disturbance scoring requires 64 CAD square centres')
        grid = np.array([sorted(sites[i:i + 8], key=lambda p: p[0]) for i in range(0, 64, 8)])[:, :, :2]
        axes = np.column_stack((np.diff(grid, axis=1).mean(axis=(0, 1)), np.diff(grid, axis=0).mean(axis=(0, 1))))
        self.square_inverse = np.linalg.inv(axes)
        flat = grid.reshape(64, 2)
        nearest = np.linalg.norm(self.initial_centers[:, None, :2] - flat[None], axis=2).argmin(axis=1)
        self.original_square_centers = flat[nearest]
        self.on_board = np.max(np.abs((self.initial_centers[:, :2] - self.original_square_centers) @ self.square_inverse.T), axis=1) <= .5
        self.parked_zones = []
        for center in self.initial_centers:
            self.parked_zones.append(next((z for z in self.geometry.get('capture_zones', {}).values()
                                          if np.all(center[:2] >= z['min_xy_m']) and np.all(center[:2] <= z['max_xy_m'])), None))
        self.previous_capture = None if self.capture_index is None else self.initial_centers[self.capture_index].copy()
        self.capture_lift_streak = self.capture_place_streak = self.capture_upright_streak = 0
        self.capture_lifted = self.capture_bonus = False
        adapted = self._adapt(initial)
        adapted['task_geometry'] = {**self.geometry, 'move': 'e2e4'}
        super().__init__(adapted, config)

    def _adapt(self, observation):
        pieces = observation['pieces']
        centers = np.asarray(pieces['piece_centers_m'])
        local = (centers[:, :2] - self.original_square_centers) @ self.square_inverse.T
        displaced = np.max(np.abs(local), axis=1) > .5
        for i in np.flatnonzero(~self.on_board):
            zone = self.parked_zones[i]
            displaced[i] = (not (np.all(centers[i, :2] >= zone['min_xy_m']) and np.all(centers[i, :2] <= zone['max_xy_m']))) if zone else np.linalg.norm(centers[i] - self.initial_centers[i]) > .003
        displaced |= np.asarray(pieces['piece_upright_cosines']) < np.cos(np.deg2rad(30))
        displaced[self.source_index] = False
        if self.capture_index is not None:
            displaced[self.capture_index] = False
        return {**observation, 'pieces': {**pieces,
            'pawn_center_m': centers[self.source_index].tolist(),
            'pawn_upright_cosine': pieces['piece_upright_cosines'][self.source_index],
            'other_pieces_displaced_over_3mm': int(displaced.sum())}}

    def update(self, observation):
        if self.done:
            raise RuntimeError('Task finished; reset before stepping')
        capture_ready, bonus = True, 0.
        lift_bonus = 0.
        order_violation = False
        capture_status = None
        if self.capture_index is not None:
            c = self.config
            center = np.asarray(observation['pieces']['piece_centers_m'][self.capture_index])
            height = center[2] - self.initial_centers[self.capture_index, 2]
            self.capture_lift_streak = self.capture_lift_streak + 1 if height > c.lift_m else 0
            if not self.capture_lifted and self.capture_lift_streak >= c.lift_steps:
                self.capture_lifted = True
                lift_bonus = 1.
            zone = self.geometry['capture_zone']
            inside = bool(np.all(center[:2] >= zone['min_xy_m']) and np.all(center[:2] <= zone['max_xy_m']))
            clearance = np.linalg.norm(np.asarray(observation['gripper_tip_proxy_m']) - center)
            upright_rest = (abs(center[2] - self.geometry['capture_rest_center_z_m']) < c.height_tolerance_m and
                            observation['pieces']['piece_upright_cosines'][self.capture_index] > c.upright_cosine)
            # A side-lying piece has a lower centre than the upright rest pose.
            near_table = (zone['surface_z_m'] < center[2] <
                          self.geometry['capture_rest_center_z_m'] + c.height_tolerance_m)
            rest = (self.capture_lifted and inside and near_table and
                    clearance > c.release_clearance_m and np.linalg.norm(center - self.previous_capture) < .0005)
            self.previous_capture = center.copy()
            self.capture_place_streak = self.capture_place_streak + 1 if rest else 0
            self.capture_upright_streak = self.capture_upright_streak + 1 if rest and upright_rest else 0
            capture_ready = self.capture_place_streak >= c.stable_steps
            if capture_ready and not self.capture_bonus:
                bonus, self.capture_bonus = .25, True
            attacker = np.asarray(observation['pieces']['piece_centers_m'][self.source_index])
            attacker_local = (attacker[:2] - self.original_square_centers[self.source_index]) @ self.square_inverse.T
            attacker_inside = bool(np.max(np.abs(attacker_local)) <= .5)
            order_violation = not self.capture_bonus and not attacker_inside
            capture_status = {'attacker_inside_original_square': attacker_inside,
                              'order_rule': 'attacker_center_inside_original_square_until_victim_release',
                              'inside_actor_rhs': inside, 'lifted': bool(self.capture_lifted),
                              'height_above_start_m': float(height),
                              'lift_streak': self.capture_lift_streak,
                              'released_stable': bool(capture_ready), 'placement_streak': self.capture_place_streak,
                              'upright_stable': self.capture_upright_streak >= c.stable_steps,
                              'upright_streak': self.capture_upright_streak,
                              'near_table': bool(near_table),
                              'removal_quality': ('upright' if self.capture_upright_streak >= c.stable_steps else
                                                  'non_upright' if capture_ready else 'incomplete')}
        reward, terminated, truncated, score = super().update(self._adapt(observation))
        if (self.capture_index is not None and self.config.capture_home_retry and
                score['reason'] == 'failed_grasp'):
            # Home is also a valid intermediate/retry pose during a two-piece
            # capture. An ungrasped attacker does not prove that this sequence
            # has finished. Keep the complete-success criteria and full budget.
            terminated = False
            truncated = self.step_count >= self.config.max_steps
            self.done = truncated
            score.update(reason='timeout' if truncated else None, success=False)
            score['home_failure_deferred'] = True
        if order_violation and score['reason'] != 'disturbed_other_piece':
            reward -= score['reward_terms'].pop('success', 0.)
            score['reward_terms']['capture_order'] = -2.
            reward -= 2.
            terminated, truncated, self.done = True, False, True
            score.update(success=False, reason='capture_order_violation')
        if score['success'] and not capture_ready:
            reward -= score['reward_terms'].pop('success')
            score.update(success=False, reason='capture_incomplete')
        if score['success'] and capture_status is not None and capture_status['upright_stable']:
            # Pay at completed move, so toppling after initial release loses this bonus.
            score['reward_terms']['capture_upright_finish'] = .75
            reward += .75
        if lift_bonus:
            score['reward_terms']['capture_lift'] = lift_bonus
            reward += lift_bonus
        if bonus:
            score['reward_terms']['capture_release'] = bonus
            reward += bonus
        score.update(move=self.geometry['move'], capture=capture_status,
                     disturbance_rule='original_square_center_or_tilt_over_30deg',
                     other_pieces_disturbed=score.pop('other_pieces_displaced_over_3mm'),
                     phase='remove_captured' if self.capture_index is not None and not self.capture_bonus else 'move_attacker')
        return reward, terminated, truncated, score
