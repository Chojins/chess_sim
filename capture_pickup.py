"""Diagnostic pickup-only curriculum; checkpoints are not full capture policies."""
import numpy as np

from chess_task import LegalMoveTask
from residual_learning import approach_potential, reward_clearance


def pickup_potential(gate, info):
    capture = info.get('score', {}).get('capture') or {}
    height = max(0., capture.get('height_above_start_m', 0.))
    proximity = np.clip(1 - reward_clearance(info) / .020, 0, 1)
    return approach_potential(gate, reward_clearance(info)) + 2 * float(np.clip(height / .020, 0, 1) * proximity)


class CapturePickupTask(LegalMoveTask):
    version = 'capture_pickup_v1'

    def __init__(self, initial, config=None):
        super().__init__(initial, config)
        if self.capture_index is None:
            raise ValueError('Pickup-only task requires a capture victim')
        self.hold_streak = 0
        self.pickup_limit = 240

    def update(self, observation):
        reward, terminated, truncated, score = super().update(observation)
        victim = self.capture_index
        center = np.asarray(observation['pieces']['piece_centers_m'][victim])
        target = np.asarray(observation['pieces']['piece_grasp_targets_m'][victim])
        clearance = float(np.linalg.norm(np.asarray(observation['gripper_grasp_center_m']) - target))
        height = center[2] - self.initial_centers[victim, 2]
        held = (height >= .020 and clearance < .015 and
                observation['pieces']['piece_upright_cosines'][victim] > .95)
        self.hold_streak = self.hold_streak + 1 if held else 0
        if not terminated and not truncated and self.hold_streak >= 10:
            score['reward_terms']['pickup_success'] = 10.
            reward += 10.
            score.update(success=True, reason='success')
            terminated, self.done = True, True
        elif not terminated and self.step_count >= self.pickup_limit:
            score.update(success=False, reason='timeout')
            truncated, self.done = True, True
        score.update(objective='capture_pickup_only', hold_streak=self.hold_streak,
                     victim_grasp_clearance_m=clearance)
        return reward, terminated, truncated, score
