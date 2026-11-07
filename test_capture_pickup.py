import copy
import unittest
import numpy as np
from capture_pickup import CapturePickupTask, pickup_potential
from chess_task import TaskConfig
from test_board_tasks import BoardTaskTests


class PickupTests(unittest.TestCase):
    def initial(self):
        obs = BoardTaskTests().observation()
        obs['pieces']['piece_grasp_targets_m'] = (np.array(obs['pieces']['piece_centers_m']) + [0, 0, .005]).tolist()
        obs['gripper_grasp_center_m'] = [0, 0, .2]
        return obs

    def held(self, obs, task, height=.025):
        i = task.capture_index
        obs['measured_joint_deg'][0] = 20
        obs['pieces']['piece_centers_m'][i] = (task.initial_centers[i] + [0, 0, height]).tolist()
        obs['pieces']['piece_grasp_targets_m'][i] = (task.initial_centers[i] + [0, 0, height+.005]).tolist()
        obs['gripper_grasp_center_m'] = obs['pieces']['piece_grasp_targets_m'][i].copy()

    def test_real_hold_succeeds_only_after_ten_frames(self):
        obs = self.initial(); task = CapturePickupTask(copy.deepcopy(obs), TaskConfig(max_steps=800))
        self.held(obs, task)
        for _ in range(9):
            _, term, trunc, score = task.update(obs)
            self.assertFalse(term or trunc or score['success'])
        reward, term, _, score = task.update(obs)
        self.assertTrue(term and score['success'])
        self.assertGreater(reward, 9.)
        self.assertFalse(score['capture']['released_stable'])
        self.assertEqual(score['objective'], 'capture_pickup_only')

    def test_height_uprightness_and_proximity_all_required(self):
        for condition in ['low', 'distant', 'tipped']:
            obs = self.initial(); task = CapturePickupTask(copy.deepcopy(obs), TaskConfig(max_steps=800))
            self.held(obs, task, .01 if condition == 'low' else .025)
            if condition == 'distant': obs['gripper_grasp_center_m'][2] += .05
            if condition == 'tipped': obs['pieces']['piece_upright_cosines'][task.capture_index] = .5
            for _ in range(12): score = task.update(obs)[3]
            self.assertFalse(score['success'])
            self.assertEqual(task.hold_streak, 0)

    def test_timeout_remains_truncation(self):
        obs = self.initial(); task = CapturePickupTask(copy.deepcopy(obs), TaskConfig(max_steps=800))
        obs['measured_joint_deg'][0] = 20
        for _ in range(240): result = task.update(obs)
        self.assertFalse(result[1]); self.assertTrue(result[2])
        self.assertEqual(result[3]['reason'], 'timeout')

    def test_potential_requires_lift_and_nearby_tool(self):
        obs = self.initial(); task = CapturePickupTask(copy.deepcopy(obs)); self.held(obs, task)
        info = {'telemetry': {'observation': obs}, 'score': {'capture': {'height_above_start_m': .025}, 'phase': 'remove_captured'}}
        self.assertAlmostEqual(pickup_potential(1, info), 3.)
        obs['gripper_grasp_center_m'][0] += .1
        self.assertEqual(pickup_potential(1, info), 0.)
