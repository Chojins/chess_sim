import unittest
from unittest.mock import patch

import numpy as np

from chess_task import TaskConfig
from grasp_geometry import GRASP_REFERENCE
from residual_learning import reward_clearance
from train_openings import OpeningChessEnv, move_features


class OpeningLearningTests(unittest.TestCase):
    def test_capture_reward_targets_victim_until_release(self):
        observation = {'gripper_tip_proxy_m': [0, 0, 0],
                       'task_geometry': {'capture_index': 1},
                       'pieces': {'pawn_center_m': [0, 0, .2], 'piece_centers_m': [[0, 0, .2], [0, 0, .01]]}}
        info = {'telemetry': {'observation': observation}}
        self.assertAlmostEqual(reward_clearance(info), .01)
        info['score'] = {'phase': 'remove_captured', 'clearance_m': .2}
        self.assertAlmostEqual(reward_clearance(info), .01)
        info['score']['phase'] = 'move_attacker'
        self.assertAlmostEqual(reward_clearance(info), .2)

    def test_round_robin_and_fixed_evaluation_case(self):
        cases = [{'id': 'normal', 'history': [], 'move': 'e2e4', 'capture': False},
                 {'id': 'capture', 'history': ['e2e4', 'd7d5'], 'move': 'e4d5', 'capture': True}]
        env = OpeningChessEnv.__new__(OpeningChessEnv)
        env.cases, env.fixed_case, env.task_cursor = cases, None, 0
        env.task_config = TaskConfig()
        with patch('chess_env.ChessEnv.reset', return_value=({}, {'telemetry': {'observation': {'grasp_reference': GRASP_REFERENCE, 'pieces': {'piece_grasp_targets_m': []}}}})) as reset:
            for i in range(4):
                env.reset(seed=501 if i == 0 else None)
                self.assertEqual(env.current_case, cases[i % 2])
                self.assertEqual(env.task_config.max_steps, 800 if i % 2 else 400)
                self.assertEqual(reset.call_args.kwargs['options']['move'], cases[i % 2]['move'])
            env.fixed_case = cases[1]
            env.reset(seed=601)
            self.assertEqual(env.current_case, cases[1])
            self.assertEqual(move_features(cases[1]).shape, (5,))
            self.assertTrue(np.isfinite(move_features(cases[1])).all())
            self.assertEqual(move_features(cases[1])[-1], 1)


if __name__ == '__main__':
    unittest.main()
