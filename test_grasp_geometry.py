import unittest
import numpy as np
from grasp_geometry import FINGER_FRONT_M, GRASP_LOCAL_M, piece_grasp_target
from residual_learning import reward_clearance


class GraspGeometryTests(unittest.TestCase):
    def test_setback_and_piece_heights(self):
        self.assertAlmostEqual(FINGER_FRONT_M + GRASP_LOCAL_M[1], .007)
        for name, mm in [('pawn', 17), ('knight', 17), ('rook', 19), ('bishop', 27), ('king', 32), ('queen', 30)]:
            for floor in [0., .01421]:
                point = piece_grasp_target('/World/chess_pieces_blue/' + name + '_01',
                                           [1, 2, floor], [1.02, 2.02, floor + .05])
                np.testing.assert_allclose(point, [1.01, 2.01, floor + mm / 1000])
        with self.assertRaises(ValueError):
            piece_grasp_target('/pawn', [0, 0, 0], [.01, .01, .01])

    def test_reward_uses_grasp_height_for_victim_then_attacker(self):
        observation = {'gripper_tip_proxy_m': [0, 0, .08], 'gripper_grasp_center_m': [0, 0, .017],
                       'task_geometry': {'source_index': 0, 'capture_index': 1},
                       'pieces': {'piece_centers_m': [[0, 0, .009], [0, 0, .01]],
                                  'piece_grasp_targets_m': [[0, 0, .017], [0, 0, .019]]}}
        info = {'telemetry': {'observation': observation}, 'score': {'phase': 'remove_captured'}}
        self.assertAlmostEqual(reward_clearance(info), .002)
        info['score']['phase'] = 'move_attacker'
        self.assertAlmostEqual(reward_clearance(info), 0.)


if __name__ == '__main__':
    unittest.main()
