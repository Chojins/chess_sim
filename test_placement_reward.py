import copy
import unittest
import numpy as np
from residual_learning import placement_potential, grasp_gate


class PlacementRewardTests(unittest.TestCase):
    def info(self):
        return {'score': {'lifted': True, 'phase': 'move_attacker'}, 'telemetry': {'observation': {
            'task_geometry': {'source_index': 0, 'capture_index': 1, 'target_center_m': [0, 0, .03],
                              'capture_zone': {'min_xy_m': [.1, .1], 'max_xy_m': [.2, .2]},
                              'capture_rest_center_z_m': .015},
            'pieces': {'piece_centers_m': [[0, 0, .03], [.15, .15, .015]]}}}}

    def test_alignment_and_lowering(self):
        info = self.info()
        self.assertEqual(placement_potential(info), 1.)
        info['telemetry']['observation']['pieces']['piece_centers_m'][0][1] = .0225
        self.assertAlmostEqual(placement_potential(info), np.exp(-1))
        info['telemetry']['observation']['pieces']['piece_centers_m'][0][2] += .02
        self.assertAlmostEqual(placement_potential(info), np.exp(-2))
        info['score']['lifted'] = False
        self.assertEqual(placement_potential(info), 0.)

    def test_capture_zone_then_attacker(self):
        info = self.info();info['score'].update(phase='remove_captured', capture={'lifted': True})
        info['telemetry']['observation']['pieces']['piece_centers_m'][0][0] = 1.
        self.assertEqual(placement_potential(info), 1.)
        info['score']['capture']['lifted'] = False
        self.assertEqual(placement_potential(info), 0.)
        info['score']['phase'] = 'move_attacker'
        self.assertLess(placement_potential(info), .001)

    def test_release_gate_already_active(self):
        self.assertEqual(grasp_gate([0, 116.8, 0, 0, 0, 0]), 1.)
