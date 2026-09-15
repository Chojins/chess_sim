import unittest

from chess_task import ChessTask, TaskConfig


def initial():
    return {'task_geometry': {'move': 'e2e4', 'source_center_m': [0, 0, .026], 'target_center_m': [0, .045, .026]},
            'measured_joint_deg': [0] * 6, 'gripper_tip_proxy_m': [0, 0, .2],
            'pieces': {'pawn_center_m': [0, 0, .026], 'pawn_upright_cosine': 1., 'other_pieces_displaced_over_3mm': 0}}


class TaskTests(unittest.TestCase):
    def setUp(self):
        self.obs = initial()
        self.task = ChessTask(self.obs)

    def advance(self, count, *, height=0, target=False, home=False, clear=True):
        self.obs['pieces']['pawn_center_m'] = [0, .045 if target else 0, .026 + height]
        self.obs['measured_joint_deg'][0] = 0 if home else 20
        self.obs['gripper_tip_proxy_m'] = [0, .045, .2 if clear else .03]
        result = None
        for _ in range(count):
            result = self.task.update(self.obs)
        return result

    def test_completed_pick_place_home(self):
        first = self.advance(3, height=.01)
        self.assertIn('lift', first[3]['reward_terms'])
        self.advance(3, height=.01, target=True)
        self.advance(1, target=True)
        reward, term, trunc, info = self.advance(10, target=True, home=True)
        self.assertTrue(info['success'] and term and not trunc)
        self.assertGreater(reward, 9)
        with self.assertRaises(RuntimeError):
            self.task.update(self.obs)

    def test_pushing_to_target_is_not_success(self):
        self.advance(1, target=True)
        _, term, _, info = self.advance(10, target=True, home=True)
        self.assertTrue(term)
        self.assertEqual(info['reason'], 'failed_grasp')

    def test_single_frame_bounce_does_not_count(self):
        self.advance(1, height=.01)
        self.advance(1)
        _, _, _, info = self.advance(10, home=True)
        self.assertFalse(info['lifted'])

    def test_lift_reward_cannot_be_repeated(self):
        self.advance(3, height=.01)
        self.advance(3)
        _, _, _, info = self.advance(3, height=.01)
        self.assertNotIn('lift', info['reward_terms'])

    def test_held_piece_and_disturbance(self):
        self.advance(3, height=.01)
        self.advance(1, target=True, clear=False)
        _, _, _, info = self.advance(10, target=True, home=True, clear=False)
        self.assertFalse(info['success'])
        self.task = ChessTask(initial())
        self.obs['pieces']['other_pieces_displaced_over_3mm'] = 1
        _, term, trunc, info = self.task.update(self.obs)
        self.assertTrue(term and not trunc)
        self.assertEqual(info['reason'], 'disturbed_other_piece')

    def test_timeout_and_reset(self):
        task = ChessTask(initial(), TaskConfig(max_steps=1))
        _, term, trunc, info = task.update(initial())
        self.assertTrue(trunc and not term)
        self.assertEqual(info['reason'], 'timeout')
        self.assertFalse(ChessTask(initial()).lifted)

    def test_wrong_square_and_tipped_placement(self):
        self.advance(3, height=.01)
        _, _, _, info = self.advance(10, home=True)
        self.assertEqual(info['reason'], 'wrong_placement')
        self.obs = initial()
        self.task = ChessTask(self.obs)
        self.advance(3, height=.01)
        self.obs['pieces']['pawn_upright_cosine'] = .5
        _, _, _, info = self.advance(10, target=True, home=True)
        self.assertEqual(info['reason'], 'unstable_or_unreleased')


if __name__ == '__main__':
    unittest.main()
