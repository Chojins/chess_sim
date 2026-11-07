"""Training-readiness criteria preserve the user's capture-quality preference."""
import copy
import unittest
from run_geometric_pipeline import screening_readiness


class ScreeningTests(unittest.TestCase):
    def results(self):
        return [{'seed': seed, 'repeat': 0, 'task_id': 'scandinavian_capture',
                 'score': {'success': seed in (1, 5), 'capture': {
                     'lifted': True, 'inside_actor_rhs': seed != 3,
                     'released_stable': seed != 3, 'upright_stable': seed == 1}}}
                for seed in range(1, 6)]

    def test_stable_toppled_victims_allow_training_but_remain_lower_quality(self):
        report = screening_readiness(self.results())
        self.assertTrue(report['passed'])
        self.assertEqual(report['complete_successes'], 2)
        self.assertEqual(report['stable_rhs_removals'], 4)
        self.assertEqual(report['upright_removals'], 1)

    def test_victim_left_on_board_or_still_held_cannot_satisfy_removal_gate(self):
        for key in ('inside_actor_rhs', 'released_stable', 'lifted'):
            rows = self.results(); rows[1]['score']['capture'][key] = False
            with self.subTest(key=key):
                self.assertFalse(screening_readiness(rows)['passed'])

    def test_upright_victims_alone_do_not_replace_legal_move_success(self):
        rows = self.results(); rows[-1]['score']['success'] = False
        for row in rows:
            row['score']['capture']['upright_stable'] = True
        self.assertFalse(screening_readiness(rows)['passed'])

    def test_short_or_duplicated_screen_cannot_qualify(self):
        rows = self.results()
        self.assertFalse(screening_readiness(rows[:4])['passed'])
        rows[-1] = copy.deepcopy(rows[0])
        with self.assertRaises(ValueError):
            screening_readiness(rows)


if __name__ == '__main__':
    unittest.main()
