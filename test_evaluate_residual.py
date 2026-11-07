"""Check counterbalancing and paired counts before a long simulation run."""
import unittest

from evaluate_residual import schedule, summarize


class EvaluationTests(unittest.TestCase):
    def test_counterbalanced_repeated_pairs(self):
        plan = schedule([201, 202, 203], 2)
        self.assertEqual(len(plan), 12)
        for seed in (201, 202, 203):
            first = [p['label'] for p in plan if p['seed'] == seed and p['repeat'] == 0]
            second = [p['label'] for p in plan if p['seed'] == seed and p['repeat'] == 1]
            self.assertEqual(first, second[::-1])
            self.assertEqual(set(first), {'baseline', 'learned'})
        for seeds, repeats in [([1, 1], 2), ([-1], 2), ([], 2), ([1], 1)]:
            with self.assertRaises(ValueError):
                schedule(seeds, repeats)

    def test_partial_and_complete_counts(self):
        rows = [{'seed': 201, 'repeat': repeat, 'label': label,
                 'score': {'success': success, 'reason': 'success' if success else 'failed_grasp'}}
                for repeat, label, success in [(0, 'baseline', False), (0, 'learned', True),
                                               (1, 'learned', False), (1, 'baseline', True)]]
        self.assertEqual(summarize(rows[:1])['matched_pairs'], {})
        result = summarize(rows)
        self.assertEqual(result['matched_pairs'], {'learned_only': 1, 'baseline_only': 1})
        self.assertEqual(result['seeds_with_repeat_disagreement'], {'baseline': 1, 'learned': 1})
        for label in ('baseline', 'learned'):
            self.assertEqual(result[label]['success_fraction'], .5)


if __name__ == '__main__':
    unittest.main()
