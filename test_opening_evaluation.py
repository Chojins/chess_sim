import unittest

from evaluate_openings import environment_settings, opening_schedule


class EvaluationSettingsTests(unittest.TestCase):
    def base(self):
        return {'gamma':.99, 'potential_weight':2., 'residual_bounds_deg':[2,2,2,2,2,3],
                'objective':'complete move'}

    def test_alignment_checkpoint_restores_wider_bounds_filter_and_pickup_objective(self):
        source=self.base()
        source.update(objective='pickup-only diagnostic; not full capture success',
                      capture_alignment=True,pickup_limit_steps=240,
                      residual_bounds_deg=[6,6,6,6,6,3],
                      residual_filter={'alpha':.25,'max_change_deg':[.4,.4,.4,.4,.4,.6]})
        pickup,settings=environment_settings(source)
        self.assertTrue(pickup)
        self.assertEqual(settings['bounds_deg'],[6,6,6,6,6,3])
        self.assertTrue(settings['capture_alignment'])
        self.assertEqual(settings['residual_alpha'],.25)
        self.assertEqual(settings['max_residual_change_deg'],[.4,.4,.4,.4,.4,.6])

    def test_old_full_move_models_keep_original_semantics(self):
        pickup,settings=environment_settings(self.base())
        self.assertFalse(pickup or settings['capture_alignment'])
        self.assertEqual(settings['bounds_deg'],[2,2,2,2,2,3])
        self.assertEqual(settings['residual_alpha'],1.)
        self.assertIsNone(settings['max_residual_change_deg'])

    def test_incompatible_alignment_objective_and_pickup_limit_are_rejected(self):
        source=self.base();source['capture_alignment']=True
        with self.assertRaises(ValueError):environment_settings(source)
        source.update(objective='pickup-only diagnostic; not full capture success',pickup_limit_steps=480)
        with self.assertRaises(ValueError):environment_settings(source)

    def test_complete_move_alignment_requires_explicit_timing_protocol(self):
        source=self.base()
        source.update(capture_alignment=True, live_timing={'config': {}}, residual_bounds_deg=[6,6,6,6,6,3])
        pickup, settings=environment_settings(source)
        self.assertFalse(pickup)
        self.assertTrue(settings['capture_alignment'])

    def test_single_policy_reliability_keeps_all_seeds_and_repeats(self):
        plan = opening_schedule([1, 2, 3], 2, 'learned')
        self.assertEqual(len(plan), 6)
        self.assertEqual({(p['seed'], p['repeat']) for p in plan},
                         {(seed, repeat) for seed in (1, 2, 3) for repeat in (0, 1)})
        self.assertEqual({p['label'] for p in plan}, {'learned'})
        self.assertEqual(len(opening_schedule([1, 2, 3], 2)), 12)
        with self.assertRaises(ValueError):
            opening_schedule([1, 2, 3], 2, 'unknown')

    def test_single_policy_screening_needs_no_counterbalanced_pair(self):
        self.assertEqual(opening_schedule([11, 12], 1, 'learned'),
                         [{'repeat': 0, 'seed': seed, 'label': 'learned'} for seed in [11, 12]])
        for seeds, repeats in [([11, 11], 1), ([-1], 1), ([11], 0), ([], 1)]:
            with self.assertRaises(ValueError):
                opening_schedule(seeds, repeats, 'learned')
        with self.assertRaises(ValueError):
            opening_schedule([11, 12], 1)


if __name__=='__main__':
    unittest.main()
