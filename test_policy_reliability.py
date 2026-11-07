import unittest
from verify_policy_reliability import assess


class ReliabilityTests(unittest.TestCase):
    def manifest(self):
        return {'mode': 'frozen_evaluation_only', 'task_version': 'legal_move_v5', 'objective': 'complete move',
                'cases': [{'id': 'capture', 'capture': True}, {'id': 'ordinary', 'capture': False}]}

    def rows(self, capture=18, ordinary=20):
        return [{'label': 'learned', 'task_id': task, 'seed': 5000 + i // 2, 'repeat': i % 2,
                 'score': {'success': i < successes, 'reason': 'success' if i < successes else 'failed_grasp',
                           'capture': {'upright_stable': i < 12} if task == 'capture' else None}}
                for task, successes in [('capture', capture), ('ordinary', ordinary)] for i in range(20)]

    def test_complete_moves_pass_per_task_while_capture_quality_is_reported(self):
        report = assess(self.manifest(), self.rows())
        self.assertTrue(report['passed'])
        self.assertEqual(report['per_task']['capture']['upright_training_quality_successes'], 12)

    def test_ordinary_success_does_not_mask_unreliable_capture(self):
        self.assertFalse(assess(self.manifest(), self.rows(capture=16))['passed'])

    def test_pickup_or_selection_success_cannot_certify_complete_teacher(self):
        for change in ({'objective': 'pickup-only diagnostic; not full capture success'}, {'mode': 'training'}):
            with self.assertRaises(ValueError):
                assess({**self.manifest(), **change}, self.rows(capture=20))

    def test_capture_only_and_small_evaluations_do_not_pass(self):
        manifest = self.manifest()
        manifest['cases'] = manifest['cases'][:1]
        self.assertFalse(assess(manifest, self.rows())['passed'])
        self.assertFalse(assess(self.manifest(), self.rows()[:4])['passed'])


if __name__ == '__main__':
    unittest.main()
