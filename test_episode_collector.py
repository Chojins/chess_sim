import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from episode_collector import EpisodeCollector


class CollectorTests(unittest.TestCase):
    def initial(self, capture=None):
        return {'telemetry': {'observation': {'task_geometry': {'capture_index': capture}}}}

    def record(self, collector, frame=3, timestamp=0.):
        state = np.arange(6, dtype=np.float32)
        images = {'state': state, 'laptop_rgb': np.zeros((480, 640, 3), np.uint8),
                  'phone_rgb': np.full((480, 640, 3), 130, np.uint8)}
        before = {'frame_id': frame, 'simulation_time': timestamp,
                  'observation': {'observation_state': state.tolist()}}
        executed = state + .25
        info = {'telemetry': {'frame_id': frame + 1, 'simulation_time': timestamp + 1 / 30},
                'residual': {'requested_legacy': (state + 9).tolist(), 'executed_legacy': executed.tolist()}}
        collector.write(images, before, info)
        return state, images, executed

    def score(self):
        return {'success': True, 'reason': 'success', 'placement_streak': 10, 'home_streak': 10,
                'lifted': True, 'transported': True, 'other_pieces_disturbed': 0}

    def test_lossless_images_before_state_and_executed_actions_are_aligned(self):
        with tempfile.TemporaryDirectory() as directory:
            collector = EpisodeCollector(Path(directory) / 'episode', self.initial(), {'policy_mode': 'deterministic'})
            state, images, executed = self.record(collector)
            manifest = collector.finalize(self.score())
            self.assertTrue(manifest['accepted_for_act_training'])
            with np.load(collector.directory / 'frame_000000.npz') as frame:
                np.testing.assert_equal(frame['observation_state'], state)
                np.testing.assert_equal(frame['action_legacy'], executed)
                np.testing.assert_equal(frame['phone_rgb'], images['phone_rgb'])
                self.assertEqual(frame['frame_id'], 3)
                self.assertEqual(frame['next_frame_id'], 4)

    def test_pickup_success_errors_and_incomplete_capture_are_not_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            for name, score, error, capture in [('pickup', {**self.score(), 'objective': 'capture_pickup_only'}, None, None),
                                               ('error', self.score(), 'transport failure', None),
                                               ('unfinished_capture', self.score(), None, 24),
                                               ('bad_home', {**self.score(), 'home_streak': 2}, None, None)]:
                collector = EpisodeCollector(Path(directory) / name, self.initial(capture), {})
                self.record(collector)
                self.assertFalse(collector.finalize(score, error)['accepted_for_act_training'])

    def test_lost_frame_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            collector = EpisodeCollector(Path(directory) / 'episode', self.initial(), {})
            self.record(collector)
            with self.assertRaises(ValueError):
                self.record(collector, frame=5, timestamp=3 / 30)

    def test_upright_capture_quality_is_required_for_training_only(self):
        with tempfile.TemporaryDirectory() as directory:
            for upright in (True, False):
                collector = EpisodeCollector(Path(directory) / str(upright), self.initial(24), {})
                self.record(collector)
                score = {**self.score(), 'capture': {'released_stable': True, 'upright_stable': upright}}
                self.assertEqual(collector.finalize(score)['accepted_for_act_training'], upright)

    def test_complete_placement_does_not_require_transient_transport_reward(self):
        with tempfile.TemporaryDirectory() as directory:
            collector=EpisodeCollector(Path(directory)/'valid',self.initial(24),{})
            self.record(collector)
            score={**self.score(),'transported':False,
                   'capture':{'released_stable':True,'upright_stable':True}}
            self.assertTrue(collector.finalize(score)['accepted_for_act_training'])
            collector=EpisodeCollector(Path(directory)/'not_lifted',self.initial(24),{})
            self.record(collector)
            self.assertFalse(collector.finalize({**score,'lifted':False})['accepted_for_act_training'])


if __name__ == '__main__':
    unittest.main()
