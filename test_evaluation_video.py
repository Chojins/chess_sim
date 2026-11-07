import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from evaluation_video import EpisodeVideo
from train_residual import evaluate


class VideoTests(unittest.TestCase):
    def test_camera_order_and_recording_frame_count(self):
        left = np.full((32, 48, 3), [240, 10, 10], np.uint8)
        right = np.full((32, 48, 3), [10, 10, 240], np.uint8)
        images = {'laptop_rgb': left, 'phone_rgb': right}
        with tempfile.TemporaryDirectory() as directory:
            video = EpisodeVideo(directory, images)
            try:
                for _ in range(4): video.write(images)
            finally:
                video.close()
            self.assertEqual(video.verify()['frames'], 4)
            capture = cv2.VideoCapture(str(video.path))
            ok, frame = capture.read(); capture.release()
            self.assertTrue(ok)
            # OpenCV decodes BGR: fixed camera red left, wrist blue right.
            self.assertGreater(frame[16, 24, 2], 200)
            self.assertGreater(frame[16, 72, 0], 200)
            np.testing.assert_equal(images['laptop_rgb'], left)

    def test_evaluation_records_reset_and_every_transition(self):
        class Env:
            @property
            def unwrapped(self): return self
            def _observation(self):
                image = np.full((32, 48, 3), self.steps * 20, np.uint8)
                return {'laptop_rgb': image, 'phone_rgb': image, 'state': np.zeros(6)}
            def reset(self, **kwargs):
                self.steps = 0
                return np.zeros(8), {}
            def step(self, action):
                self.steps += 1
                return np.zeros(8), 1., self.steps == 3, False, {'score': {'success': self.steps == 3}, 'timing_ms': {'server_physics_ms': 2.}}
        with tempfile.TemporaryDirectory() as directory, patch('train_residual.transition_record', return_value={}):
            rows = evaluate(Env(), None, [9], Path(directory), 'test', record_video=True, profile=True)
            self.assertEqual(rows[0]['steps'], 3)
            self.assertEqual(rows[0]['video']['frames'], 4)
            self.assertEqual(rows[0]['return'], 3.)
            self.assertEqual(rows[0]['timing_ms']['server_physics_ms']['mean'], 2.)
            self.assertTrue((Path(directory) / 'test_9/timing_steps.json').exists())

    def test_exact_residual_replay_preserves_before_and_next_observations(self):
        class Env:
            @property
            def unwrapped(self): return self
            def _observation(self): return {'state': np.zeros(6)}
            def reset(self, **kwargs):
                self.steps = 0
                return np.zeros(8, np.float32), {}
            def step(self, action):
                self.steps += 1
                return np.full(8, self.steps, np.float32), .5, self.steps == 3, False, {'score': {'success': self.steps == 3}}
        with tempfile.TemporaryDirectory() as directory, patch('train_residual.transition_record', return_value={}):
            rows = evaluate(Env(), None, [9], Path(directory), 'teacher', record_learning_transitions=True)
            self.assertEqual(rows[0]['learning_transitions']['frames'], 3)
            with np.load(Path(directory) / 'teacher_9/learning_transitions.npz') as data:
                np.testing.assert_equal(data['observations'], np.repeat(np.arange(4)[:, None], 8, axis=1))
                np.testing.assert_equal(data['actions'], np.zeros((3, 6)))
                np.testing.assert_equal(data['terminated'], [False, False, True])
                np.testing.assert_equal(data['truncated'], [False] * 3)
                np.testing.assert_equal(data['rewards'], [.5] * 3)


if __name__ == '__main__':
    unittest.main()
