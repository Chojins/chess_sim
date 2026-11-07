import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.buffers import ReplayBuffer
from residual_replay import collect_teacher_replay


class SuccessfulReplayTests(unittest.TestCase):
    def test_fresh_zero_residual_teacher_can_be_saved_and_independently_reloaded(self):
        import gymnasium as gym
        from stable_baselines3 import SAC
        from capture_demonstrations import DemonstrationSAC
        from train_residual import make_learner
        class Env(gym.Env):
            observation_space = spaces.Box(-np.inf, np.inf, (4122,), np.float32)
            action_space = spaces.Box(-1., 1., (6,), np.float32)
            def reset(self, *, seed=None, options=None):
                super().reset(seed=seed)
                return np.zeros(4122, np.float32), {}
            def step(self, action):
                return np.zeros(4122, np.float32), 0., True, False, {}
        model = make_learner(Env(), 'cpu', 35, buffer_size=20, learner_class=DemonstrationSAC)
        observations = np.random.default_rng(35).normal(size=(4, 4122)).astype(np.float32)
        np.testing.assert_equal(model.predict(observations, deterministic=True)[0], np.zeros((4, 6)))
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'replay_teacher.zip'
            model.save(path)
            restored = SAC.load(path, device='cpu')
            np.testing.assert_equal(restored.predict(observations, deterministic=True)[0], np.zeros((4, 6)))

    def test_failed_rollouts_and_validation_seed_never_enter_training_replay(self):
        observation_space = spaces.Box(-np.inf, np.inf, (8,), np.float32)
        action_space = spaces.Box(-1., 1., (6,), np.float32)
        env = SimpleNamespace(unwrapped=SimpleNamespace(fixed_case=None), observation_space=observation_space)
        model = SimpleNamespace(replay_buffer=ReplayBuffer(20, observation_space, action_space, device='cpu'))
        def recorded(env, model, seeds, parent, label, **kwargs):
            self.assertTrue(kwargs['record_learning_transitions'])
            seed = seeds[0]
            path = parent / f'{label}_{seed}'
            path.mkdir()
            obs = np.full((3, 8), seed, np.float32) + np.arange(3)[:, None]
            np.savez_compressed(path / 'learning_transitions.npz', observations=obs,
                                actions=np.full((2, 6), .2, np.float32), rewards=np.array([1., 10.], np.float32),
                                terminated=np.array([False, True]), truncated=np.array([False, False]))
            return [{'seed': seed, 'steps': 2, 'score': {'success': seed != 73, 'other_pieces_disturbed': 0}}]
        with tempfile.TemporaryDirectory() as temporary, patch('residual_replay.evaluate', side_effect=recorded):
            data, report = collect_teacher_replay(env, model, [{'id': 'capture'}], [71, 72, 73], Path(temporary) / 'replay')
        self.assertEqual(report['replay_transitions'], 2)
        self.assertEqual(report['validation_seed'], 72)
        self.assertEqual(report['replay_seeds'], [71])
        self.assertEqual(model.replay_buffer.size(), 2)
        np.testing.assert_equal(model.replay_buffer.observations[:2, 0], np.full((2, 8), 71) + np.arange(2)[:, None])
        np.testing.assert_equal(model.replay_buffer.next_observations[:2, 0], np.full((2, 8), 72) + np.arange(2)[:, None])
        np.testing.assert_equal(model.replay_buffer.rewards[:2, 0], [1., 10.])
        np.testing.assert_equal(model.replay_buffer.dones[:2, 0], [0., 1.])
        self.assertEqual(data['observations'].shape, (4, 8))
        np.testing.assert_equal(data['validation'], [False, False, True, True])
        self.assertIsNone(env.unwrapped.fixed_case)


if __name__ == '__main__':
    unittest.main()
