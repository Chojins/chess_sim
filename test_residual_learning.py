"""Offline numerical tests. Synthetic learner test is not chess training."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
from stable_baselines3 import SAC

from residual_learning import LearningResidualEnv, approach_potential, grasp_gate
from chess_env import ChessEnv
from test_chess_env import FakeServer
from train_residual import make_learner


class ToyEnv(gym.Env):
    observation_space = spaces.Box(-1, 1, (8,), np.float32)
    action_space = spaces.Box(-1, 1, (6,), np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        return np.zeros(8, np.float32), {}

    def step(self, action):
        self.steps += 1
        obs = np.r_[action, self.steps / 20, 0].astype(np.float32)
        return obs, -float(np.mean(action ** 2)), self.steps == 20, False, {}


class ResidualTests(unittest.TestCase):
    def test_wrapper_gating_and_episode_boundary_shaping(self):
        class Policy:
            config = {'n_action_steps': 100}
            def reset(self):
                pass
            def action(self, state, *_):
                return state.copy()

        class Encoder:
            output_dim = 4
            def __call__(self, obs):
                return np.zeros(self.output_dim, np.float32)

        for shoulder, terminal in [(0., False), (70., False), (70., True)]:
            with self.subTest(shoulder=shoulder, terminal=terminal):
                server = FakeServer()
                state = server.meta['observation']
                state['observation_state'][1] = -shoulder
                state['measured_joint_deg'][1] = shoulder
                state['gripper_tip_proxy_m'] = (np.array(state['pieces']['pawn_center_m']) + [0, 0, .015]).tolist()
                with patch('chess_env.PairedCameraClient', return_value=server), patch('chess_env.command', side_effect=server.command):
                    env = LearningResidualEnv(ChessEnv(max_steps=1), Policy(), encoder=Encoder())
                    obs, _ = env.reset(seed=1)
                    self.assertTrue(env.observation_space.contains(obs))
                    old_potential, gate = env.potential, env.gate
                    score = {'clearance_m': .015}
                    with patch.object(env.unwrapped.task, 'update', return_value=(0., terminal, not terminal, score)):
                        obs, reward, terminated, truncated, info = env.step(np.ones(6, np.float32))
                    self.assertEqual((terminated, truncated), (terminal, not terminal))
                    np.testing.assert_allclose(info['learning']['applied_residual'], np.full(6, gate))
                    np.testing.assert_allclose(obs[16:22], np.full(6, gate))
                    self.assertTrue(env.observation_space.contains(obs))
                    expected_next = 0. if terminal else approach_potential(env.gate, .015)
                    self.assertAlmostEqual(info['learning']['shaping_reward'], 2 * (.99 * expected_next - old_potential))
                    self.assertAlmostEqual(reward, info['learning']['shaping_reward'] - info['residual']['penalty'])
                    with self.assertRaises(RuntimeError):
                        env.step(np.zeros(6, np.float32))
                    env.close()

    def test_gate_and_potential(self):
        self.assertEqual(grasp_gate([0, 18, 0, 0, 0, 0]), 0)
        self.assertEqual(grasp_gate([0, 100, 0, 0, 0, 0]), 1)
        self.assertEqual(grasp_gate([0, 150, 0, 0, 0, 0]), 0)
        self.assertAlmostEqual(approach_potential(1, .015), .5)
        self.assertEqual(approach_potential(0, 0), 0)
        gamma = .99
        potentials = [0., .5, .8, 0.]
        shaped = [gamma * b - a for a, b in zip(potentials, potentials[1:])]
        self.assertAlmostEqual(sum(gamma ** t * r for t, r in enumerate(shaped)), 0.)
        # A timeout keeps its next-state potential for bootstrapping.
        self.assertNotEqual(gamma * .8 - .5, -.5)

    def test_sac_updates_and_round_trip(self):
        torch.set_num_threads(2)
        env = ToyEnv()
        model = make_learner(env, 'cpu', 8, buffer_size=512)
        obs, _ = env.reset(seed=8)
        np.testing.assert_equal(model.predict(obs, deterministic=True)[0], np.zeros(6))
        before = [p.detach().clone() for p in model.actor.parameters()]
        model.learn(320)
        self.assertGreater(model._n_updates, 0)
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(before, model.actor.parameters())))
        self.assertTrue(all(torch.isfinite(p).all() for p in model.policy.parameters()))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'residual'
            model.save(path)
            loaded = SAC.load(path, device='cpu')
            np.testing.assert_allclose(model.predict(obs, deterministic=True)[0],
                                       loaded.predict(obs, deterministic=True)[0], atol=1e-7)


if __name__ == '__main__':
    unittest.main()
