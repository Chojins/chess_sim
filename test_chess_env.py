"""Transport failure and residual action semantics without running Isaac."""
import copy
import unittest
from unittest.mock import patch

import numpy as np

from chess_env import ChessEnv, ResidualACTEnv
from test_chess_task import initial


class FakeServer:
    def __init__(self):
        obs = initial()
        obs.update(mode='policy_control', observation_state=[0] * 6,
            joint_limits_deg=[[-100, 100]] * 5 + [[-3, 45]],
            joint_map={'scale': [-1, -1, 1, 1, -1, 1.16279296875],
                       'sim_home_deg': [0] * 6, 'legacy_home': [0] * 6})
        self.meta = {'frame_id': 0, 'simulation_time': 0., 'observation': obs}
        self.images = {name: np.zeros((480, 640, 3), np.uint8) for name in ('red', 'blue')}
        self.failure = None

    def read(self, **kwargs):
        result = copy.deepcopy(self.meta)
        if self.failure == 'mismatch':
            result['frame_id'] += 1
        return result, self.images

    def command(self, url, path, frame, action=None, options=None):
        assert frame == self.meta['frame_id']
        self.meta['frame_id'] += 1
        if path == '/step':
            self.meta['simulation_time'] += 1 / 30
            self.meta['observation']['observation_state'] = list(action)
            self.meta['observation']['measured_joint_deg'] = (
                np.asarray(action) * self.meta['observation']['joint_map']['scale']).tolist()
        else:
            self.meta['simulation_time'] = 0.
        if self.failure == 'lost_reply':
            raise OSError('Reply lost after execution')
        return {'frame_id': self.meta['frame_id']}


class EnvTests(unittest.TestCase):
    def setUp(self):
        self.server = FakeServer()
        self.patches = [patch('chess_env.PairedCameraClient', return_value=self.server),
                        patch('chess_env.command', side_effect=self.server.command)]
        for item in self.patches:
            item.start()
            self.addCleanup(item.stop)

    def test_uncertain_step_requires_reset(self):
        env = ChessEnv()
        obs, _ = env.reset()
        frame = env.latest['frame_id']
        self.server.failure = 'lost_reply'
        with self.assertRaises(OSError):
            env.step(obs['state'])
        self.assertEqual(env.latest['frame_id'], frame)
        self.assertEqual(env.task.step_count, 0)
        with self.assertRaises(RuntimeError):
            env.step(obs['state'])
        self.server.failure = None
        env.reset()
        self.assertFalse(env.needs_reset)

    def test_malformed_server_requests_fail_before_mutation(self):
        from sim_control import SimControl
        control = SimControl.__new__(SimControl)
        # No driver is installed: attempting to mutate it would fail this test.
        for options in [{'pawn_offset_mm': {'x': 0}}, {'pawn_offset_mm': [3, 0]}, {'unknown': 1}]:
            with self.assertRaises(ValueError):
                control.reset(options)
        with self.assertRaises(ValueError):
            control.step({'bad': 'action'})

    def test_mismatched_pair_is_not_a_transition(self):
        env = ChessEnv()
        obs, _ = env.reset()
        self.server.failure = 'mismatch'
        with self.assertRaises(RuntimeError):
            env.step(obs['state'])
        self.assertTrue(env.needs_reset)
        self.assertEqual(env.task.step_count, 0)

    def test_residual_uses_physical_degrees_and_timeout_has_next_base(self):
        class Policy:
            config = {'n_action_steps': 100}
            calls = 0
            def reset(self):
                pass
            def action(self, state, *_):
                self.calls += 1
                return state.copy()
        policy = Policy()
        env = ChessEnv(max_steps=1)
        residual = ResidualACTEnv(env, policy)
        obs, _ = residual.reset()
        obs, _, terminated, truncated, info = residual.step(np.ones(6, np.float32) * .5)
        self.assertTrue(truncated and not terminated)
        actual_delta_deg = np.array(info['residual']['executed_legacy']) * env.scale
        np.testing.assert_allclose(actual_delta_deg, [1, 1, 1, 1, 1, 1.5], atol=1e-6)
        self.assertEqual(policy.calls, 2)  # timeout observation must support bootstrapping
        self.assertTrue(residual.observation_space.contains(obs))


if __name__ == '__main__':
    unittest.main()
