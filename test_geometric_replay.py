"""Protect reward matching, successful-only replay and held-out episodes."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.buffers import ReplayBuffer
from capture_demonstrations import BOUNDS, FILTER, file_hash
from geometric_replay import matched_semantics, prefill


class GeometricReplayTests(unittest.TestCase):
    def fixture(self, root):
        manifest = dict(task_version='legal_move_v5', objective='complete move',
                        weights_sha256='frozen', grasp_reference={'version': 'approved'},
                        observation_size=4122, action_window=100, capture_alignment=True,
                        capture_home_retry=True, residual_bounds_deg=BOUNDS.tolist(),
                        joint_map={'scale': [-1, -1, 1, 1, -1, 1.16]},
                        residual_filter=FILTER, gamma=.99, potential_weight=2.,
                        cases=[{'id': 'capture'}], source_seeds=[81, 82], validation_seed=82,
                        collection_roots=[str(root/'collection')], episodes=[],
                        live_timing={'config': {'minimum_arm_adaptation_steps': 32}, 'kinematics_sha256': 'robot'})
        obs, actions = [], []
        for number, seed in enumerate([81, 82]):
            episode = root/str(seed); episode.mkdir()
            replay = episode/'learning_transitions.npz'
            features = np.full((3, 4122), seed, np.float32)+np.arange(3, dtype=np.float32)[:, None]
            raw = np.full((2, 6), .2+number*.1, np.float32)
            np.savez_compressed(replay, observations=features, actions=raw,
                                rewards=np.array([1., 10.], np.float32),
                                terminated=np.array([False, True]), truncated=np.zeros(2, bool))
            (episode/'steps.jsonl').write_text('{}\n{}\n')
            (episode/'decisions.jsonl').write_text('{}\n{}\n')
            (episode/'result.json').write_text(json.dumps({'score': {'success': True,
                'other_pieces_disturbed': 0, 'capture': {'upright_stable': True}}}))
            (episode/'initial.json').write_text(json.dumps({'telemetry': {'observation': {'joint_map': manifest['joint_map']}}}))
            manifest['episodes'].append(dict(path=str(episode), seed=seed, transitions=2,
                replay_file=replay.name, replay_sha256=file_hash(replay), trace_sha256=file_hash(episode/'steps.jsonl'),
                decisions=str(episode/'decisions.jsonl'), decisions_sha256=file_hash(episode/'decisions.jsonl')))
            obs.extend(features[:-1]); actions.extend(raw)
        path = root/'demonstrations.npz'
        np.savez_compressed(path, observations=np.array(obs), actions=np.array(actions), weights=np.ones(4),
                            validation=np.array([False, False, True, True]), episode=np.array([0, 0, 1, 1]))
        manifest['dataset_sha256'] = file_hash(path)
        path.with_suffix('.json').write_text(json.dumps(manifest))
        deployment = {**manifest, 'training_seed': 90, 'evaluation_seeds': [91]}
        return path, manifest, deployment

    def test_validation_is_excluded_and_real_next_observations_are_preserved(self):
        with tempfile.TemporaryDirectory() as temporary, patch('verify_geometric_teacher.audit') as audit:
            path, _, deployment = self.fixture(Path(temporary))
            model = SimpleNamespace(replay_buffer=ReplayBuffer(20,
                spaces.Box(-np.inf, np.inf, (4122,), np.float32), spaces.Box(-1., 1., (6,), np.float32), device='cpu'))
            report = prefill(path, deployment, model)
            audit.assert_called_once()
            self.assertEqual(report['replay_seeds'], [81])
            self.assertEqual(report['replay_transitions'], model.replay_buffer.size())
            self.assertEqual(model.replay_buffer.size(), 2)
            np.testing.assert_equal(model.replay_buffer.observations[:2, 0, 0], [81, 82])
            np.testing.assert_equal(model.replay_buffer.next_observations[:2, 0, 0], [82, 83])
            np.testing.assert_equal(model.replay_buffer.rewards[:2, 0], [1, 10])
            np.testing.assert_equal(model.replay_buffer.dones[:2, 0], [0, 1])

    def test_timing_reward_and_home_semantics_must_match(self):
        with tempfile.TemporaryDirectory() as temporary:
            _, source, deployment = self.fixture(Path(temporary))
            for key, value in [('gamma', .95), ('capture_home_retry', False),
                               ('joint_map', {'scale': [1, 1, 1, 1, 1, 1]}),
                               ('live_timing', {'config': {'minimum_arm_adaptation_steps': 18}, 'kinematics_sha256': 'robot'})]:
                revised = copy.deepcopy(deployment); revised[key] = value
                with self.subTest(key=key), self.assertRaises(ValueError):
                    matched_semantics(source, revised)

    def test_changed_transition_file_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary, patch('verify_geometric_teacher.audit'):
            path, manifest, deployment = self.fixture(Path(temporary))
            replay = Path(manifest['episodes'][0]['path'])/'learning_transitions.npz'
            replay.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'provenance'):
                prefill(path, deployment)

    def test_toppled_capture_is_excluded_even_when_chess_move_succeeded(self):
        with tempfile.TemporaryDirectory() as temporary, patch('verify_geometric_teacher.audit'):
            path, manifest, deployment = self.fixture(Path(temporary))
            result = Path(manifest['episodes'][0]['path'])/'result.json'
            result.write_text(json.dumps({'score': {'success': True, 'other_pieces_disturbed': 0,
                                                  'capture': {'upright_stable': False}}}))
            with self.assertRaisesRegex(ValueError, 'toppled'):
                prefill(path, deployment)


if __name__ == '__main__':
    unittest.main()
