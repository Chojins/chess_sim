"""Reject tampered collection/physics traces before ACT export."""
import copy
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from chess_task import LegalMoveTask, TaskConfig
from episode_collector import EpisodeCollector
import test_board_tasks as fixtures
from verify_collected_episodes import audit_episode


class CollectionAuditTests(unittest.TestCase):
    def fixture(self, root, over_target_while_lifted=True):
        root = Path(root)
        obs = fixtures.BoardTaskTests().observation()
        obs['observation_state'] = [0.] * 6
        obs['joint_map'] = {'scale': [1.] * 6, 'legacy_home': [0.] * 6, 'sim_home_deg': [0.] * 6,
                            'legacy_names': ['pan', 'lift', 'elbow', 'wrist', 'roll', 'jaw']}
        initial = {'telemetry': {'frame_id': 100, 'simulation_time': 0., 'observation': copy.deepcopy(obs)},
                   'task_version': LegalMoveTask.version, 'task_config': asdict(TaskConfig())}
        (root / 'initial.json').write_text(json.dumps(initial))
        images = {'state': np.zeros(6, np.float32), 'laptop_rgb': np.zeros((480, 640, 3), np.uint8),
                  'phone_rgb': np.broadcast_to(np.array([15, 130, 240], np.uint8), (480, 640, 3)).copy()}
        np.savez_compressed(root / 'initial_images.npz', **images)
        provenance = {'act_sha256': 'synthetic-test-only', 'calibration': obs['joint_map'], 'policy_mode': 'deterministic evaluation; exploration disabled',
                      'residual_mode': 'frozen actor', 'residual_actor_state_sha256': 'example'}
        collector = EpisodeCollector(root / 'lossless_episode', initial, provenance)
        task = LegalMoveTask(copy.deepcopy(obs))
        source, victim = task.source_index, task.capture_index
        zone = obs['task_geometry']['capture_zone']
        parking = [*((np.array(zone['min_xy_m']) + zone['max_xy_m']) / 2), .012]
        target = np.asarray(obs['task_geometry']['target_center_m'])
        before, rows = copy.deepcopy(initial['telemetry']), []
        with (root / 'steps.jsonl').open('w') as stream:
            for index in range(40):
                obs['measured_joint_deg'][0] = 20. if index < 18 else 0.
                obs['pieces']['piece_centers_m'][victim] = (task.initial_centers[victim] + [0, 0, .01]).tolist() if index < 3 else parking
                if index >= 14:
                    lifted_point=target if over_target_while_lifted else task.initial_centers[source]
                    obs['pieces']['piece_centers_m'][source] = (lifted_point + [0, 0, .01]).tolist() if index < 17 else target.tolist()
                obs['observation_state'] = obs['measured_joint_deg'].copy()
                _, terminated, truncated, score = task.update(obs)
                frame, timestamp = 101 + index, (index + 1) / 30
                row = {'step': index + 1, 'frame_id': frame, 'simulation_time': timestamp,
                       'state': obs['observation_state'].copy(), 'placement_telemetry': copy.deepcopy(obs),
                       'residual': {'executed_legacy': obs['observation_state'].copy()},
                       'score': score, 'terminated': terminated, 'truncated': truncated}
                before_images = {**images, 'state': np.asarray(before['observation']['observation_state'], np.float32)}
                collector.write(before_images, before, {'telemetry': {'frame_id': frame, 'simulation_time': timestamp}, 'residual': row['residual']})
                stream.write(json.dumps(row) + '\n')
                rows.append(row)
                before = {'frame_id': frame, 'simulation_time': timestamp, 'observation': copy.deepcopy(obs)}
                if terminated or truncated:
                    break
        self.assertTrue(score['success'])
        self.assertTrue(collector.finalize(score)['accepted_for_act_training'])
        return root / 'lossless_episode', rows

    def test_complete_capture_is_recomputed_and_all_frame_hashes_are_indexed(self):
        with tempfile.TemporaryDirectory() as root:
            directory, rows = self.fixture(root)
            audited = audit_episode(directory)
            self.assertTrue(audited['capture'])
            self.assertEqual(audited['frames'], len(rows))
            self.assertEqual(len(audited['file_sha256']), len(rows) + 4)

    def test_full_physics_success_without_transient_transport_bonus_is_valid(self):
        with tempfile.TemporaryDirectory() as root:
            directory,rows=self.fixture(root,over_target_while_lifted=False)
            self.assertFalse(rows[-1]['score']['transported'])
            self.assertTrue(rows[-1]['score']['lifted'])
            self.assertTrue(rows[-1]['score']['success'])
            self.assertEqual(audit_episode(directory)['frames'],len(rows))

    def test_accepted_flag_does_not_hide_an_invalid_physics_outcome(self):
        with tempfile.TemporaryDirectory() as root:
            directory, rows = self.fixture(root)
            victim = rows[-1]['placement_telemetry']['task_geometry']['capture_index']
            rows[-1]['placement_telemetry']['pieces']['piece_centers_m'][victim][0] += 1.
            (directory.parent / 'steps.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
            with self.assertRaisesRegex(ValueError, 'Recomputed physics outcome differs'):
                audit_episode(directory)

    def test_action_must_match_executed_target_not_proposal(self):
        with tempfile.TemporaryDirectory() as root:
            directory, rows = self.fixture(root)
            rows[0]['residual']['executed_legacy'][0] += .5
            (directory.parent / 'steps.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
            with self.assertRaises(AssertionError):
                audit_episode(directory)


if __name__ == '__main__':
    unittest.main()
