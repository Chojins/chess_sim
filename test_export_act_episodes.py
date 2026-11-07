"""Real LeRobot writer/reader round trip, using a synthetic temporary episode."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from export_act_episodes import export_episodes
import test_collected_episode_audit as fixtures
from verify_collected_episodes import audit_episode


@unittest.skipUnless(importlib.util.find_spec('datasets'), 'Run with the separate dataset environment')
class DatasetExportTests(unittest.TestCase):
    def test_lossless_camera_and_legacy_action_round_trip(self):
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / 'source'
            source.mkdir()
            episode, rows = fixtures.CollectionAuditTests().fixture(source)
            index = root / 'index.json'
            index.write_text(json.dumps({'format': 'chess_sim_audited_episodes_v1', 'episodes': [audit_episode(episode)], 'frames': len(rows)}))
            target = root / 'dataset'
            result = export_episodes(index, target, 'local/chess-sim-export-test')
            self.assertFalse(result['uploaded'])
            dataset = LeRobotDataset('local/chess-sim-export-test', root=target)
            self.assertEqual(dataset.num_episodes, 1)
            self.assertEqual(len(dataset), len(rows))
            self.assertEqual(dataset.fps, 30)
            for number in (0, 1, len(rows) - 1):
                data = dataset[number]
                with np.load(episode / f'frame_{number:06d}.npz') as expected:
                    np.testing.assert_equal(data['observation.state'].numpy(), expected['observation_state'])
                    np.testing.assert_equal(data['action'].numpy(), expected['action_legacy'])
                    for camera, name in (('laptop', 'laptop_rgb'), ('phone', 'phone_rgb')):
                        decoded = data[f'observation.images.{camera}'].permute(1, 2, 0).numpy()
                        np.testing.assert_equal(np.rint(decoded * 255).astype(np.uint8), expected[name])
                self.assertAlmostEqual(float(data['timestamp']), number / 30, places=6)


if __name__ == '__main__':
    unittest.main()
