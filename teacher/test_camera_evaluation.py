"""A camera reply must match the state-teacher command, without retries."""
import unittest
from unittest.mock import Mock,patch

import numpy as np

from teacher.evaluate import CameraStateClient


class CameraEvaluationTests(unittest.TestCase):
    def test_reset_and_uncertain_camera_step_require_matching_frames(self):
        client=CameraStateClient('http://127.0.0.1:8080')
        client.client=Mock()
        def pair(frame,timestamp):
            return ({'frame_id':frame,'simulation_time':timestamp,'observation':{'mode':'policy_control'}},{})
        client.client.read.side_effect=[pair(10,0),pair(11,0),pair(13,1/30)]
        with patch('teacher.evaluate.command',return_value={'frame_id':11}) as command:
            client.reset({'move':'e2e4'})
            self.assertFalse(client.uncertain)
            command.return_value={'frame_id':12}
            with self.assertRaises(RuntimeError):
                client.step(np.zeros(6))
            self.assertTrue(client.uncertain)
            with self.assertRaises(RuntimeError):
                client.step(np.zeros(6))
            self.assertEqual(command.call_count,2)


if __name__=='__main__':
    unittest.main()
