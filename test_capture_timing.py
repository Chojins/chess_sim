import unittest
import numpy as np
from capture_timing import closure_window, settled_fingers


class TimingTests(unittest.TestCase):
    def test_choose_low_pose_before_open_jaw_lift_using_only_commands(self):
        class Kinematics:
            def point(self, q):
                return np.array([0, 0, q[2]])
        commands = np.zeros((8, 6))
        commands[:, 1] = 100
        commands[:, 2] = [.1, .08, .04, .03, .04, .05, .07, .09]
        commands[:, 5] = [8, 9, 10, 10, 8, 4, 0, -1]
        hold, close, z = closure_window(commands, Kinematics())
        self.assertEqual((hold, close), (3, 6))
        self.assertEqual(commands[hold, 5], 10)
        self.assertGreater(z[close], z[hold])

    def test_jaw_settling_accepts_contact_gap_and_rejects_open_or_moving(self):
        held = np.array([0, 100, -50, 30, 0, 0])
        def obs(jaw, opposite=None, arm_offset=0):
            joints = held.copy().astype(float)
            joints[1] += arm_offset
            return {'measured_joint_deg': joints.tolist(),
                    'measured_finger_joint_deg': {'jaw1_hinge': jaw, 'jaw2_hinge': -jaw if opposite is None else opposite}}
        self.assertTrue(settled_fingers(obs(3.1), obs(3.2), held))
        self.assertTrue(settled_fingers(obs(0), obs(.1), held))  # Empty closure is not proof of pickup.
        self.assertFalse(settled_fingers(obs(10), obs(10), held))
        self.assertFalse(settled_fingers(obs(3), obs(4), held))
        self.assertFalse(settled_fingers(obs(3, opposite=-9), obs(3, opposite=-9), held))
        self.assertFalse(settled_fingers(obs(3, arm_offset=2), obs(3), held))


if __name__ == '__main__':
    unittest.main()
