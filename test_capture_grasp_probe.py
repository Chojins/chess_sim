import unittest

from capture_grasp_probe import alignment


class AlignmentTests(unittest.TestCase):
    def setUp(self):
        self.plan={'settle_position_tolerance_m':.003,'settle_joint_tolerance_deg':.6}
        self.case={'capture_index':0,'fitted_grasp_deg':[1,2,3,4,5,10]}
        self.observation={'measured_joint_deg':[1,2,3,4,5,10],
                          'pieces':{'piece_grasp_targets_m':[[0,0,.017]],'piece_upright_cosines':[1.]},
                          'gripper_grasp_center_m':[0,0,.017],
                          'measured_finger_joint_deg':{'jaw1_hinge':10,'jaw2_hinge':-10}}

    def test_measured_pose_is_required_before_closing(self):
        self.assertTrue(alignment(self.observation,self.plan,self.case)[0])
        self.observation['gripper_grasp_center_m'][2]+=.004
        self.assertFalse(alignment(self.observation,self.plan,self.case)[0])

    def test_arm_tracking_and_both_actual_fingers_are_checked(self):
        self.observation['measured_joint_deg'][1]+=.7
        self.assertFalse(alignment(self.observation,self.plan,self.case)[0])
        self.observation['measured_joint_deg'][1]-=.7
        self.observation['measured_finger_joint_deg']['jaw2_hinge']=-2
        self.assertFalse(alignment(self.observation,self.plan,self.case)[0])

    def test_target_follows_actual_victim(self):
        self.observation['pieces']['piece_grasp_targets_m'][0][0]=.005
        self.assertFalse(alignment(self.observation,self.plan,self.case)[0])

    def test_tipped_victim_cannot_trigger_closing(self):
        self.observation['pieces']['piece_upright_cosines'][0]=.8
        self.assertFalse(alignment(self.observation,self.plan,self.case)[0])


if __name__=='__main__':
    unittest.main()
