"""Generic IK geometry, rate bounds and non-integrating gripper requests."""
import json
import unittest
import numpy as np
from review_grasp_geometry import opening_center
from teacher.environment import ASSETS
from teacher.task_space import CartesianControl


class TaskSpaceTests(unittest.TestCase):
    def setUp(self):
        self.spec=json.loads((ASSETS/'kinematics.json').read_text())
        self.anchors=json.loads((ASSETS/'finger_reference.json').read_text())['anchors_jaw_local_m']
        limits=np.asarray(self.spec['limits']);self.low,self.high=limits[:,0]+1e-4,limits[:,1]-1e-4
        self.control=CartesianControl(self.spec,self.anchors,self.low,self.high,[15.]*5+[45.])
        self.q=np.array([-12.5,88.7,-29.7,6.6,87.,self.low[5]])
        self.actual={'measured_finger_joint_deg':{'jaw1_hinge':2.1,'jaw2_hinge':-2.1}}
    def test_small_inverse_target_matches_fk_and_joint_target(self):
        target=self.q+np.array([.02,.02,-.02,.01,.02,0])
        action=self.control.reference_action(self.q,self.actual,target)
        result=self.control.command(self.q,self.actual,action)
        np.testing.assert_allclose(result[:5],target[:5],atol=2e-4)
        np.testing.assert_allclose(self.control.point(result,self.actual['measured_finger_joint_deg']),
            self.control.point(target,self.actual['measured_finger_joint_deg']),atol=1e-6)
    def test_all_commands_obey_joint_speed_and_authored_limits(self):
        for action in [np.ones(6),-np.ones(6),np.array([1,-1,1,-1,1,-1])]:
            result=self.control.command(self.q,self.actual,action)
            self.assertLessEqual(float(np.max(np.abs(result[:5]-self.q[:5]))),.5000001)
            self.assertLessEqual(abs(result[5]-self.q[5]),1.5000001)
            self.assertTrue(np.all(result>=self.low));self.assertTrue(np.all(result<=self.high))
    def test_small_gripper_output_does_not_accumulate_into_a_release(self):
        q=self.q.copy();closed=np.zeros(6);closed[5]=-.01
        for _ in range(100): q=self.control.command(q,self.actual,closed)
        self.assertAlmostEqual(q[5],self.low[5]);np.testing.assert_allclose(q[:5],self.q[:5])
        opened=np.zeros(6);opened[5]=.01
        for _ in range(20): q=self.control.command(q,self.actual,opened)
        self.assertAlmostEqual(q[5],self.control.config.open_jaw_deg)
    def test_opening_midpoint_uses_both_actual_finger_angles(self):
        q=self.q.copy();q[5]=2.1
        expected=opening_center(self.spec,self.anchors,q,-2.1)
        np.testing.assert_allclose(self.control.point(self.q,self.actual['measured_finger_joint_deg']),expected)
        commanded=opening_center(self.spec,self.anchors,self.q,-self.q[5])
        self.assertGreater(np.linalg.norm(expected-commanded),.0001)


if __name__=='__main__': unittest.main()
