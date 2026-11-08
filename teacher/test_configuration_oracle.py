import json
import importlib.util
from pathlib import Path
import unittest
import numpy as np
if importlib.util.find_spec('scipy'):
    from teacher.configuration_oracle import ConfigurationReference
else:ConfigurationReference=None
from teacher.environment import ASSETS
from teacher.reactive_oracle import FlowConfig
from teacher.task_space import CartesianControl


@unittest.skipUnless(ConfigurationReference is not None,'configuration-reference fitting uses the dataset environment')
class ConfigurationReferenceTests(unittest.TestCase):
    def setUp(self):
        spec=json.loads((ASSETS/'kinematics.json').read_text());anchors=json.loads((ASSETS/'finger_reference.json').read_text())['anchors_jaw_local_m']
        limits=np.array(spec['limits']);self.control=CartesianControl(spec,anchors,limits[:,0]+1e-4,limits[:,1]-1e-4,[15.]*5+[45.])
        self.oracle=ConfigurationReference(self.control,FlowConfig(grip_pitch_deg=85))
        self.home=np.array([-2.,18.,-50.,56.,90.,6.]);self.goal=np.array([-5.,105.,-50.,30.,90.,6.])
        self.fingers={'jaw1_hinge':6.,'jaw2_hinge':-6.}

    def observation(self,q,grasp):
        x=np.zeros(76,np.float32);x[:6]=q/180;x[21:23]=[.6,-.6];x[70]=1;x[34]=1;x[18]=.9
        x[12:15]=grasp/.03;x[62:67]=(self.home[:5]-q[:5])/90
        x[74]=np.sum(self.home[1:4]-q[1:4])/90;x[75]=(self.home[4]-q[4])/90
        return x

    def test_reach_does_not_unfold_elbow_through_the_servo(self):
        target=self.control.point(self.goal,self.fingers)-[0,0,.025]
        x=self.observation(self.home,target-self.control.point(self.home,self.fingers))
        action,_=self.oracle.predict(x);command=self.control.command(self.home,{'measured_finger_joint_deg':self.fingers},action)
        self.assertGreater(command[1],self.home[1])
        self.assertLess(command[2]-self.home[2],.02)
        self.assertTrue(np.all(np.abs(command-self.home)<=self.control.max_step+1e-5))

    def test_gripper_closes_at_grasp_and_outputs_have_no_clock_or_hidden_phase(self):
        x=self.observation(self.goal,np.zeros(3));a,_=self.oracle.predict(x)
        self.assertEqual(a[5],-1)
        target=self.control.command(self.goal,{'measured_finger_joint_deg':self.fingers},a)
        self.assertLess(target[5],self.goal[5]);np.testing.assert_allclose(target[:5],self.goal[:5],atol=1e-4)
        self.oracle.predict(self.observation(self.home,np.array([.01,.10,-.12])))
        x[67]=.95;b,_=self.oracle.predict(x)
        np.testing.assert_array_equal(a,b)

    def test_open_gripper_returns_after_small_placement_error_instead_of_regrasping(self):
        oracle=ConfigurationReference(self.control,FlowConfig(grip_pitch_deg=85,home_basin_m=.008))
        x=self.observation(self.goal,np.array([.004,0,-.075]));x[18:21]=[0,.004/.1,0]
        x[71:74]=[.01,.05,.08]
        action,_=oracle.predict(x)
        target=self.control.command(self.goal,{'measured_finger_joint_deg':self.fingers},action)
        self.assertEqual(action[5],1)
        self.assertLess(np.linalg.norm(target[:5]-self.home[:5]),np.linalg.norm(self.goal[:5]-self.home[:5]))
        # At first release, hold the arm while the fingers finish opening.
        x[21:23]=[.43,-.43];x[5]=4.3/180;x[12:15]=[.004/.03,0,0]
        action,_=oracle.predict(x)
        self.assertEqual(action[5],1);np.testing.assert_allclose(action[:5],0,atol=1e-5)


if __name__=='__main__':unittest.main()
