import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import numpy as np
from review_grasp_geometry import joint_transforms,transform
from teacher.robot_clearance import CameraClearance,ClearanceReward,clearance_reward_contract,clearance_buffer_cost


class ClearanceBufferTests(unittest.TestCase):
    def test_buffer_penalty_precedes_the_hard_gate_and_never_rewards_small_gaps(self):
        contract={'buffer_m':.008,'minimum_m':.005,'weight':.04}
        self.assertEqual(clearance_buffer_cost(.010,contract),0.)
        self.assertEqual(clearance_buffer_cost(.008,contract),0.)
        self.assertGreater(clearance_buffer_cost(.007,contract),0.)
        self.assertAlmostEqual(clearance_buffer_cost(.005,contract),.04)
        self.assertGreater(clearance_buffer_cost(.0049,contract),clearance_buffer_cost(.005,contract))
        for value in (float('nan'),-.001):
            with self.assertRaises(ValueError):clearance_buffer_cost(value,contract)
        with self.assertRaises(ValueError):clearance_reward_contract('/missing',.005,.005,.04)


@unittest.skipUnless(importlib.util.find_spec('fcl'),'mesh-clearance audit dependency is in the dataset environment')
class RobotClearanceTests(unittest.TestCase):
    def test_known_triangle_gaps_survive_body_transforms_and_base_rotation(self):
        spec=json.loads(Path('teacher/assets/kinematics.json').read_text());q=np.array([-2,18,-50,56,90,6.])
        frames=joint_transforms(spec,q);triangle=np.array([[0,0,0],[.02,0,0],[0,.02,0]],float)
        arrays={}
        for i,name in enumerate((CameraClearance.camera_name,*CameraClearance.obstacle_names)):
            matrix=frames['/World/so100_blue/'+name.split('__')[0]]
            world=triangle+[0,0,i*.003]
            arrays[name+'_vertices']=transform(world,np.linalg.inv(matrix));arrays[name+'_faces']=np.array([[0,1,2]],np.int32)
        with TemporaryDirectory() as tmp:
            path=Path(tmp)/'meshes.npz';np.savez(path,**arrays);model=CameraClearance(path)
            expected=np.arange(1,5)*.003
            np.testing.assert_allclose(model.distances(q),expected,atol=1e-7,rtol=0)
            contract=clearance_reward_contract(path)
            measured=ClearanceReward(contract).evaluate({'measured_joint_deg':q,'measured_finger_joint_deg':{'jaw2_hinge':-6.}})
            self.assertAlmostEqual(measured['minimum_gap_m'],.003,places=7)
            self.assertAlmostEqual(measured['cost'],clearance_buffer_cost(.003,contract),places=6)
            with self.assertRaises(ValueError):ClearanceReward({**contract,'sha256':'changed'})
            q[0]+=25
            np.testing.assert_allclose(model.distances(q),expected,atol=1e-7,rtol=0)


if __name__=='__main__':unittest.main()
