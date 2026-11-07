"""Supervised setpoint perturbations respect joint limits and physical units."""
import unittest
import torch
import json
from teacher.bootstrap_move import augment_setpoints,augment_free_arm
from teacher.environment import ASSETS
from teacher.kinematics import OpeningMidpoint


class BootstrapMoveTests(unittest.TestCase):
    def test_same_next_target_and_unchanged_geometry(self):
        x=torch.zeros((1,71));labels=torch.ones((1,6))*.2
        low=torch.full((6,),-10.);high=-low;step=torch.tensor([.5]*5+[1.5])
        offset=torch.tensor([[.02,-.02,.01,-.01,.03,-.03]])
        changed,action=augment_setpoints(x,labels,offset,low,high,step)
        next_target=changed[:,:6]*180+changed[:,23:29]*5+action*step
        torch.testing.assert_close(next_target,labels*step)
        keep=list(range(23))+list(range(29,71))
        torch.testing.assert_close(changed[:,keep],x[:,keep])
    def test_closing_jaw_perturbation_stays_inside_joint_limit(self):
        x=torch.zeros((1,71));x[0,5]=2/180;x[0,28]=(-2.9999-2)/5
        low=torch.tensor([-10.]*5+[-2.9999]);high=torch.tensor([10.]*6);step=torch.tensor([.5]*5+[1.5])
        changed,action=augment_setpoints(x,torch.zeros((1,6)),torch.full((1,6),-.1),low,high,step)
        command=changed[:,:6]*180+changed[:,23:29]*5
        self.assertGreaterEqual(command[0,5].item(),low[5].item()-1e-6)
        self.assertAlmostEqual(action[0,5].item(),0,places=5)
    def test_free_arm_noise_preserves_grasp_world_position_and_next_target(self):
        spec=json.loads((ASSETS/'kinematics.json').read_text());anchors=json.loads((ASSETS/'finger_reference.json').read_text())['anchors_jaw_local_m']
        midpoint=OpeningMidpoint(spec,anchors);q=torch.tensor([[10.,60,-45,20,90,5]])
        x=torch.zeros((1,71));x[:,:6]=q/180;x[:,21:23]=torch.tensor([[.5,-.5]]);x[:,12:15]=torch.tensor([[.1,.2,.3]])
        aux={'free_arm_before':torch.tensor([True]),'home_feedback':torch.tensor([False]),
            'home_joint_deg':q.clone(),'all_centers_m':torch.arange(96,dtype=torch.float32).reshape(1,32,3)/1000,
            'all_upright':torch.ones((1,32)),'source_index':torch.tensor([8])}
        step=torch.tensor([.5]*5+[1.5]);low=torch.tensor(spec['limits'])[:,0]+1e-4;high=torch.tensor(spec['limits'])[:,1]-1e-4
        label=torch.full((1,6),.2);noise=torch.tensor([[.1,.2,-.3,.1,0,0]])
        changed,action=augment_free_arm(x,label,aux,noise,midpoint,low,high,step,0)
        before=midpoint(q,torch.tensor([-5.]))+x[:,12:15]*.03
        after=midpoint(changed[:,:6]*180,torch.tensor([-5.]))+changed[:,12:15]*.03
        torch.testing.assert_close(after,before,atol=1e-6,rtol=1e-6)
        command=changed[:,:6]*180+changed[:,23:29]*5
        torch.testing.assert_close(command+action*step,q+label*step,atol=1e-5,rtol=1e-5)


if __name__=='__main__': unittest.main()
