"""Counterfactual supervision preserves scene geometry and desired tool goals."""
from dataclasses import asdict
import json
import unittest
import numpy as np
import torch
from teacher.cartesian_augmentation import augment_cartesian
from teacher.environment import ASSETS
from teacher.kinematics import OpeningMidpoint
from teacher.task_space import CartesianControl,CartesianConfig


class CartesianAugmentationTests(unittest.TestCase):
    def test_grasp_below_COM_preserves_positive_piece_uprightness(self):
        class FixedPose:
            def forward_pose(self,q,jaw2):
                return torch.zeros((len(q),3)),torch.eye(3).repeat(len(q),1,1)
        x=torch.zeros((1,76));x[:,17]=.5;x[:,41]=1;x[:,70]=-1
        aux={'free_arm_before':torch.tensor([False]),'released_before':torch.tensor([False]),
            'all_centers_m':torch.arange(96,dtype=torch.float32).reshape(1,32,3)/1000,
            'all_upright':torch.ones((1,32)),'source_index':torch.tensor([0]),'home_joint_deg':torch.zeros((1,6))}
        changed,_=augment_cartesian(x,torch.zeros((1,6)),aux,torch.zeros((1,6)),FixedPose(),
            torch.full((6,),-180.),torch.full((6,),180.),asdict(CartesianConfig()))
        self.assertEqual(float(changed[0,41]),1.)
        torch.testing.assert_close(changed[0,68:71],torch.tensor([0.,0.,-1.]))

    def test_free_and_rigid_held_cases_keep_correct_geometry_and_next_pose(self):
        spec=json.loads((ASSETS/'kinematics.json').read_text());anchors=json.loads((ASSETS/'finger_reference.json').read_text())['anchors_jaw_local_m']
        midpoint=OpeningMidpoint(spec,anchors);q=torch.tensor([[-10.,90,-40,10,85,2.1]]).repeat(2,1)
        point,rotation=midpoint.forward_pose(q,torch.full((2,),-2.1))
        x=torch.zeros((2,76));x[:,:6]=q/180;x[:,21:23]=torch.tensor([.21,-.21]);x[:,68:71]=torch.tensor([0.,0,1.]);x[:,41]=1
        x[:,12:15]=torch.tensor([.1,0,0]);x[1,17]=.5
        centers=torch.arange(96,dtype=torch.float32).reshape(1,32,3).repeat(2,1,1)/1000
        centers[:,8]=point+torch.tensor([.003,0,-.005])
        aux={'free_arm_before':torch.tensor([True,False]),'released_before':torch.tensor([False,False]),
            'all_centers_m':centers,'all_upright':torch.ones((2,32)),'source_index':torch.tensor([8,8]),'home_joint_deg':q.clone()}
        limits=torch.tensor(spec['limits']);low,high=limits[:,0]+1e-4,limits[:,1]-1e-4
        noise=torch.tensor([[.005,.005,-.005,.005,.005,0.]]).repeat(2,1);labels=torch.full((2,6),.05);labels[:,5]=-1
        cfg=CartesianConfig();changed,action=augment_cartesian(x,labels,aux,noise,midpoint,low,high,asdict(cfg))
        pnew,rnew=midpoint.forward_pose(changed[:,:6]*180,torch.full((2,),-2.1))
        old_grasp=point+x[:,12:15]*.03;new_grasp=pnew+changed[:,12:15]*.03
        torch.testing.assert_close(new_grasp[0],old_grasp[0],atol=1e-6,rtol=1e-6)
        old_local=(old_grasp[1]-point[1])@rotation[1].T
        new_local=(new_grasp[1]-pnew[1])@rnew[1].T
        torch.testing.assert_close(new_local,old_local,atol=1e-6,rtol=1e-6)
        control=CartesianControl(spec,anchors,low.numpy(),high.numpy(),[15.]*5+[45.],cfg)
        for i in range(2):
            fingers={'jaw1_hinge':2.1,'jaw2_hinge':-2.1}
            before=control.coordinates(q[i].numpy(),fingers)+labels[i,:5].numpy()
            command=(changed[i,:6]*180+changed[i,23:29]*5).numpy()
            after=control.coordinates(command,fingers)+action[i,:5].numpy()
            np.testing.assert_allclose(after,before,atol=2e-4)
        torch.testing.assert_close(action[:,5],labels[:,5])


if __name__=='__main__': unittest.main()
