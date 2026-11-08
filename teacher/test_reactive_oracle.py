import unittest
import numpy as np
import torch
from teacher.reactive_oracle import ReactiveOracle,FlowConfig
from teacher.task_space import CartesianConfig
from teacher.demonstration_filter import settled_dwell
from teacher.collect_corrections import TrainingMixture
from teacher.geometry_features import geometry_values


class OracleTests(unittest.TestCase):
    def test_orientation_profile_changes_height_transition_without_clock(self):
        with self.assertRaises(ValueError): FlowConfig(orientation_low_m=.15,orientation_high_m=.10)
        x=self.pose();x[12:15]=[.1,0,-.15/.03];x[74]=(24-65)/90
        early=ReactiveOracle(CartesianConfig(),FlowConfig(grip_pitch_deg=85,orientation_high_m=.23,orientation_low_m=.17))
        default=ReactiveOracle(CartesianConfig(),FlowConfig(grip_pitch_deg=85))
        self.assertGreater(early.predict(x)[0][3],default.predict(x)[0][3])
        changed=x.copy();changed[67]=.8
        np.testing.assert_array_equal(early.predict(changed)[0],early.predict(x)[0])

    def test_tight_approach_keeps_fingers_above_head_until_laterally_centered(self):
        oracle=ReactiveOracle(CartesianConfig(),FlowConfig(funnel_inner_m=.0005,funnel_outer_m=.0025,close_xy_m=.0008,close_z_m=.001))
        x=self.pose();x[12:15]=[.005/.03,0,0]
        action,_=oracle.predict(x);self.assertGreater(action[2],0);self.assertEqual(action[5],1)
        x[12:15]=[.0015/.03,0,0];action,_=oracle.predict(x)
        self.assertEqual(action[5],1)
        x[12:15]=0;action,_=oracle.predict(x);self.assertEqual(action[5],-1)
    def setUp(self):
        self.oracle=ReactiveOracle(CartesianConfig())
    def pose(self):
        x=np.zeros(76,np.float32);x[1:4]=[0,65/180,0];x[4]=90/180
        x[21:23]=[.6,-.6];x[5]=6/180;x[34]=1;x[70]=1;x[18]=.9
        return x
    def test_aligned_open_grip_closes_before_lift(self):
        x=self.pose();action,_=self.oracle.predict(x)
        np.testing.assert_allclose(action[:3],0,atol=1e-6);self.assertEqual(action[5],-1)
        # While the jaw is still closing, no premature lift is requested.
        x[21:23]=[.4,-.4];x[5]=4/180;x[28]=-1.4;x[34]=-1;x[11]=-10/60
        action,_=self.oracle.predict(x)
        np.testing.assert_allclose(action[:3],0,atol=1e-6);self.assertEqual(action[5],-1)
    def test_held_pawn_lifts_before_transport(self):
        x=self.pose();x[5]=2/180;x[21:23]=[.2,-.2];x[28]=-1;x[34]=-1
        action,_=self.oracle.predict(x)
        self.assertGreater(action[2],0);self.assertEqual(action[0],0);self.assertEqual(action[5],-1)
        x[17]=.032/.03
        action,_=self.oracle.predict(x);self.assertGreater(action[0],0)
    def test_contact_solver_velocity_does_not_block_obstructed_grasp_lift(self):
        x=self.pose();x[5]=2.4/180;x[21:23]=[.24,-.24];x[28]=(-3-2.4)/5;x[34]=-1;x[11]=8/60
        action,_=self.oracle.predict(x);self.assertGreater(action[2],0)
        # A still-wide aperture does not count as a completed grasp.
        x[5]=4/180;x[21:23]=[.4,-.4];x[28]=(-3-4)/5
        action,_=self.oracle.predict(x);np.testing.assert_allclose(action[:3],0,atol=1e-6)
    def test_open_placed_pawn_returns_home_without_grasp(self):
        x=self.pose();x[18:21]=0;x[71:74]=[.1,.2,.7];x[12:15]=[0,0,-.12/.03]
        action,_=self.oracle.predict(x)
        self.assertGreater(action[2],0);self.assertEqual(action[5],1)
    def test_invalid_shape_and_finite_batch(self):
        with self.assertRaises(ValueError): self.oracle.actions(np.zeros((2,71)))
        with self.assertRaises(ValueError): self.oracle.actions(np.full((1,76),np.nan))
        a=self.oracle.actions(np.array([self.pose(),self.pose()]))
        self.assertEqual(a.shape,(2,6));self.assertTrue(np.isfinite(a).all());self.assertLessEqual(np.abs(a).max(),1)
    def test_dwell_filter_keeps_jaw_transitions_and_final_home(self):
        x=np.zeros((4,76),np.float32);a=np.zeros((4,6),np.float32)
        x[:,34]=1;a[:,5]=1
        a[1,5]=-1;x[2,11]=.2;x[3,62]=0
        keep=~settled_dwell(x,a,np.array([False,False,False,True]))
        np.testing.assert_array_equal(keep,[False,True,True,True])
    def test_mixture_counts_interventions_and_frozen_neural_mode(self):
        class Constant:
            def __init__(self,value): self.value=value
            def predict(self,observation,deterministic=True): return np.full(6,self.value),None
        neural=Constant(.25);oracle=Constant(.75)
        mixture=TrainingMixture(neural,oracle,0,1)
        for _ in range(4): np.testing.assert_array_equal(mixture.predict(self.pose())[0],np.full(6,.25))
        self.assertEqual(mixture.interventions,0);self.assertEqual(mixture.steps,4)
        mixture=TrainingMixture(neural,oracle,1,1)
        np.testing.assert_array_equal(mixture.predict(self.pose())[0],np.full(6,.75))
        self.assertEqual(mixture.interventions,1)
    def test_geometry_encoder_ignores_clock_and_previous_arm_actions(self):
        x=torch.from_numpy(np.array([self.pose(),self.pose()]))
        x[1,67]=.8;x[1,29:34]=.9
        encoded=geometry_values(x)
        self.assertEqual(encoded.shape,(2,39));torch.testing.assert_close(encoded[0],encoded[1])
        self.assertAlmostEqual(float(encoded[0,21]),65/90,places=6)


if __name__=='__main__': unittest.main()
