import copy
import unittest

import numpy as np

from capture_alignment import capture_alignment, closing_alignment_penalty
from residual_learning import filter_residual
from verify_residual_pilot import check_row


class CaptureAlignmentTests(unittest.TestCase):
    def setUp(self):
        self.observation={'task_geometry':{'capture_index':0},
                          'pieces':{'piece_grasp_targets_m':[[0,0,.017]],'piece_upright_cosines':[1.]},
                          'gripper_grasp_center_m':[0,0,.017],
                          'target_legacy_action':[0,0,0,0,0,5.],
                          'joint_map':{'scale':[-1,-1,1,1,-1,1.16279296875]}}
        self.info={'telemetry':{'observation':self.observation},
                   'score':{'phase':'remove_captured','capture':{'lifted':False}}}

    def test_requested_height_has_maximum_quality_and_high_miss_does_not(self):
        self.assertEqual(capture_alignment(self.info),(1.,0.))
        self.observation['gripper_grasp_center_m'][2]+=.015
        quality,distance=capture_alignment(self.info)
        self.assertLess(quality,.012)
        self.assertAlmostEqual(distance,.015)

    def test_misaligned_commanded_closure_costs_but_centred_closure_does_not(self):
        before=copy.deepcopy(self.observation)
        self.observation['target_legacy_action'][5]-=1.
        self.assertEqual(closing_alignment_penalty(before,self.info,1)[0],0.)
        self.observation['gripper_grasp_center_m'][1]-=.012
        penalty,closing,quality=closing_alignment_penalty(before,self.info,1)
        self.assertLess(penalty,0.)
        self.assertAlmostEqual(closing,1.16279296875)
        self.assertAlmostEqual(penalty,-2*closing/12.5*(1-quality))

    def test_opening_holding_far_away_or_after_lift_is_not_penalised(self):
        self.observation['gripper_grasp_center_m'][2]+=.015
        before=copy.deepcopy(self.observation)
        for delta in [0.,1.]:
            self.observation['target_legacy_action'][5]=before['target_legacy_action'][5]+delta
            self.assertEqual(closing_alignment_penalty(before,self.info,1)[0],0.)
        self.observation['target_legacy_action'][5]=before['target_legacy_action'][5]-1.
        self.assertEqual(closing_alignment_penalty(before,self.info,0)[0],0.)
        self.info['score']['capture']['lifted']=True
        self.assertEqual(closing_alignment_penalty(before,self.info,1)[0],0.)
        self.info['score']['capture']['lifted']=False
        self.observation['gripper_grasp_center_m'][2]+=.1
        self.assertEqual(closing_alignment_penalty(before,self.info,1)[0],0.)

    def test_tipping_or_switching_to_attacker_cannot_gain_alignment_potential(self):
        self.observation['pieces']['piece_upright_cosines'][0]=.5
        self.assertEqual(capture_alignment(self.info)[0],0.)
        self.info['score']['phase']='move_attacker'
        self.assertEqual(capture_alignment(self.info)[0],0.)


class ResidualFilterTests(unittest.TestCase):
    def test_wider_corrections_reach_bounds_smoothly_and_can_reverse(self):
        bounds=np.array([6,6,6,6,6,3]);rates=np.array([.4,.4,.4,.4,.4,.6])
        previous=np.zeros(6)
        for target in [np.ones(6),-np.ones(6),np.zeros(6)]:
            for _ in range(80):
                applied=filter_residual(target,1,previous,bounds,.25,rates)
                self.assertTrue(np.all(np.abs(applied-previous)*bounds<=rates+1e-6))
                self.assertTrue(np.all(np.abs(applied)<=1.))
                previous=applied
            np.testing.assert_allclose(previous,target,atol=1e-5)

    def test_zero_policy_remains_exact_act_and_gate_fades_to_zero(self):
        np.testing.assert_equal(filter_residual(np.zeros(6),1,np.zeros(6),[6]*6,.25,[.4]*6),np.zeros(6))
        previous=np.ones(6)
        for _ in range(60):previous=filter_residual(np.ones(6),0,previous,[6]*6,.25,[.4]*6)
        np.testing.assert_allclose(previous,np.zeros(6),atol=1e-5)

    def test_saved_run_audit_accepts_filter_memory_and_checks_closing_cost(self):
        previous=np.full(6,.8);bounds=np.array([6,6,6,6,6,3])
        config={'alpha':.25,'max_change_deg':[.4,.4,.4,.4,.4,.6]}
        applied=filter_residual(np.zeros(6),1,previous,bounds,.25,config['max_change_deg'])
        penalty=.01*float(np.mean(applied**2))
        row={'action':[0.]*6,'state':[0.]*6,'reward':-penalty-.2,
             'residual':{'penalty':penalty},'score':{'reward_terms':{'time':0.}},
             'learning':{'applied_residual':applied.tolist(),'correction_gate':1.,
                         'task_reward':0.,'shaping_reward':0.,'closing_alignment_penalty':-.2}}
        check_row(row,previous_applied=previous,residual_bounds=bounds,filter_config=config)
        row['reward']+=.2
        with self.assertRaises(AssertionError):
            check_row(row,previous_applied=previous,residual_bounds=bounds,filter_config=config)


if __name__=='__main__':
    unittest.main()
