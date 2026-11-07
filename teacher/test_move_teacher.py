"""Shared completion objective, pre-opening checks and adaptive reset isolation."""
import copy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import numpy as np
import gymnasium as gym
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor

from teacher.grasp_checkpoints import GraspSelection,GraspCheckpoints
from teacher.move_teacher import MoveTask,MoveConfig
from teacher.success_sampler import SuccessSampler
from teacher.test_resets import observation
from teacher.test_grasp_checkpoints import ToyResetEnv,VALIDATION


class SamplerTests(unittest.TestCase):
    def test_prefers_mixed_outcomes_and_retains_coverage(self):
        s=SuccessSampler(['easy','frontier','hard'])
        for i in range(100):
            s.record('easy',True);s.record('hard',False);s.record('frontier',i%2==0)
        p=s.probabilities()
        self.assertGreater(p[1],p[0]);self.assertGreater(p[1],p[2]);self.assertGreaterEqual(p.min(),.2/3)
        self.assertAlmostEqual(float(p.sum()),1)
        with self.assertRaises(ValueError): s.record('heldout',True)
    def test_window_and_rng_reproducibility(self):
        s=SuccessSampler(['a','b'],window=4)
        for success in [False]*8+[True]*4: s.record('a',success)
        self.assertEqual(s.state()['history']['a'],[True]*4)
        first=np.random.default_rng(123);second=np.random.default_rng(123)
        self.assertEqual([s.sample(first) for _ in range(30)],[s.sample(second) for _ in range(30)])
    def test_complete_move_selection_keeps_partial_reset_scope(self):
        entries=[{'name':'home','family':'home'},{'name':'late','family':'near_release'}]
        rows=[{'reset_name':e['name'],'reset_family':e['family'],'score':{
            'success':e['name']=='late','complete_chess_move':e['name']=='late','objective':'unified_e2e4'}} for e in entries]
        with TemporaryDirectory() as tmp:
            p=Path(tmp,'actor.zip');p.write_bytes(b'new')
            selection=GraspSelection(tmp,entries,prefix='move',complete_moves=True)
            selection.consider(p,'late_only',1,rows)
            self.assertEqual(selection.best['per_family']['home']['successes'],0)
            self.assertEqual(selection.best['per_family']['near_release']['successes'],1)
            with self.assertRaises(ValueError): GraspSelection(tmp,entries).consider(p,'bad',1,rows)
    def test_real_sac_checkpoint_validation_preserves_adaptive_training_sequence(self):
        class AdaptiveToy(ToyResetEnv):
            def __init__(self):
                super().__init__(episode_steps=3);self.sampler=SuccessSampler([e['name'] for e in self.entries])
            def reset(self,*,seed=None,options=None):
                gym.Env.reset(self,seed=seed)
                self.entry=self.fixed_entry or self.entries[self.sampler.sample(self.np_random)]
                self.tick=0;return self.obs(),{}
            def step(self,action):
                result=super().step(action)
                if result[2] and self.fixed_entry is None: self.sampler.record(self.entry['name'],bool(action[0]>0))
                return result
        def train(root,evaluate):
            torch.set_num_threads(1);env=AdaptiveToy()
            model=SAC('MlpPolicy',Monitor(env),seed=27,device='cpu',learning_starts=100,
                buffer_size=32,batch_size=2,policy_kwargs={'net_arch':[16,16]},verbose=0)
            def validation(raw,actor,label,step):
                before=copy.deepcopy(raw.sampler.state());rows=[]
                for entry in VALIDATION:
                    raw.fixed_entry=entry;raw.reset(seed=999)
                    for _ in range(3): raw.step(np.zeros(2,np.float32))
                    rows.append({'reset_name':entry['name'],'reset_family':entry['family'],
                        'score':{'success':True,'complete_chess_move':True,'objective':'unified_e2e4'}})
                self.assertEqual(raw.sampler.state(),before);return rows
            callback=GraspCheckpoints(root,env,GraspSelection(root,VALIDATION,prefix='move',complete_moves=True),1,11,validation) if evaluate else None
            model.learn(11,callback=callback)
            n=model.replay_buffer.size()
            return env.sampler.state(),[getattr(model.replay_buffer,k)[:n].copy() for k in ('observations','actions','next_observations','dones')]
        with TemporaryDirectory() as first,TemporaryDirectory() as second:
            expected,expected_data=train(first,False);actual,data=train(second,True)
            self.assertEqual(actual,expected)
            self.assertEqual(sum(map(len,actual['history'].values())),3)
            for observed,wanted in zip(data,expected_data): np.testing.assert_array_equal(observed,wanted)


class MoveTests(unittest.TestCase):
    def setUp(self):
        self.baseline=observation();self.actual=copy.deepcopy(self.baseline)
        self.i=self.baseline['task_geometry']['source_index'];self.command=np.array([0.,0,0,0,0,-3])
        self.actual['measured_finger_joint_deg']={'jaw1_hinge':2.,'jaw2_hinge':-2.}
    def point(self): return np.array(self.actual['pieces']['piece_grasp_targets_m'][self.i])
    def move(self,center):
        self.actual['pieces']['piece_centers_m'][self.i]=np.array(center).tolist()
        self.actual['pieces']['piece_grasp_targets_m'][self.i]=(np.array(center)+[0,0,.005]).tolist()
    def task(self,history=None):
        return MoveTask(self.baseline,self.actual,self.point(),self.command,history or {},MoveConfig(stable_steps=3))
    def test_home_start_cannot_succeed_by_staying_home(self):
        task=self.task()
        for _ in range(8):
            _,done,_,score=task.update(self.actual,self.point()+[0,0,.1],self.command)
            self.assertFalse(done);self.assertFalse(score['success'])
    def test_lift_alone_is_not_completion(self):
        task=self.task();self.move(task.original+[0,0,.015])
        for _ in range(5): result=task.update(self.actual,self.point(),self.command)
        self.assertTrue(result[3]['lifted']);self.assertFalse(result[1]);self.assertFalse(result[3]['success'])
    def test_place_release_requires_final_home(self):
        self.move(np.array(self.baseline['pieces']['piece_centers_m'][self.i])+[0,0,.015])
        task=self.task({'prior_lift_verified':True,'prior_max_lift_m':.015,'held_start':True})
        self.move(task.target);self.actual['measured_joint_deg'][1]=80
        task.update(self.actual,self.point(),self.command)
        self.command[5]=6;self.actual['measured_finger_joint_deg']={'jaw1_hinge':5.,'jaw2_hinge':-5.}
        task.update(self.actual,self.point(),self.command)
        for _ in range(4): result=task.update(self.actual,self.point()+[0,0,.1],self.command)
        self.assertFalse(result[1]);self.assertTrue(result[3]['released_on_target'])
        self.actual['measured_joint_deg'][1]=0
        for _ in range(3): result=task.update(self.actual,self.point()+[0,0,.1],self.command)
        self.assertTrue(result[3]['success']);self.assertTrue(result[3]['complete_chess_move'])
    def test_unverified_held_and_released_resets_rejected(self):
        with self.assertRaises(ValueError): self.task({'held_start':True})
        with self.assertRaises(ValueError): self.task({'prior_release_verified':True})
    def test_near_complete_reset_still_requires_fresh_stability_window(self):
        self.move(self.baseline['task_geometry']['target_center_m'])
        history={'prior_lift_verified':True,'prior_max_lift_m':.015,'prior_release_verified':True,
            'release_intent':{'inside_target_square':True,'height_m':0,'center_m':self.actual['pieces']['piece_centers_m'][self.i]}}
        task=self.task(history)
        self.assertFalse(task.done);self.assertEqual(task.streak,0)
        for _ in range(2): self.assertFalse(task.update(self.actual,self.point()+[0,0,.1],self.command)[1])
        self.assertTrue(task.update(self.actual,self.point()+[0,0,.1],self.command)[3]['success'])
    def test_preopening_position_survives_magnetic_move(self):
        self.move(np.array(self.baseline['pieces']['piece_centers_m'][self.i])+[0,0,.015])
        task=self.task({'prior_lift_verified':True,'prior_max_lift_m':.015,'held_start':True})
        self.move(task.target);self.command[5]=6
        self.actual['measured_finger_joint_deg']={'jaw1_hinge':5.,'jaw2_hinge':-5.}
        result=task.update(self.actual,self.point(),self.command)
        self.assertEqual(result[3]['reason'],'release_outside_target_or_too_high')


if __name__=='__main__': unittest.main()
