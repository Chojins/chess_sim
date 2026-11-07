"""Real SAC tests for validation isolation and retained early checkpoints."""
import copy
from pathlib import Path
import random
from tempfile import TemporaryDirectory
import unittest

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor

from teacher.grasp_checkpoints import ConservativeSAC,GraspCheckpoints,GraspSelection


class ToyResetEnv(gym.Env):
    def __init__(self,episode_steps=2):
        self.entries=[{'name':'train_open','family':'open_approach','id':0},
                      {'name':'train_closed','family':'closed_contact','id':1}]
        self.fixed_entry=None;self.entry=None;self.tick=0;self.episode_steps=episode_steps
        self.action_space=gym.spaces.Box(-1,1,(2,),np.float32)
        self.observation_space=gym.spaces.Box(-np.inf,np.inf,(3,),np.float32)

    def reset(self,*,seed=None,options=None):
        super().reset(seed=seed)
        self.entry=self.fixed_entry or self.entries[int(self.np_random.integers(2))]
        self.tick=0
        return self.obs(),{}

    def obs(self):
        return np.array([self.entry['id'],self.tick,0],np.float32)

    def step(self,action):
        self.tick+=1
        return self.obs(),float(np.sum(action)),self.tick>=self.episode_steps,False,{}


VALIDATION=[{'name':'val_open','family':'open_approach','id':1000},
            {'name':'val_closed','family':'closed_contact','id':1001}]


def results(entries,success=True):
    return [{'reset_name':entry['name'],'reset_family':entry['family'],
             'score':{'success':success,'complete_chess_move':False}} for entry in entries]


class GraspCheckpointTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_critic_warmup_freezes_actor_then_uses_smaller_learning_rate(self):
        model=ConservativeSAC('MlpPolicy',ToyResetEnv(),device='cpu',seed=8,learning_rate=.003,
            actor_lr_scale=.1,critic_warmup_updates=4,learning_starts=0,batch_size=2,buffer_size=32,
            ent_coef=.002,policy_kwargs={'net_arch':[16,16]},verbose=0)
        actor=copy.deepcopy(model.actor.state_dict());critic=copy.deepcopy(model.critic.state_dict())
        model.learn(4)
        self.assertTrue(all(torch.equal(actor[k],v) for k,v in model.actor.state_dict().items()))
        self.assertTrue(any(not torch.equal(critic[k],v) for k,v in model.critic.state_dict().items()))
        model.learn(2,reset_num_timesteps=False)
        self.assertTrue(any(not torch.equal(actor[k],v) for k,v in model.actor.state_dict().items()))
        self.assertAlmostEqual(model.actor.optimizer.param_groups[0]['lr'],.0003)
        self.assertAlmostEqual(model.critic.optimizer.param_groups[0]['lr'],.003)
        self.assertIsNone(model.ent_coef_optimizer)
        self.assertAlmostEqual(float(model.ent_coef_tensor),.002)

    def test_late_failure_and_ties_preserve_the_earlier_selected_file(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);early=root/'early.zip';late=root/'late.zip'
            early.write_bytes(b'working-early-policy');late.write_bytes(b'regressed-late-policy')
            selector=GraspSelection(root,VALIDATION)
            self.assertTrue(selector.consider(early,'early',1000,results(VALIDATION)))
            self.assertFalse(selector.consider(late,'regressed',2000,results(VALIDATION,False)))
            self.assertFalse(selector.consider(late,'tie',3000,results(VALIDATION)))
            self.assertEqual((root/'grasp_best.zip').read_bytes(),early.read_bytes())
            self.assertEqual(selector.best['label'],'early')
            with self.assertRaises(ValueError): selector.consider(late,'duplicate',4000,[results(VALIDATION)[0]]*2)

    def test_balanced_families_rank_above_one_family_only(self):
        entries=[{'name':str(i),'family':'open_approach' if i<2 else 'closed_contact'} for i in range(4)]
        with TemporaryDirectory() as tmp:
            checkpoint=Path(tmp,'actor.zip');checkpoint.write_bytes(b'actor')
            selector=GraspSelection(tmp,entries)
            one=results(entries);one[2]['score']['success']=False;one[3]['score']['success']=False
            selector.consider(checkpoint,'one-family',100,one)
            balanced=results(entries);balanced[1]['score']['success']=False;balanced[3]['score']['success']=False
            self.assertTrue(selector.consider(checkpoint,'balanced',200,balanced))

    def test_placement_names_do_not_overwrite_the_qualified_grasp(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'new.zip';source.write_bytes(b'placement')
            (root/'grasp_best.zip').write_bytes(b'frozen-grasp')
            selector=GraspSelection(root,VALIDATION,prefix='place')
            selector.consider(source,'initial',0,results(VALIDATION))
            self.assertEqual((root/'grasp_best.zip').read_bytes(),b'frozen-grasp')
            self.assertEqual((root/'place_best.zip').read_bytes(),b'placement')
    def test_optional_complete_move_efficiency_never_beats_reliability(self):
        entries=[{'name':'val_home','family':'home'}]
        def row(steps,success=True):
            return [{'reset_name':'val_home','reset_family':'home','steps':steps,
                'score':{'success':success,'complete_chess_move':success,'objective':'unified_e2e4'}}]
        with TemporaryDirectory() as tmp:
            root=Path(tmp);checkpoint=root/'actor.zip';checkpoint.write_bytes(b'actor')
            selector=GraspSelection(root,entries,prefix='move',complete_moves=True,prefer_faster=True)
            self.assertTrue(selector.consider(checkpoint,'initial',0,row(100)))
            self.assertFalse(selector.consider(checkpoint,'tiny_timing_change',1,row(95)))
            self.assertTrue(selector.consider(checkpoint,'faster_success',2,row(80)))
            self.assertFalse(selector.consider(checkpoint,'fast_failure',3,row(10,False)))
            self.assertEqual(selector.best['label'],'faster_success')

    def test_episode_boundary_evaluation_preserves_replay_rng_and_unused_training_reset(self):
        def train(callbacks,root):
            env=ToyResetEnv(episode_steps=3)
            model=SAC('MlpPolicy',Monitor(env),device='cpu',seed=27,learning_starts=100,
                buffer_size=32,batch_size=2,policy_kwargs={'net_arch':[16,16]},verbose=0)
            steps=[]
            def evaluate(raw,policy,label,step):
                steps.append(step)
                # Deliberately consume all RNGs and produce unmistakably
                # non-training observations. None may affect the learner.
                random.random();np.random.rand(9);torch.rand(9)
                for entry in VALIDATION:
                    raw.fixed_entry=entry;raw.reset(seed=999)
                    for _ in range(3): raw.step(np.zeros(2,np.float32))
                return results(VALIDATION)
            cb=GraspCheckpoints(root,env,GraspSelection(root,VALIDATION),1,8,evaluate) if callbacks else None
            model.learn(8,callback=cb)
            n=model.replay_buffer.size()
            return model,[getattr(model.replay_buffer,key)[:n].copy() for key in ('observations','next_observations','actions','dones')],steps
        with TemporaryDirectory() as first,TemporaryDirectory() as second:
            control,expected,_=train(False,first)
            actual,data,steps=train(True,second)
            self.assertEqual(steps,[3,6])
            self.assertEqual(actual.replay_buffer.size(),8)
            for observed,wanted in zip(data,expected): np.testing.assert_array_equal(observed,wanted)
            self.assertLess(float(data[0][:,:,0].max()),1000)
            loaded=SAC.load(Path(second,'grasp_best.zip'),device='cpu')
            action,_=loaded.predict(np.zeros(3,np.float32),deterministic=True)
            self.assertTrue(np.isfinite(action).all())


if __name__=='__main__':
    unittest.main()
