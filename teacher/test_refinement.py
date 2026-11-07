"""Reference replay isolation, timeout semantics and fitted-actor warmup."""
from dataclasses import asdict
import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
import numpy as np
import torch
from stable_baselines3.common.logger import configure

from teacher.bootstrap_move import OfflineSpace
from teacher.environment import TeacherConfig
from teacher.grasp_checkpoints import file_hash
from teacher.move_teacher import MoveConfig
from teacher.refinement import RefinementSAC


class RefinementTests(unittest.TestCase):
    def setup_data(self,root,source_split='train',gamma=.997):
        config=TeacherConfig(max_steps=800,gamma=.997,jitter_mm=0,action_mode='delta',
            max_velocity_deg_s=(15.,15.,15.,15.,15.,45.),acceleration_weight=.0005,acceleration_cost_cap=.003)
        env=SimpleNamespace(base=SimpleNamespace(config=config),move_config=MoveConfig(),schema={'test':71},entries=[{'seed':1}])
        source=root/'manifest.json';source.write_text('{}')
        metadata={'training_only':True,'sources':[{'split':source_split}],'manifest':str(source),'manifest_sha256':file_hash(source),
            'action_config':{**json.loads(json.dumps(asdict(config))),'gamma':gamma},'move_config':asdict(env.move_config),'observation_schema':env.schema}
        path=root/'data.npz'
        np.savez(path,observation=np.zeros((4,71),np.float32),next_observation=np.zeros((4,71),np.float32),
            action=np.zeros((4,6),np.float32),reward=np.ones(4),terminated=np.array([False]*4),
            truncated=np.array([False,False,False,True]),seed=np.ones(4,dtype=np.int64),
            home_feedback=np.array([False,False,True,True]),metadata_json=np.array(json.dumps(metadata)))
        return path,env
    def model(self):
        torch.set_num_threads(1)
        return RefinementSAC('MlpPolicy',OfflineSpace(71),seed=17,device='cpu',learning_rate=.0003,
            actor_lr_scale=1/30,critic_warmup_updates=4,buffer_size=32,learning_starts=0,batch_size=2,
            ent_coef=.002,policy_kwargs={'net_arch':[16,16]},verbose=0)
    def test_reference_replay_and_bc_anchor_obey_critic_warmup(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);path,env=self.setup_data(root);model=self.model()
            info=model.configure_reference(path,env,12)
            self.assertEqual(model.replay_buffer.size(),4);self.assertEqual(info['prior_examples'],2)
            self.assertEqual(model.replay_buffer.timeouts[3,0],1)
            model.set_logger(configure(str(root/'optimizer'),[]));model._current_progress_remaining=1.
            before=copy.deepcopy(model.actor.state_dict())
            model.train(2,2);model.train(2,2)
            self.assertTrue(all(torch.equal(before[k],v) for k,v in model.actor.state_dict().items()))
            model.train(1,2)
            self.assertTrue(any(not torch.equal(before[k],v) for k,v in model.actor.state_dict().items()))
            # Reference supervision shares SAC's update, rather than causing a
            # second momentum-coupled Adam step for each gradient update.
            counts=[int(state['step']) for state in model.actor.optimizer.state.values() if 'step' in state]
            self.assertTrue(counts);self.assertEqual(set(counts),{5})
    def test_warmup_collects_from_fitted_actor_instead_of_random_targets(self):
        model=self.model()
        with torch.no_grad():
            for p in model.actor.parameters(): p.zero_()
            model.actor.mu.bias.fill_(1.);model.actor.log_std.bias.fill_(-20.)
        model._last_obs=np.zeros((1,71),np.float32)
        action,_=model._sample_action(512)
        np.testing.assert_allclose(action,np.tanh(1.),atol=1e-6)
    def test_validation_and_wrong_reward_contract_rejected_before_mutation(self):
        for split,gamma in [('validation',.997),('train',.99)]:
            with TemporaryDirectory() as tmp:
                path,env=self.setup_data(Path(tmp),split,gamma);model=self.model()
                with self.assertRaises(ValueError): model.configure_reference(path,env,12)
                self.assertEqual(model.replay_buffer.size(),0)
    def test_consistent_flow_prior_preserves_home_as_well_as_grasp(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);path,env=self.setup_data(root)
            with np.load(path,allow_pickle=False) as source:
                arrays={key:source[key].copy() for key in source.files}
            metadata=json.loads(str(arrays['metadata_json']));metadata['reference_label']='analytic_geometry_oracle'
            arrays['metadata_json']=np.array(json.dumps(metadata));np.savez(path,**arrays)
            model=self.model();info=model.configure_reference(path,env,12)
            self.assertEqual(info['prior_examples'],4);self.assertEqual(model.replay_buffer.size(),4)


if __name__=='__main__': unittest.main()
