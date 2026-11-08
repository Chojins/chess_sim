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
    def test_arm_only_updates_preserve_jaw_shared_features_and_saved_inference(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);path,env=self.setup_data(root);model=self.model()
            model.configure_reference(path,env,12);model.configure_arm_head_only()
            model.set_logger(configure(str(root/'optimizer'),[]));model._current_progress_remaining=1.
            before=copy.deepcopy(model.actor.state_dict());x=np.zeros((2,71),np.float32)
            initial,_=model.predict(x,deterministic=True)
            model.train(4,2);model.train(4,2)
            self.assertTrue(model.assert_arm_head_preserved()['passed'])
            self.assertFalse(torch.equal(before['mu.weight'][:5],model.actor.mu.weight[:5]))
            after,_=model.predict(x,deterministic=True);np.testing.assert_array_equal(initial[:,5],after[:,5])
            model.save(root/'arm_only');restored=RefinementSAC.load(root/'arm_only.zip',device='cpu')
            np.testing.assert_array_equal(after,restored.predict(x,deterministic=True)[0])
            self.assertFalse(hasattr(restored,'preserved_grasp_state'))

    def test_reference_with_old_camera_rewards_is_rejected_before_replay_mutation(self):
        with TemporaryDirectory() as tmp:
            path,env=self.setup_data(Path(tmp));env.camera_clearance_reward_contract={'buffer_m':.008}
            model=self.model()
            with self.assertRaisesRegex(ValueError,'camera-clearance reward'):model.configure_reference(path,env,12)
            self.assertEqual(model.replay_buffer.size(),0)

    def test_counterfactual_supervision_does_not_enter_physical_replay(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);path,env=self.setup_data(root);model=self.model();model.configure_reference(path,env,12)
            pool=root/'labels.npz';np.savez(pool,observation=np.zeros((8,71),np.float32),label=np.zeros((8,6),np.float32),
                counterfactual_observation=np.array([False]*4+[True]*4))
            manifest={'oracle_training_only':True,'validation_used_for_gradients':False,'physics_settings':env.base.actual['physics_settings'],
                'training_sources':[{'episodes':[{'split':'train'}]}],'supervised_pool_sha256':file_hash(pool)}
            result=model.configure_supervised_prior(pool,manifest)
            self.assertEqual(model.replay_buffer.size(),4);self.assertEqual(len(model.prior_observations),8)
            self.assertEqual(result['counterfactual_observations'],4);self.assertEqual(result['synthetic_replay_transitions'],0)
            model.actor.optimizer.zero_grad()
            loss=model._supervised_loss(model.prior_observations,model.prior_actions,True,model.prior_jaw_probability)
            loss.backward()
            self.assertTrue(all(p.grad is None or float(p.grad.abs().max())<1e-7 for p in model.actor.parameters()))

    def setup_data(self,root,source_split='train',gamma=.997):
        config=TeacherConfig(max_steps=800,gamma=.997,jitter_mm=0,action_mode='delta',
            max_velocity_deg_s=(15.,15.,15.,15.,15.,45.),acceleration_weight=.0005,acceleration_cost_cap=.003)
        physics={'contacts':{'robot_self_collisions':False}}
        env=SimpleNamespace(base=SimpleNamespace(config=config,actual={'physics_settings':physics}),move_config=MoveConfig(),schema={'test':71},entries=[{'seed':1}])
        source=root/'manifest.json';source.write_text('{}')
        (root/'initial.json').write_text(json.dumps({'telemetry':{'observation':{'physics_settings':physics}}}))
        metadata={'training_only':True,'sources':[{'split':source_split,'path':str(root)}],'manifest':str(source),'manifest_sha256':file_hash(source),
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
