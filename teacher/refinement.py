"""Conservative SAC refinement with training-only reference replay and BC prior."""
import json
from pathlib import Path
import numpy as np
import torch
from teacher.grasp_checkpoints import ConservativeSAC,file_hash


class RefinementSAC(ConservativeSAC):
    def _sample_action(self,learning_starts,action_noise=None,n_envs=1):
        # A fitted actor initializes fresh replay; random warmup would discard
        # its learned grasp and transport on the first several hundred steps.
        return super()._sample_action(0,action_noise,n_envs)
    def configure_reference(self,path,env,seed,weight=.5,decay_steps=8000.,floor_fraction=.1):
        if (not np.isfinite([weight,decay_steps,floor_fraction]).all() or weight<0 or decay_steps<=0
                or not 0<=floor_fraction<=1): raise ValueError('Invalid reference regularization')
        with np.load(path,allow_pickle=False) as archive:
            metadata=json.loads(str(archive['metadata_json']));arrays={k:archive[k].copy() for k in
                ('observation','next_observation','action','reward','terminated','truncated','seed','home_feedback')}
        if not metadata['training_only'] or any(s['split']!='train' for s in metadata['sources']):
            raise ValueError('Only training references may enter replay or prior')
        if file_hash(metadata['manifest'])!=metadata['manifest_sha256']: raise ValueError('Reference source manifest changed')
        from teacher.physics_contract import source_physics,physics_signature
        manifest=json.loads(Path(metadata['manifest']).read_text())
        self.reference_physics=source_physics(manifest,metadata['sources'])
        if self.reference_physics!=physics_signature(env.base.actual['physics_settings']):
            raise ValueError('Reference replay collision/dynamics contract differs from the live simulator')
        from teacher.robot_contacts import contact_costs
        if contact_costs(metadata.get('robot_contact_costs'))!=contact_costs(getattr(env,'robot_contact_costs',None)):
            raise ValueError('Reference replay robot-contact reward contract differs')
        if bool(metadata.get('terminate_on_robot_contact'))!=bool(getattr(env,'terminate_on_robot_contact',False)):
            raise ValueError('Reference replay robot-contact terminal contract differs')
        if metadata.get('camera_clearance_reward_contract')!=getattr(env,'camera_clearance_reward_contract',None):
            raise ValueError('Reference replay camera-clearance reward contract differs')
        if metadata['action_config']!=json.loads(json.dumps(env.base.config.__dict__)):
            raise ValueError('Reference rewards/actions differ from the refinement contract')
        if metadata['move_config']!=env.move_config.__dict__ or metadata['observation_schema']!=env.schema:
            raise ValueError('Reference task or observation contract changed')
        if metadata.get('policy_action_space')!=getattr(env,'policy_action_space',None):
            raise ValueError('Reference policy action units changed')
        if metadata.get('cartesian_config') is not None:
            from dataclasses import asdict
            if metadata['cartesian_config']!=asdict(env.cartesian_config): raise ValueError('Cartesian action scaling changed')
        if not set(arrays['seed']).issubset({r['seed'] for r in env.entries}):
            raise ValueError('A validation/test seed entered reference replay')
        if self.replay_buffer.size() or len(arrays['action'])>=self.replay_buffer.buffer_size:
            raise ValueError('Need empty replay with room for fresh online experience')
        for i in range(len(arrays['action'])):
            terminated=bool(arrays['terminated'][i]);truncated=bool(arrays['truncated'][i])
            self.replay_buffer.add(arrays['observation'][i:i+1],arrays['next_observation'][i:i+1],arrays['action'][i:i+1],
                np.array([arrays['reward'][i]]),np.array([terminated or truncated]),
                [{'TimeLimit.truncated':bool(truncated and not terminated)}])
        consistent_flow=metadata.get('reference_label')=='analytic_geometry_oracle'
        prior_mask=np.ones(len(arrays['action']),dtype=bool) if consistent_flow else ~arrays['home_feedback'].astype(bool)
        if metadata.get('policy_action_space'):
            from teacher.demonstration_filter import settled_dwell
            # Replay retains every real transition. Only supervised prior labels
            # omit settled waits that require an unobserved trajectory clock.
            prior_mask&=~settled_dwell(arrays['observation'],arrays['action'],arrays['home_feedback'])
        if not prior_mask.any(): raise ValueError('No grasp/carry prior examples')
        self.prior_observations=torch.as_tensor(arrays['observation'][prior_mask],device=self.device,dtype=torch.float32)
        self.prior_actions=torch.as_tensor(arrays['action'][prior_mask],device=self.device,dtype=torch.float32)
        self.prior_rng=np.random.default_rng(seed);self.prior_weight=weight
        self.prior_decay_steps=decay_steps;self.prior_floor_fraction=floor_fraction
        self.prior_cartesian=bool(metadata.get('policy_action_space'))
        return {'path':str(Path(path).resolve()),'sha256':file_hash(path),'transitions':len(arrays['action']),
            'seeds':sorted({int(s) for s in arrays['seed']}),'initial_prior_weight':weight,
            'prior_examples':int(prior_mask.sum()),'prior_scope':('all phases of the consistent geometric reference' if consistent_flow
                else 'grasp/carry and pre-clearance; no old homing-speed anchor'),
            'prior_decay_steps':decay_steps,'minimum_prior_weight_fraction':floor_fraction,
            'optimizer_update':'reference loss added to the same actor update as SAC',
            'reward_relabeling':metadata.get('reward_relabeling'),
            'collision_reward_relabeling':metadata.get('collision_reward_relabeling'),
            'camera_reward_relabeling':metadata.get('camera_reward_relabeling')}

    def configure_arm_head_only(self):
        if hasattr(self,'preserved_grasp_state'):raise ValueError('Arm-only refinement already configured')
        self.preserved_grasp_state={k:v.detach().clone() for k,v in self.actor.state_dict().items()}
        for parameter in self.actor.parameters():parameter.requires_grad_(False)
        for parameter in (self.actor.mu.weight,self.actor.mu.bias):
            parameter.requires_grad_(True)
            parameter.register_hook(lambda gradient:torch.cat([gradient[:5],torch.zeros_like(gradient[5:])]))
        return {'trainable':'five arm mean-output rows only','preserved':'shared actor features/trunk, jaw mean and exploration parameters',
            'runtime':'ordinary saved neural actor; no corrective action override'}

    def assert_arm_head_preserved(self):
        for key,value in self.actor.state_dict().items():
            expected=self.preserved_grasp_state[key]
            if key in ('mu.weight','mu.bias'):value,expected=value[5:],expected[5:]
            torch.testing.assert_close(value,expected,rtol=0,atol=0)
        return {'passed':True,'shared_features_trunk_jaw_and_exploration_exact':True}

    def configure_placement_corrections(self,path,env,weight):
        if not np.isfinite(weight) or weight<=0:raise ValueError('Invalid placement corrective weight')
        from teacher.fit_oracle import load_training
        from teacher.placement_corrections import placement_mask
        with np.load(path,allow_pickle=False) as archive:
            metadata=json.loads(str(archive['metadata_json']))
            values=archive['observation'].copy();labels=archive['label'].copy();goals=archive['goal_id'].copy()
        if not metadata['training_only'] or metadata['validation_used_for_gradients'] is not False:
            raise ValueError('Placement corrections must be training-only')
        paths=[Path(s['path']) for s in metadata['training_sources']]
        for source in metadata['training_sources']:
            if file_hash(source['path'])!=source['sha256']:raise ValueError('Corrective source archive changed')
            if any(e['split']!='train' or e['seed'] not in {r['seed'] for r in env.entries} for e in source['episodes']):
                raise ValueError('Held-out reset entered placement corrections')
        _,_,source=load_training(paths)
        if (source['physics_settings']!=self.reference_physics or source['observation_schema']!=env.schema
                or source['joint_map']!=env.base.mapping):raise ValueError('Placement corrective physics/encoder contract differs')
        if (values.shape!=(len(labels),*self.observation_space.shape) or labels.shape!=(len(values),6)
                or goals.shape!=(len(values),) or not len(values) or not np.isfinite(values).all()
                or not np.isfinite(labels).all() or np.max(np.abs(labels))>1 or not placement_mask(values).all()):
            raise ValueError('Invalid held-piece placement correction pool')
        self.placement_observations=torch.as_tensor(values,device=self.device,dtype=torch.float32)
        self.placement_labels=torch.as_tensor(labels,device=self.device,dtype=torch.float32)
        self.placement_weight=weight
        self.placement_goal_indices=[np.flatnonzero(goals==goal) for goal in np.unique(goals)]
        return {'path':str(Path(path).resolve()),'sha256':file_hash(path),'observations':len(values),'weight':weight,
            'sampling':'uniform goals, then uniform held-piece placement states','scope':'Arm-only supervision; no jaw correction, synthetic rewards or physical replay insertion',
            'training_sources':metadata['training_sources']}

    def configure_supervised_prior(self,path,warm_manifest):
        if not hasattr(self,'prior_observations'):raise ValueError('Configure physical replay before a supervised-only prior')
        if (not warm_manifest.get('oracle_training_only') or warm_manifest.get('validation_used_for_gradients') is not False
                or warm_manifest.get('physics_settings')!=self.reference_physics):
            raise ValueError('Supervised prior provenance differs from training physics or roles')
        if any(e['split']!='train' for s in warm_manifest['training_sources'] for e in s['episodes']):
            raise ValueError('A held-out episode entered the supervised prior')
        if file_hash(path)!=warm_manifest['supervised_pool_sha256']:raise ValueError('Supervised label pool changed')
        with np.load(path,allow_pickle=False) as source:
            observations=source['observation'].copy();labels=source['label'].copy();counterfactual=source['counterfactual_observation'].copy()
        if (observations.shape!=(len(labels),*self.observation_space.shape) or labels.shape!=(len(observations),6)
                or counterfactual.shape!=(len(observations),) or not np.isfinite(observations).all()
                or not np.isfinite(labels).all() or np.max(np.abs(labels))>1):
            raise ValueError('Invalid supervised-only label pool')
        replay_size=self.replay_buffer.size()
        self.prior_observations=torch.as_tensor(observations,device=self.device,dtype=torch.float32)
        # Anchor the verified warm policy itself. Re-fitting an imperfect expert
        # approximation during critic warmup can shift a working grasp boundary.
        with torch.no_grad():
            logits=self.actor.get_action_dist_params(self.prior_observations)[0]
            self.prior_actions=torch.tanh(logits).detach()
            self.prior_jaw_probability=torch.sigmoid(logits[:,5]).detach()
        return {'path':str(Path(path).resolve()),'sha256':file_hash(path),'supervised_observations':len(labels),
            'counterfactual_observations':int(counterfactual.sum()),'physical_replay_transitions':replay_size,
            'synthetic_replay_transitions':0,'target':'Frozen warm neural mean and jaw probabilities on this observation pool',
            'scope':'Actor preservation only; no synthetic rewards, dynamics or replay transitions'}
    def configure_online_oracle(self,env,weight,config=None,action_field=None):
        if not np.isfinite(weight) or weight<0: raise ValueError('Invalid corrective supervision weight')
        if not weight: return None
        from teacher.reactive_oracle import ReactiveOracle,FlowConfig
        if getattr(env,'policy_action_space',None)!='opening_xyz_pitch_roll_binary_jaw_v1':
            raise ValueError('Corrective oracle requires the Cartesian geometry contract')
        if action_field=='configuration_reach_and_return_with_task_space_transport':
            from teacher.configuration_oracle import ConfigurationReference
            self.online_oracle=ConfigurationReference(env.controller,FlowConfig(**(config or {})))
        else:self.online_oracle=ReactiveOracle(env.cartesian_config,FlowConfig(**(config or {})))
        self.online_oracle_weight=weight
        return {'weight':weight,'version':self.online_oracle.version,
            'scope':'training-only corrective labels on real replay observations; no runtime intervention',
            'replay_labels_changed':False}
    def _supervised_loss(self,observations,actions,cartesian,jaw_probability=None):
        logits=self.actor.get_action_dist_params(observations)[0];predicted=torch.tanh(logits)
        if cartesian:
            return ((predicted[:,:5]-actions[:,:5]).square().mean()
                +torch.nn.functional.binary_cross_entropy_with_logits(logits[:,5],(actions[:,5]>0).float() if jaw_probability is None else jaw_probability))
        return (predicted-actions).square().mean()
    def train(self,gradient_steps,batch_size=64):
        if not hasattr(self,'prior_observations'):
            return super().train(gradient_steps,batch_size)
        # Combine SAC and supervision BEFORE its optimizer step. Separate Adam
        # steps can retain the preceding critic gradient in the BC step's
        # momentum, defeating the intended preservation of a working actor.
        def regularize(optimizer,args,kwargs):
            weight=self.prior_weight*max(self.prior_floor_fraction,float(np.exp(-self.num_timesteps/self.prior_decay_steps)))
            indices=self.prior_rng.integers(len(self.prior_observations),size=batch_size)
            probability=self.prior_jaw_probability[indices] if hasattr(self,'prior_jaw_probability') else None
            loss=self._supervised_loss(self.prior_observations[indices],self.prior_actions[indices],self.prior_cartesian,probability)
            total=weight*loss
            if hasattr(self,'placement_observations'):
                chosen=self.placement_goal_indices[int(self.prior_rng.integers(len(self.placement_goal_indices)))]
                selected=self.prior_rng.choice(chosen,size=batch_size)
                prediction=self.actor(self.placement_observations[selected],deterministic=True)
                corrective=(prediction[:,:5]-self.placement_labels[selected,:5]).square().mean()
                total=total+self.placement_weight*corrective
                self.logger.record('train/placement_corrective_loss',corrective.item())
            if hasattr(self,'online_oracle'):
                sample=self.replay_buffer.sample(batch_size,env=self._vec_normalize_env)
                labels=torch.as_tensor(self.online_oracle.actions(sample.observations.detach().cpu().numpy()),device=self.device)
                corrective=self._supervised_loss(sample.observations,labels,True)
                total=total+self.online_oracle_weight*corrective
                self.logger.record('train/online_corrective_loss',corrective.item())
            total.backward();torch.nn.utils.clip_grad_norm_(self.actor.parameters(),1.)
            self.logger.record('train/reference_prior_loss',loss.item());self.logger.record('train/reference_prior_weight',weight)
        hook=self.actor.optimizer.register_step_pre_hook(regularize)
        try: super().train(gradient_steps,batch_size)
        finally: hook.remove()
    def _excluded_save_params(self):
        return super()._excluded_save_params()+['prior_observations','prior_actions','prior_jaw_probability','prior_rng','online_oracle',
            'placement_observations','placement_labels','placement_goal_indices','preserved_grasp_state']
