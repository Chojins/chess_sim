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
            'reward_relabeling':metadata.get('reward_relabeling')}
    def configure_online_oracle(self,env,weight):
        if not np.isfinite(weight) or weight<0: raise ValueError('Invalid corrective supervision weight')
        if not weight: return None
        from teacher.reactive_oracle import ReactiveOracle
        if getattr(env,'policy_action_space',None)!='opening_xyz_pitch_roll_binary_jaw_v1':
            raise ValueError('Corrective oracle requires the Cartesian geometry contract')
        self.online_oracle=ReactiveOracle(env.cartesian_config);self.online_oracle_weight=weight
        return {'weight':weight,'version':self.online_oracle.version,
            'scope':'training-only corrective labels on real replay observations; no runtime intervention',
            'replay_labels_changed':False}
    def _supervised_loss(self,observations,actions,cartesian):
        logits=self.actor.get_action_dist_params(observations)[0];predicted=torch.tanh(logits)
        if cartesian:
            return ((predicted[:,:5]-actions[:,:5]).square().mean()
                +torch.nn.functional.binary_cross_entropy_with_logits(logits[:,5],(actions[:,5]>0).float()))
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
            loss=self._supervised_loss(self.prior_observations[indices],self.prior_actions[indices],self.prior_cartesian)
            total=weight*loss
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
        return super()._excluded_save_params()+['prior_observations','prior_actions','prior_rng','online_oracle']
