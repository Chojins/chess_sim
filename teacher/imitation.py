"""Fit state-policy actions to freshly retimed, independently checked episodes."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from chess_task import LegalMoveTask, TaskConfig
from teacher.environment import TeacherFeatures


class StandardizedFeatures(BaseFeaturesExtractor):
    """Training-only statistics travel inside the saved policy checkpoint."""
    def __init__(self, observation_space, mean, scale, time_max=None):
        super().__init__(observation_space,features_dim=observation_space.shape[0])
        self.register_buffer('mean',torch.as_tensor(mean,dtype=torch.float32))
        self.register_buffer('scale',torch.as_tensor(scale,dtype=torch.float32))
        if time_max is not None:
            self.register_buffer('time_max',torch.tensor(time_max,dtype=torch.float32))
        if self.mean.shape != observation_space.shape or self.scale.shape != self.mean.shape or torch.any(self.scale<=0):
            raise ValueError('Invalid observation normalization')

    def forward(self, observations):
        if hasattr(self,'time_max'):
            observations=observations.clone()
            observations[:,-1]=observations[:,-1].clamp(max=self.time_max)
        return ((observations-self.mean)/self.scale).clamp(-20,20)


def load_demonstrations(paths, validation_indices, specification, anchors, config):
    observations,actions,validation,episodes,sources=[],[],[],[],[]
    validation_indices=set(validation_indices)
    if not validation_indices or validation_indices==set(range(len(paths))):
        raise ValueError('Need separate whole training and validation episodes')
    if not validation_indices <= set(range(len(paths))):
        raise ValueError('Unknown validation episode index')
    seen=set()
    for episode,path in enumerate(map(Path,paths)):
        result=json.loads((path/'result.json').read_text())
        if not result['complete'] or result['config']['config']!=json.loads(json.dumps(asdict(config))):
            raise ValueError('Demonstration outcome or teacher control configuration differs')
        trace=path/'steps.jsonl'
        digest=hashlib.sha256(trace.read_bytes()).hexdigest()
        if digest in seen:
            raise ValueError('Duplicate demonstration')
        seen.add(digest)
        initial=json.loads((path/'initial.json').read_text())['telemetry']['observation']
        task=LegalMoveTask(initial,TaskConfig(max_steps=config.max_steps,capture_home_retry=True))
        encoder=TeacherFeatures(initial,specification,anchors,config)
        mapping=initial['joint_map']
        command=np.asarray(mapping['sim_home_deg'])+np.asarray(mapping['scale'])*(
            np.asarray(initial['target_legacy_action'])-mapping['legacy_home'])
        limits=np.asarray(initial['joint_limits_deg'])
        low,high=limits[:,0]+1e-4,limits[:,1]-1e-4
        command=np.clip(command,low,high)
        previous_q=np.asarray(initial['measured_joint_deg'])
        velocity=np.zeros(6)
        actual=initial
        rows=[json.loads(s) for s in trace.read_text().splitlines()]
        with np.load(path/'demonstration.npz',allow_pickle=False) as archive:
            if len(archive['observations']) != len(rows):
                raise ValueError('Missing demonstration transitions')
            for index,row in enumerate(rows):
                encoded=encoder.encode(actual,task,command,velocity,previous_q)
                np.testing.assert_allclose(encoded,archive['observations'][index],atol=1e-6)
                action=np.asarray(row['action'],np.float32)
                np.testing.assert_array_equal(action,archive['actions'][index])
                if action.shape!=(6,) or not np.isfinite(action).all() or np.max(np.abs(action))>1:
                    raise ValueError('Invalid demonstration action')
                max_step=np.asarray(config.max_velocity_deg_s)/30
                desired=(command+action*max_step if config.action_mode!='absolute' else low+(action+1)*.5*(high-low))
                if config.action_mode=='arm_delta_gripper_position':
                    desired[5]=low[5]+(action[5]+1)*.5*(high[5]-low[5])
                target=np.clip(command+np.clip(desired-command,-max_step,max_step),low,high)
                np.testing.assert_allclose(target,row['teacher']['command_joint_deg'],atol=1e-5)
                physical=np.asarray(mapping['sim_home_deg'])+np.asarray(mapping['scale'])*(
                    np.asarray(row['teacher']['executed_legacy'])-mapping['legacy_home'])
                np.testing.assert_allclose(physical,target,atol=1e-5)
                next_actual=row['observation']
                if row['step']!=index+1 or next_actual['control_step']!=index+1:
                    raise ValueError('Noncontiguous demonstration steps')
                np.testing.assert_allclose(next_actual['physics_time']-actual['physics_time'],1/30,atol=1e-6)
                _,terminated,truncated,score=task.update(next_actual)
                if score!=row['score'] or terminated!=row['terminated'] or truncated!=row['truncated']:
                    raise ValueError('Saved outcome differs from independent physics scoring')
                if bool(terminated or truncated)!=(index==len(rows)-1):
                    raise ValueError('Premature/missing demonstration outcome')
                observations.append(encoded);actions.append(action)
                validation.append(episode in validation_indices);episodes.append(episode)
                previous_q=np.asarray(actual['measured_joint_deg'])
                velocity=(target-command)*30
                command,actual=target,next_actual
        if not score['success'] or score!=result['score']:
            raise ValueError('Demonstration does not complete the move')
        sources.append({'path':str(path.resolve()),'trace_sha256':digest,'frames':len(rows),
                        'validation':episode in validation_indices})
    return {'observations':np.asarray(observations,np.float32),'actions':np.asarray(actions,np.float32),
            'validation':np.asarray(validation,bool),'episode':np.asarray(episodes,np.int32)},sources


def fit_actor(model,data,env,updates,seed,output,state_noise_deg=0.):
    if not np.isfinite(state_noise_deg) or state_noise_deg<0:
        raise ValueError('State augmentation must be finite and nonnegative')
    from teacher.kinematics import OpeningMidpoint
    midpoint=OpeningMidpoint(env.kinematics,env.anchors)
    rng=np.random.default_rng(seed)
    train=np.flatnonzero(~data['validation'])
    validation=data['validation']
    x=torch.as_tensor(data['observations'])
    y=torch.as_tensor(data['actions'])
    max_step=torch.as_tensor(np.asarray(env.config.max_velocity_deg_s)/30,dtype=torch.float32)
    low=torch.as_tensor(env.low,dtype=torch.float32)
    high=torch.as_tensor(env.high,dtype=torch.float32)
    action_scale=max_step.clone() if env.config.action_mode!='absolute' else (high-low)/2
    if env.config.action_mode=='arm_delta_gripper_position':
        action_scale[5]=(high[5]-low[5])/2
    best_loss=float('inf');best_actor=None;best_update=0;history=[]
    for update in range(updates+1):
        if update%100==0 or update==updates:
            with torch.no_grad():
                error=(model.actor(x[validation],deterministic=True)-y[validation])*action_scale
                loss=error.square().mean().item()
            history.append({'update':update,'validation_command_increment_rmse_deg':float(np.sqrt(loss))})
            if loss<best_loss:
                best_loss=loss;best_update=update
                best_actor={k:v.detach().clone() for k,v in model.actor.state_dict().items()}
            Path(output,'status.json').write_text(json.dumps({'stage':'behavioral_cloning','update':update,
                'requested_updates':updates,'validation_increment_rmse_deg':float(np.sqrt(loss))}))
        if update==updates:
            break
        indices=rng.choice(train,size=256,replace=True)
        inputs=x[indices].clone()
        labels=y[indices].clone()
        # A small setpoint perturbation teaches feedback instead of merely
        # copying the previous command. Measured state and piece poses stay
        # intact; the counterfactual target is the same demonstrated next pose.
        command=inputs[:,12:18]*180
        state_offset=torch.zeros_like(command)
        if state_noise_deg:
            q=inputs[:,:6]*180
            state_offset[:,:5]=torch.as_tensor(rng.normal(0,state_noise_deg,(len(inputs),5)),dtype=torch.float32)
            perturbed_q=(q+state_offset).clamp(low,high)
            state_offset=perturbed_q-q
            jaw2=inputs[:,34]*45
            with torch.no_grad():
                point_shift=(midpoint(perturbed_q,jaw2)-midpoint(q,jaw2))/.25
            inputs[:,:6]=perturbed_q/180
            inputs[:,30:33]+=point_shift
            pieces=inputs[:,35:259].reshape(-1,32,7)
            pieces[:,:,3:6]-=point_shift[:,None,:]
        offset=state_offset+torch.as_tensor(rng.normal(0,.25,command.shape),dtype=torch.float32)
        perturbed=(command+offset).clamp(low,high)
        offset=perturbed-command
        inputs[:,12:18]=perturbed/180
        if env.config.action_mode=='delta':
            labels=(labels-offset/max_step).clamp(-1,1)
        elif env.config.action_mode=='arm_delta_gripper_position':
            labels[:,:5]=(labels[:,:5]-offset[:,:5]/max_step[:5]).clamp(-1,1)
        prediction=model.actor(inputs,deterministic=True)
        loss=((prediction-labels)*action_scale).square().mean()
        model.actor.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.actor.parameters(),5.)
        model.actor.optimizer.step()
    model.actor.load_state_dict(best_actor)
    model.actor.optimizer.state.clear()
    report={'updates':updates,'selected_update':best_update,
            'validation_increment_rmse_deg':float(np.sqrt(best_loss)),
            'training_frames':len(train),'validation_frames':int(validation.sum()),
            'setpoint_augmentation_std_deg':.25,'validation_used_for_gradients':False,
            'robot_pose_augmentation_std_deg':state_noise_deg,
            'pose_augmentation':'arm joints and all correlated finger-midpoint/relative-piece features recomputed together; no simulator transitions fabricated',
            'history':history}
    Path(output,'imitation.json').write_text(json.dumps(report,indent=2))
    return report
