"""Fit a unified actor from audited training references; critics remain fresh.

This is an explicit comparison to scratch RL, not the SGS paper's demo-free
recipe. Counterfactual setpoints supply BC labels only, never replay transitions.
"""
import argparse
import copy
import json
from pathlib import Path
import gymnasium as gym
import numpy as np
import torch

from teacher.grasp_checkpoints import ConservativeSAC,file_hash
from teacher.imitation import StandardizedFeatures
from teacher.train_grasp import save_status
from teacher.environment import ASSETS
from teacher.kinematics import OpeningMidpoint


def augment_setpoints(inputs,labels,offset,low,high,max_step):
    """Keep the recorded next target, with a bounded counterfactual setpoint."""
    inputs=inputs.clone()
    q=inputs[:,:6]*180
    command=q+inputs[:,23:29]*5
    target=(command+labels*max_step).clamp(low,high)
    perturbed=(command+offset).clamp(low,high)
    inputs[:,23:29]=(perturbed-q)/5
    return inputs,((target-perturbed)/max_step).clamp(-1,1)


def augment_free_arm(inputs,labels,aux,noise,midpoint,low,high,max_step,home_tau):
    """Geometry-consistent BC corrections in free space, never fake replay."""
    inputs=inputs.clone();q=inputs[:,:6]*180;command=q+inputs[:,23:29]*5
    target=(command+labels*max_step).clamp(low,high)
    offset=noise*aux['free_arm_before'][:,None];offset[:,5]=0
    perturbed_q=(q+offset).clamp(low,high);offset=perturbed_q-q
    with torch.no_grad():
        point_before=midpoint(q,inputs[:,22]*10);point_after=midpoint(perturbed_q,inputs[:,22]*10)
    inputs[:,:6]=perturbed_q/180;inputs[:,12:15]-=(point_after-point_before)/.03
    new_command=(command+offset).clamp(low,high)
    inputs[:,23:29]=(new_command-perturbed_q)/5
    inputs[:,62:67]=(aux['home_joint_deg'][:,:5]-perturbed_q[:,:5])/90
    centers=aux['all_centers_m'];distances=torch.linalg.norm(centers-point_after[:,None,:],dim=2)
    distances[torch.arange(len(inputs)),aux['source_index'].long()]=float('inf')
    nearest=distances.topk(4,largest=False).indices
    points=centers.gather(1,nearest[:,:,None].expand(-1,-1,3))
    upright=aux['all_upright'].gather(1,nearest)
    inputs[:,42:58]=torch.cat([(points-point_after[:,None,:])/.05,upright[:,:,None]],dim=2).flatten(1)
    labels=((target-new_command)/max_step).clamp(-1,1)
    if home_tau>0:
        # Corrective supervision only for already released, clear empty-hand
        # states. Runtime inference remains one learned actor with no override.
        home_action=((aux['home_joint_deg']-new_command)/(max_step*30*home_tau)).clamp(-1,1)
        labels=torch.where(aux['home_feedback'][:,None],home_action,labels)
    return inputs,labels


class OfflineSpace(gym.Env):
    def __init__(self,size):
        self.observation_space=gym.spaces.Box(-np.inf,np.inf,(size,),np.float32)
        self.action_space=gym.spaces.Box(-1.,1.,(6,),np.float32)
    def reset(self,*,seed=None,options=None):
        super().reset(seed=seed);return np.zeros(self.observation_space.shape,np.float32),{}
    def step(self,action): raise RuntimeError('Offline fitting cannot fabricate simulator transitions')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--updates',type=int,default=3000);parser.add_argument('--seed',type=int,default=86001)
    parser.add_argument('--free-arm-noise-deg',type=float,default=0.)
    parser.add_argument('--home-feedback-tau',type=float,default=0.,help='Training-only proportional home correction for released, clear states')
    parser.add_argument('--warm-actor',type=Path,help='Preserve an existing fitted actor while learning corrective home feedback')
    parser.add_argument('--preserve-nonhome-weight',type=float,default=0.)
    parser.add_argument('--cartesian-pose-noise-deg',type=float,default=0.)
    parser.add_argument('--history-dropout',type=float,default=0.)
    parser.add_argument('--drop-settled-dwell-labels',action='store_true')
    args=parser.parse_args()
    if args.updates<1 or args.seed<0 or not np.isfinite([args.free_arm_noise_deg,args.home_feedback_tau]).all() or min(args.free_arm_noise_deg,args.home_feedback_tau)<0:
        parser.error('Invalid fit budget')
    if args.preserve_nonhome_weight<0 or (args.preserve_nonhome_weight and not (args.warm_actor and args.home_feedback_tau)):
        parser.error('Non-home preservation needs a warm actor and home corrective supervision')
    if not np.isfinite(args.cartesian_pose_noise_deg) or args.cartesian_pose_noise_deg<0 or not 0<=args.history_dropout<1:
        parser.error('Invalid geometry augmentation or history dropout')
    args.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2);torch.manual_seed(args.seed)
    with np.load(args.data,allow_pickle=False) as archive:
        metadata=json.loads(str(archive['metadata_json']));x=np.asarray(archive['observation'],np.float32);y=np.asarray(archive['action'],np.float32)
        released=archive['released_before'].copy() if args.drop_settled_dwell_labels else None
        auxiliary={k:torch.as_tensor(archive[k]) for k in ('home_feedback','free_arm_before','all_centers_m','all_upright','source_index','home_joint_deg','released_before')} if args.free_arm_noise_deg or args.home_feedback_tau or args.cartesian_pose_noise_deg else None
    cartesian=bool(metadata.get('policy_action_space'))
    if cartesian and (args.free_arm_noise_deg or args.home_feedback_tau or args.warm_actor):
        raise ValueError('Joint-space corrective augmentation is not applicable to Cartesian policy labels')
    if args.cartesian_pose_noise_deg and not cartesian: raise ValueError('Cartesian augmentation requires task-space labels')
    original_frames=len(x);removed=0
    if args.drop_settled_dwell_labels:
        if not cartesian: raise ValueError('Dwell-label filtering requires Cartesian open/close commands')
        from teacher.demonstration_filter import settled_dwell
        keep=~settled_dwell(x,y,released);removed=int((~keep).sum());x=x[keep];y=y[keep]
        if auxiliary is not None: auxiliary={k:v[keep] for k,v in auxiliary.items()}
    if (not metadata['training_only'] or any(s['split']!='train' for s in metadata['sources'])
            or x.shape[1]!=(76 if cartesian else 71) or y.shape!=(len(x),6) or not np.isfinite(x).all() or not np.isfinite(y).all()):
        raise ValueError('Need valid training-only unified reference transitions')
    source=json.loads(Path(metadata['manifest']).read_text())
    if file_hash(metadata['manifest'])!=metadata['manifest_sha256']: raise ValueError('Reference manifest changed')
    rng=np.random.default_rng(args.seed);mean=x.mean(0);scale=np.maximum(x.std(0),.05)
    if args.warm_actor:
        parent_status=json.loads((args.warm_actor.parent/'status.json').read_text())
        if parent_status.get('critics_trained') or parent_status.get('replay_size')!=0: raise ValueError('Parent must retain fresh critics/replay')
        model=ConservativeSAC.load(args.warm_actor,env=OfflineSpace(x.shape[1]),device='cpu',learning_rate=.0001)
        for group in model.actor.optimizer.param_groups: group['lr']=.0001
        frozen=copy.deepcopy(model.actor)
        for p in frozen.parameters(): p.requires_grad_(False)
    else:
        model=ConservativeSAC('MlpPolicy',OfflineSpace(x.shape[1]),seed=args.seed,device='cpu',learning_rate=3e-4,
            gamma=metadata['action_config']['gamma'],buffer_size=30000,learning_starts=512,batch_size=128,ent_coef=.002,
            policy_kwargs={'net_arch':[256,256],'features_extractor_class':StandardizedFeatures,
                'features_extractor_kwargs':{'mean':mean.tolist(),'scale':scale.tolist()}},verbose=0)
        with torch.no_grad(): model.actor.log_std.weight.zero_();model.actor.log_std.bias.fill_(-3.)
    manifest={**source,'algorithm':'actor behavioral cloning; critics and replay fresh','actor_learning_rate':.0001 if args.warm_actor else .0003,
        'critic_learning_rate':None,'bc_loss':'normalized action MSE','actor_log_std_bias':-3.,
        'initialization':'behavioral cloning from audited training references; critics/replay fresh',
        'comparison':'data-assisted bootstrap, not demo-free SGS','training_data':str(args.data.resolve()),
        'training_data_sha256':file_hash(args.data),'source_episodes':metadata['sources'],'seed':args.seed,
        'bc_updates':args.updates,'validation_used_for_gradients':False,'final_test_used_for_selection':False,
        'observation_normalization':'training frames only; standard deviation floor .05',
        'setpoint_augmentation_std_deg':0. if cartesian else .1,'normalised_action_loss':True,'ready_for_act_collection':False,
        'free_arm_augmentation_std_deg':args.free_arm_noise_deg,'home_feedback_tau_s':args.home_feedback_tau,
        'home_feedback_scope':'supervised corrective labels for released/clear states only; never a runtime controller',
        'warm_actor':str(args.warm_actor.resolve()) if args.warm_actor else None,
        'warm_actor_sha256':file_hash(args.warm_actor) if args.warm_actor else None,
        'preserve_nonhome_weight':args.preserve_nonhome_weight,
        'cartesian_pose_augmentation_std_deg':args.cartesian_pose_noise_deg,'previous_action_dropout_fraction':args.history_dropout,
        'drop_settled_dwell_labels':args.drop_settled_dwell_labels,'source_physical_frames':original_frames,
        'removed_settled_dwell_labels':removed,'bc_used_frames':len(x),
        'cartesian_augmentation':'Opening FK and all correlated piece/home/neighbour features updated; held pawn moves rigidly; corrective supervised labels only, never replay transitions',
        'source_hashes':{str(p):file_hash(p) for p in Path('teacher').glob('*.py')}}
    if cartesian: manifest['bc_loss']='normalized arm task-space MSE plus binary gripper BCE'
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    snapshot=args.output/'source';snapshot.mkdir()
    for p in Path('teacher').glob('*.py'): (snapshot/p.name).write_bytes(p.read_bytes())
    xt=torch.as_tensor(x);yt=torch.as_tensor(y);max_step=torch.as_tensor(metadata['action_config']['max_velocity_deg_s'])/30
    limits=torch.as_tensor(json.loads((ASSETS/'kinematics.json').read_text())['limits'],dtype=torch.float32)
    low,high=limits[:,0]+1e-4,limits[:,1]-1e-4
    spec=json.loads((ASSETS/'kinematics.json').read_text());anchors=json.loads((ASSETS/'finger_reference.json').read_text())['anchors_jaw_local_m']
    midpoint=OpeningMidpoint(spec,anchors)
    history=[]
    for update in range(args.updates+1):
        if update%250==0 or update==args.updates:
            with torch.no_grad():
                prediction_all=model.actor(xt,deterministic=True);error=prediction_all-yt
            row={'stage':'unified_reference_bc','update':update,'requested_updates':args.updates,
                'training_action_rmse':float(error.square().mean().sqrt())}
            if cartesian:
                row['training_translation_rmse_mm']=float((error[:,:3]*metadata['cartesian_config']['translation_step_m']*1000).square().mean().sqrt())
                row['training_gripper_sign_accuracy']=float(((prediction_all[:,5]>0)==(yt[:,5]>0)).float().mean())
            else: row['training_increment_rmse_deg']=float((error*max_step).square().mean().sqrt())
            history.append(row);save_status(args.output/'status.json',**row);print('UNIFIED_BC_PROGRESS',json.dumps(row),flush=True)
        if update==args.updates: break
        indices=rng.integers(len(x),size=256);inputs=xt[indices].clone();labels=yt[indices].clone()
        if auxiliary is not None and not cartesian:
            aux={k:v[indices] for k,v in auxiliary.items()}
            if args.preserve_nonhome_weight:
                aux['free_arm_before']=aux['free_arm_before'] & aux['home_feedback']
            noise=torch.as_tensor(rng.normal(0,args.free_arm_noise_deg,(len(indices),6)),dtype=torch.float32)
            inputs,labels=augment_free_arm(inputs,labels,aux,noise,midpoint,low,high,max_step,args.home_feedback_tau)
        if cartesian and auxiliary is not None:
            from teacher.cartesian_augmentation import augment_cartesian
            aux={k:v[indices] for k,v in auxiliary.items()}
            noise=torch.as_tensor(rng.normal(0,args.cartesian_pose_noise_deg,(len(indices),6)),dtype=torch.float32)
            inputs,labels=augment_cartesian(inputs,labels,aux,noise,midpoint,low,high,metadata['cartesian_config'])
        if args.history_dropout:
            inputs[torch.as_tensor(rng.random(len(indices))<args.history_dropout),29:35]=0
        offset=torch.as_tensor(rng.normal(0,.1,(len(indices),6)),dtype=torch.float32)
        # Only commanded setpoint error changes. Measured joints, geometry,
        # contacts and velocities remain as recorded. This is supervised data,
        # not an invented physical transition or reward.
        if not cartesian: inputs,labels=augment_setpoints(inputs,labels,offset,low,high,max_step)
        predicted=model.actor(inputs,deterministic=True);loss=(predicted-labels).square().mean()
        if cartesian:
            mean_action=model.actor.get_action_dist_params(inputs)[0]
            loss=(predicted[:,:5]-labels[:,:5]).square().mean()+torch.nn.functional.binary_cross_entropy_with_logits(mean_action[:,5],(labels[:,5]>0).float())
        if args.preserve_nonhome_weight:
            home=aux['home_feedback'];rest=~home
            with torch.no_grad(): original=frozen(inputs,deterministic=True)
            loss=(predicted[home]-labels[home]).square().mean() if home.any() else predicted.sum()*0
            if rest.any(): loss=loss+args.preserve_nonhome_weight*(predicted[rest]-original[rest]).square().mean()
        model.actor.optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.actor.parameters(),5.)
        model.actor.optimizer.step()
    model.actor.optimizer.state.clear();model.save(args.output/'move_bc')
    (args.output/'fit_history.json').write_text(json.dumps(history,indent=2))
    save_status(args.output/'status.json',stage='unified_reference_bc_complete',updates=args.updates,
        training_frames=len(x),training_seeds=sorted({s['seed'] for s in metadata['sources']}),
        training_action_rmse=history[-1]['training_action_rmse'],checkpoint_sha256=file_hash(args.output/'move_bc.zip'),
        critics_trained=False,replay_size=0,closed_loop_evaluated=False,ready_for_act_collection=False)


if __name__=='__main__': main()
