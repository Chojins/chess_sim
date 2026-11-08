"""Fit a neural teacher from training-only configuration-reference supervision.

Counterfactual pose labels are precomputed once. They never enter physical SAC
replay or an ACT episode. Inference uses the saved neural actor alone.
"""
import argparse
from dataclasses import asdict
import json
import multiprocessing as mp
from pathlib import Path
import time
import numpy as np
import torch
from teacher.cartesian_augmentation import augment_cartesian
from teacher.environment import ASSETS
from teacher.fit_oracle import geometry_warm_actor,load_training
from teacher.geometry_features import PostureGeometryFeatures,posture_values
from teacher.grasp_checkpoints import ConservativeSAC,file_hash
from teacher.kinematics import OpeningMidpoint
from teacher.train_grasp import save_status


def label_init(source):
    global label_oracle
    from teacher.configuration_oracle import ConfigurationReference
    from teacher.reactive_oracle import FlowConfig
    from teacher.task_space import CartesianControl,CartesianConfig
    torch.set_num_threads(1)
    spec=json.loads((ASSETS/'kinematics.json').read_text());limits=np.asarray(spec['limits'])
    anchors=json.loads((ASSETS/'finger_reference.json').read_text())['anchors_jaw_local_m']
    controller=CartesianControl(spec,anchors,limits[:,0]+1e-4,limits[:,1]-1e-4,
        source['action_config']['max_velocity_deg_s'],CartesianConfig(**source['cartesian_config']))
    label_oracle=ConfigurationReference(controller,FlowConfig(**source['oracle_config']))


def label_chunk(payload):
    offset,values=payload
    return offset,label_oracle.actions(values)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,nargs='+',required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--warm-actor',type=Path,required=True);p.add_argument('--updates',type=int,default=20000)
    p.add_argument('--augmentations',type=int,default=1);p.add_argument('--noise-deg',type=float,default=.35)
    p.add_argument('--label-workers',type=int,default=4);p.add_argument('--seed',type=int,default=131001)
    p.add_argument('--home-basin-mm',type=float,help='Override the training-label return basin for an already-open gripper')
    p.add_argument('--placement-jitter-mm',type=float,default=0.,help='Supervised-only XY pose variation after verified training releases')
    p.add_argument('--return-arm-head-only',action='store_true',help='Preserve the warm jaw/trunk exactly; tune only the five arm output rows after release')
    p.add_argument('--preservation-weight',type=float,default=10.,help='Non-release arm imitation weight in return-only tuning')
    args=p.parse_args()
    if min(args.updates,args.label_workers)<1 or args.augmentations<0 or not np.isfinite(args.noise_deg) or args.noise_deg<0:
        p.error('Invalid fitting or augmentation settings')
    if not np.isfinite(args.placement_jitter_mm) or not 0<=args.placement_jitter_mm<=5 or (args.home_basin_mm is not None and (not np.isfinite(args.home_basin_mm) or not 3<=args.home_basin_mm<=10)):
        p.error('Invalid released-piece supervised variation')
    if not np.isfinite(args.preservation_weight) or args.preservation_weight<1:p.error('Invalid preservation weight')
    args.output.mkdir(parents=True,exist_ok=False);started=time.perf_counter();torch.set_num_threads(2);torch.manual_seed(args.seed)
    try:
        save_status(args.output/'status.json',stage='auditing_configuration_sources')
        data,sources,source=load_training(args.data)
        if source.get('oracle_action_field')!='configuration_reach_and_return_with_task_space_transport':
            raise ValueError('Need a configuration-reference source contract')
        if not source['physics_settings']['contacts'].get('robot_self_collisions'):
            raise ValueError('Configuration fit requires audited self-collision physics')
        original_label_config=dict(source['oracle_config'])
        if args.home_basin_mm is not None:source={**source,'oracle_config':{**source['oracle_config'],'home_basin_m':args.home_basin_mm/1000}}
        parent=json.loads((args.warm_actor.parent/'manifest.json').read_text())
        if any(parent[k]!=source[k] for k in ('action_config','observation_schema','cartesian_config','joint_map')):
            raise ValueError('Warm actor control contract differs')
        rng=np.random.default_rng(args.seed);physical=torch.tensor(data['observation'],dtype=torch.float32)
        auxiliary={k:torch.as_tensor(v) for k,v in data.items() if k not in ('observation','action','goal_id')}
        spec=json.loads((ASSETS/'kinematics.json').read_text());limits=torch.tensor(spec['limits'])
        anchors=json.loads((ASSETS/'finger_reference.json').read_text())['anchors_jaw_local_m'];midpoint=OpeningMidpoint(spec,anchors)
        inputs=[physical]
        for _ in range(args.augmentations):
            noise=torch.from_numpy(rng.normal(0,args.noise_deg,(len(physical),6)).astype(np.float32))
            altered,_=augment_cartesian(physical,torch.as_tensor(data['action']),auxiliary,noise,midpoint,
                limits[:,0]+1e-4,limits[:,1]-1e-4,source['cartesian_config'])
            if args.placement_jitter_mm:
                opened=torch.minimum(physical[:,21],-physical[:,22])*10>=4.
                eligible=auxiliary['released_before'] & opened
                offset=torch.from_numpy(rng.uniform(-args.placement_jitter_mm,args.placement_jitter_mm,(len(physical),2)).astype(np.float32))/1000
                offset*=eligible[:,None]
                altered[:,12:14]+=offset/.03;altered[:,15:17]+=offset/.03;altered[:,18:20]-=offset/.1
            inputs.append(altered)
        pool=torch.cat(inputs);goal_ids=np.tile(data['goal_id'],len(inputs));labels=np.empty((len(pool),6),np.float32)
        manifest={**source,'algorithm':'neural imitation of configuration-reference labels; no RL updates',
            'teacher_label':'neural_actor','learned_policy':True,'runtime_oracle':False,'oracle_training_only':True,
            'oracle_version':'configuration_feedback_reference_v1','training_sources':sources,
            'warm_actor':str(args.warm_actor.resolve()),'warm_actor_sha256':file_hash(args.warm_actor),
            'initialization':'actor weights only; fresh critics, optimizer and empty replay',
            'actor_features':'posture_geometry_57','bc_updates':args.updates,'seed':args.seed,
            'pose_noise_deg':args.noise_deg,'counterfactual_variants_per_physical_state':args.augmentations,
            'source_label_config':original_label_config,'placement_jitter_mm':args.placement_jitter_mm,
            'label_pool_scope':'Supervised observations/actions only; never physical replay or accepted ACT data',
            'sampling':'uniform goals, then uniform pooled observations within each goal',
            'training_enabled':True,'validation_used_for_gradients':False,'ready_for_act_collection':False,
            'source_hashes':{str(path):file_hash(path) for path in Path('teacher').glob('*.py')}}
        (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2));snapshot=args.output/'source';snapshot.mkdir()
        for path in Path('teacher').glob('*.py'): (snapshot/path.name).write_bytes(path.read_bytes())
        completed=0
        save_status(args.output/'status.json',stage='labelling_supervised_configuration_pool',completed=0,frames=len(pool))
        chunks=[(n,pool[n:n+64].numpy()) for n in range(0,len(pool),64)]
        with mp.get_context('spawn').Pool(args.label_workers,initializer=label_init,initargs=(source,)) as workers:
            for offset,value in workers.imap_unordered(label_chunk,chunks):
                labels[offset:offset+len(value)]=value;completed+=len(value)
                save_status(args.output/'status.json',stage='labelling_supervised_configuration_pool',completed=completed,
                    frames=len(pool),elapsed_seconds=time.perf_counter()-started)
        model=geometry_warm_actor(ConservativeSAC.load(args.warm_actor,device='cpu'),pool,args.seed,
            source['action_config']['gamma'],PostureGeometryFeatures,posture_values)
        with torch.no_grad():model.actor.log_std.weight.zero_();model.actor.log_std.bias.fill_(-3.)
        released=torch.from_numpy(np.tile(data['released_before'].astype(bool),len(inputs)))
        preserved_state=None;baseline=None
        if args.return_arm_head_only:
            preserved_state={k:v.clone() for k,v in model.actor.state_dict().items()}
            with torch.no_grad():baseline=model.actor(pool,deterministic=True).numpy()
            labels[~released.numpy(),:5]=baseline[~released.numpy(),:5];labels[:,5]=baseline[:,5]
            for parameter in model.actor.parameters():parameter.requires_grad_(False)
            model.actor.mu.weight.requires_grad_(True);model.actor.mu.bias.requires_grad_(True)
            model.actor.mu.weight.register_hook(lambda gradient:torch.cat([gradient[:5],torch.zeros_like(gradient[5:])]))
            model.actor.mu.bias.register_hook(lambda gradient:torch.cat([gradient[:5],torch.zeros_like(gradient[5:])]))
            manifest.update(return_arm_head_only=True,preservation_weight=args.preservation_weight,
                label_scope='Configuration feedback after verified releases; frozen-parent arm predictions before release; parent jaw preserved exactly')
        np.savez_compressed(args.output/'supervised_pool.npz',observation=pool.numpy(),label=labels,
            source_frame=np.tile(np.arange(len(physical)),len(inputs)),goal_id=goal_ids,
            counterfactual_observation=np.repeat(np.arange(len(inputs))>0,len(physical)))
        goals,goal_index,counts=np.unique(goal_ids,return_inverse=True,return_counts=True)
        probabilities=1/(len(goals)*counts[goal_index]);target=torch.from_numpy(labels);history=[]
        manifest.update(training_frames_by_goal=dict(zip(goals.tolist(),counts.tolist())),
            supervised_pool_sha256=file_hash(args.output/'supervised_pool.npz'))
        (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2))
        for update in range(args.updates+1):
            if update%1000==0 or update==args.updates:
                with torch.no_grad():prediction=model.actor(pool,deterministic=True)
                row={'stage':'configuration_actor_fit','update':update,'requested_updates':args.updates,'frames':len(pool),
                    'training_arm_rmse':float((prediction[:,:5]-target[:,:5]).square().mean().sqrt()),
                    'training_jaw_sign_accuracy':float(((prediction[:,5]>0)==(target[:,5]>0)).float().mean()),
                    'elapsed_seconds':time.perf_counter()-started}
                save_status(args.output/'status.json',**row);history.append(row);print('CONFIGURATION_FIT',json.dumps(row),flush=True)
            if update==args.updates:break
            ids=rng.choice(len(pool),size=256,p=probabilities);values=pool[ids];expected=target[ids]
            prediction=model.actor(values,deterministic=True);logits=model.actor.get_action_dist_params(values)[0]
            arm_error=(prediction[:,:5]-expected[:,:5]).square().mean(1)
            if args.return_arm_head_only:
                weights=torch.where(released[ids],1.,args.preservation_weight)
                loss=(arm_error*weights).mean()
            else:loss=arm_error.mean()+torch.nn.functional.binary_cross_entropy_with_logits(logits[:,5],(expected[:,5]>0).float())
            model.actor.optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.actor.parameters(),5.);model.actor.optimizer.step()
        if args.return_arm_head_only:
            for key,value in model.actor.state_dict().items():
                if key in ('mu.weight','mu.bias'):torch.testing.assert_close(value[5:],preserved_state[key][5:],rtol=0,atol=0)
                else:torch.testing.assert_close(value,preserved_state[key],rtol=0,atol=0)
            with torch.no_grad():after=model.actor(pool,deterministic=True).numpy()
            integrity={'passed':True,'jaw_and_trunk_parameters_exact':True,'jaw_output_max_abs_error':float(abs(after[:,5]-baseline[:,5]).max()),
                'non_release_arm_output_rmse':float(np.sqrt(np.mean((after[~released.numpy(),:5]-baseline[~released.numpy(),:5])**2)))}
            (args.output/'preservation_integrity.json').write_text(json.dumps(integrity,indent=2))
            for parameter in model.actor.parameters():parameter.requires_grad_(True)
        model.actor.optimizer.state.clear();model.save(args.output/'move_bc')
        (args.output/'fit_history.json').write_text(json.dumps(history,indent=2))
        save_status(args.output/'status.json',stage='configuration_actor_fit_complete',updates=args.updates,
            physical_frames=len(physical),supervised_frames=len(pool),checkpoint_sha256=file_hash(args.output/'move_bc.zip'),
            critics_trained=False,replay_size=0,closed_loop_evaluated=False,ready_for_act_collection=False)
    except BaseException as error:
        save_status(args.output/'status.json',stage='failed',error=repr(error));raise


if __name__=='__main__':main()
