"""Fit a neural actor to geometric corrective labels on training physics states.

The oracle is a supervised training aid only. Saved inference is the neural SAC
actor, with no oracle calls or trajectory/phase controller at runtime. Synthetic
pose variants are supervised inputs only, never physical RL transitions.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import numpy as np
import torch
from teacher.bootstrap_move import OfflineSpace
from teacher.cartesian_augmentation import augment_cartesian
from teacher.environment import ASSETS
from teacher.grasp_checkpoints import ConservativeSAC, file_hash
from teacher.imitation import StandardizedFeatures
from teacher.kinematics import OpeningMidpoint
from teacher.reactive_oracle import ReactiveOracle
from teacher.task_space import CartesianConfig
from teacher.train_grasp import save_status


def load_training(paths):
    arrays=[];sources=[];manifest=None;training_seeds=set();reserved_seeds=set()
    keys=('observation','action','free_arm_before','released_before','all_centers_m',
          'all_upright','source_index','home_joint_deg')
    for path in paths:
        with np.load(path,allow_pickle=False) as archive:
            meta=json.loads(str(archive['metadata_json']))
            if not meta['training_only'] or any(s['split']!='train' for s in meta['sources']):
                raise ValueError('Oracle fitting accepts training states only')
            if file_hash(meta['manifest'])!=meta['manifest_sha256']:
                raise ValueError('Training source manifest changed')
            current=json.loads(Path(meta['manifest']).read_text())
            if file_hash(current['bank'])!=current['bank_sha256']: raise ValueError('Reset bank changed')
            bank=json.loads(Path(current['bank']).read_text());entries={e['name']:e for e in bank['entries']}
            reserved_seeds.update(e['seed'] for e in bank['entries'] if e['split']!='train')
            for s in meta['sources']:
                e=entries[s['reset_name']]
                if e['split']!='train' or e['seed']!=s['seed']: raise ValueError('Training episode membership differs from bank')
                training_seeds.add(s['seed'])
            if current.get('policy_action_space')!='opening_xyz_pitch_roll_binary_jaw_v1':
                raise ValueError('Oracle fitting requires the task-space action contract')
            if manifest is not None and any(current[k]!=manifest[k] for k in
                    ('case','action_config','move_config','observation_schema','joint_map','cartesian_config')):
                raise ValueError('Training contracts differ')
            manifest=current
            row={k:archive[k].copy() for k in keys}
            if row['observation'].ndim!=2 or row['observation'].shape[1]!=76 or not np.isfinite(row['observation']).all():
                raise ValueError('Invalid physical training inputs')
            arrays.append(row);sources.append({'path':str(path.resolve()),'sha256':file_hash(path),
                'frames':len(row['observation']),'episodes':meta['sources']})
    if training_seeds & reserved_seeds: raise ValueError('Cross-bank training/evaluation seed leakage')
    return {k:np.concatenate([a[k] for a in arrays]) for k in keys},sources,manifest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,nargs='+',required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--updates',type=int,default=12000);p.add_argument('--seed',type=int,default=98001)
    p.add_argument('--noise-deg',type=float,default=.5)
    p.add_argument('--warm-actor',type=Path)
    p.add_argument('--features',choices=['full','geometry'],default='full')
    args=p.parse_args()
    if args.updates<1 or not np.isfinite(args.noise_deg) or args.noise_deg<0: p.error('Invalid fit budget/augmentation')
    data,sources,source=load_training(args.data)
    args.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2);torch.manual_seed(args.seed)
    rng=np.random.default_rng(args.seed);x=torch.tensor(data['observation'],dtype=torch.float32)
    auxiliary={k:torch.as_tensor(v) for k,v in data.items() if k not in ('observation','action')}
    oracle=ReactiveOracle(CartesianConfig(**source['cartesian_config']))
    y=torch.from_numpy(oracle.actions(x.numpy()))
    if args.warm_actor:
        parent=json.loads((args.warm_actor.parent/'manifest.json').read_text())
        if any(parent[k]!=source[k] for k in ('action_config','observation_schema','cartesian_config','joint_map')):
            raise ValueError('Warm actor contract differs')
        model=ConservativeSAC.load(args.warm_actor,env=OfflineSpace(76),device='cpu')
        for group in model.actor.optimizer.param_groups: group['lr']=.0001
    else:
        extractor=StandardizedFeatures;features=x
        if args.features=='geometry':
            from teacher.geometry_features import GeometryFeatures,geometry_values
            extractor=GeometryFeatures;features=geometry_values(x)
        model=ConservativeSAC('MlpPolicy',OfflineSpace(76),seed=args.seed,device='cpu',
            learning_rate=.0003,gamma=source['action_config']['gamma'],buffer_size=30000,
            learning_starts=512,batch_size=128,ent_coef=.002,
            policy_kwargs={'net_arch':[256,256],'features_extractor_class':extractor,
                'features_extractor_kwargs':{'mean':features.mean(0).tolist(),'scale':features.std(0).clamp(min=.15).tolist()}},verbose=0)
    with torch.no_grad(): model.actor.log_std.weight.zero_();model.actor.log_std.bias.fill_(-3.)
    manifest={**source,'algorithm':'neural actor imitation of analytic corrective labels; no RL updates',
        'teacher_label':'neural_actor','learned_policy':True,'oracle_training_only':True,
        'runtime_oracle':False,'training_sources':sources,'oracle_config':asdict(oracle.config),
        'oracle_version':oracle.version,
        'bc_updates':args.updates,'seed':args.seed,'pose_noise_deg':args.noise_deg,
        'actor_features':args.features if not args.warm_actor else 'inherited from warm actor',
        'warm_actor':str(args.warm_actor.resolve()) if args.warm_actor else None,
        'warm_actor_sha256':file_hash(args.warm_actor) if args.warm_actor else None,
        'training_enabled':True,'validation_used_for_gradients':False,'ready_for_act_collection':False,
        'source_hashes':{str(path):file_hash(path) for path in Path('teacher').glob('*.py')}}
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    snap=args.output/'source';snap.mkdir()
    for path in Path('teacher').glob('*.py'): (snap/path.name).write_bytes(path.read_bytes())
    spec=json.loads((ASSETS/'kinematics.json').read_text())
    anchors=json.loads((ASSETS/'finger_reference.json').read_text())['anchors_jaw_local_m']
    midpoint=OpeningMidpoint(spec,anchors);limits=torch.tensor(spec['limits']);low,high=limits[:,0]+1e-4,limits[:,1]-1e-4
    history=[]
    for update in range(args.updates+1):
        if update%1000==0 or update==args.updates:
            with torch.no_grad(): prediction=model.actor(x,deterministic=True)
            row={'stage':'oracle_actor_fit','update':update,'requested_updates':args.updates,'frames':len(x),
                'training_arm_rmse':float((prediction[:,:5]-y[:,:5]).square().mean().sqrt()),
                'training_jaw_sign_accuracy':float(((prediction[:,5]>0)==(y[:,5]>0)).float().mean())}
            save_status(args.output/'status.json',**row);history.append(row);print('ORACLE_FIT',json.dumps(row),flush=True)
        if update==args.updates: break
        ids=rng.integers(len(x),size=256);inputs=x[ids].clone()
        if args.noise_deg:
            aux={k:v[ids] for k,v in auxiliary.items()}
            noise=torch.from_numpy(rng.normal(0,args.noise_deg,(len(ids),6)).astype(np.float32))
            inputs,_=augment_cartesian(inputs,y[ids],aux,noise,midpoint,low,high,source['cartesian_config'])
        labels=torch.from_numpy(oracle.actions(inputs.numpy()))
        prediction=model.actor(inputs,deterministic=True)
        logits=model.actor.get_action_dist_params(inputs)[0]
        loss=(prediction[:,:5]-labels[:,:5]).square().mean()+torch.nn.functional.binary_cross_entropy_with_logits(logits[:,5],(labels[:,5]>0).float())
        model.actor.optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.actor.parameters(),5.)
        model.actor.optimizer.step()
    model.actor.optimizer.state.clear();model.save(args.output/'move_bc')
    (args.output/'fit_history.json').write_text(json.dumps(history,indent=2))
    save_status(args.output/'status.json',stage='oracle_actor_fit_complete',updates=args.updates,frames=len(x),
        checkpoint_sha256=file_hash(args.output/'move_bc.zip'),critics_trained=False,replay_size=0,
        closed_loop_evaluated=False,ready_for_act_collection=False)


if __name__=='__main__': main()
