"""Training-only arm supervision for physically held-piece transport/placement.

Pose variants preserve the held piece relative to the opening. They are labels,
never physical replay, new episodes, jaw-timing overrides or accepted ACT data.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import numpy as np
import torch
from teacher.cartesian_augmentation import augment_cartesian
from teacher.environment import ASSETS
from teacher.fit_oracle import load_training
from teacher.grasp_checkpoints import file_hash
from teacher.kinematics import OpeningMidpoint


def placement_mask(values):
    x=np.asarray(values);q=x[:,:6]*180;command=q+x[:,23:29]*5
    f1=x[:,21]*10;f2=-x[:,22]*10
    held=(command[:,5]<-2.5)&(np.maximum(f1,f2)<3.5)&(f1>command[:,5]+.8)&(f2>command[:,5]+.8)
    held&=np.linalg.norm(x[:,12:15]*.03,axis=1)<.012
    return held&(x[:,17]*.03>.003)&(x[:,17]*.03<.065)&(np.linalg.norm(x[:,18:20]*.1,axis=1)<.03)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,nargs='+',required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--variants',type=int,default=2);p.add_argument('--noise-deg',type=float,default=.8)
    p.add_argument('--seed',type=int,default=138002)
    args=p.parse_args()
    if args.output.exists() or args.variants<0 or not np.isfinite(args.noise_deg) or not 0<=args.noise_deg<=2:
        p.error('Need a new output and valid supervised variation')
    torch.set_num_threads(2);data,sources,source=load_training(args.data)
    if source.get('oracle_action_field')!='configuration_reach_and_return_with_task_space_transport':
        raise ValueError('Placement supervision must retain the collision-aware reference')
    keep=placement_mask(data['observation'])&~data['released_before'].astype(bool)
    if not keep.any():raise ValueError('No physically held placement states')
    from teacher.configuration_oracle import ConfigurationReference
    from teacher.reactive_oracle import FlowConfig
    from teacher.task_space import CartesianControl,CartesianConfig
    config=CartesianConfig(**source['cartesian_config']);spec=json.loads((ASSETS/'kinematics.json').read_text())
    limits=torch.tensor(spec['limits']);anchors=json.loads((ASSETS/'finger_reference.json').read_text())['anchors_jaw_local_m']
    control=CartesianControl(spec,anchors,limits[:,0].numpy()+1e-4,limits[:,1].numpy()-1e-4,
        source['action_config']['max_velocity_deg_s'],config)
    oracle=ConfigurationReference(control,FlowConfig(**source['oracle_config']))
    midpoint=OpeningMidpoint(spec,anchors);physical=torch.tensor(data['observation'][keep],dtype=torch.float32)
    auxiliary={k:torch.as_tensor(v[keep]) for k,v in data.items() if k not in ('observation','action','goal_id')}
    inputs=[physical];rng=np.random.default_rng(args.seed)
    for _ in range(args.variants):
        noise=torch.from_numpy(rng.normal(0,args.noise_deg,(len(physical),6)).astype(np.float32))
        value,_=augment_cartesian(physical,torch.as_tensor(data['action'][keep]),auxiliary,noise,
            midpoint,limits[:,0]+1e-4,limits[:,1]-1e-4,asdict(config))
        inputs.append(value)
    pool=torch.cat(inputs).numpy();selected=placement_mask(pool)
    labels=oracle.actions(pool[selected]);source_frames=np.tile(np.flatnonzero(keep),len(inputs))[selected]
    counterfactual=np.repeat(np.arange(len(inputs))>0,len(physical))[selected]
    goals=np.tile(data['goal_id'][keep],len(inputs))[selected]
    metadata={'format':'training_only_held_piece_arm_corrections_v1','training_only':True,
        'validation_used_for_gradients':False,'training_sources':sources,'physics_settings':source['physics_settings'],
        'oracle_version':oracle.version,'oracle_config':asdict(oracle.config),'physical_states':int(keep.sum()),
        'supervised_states':int(selected.sum()),'counterfactual_states':int(counterfactual.sum()),
        'noise_deg':args.noise_deg,'variants':args.variants,'seed':args.seed,
        'scope':'Supervised arm labels only; no physical replay, synthetic reward, jaw correction or runtime override',
        'source_hashes':{str(path):file_hash(path) for path in (Path(__file__),Path('teacher/configuration_oracle.py'),Path('teacher/reactive_oracle.py'),Path('teacher/cartesian_augmentation.py'))}}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('xb') as stream:
        np.savez_compressed(stream,observation=pool[selected],label=labels,source_frame=source_frames,
            goal_id=goals,counterfactual_observation=counterfactual,metadata_json=np.array(json.dumps(metadata)))
    print(json.dumps({'passed':True,'path':str(args.output),'sha256':file_hash(args.output),
        'physical_states':int(keep.sum()),'supervised_states':int(selected.sum()),'counterfactual_states':int(counterfactual.sum()),
        'states_per_goal':{str(g):int((goals==g).sum()) for g in np.unique(goals)}}),flush=True)


if __name__=='__main__':main()
