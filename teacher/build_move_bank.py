"""Reach dense unified-task resets through real approach, grasp, carry and home."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import numpy as np
import torch

from teacher.build_reset_bank import checked_reference
from teacher.grasp_checkpoints import ConservativeSAC,file_hash
from teacher.near_grasp import NearGraspEnv,LiftConfig
from teacher.move_teacher import MoveTask,MoveConfig
from teacher.place_reference import PlaceReference
from teacher.train_grasp import save_status
from teacher.transport import StateClient
from teacher.transport_place import TransportPlaceEnv


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--url',default='http://127.0.0.1:8081')
    parser.add_argument('--name-prefix',default='move39')
    parser.add_argument('--qualification',type=Path,default=Path('outputs/near-grasp-qualification-v37'))
    parser.add_argument('--reference',type=Path,default=Path('outputs/e2e4-teacher-bootstrap-v31/reference_0'))
    parser.add_argument('--train-seeds',type=int,nargs='+',default=[81001,81002,81003])
    parser.add_argument('--validation-seeds',type=int,nargs='+',default=[82001])
    parser.add_argument('--test-seeds',type=int,nargs='+',default=[83001])
    args=parser.parse_args();partitions={'train':args.train_seeds,'validation':args.validation_seeds,'test':args.test_seeds}
    seeds=sum(partitions.values(),[])
    if not all(partitions.values()) or len(set(seeds))!=len(seeds): parser.error('Need disjoint nonempty seed sets')
    qualified=json.loads((args.qualification/'status.json').read_text())
    contract=json.loads((args.qualification/'manifest.json').read_text());frozen=args.qualification/'frozen_actor.zip'
    if not qualified['passed'] or file_hash(frozen)!=qualified['checkpoint_sha256']: raise ValueError('Qualified grasp changed')
    rows,_=checked_reference(args.reference);args.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
    actor=ConservativeSAC.load(frozen,device='cpu');client=StateClient(args.url);grasp=transport=None
    entries=[];reference_hash=file_hash(args.reference/'steps.jsonl')
    def save(latest,baseline,task,seed,split,family,held=False):
        history={'move':'e2e4','seed':seed,'split':split,'family':family,'held_start':held,
            'prior_lift_verified':bool(task.lifted),'prior_max_lift_m':task.max_lift,
            'prior_release_verified':bool(task.released),'release_intent':task.release_location,
            'grasp_checkpoint_sha256':qualified['checkpoint_sha256'],'reference_sha256':reference_hash}
        saved=client.save_reset(f'{args.name_prefix}_s{seed}_{family}',history)['reset_context']
        entries.append({**saved,**history,'physical_step':task.steps,'prefix_step':task.steps})
        return entries[-1]
    try:
        for split,partition in partitions.items():
            for seed in partition:
                rng=np.random.default_rng(seed)
                latest=client.reset({'history':[],'move':'e2e4','pawn_offset_mm':rng.uniform(-.5,.5,2).tolist()})
                baseline=latest['observation'];mapping=baseline['joint_map']
                command=np.array(mapping['sim_home_deg'])+np.array(mapping['scale'])*(np.array(baseline['target_legacy_action'])-mapping['legacy_home'])
                from review_grasp_geometry import opening_center
                from teacher.environment import ASSETS
                spec=json.loads((ASSETS/'kinematics.json').read_text())
                anchors=json.loads((ASSETS/'finger_reference.json').read_text())['anchors_jaw_local_m']
                def point(actual): return opening_center(spec,anchors,actual['measured_joint_deg'],actual['measured_finger_joint_deg']['jaw2_hinge'])
                task=MoveTask(baseline,baseline,point(baseline),command,{},MoveConfig())
                save(latest,baseline,task,seed,split,'home')
                for tick,row in enumerate(rows[:140],1):
                    latest=client.step(row['teacher']['executed_legacy']);actual=latest['observation']
                    _,done,truncated,score=task.update(actual,point(actual),row['teacher']['command_joint_deg'])
                    if done or truncated: raise ValueError(f'Approach failed: {score}')
                    if tick==100: save(latest,baseline,task,seed,split,'approach')
                    if tick==140: save(latest,baseline,task,seed,split,'open_grasp')
                approach={**client.save_reset(f'{args.name_prefix}_s{seed}_grasp_start',{'family':'open_approach','split':'train',
                    'seed':seed,'prefix_step':140})['reset_context'],'family':'open_approach','split':'train','prefix_step':140}
                temporary=args.output/'approach.json';temporary.write_text(json.dumps({'entries':[approach]}))
                if grasp is None: grasp=NearGraspEnv(temporary,args.url,lift_config=LiftConfig(**contract['lift_config']))
                grasp.fixed_entry=approach;obs,_=grasp.reset(seed=seed)
                trajectory=[]
                while True:
                    action,_=actor.predict(obs,deterministic=True)
                    obs,reward,done,truncated,info=grasp.step(action)
                    actual=grasp.base.actual;command=grasp.base.command
                    _,move_done,move_truncated,score=task.update(actual,point(actual),command)
                    trajectory.append({'source':'frozen_grasp','action':action.tolist(),'command_joint_deg':command.tolist(),
                                       'observation':actual,'score':score})
                    if move_done or move_truncated: raise ValueError('Unified task rejected grasp prefix')
                    if done or truncated:
                        if not info['score']['success']: raise ValueError(f'Frozen grasp failed seed {seed}')
                        break
                client.read();held=save(latest,baseline,task,seed,split,'held_source',True)
                transport_manifest=args.output/'held.json'
                transport_manifest.write_text(json.dumps({'entries':[dict(held,split='train')]}))
                if transport is None: transport=TransportPlaceEnv(transport_manifest,args.url)
                transport.fixed_entry=held;obs,_=transport.reset(seed=seed)
                probe=PlaceReference(transport)
                saved_families=set()
                while True:
                    action,_=probe.predict(obs,deterministic=True)
                    obs,reward,done,truncated,info=transport.step(action)
                    actual=transport.base.actual;command=transport.base.command
                    _,move_done,move_truncated,score=task.update(actual,point(actual),command)
                    trajectory.append({'source':'carry_feasibility','action':action.tolist(),
                        'command_joint_deg':command.tolist(),'observation':actual,'score':score})
                    if move_done or move_truncated: raise ValueError(f'Unified carry failed: {score}')
                    center,height,error,contact=task.measure(actual,point(actual),command)
                    fraction=float((center[1]-task.original[1])/(task.target[1]-task.original[1]))
                    family=None
                    if not task.released and contact:
                        if height>=.027 and fraction<.25: family='lift_clear'
                        elif .4<=fraction<=.7 and height>.025: family='carry_mid'
                        elif np.linalg.norm(center[:2]-task.target[:2])<.008 and height<.005: family='near_release'
                    if family and family not in saved_families:
                        client.read();save(info['telemetry'],baseline,task,seed,split,family,True);saved_families.add(family)
                        transport.base.client.read()  # save_reset advanced the shared revision.
                    if done or truncated:
                        if not info['score']['success']: raise ValueError(f'Carry feasibility failed {info["score"]}')
                        break
                if saved_families!={'lift_clear','carry_mid','near_release'}: raise ValueError('Dense carry coverage incomplete')
                client.read();latest=client.read();save(latest,baseline,task,seed,split,'retreat')
                # A final home continuation is only a feasibility/reset-building
                # reference. Learned actions are never replaced by this route.
                start=transport.base.command.copy();home=np.array(baseline['measured_joint_deg']);home[5]=5.4
                count=max(1,int(np.ceil(np.max(np.abs(home-start)*30/[15,15,15,15,15,45]))))
                saved_home=False;saved_settle=False
                for fraction in list(np.linspace(0,1,count+1)[1:])+[1.]*25:
                    goal=start+(home-start)*fraction
                    legacy=np.array(mapping['legacy_home'])+(goal-mapping['sim_home_deg'])/mapping['scale']
                    latest=client.step(legacy);actual=latest['observation']
                    _,done,truncated,score=task.update(actual,point(actual),goal)
                    trajectory.append({'source':'home_feasibility','command_joint_deg':goal.tolist(),'observation':actual,'score':score})
                    if not saved_home and 6<score['home_error_deg']<12:
                        save(latest,baseline,task,seed,split,'near_home');saved_home=True
                    if not saved_settle and score['home_error_deg']<3.5 and not done:
                        save(latest,baseline,task,seed,split,'home_settle');saved_settle=True
                    if done or truncated:
                        if not score['success']: raise ValueError(f'Home continuation failed: {score}')
                        break
                if not task.done or not score['success'] or not saved_home or not saved_settle:
                    raise ValueError('No complete reset-building continuation')
                trace=args.output/f'seed_{seed}.jsonl'
                trace.write_text(''.join(json.dumps(r)+'\n' for r in trajectory))
                for entry in entries:
                    if entry['seed']==seed:
                        entry['continuation_sha256']=file_hash(trace)
                        entry['horizon']=min(800,max(96,int(np.ceil((task.steps-entry['physical_step'])*1.7+64))))
                save_status(args.output/'status.json',stage='building',seed=seed,checkpoints=len(entries))
                print('MOVE_RESETS_SAVED',json.dumps({'seed':seed,'split':split,'checkpoints':len(entries)}),flush=True)
        for entry in entries:
            context=client.restore_reset(entry['name'])['reset_context']
            if context['checkpoint_sha256']!=entry['sha256'] or max(context['parity'].values())>1e-5:
                raise ValueError('Reset failed parity')
            entry['restore_parity']=context['parity']
        manifest={'format':'unified_move_reset_set_v1','case':{'id':'initial_e2e4','history':[],'move':'e2e4'},
            'entries':entries,'partitions':partitions,'reference':str(args.reference.resolve()),
            'continuation_directory':str(args.output.resolve()),
            'reference_sha256':reference_hash,'grasp_checkpoint':str(frozen.resolve()),
            'grasp_checkpoint_sha256':qualified['checkpoint_sha256'],
            'goal':'Pawn placed/released upright on e4 and blue arm back at original viewing pose',
            'limitations':'Reset-building references prove feasibility, not learned-policy success; PhysX caches rebuild.'}
        (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2))
        save_status(args.output/'status.json',stage='ready',checkpoints=len(entries),complete_reference_continuations=len(seeds))
        print('UNIFIED_BANK_READY',str(args.output/'manifest.json'),flush=True)
    except BaseException as error:
        save_status(args.output/'status.json',stage='failed',error=repr(error));raise
    finally:
        if grasp: grasp.close()
        if transport: transport.close()
        client.close()


if __name__=='__main__': main()
