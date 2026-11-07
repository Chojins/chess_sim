"""Physically reach held-source and held-target resets without moving object poses."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from chess_task import LegalMoveTask,TaskConfig
from review_grasp_geometry import opening_center
from teacher.build_reset_bank import checked_reference
from teacher.environment import ASSETS
from teacher.grasp_checkpoints import ConservativeSAC,file_hash
from teacher.near_grasp import NearGraspEnv,LiftConfig
from teacher.train_grasp import save_status
from teacher.transport import StateClient
from teacher.transport_place import TransportPlaceTask


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--url',default='http://127.0.0.1:8081')
    parser.add_argument('--qualification',type=Path,default=Path('outputs/near-grasp-qualification-v37'))
    parser.add_argument('--reference',type=Path,default=Path('outputs/e2e4-teacher-bootstrap-v31/reference_0'))
    parser.add_argument('--train-seeds',type=int,nargs='+',default=[72001,72002,72003])
    parser.add_argument('--validation-seeds',type=int,nargs='+',default=[73001,73002])
    parser.add_argument('--test-seeds',type=int,nargs='+',default=[74001,74002])
    args=parser.parse_args()
    partitions={'train':args.train_seeds,'validation':args.validation_seeds,'test':args.test_seeds}
    seeds=sum(partitions.values(),[])
    if len(set(seeds))!=len(seeds) or not all(partitions.values()):
        parser.error('Need nonempty, disjoint seed partitions')
    qualified=json.loads((args.qualification/'status.json').read_text())
    contract=json.loads((args.qualification/'manifest.json').read_text())
    frozen=args.qualification/'frozen_actor.zip'
    if not qualified['passed'] or file_hash(frozen)!=qualified['checkpoint_sha256']:
        raise ValueError('Grasp actor must retain its qualified checkpoint hash')
    rows,source_initial=checked_reference(args.reference)
    args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2)
    actor=ConservativeSAC.load(frozen,device='cpu')
    spec=json.loads((ASSETS/'kinematics.json').read_text())
    anchors=json.loads((ASSETS/'finger_reference.json').read_text())['anchors_jaw_local_m']
    entries=[];client=StateClient(args.url);grasp=None
    source_hash=file_hash(args.reference/'steps.jsonl')
    def save(actual,baseline,command,history,seed,split,family,prefix_step):
        i=baseline['task_geometry']['source_index']
        point=opening_center(spec,anchors,actual['measured_joint_deg'],actual['measured_finger_joint_deg']['jaw2_hinge'])
        provenance={'seed':seed,'split':split,'family':family,'move':'e2e4','prefix_step':prefix_step,
            'reference_sha256':source_hash,'grasp_checkpoint_sha256':qualified['checkpoint_sha256'],
            'prior_lift_verified':bool(history['lifted']),'prior_max_lift_m':history['max_lift_m'],
            'prior_transported':bool(history.get('transported',False)),
            'pawn_height_m':float(actual['pieces']['piece_centers_m'][i][2]-baseline['pieces']['piece_centers_m'][i][2])}
        TransportPlaceTask(baseline,actual,point,command,provenance)
        saved=client.save_reset(f'place_s{seed}_{family}',provenance)['reset_context']
        entry={**saved,**provenance};entries.append(entry)
        return entry
    try:
        for split,partition in partitions.items():
            for seed in partition:
                rng=np.random.default_rng(seed)
                first=client.reset({'history':[],'move':'e2e4','pawn_offset_mm':rng.uniform(-.5,.5,2).tolist()})
                baseline=first['observation']
                if baseline['joint_map']!=source_initial['joint_map']:
                    raise ValueError('Reference calibration changed')
                scorer=LegalMoveTask(baseline,TaskConfig(max_steps=600))
                for tick,row in enumerate(rows[:200],1):
                    latest=client.step(row['teacher']['executed_legacy']);actual=latest['observation']
                    _,terminated,truncated,score=scorer.update(actual)
                    if terminated or truncated:
                        raise ValueError(f'Physical prefix failed: {score}')
                    if tick==140:
                        approach={**client.save_reset(f'place_s{seed}_approach',
                            {'family':'open_approach','split':'train','seed':seed,'prefix_step':140})['reset_context'],
                            'family':'open_approach','split':'train','prefix_step':140}
                    if tick in (185,200):
                        save(actual,baseline,row['teacher']['command_joint_deg'],score,seed,split,
                             'near_target' if tick==185 else 'near_release',tick)
                # The third reset family is the real endpoint of the frozen
                # learned grasp, not an idealized/reference grasp pose.
                temporary=args.output/'approach.json'
                temporary.write_text(json.dumps({'entries':[approach]}))
                if grasp is None:
                    grasp=NearGraspEnv(temporary,args.url,lift_config=LiftConfig(**contract['lift_config']))
                grasp.fixed_entry=approach
                obs,initial=grasp.reset(seed=seed)
                trace=args.output/f'grasp_s{seed}.jsonl'
                with trace.open('x') as stream:
                    while True:
                        action,_=actor.predict(obs,deterministic=True)
                        obs,reward,terminated,truncated,info=grasp.step(action)
                        stream.write(json.dumps({'action':action.tolist(),'score':info['score'],
                                                 'observation':info['telemetry']['observation']})+'\n')
                        if terminated or truncated:
                            if not info['score']['success']:
                                raise ValueError(f'Frozen grasp failed on new bank seed {seed}: {info["score"]}')
                            break
                actual=grasp.base.actual
                history={'lifted':True,'max_lift_m':info['score']['height_m'],'transported':False}
                # Synchronize this client's revision after the grasp client.
                client.read()
                entry=save(actual,baseline,grasp.base.command,history,seed,split,'held_source',140)
                entry['grasp_trace_sha256']=file_hash(trace)
                (args.output/f'grasp_s{seed}_initial.json').write_text(json.dumps(initial,indent=2))
                save_status(args.output/'status.json',stage='building',checkpoints=len(entries),seed=seed)
                print('TRANSPORT_RESETS_SAVED',json.dumps({'seed':seed,'split':split,'checkpoints':len(entries)}),flush=True)
        for entry in entries:
            restored=client.restore_reset(entry['name'])['reset_context']
            if restored['checkpoint_sha256']!=entry['sha256'] or max(restored['parity'].values())>1e-5:
                raise ValueError('Transport reset failed physical parity')
            entry['restore_parity']=restored['parity']
        manifest={'format':'transport_place_reset_set_v1','case':{'id':'initial_e2e4','history':[],'move':'e2e4'},
            'reference':str(args.reference.resolve()),'reference_sha256':source_hash,
            'grasp_checkpoint':str(frozen.resolve()),'grasp_checkpoint_sha256':qualified['checkpoint_sha256'],
            'entries':entries,'partitions':partitions,'complete_chess_move':False,
            'note':'All objects physically reached through reference replay or frozen grasp; original board baseline retained.'}
        (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2))
        save_status(args.output/'status.json',stage='ready',checkpoints=len(entries))
        print('TRANSPORT_BANK_READY',str(args.output/'manifest.json'),flush=True)
    except BaseException as error:
        save_status(args.output/'status.json',stage='failed',error=repr(error));raise
    finally:
        if grasp: grasp.close()
        client.close()


if __name__=='__main__':
    main()
