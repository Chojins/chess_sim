"""Create disjoint near-grasp reset sets by physically replaying a checked prefix."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from chess_task import LegalMoveTask,TaskConfig
from review_grasp_geometry import opening_center
from teacher.environment import ASSETS
from teacher.transport import StateClient


def checked_reference(path):
    path=Path(path)
    report=json.loads((path/'result.json').read_text())
    initial=json.loads((path/'initial.json').read_text())['telemetry']['observation']
    if (not report['complete'] or initial['task_geometry']['move']!='e2e4'
            or initial['task_geometry']['history'] or initial['task_geometry']['capture_index'] is not None):
        raise ValueError('Need a complete ordinary e2e4 physical reference')
    task=LegalMoveTask(initial,TaskConfig(max_steps=report['config']['config']['max_steps'],capture_home_retry=True))
    rows=[json.loads(s) for s in (path/'steps.jsonl').read_text().splitlines()]
    last_time=initial['physics_time']
    for index,row in enumerate(rows):
        if row['step']!=index+1:
            raise ValueError('Reference lost a physical step')
        np.testing.assert_allclose(row['observation']['physics_time']-last_time,1/30,atol=1e-6)
        _,terminated,truncated,score=task.update(row['observation'])
        if score!=row['score'] or bool(terminated or truncated)!=(index==len(rows)-1):
            raise ValueError('Reference outcome does not independently reproduce')
        if terminated!=row['terminated'] or truncated!=row['truncated']:
            raise ValueError('Reference terminal flags differ')
        last_time=row['observation']['physics_time']
    if score!=report['score'] or not score['success']:
        raise ValueError('Reference is not a complete success')
    return rows,initial


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference',type=Path,default=Path('outputs/e2e4-teacher-bootstrap-v31/reference_0'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--url',default='http://127.0.0.1:8081')
    parser.add_argument('--train-seeds',type=int,nargs='+',default=[50001,50002,50003,50004])
    parser.add_argument('--eval-seeds',type=int,nargs='+',default=[51001,51002])
    parser.add_argument('--prefix-steps',type=int,nargs='+',default=[120,130,140])
    parser.add_argument('--allow-contact-starts',action='store_true',help='Also accept naturally closed, obstructed fingers before any pawn lift')
    parser.add_argument('--evaluation-only',action='store_true',help='Build only the requested evaluation seeds; no training partition')
    args=parser.parse_args()
    if args.evaluation_only:
        args.train_seeds=[]
    seeds=args.train_seeds+args.eval_seeds
    if (len(set(seeds))!=len(seeds) or not seeds or min(seeds)<0 or not args.prefix_steps
            or min(args.prefix_steps)<1 or len(set(args.prefix_steps))!=len(args.prefix_steps)):
        parser.error('Need unique disjoint nonnegative seeds and positive unique prefix steps')
    rows,source_initial=checked_reference(args.reference)
    if max(args.prefix_steps)>=len(rows):
        parser.error('Reset prefixes must precede the complete reference endpoint')
    args.output.mkdir(parents=True,exist_ok=False)
    spec=json.loads((ASSETS/'kinematics.json').read_text())
    anchors=json.loads((ASSETS/'finger_reference.json').read_text())['anchors_jaw_local_m']
    source_digest=hashlib.sha256((args.reference/'steps.jsonl').read_bytes()).hexdigest()
    rng=np.random.default_rng()
    entries=[]
    client=StateClient(args.url)
    try:
        for seed in seeds:
            rng=np.random.default_rng(seed)
            first=client.reset({'history':[],'move':'e2e4','pawn_offset_mm':rng.uniform(-.5,.5,2).tolist()})
            baseline=first['observation'];index=baseline['task_geometry']['source_index']
            if baseline['joint_map']!=source_initial['joint_map']:
                raise ValueError('Reset bank calibration differs from reference')
            scorer=LegalMoveTask(baseline,TaskConfig(max_steps=600))
            split='train' if seed in args.train_seeds else 'evaluation'
            previous_fingers=None
            for tick,row in enumerate(rows[:max(args.prefix_steps)],1):
                latest=client.step(row['teacher']['executed_legacy'])
                actual=latest['observation']
                finger_values=np.array([actual['measured_finger_joint_deg'][key] for key in ('jaw1_hinge','jaw2_hinge')])
                settled=previous_fingers is not None and np.max(np.abs(finger_values-previous_fingers))<=.2
                previous_fingers=finger_values
                _,terminated,truncated,score=scorer.update(actual)
                if terminated or truncated:
                    raise RuntimeError(f'Reference prefix failed before reset save: {score}')
                if tick not in args.prefix_steps:
                    continue
                point=opening_center(spec,anchors,actual['measured_joint_deg'],actual['measured_finger_joint_deg']['jaw2_hinge'])
                error=float(np.linalg.norm(point-actual['pieces']['piece_grasp_targets_m'][index]))
                height=actual['pieces']['piece_centers_m'][index][2]-baseline['pieces']['piece_centers_m'][index][2]
                fingers=actual['measured_finger_joint_deg']
                target_jaw=row['teacher']['command_joint_deg'][5]
                open_start=fingers['jaw1_hinge']>=4 and fingers['jaw2_hinge']<=-4
                contact_start=(args.allow_contact_starts and target_jaw<0 and settled
                    and fingers['jaw1_hinge']>target_jaw+.8 and -fingers['jaw2_hinge']>target_jaw+.8)
                if (error>.015 or abs(height)>.002 or actual['pieces']['piece_upright_cosines'][index]<.98
                        or not (open_start or contact_start)):
                    raise RuntimeError(f'Not an upright, pre-lift near-grasp reset: seed={seed} tick={tick} error={error}')
                name=f'e2e4_s{seed}_step{tick}'
                provenance={'split':split,'seed':seed,'prefix_step':tick,'reference_sha256':source_digest,
                            'move':'e2e4','grasp_error_m':error,'pawn_height_m':float(height),
                            'family':'open_approach' if open_start else 'closed_contact'}
                saved=client.save_reset(name,provenance)['reset_context']
                entries.append({**saved,**provenance})
            (args.output/'status.json').write_text(json.dumps({'stage':'building','completed_seeds':len({r['seed'] for r in entries}),
                                                              'checkpoints':len(entries)}))
            print('RESET_PREFIX_SAVED',json.dumps({'seed':seed,'split':split,'checkpoints':len(args.prefix_steps)}),flush=True)
        # Restore every entry and check pose/velocity parity before accepting the bank.
        for entry in entries:
            restored=client.restore_reset(entry['name'])
            context=restored['reset_context']
            if context['checkpoint_sha256']!=entry['sha256']:
                raise RuntimeError('Reset checkpoint changed')
            entry['restore_parity']=context['parity']
        manifest={'format':'near_grasp_reset_set_v1','move':'e2e4','case':{'id':'initial_e2e4','history':[],'move':'e2e4'},
                  'reference':str(args.reference.resolve()),'reference_sha256':source_digest,'entries':entries,
                  'train_seeds':args.train_seeds,'evaluation_seeds':args.eval_seeds,
                  'note':'Physical states reached through reference replay; no lifted/attached object initialization',
                  'limitations':'Pose/velocity parity is checked. PhysX contact caches are rebuilt, not serialized.'}
        (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2))
        (args.output/'status.json').write_text(json.dumps({'stage':'ready','checkpoints':len(entries),
            'train_checkpoints':sum(r['split']=='train' for r in entries),'evaluation_checkpoints':sum(r['split']=='evaluation' for r in entries)}))
        print('RESET_BANK_READY',str(args.output/'manifest.json'),flush=True)
    finally:
        client.close()


if __name__=='__main__':
    main()
