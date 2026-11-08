"""Frozen-policy qualification on newly reached home starts, without selection."""
import argparse
import json
from pathlib import Path
import shutil
import numpy as np
import torch

from teacher.grasp_checkpoints import ConservativeSAC,file_hash
from teacher.review_policy import make_env
from teacher.train_grasp import run_trial,save_status
from teacher.transport import StateClient


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True);parser.add_argument('--checkpoint',default='move_best.zip')
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--url',default='http://127.0.0.1:8082')
    parser.add_argument('--seeds',type=int,nargs='+',default=list(range(90001,90021)))
    parser.add_argument('--minimum-trials',type=int,default=20);parser.add_argument('--minimum-success-fraction',type=float,default=.9)
    parser.add_argument('--minimum-per-case',type=int,help='Multi-goal pilot minimum per move; defaults to 5 (single-goal uses minimum-trials)')
    parser.add_argument('--pawn-jitter-mm',type=float,default=.5)
    parser.add_argument('--camera-clearance-meshes',type=Path,help='Add a predeclared sampled camera/servo CAD margin gate')
    parser.add_argument('--minimum-camera-clearance-mm',type=float,default=5.)
    args=parser.parse_args()
    if (len(set(args.seeds))!=len(args.seeds) or not args.seeds or min(args.seeds)<0 or args.minimum_trials<1
            or not 0<args.minimum_success_fraction<=1 or not np.isfinite(args.pawn_jitter_mm) or not 0<=args.pawn_jitter_mm<=2):
        parser.error('Invalid qualification seeds, thresholds or jitter')
    source=json.loads((args.run/'manifest.json').read_text())
    if source['objective'] not in ('unified_e2e4','ordinary_move'): raise ValueError('Only ordinary-move policies can qualify here')
    cases=source.get('cases',[source['case']]) if source['objective']=='ordinary_move' else [source['case']]
    minimum_case=args.minimum_per_case if args.minimum_per_case is not None else (5 if len(cases)>1 else args.minimum_trials)
    if minimum_case<1: parser.error('minimum-per-case must be positive')
    if not np.isfinite(args.minimum_camera_clearance_mm) or args.minimum_camera_clearance_mm<0:parser.error('Invalid camera clearance margin')
    old_bank=json.loads(Path(source['bank']).read_text())
    if set(args.seeds)&{r['seed'] for r in old_bank['entries']}: raise ValueError('Qualification seeds overlap development resets')
    checkpoint=args.run/args.checkpoint;actor_hash=file_hash(checkpoint)
    args.output.mkdir(parents=True,exist_ok=False);frozen=args.output/'frozen_actor.zip';shutil.copyfile(checkpoint,frozen)
    if file_hash(frozen)!=actor_hash: raise ValueError('Checkpoint changed during freezing')
    client=StateClient(args.url);env=None;torch.set_num_threads(2);entries=[]
    try:
        save_status(args.output/'status.json',stage='building_fresh_home_starts',training_enabled=False)
        for case,seed in ((c,s) for c in cases for s in args.seeds):
            offset=np.random.default_rng(seed).uniform(-args.pawn_jitter_mm,args.pawn_jitter_mm,2)
            latest=client.reset({'history':case['history'],'move':case['move'],'pawn_offset_mm':offset.tolist()})
            actual=latest['observation']
            if actual['joint_map']!=source['joint_map']: raise ValueError('Qualification calibration differs')
            history={'seed':seed,'split':'test','family':'home','held_start':False,'move':case['move'],'case':case,
                'prior_lift_verified':False,'prior_max_lift_m':0.,'prior_release_verified':False,'release_intent':None,
                'checkpoint_sha256':actor_hash,'pawn_offset_mm':offset.tolist()}
            prefix='q'+file_hash(frozen)[:8]+'_'+args.output.name[-16:].replace('-','_')
            saved=client.save_reset(f'{prefix}_{case["id"]}_{seed}',history)['reset_context']
            context=client.restore_reset(saved['name'])['reset_context']
            if context['checkpoint_sha256']!=saved['sha256'] or max(context['parity'].values())>1e-5:
                raise ValueError('Fresh home reset failed parity')
            entries.append({**saved,**history,'horizon':source['move_config']['horizon'],'restore_parity':context['parity']})
        bank_path=args.output/'home_reset_bank.json'
        bank={'format':'fresh_home_qualification_v1','case':source['case'],'cases':cases,'entries':entries,
            'source_development_bank':source['bank'],'checkpoint_sha256':actor_hash}
        bank_path.write_text(json.dumps(bank,indent=2))
        manifest={**source,'bank':str(bank_path.resolve()),'bank_sha256':file_hash(bank_path),
            'source_run':str(args.run.resolve()),'checkpoint':str(frozen.resolve()),'checkpoint_sha256':actor_hash,
            'source_checkpoint':str(checkpoint.resolve()),
            'mode':'frozen_home_start_qualification','training_enabled':False,'checkpoint_selection_enabled':False,
            'minimum_trials':args.minimum_trials,'minimum_success_fraction':args.minimum_success_fraction,
            'minimum_per_case':minimum_case,
            'qualification_seeds':args.seeds,'pawn_jitter_mm':args.pawn_jitter_mm,
            'scope':('Blue ordinary e2e4 from home on this board/setup only; no captures, other moves or hardware transfer claim.'
                if source['objective']=='unified_e2e4' else 'Listed blue ordinary goals on this setup only; no captures, red control or hardware transfer claim.')}
        from teacher.physics_contract import physics_signature
        manifest['physics_settings']=physics_signature(latest['observation']['physics_settings'])
        if args.camera_clearance_meshes:
            manifest['camera_clearance_contract']={'meshes':str(args.camera_clearance_meshes.resolve()),
                'sha256':file_hash(args.camera_clearance_meshes),'minimum_m':args.minimum_camera_clearance_mm/1000}
        (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2))
        env,audit=make_env(manifest,args.url,'test');actor=ConservativeSAC.load(frozen,device='cpu')
        if actor.observation_space.shape!=env.observation_space.shape: raise ValueError('Qualified actor layout changed')
        save_status(args.output/'status.json',stage='frozen_home_start_evaluation',trials=len(entries),training_enabled=False)
        directory=args.output/'evaluation';directory.mkdir();rows=[]
        for index,entry in enumerate(entries):
            row=run_trial(env,actor,entry,95001+index,directory/entry['name']);rows.append(row)
            save_status(args.output/'status.json',stage='frozen_home_start_evaluation',trials=len(entries),
                completed_trials=len(rows),provisional_successes=sum(r['score']['success'] for r in rows),
                training_enabled=False,checkpoint_sha256=actor_hash)
            print('HOME_QUALIFICATION_TRIAL',json.dumps(row),flush=True)
        (directory/'results.json').write_text(json.dumps(rows,indent=2))
        checks=[audit(args.output/'evaluation'/r['name'],manifest) for r in entries]
        if not all(c['home_start'] for c in checks) or [r['score']['success'] for r in rows]!=[c['success'] for c in checks]:
            raise ValueError('Qualification outcomes did not reproduce from raw physics')
        if file_hash(frozen)!=actor_hash: raise ValueError('Frozen actor changed during qualification')
        successes=sum(c['success'] for c in checks);rate=successes/len(checks)
        per_case={}
        for case in cases:
            selected=[r for r in rows if r.get('case',source['case'])==case]
            count=sum(r['score']['success'] for r in selected);fraction=count/len(selected) if selected else 0.
            per_case[case['id']]={'case':case,'trials':len(selected),'successes':count,'success_fraction':fraction,
                'passed':len(selected)>=minimum_case and fraction>=args.minimum_success_fraction}
        unsafe=[c['path'] for c in checks if c.get('robot_collision_audit',{}).get('contact_free') is False]
        too_close=[c['path'] for c in checks if c.get('camera_clearance_audit',{}).get('passed') is False]
        report={'stage':'home_start_qualification_complete','passed':len(checks)>=args.minimum_trials and rate>=args.minimum_success_fraction and all(c['passed'] for c in per_case.values()) and not unsafe and not too_close,
            'successes':successes,'trials':len(checks),'success_fraction':rate,'audited_transitions':sum(c['steps'] for c in checks),
            'per_case':per_case,'minimum_per_case':minimum_case,
            'robot_collision_audit':{'enabled':any('robot_collision_audit' in c for c in checks),
                'unsafe_trials':unsafe,'note':'Self-disabled historical physics cannot establish self-clearance'},
            'camera_clearance_audit':{'enabled':any('camera_clearance_audit' in c for c in checks),'unsafe_trials':too_close,
                'minimum_observed_m':min((c['camera_clearance_audit']['minimum_gap_m'] for c in checks if 'camera_clearance_audit' in c),default=None),
                'minimum_required_m':manifest.get('camera_clearance_contract',{}).get('minimum_m')},
            'checkpoint_sha256':actor_hash,'source_checkpoint_current_sha256':file_hash(checkpoint),
            'training_enabled':False,'checkpoint_selection_enabled':False,'ready_for_act_collection':False,
            'next':'Review complete camera motion and qualify collection/export separately',
            'scope':manifest['scope']}
        (args.output/'integrity.json').write_text(json.dumps({'passed':True,'trials':len(checks),'results':checks},indent=2))
        save_status(args.output/'status.json',**report);print('HOME_START_QUALIFICATION_COMPLETE',json.dumps(report),flush=True)
    except BaseException as error:
        save_status(args.output/'status.json',stage='failed',error=repr(error),training_enabled=False);raise
    finally:
        if env: env.close()
        client.close()


if __name__=='__main__': main()
