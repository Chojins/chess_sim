"""Audit requested ordinary goals with a frozen actor or labelled analytic aid."""
import argparse
from dataclasses import asdict,replace
import html
import json
from pathlib import Path
import numpy as np
import torch
from camera_client import PairedCameraClient
from evaluation_video import EpisodeVideo
from policy_overlay import highlight_move
from teacher.grasp_checkpoints import ConservativeSAC,file_hash
from teacher.move_teacher import MoveConfig
from teacher.ordinary_moves import ordinary_case,validate_case_geometry,OrdinaryTaskSpaceEnv
from teacher.reactive_oracle import ReactiveOracle,FlowConfig
from teacher.review_policy import matched_images,motion_metrics
from teacher.task_space import CartesianConfig
from teacher.train_grasp import run_trial,save_status
from teacher.transport import StateClient
from teacher.verify_task_space import audit_trial
from teacher.build_ordinary_bank import selected_cases


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,default=Path('outputs/rl-home-qualification-v62'))
    p.add_argument('--checkpoint',default='frozen_actor.zip');p.add_argument('--output',type=Path,required=True)
    p.add_argument('--url',default='http://127.0.0.1:8081');p.add_argument('--bank',type=Path)
    p.add_argument('--moves',nargs='+');p.add_argument('--history',nargs='*',default=[]);p.add_argument('--cases-file',type=Path)
    p.add_argument('--seeds',nargs='+',type=int,default=[120001,120002])
    p.add_argument('--split',choices=['train','validation','test','evaluation'],default='evaluation')
    p.add_argument('--mode',choices=['frozen','oracle'],default='frozen');p.add_argument('--record-video',action='store_true')
    p.add_argument('--tight-approach',action='store_true',help='Analytic aid only: centre above the piece before descending and closing')
    p.add_argument('--grip-pitch',type=float);p.add_argument('--grip-roll',type=float)
    p.add_argument('--flow-config',type=Path,help='Analytic aid only: explicit FlowConfig JSON profile')
    p.add_argument('--configuration-reach',action='store_true',help='Analytic aid only: global grasp IK with joint-space approach/return; dataset interpreter required')
    p.add_argument('--one-per-case',action='store_true',help='Use the first selected reset per goal')
    p.add_argument('--horizon',type=int,help='Shorten a diagnostic probe; completion scoring otherwise remains identical')
    args=p.parse_args()
    cases=selected_cases(args.moves,args.history,args.cases_file)
    if args.bank and not (args.moves or args.history or args.cases_file):
        cases=[ordinary_case(c['history'],c['move']) for c in json.loads(args.bank.read_text())['cases']]
    if len(set(args.seeds))!=len(args.seeds) or min(args.seeds)<0:
        p.error('Need unique goals and nonnegative unique seeds')
    if (args.configuration_reach or args.flow_config or args.tight_approach or args.grip_pitch is not None or args.grip_roll is not None) and args.mode!='oracle':
        p.error('Approach profile applies only to the analytic training aid')
    if args.flow_config and args.tight_approach: p.error('Put approach tolerances inside the explicit flow profile')
    source=json.loads((args.run/'manifest.json').read_text());checkpoint=args.run/args.checkpoint
    if args.horizon is not None:
        if not 1<=args.horizon<=source['move_config']['horizon']: p.error('Diagnostic horizon must shorten the source horizon')
        source={**source,'source_horizon':source['move_config']['horizon'],
            'move_config':{**source['move_config'],'horizon':args.horizon},
            'action_config':{**source['action_config'],'max_steps':args.horizon},'diagnostic_shortened_horizon':True}
    digest=file_hash(checkpoint) if args.mode=='frozen' else None
    args.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2);env=None;client=StateClient(args.url)
    try:
        if args.bank:
            bank_path=args.bank;bank=json.loads(bank_path.read_text())
            entries=[e for e in bank['entries'] if e['split']==args.split and e['case'] in cases]
        else:
            entries=[]
            for case in cases:
                for seed in args.seeds:
                    offset=np.random.default_rng(seed).uniform(-.5,.5,2)
                    latest=client.reset({'history':case['history'],'move':case['move'],'pawn_offset_mm':offset.tolist()})
                    validate_case_geometry(case,latest['observation']['task_geometry'])
                    if latest['observation']['joint_map']!=source['joint_map']: raise ValueError('Calibration differs from frozen source')
                    history={'case':case,'seed':seed,'split':args.split,'family':'home','held_start':False,'move':case['move'],
                        'prior_lift_verified':False,'prior_max_lift_m':0.,'prior_release_verified':False,'release_intent':None,
                        'pawn_offset_mm':offset.tolist()}
                    name=f'{args.output.name[:40]}_{case["move"]}_s{seed}'
                    saved=client.save_reset(name,history)['reset_context'];entries.append({**saved,**history,'horizon':800})
            bank={'format':'ordinary_move_home_bank_v1','case':cases[0],'cases':cases,'entries':entries}
            bank_path=args.output/'reset_bank.json';bank_path.write_text(json.dumps(bank,indent=2))
        if not entries: raise ValueError('No requested ordinary reset entries')
        if args.one_per_case:
            first={}
            for entry in entries:first.setdefault(entry['case']['id'],entry)
            entries=list(first.values())
        manifest={**source,'objective':'ordinary_move','task_version':'ordinary_move_v1','case':cases[0],'cases':cases,
            'bank':str(bank_path.resolve()),'bank_sha256':file_hash(bank_path),
            'source_checkpoint':str(checkpoint.resolve()) if digest else None,'checkpoint_sha256':digest,
            'source_objective':source['objective'],'learned_policy':args.mode=='frozen',
            'teacher_label':'frozen_neural_actor' if args.mode=='frozen' else 'analytic_geometry_oracle',
            'zero_shot_evaluation':args.mode=='frozen','runtime_oracle':args.mode=='oracle',
            'training_enabled':False,'ready_for_act_collection':False,
            'source_hashes':{str(path):file_hash(path) for path in Path('teacher').glob('*.py')}}
        (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2))
        env=OrdinaryTaskSpaceEnv(bank_path,args.url,split=args.split,config=MoveConfig(**source['move_config']),adaptive=False,
            gamma=source['action_config']['gamma'],cartesian_config=CartesianConfig(**source['cartesian_config']),
            robot_contact_costs=source.get('robot_contact_costs'),terminate_on_robot_contact=source.get('terminate_on_robot_contact',False))
        if (env.schema!=source['observation_schema'] or env.base.mapping!=source['joint_map']
                or json.loads(json.dumps(asdict(env.base.config)))!=source['action_config']):
            raise ValueError('Frozen source action/observation contract changed')
        from teacher.physics_contract import physics_signature
        if args.mode=='frozen' and 'physics_settings' in source and source['physics_settings']!=physics_signature(env.base.actual['physics_settings']):
            raise ValueError('Frozen teacher collision/dynamics contract differs from the simulator')
        manifest['physics_settings']=physics_signature(env.base.actual['physics_settings'])
        (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2))
        profile=FlowConfig(funnel_inner_m=.0005,funnel_outer_m=.0025,close_xy_m=.0008,close_z_m=.001) if args.tight_approach else FlowConfig()
        if args.flow_config: profile=FlowConfig(**json.loads(args.flow_config.read_text()))
        if args.grip_pitch is not None: profile=replace(profile,grip_pitch_deg=args.grip_pitch)
        if args.grip_roll is not None: profile=replace(profile,grip_roll_deg=args.grip_roll)
        actor=ConservativeSAC.load(checkpoint,device='cpu') if args.mode=='frozen' else ReactiveOracle(env.cartesian_config,profile)
        if args.configuration_reach:
            from teacher.configuration_oracle import ConfigurationReference
            actor=ConfigurationReference(env.controller,profile)
        if args.mode=='oracle':
            manifest.update(oracle_version=actor.version,oracle_config=asdict(actor.config))
            if args.configuration_reach: manifest['oracle_action_field']='configuration_reach_and_return_with_task_space_transport'
            (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2))
        camera=PairedCameraClient(args.url,timeout=30) if args.record_video else None
        calibration=json.loads(Path('calibration/sim_board.json').read_text())['cameras']['red']
        results=[];directory=args.output/'trials';directory.mkdir()
        for entry in entries:
            path=directory/entry['name'];video=None
            def capture(latest):
                nonlocal video
                images=matched_images(camera,latest)
                fixed=highlight_move(np.ascontiguousarray(images['red'][...,::-1]),calibration,entry['case']['move'])
                pair={'laptop_rgb':np.ascontiguousarray(fixed[...,::-1]),'phone_rgb':images['blue']}
                if video is None: video=EpisodeVideo(path,pair)
                video.write(pair)
            try: result=run_trial(env,actor,entry,entry['seed'],path,capture=capture if camera else None)
            finally:
                if video: video.close()
            result['case']=entry['case']
            if video: result['video']=video.verify()
            rows=[json.loads(s) for s in (path/'steps.jsonl').read_text().splitlines()]
            initial=json.loads((path/'initial.json').read_text())
            result['motion']=motion_metrics(rows,initial['telemetry']['observation'])
            (path/'result.json').write_text(json.dumps(result,indent=2));checked=audit_trial(path,manifest)
            from teacher.robot_clearance import append_clearance_audit
            checked=append_clearance_audit(path,manifest,checked)
            if 'robot_collision_audit' in checked:
                result['robot_collision_audit']=checked['robot_collision_audit']
                (path/'result.json').write_text(json.dumps(result,indent=2))
            if 'camera_clearance_audit' in checked:
                result['camera_clearance_audit']=checked['camera_clearance_audit']
                (path/'result.json').write_text(json.dumps(result,indent=2))
            results.append(result)
            (args.output/'results.json').write_text(json.dumps(results,indent=2))
            counts={c['move']:{'successes':sum(r['score']['success'] for r in results if r['case']==c),
                'trials':sum(r['case']==c for r in results)} for c in cases}
            save_status(args.output/'status.json',stage='ordinary_goal_probe',completed=len(results),requested=len(entries),
                per_move=counts,learned_policy=args.mode=='frozen',training_enabled=False)
            print('ORDINARY_GOAL_TRIAL',json.dumps(result),flush=True)
        if digest and file_hash(checkpoint)!=digest: raise ValueError('Frozen source changed')
        if video is not None:
            body=['<!doctype html><meta charset="utf-8"><title>Ordinary goal probe</title>',
                '<h1>Requested ordinary moves</h1><p>Actual paired simulator cameras, 30 Hz simulated playback. '
                f'Policy: {html.escape(manifest["teacher_label"])}. No gradients or ACT export.</p>']
            for r in results:
                body.append(f'<h2>{html.escape(r["case"]["move"])} — {html.escape(r["score"]["reason"])}, seed {r["seed"]}</h2>'
                    f'<video controls preload="metadata" width="1200" src="trials/{html.escape(r["reset_name"])}/cameras.mp4"></video>')
            (args.output/'videos.html').write_text('\n'.join(body))
        save_status(args.output/'status.json',stage='ordinary_goal_probe_complete',completed=len(results),per_move=counts,
            learned_policy=args.mode=='frozen',training_enabled=False,ready_for_act_collection=False)
    except BaseException as error:
        save_status(args.output/'status.json',stage='failed',error=repr(error));raise
    finally:
        if env: env.close()
        client.close()


if __name__=='__main__': main()
