"""Collect a small local ACT pilot from a qualified frozen neural teacher.

No gradients, exploration, analytic override, remote upload or robot commands.
Only independently audited home-start successes enter the accepted index.
"""
import argparse
import html
import json
from pathlib import Path
import shutil
import numpy as np
import torch
import chess
from camera_client import PairedCameraClient
from evaluation_video import EpisodeVideo
from teacher.episode_data import TeacherFrames,audit_episode,digest
from teacher.grasp_checkpoints import ConservativeSAC
from teacher.goal_spec import save_goal
from teacher.review_policy import make_env,matched_images
from teacher.train_grasp import save_status
from teacher.transport import StateClient


def qualification(source):
    status=json.loads((source/'status.json').read_text());manifest=json.loads((source/'manifest.json').read_text())
    if (not status.get('passed') or status.get('stage')!='home_start_qualification_complete'
            or status['trials']<20 or status['success_fraction']<.9
            or digest(source/'frozen_actor.zip')!=status['checkpoint_sha256']
            or manifest['checkpoint_sha256']!=status['checkpoint_sha256']):
        raise ValueError('Need a passing, immutable >=20-trial home qualification')
    if manifest['objective']=='ordinary_move':
        for case in manifest['cases']:
            row=status.get('per_case',{}).get(case['id'],{})
            if not row.get('passed') or row.get('trials',0)<5:
                raise ValueError('Each ordinary goal needs a passing >=5-trial pilot qualification')
    return status,manifest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--qualification',type=Path,required=True);p.add_argument('--motion-review',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--seeds',type=int,nargs='+',required=True)
    p.add_argument('--url',default='http://127.0.0.1:8082')
    args=p.parse_args();status,source=qualification(args.qualification)
    review=json.loads((args.motion_review/'results.json').read_text())
    if review['checkpoint_sha256']!=status['checkpoint_sha256'] or review['home_start_successes']<1:
        raise ValueError('Need a successful matched-camera home review of this exact actor')
    cases=source.get('cases',[source['case']]) if source['objective']=='ordinary_move' else [source['case']]
    if source['objective']=='ordinary_move':
        reviewed={r['case']['id'] for r in review['results'] if r['score']['success'] and r['reset_family']=='home'}
        if not {c['id'] for c in cases}.issubset(reviewed): raise ValueError('Camera review must cover every ordinary goal')
    used=set(source['qualification_seeds'])
    for key in ('bank','source_development_bank'):
        if key in source:
            used.update(e['seed'] for e in json.loads(Path(source[key]).read_text())['entries'])
    development=json.loads(Path(source['bank']).read_text())
    if development.get('source_development_bank'):
        used.update(e['seed'] for e in json.loads(Path(development['source_development_bank']).read_text())['entries'])
    if not args.seeds or len(set(args.seeds))!=len(args.seeds) or min(args.seeds)<0 or used.intersection(args.seeds):
        raise ValueError('Collection needs fresh, unique seeds')
    args.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
    frozen=args.output/'frozen_actor.zip';shutil.copyfile(args.qualification/'frozen_actor.zip',frozen)
    client=StateClient(args.url);env=None;entries=[];accepted=[];results=[]
    try:
        save_status(args.output/'status.json',stage='building_collection_home_starts')
        for case,seed in ((c,s) for c in cases for s in args.seeds):
            offset=np.random.default_rng(seed).uniform(-source['pawn_jitter_mm'],source['pawn_jitter_mm'],2)
            client.reset({'history':case['history'],'move':case['move'],'pawn_offset_mm':offset.tolist()})
            history={'seed':seed,'split':'train','family':'home','held_start':False,'move':case['move'],'case':case,
                'prior_lift_verified':False,'prior_max_lift_m':0.,'prior_release_verified':False,'release_intent':None,
                'pawn_offset_mm':offset.tolist(),'collection_checkpoint_sha256':status['checkpoint_sha256']}
            saved=client.save_reset(f'collect_{args.output.name}_{case["id"]}_{seed}',history)['reset_context']
            entries.append({**saved,**history,'horizon':source['move_config']['horizon']})
        bank=args.output/'home_reset_bank.json';bank.write_text(json.dumps({'case':source['case'],'cases':cases,'entries':entries},indent=2))
        calibration=json.loads(Path('calibration/sim_board.json').read_text())['cameras']['red']
        manifest={**source,'bank':str(bank.resolve()),'bank_sha256':digest(bank),
            'mode':'frozen_neural_teacher_ACT_pilot','teacher_label':'neural_actor','learned_policy':True,
            'checkpoint':str(frozen.resolve()),'checkpoint_sha256':digest(frozen),
            'qualification':str(args.qualification.resolve()),'qualification_status_sha256':digest(args.qualification/'status.json'),
            'motion_review':str(args.motion_review.resolve()),'motion_review_results_sha256':digest(args.motion_review/'results.json'),
            'runtime_oracle':False,'oracle_interventions':0,'training_enabled':False,'collection_seeds':args.seeds,
            'ready_for_act_collection':True,'collection_readiness_scope':'Listed qualified blue ordinary goals on this setup only',
            'collection_source_hashes':{str(path):digest(path) for path in Path('teacher').glob('*.py')},
            'camera_calibration':calibration,'camera_calibration_sha256':digest('calibration/sim_board.json'),
            'uploaded':False,'rgb_student':False,'scope':'Listed blue ordinary goals only; no broader move generalization claim'}
        (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2))
        env,audit=make_env(manifest,args.url,'train');model=ConservativeSAC.load(frozen,device='cpu')
        camera=PairedCameraClient(args.url,timeout=30)
        for entry in entries:
            directory=args.output/entry['name'];directory.mkdir();env.fixed_entry=entry
            case=entry['case'];board=chess.Board()
            for move in case['history']: board.push_uci(move)
            goal=save_goal(directory,board.fen(),case['move'])
            obs,initial=env.reset(seed=entry['seed']);(directory/'initial.json').write_text(json.dumps(initial,indent=2))
            frames=TeacherFrames(directory/'frames');rows=[]
            save_status(args.output/'status.json',stage='collecting_neural_ACT_pilot',completed=len(results),seed=entry['seed'])
            with (directory/'steps.jsonl').open('x') as stream:
                while True:
                    before=env.base.latest;images=matched_images(camera,before)
                    action,_=model.predict(obs,deterministic=True);obs,reward,done,truncated,info=env.step(action)
                    frames.write(images,before,info,calibration,case['move'])
                    row={'step':len(rows)+1,'action':action.tolist(),'reward':reward,'terminated':done,'truncated':truncated,
                        'score':info['score'],'teacher':info['teacher'],'reward_components':info['reward_components'],
                        'observation':info['telemetry']['observation'],'sequence':info['telemetry']['sequence'],
                        'simulation_time':info['telemetry']['simulation_time']}
                    stream.write(json.dumps(row)+'\n');rows.append(row)
                    if done or truncated: break
            result={'reset_name':entry['name'],'reset_family':'home','seed':entry['seed'],'steps':len(rows),'score':rows[-1]['score'],'case':case}
            (directory/'result.json').write_text(json.dumps(result,indent=2));checked=audit(directory,manifest)
            contact_free=checked.get('robot_collision_audit',{}).get('contact_free',True)
            contact_free&=checked.get('camera_clearance_audit',{}).get('passed',True)
            episode={'format':'chess_sim_neural_teacher_episode_v1','fps':30,'frames':frames.frames,'score':result['score'],
                'accepted_for_act_training':bool(result['score']['success'] and contact_free),'provenance':{
                    'teacher_checkpoint_sha256':digest(frozen),'calibration':source['joint_map'],
                    'qualification_status_sha256':manifest['qualification_status_sha256'],
                    'action_units':'unchanged source legacy joint units; executed physical targets',
                    'image_order':'RGB','laptop':'fixed red camera with legacy move highlight','phone':'blue wrist camera',
                    'policy_mode':'frozen deterministic neural actor; exploration disabled; no analytic intervention',
                    'goal':goal,
                    'uploaded':False}}
            (directory/'episode.json').write_text(json.dumps(episode,indent=2))
            if episode['accepted_for_act_training']: accepted.append(audit_episode(directory))
            # A derivative preview of exactly the saved before-action frames.
            # RGB training data stay lossless; H.264 is only for viewing.
            video=None
            try:
                for frame_path in sorted((directory/'frames').glob('frame_*.npz')):
                    with np.load(frame_path,allow_pickle=False) as frame:
                        pair={k:frame[k].copy() for k in ('laptop_rgb','phone_rgb')}
                    if video is None: video=EpisodeVideo(directory,pair)
                    video.write(pair)
            finally:
                if video: video.close()
            if video: result['video']=video.verify()
            results.append(result)
            (args.output/'accepted_episodes.json').write_text(json.dumps({'format':'chess_sim_neural_teacher_audited_v1',
                'episodes':accepted,'frames':sum(e['frames'] for e in accepted),'uploaded':False},indent=2))
        if digest(frozen)!=status['checkpoint_sha256']: raise ValueError('Frozen teacher changed')
        (args.output/'results.json').write_text(json.dumps(results,indent=2))
        body=['<!doctype html><meta charset="utf-8"><title>Neural teacher ACT pilot</title>',
            '<h1>Neural teacher ACT pilot</h1><p>Frozen qualified neural actor; no exploration or analytic intervention. '
            'Listed qualified ordinary goals only. Left: fixed red view with move highlight. Right: blue wrist. '
            'Preview uses the saved before-action frames at simulated 30 Hz.</p>']
        for result in results:
            body.append(f'<h2>{html.escape(result["case"]["move"])} — seed {result["seed"]}: {html.escape(result["score"]["reason"])}</h2>'
                f'<video controls preload="metadata" width="1200" src="{html.escape(result["reset_name"])}/cameras.mp4"></video>'
                f'<p>Desired board (separate future policy input):</p><img width="320" '
                f'src="{html.escape(result["reset_name"])}/desired_board.svg" alt="Desired chess board">')
        (args.output/'videos.html').write_text('\n'.join(body))
        save_status(args.output/'status.json',stage='neural_ACT_pilot_complete',attempts=len(results),
            accepted_episodes=len(accepted),accepted_frames=sum(e['frames'] for e in accepted),uploaded=False,
            ready_for_act_collection=bool(accepted),scope=manifest['collection_readiness_scope'])
    except BaseException as error:
        save_status(args.output/'status.json',stage='failed',error=repr(error));raise
    finally:
        if env: env.close()
        client.close()


if __name__=='__main__': main()
