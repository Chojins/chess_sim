"""Record a frozen policy with matched state/RGB pairs; no learning or ACT export."""
import argparse
from dataclasses import asdict
import html
import json
from pathlib import Path
import numpy as np
import torch

from camera_client import PairedCameraClient
from evaluation_video import EpisodeVideo
from policy_overlay import highlight_move
from teacher.grasp_checkpoints import ConservativeSAC,file_hash


def make_env(manifest,url,split):
    objective=manifest['objective']
    if objective=='unified_e2e4':
        from teacher.move_teacher import MoveEnv,MoveConfig
        if manifest.get('policy_action_space'):
            from teacher.task_space import TaskSpaceMoveEnv,CartesianConfig
            from teacher.verify_task_space import audit_trial
            env=TaskSpaceMoveEnv(manifest['bank'],url,split=split,config=MoveConfig(**manifest['move_config']),adaptive=False,
                gamma=manifest['action_config']['gamma'],cartesian_config=CartesianConfig(**manifest['cartesian_config']))
        else:
            from teacher.verify_move import audit_trial
            env=MoveEnv(manifest['bank'],url,split=split,config=MoveConfig(**manifest['move_config']),adaptive=False,
                gamma=manifest['action_config']['gamma'])
    elif objective=='transport_place_only':
        from teacher.transport_place import TransportPlaceEnv,PlaceConfig
        from teacher.verify_transport import audit_trial
        env=TransportPlaceEnv(manifest['bank'],url,split=split,config=PlaceConfig(**manifest['place_config']))
    elif objective=='grasp_lift_only':
        from teacher.near_grasp import NearGraspEnv,LiftConfig
        from teacher.verify_grasp import audit_trial
        env=NearGraspEnv(manifest['bank'],url,split=split,lift_config=LiftConfig(**manifest['lift_config']))
    else: raise ValueError('Unsupported frozen policy objective')
    if json.loads(json.dumps(asdict(env.base.config)))!=manifest['action_config']:
        raise ValueError('Review action contract differs from training')
    if env.base.mapping!=manifest['joint_map'] or env.schema!=manifest['observation_schema']:
        raise ValueError('Review calibration or observation layout changed')
    return env,audit_trial


def matched_images(camera,latest):
    metadata,images=camera.read(color_order='rgb')
    if (latest.get('camera_render_products')!=2 or metadata['frame_id']!=latest['sequence']
            or abs(metadata['simulation_time']-latest['simulation_time'])>1e-6
            or metadata.get('observation')!=latest['observation']):
        raise RuntimeError('Review RGB pair does not match the executed control state')
    if set(images)!={'red','blue'}: raise ValueError('Need both authored wrist cameras')
    return images


def motion_metrics(rows,initial=None):
    if not rows: return {}
    actual=np.asarray([r['observation']['measured_joint_deg'] for r in rows])
    command=np.asarray([r['teacher']['command_joint_deg'] for r in rows])
    if initial is not None:
        mapping=initial['joint_map']
        first=np.asarray(mapping['sim_home_deg'])+np.asarray(mapping['scale'])*(np.asarray(initial['target_legacy_action'])-mapping['legacy_home'])
        actual=np.vstack([initial['measured_joint_deg'],actual]);command=np.vstack([first,command])
    velocity=np.diff(actual[:,:5],axis=0)*30
    acceleration=np.diff(velocity,axis=0)*30
    return {'arm_path_deg':np.abs(np.diff(actual[:,:5],axis=0)).sum(axis=0).tolist(),
        'max_measured_arm_speed_deg_s':float(np.max(np.abs(velocity))) if len(velocity) else 0.,
        'rms_measured_arm_acceleration_deg_s2':float(np.sqrt(np.mean(acceleration**2))) if len(acceleration) else 0.,
        'max_arm_target_speed_deg_s':float(np.max(np.abs(np.diff(command[:,:5],axis=0)*30))) if len(command)>1 else 0.,
        'note':'Diagnostics only; these values do not certify motion quality or ACT collection readiness.'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True);parser.add_argument('--checkpoint',default='move_best.zip')
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--url',default='http://127.0.0.1:8082')
    parser.add_argument('--split',default='test',choices=['train','validation','test','evaluation'])
    parser.add_argument('--families',nargs='+');parser.add_argument('--maximum-trials',type=int,default=0)
    args=parser.parse_args()
    if args.maximum_trials<0: parser.error('maximum-trials must be nonnegative')
    args.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2);env=None
    source=json.loads((args.run/'manifest.json').read_text());checkpoint=args.run/args.checkpoint;digest=file_hash(checkpoint)
    try:
        env,audit=make_env(source,args.url,args.split);model=ConservativeSAC.load(checkpoint,device='cpu')
        if model.observation_space.shape!=env.observation_space.shape: raise ValueError('Actor input size changed')
        entries=[r for r in env.entries if not args.families or r['family'] in args.families]
        if args.families and set(args.families)!={r['family'] for r in entries}: raise ValueError('Requested family missing')
        if args.maximum_trials: entries=entries[:args.maximum_trials]
        if not entries: raise ValueError('No requested review trials')
        manifest={**source,'review_source_run':str(args.run.resolve()),'checkpoint':str(checkpoint.resolve()),
            'checkpoint_sha256':digest,'mode':'frozen_privileged_policy_camera_review','rgb_student':False,
            'training_enabled':False,'checkpoint_selection_enabled':False,'accepted_for_act_training':False,
            'recorded_resets':[r['name'] for r in entries]}
        (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2))
        calibration=json.loads(Path('calibration/sim_board.json').read_text())['cameras']['red']
        camera=PairedCameraClient(args.url,timeout=30);results=[]
        def pair(latest):
            images=matched_images(camera,latest)
            fixed=highlight_move(np.ascontiguousarray(images['red'][...,::-1]),calibration,source['case']['move'])
            return {'laptop_rgb':np.ascontiguousarray(fixed[...,::-1]),'phone_rgb':images['blue']}
        for entry in entries:
            directory=args.output/entry['name'];directory.mkdir();env.fixed_entry=entry
            obs,initial=env.reset(seed=entry['seed']);(directory/'initial.json').write_text(json.dumps(initial,indent=2))
            images=pair(env.base.latest);video=EpisodeVideo(directory,images);video.write(images);rows=[]
            try:
                with (directory/'steps.jsonl').open('x') as stream:
                    while True:
                        action,_=model.predict(obs,deterministic=True)
                        obs,reward,done,truncated,info=env.step(action)
                        row={'step':len(rows)+1,'action':action.tolist(),'reward':reward,'terminated':done,'truncated':truncated,
                            'score':info['score'],'teacher':info['teacher'],'reward_components':info['reward_components'],
                            'observation':info['telemetry']['observation']}
                        stream.write(json.dumps(row)+'\n');rows.append(row);video.write(pair(env.base.latest))
                        if done or truncated: break
            finally: video.close()
            result={'reset_name':entry['name'],'reset_family':entry['family'],'seed':entry['seed'],'steps':len(rows),
                'score':rows[-1]['score'],'video':video.verify(),'motion':motion_metrics(rows,initial['telemetry']['observation']),
                'scope':'home-start' if entry['family']=='home' else 'intermediate-start continuation',
                'collection':{'accepted_for_act_training':False}}
            (directory/'result.json').write_text(json.dumps(result,indent=2));audit(directory,manifest);results.append(result)
            print('FROZEN_CAMERA_TRIAL',json.dumps(result),flush=True)
        if file_hash(checkpoint)!=digest: raise ValueError('Frozen checkpoint changed while recording')
        body=['<!doctype html><meta charset="utf-8"><title>Frozen teacher review</title>',
            '<h1>Frozen teacher camera review</h1><p>Actual synchronized simulator cameras. Left: fixed red view with move highlight; '
            'right: blue wrist. Playback is simulated 30 Hz. Policy inputs are simulator state. No learning, intervention, or ACT export.</p>']
        for r in results:
            body.append(f'<h2>{html.escape(r["reset_family"])} — {html.escape(r["score"]["reason"])}</h2>'
                f'<p>{html.escape(r["scope"])}, seed {r["seed"]}</p><video controls preload="metadata" width="1200" '
                f'src="{html.escape(r["reset_name"])}/cameras.mp4"></video>')
        (args.output/'videos.html').write_text('\n'.join(body))
        report={'trials':len(results),'goal_completions':sum(r['score']['success'] for r in results),
            'home_start_successes':sum(r['score']['success'] for r in results if r['reset_family']=='home'),
            'home_start_trials':sum(r['reset_family']=='home' for r in results),'checkpoint_sha256':digest,
            'ready_for_act_collection':False,'results':results}
        (args.output/'results.json').write_text(json.dumps(report,indent=2));print('FROZEN_CAMERA_REVIEW_COMPLETE',len(results),flush=True)
    finally:
        if env: env.close()


if __name__=='__main__': main()
