"""Recorded-trajectory, privileged capture grasp probe; no RL/model training.

The plan is generated offline from actual USD FK. Only local simulation
endpoints are accessed. Complete captures and deployment are out of scope.
"""
import argparse
import hashlib
import html
import json
from pathlib import Path

import numpy as np

from capture_pickup import CapturePickupTask
from chess_env import ChessEnv
from evaluation_video import EpisodeVideo
from grasp_geometry import GRASP_REFERENCE


def alignment(observation, plan, case):
    """Measured convergence: opening, tool position and arm tracking."""
    joints = np.asarray(observation['measured_joint_deg'])
    target = np.asarray(observation['pieces']['piece_grasp_targets_m'][case['capture_index']])
    distance = float(np.linalg.norm(np.asarray(observation['gripper_grasp_center_m'])-target))
    error = float(np.max(np.abs(joints[:5]-np.asarray(case['fitted_grasp_deg'])[:5])))
    fingers = observation['measured_finger_joint_deg']
    opening = fingers['jaw1_hinge'] >= 8. and fingers['jaw2_hinge'] <= -8.
    upright = observation['pieces']['piece_upright_cosines'][case['capture_index']] > .95
    aligned = distance <= plan['settle_position_tolerance_m'] and error <= plan['settle_joint_tolerance_deg'] and opening and upright
    return bool(aligned), {'tool_target_distance_m':distance,'arm_tracking_max_deg':error,
                          'both_fingers_open':bool(opening),'victim_upright':bool(upright)}


def write_index(output, results):
    body=['<!doctype html><meta charset="utf-8"><title>Capture grasp physics probe</title>',
          '<h1>Capture grasp physics probe</h1>',
          '<p>Privileged recorded-trajectory diagnostic, not a trained policy or complete capture. '
          'Left: highlighted fixed camera; right: moving wrist camera. Playback uses simulation time.</p><ul>']
    for result in results:
        title=f"{result['case_id']} / {result['label']}: {result['score']['reason']}"
        body.append(f'<li><a href="{html.escape(result["video_path"])}">{html.escape(title)}</a></li>')
    body.append('</ul>')
    (output/'videos.html').write_text('\n'.join(body))


def run_episode(env, case, label, plan, output):
    path=output/case['id']/label
    path.mkdir(parents=True,exist_ok=False)
    obs,initial=env.reset(options=case['reset_options'])
    observation=initial['telemetry']['observation']
    if observation.get('grasp_reference')!=GRASP_REFERENCE or 'measured_finger_joint_deg' not in observation:
        raise RuntimeError('Restart the camera server with current grasp and two-finger telemetry')
    if np.max(np.abs(np.array(observation['measured_joint_deg'])-case['expected_initial_joint_deg']))>1.:
        raise RuntimeError('Reset joint pose differs from the recorded source by more than 1 degree')
    target=np.array(observation['pieces']['piece_grasp_targets_m'][case['capture_index']])
    if np.linalg.norm(target-case['target_m'])>.001:
        raise RuntimeError('Victim target changed by more than 1mm; reprepare the offline plan')
    env.task=CapturePickupTask(observation,env.task_config)
    env.task.pickup_limit=480 if label=='centred' else 240
    initial['task_version']=env.task.version
    (path/'initial.json').write_text(json.dumps(initial,indent=2))
    video=EpisodeVideo(path,obs)
    video.write(obs)
    steps=0;done=False;info=None;settled=False;failure=None
    reward_sum=0.
    with (path/'steps.jsonl').open('x') as stream:
        def advance(q,phase):
            nonlocal steps,done,info,obs,reward_sum
            q=np.asarray(q,float)
            legacy=env.legacy_home+(q-env.sim_home)/env.scale
            action=np.clip(legacy,env.action_space.low,env.action_space.high).astype(np.float32)
            # Float32 endpoint rounding is handled by ChessEnv's inward limits;
            # all plan targets were independently checked against USD limits.
            obs,reward,terminated,truncated,info=env.step(action)
            video.write(obs)
            steps+=1;done=terminated or truncated;reward_sum+=reward
            actual=info['telemetry']['observation']
            _,metrics=alignment(actual,plan,case)
            stream.write(json.dumps({'step':steps,'phase':phase,'commanded_joint_deg':q.tolist(),
                                     'action_legacy':action.tolist(),'reward':reward,'terminated':terminated,'truncated':truncated,
                                     'alignment':metrics,'frame_id':info['telemetry']['frame_id'],
                                     'simulation_time':info['telemetry']['simulation_time'],
                                     'score':info['score'],'observation':actual})+'\n')
            return actual
        try:
            if label=='recorded':
                for q in case['recorded_commands_deg']:
                    advance(q,'recorded')
                    if done:break
            else:
                for phase,key in [('prefix','prefix_commands_deg'),('align','approach_deg')]:
                    for q in case[key]:
                        advance(q,phase)
                        if done:break
                    if done:break
                streak=0
                if not done:
                    for _ in range(plan['settle_limit']):
                        actual=advance(case['fitted_grasp_deg'],'settle_open')
                        ready,_=alignment(actual,plan,case)
                        streak=streak+1 if ready else 0
                        if done:break
                        if streak>=plan['settle_frames']:
                            settled=True
                            break
                if not done and not settled:
                    failure='alignment_timeout'
                if not done and settled:
                    for q in case['close_deg']:
                        advance(q,'close')
                        if done:break
                if not done and settled:
                    for _ in range(10):
                        advance(case['close_deg'][-1],'hold_closed')
                        if done:break
                if not done and settled:
                    for q in case['lift_deg']:
                        advance(q,'lift')
                        if done:break
                if not done and settled:
                    for _ in range(plan['raised_hold_limit']):
                        advance(case['raised_hold_deg'],'hold_raised')
                        if done:break
            if not done:
                failure=failure or 'probe_end_without_stable_pickup'
        finally:
            video.close()
    if info is None:
        raise RuntimeError('Probe produced no transitions')
    score=info['score'].copy()
    if failure:
        score.update(success=False,reason=failure)
    result={'case_id':case['id'],'label':label,'steps':steps,'return':reward_sum,'score':score,
            'alignment_settled':settled,'video':video.verify(),
            'video_path':str((path/'cameras.mp4').relative_to(output))}
    if result['video']['frames']!=steps+1:
        raise AssertionError('Video does not match control steps')
    (path/'result.json').write_text(json.dumps(result,indent=2))
    np.savez_compressed(path/'terminal_images.npz',**obs)
    print('PROBE_EPISODE',json.dumps({k:result[k] for k in ['case_id','label','steps','alignment_settled','score']}),flush=True)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    plan_path=args.output/'plan.json'
    plan=json.loads(plan_path.read_text())
    if (args.output/'manifest.json').exists():
        raise FileExistsError('Probe was already started; choose a new output directory')
    for case in plan['cases']:
        source=Path(case['source'])/'steps.jsonl'
        if hashlib.sha256(source.read_bytes()).hexdigest()!=case['source_steps_sha256']:
            raise RuntimeError('Recorded source changed after plan generation')
    manifest={'status':'running','objective':plan['objective'],'privileged':True,
              'episodes':len(plan['cases'])*2,'plan_sha256':hashlib.sha256(plan_path.read_bytes()).hexdigest(),
              'schedule':'recorded then centred for each source; diagnostic paired replay, not held-out policy evaluation',
              'grasp_reference':GRASP_REFERENCE,'rl_training':False}
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    results=[];write_index(args.output,results)
    env=ChessEnv(max_steps=480,pawn_jitter_mm=0.)
    try:
        for case in plan['cases']:
            for label in ['recorded','centred']:
                result=run_episode(env,case,label,plan,args.output)
                results.append(result)
                (args.output/'results.json').write_text(json.dumps(results,indent=2))
                write_index(args.output,results)
        manifest['status']='complete'
        summary={label:{'successes':sum(r['score']['success'] for r in results if r['label']==label),
                        'episodes':sum(r['label']==label for r in results)} for label in ['recorded','centred']}
        (args.output/'summary.json').write_text(json.dumps(summary,indent=2))
        print('PROBE_RESULT',json.dumps(summary),flush=True)
    except Exception as error:
        manifest.update(status='failed',error=str(error))
        raise
    finally:
        (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2))
        env.close()


if __name__=='__main__':
    main()
