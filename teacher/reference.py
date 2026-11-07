"""Audit and synchronously retime a successful ordinary-move reference.

This is a fixed recorded reference, not a learned policy or an RL result. It
creates fresh physical transitions under the state teacher's control interface.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from chess_task import LegalMoveTask, TaskConfig
from teacher.environment import StateTeacherEnv, TeacherConfig

ORDINARY_CASE = {'id': 'initial_e2e4', 'history': [], 'move': 'e2e4'}


def audit_reference(directory):
    directory = Path(directory)
    initial = json.loads((directory/'initial.json').read_text())
    actual = initial['telemetry']['observation']
    geometry = actual['task_geometry']
    if geometry['move'] != 'e2e4' or geometry['capture_index'] is not None or geometry['history']:
        raise ValueError('First ordinary reference must be initial e2e4 without a capture')
    if initial['task_version'] != LegalMoveTask.version or geometry['acting_robot'] != 'blue':
        raise ValueError('Unexpected task version or robot')
    task = LegalMoveTask(actual, TaskConfig(**initial['task_config']))
    trace = directory/'steps.jsonl'
    rows = [json.loads(s) for s in trace.read_text().splitlines()]
    frame, timestamp = initial['telemetry']['frame_id'], initial['telemetry']['simulation_time']
    mapping = actual['joint_map']
    limits = np.asarray(actual['joint_limits_deg'])
    for index, row in enumerate(rows):
        if row['step'] != index+1 or row['frame_id'] != frame+1:
            raise ValueError('Noncontiguous reference command/frame')
        if not np.isclose(row['simulation_time']-timestamp, 1/30, atol=1e-6):
            raise ValueError('Reference lost a physical timestep')
        target = np.asarray(mapping['sim_home_deg'])+np.asarray(mapping['scale'])*(
            np.asarray(row['residual']['executed_legacy'])-mapping['legacy_home'])
        if target.shape != (6,) or not np.isfinite(target).all() or np.any(target<limits[:,0]) or np.any(target>limits[:,1]):
            raise ValueError('Reference command outside physical limits')
        _, terminated, truncated, score = task.update(row['placement_telemetry'])
        if score != row['score'] or terminated != row['terminated'] or truncated != row['truncated']:
            raise ValueError('Reference physics outcome differs from independent scoring')
        if bool(terminated or truncated) != (index==len(rows)-1):
            raise ValueError('Premature or missing reference terminal transition')
        frame, timestamp = row['frame_id'], row['simulation_time']
    if not rows or not score['success']:
        raise ValueError('Reference is not a successful complete move')
    return initial, rows, {'directory':str(directory.resolve()), 'trace_sha256':hashlib.sha256(trace.read_bytes()).hexdigest(),
                          'initial_sha256':hashlib.sha256((directory/'initial.json').read_bytes()).hexdigest(),
                          'frames':len(rows), 'physics_outcome_recomputed':True}


def record_reference(env, source, output, seed):
    initial, rows, provenance = audit_reference(source)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    obs, info = env.reset(seed=seed)
    mapping = initial['telemetry']['observation']['joint_map']
    if mapping != env.mapping:
        raise ValueError('Reference and live teacher calibration differ')
    (output/'initial.json').write_text(json.dumps(info, indent=2))
    previous = env.command.copy()
    observations, actions, targets = [], [], []
    score, stopped = None, False
    with (output/'steps.jsonl').open('x') as stream:
        for index, row in enumerate(rows):
            target = np.asarray(mapping['sim_home_deg'])+np.asarray(mapping['scale'])*(
                np.asarray(row['residual']['executed_legacy'])-mapping['legacy_home'])
            target = np.clip(target, env.low, env.high)
            # Stretch the complete six-joint segment together. Independent
            # clipping against the old clock can close the fingers too early.
            count = max(1,int(np.ceil(np.max(np.abs(target-previous)*30/env.config.max_velocity_deg_s))))
            for fraction in np.linspace(0.,1.,count+1)[1:]:
                goal = previous+(target-previous)*fraction
                action = env.normalized_target(goal)
                observations.append(obs.copy())
                actions.append(action.copy())
                targets.append(goal.copy())
                obs, reward, terminated, truncated, info = env.step(action)
                score = info['score']
                stream.write(json.dumps({'step':len(actions), 'reference_index':index, 'fraction':float(fraction),
                    'action':action.tolist(), 'reward':reward, 'terminated':terminated,'truncated':truncated,
                    'score':score, 'teacher':info['teacher'], 'observation':info['telemetry']['observation']})+'\n')
                if terminated or truncated:
                    stopped = True
                    break
            if stopped:
                break
            previous = target
        # A small tolerance/rate difference may require settling after the last
        # recorded source command. Hold that command, not a scripted task pose.
        for _ in range(60):
            if stopped:
                break
            action=env.normalized_target(target)
            observations.append(obs.copy()); actions.append(action.copy()); targets.append(env.command.copy())
            obs,reward,terminated,truncated,info = env.step(action)
            score=info['score']
            stream.write(json.dumps({'step':len(actions),'reference_index':len(rows),'fraction':1.,
                'action':action.tolist(),'reward':reward,'terminated':terminated,'truncated':truncated,
                'score':score,'teacher':info['teacher'],'observation':info['telemetry']['observation']})+'\n')
            stopped=terminated or truncated
    report = {'mode':'synchronously_retimed_recorded_reference', 'learned_policy':False, 'seed':seed,
              'source':provenance, 'config':env.manifest(), 'frames':len(actions), 'score':score,
              'complete':bool(stopped and score['success']), 'images_recorded':False}
    (output/'result.json').write_text(json.dumps(report,indent=2))
    if report['complete']:
        np.savez_compressed(output/'demonstration.npz', observations=np.asarray(observations,np.float32),
                            actions=np.asarray(actions,np.float32), physical_targets=np.asarray(targets,np.float64))
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seed',type=int,required=True)
    parser.add_argument('--url',default='http://127.0.0.1:8081')
    args=parser.parse_args()
    env=StateTeacherEnv(args.url,case=ORDINARY_CASE,config=TeacherConfig(max_steps=600))
    try:
        result=record_reference(env,args.source,args.output,args.seed)
        print(json.dumps({k:v for k,v in result.items() if k not in ('source','config')}),flush=True)
        if not result['complete']:
            raise SystemExit(1)
    finally:
        env.close()


if __name__=='__main__':
    main()
