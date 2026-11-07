"""Independent reconstruction of unified-task goals, actions and rewards."""
from dataclasses import replace
import json
from pathlib import Path
import numpy as np
from review_grasp_geometry import opening_center
from teacher.environment import ASSETS
from teacher.move_teacher import MoveTask,MoveConfig


def audit_trial(path,manifest):
    path=Path(path);initial=json.loads((path/'initial.json').read_text())
    context=initial['reset_context'];actual=initial['telemetry']['observation']
    entry=next(r for r in json.loads(Path(manifest['bank']).read_text())['entries'] if r['name']==initial['reset_name'])
    if context['checkpoint_sha256']!=entry['sha256'] or max(context['parity'].values())>1e-5:
        raise ValueError('Move reset integrity failed')
    for key in ('family','held_start','prior_lift_verified','prior_max_lift_m','prior_release_verified','release_intent'):
        if context['provenance'][key]!=entry[key]: raise ValueError('Physical reset history changed')
    config=manifest['action_config'];move=MoveConfig(**manifest['move_config'])
    move=replace(move,horizon=min(move.horizon,entry.get('horizon',move.horizon)))
    spec=json.loads((ASSETS/'kinematics.json').read_text());anchors=json.loads((ASSETS/'finger_reference.json').read_text())['anchors_jaw_local_m']
    def point(o): return opening_center(spec,anchors,o['measured_joint_deg'],o['measured_finger_joint_deg']['jaw2_hinge'])
    mapping=actual['joint_map'];limits=np.asarray(actual['joint_limits_deg']);low,high=limits[:,0]+1e-4,limits[:,1]-1e-4
    command=np.clip(np.array(mapping['sim_home_deg'])+np.array(mapping['scale'])*(np.array(actual['target_legacy_action'])-mapping['legacy_home']),low,high)
    task=MoveTask(context['baseline_observation'],actual,point(actual),command,context['provenance'],move)
    previous=task.potential(actual,point(actual),command);velocity=np.zeros(6);previous_time=actual['physics_time']
    rows=[json.loads(s) for s in (path/'steps.jsonl').read_text().splitlines()]
    for index,row in enumerate(rows):
        action=np.asarray(row['action'],np.float32)
        if action.shape!=(6,) or not np.isfinite(action).all() or np.max(np.abs(action))>1: raise ValueError('Invalid move action')
        target=np.clip(command+action*np.asarray(config['max_velocity_deg_s'])/30,low,high)
        np.testing.assert_allclose(target,row['teacher']['command_joint_deg'],atol=1e-5)
        legacy=np.array(mapping['legacy_home'])+(target-mapping['sim_home_deg'])/mapping['scale']
        np.testing.assert_allclose(legacy,row['teacher']['executed_legacy'],atol=1e-5)
        actual=row['observation']
        if row['step']!=index+1 or actual['control_step']!=index+1: raise ValueError('Missing control step')
        np.testing.assert_allclose(actual['physics_time']-previous_time,1/30,atol=1e-6)
        terminal,terminated,truncated,score=task.update(actual,point(actual),target)
        score['reset_family']=entry['family'];score['home_start']=entry['family']=='home'
        if score!=row['score'] or terminated!=row['terminated'] or truncated!=row['truncated']:
            raise ValueError('Unified outcome differs from raw physics')
        if bool(terminated or truncated)!=(index==len(rows)-1): raise ValueError('Invalid terminal placement')
        potential=task.potential(actual,point(actual),target)
        shaping=.5*(config['gamma']*(0 if terminated else potential)-previous)
        next_velocity=(target-command)*30;accel=(next_velocity-velocity)*30
        accel_cost=config['acceleration_weight']*float(np.mean((accel/180)**2))
        if config['acceleration_cost_cap'] is not None: accel_cost=min(accel_cost,config['acceleration_cost_cap'])
        cost=min(.003,accel_cost+config['velocity_weight']*float(np.mean((next_velocity/60)**2)))
        np.testing.assert_allclose(row['reward'],terminal+shaping-.003-cost,atol=1e-5)
        np.testing.assert_allclose(row['reward'],sum(row['reward_components'].values()),atol=1e-6)
        command,velocity,previous,previous_time=target,next_velocity,potential,actual['physics_time']
    result=json.loads((path/'result.json').read_text())
    if not rows or result['score']!=score: raise ValueError('Move summary differs from physics trace')
    return {'path':str(path.resolve()),'steps':len(rows),'success':score['success'],'reset_family':entry['family'],
        'reason':score['reason'],'home_start':entry['family']=='home','complete_chess_move':score['complete_chess_move']}
