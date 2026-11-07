"""Re-score transport placement from raw, saved physics and bounded actions."""
import json
from pathlib import Path
import numpy as np

from chess_task import LegalMoveTask,TaskConfig
from review_grasp_geometry import opening_center
from teacher.environment import ASSETS
from teacher.transport_place import TransportPlaceTask,PlaceConfig


def restore_history(task,actual,history):
    if not history.get('prior_lift_verified') or history.get('prior_max_lift_m',0)<.005:
        raise ValueError('Held state lacks verified physical lift history')
    task.lifted=True;task.departed=True;task.lift_streak=3
    task.max_lift=float(history['prior_max_lift_m'])
    task.transported=bool(history.get('prior_transported',False))
    task.previous_center=np.asarray(actual['pieces']['piece_centers_m'][task.source_index])


def audit_trial(path,manifest):
    path=Path(path);initial=json.loads((path/'initial.json').read_text())
    context=initial['reset_context'];baseline=context['baseline_observation']
    actual=initial['telemetry']['observation'];history=context['provenance']
    if max(context['parity'].values())>1e-5:
        raise ValueError('Transport reset did not match physical checkpoint')
    expected={entry['name']:entry for entry in json.loads(Path(manifest['bank']).read_text())['entries']}
    entry=expected[initial['reset_name']]
    if context['checkpoint_sha256']!=entry['sha256']:
        raise ValueError('Transport trace checkpoint differs from manifest')
    for key in ('family','prior_lift_verified','prior_max_lift_m','prior_transported'):
        if history[key]!=entry[key]: raise ValueError('Held-state history changed')
    config=manifest['action_config'];place=PlaceConfig(**manifest['place_config'])
    if config['action_mode']!='delta': raise ValueError('Need bounded incremental joint control')
    spec=json.loads((ASSETS/'kinematics.json').read_text())
    anchors=json.loads((ASSETS/'finger_reference.json').read_text())['anchors_jaw_local_m']
    def point(o):
        return opening_center(spec,anchors,o['measured_joint_deg'],o['measured_finger_joint_deg']['jaw2_hinge'])
    mapping=actual['joint_map'];limits=np.asarray(actual['joint_limits_deg'])
    low,high=limits[:,0]+1e-4,limits[:,1]-1e-4
    command=np.clip(np.asarray(mapping['sim_home_deg'])+np.asarray(mapping['scale'])*(
        np.asarray(actual['target_legacy_action'])-mapping['legacy_home']),low,high)
    task=TransportPlaceTask(baseline,actual,point(actual),command,history,place)
    full=LegalMoveTask(baseline,TaskConfig(max_steps=place.horizon,capture_home_retry=True))
    restore_history(full,actual,history)
    velocity=np.zeros(6);previous=task.potential(actual,point(actual),command)
    previous_time=actual['physics_time']
    rows=[json.loads(s) for s in (path/'steps.jsonl').read_text().splitlines()]
    for index,row in enumerate(rows):
        action=np.asarray(row['action'],np.float32)
        if action.shape!=(6,) or not np.isfinite(action).all() or np.max(np.abs(action))>1:
            raise ValueError('Invalid placement action')
        target=np.clip(command+action*np.asarray(config['max_velocity_deg_s'])/30,low,high)
        np.testing.assert_allclose(target,row['teacher']['command_joint_deg'],atol=1e-5)
        legacy=np.asarray(mapping['legacy_home'])+(target-mapping['sim_home_deg'])/mapping['scale']
        np.testing.assert_allclose(legacy,row['teacher']['executed_legacy'],atol=1e-5)
        actual=row['observation']
        if row['step']!=index+1 or actual['control_step']!=index+1: raise ValueError('Missing placement control step')
        np.testing.assert_allclose(actual['physics_time']-previous_time,1/30,atol=1e-6)
        _,full_terminated,_,full_score=full.update(actual)
        terminal,terminated,truncated,score=task.update(actual,point(actual),target)
        if full_terminated and not terminated:
            terminated=True;truncated=False;terminal=-2.
            score.update(success=False,reason='full_task_'+str(full_score['reason']))
        score['reset_family']=history['family']
        if score!=row['score'] or terminated!=row['terminated'] or truncated!=row['truncated']:
            raise ValueError('Placement outcome differs from recomputed physics')
        if bool(terminated or truncated)!=(index==len(rows)-1): raise ValueError('Invalid terminal transition')
        potential=task.potential(actual,point(actual),target)
        shaping=.5*(config['gamma']*(0 if terminated else potential)-previous)
        next_velocity=(target-command)*30;acceleration=(next_velocity-velocity)*30
        accel_cost=config['acceleration_weight']*float(np.mean((acceleration/180)**2))
        if config['acceleration_cost_cap'] is not None: accel_cost=min(accel_cost,config['acceleration_cost_cap'])
        cost=min(.003,accel_cost+config['velocity_weight']*float(np.mean((next_velocity/60)**2)))
        dense=.005*np.exp(-(score['target_error_m']/.04)**2)
        np.testing.assert_allclose(row['reward'],terminal+shaping+dense-.003-cost,atol=1e-5)
        np.testing.assert_allclose(sum(row['reward_components'].values()),row['reward'],atol=1e-6)
        command,velocity,previous,previous_time=target,next_velocity,potential,actual['physics_time']
    result=json.loads((path/'result.json').read_text())
    if not rows or result['score']!=score: raise ValueError('Final placement report differs from physical trace')
    return {'path':str(path.resolve()),'steps':len(rows),'success':score['success'],
            'reset_family':history['family'],'reason':score['reason'],'complete_chess_move':False}
