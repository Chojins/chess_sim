"""Recompute diagnostic outcomes, action limits and rewards from saved physics."""
import argparse
import json
from pathlib import Path

import numpy as np

from chess_task import LegalMoveTask,TaskConfig
from review_grasp_geometry import opening_center
from teacher.environment import ASSETS
from teacher.near_grasp import GraspLiftTask,LiftConfig


def audit_trial(path,manifest):
    path=Path(path)
    initial=json.loads((path/'initial.json').read_text())
    context=initial['reset_context'];baseline=context['baseline_observation']
    actual=initial['telemetry']['observation']
    if max(context['parity'].values())>1e-5:
        raise ValueError('Reset physical state did not match its checkpoint')
    config=manifest['action_config'];lift=LiftConfig(**manifest['lift_config'])
    if config['action_mode']!='delta':
        raise ValueError('Unsupported near-grasp action contract')
    task=GraspLiftTask(baseline,actual,lift)
    full=LegalMoveTask(baseline,TaskConfig(max_steps=lift.horizon,capture_home_retry=True))
    specification=json.loads((ASSETS/'kinematics.json').read_text())
    anchors=json.loads((ASSETS/'finger_reference.json').read_text())['anchors_jaw_local_m']
    def point(observation):
        return opening_center(specification,anchors,observation['measured_joint_deg'],
                              observation['measured_finger_joint_deg']['jaw2_hinge'])
    mapping=actual['joint_map'];limits=np.asarray(actual['joint_limits_deg'])
    low,high=limits[:,0]+1e-4,limits[:,1]-1e-4
    command=np.clip(np.asarray(mapping['sim_home_deg'])+np.asarray(mapping['scale'])*(
        np.asarray(actual['target_legacy_action'])-mapping['legacy_home']),low,high)
    velocity=np.zeros(6)
    previous=task.potential(actual,point(actual),command)
    previous_time=actual['physics_time']
    family=context['provenance'].get('family','open_approach')
    rows=[json.loads(s) for s in (path/'steps.jsonl').read_text().splitlines()]
    for index,row in enumerate(rows):
        action=np.asarray(row['action'],np.float32)
        if action.shape!=(6,) or not np.isfinite(action).all() or np.max(np.abs(action))>1:
            raise ValueError('Invalid diagnostic action')
        target=np.clip(command+action*np.asarray(config['max_velocity_deg_s'])/30,low,high)
        np.testing.assert_allclose(target,row['teacher']['command_joint_deg'],atol=1e-5)
        legacy=np.asarray(mapping['legacy_home'])+(target-mapping['sim_home_deg'])/mapping['scale']
        np.testing.assert_allclose(legacy,row['teacher']['executed_legacy'],atol=1e-5)
        actual=row['observation']
        if row['step']!=index+1 or actual['control_step']!=index+1:
            raise ValueError('Missing physical control step')
        np.testing.assert_allclose(actual['physics_time']-previous_time,1/30,atol=1e-6)
        _,full_terminated,_,full_score=full.update(actual)
        terminal_reward,terminated,truncated,score=task.update(actual,point(actual),target)
        if full_terminated and not terminated:
            terminated=True;truncated=False;terminal_reward=-2.
            score.update(success=False,reason='full_task_'+str(full_score['reason']))
        score['reset_family']=family
        if score!=row['score'] or terminated!=row['terminated'] or truncated!=row['truncated']:
            raise ValueError('Saved grasp outcome differs from recomputed physical outcome')
        if bool(terminated or truncated)!=(index==len(rows)-1):
            raise ValueError('Premature or missing grasp terminal transition')
        potential=task.potential(actual,point(actual),target)
        shaping=.5*(config['gamma']*(0 if terminated else potential)-previous)
        next_velocity=(target-command)*30
        accel=(next_velocity-velocity)*30
        accel_cost=config['acceleration_weight']*float(np.mean((accel/180)**2))
        if config['acceleration_cost_cap'] is not None:
            accel_cost=min(accel_cost,config['acceleration_cost_cap'])
        cost=min(.003,accel_cost+config['velocity_weight']*float(np.mean((next_velocity/60)**2)))
        dense=.005*np.exp(-(score['grasp_error_m']/.015)**2)
        np.testing.assert_allclose(row['reward'],terminal_reward+shaping+dense-.003-cost,atol=1e-5)
        if abs(sum(row['reward_components'].values())-row['reward'])>1e-6:
            raise ValueError('Reward components do not sum to the executed reward')
        command,velocity,previous,previous_time=target,next_velocity,potential,actual['physics_time']
    result=json.loads((path/'result.json').read_text())
    if not rows or result['score']!=score:
        raise ValueError('Final grasp report differs from its physical trace')
    return {'path':str(path.resolve()),'steps':len(rows),'success':score['success'],
            'reset_family':family,'reason':score['reason'],'complete_chess_move':False}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run',type=Path)
    args=parser.parse_args()
    manifest=json.loads((args.run/'manifest.json').read_text())
    trials=[]
    for result in sorted(args.run.glob('*/*/result.json')):
        trials.append(audit_trial(result.parent,manifest))
    if not trials:
        raise ValueError('No completed diagnostic trials')
    report={'passed':True,'trials':len(trials),'transitions':sum(r['steps'] for r in trials),
            'objective':'grasp_lift_only','complete_chess_moves_certified':0,'results':trials}
    (args.run/'integrity.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!='results'}))


if __name__=='__main__':
    main()
