"""Rebuild exact privileged RL transitions from independently audited physics.

These are state-teacher data, not camera demonstrations or an ACT export.
"""
import argparse
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace
import gymnasium as gym
import numpy as np

from teacher.environment import ASSETS,TeacherConfig,TeacherFeatures
from teacher.grasp_checkpoints import ConservativeSAC,file_hash
from teacher.move_teacher import MoveEnv,MoveConfig,task_type_for_objective
from teacher.verify_move import audit_trial


def trial_transitions(path,manifest,gamma=None,robot_contact_costs=None,camera_clearance_reward_contract=None):
    path=Path(path)
    if manifest.get('policy_action_space'):
        from teacher.verify_task_space import audit_trial as audit_cartesian
        audit_cartesian(path,manifest)
    else: audit_trial(path,manifest)
    initial=json.loads((path/'initial.json').read_text());actual=initial['telemetry']['observation']
    context=initial['reset_context'];baseline=context['baseline_observation']
    entry=next(r for r in json.loads(Path(manifest['bank']).read_text())['entries'] if r['name']==initial['reset_name'])
    config=TeacherConfig(**manifest['action_config']);move=MoveConfig(**manifest['move_config'])
    local=replace(move,horizon=min(move.horizon,entry.get('horizon',move.horizon)))
    spec=json.loads((ASSETS/'kinematics.json').read_text());anchors=json.loads((ASSETS/'finger_reference.json').read_text())['anchors_jaw_local_m']
    mapping=actual['joint_map'];limits=np.asarray(actual['joint_limits_deg']);low,high=limits[:,0]+1e-4,limits[:,1]-1e-4
    command=np.clip(np.asarray(mapping['sim_home_deg'])+np.asarray(mapping['scale'])*(
        np.asarray(actual['target_legacy_action'])-mapping['legacy_home']),low,high)
    # Use the actual encoder method, without constructing a simulator client.
    if manifest.get('policy_action_space'):
        from teacher.task_space import TaskSpaceMoveEnv,CartesianConfig
        env=TaskSpaceMoveEnv.__new__(TaskSpaceMoveEnv);env.cartesian_config=CartesianConfig(**manifest['cartesian_config'])
    else: env=MoveEnv.__new__(MoveEnv)
    env.move_config=move;env.episode_config=local;env.schema=manifest['observation_schema']
    env.observation_space=gym.spaces.Box(-np.inf,np.inf,(manifest['observation_size'],),np.float32)
    encoder=TeacherFeatures(baseline,spec,anchors,config)
    env.base=SimpleNamespace(actual=actual,encoder=encoder,command=command,kinematics=spec,anchors=anchors)
    env.task=task_type_for_objective(manifest['objective'])(baseline,actual,encoder.point(actual),command,context['provenance'],local)
    env.previous_action=np.zeros(6);obs=env._observation()
    rows=[json.loads(line) for line in (path/'steps.jsonl').read_text().splitlines()]
    target_camera=None;source_camera=None
    if camera_clearance_reward_contract is not None:
        from teacher.robot_clearance import ClearanceReward
        target_camera=ClearanceReward(camera_clearance_reward_contract)
        if manifest.get('camera_clearance_reward_contract') is not None:
            source_camera=ClearanceReward(manifest['camera_clearance_reward_contract'])
    for row in rows:
        before=env.base.actual
        _,_,grasp_error,contact=env.task.measure(before,encoder.point(before),env.base.command)
        center=np.asarray(before['pieces']['piece_centers_m'][env.task.index])
        clear=np.linalg.norm(encoder.point(before)-center)>.035
        fingers=env.task.fingers(before)
        free=bool((not env.task.lifted and fingers[0]>=4 and fingers[1]<=-4 and grasp_error>.012)
                  or (env.task.released and clear))
        auxiliary={'released_before':bool(env.task.released),'home_feedback':bool(env.task.released and clear),
            'free_arm_before':free,'all_centers_m':np.asarray(before['pieces']['piece_centers_m'],np.float32),
            'all_upright':np.asarray(before['pieces']['piece_upright_cosines'],np.float32),
            'source_index':env.task.index,'home_joint_deg':np.asarray(baseline['measured_joint_deg'],np.float32)}
        action=np.asarray(row['action'],np.float32)
        env.base.actual=row['observation'];env.base.command=np.asarray(row['teacher']['command_joint_deg'])
        env.task.update(env.base.actual,encoder.point(env.base.actual),env.base.command)
        env.previous_action=action.copy();next_obs=env._observation()
        reward=float(row['reward'])
        if gamma is not None:
            next_potential=0. if row['terminated'] else env.task.potential(env.base.actual,encoder.point(env.base.actual),env.base.command)
            reward+=.5*(gamma-config.gamma)*next_potential
        if robot_contact_costs is not None:
            from teacher.robot_contacts import contact_costs,collision_cost
            reward+=collision_cost(row['observation'],contact_costs(manifest.get('robot_contact_costs')))-collision_cost(row['observation'],robot_contact_costs)
        if target_camera is not None:
            reward+=(source_camera.evaluate(row['observation'])['cost'] if source_camera else 0.)-target_camera.evaluate(row['observation'])['cost']
        yield {'observation':obs,'next_observation':next_obs,'action':action,'reward':reward,
            'terminated':bool(row['terminated']),'truncated':bool(row['truncated']),
            'reset_name':entry['name'],'split':entry['split'],'seed':entry['seed'],'step':row['step'],
            'robot_self_contact':bool(row['observation'].get('robot_contacts',{}).get('self_contact')),
            'robot_inter_robot_contact':bool(row['observation'].get('robot_contacts',{}).get('inter_robot_contact')),**auxiliary}
        obs=next_obs


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--group',default='reference',help='Directory containing completed trials')
    parser.add_argument('--manifest-name',default='manifest.json',help='Policy manifest filename (flow bank uses policy_manifest.json)')
    parser.add_argument('--training-only',action='store_true',help='Explicitly omit whole held-out episodes before exporting training data')
    parser.add_argument('--output',type=Path,help='Export training-only privileged RL transitions')
    parser.add_argument('--verify-policy',help='Check reconstructed inputs reproduce this frozen actor’s deterministic actions')
    parser.add_argument('--gamma',type=float,help='Recompute only potential shaping for a different discount; physical transitions stay unchanged')
    parser.add_argument('--self-contact-cost',type=float,help='Relabel recorded robot-contact costs for RL replay')
    parser.add_argument('--inter-robot-contact-cost',type=float)
    parser.add_argument('--terminate-on-robot-contact',action='store_true',help='Mark clean reference replay as contact-terminal compatible; rejects any contacting source')
    parser.add_argument('--camera-clearance-reward-meshes',type=Path,help='Recompute camera buffer costs from recorded joints, without changing actions/dynamics')
    parser.add_argument('--camera-clearance-buffer-mm',type=float,default=8.)
    parser.add_argument('--camera-clearance-minimum-mm',type=float,default=5.)
    parser.add_argument('--camera-clearance-weight',type=float,default=.04)
    args=parser.parse_args()
    if not args.output and not args.verify_policy: parser.error('Choose training export or frozen-input verification')
    if args.gamma is not None and not 0<args.gamma<1: parser.error('gamma must lie between zero and one')
    from teacher.robot_contacts import contact_costs
    target_costs=None
    target_camera=None
    if args.camera_clearance_reward_meshes:
        from teacher.robot_clearance import clearance_reward_contract
        target_camera=clearance_reward_contract(args.camera_clearance_reward_meshes,args.camera_clearance_buffer_mm/1000,args.camera_clearance_minimum_mm/1000,args.camera_clearance_weight)
    if args.self_contact_cost is not None or args.inter_robot_contact_cost is not None:
        target_costs=contact_costs({'self':args.self_contact_cost or 0.,'inter_robot':args.inter_robot_contact_cost or 0.})
    if args.output and args.output.exists(): raise ValueError('Refusing to overwrite transition export')
    manifest_path=args.run/args.manifest_name;manifest=json.loads(manifest_path.read_text())
    if manifest['objective'] not in ('unified_e2e4','ordinary_move'): raise ValueError('Only ordinary move state features are supported')
    actor=ConservativeSAC.load(args.run/args.verify_policy,device='cpu') if args.verify_policy else None
    episodes=sorted((args.run/args.group).glob('*/result.json'))
    if args.training_only:
        train_names={r['name'] for r in json.loads(Path(manifest['bank']).read_text())['entries'] if r['split']=='train'}
        episodes=[p for p in episodes if json.loads(p.read_text())['reset_name'] in train_names]
    if not episodes: raise ValueError('No completed trials')
    transitions=[];sources=[];max_error=0.
    for episode_id,result in enumerate(episodes):
        path=result.parent;rows=list(trial_transitions(path,manifest,args.gamma,target_costs,target_camera))
        if args.output and any(r['split']!='train' for r in rows):
            raise ValueError('Validation/final-test data cannot enter a training transition export')
        if actor:
            for row in rows:
                predicted,_=actor.predict(row['observation'],deterministic=True)
                error=float(np.max(np.abs(predicted-row['action'])));max_error=max(max_error,error)
                if error>1e-5: raise ValueError(f'Frozen policy inputs do not reproduce recorded action: {path}, step {row["step"]}, error {error}')
        for row in rows: row['episode_id']=episode_id
        transitions.extend(rows);sources.append({'path':str(path.resolve()),'steps':len(rows),
            'initial_sha256':file_hash(path/'initial.json'),'steps_sha256':file_hash(path/'steps.jsonl'),
            'reset_name':rows[0]['reset_name'],'split':rows[0]['split'],'seed':rows[0]['seed']})
        if manifest['objective']=='ordinary_move':
            sources[-1]['case']=json.loads((path/'initial.json').read_text())['reset_context']['provenance']['case']
    if args.terminate_on_robot_contact and any(r['robot_self_contact'] or r['robot_inter_robot_contact'] for r in transitions):
        raise ValueError('Contacting references cannot enter contact-terminal replay')
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        arrays={key:np.asarray([r[key] for r in transitions]) for key in
            ('observation','next_observation','action','reward','terminated','truncated','episode_id','seed','step',
             'released_before','home_feedback','free_arm_before','all_centers_m','all_upright','source_index','home_joint_deg',
             'robot_self_contact','robot_inter_robot_contact')}
        report={'format':'audited_privileged_rl_transitions_v1','objective':manifest['objective'],
            'manifest':str(manifest_path.resolve()),'manifest_sha256':file_hash(manifest_path),
            'bank_sha256':manifest['bank_sha256'],'action_config':manifest['action_config'],'move_config':manifest['move_config'],
            'observation_schema':manifest['observation_schema'],'sources':sources,'training_only':True,
            'reference_label':manifest.get('teacher_label'),
            'note':'Training references only. No validation/test transitions, RGB, or ACT export.'}
        from teacher.physics_contract import source_physics
        report['physics_settings']=source_physics(manifest,sources)
        if args.terminate_on_robot_contact and not report['physics_settings']['contacts'].get('robot_self_collisions'):
            raise ValueError('Contact-terminal replay requires self-contact physics and reports')
        report['robot_contact_costs']=target_costs if target_costs is not None else contact_costs(manifest.get('robot_contact_costs'))
        report['terminate_on_robot_contact']=bool(args.terminate_on_robot_contact or manifest.get('terminate_on_robot_contact',False))
        report['camera_clearance_reward_contract']=target_camera or manifest.get('camera_clearance_reward_contract')
        if target_camera is not None:
            report['camera_reward_relabeling']={'source':manifest.get('camera_clearance_reward_contract'),'target':target_camera,
                'method':'Exact CAD gaps recomputed from recorded measured joints; physical observations/actions/dynamics unchanged'}
        if target_costs is not None:
            report['collision_reward_relabeling']={'source_costs':contact_costs(manifest.get('robot_contact_costs')),
                'target_costs':target_costs,'method':'Costs recomputed from audited recorded per-step contact reports; actions and dynamics unchanged'}
        if manifest.get('policy_action_space'):
            report.update(policy_action_space=manifest['policy_action_space'],cartesian_config=manifest['cartesian_config'])
        if args.gamma is not None:
            report['action_config']={**manifest['action_config'],'gamma':args.gamma}
            report['reward_relabeling']={'source_gamma':manifest['action_config']['gamma'],'target_gamma':args.gamma,
                'method':'Potential shaping recomputed from audited recorded states; task reward, costs, actions and dynamics unchanged'}
        with args.output.open('xb') as stream: np.savez_compressed(stream,**arrays,metadata_json=np.array(json.dumps(report)))
    print(json.dumps({'passed':True,'episodes':len(sources),'transitions':len(transitions),
        'frozen_action_max_abs_error':max_error if actor else None,'training_export':str(args.output) if args.output else None}))


if __name__=='__main__': main()
