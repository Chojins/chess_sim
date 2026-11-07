"""Short-horizon grasp/lift curriculum with explicit e2e4 goals.

These successes are pickup diagnostics, never complete chess moves or ACT data.
"""
from dataclasses import dataclass
import json
from pathlib import Path

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from chess_task import LegalMoveTask,TaskConfig
from teacher.environment import StateTeacherEnv,TeacherConfig
from teacher.reference import ORDINARY_CASE


@dataclass(frozen=True)
class LiftConfig:
    horizon:int=128
    lift_m:float=.010
    hold_steps:int=10
    max_grasp_error_m:float=.012
    max_relative_slip_m:float=.003
    max_height_m:float=.080

    def __post_init__(self):
        if (type(self.horizon) is not int or self.horizon<1 or type(self.hold_steps) is not int or self.hold_steps<1
                or not all(np.isfinite(v) and v>0 for v in [self.lift_m,self.max_grasp_error_m,
                                                          self.max_relative_slip_m,self.max_height_m])
                or self.max_height_m<=self.lift_m):
            raise ValueError('Invalid lift diagnostic configuration')


class GraspLiftTask:
    version='near_grasp_lift_v1'

    def __init__(self,baseline,reset_observation,config=None):
        self.config=config or LiftConfig()
        if baseline['task_geometry']['move']!='e2e4' or baseline['task_geometry']['capture_index'] is not None:
            raise ValueError('Initial near-grasp curriculum is ordinary e2e4 only')
        self.index=baseline['task_geometry']['source_index']
        self.original=np.array(baseline['pieces']['piece_centers_m'][self.index])
        self.monitor=LegalMoveTask(baseline,TaskConfig(max_steps=self.config.horizon))
        self.steps=self.streak=0
        self.relative_anchor=None
        self.previous_fingers=np.array([reset_observation['measured_finger_joint_deg'][name]
                                        for name in ('jaw1_hinge','jaw2_hinge')])
        self.done=False
        start=np.array(reset_observation['pieces']['piece_centers_m'][self.index])
        if abs(start[2]-self.original[2])>.002 or reset_observation['pieces']['piece_upright_cosines'][self.index]<.98:
            raise ValueError('Near-grasp reset must start with the pawn upright on its original board surface')

    def measurements(self,actual,point,command):
        pieces=actual['pieces']
        center=np.asarray(pieces['piece_centers_m'][self.index])
        target=np.asarray(pieces['piece_grasp_targets_m'][self.index])
        height=float(center[2]-self.original[2])
        error=float(np.linalg.norm(point-target))
        fingers=actual['measured_finger_joint_deg']
        # Closed commands obstructed on both fingers are contact evidence.
        # No contact/magnet attachment or object repositioning is created.
        contact=bool(command[5]<0 and fingers['jaw1_hinge']>command[5]+.8
                     and -fingers['jaw2_hinge']>command[5]+.8 and error<self.config.max_grasp_error_m)
        return center,target,height,error,contact

    def potential(self,actual,point,command):
        _,_,height,error,contact=self.measurements(actual,point,command)
        return float(1.5*np.exp(-(error/.015)**2)+.5*contact+
                     2*contact*np.clip(height/self.config.lift_m,0,1))

    def update(self,actual,point,command):
        if self.done:
            raise RuntimeError('Lift diagnostic finished; reset before stepping')
        self.steps+=1
        center,target,height,error,contact=self.measurements(actual,point,command)
        upright=float(actual['pieces']['piece_upright_cosines'][self.index])
        relative=center-point
        fingers=np.array([actual['measured_finger_joint_deg'][name] for name in ('jaw1_hinge','jaw2_hinge')])
        settled=bool(np.max(np.abs(fingers-self.previous_fingers))<=.2)
        self.previous_fingers=fingers
        candidate=contact and settled and height>=self.config.lift_m and upright>.95
        if candidate:
            if self.relative_anchor is None:
                self.relative_anchor=relative.copy()
            slip=float(np.linalg.norm(relative-self.relative_anchor))
            self.streak=self.streak+1 if slip<=self.config.max_relative_slip_m else 0
            if not self.streak:
                self.relative_anchor=relative.copy()
        else:
            self.streak=0;self.relative_anchor=None;slip=0.
        disturbed=self.monitor._adapt(actual)['pieces']['other_pieces_displaced_over_3mm']
        reason=None
        if disturbed:
            reason='disturbed_other_piece'
        elif upright<np.cos(np.deg2rad(30)):
            reason='pawn_tipped'
        elif height>self.config.max_height_m or np.linalg.norm(center[:2]-self.original[:2])>.05:
            reason='piece_outside_pickup_workspace'
        elif np.linalg.norm(point-target)>.10:
            reason='gripper_left_pickup_workspace'
        elif self.streak>=self.config.hold_steps:
            reason='stable_grasp_lift'
        success=reason=='stable_grasp_lift'
        terminated=reason is not None
        truncated=not terminated and self.steps>=self.config.horizon
        if truncated:
            reason='timeout'
        self.done=terminated or truncated
        reward=10. if success else -2. if terminated else 0.
        return reward,terminated,truncated,{'objective':'grasp_lift_only','task_version':self.version,
            'success':success,'reason':reason,'complete_chess_move':False,'height_m':height,
            'grasp_error_m':error,'finger_contact_evidence':contact,'hold_streak':self.streak,
            'fingers_settled':settled,
            'relative_slip_m':slip,'upright_cosine':upright,'other_pieces_disturbed':int(disturbed)}


class NearGraspEnv(gym.Env):
    metadata={}

    def __init__(self,bank_manifest,url='http://127.0.0.1:8081',split='train',lift_config=None):
        self.bank=json.loads(Path(bank_manifest).read_text())
        self.entries=[row for row in self.bank['entries'] if row['split']==split]
        if not self.entries:
            raise ValueError('No reset entries for the requested split')
        self.fixed_entry=None
        self.lift_config=lift_config or LiftConfig()
        self.base=StateTeacherEnv(url,case=ORDINARY_CASE,config=TeacherConfig(
            max_steps=self.lift_config.horizon,jitter_mm=0,action_mode='delta',
            max_velocity_deg_s=(15.,15.,15.,15.,15.,45.),acceleration_weight=.0005,
            acceleration_cost_cap=.003))
        self.action_space=self.base.action_space
        self.schema={'joints_div180':6,'actual_joint_velocity_div60':6,'opening_to_pawn_grasp_div30mm':3,
            'pawn_displacement_div30mm':3,'desired_e4_minus_pawn_div100mm':3,'finger_angles_div10':2,
            'servo_target_error_div5deg':6,'previous_action':6,'pawn_velocity':6,'pawn_upright':1,
            'nearest_four_other_pieces':16,'move_command_from_to':4,'lift_goal_and_progress':3}
        self.observation_space=spaces.Box(-np.inf,np.inf,(sum(self.schema.values()),),np.float32)
        self.needs_reset=True

    def _observation(self):
        base=self.base;actual=base.actual
        pieces=actual['pieces'];index=self.task.index
        point=base.encoder.point(actual)
        center=np.asarray(pieces['piece_centers_m'][index])
        target=np.asarray(pieces['piece_grasp_targets_m'][index])
        q=np.asarray(actual['measured_joint_deg'])
        velocity=np.asarray(actual.get('joint_velocity_deg_s',(q-base.previous_q)*30))
        fingers=actual['measured_finger_joint_deg']
        all_centers=np.asarray(pieces['piece_centers_m'])
        candidates=np.array([i for i in range(len(all_centers)) if i!=index])
        nearest=candidates[np.argsort(np.linalg.norm(all_centers[candidates]-point,axis=1))[:4]]
        neighbours=np.c_[(all_centers[nearest]-point)/.05,np.asarray(pieces['piece_upright_cosines'])[nearest]]
        piece_velocity=np.asarray(actual['piece_velocities_m_s_rad_s'][index])/np.array([.1,.1,.1,5,5,5])
        move=actual['task_geometry']['move']
        command=[(ord(move[0])-97)/7,(int(move[1])-1)/7,(ord(move[2])-97)/7,(int(move[3])-1)/7]
        values=np.r_[q/180,velocity/60,(target-point)/.03,(center-self.task.original)/.03,
            (np.asarray(actual['task_geometry']['target_center_m'])-center)/.1,
            [fingers['jaw1_hinge']/10,fingers['jaw2_hinge']/10],(base.command-q)/5,self.previous_action,
            piece_velocity,pieces['piece_upright_cosines'][index],neighbours.ravel(),command,
            [self.lift_config.lift_m/.03,self.task.streak/self.lift_config.hold_steps,
             self.task.steps/self.lift_config.horizon]]
        obs=np.asarray(values,np.float32)
        if obs.shape!=self.observation_space.shape or not np.isfinite(obs).all():
            raise ValueError('Invalid near-grasp teacher observations')
        return obs

    def reset(self,*,seed=None,options=None):
        super().reset(seed=seed)
        if options:
            raise ValueError('Use fixed_entry to select a diagnostic reset')
        self.needs_reset=True
        self.entry=self.fixed_entry or self.entries[int(self.np_random.integers(len(self.entries)))]
        latest=self.base.client.restore_reset(self.entry['name'])
        context=latest['reset_context']
        if context['checkpoint_sha256']!=self.entry['sha256']:
            raise ValueError('Reset checkpoint changed after bank preparation')
        baseline=context['baseline_observation']
        base=self.base;base.latest=latest;base.actual=latest['observation']
        if base.actual['joint_map']!=base.mapping or baseline['task_geometry']['move']!='e2e4':
            raise ValueError('Reset calibration or move differs from the teacher')
        base.task=LegalMoveTask(baseline,TaskConfig(max_steps=base.config.max_steps,capture_home_retry=True))
        from teacher.environment import TeacherFeatures
        base.encoder=TeacherFeatures(baseline,base.kinematics,base.anchors,base.config)
        base.command=np.clip(np.asarray(base.mapping['sim_home_deg'])+np.asarray(base.mapping['scale'])*(
            np.asarray(base.actual['target_legacy_action'])-base.mapping['legacy_home']),base.low,base.high)
        base.previous_q=np.asarray(base.actual['measured_joint_deg'])
        base.velocity=np.zeros(6)
        base.previous_potential=base.encoder.potential(base.actual,base.task)
        base.needs_reset=False
        self.task=GraspLiftTask(baseline,base.actual,self.lift_config)
        self.previous_action=np.zeros(6)
        self.previous_potential=self.task.potential(base.actual,base.encoder.point(base.actual),base.command)
        self.needs_reset=False
        return self._observation(),{'reset_name':self.entry['name'],'reset_context':context,
                                    'telemetry':latest,'objective':'grasp_lift_only'}

    def step(self,action):
        if self.needs_reset:
            raise RuntimeError('reset() required before step()')
        self.needs_reset=True
        _,_,base_terminated,base_truncated,info=self.base.step(action)
        point=self.base.encoder.point(self.base.actual)
        reward,terminated,truncated,score=self.task.update(self.base.actual,point,self.base.command)
        if base_terminated and not terminated:
            terminated=True;truncated=False
            score.update(success=False,reason='full_task_'+str(info['score']['reason']))
            reward=-2.
        task_reward=reward
        potential=self.task.potential(self.base.actual,point,self.base.command)
        shaping=.5*(self.base.config.gamma*(0 if terminated else potential)-self.previous_potential)
        cost=min(.003,sum(info['teacher']['penalties'].values()))
        distance=score['grasp_error_m']
        dense=.005*np.exp(-(distance/.015)**2)
        reward=float(reward+shaping+dense-.003-cost)
        self.previous_potential=potential
        self.previous_action=np.asarray(action,np.float32).copy()
        self.needs_reset=terminated or truncated
        self.base.needs_reset=self.needs_reset
        if self.needs_reset:
            self.base.task.done=True
        info['teacher']['unused_full_move_shaping']=info['teacher']['shaping']
        info['teacher'].update(shaping=shaping,potential=potential)
        score['reset_family']=self.entry.get('family','open_approach')
        info.update(score=score,is_success=score['success'],reset_name=self.entry['name'],
                    objective='grasp_lift_only',reward_components={'task':task_reward,'shaping':shaping,
                        'dense':float(dense),'time':-.003,'cost':-cost})
        return self._observation(),reward,terminated,truncated,info

    def close(self):
        self.base.close()
