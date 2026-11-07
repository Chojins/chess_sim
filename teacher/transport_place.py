"""Held-pawn transport, on-square release and retreat; no home-start claim."""
from dataclasses import dataclass
import json
from pathlib import Path

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from chess_task import LegalMoveTask,TaskConfig
from teacher.environment import StateTeacherEnv,TeacherConfig,TeacherFeatures
from teacher.reference import ORDINARY_CASE


@dataclass(frozen=True)
class PlaceConfig:
    horizon:int=192
    placement_m:float=.005
    height_tolerance_m:float=.003
    release_height_limit_m:float=.008
    clearance_m:float=.035
    stable_steps:int=10
    max_grasp_error_m:float=.012
    max_height_m:float=.080

    def __post_init__(self):
        if (type(self.horizon) is not int or self.horizon<1 or type(self.stable_steps) is not int or self.stable_steps<1
                or not all(np.isfinite(v) and v>0 for k,v in vars(self).items() if k not in ('horizon','stable_steps'))):
            raise ValueError('Invalid transport/place configuration')


class TransportPlaceTask:
    version='held_transport_place_v2'

    def __init__(self,baseline,actual,point,command,provenance,config=None):
        self.config=config or PlaceConfig()
        geometry=baseline['task_geometry']
        if geometry['move']!='e2e4' or geometry['capture_index'] is not None:
            raise ValueError('Initial placement curriculum is ordinary e2e4 only')
        self.index=geometry['source_index'];self.original=np.asarray(baseline['pieces']['piece_centers_m'][self.index])
        self.target=np.asarray(geometry['target_center_m'])
        self.monitor=LegalMoveTask(baseline,TaskConfig(max_steps=self.config.horizon))
        self.steps=self.streak=0;self.done=False;self.released=False
        self.pending_release=None;self.release_location=None
        self.previous_center=np.asarray(actual['pieces']['piece_centers_m'][self.index])
        self.previous_fingers=np.array([actual['measured_finger_joint_deg'][k] for k in ('jaw1_hinge','jaw2_hinge')])
        self.initial_relative=self.previous_center-np.asarray(point)
        self.prior_max_lift=float(provenance.get('prior_max_lift_m',0.))
        if not np.isfinite(self.prior_max_lift) or self.prior_max_lift<.005 or not provenance.get('prior_lift_verified'):
            raise ValueError('A held reset needs verified physical lift history')
        _,_,height,error,contact=self.measurements(actual,point,command)
        self.previous_grasp_error=error
        if not contact or height<-.002 or actual['pieces']['piece_upright_cosines'][self.index]<.95:
            raise ValueError('Transport reset must contain an upright, actually held pawn')
        if error>self.config.max_grasp_error_m:
            raise ValueError('Reset grasp is too far from the specified material point')

    def measurements(self,actual,point,command):
        center=np.asarray(actual['pieces']['piece_centers_m'][self.index])
        grasp=np.asarray(actual['pieces']['piece_grasp_targets_m'][self.index])
        error=float(np.linalg.norm(np.asarray(point)-grasp))
        fingers=actual['measured_finger_joint_deg']
        contact=bool(command[5]<0 and fingers['jaw1_hinge']>command[5]+.8
                     and -fingers['jaw2_hinge']>command[5]+.8 and error<self.config.max_grasp_error_m)
        return center,grasp,float(center[2]-self.original[2]),error,contact

    def potential(self,actual,point,command):
        center,_,height,_,_=self.measurements(actual,point,command)
        error=float(np.linalg.norm(center[:2]-self.target[:2]))
        approach=3*np.exp(-(error/.04)**2)
        lower=np.exp(-(error/.015)**2)*np.exp(-(height/.01)**2)
        retreat=float(self.released)*np.clip((np.linalg.norm(np.asarray(point)-center)-.015)/self.config.clearance_m,0,1)
        return float(approach+lower+2*self.released+retreat)

    def update(self,actual,point,command):
        if self.done:
            raise RuntimeError('Transport/place task finished; reset before stepping')
        self.steps+=1
        c=self.config
        center,grasp,height,grasp_error,contact=self.measurements(actual,point,command)
        upright=float(actual['pieces']['piece_upright_cosines'][self.index])
        error=float(np.linalg.norm(center[:2]-self.target[:2]))
        clearance=float(np.linalg.norm(np.asarray(point)-center))
        fingers=actual['measured_finger_joint_deg']
        angles=np.array([fingers[k] for k in ('jaw1_hinge','jaw2_hinge')])
        # A command beyond the obstructed finger angle starts opening the gap.
        # Use the PREVIOUS physical frame: this update follows the commanded
        # physics step, which may already detach or magnetically move the pawn.
        opening_threshold=max(0.,min(self.previous_fingers[0],-self.previous_fingers[1]))+.2
        if (not self.released and self.pending_release is None and command[5]>opening_threshold
                and self.previous_grasp_error<.018):
            before=self.previous_center
            local=(before[:2]-self.target[:2])@self.monitor.square_inverse.T
            self.pending_release={'center_m':before.tolist(),'height_m':float(before[2]-self.original[2]),
                                  'inside_target_square':bool(np.max(np.abs(local))<=.5),
                                  'step':self.steps}
        if (self.pending_release is not None and not self.released and command[5]<0
                and fingers['jaw1_hinge']<3.5 and fingers['jaw2_hinge']>-3.5):
            self.pending_release=None  # An aborted opening attempt did not release the pawn.
        release_event=False;bad_release=False
        if (not self.released and self.pending_release is not None
                and fingers['jaw1_hinge']>=4 and fingers['jaw2_hinge']<=-4):
            self.release_location=dict(self.pending_release)
            bad_release=(not self.release_location['inside_target_square']
                         or self.release_location['height_m']>c.release_height_limit_m)
            if not bad_release:
                self.released=True;release_event=True
        stable=(self.released and error<c.placement_m and abs(height)<c.height_tolerance_m
                and upright>.95 and clearance>c.clearance_m
                and np.linalg.norm(center-self.previous_center)<.0005)
        self.streak=self.streak+1 if stable else 0
        disturbed=self.monitor._adapt(actual)['pieces']['other_pieces_displaced_over_3mm']
        reason=None
        if disturbed: reason='disturbed_other_piece'
        elif upright<np.cos(np.deg2rad(30)): reason='pawn_tipped'
        elif bad_release: reason='release_outside_target_or_too_high'
        elif height>c.max_height_m or np.linalg.norm(center[:2]-self.target[:2])>.10: reason='piece_outside_transport_workspace'
        elif not self.released and self.pending_release is None and grasp_error>.025: reason='lost_grasp_before_release'
        elif self.streak>=c.stable_steps: reason='stable_place_release'
        success=reason=='stable_place_release'
        terminated=reason is not None
        truncated=not terminated and self.steps>=c.horizon
        if truncated: reason='timeout'
        self.done=terminated or truncated
        self.previous_center=center.copy();self.previous_fingers=angles;self.previous_grasp_error=grasp_error
        reward=(10. if success else -2. if terminated else 0.)+(2. if release_event and not bad_release else 0.)
        return reward,terminated,truncated,{'objective':'transport_place_only','task_version':self.version,
            'success':success,'reason':reason,'complete_chess_move':False,'target_error_m':error,
            'height_m':height,'grasp_error_m':grasp_error,'finger_contact_evidence':contact,
            'released_on_target':self.released,'release_intent':self.release_location or self.pending_release,
            'release_bonus':bool(release_event and not bad_release),'clearance_m':clearance,
            'placement_streak':self.streak,'upright_cosine':upright,'other_pieces_disturbed':int(disturbed)}


class TransportPlaceEnv(gym.Env):
    metadata={}

    def __init__(self,bank_manifest,url='http://127.0.0.1:8081',split='train',config=None):
        self.bank=json.loads(Path(bank_manifest).read_text())
        self.entries=[row for row in self.bank['entries'] if row['split']==split]
        if not self.entries: raise ValueError('No transport states for this split')
        self.fixed_entry=None;self.place_config=config or PlaceConfig()
        self.base=StateTeacherEnv(url,case=ORDINARY_CASE,config=TeacherConfig(
            max_steps=self.place_config.horizon,jitter_mm=0,action_mode='delta',
            max_velocity_deg_s=(15.,15.,15.,15.,15.,45.),acceleration_weight=.0005,acceleration_cost_cap=.003))
        self.action_space=self.base.action_space
        self.schema={'joints_div180':6,'actual_joint_velocity_div60':6,'opening_to_pawn_grasp_div30mm':3,
            'pawn_displacement_div30mm':3,'desired_e4_minus_pawn_div100mm':3,'finger_angles_div10':2,
            'servo_target_error_div5deg':6,'previous_action':6,'pawn_velocity':6,'pawn_upright':1,
            'nearest_four_other_pieces':16,'move_command_from_to':4,'placement_goal_and_progress':3,
            'released_on_target':1,'pawn_up_vector':3}
        self.observation_space=spaces.Box(-np.inf,np.inf,(sum(self.schema.values()),),np.float32)
        self.needs_reset=True

    def _observation(self):
        base=self.base;o=base.actual;i=self.task.index
        centers=np.asarray(o['pieces']['piece_centers_m']);center=centers[i]
        grasp=np.asarray(o['pieces']['piece_grasp_targets_m'][i]);point=base.encoder.point(o)
        q=np.asarray(o['measured_joint_deg']);f=o['measured_finger_joint_deg']
        candidates=np.array([j for j in range(len(centers)) if j!=i])
        nearest=candidates[np.argsort(np.linalg.norm(centers[candidates]-point,axis=1))[:4]]
        neighbours=np.c_[(centers[nearest]-point)/.05,np.asarray(o['pieces']['piece_upright_cosines'])[nearest]]
        move=o['task_geometry']['move'];coords=[(ord(move[0])-97)/7,(int(move[1])-1)/7,(ord(move[2])-97)/7,(int(move[3])-1)/7]
        up=grasp-center;up=up/max(np.linalg.norm(up),1e-8)
        result=np.r_[q/180,np.asarray(o['joint_velocity_deg_s'])/60,(grasp-point)/.03,
            (center-self.task.original)/.03,(self.task.target-center)/.1,[f['jaw1_hinge']/10,f['jaw2_hinge']/10],
            (base.command-q)/5,self.previous_action,np.asarray(o['piece_velocities_m_s_rad_s'][i])/[.1,.1,.1,5,5,5],
            o['pieces']['piece_upright_cosines'][i],neighbours.ravel(),coords,
            [self.place_config.height_tolerance_m/.03,self.task.streak/self.place_config.stable_steps,
             self.task.steps/self.place_config.horizon],float(self.task.released),up].astype(np.float32)
        if result.shape!=self.observation_space.shape or not np.isfinite(result).all():
            raise ValueError('Invalid transport observations')
        return result

    def reset(self,*,seed=None,options=None):
        super().reset(seed=seed)
        if options: raise ValueError('Use fixed_entry to choose a transport state')
        self.needs_reset=True
        self.entry=self.fixed_entry or self.entries[int(self.np_random.integers(len(self.entries)))]
        latest=self.base.client.restore_reset(self.entry['name']);context=latest['reset_context']
        if context['checkpoint_sha256']!=self.entry['sha256']: raise ValueError('Transport checkpoint changed')
        baseline=context['baseline_observation'];base=self.base;base.latest=latest;base.actual=latest['observation']
        if base.actual['joint_map']!=base.mapping: raise ValueError('Transport calibration changed')
        base.task=LegalMoveTask(baseline,TaskConfig(max_steps=base.config.max_steps,capture_home_retry=True))
        # Restore verified history rather than changing the original board
        # baseline to the already lifted/transported pawn's current pose.
        history=context['provenance']
        if not history.get('prior_lift_verified'): raise ValueError('Missing pre-reset lift history')
        base.task.lifted=True;base.task.departed=True;base.task.lift_streak=3
        base.task.max_lift=float(history['prior_max_lift_m'])
        base.task.transported=bool(history.get('prior_transported',False))
        base.task.previous_center=np.asarray(base.actual['pieces']['piece_centers_m'][base.task.source_index])
        base.encoder=TeacherFeatures(baseline,base.kinematics,base.anchors,base.config)
        base.command=np.clip(np.asarray(base.mapping['sim_home_deg'])+np.asarray(base.mapping['scale'])*(
            np.asarray(base.actual['target_legacy_action'])-base.mapping['legacy_home']),base.low,base.high)
        base.previous_q=np.asarray(base.actual['measured_joint_deg']);base.velocity=np.zeros(6)
        base.previous_potential=base.encoder.potential(base.actual,base.task);base.needs_reset=False
        point=base.encoder.point(base.actual)
        self.task=TransportPlaceTask(baseline,base.actual,point,base.command,history,self.place_config)
        self.previous_action=np.zeros(6);self.previous_potential=self.task.potential(base.actual,point,base.command)
        self.needs_reset=False
        return self._observation(),{'reset_name':self.entry['name'],'reset_context':context,
                                    'telemetry':latest,'objective':'transport_place_only'}

    def step(self,action):
        if self.needs_reset: raise RuntimeError('reset() required before stepping')
        self.needs_reset=True
        _,_,base_terminated,_,info=self.base.step(action)
        point=self.base.encoder.point(self.base.actual)
        task_reward,terminated,truncated,score=self.task.update(self.base.actual,point,self.base.command)
        if base_terminated and not terminated:
            terminated=True;truncated=False;task_reward=-2.
            score.update(success=False,reason='full_task_'+str(info['score']['reason']))
        potential=self.task.potential(self.base.actual,point,self.base.command)
        shaping=.5*(self.base.config.gamma*(0 if terminated else potential)-self.previous_potential)
        cost=min(.003,sum(info['teacher']['penalties'].values()))
        dense=float(.005*np.exp(-(score['target_error_m']/.04)**2))
        reward=float(task_reward+shaping+dense-.003-cost)
        self.previous_potential=potential;self.previous_action=np.asarray(action,np.float32).copy()
        self.needs_reset=terminated or truncated;self.base.needs_reset=self.needs_reset
        if self.needs_reset: self.base.task.done=True
        info['teacher']['unused_full_move_shaping']=info['teacher']['shaping']
        info['teacher'].update(shaping=shaping,potential=potential)
        score['reset_family']=self.entry['family']
        info.update(score=score,is_success=score['success'],reset_name=self.entry['name'],objective='transport_place_only',
            reward_components={'task':task_reward,'shaping':shaping,'dense':dense,'time':-.003,'cost':-cost})
        return self._observation(),reward,terminated,truncated,info

    def close(self):
        self.base.close()
