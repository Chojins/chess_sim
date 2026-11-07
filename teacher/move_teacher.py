"""One action contract and one placed/released/home goal for every e2e4 start."""
from dataclasses import dataclass,replace
import json
from pathlib import Path
import gymnasium as gym
from gymnasium import spaces
import numpy as np

from chess_task import LegalMoveTask,TaskConfig
from teacher.environment import StateTeacherEnv,TeacherConfig,TeacherFeatures
from teacher.reference import ORDINARY_CASE
from teacher.success_sampler import SuccessSampler


@dataclass(frozen=True)
class MoveConfig:
    horizon:int=800
    stable_steps:int=10
    placement_m:float=.005
    release_height_limit_m:float=.008
    clearance_m:float=.035
    home_deg:float=5.
    max_height_m:float=.10
    def __post_init__(self):
        if (type(self.horizon) is not int or type(self.stable_steps) is not int
                or not all(np.isfinite(v) and v>0 for v in vars(self).values())):
            raise ValueError('Invalid move configuration')


class MoveTask:
    version='unified_e2e4_v1'
    def __init__(self,baseline,actual,point,command,history,config=None):
        self.config=config or MoveConfig();g=baseline['task_geometry']
        if g['move']!='e2e4' or g['capture_index'] is not None:
            raise ValueError('Only ordinary blue e2e4 is implemented')
        self.index=g['source_index'];self.target=np.array(g['target_center_m'])
        self.original=np.array(baseline['pieces']['piece_centers_m'][self.index])
        self.home=np.array(baseline['measured_joint_deg'])[:5]
        self.monitor=LegalMoveTask(baseline,TaskConfig(max_steps=self.config.horizon))
        self.steps=self.streak=self.lift_streak=0;self.done=False
        self.max_lift=float(history.get('prior_max_lift_m',0))
        self.lifted=bool(history.get('prior_lift_verified',False))
        if not np.isfinite(self.max_lift) or (self.lifted and self.max_lift<.005):
            raise ValueError('Invalid lift history')
        self.released=bool(history.get('prior_release_verified',False))
        self.release_location=history.get('release_intent') if self.released else None
        if self.released and (not self.lifted or not self.release_location
                or not self.release_location['inside_target_square']
                or not -.002<=self.release_location['height_m']<=self.config.release_height_limit_m):
            raise ValueError('Invalid release history')
        self.pending_release=None
        self.previous_center=np.array(actual['pieces']['piece_centers_m'][self.index])
        self.previous_fingers=self.fingers(actual)
        self.previous_error=float(np.linalg.norm(np.asarray(point)-actual['pieces']['piece_grasp_targets_m'][self.index]))
        if history.get('held_start'):
            _,height,error,contact=self.measure(actual,point,command)
            if not self.lifted or not contact or height<-.002 or actual['pieces']['piece_upright_cosines'][self.index]<.95:
                raise ValueError('Held reset is not a verified physical grasp')
        if self.released and (np.linalg.norm(self.previous_center[:2]-self.target[:2])>.005
                or abs(self.previous_center[2]-self.original[2])>.003):
            raise ValueError('Released reset is not placed on target')
    @staticmethod
    def fingers(actual):
        return np.array([actual['measured_finger_joint_deg'][k] for k in ('jaw1_hinge','jaw2_hinge')])
    def measure(self,actual,point,command):
        center=np.asarray(actual['pieces']['piece_centers_m'][self.index])
        error=float(np.linalg.norm(np.asarray(point)-actual['pieces']['piece_grasp_targets_m'][self.index]))
        f=self.fingers(actual)
        contact=bool(command[5]<0 and f[0]>command[5]+.8 and -f[1]>command[5]+.8 and error<.012)
        return center,float(center[2]-self.original[2]),error,contact
    def potential(self,actual,point,command):
        center,height,error,contact=self.measure(actual,point,command)
        approach=2*np.exp(-(error/.04)**2)*(not self.lifted)
        carry=3*np.exp(-(np.linalg.norm(center[:2]-self.target[:2])/.04)**2)*self.lifted
        home=2*np.exp(-(np.max(np.abs(np.asarray(actual['measured_joint_deg'])[:5]-self.home))/45)**2)*self.released
        return float(approach+.5*contact+2*self.lifted+carry+2*self.released+home)
    def update(self,actual,point,command):
        if self.done: raise RuntimeError('Move finished; reset required')
        self.steps+=1;c=self.config
        center,height,error,contact=self.measure(actual,point,command)
        f=self.fingers(actual);upright=float(actual['pieces']['piece_upright_cosines'][self.index])
        self.max_lift=max(self.max_lift,height)
        self.lift_streak=self.lift_streak+1 if height>.005 and contact else 0
        opening=max(0.,min(self.previous_fingers[0],-self.previous_fingers[1]))+.2
        if self.lifted and not self.released and self.pending_release is None and command[5]>opening and self.previous_error<.018:
            before=self.previous_center;local=(before[:2]-self.target[:2])@self.monitor.square_inverse.T
            self.pending_release={'center_m':before.tolist(),'height_m':float(before[2]-self.original[2]),
                'inside_target_square':bool(np.max(np.abs(local))<=.5),'step':self.steps}
        self.lifted|=self.lift_streak>=3
        if self.pending_release is not None and command[5]<0 and f[0]<3.5 and f[1]>-3.5 and not self.released:
            self.pending_release=None
        bad_release=False
        if self.pending_release is not None and not self.released and f[0]>=4 and f[1]<=-4:
            self.release_location=dict(self.pending_release)
            bad_release=(not self.release_location['inside_target_square']
                or not -.002<=self.release_location['height_m']<=c.release_height_limit_m)
            if not bad_release: self.released=True
        target_error=float(np.linalg.norm(center[:2]-self.target[:2]))
        home_error=float(np.max(np.abs(np.asarray(actual['measured_joint_deg'])[:5]-self.home)))
        clearance=float(np.linalg.norm(np.asarray(point)-center))
        placed=(self.released and target_error<c.placement_m and abs(height)<.003 and upright>.95
            and clearance>c.clearance_m and np.linalg.norm(center-self.previous_center)<.0005)
        self.streak=self.streak+1 if placed and home_error<c.home_deg else 0
        disturbed=self.monitor._adapt(actual)['pieces']['other_pieces_displaced_over_3mm']
        reason=None
        if disturbed: reason='disturbed_other_piece'
        elif upright<np.cos(np.deg2rad(30)): reason='pawn_tipped'
        elif bad_release: reason='release_outside_target_or_too_high'
        elif height>c.max_height_m or np.linalg.norm(center[:2]-self.target[:2])>.15: reason='piece_outside_workspace'
        elif self.streak>=c.stable_steps: reason='placed_released_home'
        success=reason=='placed_released_home';terminated=reason is not None
        truncated=not terminated and self.steps>=c.horizon
        if truncated: reason='timeout'
        self.done=terminated or truncated
        self.previous_center=center.copy();self.previous_fingers=f;self.previous_error=error
        return (10. if success else -2. if terminated else 0.),terminated,truncated,{
            'objective':'unified_e2e4','task_version':self.version,'success':success,'reason':reason,
            'complete_chess_move':success,'target_error_m':target_error,'height_m':height,'max_lift_m':self.max_lift,
            'grasp_error_m':error,'finger_contact_evidence':contact,'lifted':bool(self.lifted),
            'released_on_target':self.released,'release_intent':self.release_location or self.pending_release,
            'clearance_m':clearance,'home_error_deg':home_error,'completion_streak':self.streak,
            'upright_cosine':upright,'other_pieces_disturbed':int(disturbed)}


class MoveEnv(gym.Env):
    metadata={}
    def __init__(self,bank_manifest,url='http://127.0.0.1:8081',split='train',config=None,adaptive=True,gamma=.99):
        self.bank=json.loads(Path(bank_manifest).read_text());self.entries=[r for r in self.bank['entries'] if r['split']==split]
        if not self.entries: raise ValueError('No move resets for split')
        self.fixed_entry=None;self.move_config=config or MoveConfig();self.adaptive=adaptive
        self.sampler=SuccessSampler([r['name'] for r in self.entries]);self.training_names=set(self.sampler.names)
        self.base=StateTeacherEnv(url,case=ORDINARY_CASE,config=TeacherConfig(max_steps=self.move_config.horizon,
            gamma=gamma,jitter_mm=0,action_mode='delta',max_velocity_deg_s=(15.,15.,15.,15.,15.,45.),
            acceleration_weight=.0005,acceleration_cost_cap=.003))
        self.action_space=self.base.action_space
        self.schema={'joints_div180':6,'actual_velocity_div60':6,'opening_to_grasp_div30mm':3,
            'pawn_displacement_div30mm':3,'goal_minus_pawn_div100mm':3,'fingers_div10':2,
            'servo_error_div5deg':6,'previous_action':6,'pawn_velocity':6,'upright':1,
            'nearest_four_neighbours':16,'move_from_to':4,'home_error_div90deg':5,'time_remaining':1,'pawn_up':3}
        self.observation_space=spaces.Box(-np.inf,np.inf,(sum(self.schema.values()),),np.float32);self.needs_reset=True
    def _feature_values(self):
        base=self.base;o=base.actual;i=self.task.index;p=o['pieces'];centers=np.asarray(p['piece_centers_m']);center=centers[i]
        grasp=np.asarray(p['piece_grasp_targets_m'][i]);point=base.encoder.point(o);q=np.asarray(o['measured_joint_deg'])
        candidates=np.array([j for j in range(len(centers)) if j!=i]);nearest=candidates[np.argsort(np.linalg.norm(centers[candidates]-point,axis=1))[:4]]
        neighbours=np.c_[(centers[nearest]-point)/.05,np.asarray(p['piece_upright_cosines'])[nearest]]
        f=self.task.fingers(o);up=grasp-center;up/=max(np.linalg.norm(up),1e-8)
        values=np.r_[q/180,np.asarray(o['joint_velocity_deg_s'])/60,(grasp-point)/.03,(center-self.task.original)/.03,
            (self.task.target-center)/.1,f/10,(base.command-q)/5,self.previous_action,
            np.asarray(o['piece_velocities_m_s_rad_s'][i])/[.1,.1,.1,5,5,5],p['piece_upright_cosines'][i],
            neighbours.ravel(),[4/7,1/7,4/7,3/7],(self.task.home-q[:5])/90,
            1-self.task.steps/self.episode_config.horizon,up].astype(np.float32)
        return values
    def _observation(self):
        values=self._feature_values()
        if values.shape!=self.observation_space.shape or not np.isfinite(values).all(): raise ValueError('Invalid move features')
        return values
    def reset(self,*,seed=None,options=None):
        super().reset(seed=seed)
        if options: raise ValueError('Use fixed_entry')
        self.needs_reset=True
        self.entry=self.fixed_entry or self.entries[self.sampler.sample(self.np_random) if self.adaptive else int(self.np_random.integers(len(self.entries)))]
        self.episode_config=replace(self.move_config,horizon=min(self.move_config.horizon,self.entry.get('horizon',self.move_config.horizon)))
        latest=self.base.client.restore_reset(self.entry['name']);context=latest['reset_context'];base=self.base
        if context['checkpoint_sha256']!=self.entry['sha256']: raise ValueError('Move reset changed')
        baseline=context['baseline_observation'];base.latest=latest;base.actual=latest['observation']
        if base.actual['joint_map']!=base.mapping: raise ValueError('Calibration changed')
        base.task=LegalMoveTask(baseline,TaskConfig(max_steps=self.move_config.horizon,capture_home_retry=True))
        # Base scoring supplies generic controller telemetry only. MoveTask is
        # the sole terminal/reward authority; its original baseline is retained.
        history=context['provenance']
        base.task.lifted=bool(history.get('prior_lift_verified',False));base.task.max_lift=float(history.get('prior_max_lift_m',0))
        base.task.departed=bool(base.task.lifted)
        base.task.previous_center=np.array(base.actual['pieces']['piece_centers_m'][baseline['task_geometry']['source_index']])
        base.encoder=TeacherFeatures(baseline,base.kinematics,base.anchors,base.config)
        base.command=np.clip(np.array(base.mapping['sim_home_deg'])+np.array(base.mapping['scale'])*(
            np.asarray(base.actual['target_legacy_action'])-base.mapping['legacy_home']),base.low,base.high)
        base.previous_q=np.asarray(base.actual['measured_joint_deg']);base.velocity=np.zeros(6)
        base.previous_potential=base.encoder.potential(base.actual,base.task);base.needs_reset=False
        self.task=MoveTask(baseline,base.actual,base.encoder.point(base.actual),base.command,history,self.episode_config)
        self.previous_action=np.zeros(6);self.previous_potential=self.task.potential(base.actual,base.encoder.point(base.actual),base.command)
        self.needs_reset=False
        return self._observation(),{'reset_name':self.entry['name'],'reset_context':context,'telemetry':latest,'objective':'unified_e2e4'}
    def step(self,action):
        if self.needs_reset: raise RuntimeError('Reset before stepping')
        self.needs_reset=True
        # The old task can finish earlier (for example after an empty home
        # return); do not let it terminate this task or refuse future steps.
        self.base.task.done=False
        _,_,_,_,info=self.base.step(action)
        point=self.base.encoder.point(self.base.actual)
        terminal,terminated,truncated,score=self.task.update(self.base.actual,point,self.base.command)
        potential=self.task.potential(self.base.actual,point,self.base.command)
        shaping=.5*(self.base.config.gamma*(0 if terminated else potential)-self.previous_potential)
        cost=min(.003,sum(info['teacher']['penalties'].values()));reward=float(terminal+shaping-.003-cost)
        self.previous_action=np.asarray(action,np.float32).copy();self.previous_potential=potential
        self.needs_reset=terminated or truncated;self.base.needs_reset=self.needs_reset
        score['reset_family']=self.entry['family'];score['home_start']=self.entry['family']=='home'
        if self.needs_reset and self.fixed_entry is None and self.entry['name'] in self.training_names:
            self.sampler.record(self.entry['name'],bool(score['success']))
        info['teacher']['unused_full_move_shaping']=info['teacher']['shaping']
        info['teacher'].update(shaping=shaping,potential=potential)
        info.update(score=score,is_success=score['success'],reset_name=self.entry['name'],objective='unified_e2e4',
            reward_components={'task':terminal,'shaping':shaping,'time':-.003,'cost':-cost})
        return self._observation(),reward,terminated,truncated,info
    def close(self): self.base.close()
