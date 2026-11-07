"""Generic differential IK and absolute open/close jaw; no chess-phase logic."""
from dataclasses import dataclass,asdict
import numpy as np
from gymnasium import spaces
from review_grasp_geometry import opening_center
from teacher.move_teacher import MoveEnv


@dataclass(frozen=True)
class CartesianConfig:
    translation_step_m:float=.003
    pitch_step_deg:float=1.5
    roll_step_deg:float=.5
    damping:float=.001
    open_jaw_deg:float=6.
    reference_open_threshold_deg:float=4.
    def __post_init__(self):
        if not all(np.isfinite(v) and v>0 for v in vars(self).values()): raise ValueError('Invalid Cartesian controller')


class CartesianControl:
    version='opening_xyz_pitch_roll_binary_jaw_v1'
    def __init__(self,spec,anchors,low,high,max_velocity,config=None):
        self.spec=spec;self.anchors=anchors;self.low=np.asarray(low);self.high=np.asarray(high)
        self.max_step=np.asarray(max_velocity)/30;self.config=config or CartesianConfig()
        if self.high[5]<self.config.open_jaw_deg: raise ValueError('Opening exceeds gripper limit')
    def point(self,q,fingers):
        measured=np.asarray(q).copy();measured[5]=fingers['jaw1_hinge']
        return opening_center(self.spec,self.anchors,measured,fingers['jaw2_hinge'])
    def coordinates(self,q,fingers):
        c=self.config
        return np.r_[self.point(q,fingers)/c.translation_step_m,
            np.sum(q[1:4])/c.pitch_step_deg,q[4]/c.roll_step_deg]
    def command(self,current,actual,action):
        action=np.asarray(action,np.float32)
        if action.shape!=(6,) or not np.isfinite(action).all() or np.max(np.abs(action))>1:
            raise ValueError('Expected six normalized Cartesian/jaw actions')
        q=np.asarray(current,float);f=actual['measured_finger_joint_deg'];origin=self.coordinates(q,f)
        jac=np.stack([(self.coordinates(q+np.eye(6)[j]*.001,f)-origin)/.001 for j in range(5)],axis=1)
        dq=np.linalg.solve(jac.T@jac+self.config.damping**2*np.eye(5),jac.T@action[:5])
        dq/=max(1.,float(np.max(np.abs(dq)/self.max_step[:5])))
        target=q.copy();target[:5]+=dq
        jaw=self.config.open_jaw_deg if action[5]>0 else self.low[5]
        target[5]+=np.clip(jaw-q[5],-self.max_step[5],self.max_step[5])
        return np.clip(target,self.low,self.high)
    def reference_action(self,current,actual,target):
        current=np.asarray(current);target=np.asarray(target);f=actual['measured_finger_joint_deg']
        delta=self.coordinates(target,f)-self.coordinates(current,f)
        delta/=max(1.,float(np.max(np.abs(delta))))
        return np.r_[delta,1. if target[5]>self.config.reference_open_threshold_deg else -1.].astype(np.float32)


class TaskSpaceMoveEnv(MoveEnv):
    policy_action_space=CartesianControl.version
    def __init__(self,*args,cartesian_config=None,**kwargs):
        self.cartesian_config=cartesian_config or CartesianConfig()
        super().__init__(*args,**kwargs)
        self.controller=CartesianControl(self.base.kinematics,self.base.anchors,self.base.low,self.base.high,
            self.base.config.max_velocity_deg_s,self.cartesian_config)
        renamed={'previous_action':'previous_tool_action','time_remaining':'unused_time_slot'}
        self.schema={renamed.get(k,k):v for k,v in self.schema.items()}
        self.schema.update(home_opening_delta_div200mm=3,home_pitch_roll_delta_div90deg=2)
        self.observation_space=spaces.Box(-np.inf,np.inf,(76,),np.float32)
    def _observation(self):
        values=self._feature_values();values[67]=0.  # Do not learn a trajectory clock.
        q=np.asarray(self.base.actual['measured_joint_deg']);home=np.r_[self.task.home,self.cartesian_config.open_jaw_deg]
        home_point=opening_center(self.base.kinematics,self.base.anchors,home,-home[5])
        point=self.base.encoder.point(self.base.actual)
        result=np.r_[values,(home_point-point)/.2,
            np.sum(home[1:4]-q[1:4])/90,(home[4]-q[4])/90].astype(np.float32)
        if result.shape!=self.observation_space.shape or not np.isfinite(result).all(): raise ValueError('Invalid task-space features')
        return result
    def step(self,action):
        raw=np.asarray(action,np.float32)
        target=self.controller.command(self.base.command,self.base.actual,raw)
        joint=self.base.normalized_target(target)
        _,reward,done,truncated,info=super().step(joint)
        self.previous_action=raw.copy()
        info['teacher'].update(policy_action_space=self.controller.version,executed_joint_action=joint.tolist(),
            cartesian_config=asdict(self.cartesian_config))
        return self._observation(),reward,done,truncated,info
    def reference_action(self,target):
        return self.controller.reference_action(self.base.command,self.base.actual,target)
