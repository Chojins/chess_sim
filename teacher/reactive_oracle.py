"""Analytic geometry oracle for corrective training labels, not a learned policy.

It uses current observable geometry/encoders and smooth spatial fields. No
trajectory index, task-progress flag, timed wait, attachment or pose write.
"""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class FlowConfig:
    tau_s:float=.4
    carry_height_m:float=.035
    approach_height_m:float=.025
    pawn_grasp_offset_m:float=.005
    grip_pitch_deg:float=65.
    grip_roll_deg:float=90.
    def __post_init__(self):
        if not all(np.isfinite(v) for v in vars(self).values()) or min(
                self.tau_s,self.carry_height_m,self.approach_height_m,self.pawn_grasp_offset_m)<=0:
            raise ValueError('Invalid geometric supervision configuration')


def smoothstep(x):
    x=np.clip(x,0,1);return x*x*(3-2*x)


class ReactiveOracle:
    version='geometry_feedback_v2_obstructed_closed_jaw'
    def __init__(self,cartesian_config,config=None):
        self.control=cartesian_config;self.config=config or FlowConfig()
    def actions(self,observations):
        x=np.atleast_2d(np.asarray(observations,dtype=float));c=self.config
        if x.ndim!=2 or x.shape[1]!=76 or not np.isfinite(x).all():
            raise ValueError('Reactive oracle expects finite task-space geometry features')
        grasp=x[:,12:15]*.03;goal=x[:,18:21]*.1;height=x[:,17]*.03
        q=x[:,:6]*180;f1=x[:,21]*10;f2=-x[:,22]*10
        cmd_jaw=q[:,5]+x[:,28]*5;previous=x[:,34]
        settled=np.abs(x[:,11]*60)<6
        # PhysX reports instantaneous contact-solver jaw velocities even when
        # the aperture is stationary across control frames. Do not interpret
        # that noise as a request to stop lifting. Require the full close
        # setpoint and a physically obstructed, narrow aperture instead.
        held=(cmd_jaw<-2.5)&(np.maximum(f1,f2)<3.5)&(f1>cmd_jaw+.8)&(f2>cmd_jaw+.8)&(np.linalg.norm(grasp,axis=1)<.012)
        empty=(np.maximum(f1,f2)<.5)&settled
        closing=np.where(np.abs(previous)>.1,previous<0,cmd_jaw<0)
        aligned=(np.linalg.norm(grasp[:,:2],axis=1)<.0025)&(np.abs(grasp[:,2])<.0025)
        open_enough=np.minimum(f1,f2)>4.8
        placed=(np.linalg.norm(goal[:,:2],axis=1)<.003)&(np.abs(height)<.003)
        pheight=height+c.pawn_grasp_offset_m*x[:,70]-grasp[:,2]
        home_pitch=q[:,1:4].sum(1)+x[:,74]*90;home_roll=q[:,4]+x[:,75]*90
        low=smoothstep((.130-pheight)/(.130-.050))
        pitch=home_pitch+(c.grip_pitch_deg-home_pitch)*low
        roll=home_roll+(c.grip_roll_deg-home_roll)*low
        error=np.zeros((len(x),3));jaw=np.ones(len(x))
        # Empty-hand reach: descend through a spatial funnel after XY alignment.
        lateral=np.linalg.norm(grasp[:,:2],axis=1)
        approach=c.approach_height_m*smoothstep((lateral-.003)/(.015-.003))
        error[:,:2]=grasp[:,:2];error[:,2]=grasp[:,2]+approach
        close=(aligned&open_enough)|(closing&~empty&(np.linalg.norm(grasp,axis=1)<.015))
        jaw[close]=-1;error[close]=grasp[close]
        # A stable obstructed grip follows an arch over neighbours, then lowers
        # only within the destination square. This is a training oracle field.
        distance=np.linalg.norm(goal[:,:2],axis=1)
        high=c.carry_height_m*smoothstep((distance-.003)/(.010-.003))
        near=smoothstep((.015-distance)/(.015-.010))
        horizontal=near+(1-near)*smoothstep((height-.015)/(.028-.015))
        error[held,:2]=goal[held,:2]*horizontal[held,None]
        error[held,2]=high[held]-height[held];jaw[held]=-1
        # At a placed pawn, open first. Once the fingers clear it, the same
        # spatial field continuously lifts/retracts toward the viewing pose.
        error[placed]=0;jaw[placed]=1
        moving_home=placed&open_enough
        home_delta=x[:,71:74]*.2
        home_xy_gate=smoothstep((pheight-.035)/(.060-.035))
        error[moving_home]=home_delta[moving_home]
        error[moving_home,:2]*=home_xy_gate[moving_home,None]
        result=np.c_[error/(self.control.translation_step_m*30*c.tau_s),
            (pitch-q[:,1:4].sum(1))/(self.control.pitch_step_deg*30*c.tau_s),
            (roll-q[:,4])/(self.control.roll_step_deg*30*c.tau_s),jaw]
        return np.clip(result,-1,1).astype(np.float32)
    def predict(self,observations,deterministic=True):
        actions=self.actions(observations)
        return (actions[0] if np.asarray(observations).ndim==1 else actions),None
