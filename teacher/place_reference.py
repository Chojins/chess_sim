"""Kinematic feasibility probe only. Never invoked by the learned policy."""
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np

from capture_transport import fit_upright,rotation
from review_grasp_geometry import opening_center


def fit_payload(kin,start,target,local_up):
    """Position first; softly prefer upright without demanding unreachable tilt.

    A slightly tilted pawn can be held between the fingers. Requiring its axis
    to be exactly vertical can overconstrain this five-axis arm. Physics, not
    this solver, must establish the final upright placement.
    """
    q=np.asarray(start,float).copy()
    def error(v):
        return np.r_[(kin.point(v)-target)*1000.,(local_up@rotation(kin.specification,v)-[0,0,1])*2.]
    for _ in range(180):
        current=error(q)
        jac=np.stack([(error(q+np.eye(6)[i]*.001)-current)/.001 for i in range(5)],axis=1)
        delta=-np.linalg.solve(jac.T@jac+.02*np.eye(5),jac.T@current)
        if np.linalg.norm(current[:3])<.1: break
        for fraction in (1.,.5,.25,.125):
            candidate=q.copy();candidate[:5]=np.clip(q[:5]+np.clip(delta,-2,2)*fraction,kin.limits[:5,0],kin.limits[:5,1])
            if np.linalg.norm(error(candidate))<np.linalg.norm(current): q=candidate;break
        else: break
    if np.linalg.norm(error(q)[:3])>.5: raise ValueError('Payload position unreachable')
    return q


class PlaceReference:
    def __init__(self,env):
        self.env=env;self.queue=[];self.stage='carry'
        base=env.base;q=np.asarray(base.actual['measured_joint_deg']).copy()
        self.jaw=q[5];self.spec=base.kinematics;self.anchors=base.anchors
        center=np.asarray(base.actual['pieces']['piece_centers_m'][env.task.index])
        grasp=np.asarray(base.actual['pieces']['piece_grasp_targets_m'][env.task.index])
        point=base.encoder.point(base.actual);matrix=rotation(self.spec,q)
        self.offset=(center-point)@matrix.T
        self.up=(grasp-center)/np.linalg.norm(grasp-center)@matrix.T
        kin=SimpleNamespace(specification=self.spec,limits=np.c_[base.low,base.high])
        def held_point(values):
            geometry=np.asarray(values).copy();geometry[5]=self.jaw
            return opening_center(self.spec,self.anchors,geometry,-self.jaw)+self.offset@rotation(self.spec,geometry)
        kin.point=held_point
        targets=[]
        if env.entry['family']=='held_source':
            targets.extend([np.r_[center[:2],env.task.original[2]+.035],
                            np.r_[env.task.target[:2],env.task.original[2]+.035]])
        targets.append(env.task.target+[0,0,.001])
        q[5]=-3.
        for target in targets:
            origin=kin.point(q)
            for fraction in np.linspace(0,1,max(2,int(np.ceil(np.linalg.norm(target-origin)/.003)))+1)[1:]:
                fitted=fit_payload(kin,q,origin+(target-origin)*fraction,self.up)
                fitted[5]=-3.;self.extend(q,fitted);q=fitted
        self.last=q.copy();self.queue.extend([q.copy() for _ in range(8)])
    def extend(self,start,target):
        count=max(1,int(np.ceil(np.max(np.abs(target-start)*30/np.asarray(self.env.base.config.max_velocity_deg_s)))))
        self.queue.extend([start+(target-start)*fraction for fraction in np.linspace(0,1,count+1)[1:]])
    def predict(self,_observation,deterministic=True):
        base=self.env.base
        if not self.queue and self.stage=='carry':
            goal=self.last.copy();goal[5]=10
            self.extend(base.command,goal);self.queue.extend([goal.copy() for _ in range(12)])
            self.stage='release'
        elif not self.queue and self.stage=='release':
            q=np.asarray(base.actual['measured_joint_deg']).copy()
            q[5]=base.command[5]
            point=opening_center(self.spec,self.anchors,q,-q[5]);up=np.array([0.,0,1])@rotation(self.spec,q).T
            kin=SimpleNamespace(specification=self.spec,limits=np.c_[base.low,base.high],
                point=lambda v:opening_center(self.spec,self.anchors,v,-v[5]))
            for fraction in np.linspace(0,1,9)[1:]:
                fitted=fit_upright(kin,q,point+[0,0,.045*fraction],up)
                self.extend(q,fitted);q=fitted
            self.last=q.copy();self.stage='retreat'
        goal=self.queue.pop(0) if self.queue else self.last
        return base.normalized_target(goal),None


def run_reference(env,entry,seed,path):
    from teacher.train_grasp import run_trial
    # run_trial restores the same checkpoint again, including all velocities.
    env.fixed_entry=entry;env.reset(seed=seed)
    return run_trial(env,PlaceReference(env),entry,seed,Path(path))
