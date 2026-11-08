"""Training reference with joint-space reach/return and physical finger feedback.

Global grasp IK avoids forcing the opening along a Cartesian line through the
camera/servo obstacle. This remains an analytic aid, not a learned policy.
"""
import numpy as np
from scipy.optimize import least_squares
from teacher.reactive_oracle import ReactiveOracle,smoothstep


class ConfigurationReference(ReactiveOracle):
    version='configuration_feedback_reference_v1'

    def __init__(self,controller,config=None):
        super().__init__(controller.config,config);self.controller=controller
        self.version=type(self).version

    def actions(self,observations):
        x=np.atleast_2d(np.asarray(observations,dtype=float));actions=super().actions(x)
        c=self.config;control=self.controller
        for index,row in enumerate(x):
            q=row[:6]*180;command=q+row[23:29]*5;home=q[:5]+row[62:67]*90
            fingers={'jaw1_hinge':row[21]*10,'jaw2_hinge':row[22]*10}
            grasp=row[12:15]*.03;goal=row[18:21]*.1;height=row[17]*.03
            f1,f2=fingers['jaw1_hinge'],-fingers['jaw2_hinge']
            held=(command[5]<-2.5 and max(f1,f2)<3.5 and f1>command[5]+.8 and f2>command[5]+.8 and np.linalg.norm(grasp)<.012)
            placed=np.linalg.norm(goal[:2])<.003 and abs(height)<.003
            placed|=np.linalg.norm(goal[:2])<c.home_basin_m and abs(height)<.003 and min(f1,f2)>=4.
            target=control.command(command,{'measured_finger_joint_deg':fingers},actions[index])
            if not held and not placed:
                point=control.point(q,fingers)+grasp
                lateral=np.linalg.norm(grasp[:2])
                rise=c.approach_height_m*smoothstep((lateral-c.funnel_inner_m)/(c.funnel_outer_m-c.funnel_inner_m))
                if actions[index,5]<0: rise=0.
                desired=np.r_[(point+[0,0,rise])/control.config.translation_step_m,
                    c.grip_pitch_deg/control.config.pitch_step_deg,c.grip_roll_deg/control.config.roll_step_deg]
                def residual(arm):
                    return control.coordinates(np.r_[arm,q[5]],fingers)-desired
                initial=np.clip(home,control.low[:5]+1e-7,control.high[:5]-1e-7)
                fit=least_squares(residual,initial,bounds=(control.low[:5],control.high[:5]),max_nfev=70,
                    ftol=1e-7,xtol=1e-7,gtol=1e-7)
                if np.max(np.abs(fit.fun))>.1:
                    raise ValueError('Grasp reference is outside the authored IK limits')
                delta=(fit.x-command[:5])/(30*c.tau_s)
                delta/=max(1.,float(np.max(np.abs(delta)/control.max_step[:5])))
                target[:5]=command[:5]+delta
            elif placed and min(f1,f2)>4.8:
                pheight=height+c.pawn_grasp_offset_m*row[70]-grasp[2]
                blend=float(smoothstep((pheight-.040)/(.070-.040)))
                delta=(home-command[:5])/(30*c.tau_s)
                delta/=max(1.,float(np.max(np.abs(delta)/control.max_step[:5])))
                target[:5]=(1-blend)*target[:5]+blend*(command[:5]+delta)
            else: continue
            converted=control.reference_action(command,{'measured_finger_joint_deg':fingers},target)
            converted[5]=actions[index,5]  # Preserve physical close/open intent.
            actions[index]=converted
        return actions
