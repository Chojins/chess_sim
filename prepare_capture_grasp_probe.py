"""Prepare a bounded, recorded-trajectory physics probe with offline USD FK.

This is privileged diagnostic assistance, not a learned/deployable policy.
Run ./run_usd_python.sh prepare_capture_grasp_probe.py --output PATH.
"""
import argparse
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from pxr import Gf, Usd
from scipy.optimize import least_squares

from diagnose_capture_grasp import TRIALS
from episode_replay import EpisodeReplay
from grasp_geometry import GRASP_LOCAL_M, GRASP_REFERENCE


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    stage = Usd.Stage.Open('Sim_Stage.usd')
    stage.SetEditTarget(stage.GetSessionLayer())
    temporary = TemporaryDirectory(prefix='chess-grasp-plan-')
    fk = EpisodeReplay(stage, 'outputs/replay/episode_000000.npz',
                       'calibration/legacy_blue_joint_map_physics.json', temporary.name, kinematic=False)
    limits = np.array([[j[1].GetAttribute('physics:lowerLimit').Get(), j[1].GetAttribute('physics:upperLimit').Get()]
                       for j in fk.joints[:6]])
    def point(q):
        angles = dict(zip(fk.mapping.sim_names,q));angles['jaw2_hinge']=-q[5]
        return np.array(fk.forward(angles)[fk.root+'/gripper'].Transform(Gf.Vec3d(*GRASP_LOCAL_M)))

    cases = []
    for label in ('capture_success','capture_miss_high','capture_miss_back'):
        directory, closing_step = TRIALS[label]
        source = Path(directory)
        initial = json.loads((source/'initial.json').read_text())
        observation = initial['telemetry']['observation']
        rows = [json.loads(line) for line in (source/'steps.jsonl').read_text().splitlines()]
        executed = np.array([fk.mapping.to_sim_deg(row['residual']['executed_legacy']) for row in rows])
        raw_act = executed-np.array([row['learning']['applied_residual'] for row in rows])*[2,2,2,2,2,3]
        nominal = raw_act[closing_step-1].copy()
        target = np.array(rows[closing_step-1]['placement_telemetry']['pieces']['piece_grasp_targets_m'][observation['task_geometry']['capture_index']])
        lo = np.maximum(-6.,limits[:5,0]-nominal[:5]);hi = np.minimum(6.,limits[:5,1]-nominal[:5])
        def error(delta):
            q=nominal.copy();q[:5]+=delta
            return (point(q)-target)*1000
        fit=least_squares(error,np.zeros(5).clip(lo,hi),bounds=(lo,hi),xtol=1e-11,ftol=1e-11,gtol=1e-11)
        if np.linalg.norm(fit.fun)>1.:
            raise ValueError(f'{label}: no sufficiently centred candidate within 6 degrees')
        grasp=nominal.copy();grasp[:5]+=fit.x;grasp[5]=10.
        prefix_end=closing_step-20
        start=executed[prefix_end-1].copy()
        # Smooth joint interpolation avoids a jump to the fitted grasp. The
        # nominal ACT clock is paused for the explicit align/close phases.
        approach=[]
        for phase in np.linspace(0,1,61)[1:]:
            blend=phase**3*(10-15*phase+6*phase**2)
            q=start+(grasp-start)*blend
            approach.append(q.tolist())
        close=[]
        for phase in np.linspace(0,1,25)[1:]:
            blend=phase**3*(10-15*phase+6*phase**2)
            q=grasp.copy();q[5]=10+(-2.5-10)*blend
            close.append(q.tolist())
        # Continue the source ACT lift with the same fitted arm offset. Blend
        # into its already-raised waypoint instead of jumping past paused time.
        lift_waypoint=None
        for lift_index in range(closing_step+30,min(closing_step+90,len(raw_act))):
            candidate=raw_act[lift_index].copy();candidate[:5]+=fit.x;candidate[5]=-2.5
            if point(candidate)[2]-point(grasp)[2]>=.035:
                lift_waypoint=candidate
                break
        if lift_waypoint is None:
            raise ValueError(f'{label}: no sufficiently raised source ACT waypoint')
        closed=grasp.copy();closed[5]=-2.5
        lift=[]
        lift_steps=max(45,int(np.ceil(1.875*np.max(np.abs(lift_waypoint[:5]-closed[:5]))/.5)))
        for phase in np.linspace(0,1,lift_steps+1)[1:]:
            blend=phase**3*(10-15*phase+6*phase**2)
            q=closed+(lift_waypoint-closed)*blend
            lift.append(q.tolist())
        # Holding the raised waypoint isolates gripping/lifting from offboard
        # transport. It cannot count as a complete capture.
        max_lift_delta=float(np.max(np.abs(lift_waypoint[:5]-closed[:5])))
        proposed=np.array(approach+close+lift+[grasp.tolist(),lift_waypoint.tolist()])
        if np.any(proposed<limits[:,0]) or np.any(proposed>limits[:,1]):
            raise ValueError(f'{label}: probe exceeds authored joint limits; no clipping')
        predicted_lift=float(point(lift_waypoint)[2]-point(grasp)[2])
        if predicted_lift<.025:
            raise ValueError(f'{label}: source ACT continuation does not lift sufficiently: {predicted_lift}')
        # During grasp/hold correction is bounded relative to the chosen ACT
        # proposal. During continuation it is the same offset on a later ACT
        # proposal; an inserted smooth blend is an explicit trajectory probe.
        cases.append({'id':label,'source':directory,'source_steps_sha256':hashlib.sha256((source/'steps.jsonl').read_bytes()).hexdigest(),
                      'reset_options':initial['reset_options'],'expected_initial_joint_deg':observation['measured_joint_deg'],
                      'capture_index':observation['task_geometry']['capture_index'],'closing_sample_step':closing_step,
                      'recorded_commands_deg':executed[:240].tolist(),'prefix_commands_deg':executed[:prefix_end].tolist(),
                      'nominal_act_grasp_deg':nominal.tolist(),'correction_deg':fit.x.tolist(),
                      'fitted_grasp_deg':grasp.tolist(),'fit_error_mm':float(np.linalg.norm(fit.fun)),
                      'target_m':target.tolist(),'approach_deg':approach,'close_deg':close,'lift_deg':lift,
                      'raised_hold_deg':lift_waypoint.tolist(),'predicted_reference_lift_m':predicted_lift,
                      'largest_lift_joint_change_deg':max_lift_delta})
    plan={'version':'capture_grasp_physics_probe_v1','objective':'victim lift and stable upright hold, not full capture',
          'privileged':True,'source':'recorded ACT plus saved residual commands; no live ACT/SAC inference or training',
          'grasp_reference':GRASP_REFERENCE,'correction_bound_deg':6.,'open_jaw_deg':10.,'closed_jaw_deg':-2.5,
          'settle_position_tolerance_m':.003,'settle_joint_tolerance_deg':.6,'settle_frames':5,'settle_limit':60,
          'raised_hold_limit':45,'cases':cases}
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'replay_audit.json').write_text((Path(temporary.name)/'replay_audit.json').read_text())
    (args.output/'plan.json').write_text(json.dumps(plan,indent=2))
    temporary.cleanup()
    print(json.dumps({c['id']:{k:c[k] for k in ('correction_deg','fit_error_mm','predicted_reference_lift_m')} for c in cases},indent=2))


if __name__=='__main__':
    main()
