"""Place the OPEN simulated gripper at one pawn for human geometry review.

Privileged IK diagnostic only. No jaw closure, grasp test, learner update or
automatic restart. The simulator remains paused at the final measured pose.
"""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import numpy as np
import cv2
from chess_env import ChessEnv
from capture_demonstrations import JointKinematics
from grasp_geometry import GRASP_REFERENCE
from review_grasp_geometry import opening_center


def fit_pose(spec, anchors, nominal, target, limits):
    q = nominal.copy()
    lower = np.maximum(nominal[:5]-6., limits[:5, 0])
    upper = np.minimum(nominal[:5]+6., limits[:5, 1])
    for _ in range(100):
        point = opening_center(spec, anchors, q)
        error = (point-target)*1000
        if np.linalg.norm(error) < .05:
            break
        jac = np.stack([(opening_center(spec, anchors, q+np.eye(6)[j]*.001)-point)*1e6
                        for j in range(5)], axis=1)
        delta = -np.linalg.solve(jac.T@jac+.05*np.eye(5), jac.T@error)
        for fraction in (1., .5, .25, .125):
            candidate = q.copy()
            candidate[:5] = np.clip(q[:5]+np.clip(delta, -1., 1.)*fraction, lower, upper)
            if np.linalg.norm(opening_center(spec, anchors, candidate)-target) < np.linalg.norm(point-target):
                q = candidate
                break
        else:
            break
    if np.linalg.norm(opening_center(spec, anchors, q)-target) > .0003:
        raise RuntimeError('Cannot centre the opening within the existing six-degree correction bounds')
    return q, lower, upper


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source', type=Path, default=Path('outputs/capture-complete-replay-rl-v17/teacher_replay/scandinavian_capture/teacher_3104'))
    args = parser.parse_args()
    if (args.output/'pose.json').exists():
        raise ValueError('This pose review has already been run')
    spec = json.loads(Path('outputs/capture-supervised-kinematics-v10.json').read_text())
    reference = json.loads((args.output/'reference.json').read_text())
    stage_hash = hashlib.sha256(Path('Sim_Stage.usd').read_bytes()).hexdigest()
    if stage_hash != spec['usd_sha256'] or stage_hash != reference['usd_sha256'] or reference['grasp_reference'] != GRASP_REFERENCE:
        raise ValueError('Kinematics, meshes or grasp reference changed')
    anchors = reference['anchors_jaw_local_m']
    source_initial = json.loads((args.source/'initial.json').read_text())
    rows = [json.loads(line) for line in (args.source/'steps.jsonl').read_text().splitlines()]
    hold = next(row for row in rows if row['act_timing']['phase'] == 'arm_settle')
    env = ChessEnv(max_steps=600, pawn_jitter_mm=0.)
    env.task_config = replace(env.task_config, capture_home_retry=True)
    try:
        obs, initial = env.reset(options=source_initial['reset_options'])
        actual = initial['telemetry']['observation']
        if actual['grasp_reference'] != GRASP_REFERENCE:
            raise ValueError('Simulator grasp reference differs')
        mapping = actual['joint_map']
        limits = np.asarray(actual['joint_limits_deg'])
        np.testing.assert_allclose(limits, spec['limits'])
        victim = actual['task_geometry']['capture_index']
        if victim is None:
            raise ValueError('Expected a capture victim')
        target = np.asarray(actual['pieces']['piece_grasp_targets_m'][victim])
        nominal = np.array(mapping['sim_home_deg']) + np.array(mapping['scale'])*(np.array(hold['act_timing']['act_base_legacy'])-mapping['legacy_home'])
        nominal[5] = 10.
        fitted, lower, upper = fit_pose(spec, anchors, nominal, target, limits)
        (args.output/'initial.json').write_text(json.dumps(initial, indent=2))
        trace = (args.output/'position_steps.jsonl').open('x')
        def advance(q, phase):
            nonlocal obs, actual
            if np.any(q < limits[:, 0]) or np.any(q > limits[:, 1]) or abs(q[5]-10.) > 1e-8:
                raise ValueError('Inspection command must be in limits with the gripper open')
            legacy = env.legacy_home + (q-env.sim_home)/env.scale
            obs, _, terminated, truncated, info = env.step(np.asarray(legacy, np.float32))
            actual = info['telemetry']['observation']
            point = opening_center(spec, anchors, actual['measured_joint_deg'], actual['measured_finger_joint_deg']['jaw2_hinge'])
            np.testing.assert_allclose(JointKinematics(spec).point(actual['measured_joint_deg']), actual['gripper_grasp_center_m'], atol=1e-6)
            trace.write(json.dumps({'phase': phase, 'frame_id': info['telemetry']['frame_id'],
                'command_deg': q.tolist(), 'measured_joint_deg': actual['measured_joint_deg'],
                'opening_center_m': point.tolist(), 'error_mm': ((point-target)*1000).tolist(),
                'victim_center_m': actual['pieces']['piece_centers_m'][victim], 'score': info['score']})+'\n')
            if terminated or truncated:
                raise RuntimeError('Inspection interrupted by task outcome: '+str(info['score']['reason']))
            return point
        try:
            for row in rows[:hold['step']-1]:
                q = env.sim_home + env.scale*(np.array(row['residual']['executed_legacy'])-env.legacy_home)
                q[5] = 10.
                advance(q, 'open_approach')
            start = np.array(actual['measured_joint_deg']); start[5] = 10.
            for t in np.linspace(0., 1., 61)[1:]:
                blend = t**3*(10.-15.*t+6.*t*t)
                advance(start+(fitted-start)*blend, 'centre_opening')
            command = fitted.copy()
            streak = 0
            for _ in range(60):
                point = advance(command, 'settle_open')
                measured = np.asarray(actual['measured_joint_deg'])
                fingers = actual['measured_finger_joint_deg']
                ready = (np.linalg.norm(point-target) < .0005 and fingers['jaw1_hinge'] >= 8.
                         and fingers['jaw2_hinge'] <= -8.)
                streak = streak+1 if ready else 0
                if streak >= 8:
                    break
                # Compensate measured drive sag, inside the same small joint
                # box, without changing robot calibration or moving the pawn.
                command[:5] = np.clip(command[:5]+.5*(fitted[:5]-measured[:5]), lower, upper)
            else:
                raise RuntimeError('Measured open grasp did not settle within 0.5 mm')
        finally:
            trace.close()
        displacement = np.linalg.norm(np.array(actual['pieces']['piece_centers_m'][victim])-initial['telemetry']['observation']['pieces']['piece_centers_m'][victim])
        if displacement > .001 or actual['pieces']['piece_upright_cosines'][victim] < .999:
            raise RuntimeError('Pawn moved or tilted during the open-gripper inspection')
        report = {'mode': 'privileged open-pose inspection; no closing or training',
            'victim_index': victim, 'victim_path': actual['pieces']['piece_paths'][victim],
            'target_m': target.tolist(), 'opening_center_m': point.tolist(),
            'opening_center_error_mm': ((point-target)*1000).tolist(),
            'opening_center_error_norm_mm': float(np.linalg.norm(point-target)*1000),
            'target_height_above_pawn_bottom_mm': 17., 'finger_setback_mm': 7.,
            'nominal_joint_deg': nominal.tolist(), 'fitted_joint_deg': fitted.tolist(),
            'commanded_joint_deg': command.tolist(), 'measured_joint_deg': actual['measured_joint_deg'],
            'measured_finger_joint_deg': actual['measured_finger_joint_deg'],
            'pawn_displacement_mm': float(displacement*1000), 'grasp_reference': GRASP_REFERENCE,
            'source': str(args.source.resolve()), 'simulator_left_paused': True}
        (args.output/'pose.json').write_text(json.dumps(report, indent=2))
        (args.output/'final_observation.json').write_text(json.dumps(actual, indent=2))
        for name, rgb in env.images.items():
            cv2.imwrite(str(args.output/f'{name}_camera.png'), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(args.output/'camera_views.png'), cv2.cvtColor(np.concatenate([env.images['red'], env.images['blue']], axis=1), cv2.COLOR_RGB2BGR))
        (args.output/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Open grasp review</title>'
            '<style>body{font:17px sans-serif;max-width:1500px;margin:2em auto}img{max-width:100%}</style>'
            '<h1>Open gripper centred on the capture pawn</h1><p>17 mm above the pawn bottom; material reference 7 mm behind the finger front. '
            'This is a privileged pose inspection, not a learned grasp or contact test.</p>'
            '<h2>Actual camera close-ups</h2><img src="camera_closeups.png">'
            '<h2>Full camera frames: fixed red camera, then blue wrist camera</h2><img src="camera_views.png">'
            '<h2>Measured CAD geometry, not camera imagery</h2><img src="geometry_views.png">'
            '<p>The cyan cross is the opposing-surface midpoint at the current opening. Black circle: target. '
            'Gold cross: the existing fixed zero-angle reference.</p><a href="pose.json">Measured pose and errors</a>')
        print(json.dumps(report), flush=True)
    finally:
        env.close()


if __name__ == '__main__':
    main()
