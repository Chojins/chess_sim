"""Close and vertically lift from the human-approved live simulation pose.

Privileged physics diagnostic, not a learned policy or full capture. Starts
without resetting the scene and leaves the last pose paused for inspection.
"""
import argparse
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np
from act_client import command
from camera_client import PairedCameraClient
from chess_task import LegalMoveTask
from evaluation_video import EpisodeVideo
from policy_overlay import highlight_move
from review_grasp_geometry import joint_transforms, opening_center


def vertical_waypoint(spec, anchors, start, target, limits):
    """Position IK with preserved finger axis and wrist roll for a short lift."""
    q = np.asarray(start, float).copy()
    axis = joint_transforms(spec, q)[spec['tool_link']][1, :3]
    def errors(candidate):
        frames = joint_transforms(spec, candidate)
        return np.r_[(opening_center(spec, anchors, candidate)-target)*1000,
                     50*(frames[spec['tool_link']][1, :3]-axis)]
    for _ in range(120):
        error = errors(q)
        if np.linalg.norm(error[:3]) < .05 and np.linalg.norm(error[3:]) < .01:
            break
        jac = np.stack([(errors(q+np.eye(6)[j]*.001)-error)/.001 for j in range(4)], axis=1)
        delta = -np.linalg.solve(jac.T@jac+.03*np.eye(4), jac.T@error)
        for fraction in (1., .5, .25, .125):
            candidate = q.copy()
            candidate[:4] = np.clip(q[:4]+np.clip(delta, -1, 1)*fraction,
                                     limits[:4, 0], limits[:4, 1])
            if np.linalg.norm(errors(candidate)) < np.linalg.norm(error):
                q = candidate
                break
        else:
            break
    if np.linalg.norm(errors(q)[:3]) > .3 or np.linalg.norm(errors(q)[3:]) > .2:
        raise RuntimeError('No sufficiently vertical lift waypoint within joint limits')
    return q


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--review', type=Path, default=Path('outputs/grasp-pose-review-v18'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    review = json.loads((args.review/'pose.json').read_text())
    reference = json.loads((args.review/'reference.json').read_text())
    spec = json.loads(Path('outputs/capture-supervised-kinematics-v10.json').read_text())
    stage_hash = hashlib.sha256(Path('Sim_Stage.usd').read_bytes()).hexdigest()
    if stage_hash != reference['usd_sha256'] or stage_hash != spec['usd_sha256']:
        raise ValueError('Geometry changed since pose approval')
    client = PairedCameraClient()
    metadata, images = client.read(color_order='rgb')
    actual = metadata['observation']
    if np.max(np.abs(np.asarray(actual['measured_joint_deg'])-review['measured_joint_deg'])) > .3:
        raise ValueError('Robot is no longer at the approved pose')
    victim = review['victim_index']
    anchors = reference['anchors_jaw_local_m']
    point = opening_center(spec, anchors, actual['measured_joint_deg'], actual['measured_finger_joint_deg']['jaw2_hinge'])
    if np.linalg.norm(point-review['target_m']) > .0005:
        raise ValueError('Opening midpoint no longer matches the approved target')
    args.output.mkdir(parents=True, exist_ok=False)
    initial = actual
    original = np.asarray(actual['pieces']['piece_centers_m'][victim])
    scoring = LegalMoveTask(initial)
    limits = np.asarray(actual['joint_limits_deg'])
    mapping = actual['joint_map']
    calibration = json.loads(Path('calibration/sim_board.json').read_text())['cameras']['red']
    def camera_pair():
        fixed = highlight_move(cv2.cvtColor(images['red'], cv2.COLOR_RGB2BGR), calibration, actual['task_geometry']['move'])
        return {'laptop_rgb': cv2.cvtColor(fixed, cv2.COLOR_BGR2RGB), 'phone_rgb': images['blue']}
    (args.output/'initial.json').write_text(json.dumps(metadata, indent=2))
    (args.output/'manifest.json').write_text(json.dumps({'objective': 'approved-pose close and 35 mm vertical lift with two-second hold',
        'privileged': True, 'rl_training': False, 'full_capture': False, 'review': str(args.review.resolve()),
        'grasp_reference': reference['grasp_reference'], 'usd_sha256': stage_hash}, indent=2))
    video = EpisodeVideo(args.output, camera_pair())
    video.write(camera_pair())
    records = []
    def snapshot(name):
        cv2.imwrite(str(args.output/f'{name}_camera_views.png'), cv2.cvtColor(np.concatenate([images['red'], images['blue']], axis=1), cv2.COLOR_RGB2BGR))
        (args.output/f'{name}_observation.json').write_text(json.dumps(actual, indent=2))
    def advance(q, phase):
        nonlocal metadata, images, actual, point
        if not np.isfinite(q).all() or np.any(q < limits[:, 0]) or np.any(q > limits[:, 1]):
            raise ValueError('Probe command outside authored limits')
        action = np.array(mapping['legacy_home'])+(q-mapping['sim_home_deg'])/mapping['scale']
        response = command(client.url, '/step', metadata['frame_id'], action)
        previous = metadata
        metadata, images = client.read(color_order='rgb')
        if metadata['frame_id'] != response['frame_id'] or metadata['frame_id'] != previous['frame_id']+1:
            raise RuntimeError('Concurrent controller or mismatched camera pair')
        np.testing.assert_allclose(metadata['simulation_time']-previous['simulation_time'], 1/30, atol=1e-6)
        actual = metadata['observation']
        point = opening_center(spec, anchors, actual['measured_joint_deg'], actual['measured_finger_joint_deg']['jaw2_hinge'])
        piece = np.asarray(actual['pieces']['piece_centers_m'][victim])
        disturbances = scoring._adapt(actual)['pieces']['other_pieces_displaced_over_3mm']
        record = {'step': len(records)+1, 'phase': phase, 'frame_id': metadata['frame_id'],
                  'simulation_time': metadata['simulation_time'], 'action_legacy': action.tolist(),
                  'commanded_joint_deg': q.tolist(), 'opening_center_m': point.tolist(),
                  'victim_lift_m': float(piece[2]-original[2]),
                  'victim_relative_to_opening_m': (piece-point).tolist(),
                  'other_pieces_disturbed': disturbances, 'observation': actual}
        records.append(record)
        stream.write(json.dumps(record)+'\n')
        video.write(camera_pair())
        if disturbances:
            raise RuntimeError('Protected piece left its square or tipped')
        source = np.asarray(actual['pieces']['piece_centers_m'][scoring.source_index])
        local = (source[:2]-scoring.original_square_centers[scoring.source_index]) @ scoring.square_inverse.T
        if np.max(np.abs(local)) > .5:
            raise RuntimeError('Attacker left its square before victim removal')
        return record
    snapshot('open')
    success = False
    try:
        with (args.output/'steps.jsonl').open('x') as stream:
            closed = np.array(review['commanded_joint_deg'])
            for t in np.linspace(0., 1., 37)[1:]:
                q = closed.copy()
                q[5] = 10.-12.5*t**3*(10.-15.*t+6.*t*t)
                advance(q, 'close')
            closed[5] = -2.5
            for _ in range(30):
                advance(closed, 'settle_closed')
            recent = [row['observation']['measured_finger_joint_deg'] for row in records[-6:]]
            for name in ('jaw1_hinge', 'jaw2_hinge'):
                if np.ptp([row[name] for row in recent]) > .2:
                    raise RuntimeError('Fingers have not settled; refusing to lift')
            if max(abs(actual['measured_finger_joint_deg'][name]) for name in ('jaw1_hinge', 'jaw2_hinge')) > 6.:
                raise RuntimeError('Fingers are still open; refusing to lift')
            snapshot('closed')
            start_q = np.asarray(actual['measured_joint_deg'])
            initial_relative = np.asarray(actual['pieces']['piece_centers_m'][victim])-point
            destination = point + [0, 0, .035]
            raised_q = vertical_waypoint(spec, anchors, start_q, destination, limits)
            sag = start_q[:5]-closed[:5]
            lift_command = closed.copy()
            for t in np.linspace(0., 1., 91)[1:]:
                blend = t**3*(10.-15.*t+6.*t*t)
                lift_command[:5] = start_q[:5]+blend*(raised_q[:5]-start_q[:5])-sag
                advance(lift_command, 'lift')
            for _ in range(60):
                advance(lift_command, 'hold_raised')
            snapshot('raised')
            held = [row for row in records if row['phase']=='hold_raised']
            final_lift = held[-1]['victim_lift_m']
            slip = max(np.linalg.norm(np.asarray(row['victim_relative_to_opening_m'])-initial_relative) for row in held)
            min_upright = min(row['observation']['pieces']['piece_upright_cosines'][victim] for row in held)
            success = (min(row['victim_lift_m'] for row in held) >= .030 and slip < .003 and min_upright > .95)
            result = {'success': bool(success), 'reason': 'stable_pickup' if success else 'unstable_or_missed_pickup',
                'objective': 'pickup diagnostic, not a complete capture', 'steps': len(records),
                'final_lift_mm': final_lift*1000, 'held_for_seconds': len(held)/30,
                'maximum_relative_slip_during_hold_mm': float(slip*1000),
                'minimum_upright_cosine_during_hold': float(min_upright),
                'measured_finger_joint_deg': actual['measured_finger_joint_deg']}
    except BaseException as error:
        (args.output/'error.json').write_text(json.dumps({'error': repr(error), 'steps': len(records)}, indent=2))
        raise
    finally:
        video.close()
    result['video'] = video.verify()
    (args.output/'result.json').write_text(json.dumps(result, indent=2))
    (args.output/'videos.html').write_text('<!doctype html><meta charset="utf-8"><h1>Approved grasp: close and lift</h1>'
        '<p>Privileged simulation diagnostic, not a learned policy or complete capture.</p>'
        '<video controls width="1200" src="cameras.mp4"></video><pre>'+json.dumps(result, indent=2)+'</pre>')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
