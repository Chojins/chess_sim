"""Replay paired recorded pickups with an explicit close-before-lift hold.

This diagnostic does not train or modify ACT/SAC and uses no object-position
feedback to control motion. Piece telemetry remains the scoring target.
"""
import argparse
import html
import json
from pathlib import Path
import numpy as np
from capture_demonstrations import file_hash
from capture_pickup import CapturePickupTask
from capture_timing import settled_fingers
from chess_env import ChessEnv
from evaluation_video import EpisodeVideo
from grasp_geometry import GRASP_REFERENCE


def write_index(output, results):
    lines = ['<!doctype html><meta charset="utf-8"><title>Close before lift timing probe</title>',
             '<h1>Close before lift timing probe</h1>',
             '<p>Paired frozen recorded-command replay, not a trained policy. '
             'Grasp geometry and calibration are unchanged. The timing variant holds the recorded arm pose '
             'while the recorded jaw commands close, then resumes the arm path. '
             'Left: fixed camera; right: wrist camera; playback uses simulation time.</p>',
             '<table><tr><th>Source video</th><th>Variant</th><th>Outcome</th><th>Video</th></tr>']
    for row in results:
        link = f"{row['case_id']}/{row['label']}/cameras.mp4"
        lines.append(f"<tr><td>{row['video_number']}</td><td>{html.escape(row['label'])}</td>"
                     f"<td>{html.escape(row['score']['reason'])}</td><td><a href='{link}'>Watch video</a></td></tr>")
    lines.append('</table>')
    (output / 'videos.html').write_text('\n'.join(lines))


def run_episode(env, case, label, plan, output):
    path = output / case['id'] / label
    path.mkdir(parents=True, exist_ok=False)
    obs, initial = env.reset(options=case['reset_options'])
    current = initial['telemetry']['observation']
    if current['grasp_reference'] != GRASP_REFERENCE or current['joint_map'] != case['joint_map']:
        raise RuntimeError('Grasp/calibration differs from the recorded trace')
    if np.max(np.abs(np.array(current['measured_joint_deg']) - case['expected_initial_joint_deg'])) > 1.:
        raise RuntimeError('Initial joint pose differs from the source')
    env.task = CapturePickupTask(current, env.task_config)
    env.task.pickup_limit = 480
    initial['task_version'] = env.task.version
    (path / 'initial.json').write_text(json.dumps(initial, indent=2))
    np.savez_compressed(path / 'initial_images.npz', **obs)
    commands = np.array(case['commands_deg'])
    hold, close = case['hold_index'], case['close_index']
    video = EpisodeVideo(path, obs)
    video.write(obs)
    steps, reward_sum, done, settled, failure = 0, 0., False, False, None
    info = None
    maximum_arm_override, settling_steps = 0., 0
    try:
        with (path / 'steps.jsonl').open('x') as stream:
            def advance(q, phase, source_arm, source_jaw):
                nonlocal current, steps, reward_sum, done, info, obs, maximum_arm_override
                q = np.array(q, float)
                if np.any(q < np.array(current['joint_limits_deg'])[:, 0] - 1e-6) or np.any(q > np.array(current['joint_limits_deg'])[:, 1] + 1e-6):
                    raise RuntimeError('Recorded timing command exceeds physical limits')
                action = np.clip(env.legacy_home + (q - env.sim_home) / env.scale, env.action_space.low, env.action_space.high).astype(np.float32)
                obs, reward, terminated, truncated, info = env.step(action)
                current = info['telemetry']['observation']
                video.write(obs)
                steps += 1
                reward_sum += reward
                done = terminated or truncated
                maximum_arm_override = max(maximum_arm_override, float(np.max(np.abs(q[:5] - commands[source_arm, :5]))))
                stream.write(json.dumps({'step': steps, 'phase': phase, 'source_arm_step': source_arm + 1,
                                         'source_jaw_step': source_jaw + 1, 'commanded_joint_deg': q.tolist(),
                                         'action_legacy': action.tolist(), 'frame_id': info['telemetry']['frame_id'],
                                         'simulation_time': info['telemetry']['simulation_time'], 'reward': reward,
                                         'terminated': terminated, 'truncated': truncated, 'score': info['score'],
                                         'observation': current}) + '\n')
                stream.flush()
            if label == 'recorded':
                for i, q in enumerate(commands):
                    advance(q, 'recorded', i, i)
                    if done:
                        break
            else:
                for i in range(hold + 1):
                    advance(commands[i], 'approach', i, i)
                    if done:
                        break
                if not done:
                    for jaw_index in range(hold + 1, close + 1):
                        q = commands[hold].copy()
                        q[5] = commands[jaw_index, 5]
                        advance(q, 'hold_arm_close_jaw', hold, jaw_index)
                        if done:
                            break
                streak = 0
                if not done:
                    q = commands[hold].copy()
                    q[5] = commands[close, 5]
                    for settling_steps in range(1, plan['maximum_settle_steps'] + 1):
                        previous = current
                        advance(q, 'settle_closed', hold, close)
                        ready = settled_fingers(current, previous, q)
                        streak = streak + 1 if ready else 0
                        if done:
                            break
                        if settling_steps >= plan['minimum_settle_steps'] and streak >= plan['stable_frames']:
                            settled = True
                            break
                    if not done and not settled:
                        failure = 'finger_or_arm_settle_timeout'
                if settled and not done:
                    for arm_index in range(hold + 1, len(commands)):
                        q = commands[arm_index].copy()
                        jaw_index = max(arm_index, close)
                        q[5] = commands[jaw_index, 5]
                        advance(q, 'resume_arm_closed' if arm_index <= close else 'recorded', arm_index, jaw_index)
                        if done:
                            break
            if not done:
                failure = failure or 'trace_end_without_pickup'
    finally:
        video.close()
    if info is None:
        raise RuntimeError('No replayed transitions')
    score = dict(info['score'])
    if failure:
        score.update(success=False, reason=failure)
    result = {'case_id': case['id'], 'video_number': case['video_number'], 'label': label,
              'steps': steps, 'return': reward_sum, 'score': score,
              'closed_settled_before_lift': settled, 'settling_steps': settling_steps,
              'maximum_arm_change_from_recorded_pose_deg': maximum_arm_override,
              'video': video.verify()}
    assert maximum_arm_override < 1e-9
    assert result['video']['frames'] == steps + 1
    (path / 'result.json').write_text(json.dumps(result, indent=2))
    np.savez_compressed(path / 'terminal_images.npz', **obs)
    print('TIMING_PROBE_EPISODE', json.dumps(result), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads((args.output / 'plan.json').read_text())
    manifest_path = args.output / 'manifest.json'
    if manifest_path.exists():
        raise FileExistsError('Timing probe was already started')
    def check_sources():
        for case in plan['cases']:
            for name, expected in case['source_hashes'].items():
                if file_hash(Path(case['source']) / name) != expected:
                    raise RuntimeError('Source trace changed')
    check_sources()
    manifest = {'status': 'running', 'objective': plan['objective'], 'rl_training': False,
                'geometry_changed': False, 'plan_sha256': file_hash(args.output / 'plan.json'),
                'schedule': 'one matched pair per source, order alternated; diagnostic replay not a reliability estimate',
                'expected_episodes': 2 * len(plan['cases']), 'cases': [case['id'] for case in plan['cases']]}
    manifest_path.write_text(json.dumps(manifest, indent=2))
    results = []
    write_index(args.output, results)
    env = ChessEnv(max_steps=480, pawn_jitter_mm=0.)
    try:
        for i, case in enumerate(plan['cases']):
            for label in (['recorded', 'close_before_lift'] if i % 2 == 0 else ['close_before_lift', 'recorded']):
                result = run_episode(env, case, label, plan, args.output)
                results.append(result)
                (args.output / 'results.json').write_text(json.dumps(results, indent=2))
                write_index(args.output, results)
        check_sources()
        summary = {label: {'successes': sum(r['score']['success'] for r in results if r['label'] == label),
                           'episodes': sum(r['label'] == label for r in results),
                           'outcomes': [r['score']['reason'] for r in results if r['label'] == label]}
                   for label in ('recorded', 'close_before_lift')}
        (args.output / 'summary.json').write_text(json.dumps(summary, indent=2))
        manifest['status'] = 'complete'
        print('TIMING_PROBE_RESULT', json.dumps(summary), flush=True)
    except Exception as error:
        manifest.update(status='failed', error=str(error))
        raise
    finally:
        manifest_path.write_text(json.dumps(manifest, indent=2))
        env.close()


if __name__ == '__main__':
    main()
