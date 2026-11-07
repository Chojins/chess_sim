"""Independently verify lossless successful episodes before ACT conversion.

Uses saved physics observations to recompute the task outcome. Frames stay in
the source legacy calibration and contain executed, rather than proposed, actions.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from chess_task import LegalMoveTask, TaskConfig


def read_json(path):
    return json.loads(Path(path).read_text())


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def audit_episode(directory):
    directory = Path(directory)
    manifest = read_json(directory / 'episode.json')
    if not manifest.get('accepted_for_act_training') or manifest.get('error'):
        raise ValueError('Episode was not accepted by the collector')
    if manifest['format'] != 'chess_sim_lossless_episode_v1' or manifest['fps'] != 30:
        raise ValueError('Unsupported collection format')
    initial = read_json(directory / 'initial.json')
    if initial['task_version'] != LegalMoveTask.version or 'task_config' not in initial:
        raise ValueError('Complete move task and explicit scoring configuration are required')
    if initial != read_json(directory.parent / 'initial.json'):
        raise ValueError('Collection reset differs from evaluation reset')
    provenance = read_json(directory / 'provenance.json')
    if provenance != manifest['provenance'] or 'exploration disabled' not in provenance.get('policy_mode', ''):
        raise ValueError('Expected documented deterministic collection')
    if provenance['calibration'] != initial['telemetry']['observation']['joint_map']:
        raise ValueError('Collection calibration differs from reset telemetry')
    if provenance.get('residual_mode') == 'frozen actor' and not provenance.get('residual_actor_state_sha256'):
        raise ValueError('Missing frozen residual actor identity')
    rows = [json.loads(line) for line in (directory.parent / 'steps.jsonl').read_text().splitlines()]
    frames = sorted(directory.glob('frame_*.npz'))
    if len(rows) != len(frames) or len(rows) != manifest['frames'] or not rows:
        raise ValueError('Incomplete episode or frame count mismatch')
    trajectory = np.load(directory / 'trajectory.npz', allow_pickle=False)
    task = LegalMoveTask(initial['telemetry']['observation'], TaskConfig(**initial['task_config']))
    previous = initial['telemetry']
    file_hashes = {}
    try:
        for index, (row, path) in enumerate(zip(rows, frames)):
            if path.name != f'frame_{index:06d}.npz' or row['step'] != index + 1:
                raise ValueError('Frames or control steps are not contiguous')
            with np.load(path, allow_pickle=False) as frame:
                state = frame['observation_state']
                action = frame['action_legacy']
                np.testing.assert_allclose(state, previous['observation']['observation_state'], atol=1e-5)
                np.testing.assert_allclose(action, row['residual']['executed_legacy'], atol=1e-5)
                np.testing.assert_equal(state, trajectory['states'][index])
                np.testing.assert_equal(action, trajectory['actions'][index])
                np.testing.assert_equal(frame['timestamp'], trajectory['timestamps'][index])
                np.testing.assert_allclose(frame['timestamp'], previous['simulation_time'], atol=1e-6)
                np.testing.assert_allclose(row['simulation_time'] - frame['timestamp'], 1 / 30, atol=1e-6)
                if int(frame['frame_id']) != previous['frame_id'] or int(frame['next_frame_id']) != row['frame_id'] or row['frame_id'] != previous['frame_id'] + 1:
                    raise ValueError('Observation/action frame correspondence failed')
                for key in ('laptop_rgb', 'phone_rgb'):
                    if frame[key].shape != (480, 640, 3) or frame[key].dtype != np.uint8:
                        raise ValueError('Invalid lossless RGB camera frame')
                    if index == 0:
                        with np.load(directory.parent / 'initial_images.npz', allow_pickle=False) as images:
                            np.testing.assert_equal(frame[key], images[key])
                if not np.isfinite(np.r_[state, action]).all():
                    raise ValueError('Nonfinite training state/action')
            observation = row['placement_telemetry']
            _, terminated, truncated, score = task.update(observation)
            if score != row['score']:
                raise ValueError(f'Recomputed physics outcome differs at step {index + 1}')
            if bool(terminated or truncated) != (index == len(rows) - 1):
                raise ValueError('Premature or missing terminal transition')
            if bool(row['terminated']) != terminated or bool(row['truncated']) != truncated:
                raise ValueError('Terminal flags differ from physics outcome')
            previous = {'frame_id': row['frame_id'], 'simulation_time': row['simulation_time'],
                        'observation': {'observation_state': row['state']}}
            file_hashes[path.name] = digest(path)
    finally:
        trajectory.close()
    if score != manifest['score'] or score['reason'] != 'success' or not score['success']:
        raise ValueError('Collection is not an independently verified full-move success')
    if task.capture_index is not None and not score['capture']['upright_stable']:
        raise ValueError('Captured piece is not upright for subsequent board reset')
    for name in ('trajectory.npz', 'initial.json', 'provenance.json', 'episode.json'):
        file_hashes[name] = digest(directory / name)
    return {'path': str(directory.resolve()), 'frames': len(rows), 'move': task.geometry['move'],
            'capture': task.capture_index is not None, 'file_sha256': file_hashes,
            'trace_sha256': digest(directory.parent / 'steps.jsonl'), 'provenance': provenance}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--output', type=Path, required=True, help='New audited episode index, never overwrites')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output already exists')
    accepted, rejected = [], []
    for path in sorted(args.root.rglob('lossless_episode/episode.json')):
        if not read_json(path).get('accepted_for_act_training'):
            rejected.append(str(path.parent.resolve()))
            continue
        accepted.append(audit_episode(path.parent))
    if not accepted:
        raise ValueError('No independently verified complete episodes; refusing empty training index')
    report = {'format': 'chess_sim_audited_episodes_v1', 'episodes': accepted,
              'frames': sum(row['frames'] for row in accepted), 'rejected_episodes': rejected,
              'calibration_conversion': None, 'remote_upload': False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps({'accepted_episodes': len(accepted), 'frames': report['frames'],
                      'rejected_episodes': len(rejected), 'index': str(args.output)}))


if __name__ == '__main__':
    main()
