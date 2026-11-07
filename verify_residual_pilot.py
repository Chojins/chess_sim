"""Audit a finished residual experiment without touching the simulator."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def read_json(path):
    return json.loads(path.read_text())


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def check_row(row, initial_state=None, previous_applied=None, residual_bounds=None, filter_config=None):
    raw = np.asarray(row['action'])
    applied = np.asarray(row['learning']['applied_residual'])
    assert raw.shape == applied.shape == (6,)
    assert np.isfinite(np.r_[raw, applied, row['state'], row['reward']]).all()
    assert np.all(np.abs(raw) <= 1.000001)
    if row['learning'].get('residual_paused'):
        assert (row.get('act_timing') or {}).get('capture_transport')
        np.testing.assert_allclose(applied,previous_applied,atol=1e-7)
    elif filter_config is not None:
        assert np.all(np.abs(applied) <= 1.000001)
        previous = np.zeros(6) if previous_applied is None else np.asarray(previous_applied)
        change = filter_config['alpha'] * (raw * row['learning']['correction_gate'] - previous)
        if filter_config['max_change_deg'] is not None:
            rate = np.asarray(filter_config['max_change_deg']) / residual_bounds
            change = np.clip(change, -rate, rate)
            assert np.all(np.abs(applied - previous) * residual_bounds <= np.asarray(filter_config['max_change_deg']) + 1e-6)
        np.testing.assert_allclose(applied, np.clip(previous + change, -1., 1.), atol=1e-6)
    else:
        assert np.all(np.abs(applied) <= np.abs(raw) + 1e-6)
    penalty = row['residual']['penalty']
    np.testing.assert_allclose(penalty, .01 * np.mean(applied ** 2), atol=1e-8)
    task_reward = sum(row['score']['reward_terms'].values())
    np.testing.assert_allclose(row['learning']['task_reward'], task_reward, atol=1e-7)
    np.testing.assert_allclose(row['reward'], task_reward - penalty + row['learning']['shaping_reward'] +
                              row['learning'].get('closing_alignment_penalty', 0.), atol=1e-6)
    if initial_state is not None:
        mapping = initial_state['joint_map']
        physical = np.array(mapping['sim_home_deg']) + np.array(mapping['scale']) * (
            np.array(row['residual']['executed_legacy']) - mapping['legacy_home'])
        limits = np.array(initial_state['joint_limits_deg'])
        assert np.isfinite(physical).all()
        assert np.all(physical >= limits[:, 0]) and np.all(physical <= limits[:, 1])


def check_snapshot(path, state):
    with np.load(path, allow_pickle=False) as data:
        np.testing.assert_allclose(data['state'], state, atol=1e-5)
        for key in ('laptop_rgb', 'phone_rgb'):
            assert data[key].shape == (480, 640, 3) and data[key].dtype == np.uint8


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    root = args.directory
    manifest, summary = read_json(root / 'manifest.json'), read_json(root / 'summary.json')
    offsets, total = {}, 0
    for label in ('baseline', 'learned'):
        results = read_json(root / f'{label}_results.json')
        assert [r['seed'] for r in results] == manifest['evaluation_seeds']
        assert summary[f'{label}_successes'] == sum(r['score']['success'] for r in results)
        for result in results:
            seed = result['seed']
            path = root / f'{label}_{seed}'
            initial = read_json(path / 'initial.json')
            offset = initial['reset_options']['pawn_offset_mm']
            if seed in offsets:
                np.testing.assert_equal(offset, offsets[seed])
            offsets[seed] = offset
            telemetry = initial['telemetry']
            frame, sim_time = telemetry['frame_id'], telemetry['simulation_time']
            check_snapshot(path / 'initial_images.npz', telemetry['observation']['observation_state'])
            rows = read_rows(path / 'steps.jsonl')
            assert len(rows) == result['steps']
            for i, row in enumerate(rows):
                check_row(row, telemetry['observation'])
                assert row['step'] == i + 1 and row['frame_id'] == frame + 1
                np.testing.assert_allclose(row['simulation_time'] - sim_time, 1 / 30, atol=1e-6)
                assert bool(row['terminated'] or row['truncated']) == (i == len(rows) - 1)
                if label == 'baseline':
                    np.testing.assert_equal(row['action'], np.zeros(6))
                frame, sim_time = row['frame_id'], row['simulation_time']
            assert rows[-1]['score'] == result['score']
            np.testing.assert_allclose(sum(r['reward'] for r in rows), result['return'])
            check_snapshot(path / 'terminal_images.npz', rows[-1]['state'])
            total += len(rows)
    training = read_rows(root / 'training_steps.jsonl')
    assert len(training) == manifest['steps'] == summary['training_steps']
    for i, row in enumerate(training):
        check_row(row, telemetry['observation'])
        assert row['step'] == i + 1
        assert row['done'] == (row['score']['reason'] is not None)
        if i and not training[i - 1]['done']:
            assert row['frame_id'] == training[i - 1]['frame_id'] + 1
            np.testing.assert_allclose(row['simulation_time'] - training[i - 1]['simulation_time'], 1 / 30, atol=1e-6)
    assert summary['gradient_updates'] > 0
    assert (root / 'residual_final.zip').stat().st_size > 0
    with (Path(manifest['checkpoint']) / 'model.safetensors').open('rb') as stream:
        assert hashlib.file_digest(stream, 'sha256').hexdigest() == manifest['weights_sha256']
    import torch
    from stable_baselines3 import SAC
    torch.set_num_threads(2)
    original = SAC.load(root / 'residual_initial', device='cpu')
    learned = SAC.load(root / 'residual_final', device='cpu')
    assert all(torch.isfinite(p).all() for p in learned.policy.parameters())
    assert any(not torch.equal(a, b) for a, b in zip(original.actor.parameters(), learned.actor.parameters()))
    report = {'passed': True, 'evaluation_trials': 2 * len(offsets),
              'evaluation_transitions': total, 'training_transitions': len(training),
              'act_checkpoint_unchanged': True, 'residual_actor_updated': True,
              'saved_policy_parameters_finite': True}
    (root / 'integrity.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report))


if __name__ == '__main__':
    main()
