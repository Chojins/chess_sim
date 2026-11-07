"""Check complete paired camera/state/action records from opening probes."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from verify_residual_pilot import check_snapshot, read_json, read_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    root = args.directory
    manifest = read_json(root / 'manifest.json')
    results = read_json(root / 'results.json')
    assert [r['id'] for r in results] == [t['id'] for t in manifest['tasks']]
    total, observations = 0, 0
    for result, task in zip(results, manifest['tasks']):
        directory = root / result['id']
        initial = read_json(directory / 'initial.json')
        telemetry = initial['telemetry']
        geo = telemetry['observation']['task_geometry']
        assert geo['move'] == task['move'] and geo['fen_before'] == task['fen']
        assert geo['acting_robot'] == 'blue' and geo['version'] == 'legal_move_v1'
        mapping = telemetry['observation']['joint_map']
        limits = np.array(telemetry['observation']['joint_limits_deg'])
        rows = read_rows(directory / 'steps.jsonl')
        assert len(rows) == result['steps']
        frame, sim_time = telemetry['frame_id'], telemetry['simulation_time']
        state = telemetry['observation']['observation_state']
        if manifest['record_every_observation']:
            files = sorted((directory / 'observations').glob('*.npz'))
            assert [p.name for p in files] == [f'{i:06}.npz' for i in range(len(rows) + 1)]
            check_snapshot(files[0], state)
            observations += len(files)
        else:
            check_snapshot(directory / 'initial_images.npz', state)
        for i, row in enumerate(rows):
            assert row['step'] == i + 1 and row['frame_id'] == frame and row['next_frame_id'] == frame + 1
            np.testing.assert_allclose(row['state'], state, atol=1e-5)
            assert np.isfinite(np.r_[row['requested_action'], row['executed_action'], row['next_state'], row['reward']]).all()
            physical = np.array(mapping['sim_home_deg']) + np.array(mapping['scale']) * (
                np.array(row['executed_action']) - mapping['legacy_home'])
            assert np.all(physical >= limits[:, 0]) and np.all(physical <= limits[:, 1])
            np.testing.assert_allclose(row['simulation_time'] - sim_time, 1 / 30, atol=1e-6)
            np.testing.assert_allclose(row['reward'], sum(row['score']['reward_terms'].values()))
            assert bool(row['terminated'] or row['truncated']) == (i == len(rows) - 1)
            state, frame, sim_time = row['next_state'], row['next_frame_id'], row['simulation_time']
            if manifest['record_every_observation']:
                check_snapshot(files[i + 1], state)
        if not manifest['record_every_observation']:
            check_snapshot(directory / 'terminal_images.npz', state)
        assert rows[-1]['score'] == result['score']
        np.testing.assert_allclose(sum(r['reward'] for r in rows), result['return'])
        assert sum(r['limited'] for r in rows) == result['limited_commands']
        total += len(rows)
    with (Path(manifest['checkpoint']) / 'model.safetensors').open('rb') as stream:
        assert hashlib.file_digest(stream, 'sha256').hexdigest() == manifest['weights_sha256']
    report = {'passed': True, 'episodes': len(results), 'transitions': total,
              'paired_observations': observations, 'checkpoint_unchanged': True,
              'note': 'Record integrity only; failures are not expert demonstrations.'}
    (root / 'integrity.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report))


if __name__ == '__main__':
    main()
