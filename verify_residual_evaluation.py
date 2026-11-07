"""Audit a counterbalanced held-out residual comparison after completion."""
import argparse
import json
from pathlib import Path

import numpy as np

from evaluate_residual import digest, schedule, summarize
from verify_residual_pilot import check_row, check_snapshot, read_json, read_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    root = args.directory
    manifest = read_json(root / 'manifest.json')
    results = read_json(root / 'results.json')
    summary = read_json(root / 'summary.json')
    plan = schedule(manifest['seeds'], manifest['repeats'])
    assert plan == manifest['schedule']
    assert len(results) == len(plan) == summary['completed_trials'] == summary['planned_trials']
    assert [{key: row[key] for key in ('repeat', 'seed', 'label')} for row in results] == plan
    for key, value in summarize(results).items():
        assert summary[key] == value, key
    assert digest(manifest['residual_checkpoint']) == manifest['residual_sha256']
    assert digest(Path(manifest['checkpoint']) / 'model.safetensors') == manifest['weights_sha256']
    pilot = read_json(Path(manifest['pilot']) / 'manifest.json')
    assert not set(manifest['seeds']).intersection([pilot['training_seed'], *pilot['evaluation_seeds'], 11, 23, 37])
    offsets, starting_states, total = {}, {}, 0
    config = None
    for index, result in enumerate(results):
        assert result['order'] == index
        path = root / result['path']
        initial = read_json(path / 'initial.json')
        metadata = initial['telemetry']
        state = metadata['observation']
        settings = {key: state[key] for key in ('joint_map', 'joint_limits_deg', 'physics_settings', 'task_geometry')}
        if config is not None:
            assert settings == config, 'Simulator configuration changed during evaluation'
        config = settings
        offset = initial['reset_options']['pawn_offset_mm']
        if result['seed'] in offsets:
            np.testing.assert_equal(offset, offsets[result['seed']])
        offsets[result['seed']] = offset
        pair = (result['repeat'], result['seed'])
        if pair in starting_states:
            previous = starting_states[pair]
            # Audit actual reset state, not only the requested seed/offset.
            np.testing.assert_allclose(state['measured_joint_deg'], previous['measured_joint_deg'], atol=.05, rtol=0)
            np.testing.assert_allclose(state['pieces']['piece_centers_m'], previous['pieces']['piece_centers_m'], atol=.0001, rtol=0)
        starting_states[pair] = state
        frame, sim_time = metadata['frame_id'], metadata['simulation_time']
        check_snapshot(path / 'initial_images.npz', state['observation_state'])
        rows = read_rows(path / 'steps.jsonl')
        assert len(rows) == result['steps']
        for step, row in enumerate(rows):
            check_row(row, state)
            assert row['step'] == step + 1 and row['frame_id'] == frame + 1
            np.testing.assert_allclose(row['simulation_time'] - sim_time, 1 / 30, atol=1e-6)
            assert bool(row['terminated'] or row['truncated']) == (step == len(rows) - 1)
            if result['label'] == 'baseline':
                np.testing.assert_equal(row['action'], np.zeros(6))
            frame, sim_time = row['frame_id'], row['simulation_time']
        assert rows[-1]['score'] == result['score']
        np.testing.assert_allclose(sum(r['reward'] for r in rows), result['return'])
        check_snapshot(path / 'terminal_images.npz', rows[-1]['state'])
        total += len(rows)
    report = {'passed': True, 'episodes': len(results), 'transitions': total,
              'unique_seeds': len(offsets), 'repeats': manifest['repeats'],
              'counterbalanced_schedule': True, 'paired_reset_states_checked': True,
              'simulator_settings_constant': True, 'checkpoints_unchanged': True}
    (root / 'integrity.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report))


if __name__ == '__main__':
    main()
