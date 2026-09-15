"""Validate a completed ACT window evaluation without running the simulator."""
import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    root = args.directory
    manifest = json.loads((root / 'manifest.json').read_text())
    results = json.loads((root / 'results.json').read_text())
    expected = {(s, w) for s in manifest['seeds'] for w in manifest['windows']}
    assert len(results) == len(expected)
    assert {(r['seed'], r['action_window']) for r in results} == expected
    offsets = {}
    total = 0
    for result in results:
        seed, window = result['seed'], result['action_window']
        path = root / f'seed_{seed}_window_{window}'
        initial = json.loads((path / 'initial.json').read_text())
        offset = initial['reset_options']['pawn_offset_mm']
        if seed in offsets:
            np.testing.assert_equal(offset, offsets[seed])
        offsets[seed] = offset
        rows = [json.loads(line) for line in (path / 'steps.jsonl').read_text().splitlines()]
        assert len(rows) == result['steps']
        previous = initial['telemetry']['observation']['observation_state']
        frame = initial['telemetry']['frame_id']
        simulation_time = initial['telemetry']['simulation_time']
        for i, row in enumerate(rows):
            assert row['step'] == i + 1
            assert row['frame_id'] == frame and row['next_frame_id'] == frame + 1
            np.testing.assert_allclose(row['state'], previous, atol=1e-5)
            assert np.isfinite(np.r_[row['state'], row['action_legacy'], row['next_state'], row['reward']]).all()
            assert len(row['action_legacy']) == 6
            np.testing.assert_allclose(row['simulation_time'] - simulation_time, 1 / 30, atol=1e-6)
            np.testing.assert_allclose(row['reward'], sum(row['score']['reward_terms'].values()))
            if i < len(rows) - 1:
                assert not row['terminated'] and not row['truncated']
            previous, frame, simulation_time = row['next_state'], row['next_frame_id'], row['simulation_time']
        np.testing.assert_allclose(sum(r['reward'] for r in rows), result['total_reward'])
        if not result['error']:
            assert rows[-1]['terminated'] or rows[-1]['truncated']
            assert rows[-1]['score']['reason'] == result['reason']
            assert rows[-1]['score']['success'] == result['success']
        with np.load(path / 'terminal_images.npz', allow_pickle=False) as data:
            np.testing.assert_allclose(data['state'], previous, atol=1e-5)
            for camera in ('laptop_rgb', 'phone_rgb'):
                assert data[camera].shape == (480, 640, 3) and data[camera].dtype == np.uint8
        total += len(rows)
    print(f'PASS: {len(results)} trials, {total} transitions, paired offsets, frame/state/time chains, rewards and terminal snapshots')


if __name__ == '__main__':
    main()
