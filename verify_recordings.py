"""Check recorded simulation transitions for alignment and completeness."""
import argparse
import json
from pathlib import Path
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    checked = []
    for episode in sorted(args.run.glob('episode_*')):
        result = json.loads((episode / 'result.json').read_text())
        files = sorted(episode.glob('frame_*.npz'))
        if len(files) != result['steps']:
            raise ValueError(f'{episode}: frame count differs from result')
        last = None
        for file in files:
            with np.load(file, allow_pickle=False) as frame:
                for key in ['laptop_rgb', 'phone_rgb', 'red_raw_rgb']:
                    if frame[key].shape != (480, 640, 3) or frame[key].dtype != np.uint8:
                        raise ValueError(f'{file}: invalid {key}')
                for key in ['observation_state', 'next_observation_state', 'action_legacy']:
                    if frame[key].shape != (6,) or not np.isfinite(frame[key]).all():
                        raise ValueError(f'{file}: invalid {key}')
                assert int(frame['next_frame_id']) == int(frame['frame_id']) + 1
                np.testing.assert_allclose(frame['next_simulation_time'] - frame['simulation_time'], 1 / 30, atol=1e-6)
                if last:
                    assert last[0] == int(frame['frame_id'])
                    np.testing.assert_array_equal(last[1], frame['observation_state'])
                last = (int(frame['next_frame_id']), frame['next_observation_state'].copy())
        with np.load(episode / 'terminal.npz', allow_pickle=False) as terminal:
            meta = json.loads(str(terminal['observation_json']))
            if last:
                assert last[0] == meta['frame_id']
                np.testing.assert_allclose(last[1], meta['observation']['observation_state'], atol=1e-5)
        checked.append({'episode': episode.name, 'transitions': len(files), 'success': result['success']})
    if not checked:
        raise ValueError('No episodes found')
    (args.run / 'recording_validation.json').write_text(json.dumps(checked, indent=2))
    print('PASS', json.dumps(checked))


if __name__ == '__main__':
    main()
