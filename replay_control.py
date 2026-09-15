"""Replay recorded states through the same synchronous interface used by ACT."""
import argparse
import json
from pathlib import Path
import numpy as np
import cv2
from act_client import command
from camera_client import PairedCameraClient


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episode', type=Path, default=Path('outputs/replay/episode_000000.npz'))
    parser.add_argument('--repeat', type=int, default=2, help='Two 30 Hz steps per target gives half-speed replay')
    parser.add_argument('--output', type=Path, default=Path('outputs/control-replay'))
    parser.add_argument('--url', default='http://127.0.0.1:8080')
    args = parser.parse_args()
    if args.repeat < 1:
        parser.error('repeat must be positive')
    args.output.mkdir(parents=True, exist_ok=False)
    with np.load(args.episode) as data:
        states = data['states']
    client = PairedCameraClient(args.url)
    meta, _ = client.read()
    command(args.url, '/reset', meta['frame_id'])
    meta, images = client.read()
    initial = meta
    with (args.output / 'observations.jsonl').open('w') as stream:
        for frame, state in enumerate(states):
            for _ in range(args.repeat):
                response = command(args.url, '/step', meta['frame_id'], state)
                meta, images = client.read()
                if meta['frame_id'] != response['frame_id']:
                    raise RuntimeError('Observation changed outside this controller')
                stream.write(json.dumps({'recorded_frame': frame, **meta}) + '\n')
            if frame in [0, 175, 185, 200, 240, len(states)-1]:
                for name, image in images.items():
                    cv2.imwrite(str(args.output / f'{name}-{frame:04d}.png'), image)
            if frame % 100 == 0:
                print('REPLAY', frame, meta['observation'].get('pieces', {}).get('pawn_lift_max_m'), flush=True)
    pieces = meta['observation'].get('pieces', {})
    result = {'initial': initial, 'final': meta, 'repeat': args.repeat,
              'lifted_5mm': pieces.get('pawn_lift_max_m', 0) > 0.005,
              'placed_within_5mm': pieces.get('pawn_target_xy_error_m', 1) < 0.005}
    command(args.url, '/reset', meta['frame_id'])
    reset, _ = client.read()
    if pieces:
        error = np.abs(np.asarray(reset['observation']['pieces']['piece_centers_m']) -
                       np.asarray(initial['observation']['pieces']['piece_centers_m'])).max()
        result['reset_max_piece_coordinate_error_m'] = float(error)
        if error > 0.0003:
            raise RuntimeError(f'Piece reset error: {error} m')
    (args.output / 'result.json').write_text(json.dumps(result, indent=2))
    print('REPLAY_RESULT', result['lifted_5mm'], result['placed_within_5mm'], flush=True)


if __name__ == '__main__':
    main()
