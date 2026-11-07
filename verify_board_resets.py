"""Live checks of legal resets and capture setup without attempting a grasp."""
import json
from pathlib import Path

import chess
import numpy as np

from act_client import command
from chess_env import ChessEnv
from board_tasks import BoardLayout


def main():
    env = ChessEnv(max_steps=2)
    out = Path('outputs/openings/reset-verification')
    out.mkdir(parents=True, exist_ok=True)
    reports = []
    try:
        env.reset(options={'pawn_offset_mm': [0, 0]})
        initial = env.latest['observation']
        contacts = initial['physics_settings']['contacts']
        layout = BoardLayout(initial['pieces']['piece_paths'], initial['pieces']['piece_centers_m'],
                             contacts['square_centers_m'], contacts['board_top_m'], contacts['board_frame_bounds_m'])
        for name, history, move in [('standard', [], 'e2e4'), ('capture', ['e2e4', 'd7d5'], 'e4d5'),
                                    ('parked', ['e2e4', 'd7d5', 'e4d5', 'g8f6'], 'd2d4')]:
            desired, geometry = layout.plan(history, move)
            obs, info = env.reset(options={'history': history, 'move': move, 'pawn_offset_mm': [0, 0]})
            actual = info['telemetry']['observation']
            for key in ('move', 'history', 'fen_before', 'fen_after_expected', 'source_index', 'capture_index', 'acting_robot'):
                assert actual['task_geometry'][key] == geometry[key]
            centers = np.array(actual['pieces']['piece_centers_m'])
            # Physical settling may move centres slightly; wrong square/type must fail.
            np.testing.assert_allclose(centers, desired, atol=.002, rtol=0)
            assert actual['task_geometry']['capture_zone']['robot'] == 'blue'
            zone = geometry['capture_zone']
            np.testing.assert_allclose(np.array(zone['max_xy_m']) - zone['min_xy_m'], [.12, .15])
            frame = env.latest['frame_id']
            for bad in [{'history': [], 'move': 'e2e5'}, {'history': ['e2e4'], 'move': 'e7e5'}]:
                try:
                    command(env.url, '/reset', frame, options=bad)
                except RuntimeError as error:
                    assert 'Simulator rejected' in str(error)
                else:
                    raise AssertionError('Invalid/unvalidated reset accepted')
                assert env.client.read()[0]['frame_id'] == frame
            _, _, term, trunc, score = env.step(obs['state'])
            assert not term and not trunc and score['score']['other_pieces_disturbed'] == 0
            np.savez_compressed(out / f'{name}.npz', **obs)
            reports.append({'name': name, 'move': move, 'max_reset_error_m': float(np.abs(centers - desired).max()),
                            'capture_zone': geometry['capture_zone']})
        env.reset(options={'pawn_offset_mm': [0, 0]})
        assert env.task.version == 'e2e4_v1'
        (out / 'report.json').write_text(json.dumps({'passed': True, 'cases': reports}, indent=2))
        print('PASS: legal placement, captures, off-board parked pieces, robot-relative zones, invalid/reset rejection and legacy reset')
    finally:
        env.close()


if __name__ == '__main__':
    main()
