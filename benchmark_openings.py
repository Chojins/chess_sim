"""ACT-only opening probes with optional full paired-frame episode recording."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from act_client import LegacyACT
from chess_env import ChessEnv, ResidualACTEnv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--curriculum', type=Path, default=Path('outputs/openings/curriculum.json'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--ids', nargs='+', default=['initial_e2e4', 'initial_d2d4', 'open_game_knight', 'scandinavian_capture'])
    parser.add_argument('--record', action='store_true', help='Save every observation as paired RGB/state NPZ')
    parser.add_argument('--device', default='cuda', choices=['cpu', 'cuda'])
    args = parser.parse_args()
    tasks = {t['id']: t for t in json.loads(args.curriculum.read_text())['tasks']}
    if len(set(args.ids)) != len(args.ids) or any(name not in tasks or not tasks[name]['executable'] for name in args.ids):
        parser.error('IDs must be unique executable blue tasks in the curriculum')
    policy = LegacyACT('/home/jacob/lerobot/outputs/train/chess_game_multi_stereo/checkpoints/last/pretrained_model', args.device)
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = {'tasks': [tasks[name] for name in args.ids], 'checkpoint': policy.checkpoint,
                'weights_sha256': policy.weights_sha256, 'controller': 'ACT, zero residual, explicit joint-limit projection',
                'record_every_observation': args.record, 'format': 'paired RGB/state NPZ + transition JSONL',
                'action_units': 'legacy absolute targets; executed_action after limit projection',
                'camera_inputs': 'laptop_rgb=highlighted red fixed camera; phone_rgb=raw blue wrist',
                'training_data_status': 'diagnostic; review successes before exporting to LeRobot',
                'logical_board_advancement': 'none; each case is an independent reset'}
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    results = []
    for name in args.ids:
        case = tasks[name]
        env = ResidualACTEnv(ChessEnv(max_steps=800 if case['capture'] else 400), policy)
        directory = args.output / name
        directory.mkdir()
        frame_dir = directory / 'observations'
        if args.record: frame_dir.mkdir()
        started = time.perf_counter()
        try:
            obs, initial = env.reset(seed=401, options={'history': case['history'], 'move': case['move']})
            (directory / 'initial.json').write_text(json.dumps(initial, indent=2))
            def save_observation(index, observation):
                data = {k: observation[k] for k in ('state', 'laptop_rgb', 'phone_rgb')}
                if args.record:
                    np.savez_compressed(frame_dir / f'{index:06}.npz', **data)
                elif index == 0:
                    np.savez_compressed(directory / 'initial_images.npz', **data)
            save_observation(0, obs)
            frame = initial['telemetry']['frame_id']
            total, limited, steps = 0., 0, 0
            with (directory / 'steps.jsonl').open('w') as stream:
                while True:
                    before = obs['state'].copy()
                    requested = obs['base_action'].copy()
                    obs, reward, terminated, truncated, info = env.step(np.zeros(6, np.float32))
                    steps += 1
                    total += reward
                    limited += info['residual']['limited']
                    save_observation(steps, obs)
                    next_frame = info['telemetry']['frame_id']
                    stream.write(json.dumps({'step': steps, 'frame_id': frame, 'next_frame_id': next_frame,
                        'state': before.tolist(), 'next_state': obs['state'].tolist(),
                        'requested_action': requested.tolist(), 'executed_action': info['residual']['executed_legacy'],
                        'limited': info['residual']['limited'], 'reward': reward,
                        'terminated': terminated, 'truncated': truncated, 'score': info['score'],
                        'simulation_time': info['telemetry']['simulation_time']}) + '\n')
                    frame = next_frame
                    if terminated or truncated: break
            if not args.record:
                np.savez_compressed(directory / 'terminal_images.npz', **{k: obs[k] for k in ('state', 'laptop_rgb', 'phone_rgb')})
            result = {'id': name, 'move': case['move'], 'steps': steps, 'return': total,
                      'limited_commands': limited, 'seconds': time.perf_counter() - started, 'score': info['score']}
            (directory / 'result.json').write_text(json.dumps(result, indent=2))
            results.append(result)
            (args.output / 'results.json').write_text(json.dumps(results, indent=2))
            print('OPENING_RESULT', json.dumps(result), flush=True)
        except Exception as error:
            (directory / 'error.json').write_text(json.dumps({'error': str(error)}, indent=2))
            raise
        finally:
            env.close()
    with (Path(policy.checkpoint) / 'model.safetensors').open('rb') as stream:
        assert hashlib.file_digest(stream, 'sha256').hexdigest() == policy.weights_sha256
    print('COMPLETE', len(results), 'episodes; ACT weights unchanged', flush=True)


if __name__ == '__main__':
    main()
