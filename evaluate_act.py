"""Paired-seed ACT execution-window evaluation; saves diagnostic traces.

Requires a running updated --contacts --control server. This does not train
ACT or save expert demonstrations. A single process owns all control steps.
"""
import argparse
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
import random
import time

import numpy as np

from act_client import LegacyACT
from chess_env import ChessEnv


def summarize(results):
    summary = {}
    for window in sorted({r['action_window'] for r in results}):
        rows = [r for r in results if r['action_window'] == window]
        successes = sum(r['success'] for r in rows)
        elapsed = sum(r['elapsed_seconds'] for r in rows)
        summary[str(window)] = {'episodes': len(rows), 'successes': successes,
            'success_rate': successes / len(rows), 'outcomes': dict(Counter(r['reason'] for r in rows)),
            'steps_per_wall_second': sum(r['steps'] for r in rows) / max(elapsed, 1e-9),
            'successful_target_errors_mm': [1000 * r['score']['target_error_m'] for r in rows if r['success']]}
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=Path('/home/jacob/lerobot/outputs/train/chess_game_multi_stereo/checkpoints/last/pretrained_model'))
    parser.add_argument('--url', default='http://127.0.0.1:8080')
    parser.add_argument('--windows', nargs='+', type=int, default=[100, 25, 10])
    parser.add_argument('--seeds', nargs='+', type=int, default=[11, 23, 37])
    parser.add_argument('--pawn-jitter-mm', type=float, default=.5)
    parser.add_argument('--max-steps', type=int, default=400)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if len(set(args.windows)) != len(args.windows) or any(w < 1 for w in args.windows):
        parser.error('Action windows must be unique positive integers')
    if len(set(args.seeds)) != len(args.seeds) or any(s < 0 for s in args.seeds):
        parser.error('Seeds must be unique nonnegative integers')
    policy = LegacyACT(args.checkpoint, args.device)
    if max(args.windows) > policy.config['chunk_size'] or policy.config.get('temporal_ensemble_coeff') is not None:
        parser.error('Windows must fit checkpoint chunks; temporal ensembling is not supported here')
    env = ChessEnv(args.url, max_steps=args.max_steps, pawn_jitter_mm=args.pawn_jitter_mm)
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = {'format': 'chess_evaluation_v1', 'task_version': 'e2e4_v1',
        'task_config': asdict(env.task_config), 'weights_sha256': policy.weights_sha256,
        'checkpoint': policy.checkpoint, 'policy_config': policy.config.copy(),
        'windows': args.windows, 'seeds': args.seeds, 'pawn_jitter_mm': args.pawn_jitter_mm,
        'device': args.device, 'note': 'Paired requested offsets; renderer/PhysX bitwise determinism is not guaranteed. Diagnostic data, not expert demonstrations.'}
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    results = []
    try:
        for seed_index, seed in enumerate(args.seeds):
            # Rotate order to reduce confounding action window with run order.
            windows = args.windows[seed_index % len(args.windows):] + args.windows[:seed_index % len(args.windows)]
            for window in windows:
                random.seed(seed)
                np.random.seed(seed)
                policy.torch.manual_seed(seed)
                policy.policy.config.n_action_steps = window
                policy.config['n_action_steps'] = window
                policy.reset()
                path = args.output / f'seed_{seed}_window_{window}'
                path.mkdir()
                obs, info = env.reset(seed=seed)
                (path / 'initial.json').write_text(json.dumps(info, indent=2))
                np.savez_compressed(path / 'initial_images.npz', **obs)
                started = time.perf_counter()
                steps, total_reward, score = 0, 0., None
                reason, error = None, None
                print('TRIAL_START', seed, window, info['reset_options'], flush=True)
                with (path / 'steps.jsonl').open('w') as log:
                    for _ in range(args.max_steps):
                        try:
                            inference_start = time.perf_counter()
                            action = policy.action(obs['state'], obs['laptop_rgb'], obs['phone_rgb'])
                            inference_seconds = time.perf_counter() - inference_start
                            before = env.latest
                            next_obs, reward, terminated, truncated, info = env.step(action)
                        except (RuntimeError, ValueError, OSError) as exception:
                            error = str(exception)
                            reason = 'invalid_action' if isinstance(exception, ValueError) else 'infrastructure_error'
                            break
                        steps += 1
                        total_reward += reward
                        score = info['score']
                        log.write(json.dumps({'step': steps, 'frame_id': before['frame_id'],
                            'next_frame_id': env.latest['frame_id'], 'state': obs['state'].tolist(),
                            'action_legacy': action.tolist(), 'next_state': next_obs['state'].tolist(),
                            'simulation_time': env.latest['simulation_time'], 'inference_seconds': inference_seconds,
                            'reward': reward, 'terminated': terminated, 'truncated': truncated, 'score': score}) + '\n')
                        obs = next_obs
                        if terminated or truncated:
                            reason = score['reason']
                            break
                result = {'seed': seed, 'action_window': window, 'steps': steps,
                    'success': bool(score and score['success'] and not error), 'reason': reason,
                    'error': error, 'score': score, 'total_reward': total_reward,
                    'elapsed_seconds': time.perf_counter() - started}
                results.append(result)
                np.savez_compressed(path / 'terminal_images.npz', **obs)
                (path / 'terminal.json').write_text(json.dumps(env.latest, indent=2))
                (path / 'result.json').write_text(json.dumps(result, indent=2))
                (args.output / 'results.json').write_text(json.dumps(results, indent=2))
                (args.output / 'summary.json').write_text(json.dumps(summarize(results), indent=2))
                print('TRIAL_RESULT', json.dumps(result), flush=True)
                if reason == 'infrastructure_error':
                    raise RuntimeError(error)
    finally:
        env.close()


if __name__ == '__main__':
    main()
