"""Frozen-policy comparison on unseen seeds with counterbalanced repeated resets."""
import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from stable_baselines3 import SAC

from act_client import LegacyACT
from chess_env import ChessEnv
from residual_learning import LearningResidualEnv
from train_residual import evaluate


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def schedule(seeds, repeats):
    if not seeds or len(set(seeds)) != len(seeds) or any(s < 0 for s in seeds) or repeats < 2:
        raise ValueError('Need unique nonnegative seeds and at least two repeats')
    result = []
    for repeat in range(repeats):
        for index, seed in enumerate(seeds):
            labels = ('baseline', 'learned') if (index + repeat) % 2 == 0 else ('learned', 'baseline')
            for label in labels:
                result.append({'repeat': repeat, 'seed': seed, 'label': label})
    return result


def summarize(results):
    report = {}
    for label in ('baseline', 'learned'):
        rows = [r for r in results if r['label'] == label]
        successes = sum(r['score']['success'] for r in rows)
        report[label] = {'episodes': len(rows), 'successes': successes,
                         'success_fraction': successes / len(rows) if rows else None,
                         'outcomes': dict(Counter(r['score']['reason'] for r in rows))}
    pairs = {}
    for row in results:
        pairs.setdefault((row['repeat'], row['seed']), {})[row['label']] = row['score']['success']
    counts = Counter()
    for values in pairs.values():
        if len(values) == 2:
            key = ('both_success' if values['baseline'] and values['learned'] else
                   'learned_only' if values['learned'] else
                   'baseline_only' if values['baseline'] else 'both_failure')
            counts[key] += 1
    report['matched_pairs'] = dict(counts)
    report['per_seed'] = [
        {'seed': seed, **{label: [r['score']['success'] for r in results
                                 if r['seed'] == seed and r['label'] == label]
                         for label in ('baseline', 'learned')}}
        for seed in sorted({r['seed'] for r in results})]
    report['seeds_with_repeat_disagreement'] = {
        label: sum(len(set(row[label])) > 1 for row in report['per_seed'])
        for label in ('baseline', 'learned')}
    report['note'] = 'Repeated trials share seeds; counts are descriptive, not independent reliability samples.'
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--pilot', type=Path, default=Path('outputs/residual-pilot-v2'))
    parser.add_argument('--seeds', nargs='+', type=int, default=list(range(201, 211)))
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--url', default='http://127.0.0.1:8080')
    args = parser.parse_args()
    plan = schedule(args.seeds, args.repeats)
    pilot = json.loads((args.pilot / 'manifest.json').read_text())
    used = {pilot['training_seed'], *pilot['evaluation_seeds'], 11, 23, 37}
    if used.intersection(args.seeds):
        parser.error('Use seeds not used for training or earlier evaluation')
    if args.device == 'cuda':
        assert torch.ones(16, device='cuda').sum().item() == 16
    policy = LegacyACT(pilot['checkpoint'], args.device, n_action_steps=pilot['action_window'])
    if policy.weights_sha256 != pilot['weights_sha256']:
        raise ValueError('ACT checkpoint differs from pilot')
    model_path = args.pilot / 'residual_final.zip'
    model_hash = digest(model_path)
    model = SAC.load(model_path, device=args.device)
    model.policy.set_training_mode(False)
    model.policy.requires_grad_(False)
    env = LearningResidualEnv(ChessEnv(args.url, pawn_jitter_mm=.5), policy,
                              gamma=pilot['gamma'], shaping_weight=pilot['potential_weight'])
    if asdict(env.unwrapped.task_config) != pilot['task_config']:
        raise ValueError('Task parameters differ from pilot')
    np.testing.assert_equal(env.env.bounds, pilot['residual_bounds_deg'])
    assert model.observation_space == env.observation_space and model.action_space == env.action_space
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = {'pilot': str(args.pilot.resolve()), 'checkpoint': policy.checkpoint,
                'weights_sha256': policy.weights_sha256, 'residual_checkpoint': str(model_path.resolve()),
                'residual_sha256': model_hash, 'seeds': args.seeds, 'repeats': args.repeats,
                'schedule': plan, 'pawn_jitter_mm': .5, 'training': False,
                'task_config': pilot['task_config'], 'action_window': pilot['action_window'],
                'residual_bounds_deg': pilot['residual_bounds_deg'],
                'device': args.device, 'torch': torch.__version__}
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    results = []
    started = time.perf_counter()
    try:
        for index, item in enumerate(plan):
            pair_path = args.output / f"repeat_{item['repeat']}" / f"seed_{item['seed']}"
            pair_path.mkdir(parents=True, exist_ok=True)
            t0 = time.perf_counter()
            row = evaluate(env, model if item['label'] == 'learned' else None,
                           [item['seed']], pair_path, item['label'])[0]
            row.update(repeat=item['repeat'], order=index, elapsed_seconds=time.perf_counter() - t0,
                       path=str((pair_path / f"{item['label']}_{item['seed']}").relative_to(args.output)))
            results.append(row)
            (args.output / 'results.json').write_text(json.dumps(results, indent=2))
            partial = summarize(results)
            partial['completed_trials'], partial['planned_trials'] = len(results), len(plan)
            (args.output / 'progress.json').write_text(json.dumps(partial, indent=2))
            print('PROGRESS', len(results), '/', len(plan), json.dumps({k: partial[k] for k in ('baseline', 'learned')}), flush=True)
        assert digest(model_path) == model_hash
        assert digest(Path(policy.checkpoint) / 'model.safetensors') == policy.weights_sha256
        final = summarize(results)
        final.update(completed_trials=len(results), planned_trials=len(plan),
                     elapsed_seconds=time.perf_counter() - started, checkpoints_unchanged=True)
        (args.output / 'summary.json').write_text(json.dumps(final, indent=2))
        print('EVALUATION_COMPLETE', json.dumps(final), flush=True)
    except Exception as error:
        (args.output / 'error.json').write_text(json.dumps({'error': str(error), 'completed_trials': len(results)}, indent=2))
        raise
    finally:
        env.close()


if __name__ == '__main__':
    main()
