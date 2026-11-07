"""Collect lossless verified episodes from one frozen complete-move teacher.

Runs no gradients or exploration. Failed attempts remain labelled and only
independently re-scored complete upright moves enter the accepted index.
"""
import argparse
import json
from pathlib import Path
import time
import torch
from stable_baselines3 import SAC
from act_client import LegacyACT
from chess_task import LegalMoveTask
from evaluate_openings import audit_episode, environment_settings, restore_timing_policy
from evaluate_residual import digest
from evaluation_video import write_training_video_index
from grasp_geometry import GRASP_REFERENCE
from train_openings import OpeningChessEnv, OpeningLearningEnv
from train_residual import evaluate
from verify_collected_episodes import audit_episode as audit_collection


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, default=Path('residual_best.zip'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--episodes', type=int, default=40, help='Attempts, distributed across the selected tasks')
    parser.add_argument('--seed-start', type=int, required=True)
    parser.add_argument('--ids', nargs='+')
    parser.add_argument('--record-video', action='store_true')
    args = parser.parse_args()
    if args.episodes < 1 or args.seed_start < 0:
        parser.error('Episodes must be positive and seed start nonnegative')
    pilot = json.loads((args.pilot / 'manifest.json').read_text())
    pickup, settings = environment_settings(pilot)
    if pickup:
        parser.error('A pickup-only checkpoint cannot collect complete moves')
    tasks = {c['id']: c for c in pilot['cases']}
    ids = args.ids or list(tasks)
    if len(set(ids)) != len(ids) or any(name not in tasks for name in ids):
        parser.error('Select unique tasks from the trained pilot')
    cases = [tasks[name] for name in ids]
    seeds = list(range(args.seed_start, args.seed_start + args.episodes))
    used = set()
    for path in Path('outputs').glob('*/manifest.json'):
        prior = json.loads(path.read_text())
        for key in ('evaluation_seeds', 'source_seeds', 'seeds', 'collection_seeds'):
            used.update(prior.get(key, []))
        used.update((prior.get('demonstrations') or {}).get('source_seeds', []))
        if 'training_seed' in prior:
            used.add(prior['training_seed'])
    if used.intersection(seeds):
        parser.error('Collection seeds overlap previous training, selection, tests or collection')
    checkpoint = args.checkpoint if args.checkpoint.is_absolute() else args.pilot / args.checkpoint
    checkpoint_hash = digest(checkpoint)
    torch.set_num_threads(2)
    assert torch.ones(1, device='cuda').item() == 1
    policy = LegacyACT(pilot['checkpoint'], 'cuda', n_action_steps=pilot['action_window'])
    if policy.weights_sha256 != pilot['weights_sha256'] or pilot['grasp_reference'] != GRASP_REFERENCE:
        raise ValueError('Teacher ACT weights or grasp semantics changed')
    model = SAC.load(checkpoint, device='cuda')
    model.policy.set_training_mode(False)
    model.policy.requires_grad_(False)
    base = OpeningChessEnv(cases, capture_home_retry=pilot.get('capture_home_retry', False),
                           capture_step_limit=pilot.get('capture_step_limit', 800))
    if pilot.get('joint_map') is not None and pilot['joint_map'] != base.latest['observation']['joint_map']:
        raise ValueError('Collection calibration differs from the verified policy')
    env = OpeningLearningEnv(base, restore_timing_policy(policy, base, pilot), **settings)
    if env.observation_space != model.observation_space or env.action_space != model.action_space:
        raise ValueError('Teacher spaces differ from deployed environment')
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = {**pilot, 'source_pilot': str(args.pilot.resolve()), 'mode': 'frozen_verified_collection',
                'source_task_version': pilot['task_version'], 'task_version': LegalMoveTask.version,
                'objective': 'complete move',
                'collection_seeds': seeds, 'cases': cases, 'expected_episodes': args.episodes,
                'residual_checkpoint': str(checkpoint.resolve()), 'residual_checkpoint_sha256': checkpoint_hash,
                'video_recording': args.record_video, 'evaluation_use': 'teacher data collection; not a checkpoint-selection run'}
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    results, accepted = [], []
    started = time.perf_counter()
    try:
        for index, seed in enumerate(seeds):
            case = cases[index % len(cases)]
            base.fixed_case = case
            parent = args.output / case['id']
            parent.mkdir(exist_ok=True)
            provenance = {'act_sha256': policy.weights_sha256, 'residual_checkpoint': str(checkpoint.resolve()),
                          'residual_checkpoint_sha256': checkpoint_hash, 'task_version': LegalMoveTask.version,
                          'calibration': base.latest['observation']['joint_map'], 'grasp_reference': GRASP_REFERENCE,
                          'residual_bounds_deg': pilot['residual_bounds_deg'], 'residual_filter': pilot.get('residual_filter'),
                          'live_timing': pilot.get('live_timing'), 'policy_mode': 'deterministic evaluation; exploration disabled',
                          'capture_transport':pilot.get('capture_transport'),
                          'task_id': case['id']}
            result = evaluate(env, model, [seed], parent, 'teacher', record_video=args.record_video, collect_provenance=provenance)[0]
            result['task_id'] = case['id']
            path = parent / f'teacher_{seed}'
            audit_episode(path, result, case['move'], pilot)
            if result['collection']['accepted_for_act_training']:
                accepted.append(audit_collection(path / 'lossless_episode'))
            results.append(result)
            (args.output / 'results.json').write_text(json.dumps(results, indent=2))
            (args.output / 'accepted_episodes.json').write_text(json.dumps({
                'format': 'chess_sim_audited_episodes_v1', 'episodes': accepted,
                'frames': sum(row['frames'] for row in accepted), 'remote_upload': False,
                'rejected_episodes': [str((args.output / r['task_id'] / f"teacher_{r['seed']}" / 'lossless_episode').resolve())
                                      for r in results if not r['collection']['accepted_for_act_training']]}, indent=2))
            (args.output / 'summary.json').write_text(json.dumps({
                'completed_attempts': len(results), 'expected_attempts': args.episodes,
                'complete_move_successes': sum(r['score']['success'] for r in results),
                'accepted_training_episodes': len(accepted), 'accepted_frames': sum(r['frames'] for r in accepted),
                'note': 'Only complete upright successes independently verified from physics enter training data'}, indent=2))
            if args.record_video:
                write_training_video_index(args.output)
        assert digest(checkpoint) == checkpoint_hash
        assert digest(Path(policy.checkpoint) / 'model.safetensors') == policy.weights_sha256
        (args.output / 'status.json').write_text(json.dumps({'status': 'complete', 'elapsed_seconds': time.perf_counter() - started}))
    except BaseException as error:
        (args.output / 'status.json').write_text(json.dumps({'status': 'failed', 'error': repr(error)}))
        raise
    finally:
        env.close()


if __name__ == '__main__':
    main()
