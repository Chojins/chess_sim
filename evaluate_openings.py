"""Frozen opening-policy comparison with fresh seeds and paired repeated trials."""
import html
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from stable_baselines3 import SAC

from act_client import LegacyACT
from chess_task import LegalMoveTask
from evaluate_residual import digest, schedule, summarize
from grasp_geometry import GRASP_REFERENCE
from train_openings import OpeningChessEnv, OpeningLearningEnv
from train_residual import evaluate
from verify_residual_pilot import check_row, check_snapshot, read_json, read_rows


def opening_schedule(seeds, repeats, only_policy=None):
    if only_policy is not None:
        if only_policy not in ('baseline', 'learned'):
            raise ValueError('Unknown policy label')
        if not seeds or len(set(seeds)) != len(seeds) or any(s < 0 for s in seeds) or repeats < 1:
            raise ValueError('Need unique nonnegative seeds and at least one repeat')
        return [{'repeat': repeat, 'seed': seed, 'label': only_policy}
                for repeat in range(repeats) for seed in seeds]
    return schedule(seeds, repeats)


def environment_settings(pilot):
    """Restore physical action/reward semantics, not just model space shapes."""
    pickup_only = pilot.get('objective') == 'pickup-only diagnostic; not full capture success'
    alignment = pilot.get('capture_alignment', False)
    if alignment and not pickup_only and not pilot.get('live_timing'):
        raise ValueError('Capture alignment checkpoint requires pickup-only evaluation')
    if pickup_only and pilot.get('pickup_limit_steps', 240) != 240:
        raise ValueError('Unsupported source pickup time limit')
    filtering = pilot.get('residual_filter', {'alpha': 1., 'max_change_deg': None})
    return pickup_only, {
        'gamma': pilot['gamma'], 'shaping_weight': pilot['potential_weight'],
        'bounds_deg': pilot['residual_bounds_deg'],
        'residual_alpha': filtering['alpha'], 'max_residual_change_deg': filtering['max_change_deg'],
        'capture_alignment': alignment}


def restore_timing_policy(policy, base, pilot):
    """One deployment path shared by frozen evaluations and episode collection."""
    timing = pilot.get('live_timing')
    if not timing:
        return policy
    from capture_demonstrations import JointKinematics
    from live_act_timing import CloseBeforeLiftACT, TimingConfig
    path = Path(timing['kinematics_path'])
    if digest(path) != timing['kinematics_sha256']:
        raise ValueError('Deployment robot kinematics changed')
    specification = read_json(path)
    if specification['usd_sha256'] != digest(Path('Sim_Stage.usd')):
        raise ValueError('Deployment stage differs from exported robot kinematics')
    kin = JointKinematics(specification)
    np.testing.assert_allclose(base.latest['observation']['joint_limits_deg'], kin.limits)
    timed = CloseBeforeLiftACT(policy, base.latest['observation']['joint_map'], kin, TimingConfig(**timing['config']))
    if pilot.get('capture_transport'):
        from capture_transport import CaptureTransportACT, TransportConfig
        transport = pilot['capture_transport']
        if digest(Path('capture_transport.py')) != transport['code_sha256']:
            raise ValueError('Capture transport implementation changed since the profile was created')
        return CaptureTransportACT(timed, TransportConfig(**transport['config']))
    return timed


def audit_episode(path, result, move, pilot=None):
    initial = read_json(path / 'initial.json')
    telemetry = initial['telemetry']
    obs = telemetry['observation']
    assert obs['task_geometry']['move'] == move
    assert obs['grasp_reference'] == GRASP_REFERENCE
    check_snapshot(path / 'initial_images.npz', obs['observation_state'])
    rows = read_rows(path / 'steps.jsonl')
    assert len(rows) == result['steps']
    frame, sim_time = telemetry['frame_id'], telemetry['simulation_time']
    previous_applied = np.zeros(6)
    bounds = np.asarray(pilot['residual_bounds_deg']) if pilot else None
    filtering = pilot.get('residual_filter') if pilot else None
    timing_audit = None
    if pilot and pilot.get('live_timing'):
        from verify_live_timing import TimingAudit
        from capture_demonstrations import JointKinematics
        kin = JointKinematics(read_json(Path(pilot['live_timing']['kinematics_path'])))
        timing_audit = TimingAudit(obs['joint_map'], bounds, pilot['live_timing']['config'], pilot['action_window'], kin, pilot.get('capture_transport'))
    for i, row in enumerate(rows):
        check_row(row, obs, previous_applied, bounds, filtering)
        if timing_audit:
            timing_audit.check(row)
        previous_applied = row['learning']['applied_residual']
        assert row['step'] == i + 1 and row['frame_id'] == frame + 1
        assert row['score']['move'] == move
        if pilot and pilot.get('objective') == 'pickup-only diagnostic; not full capture success':
            assert initial['task_version'] == 'capture_pickup_v1'
            assert row['score']['objective'] == 'capture_pickup_only'
        assert bool(row['terminated'] or row['truncated']) == (i == len(rows) - 1)
        np.testing.assert_allclose(row['simulation_time'] - sim_time, 1 / 30, atol=1e-6)
        frame, sim_time = row['frame_id'], row['simulation_time']
    assert rows[-1]['score'] == result['score']
    np.testing.assert_allclose(sum(r['reward'] for r in rows), result['return'])
    check_snapshot(path / 'terminal_images.npz', rows[-1]['state'])
    return initial['reset_options'], len(rows)


def write_video_index(root, results, labels, objective=None):
    body = ['<!doctype html><meta charset="utf-8"><title>Recorded policy comparison</title>',
            '<style>body{font:16px sans-serif;max-width:1100px;margin:2em auto}td,th{padding:.5em;text-align:left}tr:nth-child(even){background:#eee}</style>',
            '<h1>Recorded policy comparison</h1><p>Videos play at simulated 30 Hz. Left: fixed camera with move overlay. Right: wrist camera. Frame 0 is reset; frame N corresponds to step N in the saved trace.</p>',
            '<p>Results are updated after each episode. Refresh this page for new recordings.</p>',
            '<table><tr><th>Policy</th><th>Task</th><th>Seed / repeat</th><th>Outcome</th><th>Recording</th></tr>']
    if objective:
        body.insert(-1, f'<p>Objective: {html.escape(objective)}.</p>')
    for row in results:
        rel = f"repeat_{row['repeat']}/{row['task_id']}/{row['label']}_{row['seed']}/cameras.mp4"
        body.append('<tr>' + ''.join(f'<td>{html.escape(str(v))}</td>' for v in
                    [labels[row['label']], row['task_id'], f"{row['seed']} / {row['repeat']}", row['score']['reason']]) +
                    f'<td><a href="{html.escape(rel)}">Watch / download video</a></td></tr>')
    body.append('</table>')
    (root / 'videos.html').write_text('\n'.join(body))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--pilot', type=Path, default=Path('outputs/opening-rl-placement-v5'))
    parser.add_argument('--baseline-checkpoint', type=Path, help='First frozen checkpoint; relative paths resolve within pilot')
    parser.add_argument('--learned-checkpoint', type=Path, help='Second frozen checkpoint; relative paths resolve within pilot')
    parser.add_argument('--baseline-label', help='Display name for first checkpoint')
    parser.add_argument('--learned-label', help='Display name for second checkpoint')
    parser.add_argument('--seeds', nargs='+', type=int, default=list(range(901, 906)))
    parser.add_argument('--ids', nargs='+', help='Evaluate only these task IDs')
    parser.add_argument('--comparison-note', default='Fresh-seed comparison')
    parser.add_argument('--record-video', action='store_true')
    parser.add_argument('--collect-episodes', action='store_true', help='Record lossless camera/state/executed-action bundles; only verified complete upright captures are accepted')
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--only-policy', choices=['baseline', 'learned'], help='Reliability test of one frozen policy; omits the paired comparison')
    parser.add_argument('--repeats', type=int, default=2)
    args = parser.parse_args()
    plan = opening_schedule(args.seeds, args.repeats, args.only_policy)
    pilot = read_json(args.pilot / 'manifest.json')
    if args.ids:
        tasks = {c['id']: c for c in pilot['cases']}
        if len(set(args.ids)) != len(args.ids) or any(name not in tasks for name in args.ids):
            parser.error('Task IDs must be unique and present in the pilot')
        pilot['cases'] = [tasks[name] for name in args.ids]
    used = {pilot['training_seed'], *pilot['evaluation_seeds']}
    # Include earlier multi-move experiments in the fresh-seed check.
    for path in Path('outputs').glob('*/manifest.json'):
        previous = read_json(path)
        used.update(previous.get('evaluation_seeds', []))
        used.update(previous.get('seeds', []))
        used.update(previous.get('source_seeds', []))
        used.update((previous.get('demonstrations') or {}).get('source_seeds', []))
        if 'training_seed' in previous:
            used.add(previous['training_seed'])
    if used.intersection(args.seeds):
        parser.error('Seeds overlap previous opening training/evaluation seeds')
    assert pilot['grasp_reference'] == GRASP_REFERENCE
    assert torch.ones(1, device='cuda').item() == 1
    paths = {}
    for label, override, default in [('baseline',args.baseline_checkpoint,'residual_initial.zip'),
                                      ('learned',args.learned_checkpoint,'residual_best.zip')]:
        if args.only_policy is not None and label != args.only_policy:
            continue
        filename = override if override is not None else Path(default)
        paths[label] = filename if filename.is_absolute() else args.pilot / filename
    hashes = {label: digest(path) for label, path in paths.items()}
    policy = LegacyACT(pilot['checkpoint'], 'cuda', n_action_steps=pilot['action_window'])
    assert policy.weights_sha256 == pilot['weights_sha256']
    models = {label: SAC.load(path, device='cuda') for label, path in paths.items()}
    pickup_only, settings = environment_settings(pilot)
    if args.collect_episodes and pickup_only:
        parser.error('Pickup diagnostics cannot collect complete-move ACT demonstrations')
    base = OpeningChessEnv(pilot['cases'], pickup_only=pickup_only,
                           capture_home_retry=pilot.get('capture_home_retry', False),
                           capture_step_limit=pilot.get('capture_step_limit', 800))
    if pilot.get('joint_map') is not None and pilot['joint_map'] != base.latest['observation']['joint_map']:
        raise ValueError('Evaluation calibration differs from the trained controller')
    adapted = restore_timing_policy(policy, base, pilot)
    env = OpeningLearningEnv(base, adapted, **settings)
    np.testing.assert_equal(env.env.bounds, pilot['residual_bounds_deg'])
    for model in models.values():
        assert model.observation_space == env.observation_space and model.action_space == env.action_space
        model.policy.set_training_mode(False)
        model.policy.requires_grad_(False)
    args.output.mkdir(parents=True, exist_ok=False)
    if pickup_only:
        from capture_pickup import CapturePickupTask
        task_version = CapturePickupTask.version
    else:
        task_version = LegalMoveTask.version
    manifest = {**pilot, 'pilot': str(args.pilot.resolve()), 'evaluation_seeds': args.seeds,
                'repeats': args.repeats, 'mode': 'frozen_evaluation_only',
                'only_policy': args.only_policy,
                'evaluation_use': 'independent fresh-seed comparison; not checkpoint selection',
                'task_version': task_version, 'source_task_version': pilot['task_version'],
                'objective': 'pickup-only diagnostic; not full capture success' if pickup_only else 'complete move',
                'source_reward': pilot['reward'],
                'reward': pilot['reward'].replace(pilot['task_version'], task_version),
                'comparison_note': args.comparison_note,
                'labels': {'baseline': args.baseline_label or ('previous best (v5)' if args.pilot.name == 'opening-rl-capture-pickup-v6' else 'starting residual'),
                           'learned': args.learned_label or ('capture-pickup best (v6)' if args.pilot.name == 'opening-rl-capture-pickup-v6' else 'selected best residual')},
                'video_recording': args.record_video, 'profiling': args.profile,
                'collect_episodes': args.collect_episodes,
                'models': {k: str(v.resolve()) for k, v in paths.items()}, 'model_hashes': hashes,
                'schedule': plan, 'expected_episodes': len(plan) * len(pilot['cases'])}
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    results, resets, transitions = [], {}, 0
    if args.record_video:
        write_video_index(args.output, results, manifest['labels'], manifest.get('objective'))
    started = time.perf_counter()
    try:
        for item in plan:
            for case in pilot['cases']:
                env.unwrapped.fixed_case = case
                parent = args.output / f"repeat_{item['repeat']}" / case['id']
                parent.mkdir(parents=True, exist_ok=True)
                label, seed = item['label'], item['seed']
                provenance = {'act_sha256': policy.weights_sha256,
                              'residual_checkpoint': str(paths[label].resolve()), 'residual_checkpoint_sha256': hashes[label],
                              'calibration': base.latest['observation']['joint_map'], 'grasp_reference': GRASP_REFERENCE,
                              'residual_bounds_deg': pilot['residual_bounds_deg'], 'residual_filter': pilot.get('residual_filter'),
                              'live_timing': pilot.get('live_timing'), 'task_version': task_version,
                              'capture_transport':pilot.get('capture_transport'),
                              'policy_mode': 'deterministic evaluation; exploration disabled',
                              'task_id': case['id'], 'repeat': item['repeat']} if args.collect_episodes else None
                result = evaluate(env, models[label], [seed], parent, label, record_video=args.record_video,
                                  profile=args.profile, collect_provenance=provenance)[0]
                result.update(task_id=case['id'], repeat=item['repeat'])
                options, count = audit_episode(parent / f'{label}_{seed}', result, case['move'], pilot)
                key = (case['id'], seed)
                if key in resets:
                    assert resets[key] == options
                resets[key] = options
                transitions += count
                results.append(result)
                if args.record_video:
                    write_video_index(args.output, results, manifest['labels'], manifest.get('objective'))
                (args.output / 'results.json').write_text(json.dumps(results, indent=2))
                report = {'per_task': {c['id']: summarize([r for r in results if r['task_id'] == c['id']])
                                       for c in pilot['cases']},
                          'completed_episodes': len(results), 'expected_episodes': manifest['expected_episodes']}
                (args.output / 'summary.json').write_text(json.dumps(report, indent=2))
        expected = {(c['id'], x['repeat'], x['seed'], x['label']) for c in pilot['cases'] for x in plan}
        assert len(results) == len(expected)
        assert {(r['task_id'], r['repeat'], r['seed'], r['label']) for r in results} == expected
        assert all(digest(paths[label]) == hashes[label] for label in paths)
        assert digest(Path(policy.checkpoint) / 'model.safetensors') == policy.weights_sha256
        if args.profile:
            measurements = []
            for result in results:
                path = args.output / f"repeat_{result['repeat']}" / result['task_id'] / f"{result['label']}_{result['seed']}" / 'timing_steps.json'
                measurements.extend(read_json(path))
            performance = {key: {'mean_ms': float(np.mean([m[key] for m in measurements if key in m])),
                                 'p95_ms': float(np.percentile([m[key] for m in measurements if key in m], 95))}
                           for key in sorted({k for m in measurements for k in m})}
            (args.output / 'performance.json').write_text(json.dumps({
                'transitions': len(measurements), 'stages': performance,
                'note': 'Wall times; nested stages overlap. Evaluation excludes training gradients and includes video overhead.'}, indent=2))
        integrity = {'passed': True, 'evaluation_episodes': len(results), 'transitions': transitions,
                     'checkpoints_unchanged': True, 'matched_resets_verified': args.only_policy is None,
                     'repeated_resets_verified': True, 'videos_verified': args.record_video,
                     'elapsed_seconds': time.perf_counter() - started}
        (args.output / 'integrity.json').write_text(json.dumps(integrity, indent=2))
        (args.output / 'status.json').write_text(json.dumps({'status': 'complete'}))
        print('EVALUATION_COMPLETE', json.dumps(integrity), flush=True)
    except Exception as error:
        (args.output / 'status.json').write_text(json.dumps({'status': 'failed', 'error': str(error)}))
        raise
    finally:
        env.close()


if __name__ == '__main__':
    main()
