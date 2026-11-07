"""Require complete-move reliability per task before declaring a teacher ready."""
import argparse
from collections import Counter
import json
from pathlib import Path
from chess_task import LegalMoveTask, TaskConfig
from evaluate_openings import audit_episode
from evaluate_residual import digest


def assess(manifest, results, label='learned', minimum_trials=20, minimum_fraction=.9):
    if manifest.get('objective') != 'complete move' or manifest.get('task_version') != LegalMoveTask.version:
        raise ValueError('Pickup-only or mismatched task scores cannot certify a complete-move teacher')
    if manifest.get('mode') != 'frozen_evaluation_only':
        raise ValueError('Checkpoint selection or training scores cannot certify reliability')
    rows = [row for row in results if row['label'] == label]
    if not rows:
        raise ValueError('No results for requested teacher')
    per_task = {}
    for case in manifest['cases']:
        trials = [row for row in rows if row['task_id'] == case['id']]
        identities = {(row['seed'], row['repeat']) for row in trials}
        if len(identities) != len(trials):
            raise ValueError('Duplicate trial identities')
        successes = sum(bool(row['score']['success']) for row in trials)
        upright = sum(bool(row['score']['success'] and (not case['capture'] or
                           (row['score'].get('capture') or {}).get('upright_stable', False))) for row in trials)
        fraction = successes / len(trials) if trials else 0.
        per_task[case['id']] = {'trials': len(trials), 'distinct_seeds': len({row['seed'] for row in trials}),
                                'complete_move_successes': successes, 'success_fraction': fraction,
                                'upright_training_quality_successes': upright,
                                'outcomes': dict(Counter(row['score']['reason'] for row in trials)),
                                'passed': len(trials) >= minimum_trials and fraction >= minimum_fraction}
    has_capture = any(case['capture'] for case in manifest['cases'])
    has_ordinary = any(not case['capture'] for case in manifest['cases'])
    return {'passed': all(row['passed'] for row in per_task.values()) and has_capture and has_ordinary,
            'teacher_label': label, 'minimum_trials_per_task': minimum_trials,
            'minimum_success_fraction_per_task': minimum_fraction,
            'capture_tested': has_capture, 'ordinary_moves_tested': has_ordinary, 'per_task': per_task,
            'scope': 'Only tested blue-robot opening states and moves; not arbitrary boards, special moves or real hardware',
            'note': 'Repeated trials share seeds. Fractions are descriptive bring-up criteria, not independent-sample confidence guarantees.'}


def recompute(path):
    initial = json.loads((path / 'initial.json').read_text())
    if initial['task_version'] != LegalMoveTask.version or 'task_config' not in initial:
        raise ValueError('Need complete task version and explicit scoring settings')
    task = LegalMoveTask(initial['telemetry']['observation'], TaskConfig(**initial['task_config']))
    rows = [json.loads(line) for line in (path / 'steps.jsonl').read_text().splitlines()]
    for row in rows:
        _, terminated, truncated, score = task.update(row['placement_telemetry'])
        failure = (row.get('act_timing') or {}).get('failure')
        if failure and not (terminated or truncated):
            score.update(success=False, reason=failure)
            terminated, task.done = True, True
        if score != row['score'] or terminated != row['terminated'] or truncated != row['truncated']:
            raise ValueError('Saved scoring differs from independently recomputed physics outcome')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path, nargs='+', help='One evaluation, or separate capture and ordinary-move evaluations with distinct task IDs')
    parser.add_argument('--output', type=Path, help='Combined readiness report; defaults to the first evaluation/reliability.json')
    parser.add_argument('--label', choices=['baseline', 'learned'], default='learned')
    parser.add_argument('--minimum-trials', type=int, default=20)
    parser.add_argument('--minimum-fraction', type=float, default=.9)
    args = parser.parse_args()
    if args.minimum_trials < 1 or not 0 < args.minimum_fraction <= 1:
        parser.error('Invalid reliability threshold')
    combined, all_results, profiles, seen = None, [], [], set()
    for root in args.directory:
        manifest = json.loads((root / 'manifest.json').read_text())
        results = json.loads((root / 'results.json').read_text())
        # Each source must independently satisfy the complete/frozen semantics,
        # even though a capture-only source cannot alone pass the overall gate.
        assess(manifest, results, args.label, args.minimum_trials, args.minimum_fraction)
        ids = {case['id'] for case in manifest['cases']}
        if ids & seen:
            raise ValueError('Separate evaluations must have distinct task IDs; do not pool different policies on one task')
        seen.update(ids)
        if combined is None:
            combined = {**manifest, 'cases': []}
        elif manifest['weights_sha256'] != combined['weights_sha256'] or manifest['grasp_reference'] != combined['grasp_reference']:
            raise ValueError('Separate teacher profiles must share ACT and grasp semantics')
        combined['cases'].extend(manifest['cases'])
        all_results.extend(results)
        integrity = json.loads((root / 'integrity.json').read_text())
        if not integrity['passed'] or len(results) != manifest['expected_episodes']:
            raise ValueError('Frozen evaluation integrity check is incomplete')
        for label, path in manifest['models'].items():
            if digest(Path(path)) != manifest['model_hashes'][label]:
                raise ValueError('Evaluated checkpoint changed')
        moves = {case['id']: case['move'] for case in manifest['cases']}
        for row in results:
            if row['label'] != args.label:
                continue
            path = root / f"repeat_{row['repeat']}" / row['task_id'] / f"{row['label']}_{row['seed']}"
            audit_episode(path, row, moves[row['task_id']], manifest)
            recompute(path)
        profiles.append({'evaluation': str(root.resolve()), 'task_ids': sorted(ids),
                         'pilot': manifest['pilot'], 'checkpoint': manifest['models'][args.label],
                         'checkpoint_sha256': manifest['model_hashes'][args.label],
                         'residual_bounds_deg': manifest['residual_bounds_deg'],
                         'capture_transport':manifest.get('capture_transport'),
                         'capture_step_limit':manifest.get('capture_step_limit',800),
                         'actor_time_reference':manifest.get('actor_time_reference'),
                         'residual_filter': manifest.get('residual_filter'), 'live_timing': manifest.get('live_timing')})
    report = assess(combined, all_results, args.label, args.minimum_trials, args.minimum_fraction)
    report['deployment_profiles'] = profiles
    output = args.output or args.directory[0] / 'reliability.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report))
    if not report['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
