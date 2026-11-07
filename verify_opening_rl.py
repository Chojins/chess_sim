"""Audit a completed multi-move residual RL pilot and its saved weights."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import SAC

from verify_residual_pilot import check_row, check_snapshot, read_json, read_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    root = args.directory
    manifest, summary = read_json(root / 'manifest.json'), read_json(root / 'summary.json')
    expected = {(c['id'], s) for c in manifest['cases'] for s in manifest['evaluation_seeds']}
    offsets, evaluation_steps = {}, 0
    labels = manifest.get('evaluation_labels', ['baseline', 'learned'])
    bounds = np.asarray(manifest['residual_bounds_deg'])
    filtering = manifest.get('residual_filter')
    kinematics = None
    if manifest.get('live_timing'):
        from capture_demonstrations import JointKinematics, file_hash
        kin_path = Path(manifest['live_timing']['kinematics_path'])
        assert file_hash(kin_path) == manifest['live_timing']['kinematics_sha256']
        kinematics = JointKinematics(read_json(kin_path))
    for label in labels:
        results = read_json(root / f'{label}_results.json')
        assert len(results) == len(expected)
        assert {(r['task_id'], r['seed']) for r in results} == expected
        for case in manifest['cases']:
            count = sum(r['score']['success'] for r in results if r['task_id'] == case['id'])
            assert summary['per_task'][case['id']][label] == {'successes': count, 'episodes': len(manifest['evaluation_seeds'])}
        for result in results:
            path = root / result['task_id'] / f'{label}_{result["seed"]}'
            initial = read_json(path / 'initial.json')
            telemetry = initial['telemetry']
            state = telemetry['observation']
            key = (result['task_id'], result['seed'])
            if key in offsets:
                np.testing.assert_equal(offsets[key], initial['reset_options'])
            offsets[key] = initial['reset_options']
            rows = read_rows(path / 'steps.jsonl')
            assert len(rows) == result['steps']
            frame, sim_time = telemetry['frame_id'], telemetry['simulation_time']
            check_snapshot(path / 'initial_images.npz', state['observation_state'])
            previous_applied = np.zeros(6)
            timing_audit = None
            if manifest.get('live_timing'):
                from verify_live_timing import TimingAudit
                timing_audit = TimingAudit(state['joint_map'], bounds, manifest['live_timing']['config'], manifest['action_window'], kinematics, manifest.get('capture_transport'))
            for i, row in enumerate(rows):
                check_row(row, state, previous_applied, bounds, filtering)
                if timing_audit:
                    timing_audit.check(row)
                previous_applied = row['learning']['applied_residual']
                assert row['step'] == i + 1 and row['frame_id'] == frame + 1
                assert row['score']['move'] == state['task_geometry']['move']
                np.testing.assert_allclose(row['simulation_time'] - sim_time, 1 / 30, atol=1e-6)
                assert bool(row['terminated'] or row['truncated']) == (i == len(rows) - 1)
                if label == 'baseline':
                    np.testing.assert_equal(row['action'], np.zeros(6))
                frame, sim_time = row['frame_id'], row['simulation_time']
            assert rows[-1]['score'] == result['score']
            np.testing.assert_allclose(sum(r['reward'] for r in rows), result['return'])
            check_snapshot(path / 'terminal_images.npz', rows[-1]['state'])
            evaluation_steps += len(rows)
    rows = read_rows(root / 'training_steps.jsonl')
    assert len(rows) == summary['training_steps'] == manifest['steps']
    allowed = {c['move'] for c in manifest['cases']}
    seen = set()
    timing_audit = None
    for i, row in enumerate(rows):
        previous_applied = rows[i-1]['learning']['applied_residual'] if i and not rows[i-1]['done'] else np.zeros(6)
        check_row(row, state, previous_applied, bounds, filtering)
        if manifest.get('live_timing'):
            if i == 0 or rows[i - 1]['done']:
                timing_audit = TimingAudit(state['joint_map'], bounds, manifest['live_timing']['config'], manifest['action_window'], kinematics, manifest.get('capture_transport'))
            timing_audit.check(row)
        assert row['step'] == i + 1 and row['score']['move'] in allowed
        seen.add(row['score']['move'])
        assert row['done'] == (row['score']['reason'] is not None)
        if i and not rows[i - 1]['done']:
            assert row['frame_id'] == rows[i - 1]['frame_id'] + 1
            assert row['score']['move'] == rows[i - 1]['score']['move']
            np.testing.assert_allclose(row['simulation_time'] - rows[i - 1]['simulation_time'], 1 / 30, atol=1e-6)
    assert seen == allowed
    with (Path(manifest['checkpoint']) / 'model.safetensors').open('rb') as stream:
        assert hashlib.file_digest(stream, 'sha256').hexdigest() == manifest['weights_sha256']
    torch.set_num_threads(2)
    before = SAC.load(root / 'residual_initial', device='cpu')
    if manifest.get('initialize_from'):
        source = Path(manifest['initialize_from'])
        assert hashlib.sha256(source.read_bytes()).hexdigest() == manifest['source_sha256']
        teacher = SAC.load(source, device='cpu')
        for name, tensor in before.actor.state_dict().items():
            if name == 'log_std' and manifest.get('exploration_log_std_reset') is not None:
                torch.testing.assert_close(tensor, torch.full_like(tensor, manifest['exploration_log_std_reset']))
            else:
                assert torch.equal(tensor, teacher.actor.state_dict()[name])
    elif manifest.get('actor_initialization') == 'fresh actor with exactly zero deterministic residual':
        assert all(torch.count_nonzero(p).item() == 0 for p in before.actor.mu.parameters())
    if manifest.get('exploration_log_std_reset') is not None:
        torch.testing.assert_close(before.actor.log_std,
                                   torch.full_like(before.actor.log_std, manifest['exploration_log_std_reset']))
    after = SAC.load(root / 'residual_final', device='cpu')
    assert after.observation_space.shape == (manifest['observation_size'],)
    assert all(torch.isfinite(p).all() for p in after.policy.parameters())
    actor_updated = any(not torch.equal(a,b) for a,b in zip(before.actor.parameters(),after.actor.parameters()))
    assert actor_updated or manifest.get('actor_stability')
    assert summary['gradient_updates'] > 0
    if manifest.get('demonstrations'):
        from capture_demonstrations import load_demonstrations
        data, demos = load_demonstrations(Path(manifest['demonstrations']['path']))
        assert demos['dataset_sha256'] == manifest['demonstrations']['sha256']
        assert summary['demonstration_updates'] == summary['gradient_updates']
        assert not {manifest['training_seed'], *manifest['evaluation_seeds']} & set(demos['source_seeds'])
        if manifest.get('actor_stability'):
            from capture_demonstrations import supervised_loss
            mask=~data['validation']
            with torch.no_grad():
                loss=supervised_loss(after.actor,torch.as_tensor(data['observations'][mask]),
                    torch.as_tensor(data['actions'][mask]),torch.as_tensor(data['weights'][mask])).item()
            budget=manifest['actor_stability']['training_loss_budget']
            if budget is not None:
                assert loss <= budget+1e-6, 'Final actor exceeded its training-demonstration retention budget'
    if manifest.get('teacher_replay'):
        if manifest['teacher_replay'].get('kind') == 'geometric_teacher':
            from geometric_replay import prefill
            assert prefill(manifest['teacher_replay']['dataset'], manifest) == manifest['teacher_replay']
        else:
            from verify_teacher_replay import audit_teacher_replay
            audit_teacher_replay(root, manifest)
        replay = manifest['teacher_replay']
        assert replay['replay_transitions'] == summary['initial_successful_replay_transitions'] > 0
        assert replay['validation_seed'] not in replay['replay_seeds']
        assert not set(replay['requested_seeds']) & {manifest['training_seed'], *manifest['evaluation_seeds']}
    if manifest.get('evaluation_interval'):
        selection = read_json(root / 'checkpoint_selection.json')
        candidates = selection['results']
        for candidate in candidates:
            rows_for_label = read_json(root / f"{candidate['label']}_results.json")
            assert candidate['successes'] == sum(r['score']['success'] for r in rows_for_label)
        best = max(candidates, key=lambda c: c['successes'])
        assert selection['best_label'] == summary['best_checkpoint_label'] == best['label']
        filename = ('residual_initial' if best['label'] == 'before' else
                    'residual_final' if best['label'] == 'learned' else f"residual_{best['step']}")
        selected = SAC.load(root / 'residual_best', device='cpu')
        candidate_model = SAC.load(root / filename, device='cpu')
        assert all(torch.equal(a, b) for a, b in zip(selected.actor.parameters(), candidate_model.actor.parameters()))
    report = {'passed': True, 'training_transitions': len(rows), 'evaluation_transitions': evaluation_steps,
              'evaluation_episodes': len(labels) * len(expected), 'all_tasks_seen_in_training': True,
              'residual_actor_updated': actor_updated, 'saved_weights_finite': True, 'act_checkpoint_unchanged': True}
    (root / 'integrity.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report))


if __name__ == '__main__':
    main()
