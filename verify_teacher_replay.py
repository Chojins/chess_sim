"""Audit exact teacher transitions, action provenance and replay/validation split."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from stable_baselines3 import SAC
from capture_demonstrations import file_hash, load_demonstrations
from evaluate_openings import audit_episode
from residual_learning import proprio_features
from train_openings import move_features


def audit_teacher_replay(root, manifest):
    root = Path(root)
    parent = root / 'teacher_replay'
    report = json.loads((parent / 'replay.json').read_text())
    assert report == manifest['teacher_replay']
    assert not set(report['requested_seeds']) & {manifest['training_seed'], *manifest['evaluation_seeds']}
    assert report['validation_seed'] not in report['replay_seeds']
    if manifest.get('initialize_from'):
        assert file_hash(manifest['initialize_from']) == manifest['source_sha256']
    if report.get('teacher_checkpoint'):
        path = Path(report['teacher_checkpoint'])
        assert path.resolve().parent == root.resolve()
        assert file_hash(path) == report['teacher_sha256']
    else:
        path = Path(manifest['initialize_from'])
    torch.set_num_threads(2)
    teacher = SAC.load(path, device='cpu')
    teacher.policy.set_training_mode(False)
    rows = json.loads((parent / 'results.json').read_text())
    assert len(rows) == report['attempts'] and [row['seed'] for row in rows] == report['requested_seeds']
    cases = {case['id']: case for case in manifest['cases']}
    successful, transitions, demo_start, episode_id = [], 0, 0, 0
    demos, metadata = load_demonstrations(parent / 'demonstrations.npz')
    for result in rows:
        case = cases[result['task_id']]
        directory = parent / case['id'] / f"teacher_{result['seed']}"
        audit_episode(directory, result, case['move'], manifest)
        if manifest['objective'] == 'complete move':
            from verify_policy_reliability import recompute
            recompute(directory)
        initial = json.loads((directory / 'initial.json').read_text())
        traces = [json.loads(line) for line in (directory / 'steps.jsonl').read_text().splitlines()]
        mapping = initial['telemetry']['observation']['joint_map']
        max_steps = initial['task_config']['max_steps']
        feature_path = directory / result['learning_transitions']['path']
        assert file_hash(feature_path) == result['learning_transitions']['sha256']
        with np.load(feature_path, allow_pickle=False) as data:
            obs, actions = data['observations'], data['actions']
            assert obs.shape == (result['steps'] + 1, manifest['observation_size'])
            np.testing.assert_allclose(actions, [row['action'] for row in traces], atol=1e-7)
            np.testing.assert_allclose(data['rewards'], [row['reward'] for row in traces], atol=1e-6)
            np.testing.assert_equal(data['terminated'], [row['terminated'] for row in traces])
            np.testing.assert_equal(data['truncated'], [row['truncated'] for row in traces])
            assert np.isfinite(obs).all() and np.isfinite(actions).all()
            np.testing.assert_allclose(np.linalg.norm(obs[:, :2048], axis=1), 1., atol=1e-5)
            np.testing.assert_allclose(np.linalg.norm(obs[:, 2048:4096], axis=1), 1., atol=1e-5)
            predicted = teacher.predict(obs[:-1], deterministic=True)[0]
            np.testing.assert_allclose(predicted, actions, atol=2e-5)
            previous = np.zeros(6, np.float32)
            state = initial['telemetry']['observation']['observation_state']
            for i, row in enumerate(traces):
                timing = row['act_timing']
                clock = i-timing.get('capture_transport_steps',0)
                budget = timing.get('actor_step_budget',max_steps)
                proprio, _ = proprio_features(state, timing['act_base_legacy'], previous, clock, budget,
                                              mapping, timing['native_chunk_phase'])
                np.testing.assert_allclose(obs[i, 4096:], np.r_[proprio, move_features(case)], atol=2e-6)
                previous, state = row['learning']['applied_residual'], row['state']
            if result['score']['success'] and result['score']['other_pieces_disturbed'] == 0:
                successful.append(result['seed'])
                stop = demo_start + result['steps']
                np.testing.assert_equal(demos['observations'][demo_start:stop], obs[:-1])
                np.testing.assert_equal(demos['actions'][demo_start:stop], actions)
                np.testing.assert_equal(demos['episode'][demo_start:stop], np.full(result['steps'], episode_id))
                is_validation = result['seed'] == report['validation_seed']
                np.testing.assert_equal(demos['validation'][demo_start:stop], np.full(result['steps'], is_validation))
                if not is_validation:
                    assert result['seed'] in report['replay_seeds']
                    transitions += result['steps']
                demo_start, episode_id = stop, episode_id + 1
            else:
                assert result['seed'] not in report['replay_seeds']
    assert len(successful) == report['successful_episodes']
    assert transitions == report['replay_transitions']
    assert demo_start == len(demos['actions'])
    assert set(metadata['source_seeds']) == set(report['requested_seeds'])
    result = {'passed': True, 'teacher_attempts': len(rows), 'successful_episodes': len(successful),
              'training_replay_transitions': transitions, 'validation_seed_excluded': True,
              'actor_actions_recomputed_from_exact_features': True, 'feature_tail_matches_observable_state_and_move': True,
              'rewards_and_terminal_flags_match_trace': True, 'source_checkpoint_unchanged': True}
    result['complete_move_outcomes_independently_recomputed'] = manifest['objective'] == 'complete move'
    (parent / 'integrity.json').write_text(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    manifest = json.loads((args.directory / 'manifest.json').read_text())
    print(json.dumps(audit_teacher_replay(args.directory, manifest)))


if __name__ == '__main__':
    main()
