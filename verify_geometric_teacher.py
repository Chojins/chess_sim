"""Recompute privileged teacher labels and audit the real-observable features."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from capture_demonstrations import BOUNDS, FILTER, file_hash
from evaluate_openings import audit_episode
from geometric_grasp_teacher import GeometricGraspTeacher
from residual_learning import grasp_gate, proprio_features, combined_potential
from capture_alignment import capture_alignment, closing_alignment_penalty
from train_openings import move_features
from verify_policy_reliability import recompute


def audit(root):
    root = Path(root)
    manifest = json.loads((root/'manifest.json').read_text())
    status = json.loads((root/'status.json').read_text())
    if status['status'] != 'complete' or not status['act_checkpoint_unchanged']:
        raise ValueError('Teacher collection is incomplete or ACT weights changed')
    for name, digest in manifest['teacher']['code_sha256'].items():
        if file_hash(name) != digest:
            raise ValueError('Teacher code changed since collection')
    kin_path = Path(manifest['live_timing']['kinematics_path'])
    if file_hash(kin_path) != manifest['live_timing']['kinematics_sha256']:
        raise ValueError('Robot kinematics changed')
    spec = json.loads(kin_path.read_text())
    results = json.loads((root/'results.json').read_text())
    if len(results) != len(manifest['source_seeds']):
        raise ValueError('Missing teacher attempts')
    cases = {case['id']: case for case in manifest['cases']}
    transitions = 0
    for result in results:
        case = cases[result['task_id']]
        path = root/case['id']/f"teacher_{result['seed']}"
        audit_episode(path, result, case['move'], manifest)
        if manifest['objective'] == 'complete move':
            recompute(path)
        initial = json.loads((path/'initial.json').read_text())
        actual = initial['telemetry']['observation']
        mapping = actual['joint_map']
        def observable_gate(observation):
            # Live features use the float32 robot state returned by ChessEnv,
            # not the higher-precision diagnostic joint telemetry.
            state = np.asarray(observation['observation_state'], np.float32)
            q = np.asarray(mapping['sim_home_deg']) + np.asarray(mapping['scale'])*(state-mapping['legacy_home'])
            return grasp_gate(q)
        base = SimpleNamespace(latest={'observation': actual}, task=SimpleNamespace(
            capture_index=actual['task_geometry']['capture_index'], source_index=actual['task_geometry']['source_index'], capture_bonus=False),
            sim_home=np.asarray(mapping['sim_home_deg']), legacy_home=np.asarray(mapping['legacy_home']), scale=np.asarray(mapping['scale']))
        timing = SimpleNamespace()
        env = SimpleNamespace(unwrapped=base, env=SimpleNamespace(bounds=BOUNDS, policy=timing),
            residual_alpha=FILTER['alpha'], max_residual_change_deg=np.array(FILTER['max_change_deg']),
            previous_action=np.zeros(6, np.float32), gate=observable_gate(actual))
        teacher = GeometricGraspTeacher(env, spec, manifest['teacher']['surface_anchors_jaw_local_m'])
        def potential(observation, score=None):
            info = {'telemetry': {'observation': observation}}
            if score is not None:
                info['score'] = score
            gate = observable_gate(observation)
            return combined_potential(gate, info) + gate*capture_alignment(info)[0]
        previous_potential = potential(actual)
        rows = [json.loads(line) for line in (path/'steps.jsonl').read_text().splitlines()]
        decisions = [json.loads(line) for line in (path.parent/f"teacher_{result['seed']}_decisions.jsonl").read_text().splitlines()]
        if len(rows) != len(decisions):
            raise ValueError('Teacher decisions and executed transitions differ')
        replay_path = path/result['learning_transitions']['path']
        if file_hash(replay_path) != result['learning_transitions']['sha256']:
            raise ValueError('Teacher transitions changed')
        with np.load(replay_path, allow_pickle=False) as data:
            features = data['observations']
            assert features.shape == (len(rows)+1, 4122)
            np.testing.assert_allclose(np.linalg.norm(features[:, :2048], axis=1), 1., atol=1e-5)
            np.testing.assert_allclose(np.linalg.norm(features[:, 2048:4096], axis=1), 1., atol=1e-5)
            np.testing.assert_equal(data['actions'], [row['action'] for row in rows])
            np.testing.assert_allclose(data['rewards'], [row['reward'] for row in rows], atol=1e-6)
            np.testing.assert_equal(data['terminated'], [row['terminated'] for row in rows])
            np.testing.assert_equal(data['truncated'], [row['truncated'] for row in rows])
            state = actual['observation_state']
            before_frame = initial['telemetry']['frame_id']
            for i, (row, decision) in enumerate(zip(rows, decisions)):
                meta = row['act_timing']
                timing.phase, timing.native_index, timing.interventions = meta['phase'], meta['native_index'], meta['interventions']
                env.env.base = np.asarray(meta['act_base_legacy'], np.float32)
                env.gate = observable_gate(actual)
                predicted, _ = teacher.predict(features[i])
                np.testing.assert_allclose(predicted, row['action'], atol=2e-5)
                assert decision['frame_id'] == before_frame
                np.testing.assert_allclose(decision['action'], predicted, atol=2e-5)
                proprio, _ = proprio_features(state, meta['act_base_legacy'], env.previous_action, i,
                    initial['task_config']['max_steps'], mapping, meta['native_chunk_phase'])
                np.testing.assert_allclose(features[i, 4096:], np.r_[proprio, move_features(case)], atol=2e-6)
                after = {**actual, **row['placement_telemetry'], 'observation_state': row['state']}
                if manifest['objective'] == 'complete move':
                    next_potential = 0. if row['terminated'] else potential(after, row['score'])
                    info = {'telemetry': {'observation': after}, 'score': row['score']}
                    cost = closing_alignment_penalty(actual, info, env.gate)[0]
                    expected = (sum(row['score']['reward_terms'].values()) - row['residual']['penalty'] +
                                manifest['potential_weight']*(manifest['gamma']*next_potential-previous_potential) + cost)
                    np.testing.assert_allclose(expected, row['reward'], atol=1e-6)
                    previous_potential = next_potential
                actual = after
                base.latest = {'observation': actual}
                base.task.capture_bonus = row['score']['phase'] == 'move_attacker'
                env.previous_action = np.asarray(row['learning']['applied_residual'], np.float32)
                state, before_frame = row['state'], row['frame_id']
        transitions += len(rows)
    report = {'passed': True, 'episodes': len(results), 'transitions': transitions,
        'teacher_actions_recomputed': True, 'actor_feature_tail_is_real_observable': True,
        'actor_has_no_piece_pose_inputs': True, 'frame_and_timing_audits_passed': True,
        'full_move_outcomes_recomputed': manifest['objective'] == 'complete move',
        'full_move_rewards_recomputed': manifest['objective'] == 'complete move',
        'privileged_simulation_teacher_only': True}
    (root/'integrity.json').write_text(json.dumps(report, indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.directory)))
