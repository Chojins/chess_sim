"""Fresh paired live inference with one frozen actor, timing disabled/enabled."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time
import numpy as np
import torch
from stable_baselines3 import SAC
from act_client import LegacyACT
from capture_demonstrations import JointKinematics, file_hash
from evaluate_openings import audit_episode, environment_settings, write_video_index
from evaluate_residual import schedule, summarize
from live_act_timing import CloseBeforeLiftACT, TimingConfig
from train_openings import OpeningChessEnv, OpeningLearningEnv
from train_residual import evaluate


class TimingEvaluationChessEnv(OpeningChessEnv):
    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        self.task.pickup_limit = 360
        info['evaluation_pickup_limit_steps'] = self.task.pickup_limit
        return obs, info


def audit_timing_episode(path, result, manifest, enabled):
    options, count = audit_episode(path, result, manifest['cases'][0]['move'], manifest)
    initial = json.loads((path / 'initial.json').read_text())
    assert initial['evaluation_pickup_limit_steps'] == 360
    observation = initial['telemetry']['observation']
    mapping = observation['joint_map']
    scale = np.array(mapping['scale'])
    bounds = np.array(manifest['residual_bounds_deg'])
    previous = None
    interventions, paused = 0, 0
    for line in (path / 'steps.jsonl').read_text().splitlines():
        row = json.loads(line)
        if not enabled:
            assert 'act_timing' not in row
            continue
        timing = row['act_timing']
        requested = np.array(timing['act_base_legacy']) + np.array(row['learning']['applied_residual']) * bounds / scale
        np.testing.assert_allclose(requested, row['residual']['requested_legacy'], atol=2e-5)
        timed = np.array(row['residual']['timed_legacy'])
        phase = timing['phase']
        if phase in ('arm_settle', 'close_jaw', 'closed_settle'):
            np.testing.assert_allclose(timed[:5], timing['arm_anchor_legacy'], atol=1e-7)
            paused += 1
            if previous and previous['phase'] in ('arm_settle', 'close_jaw', 'closed_settle'):
                assert timing['native_index'] == previous['native_index']
        else:
            np.testing.assert_allclose(timed, requested, atol=1e-7)
        if previous:
            assert timing['native_index'] - previous['native_index'] in (0, 1)
            if phase == 'resume_closed' and previous['phase'] == 'closed_settle':
                assert previous['phase_step'] >= TimingConfig().minimum_closed_settle_steps
        if timing['failure']:
            assert row['terminated'] and row['score']['reason'] == timing['failure']
        interventions = max(interventions, timing['interventions'])
        previous = timing
    assert interventions <= 1
    return options, count, {'interventions': interventions, 'paused_steps': paused}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--pilot', type=Path, default=Path('outputs/capture-supervised-rl-v10'))
    parser.add_argument('--seeds', type=int, nargs='+', default=[1801, 1802, 1803, 1804, 1805])
    parser.add_argument('--repeats', type=int, default=2)
    args = parser.parse_args()
    plan = schedule(args.seeds, args.repeats)
    source = json.loads((args.pilot / 'manifest.json').read_text())
    if len(source['cases']) != 1 or source['cases'][0]['id'] != 'scandinavian_capture':
        raise ValueError('This timing comparison supports the single capture-pickup objective')
    used = set()
    for path in Path('outputs').glob('*/manifest.json'):
        previous = json.loads(path.read_text())
        used.update(previous.get('evaluation_seeds', []))
        used.update(previous.get('seeds', []))
        used.update(previous.get('source_seeds', []))
        used.update((previous.get('demonstrations') or {}).get('source_seeds', []))
        if 'training_seed' in previous:
            used.add(previous['training_seed'])
    if used & set(args.seeds):
        raise ValueError('Evaluation seeds overlap previous training, selection or testing')
    torch.set_num_threads(2)
    assert torch.ones(1, device='cuda').item() == 1
    policy = LegacyACT(source['checkpoint'], 'cuda', n_action_steps=source['action_window'])
    assert policy.weights_sha256 == source['weights_sha256']
    actor_path = args.pilot / 'residual_best.zip'
    actor_sha = file_hash(actor_path)
    model = SAC.load(actor_path, device='cuda')
    model.policy.set_training_mode(False)
    model.policy.requires_grad_(False)
    kin_path = Path('outputs/capture-supervised-kinematics-v10.json')
    specification = json.loads(kin_path.read_text())
    assert specification['usd_sha256'] == file_hash('Sim_Stage.usd')
    kin = JointKinematics(specification)
    pickup, settings = environment_settings(source)
    assert pickup
    envs = {}
    for label in ('baseline', 'learned'):
        base = TimingEvaluationChessEnv(source['cases'], pickup_only=True)
        mapping = base.latest['observation']['joint_map']
        np.testing.assert_allclose(base.latest['observation']['joint_limits_deg'], kin.limits)
        adapted = policy if label == 'baseline' else CloseBeforeLiftACT(policy, mapping, kin)
        envs[label] = OpeningLearningEnv(base, adapted, **settings)
        assert model.observation_space == envs[label].observation_space
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = {**source, 'evaluation_seeds': args.seeds, 'mode': 'frozen_live_timing_evaluation',
                'pilot': str(args.pilot.resolve()), 'evaluation_pickup_limit_steps': 360,
                'source_pickup_limit_steps': source['pickup_limit_steps'],
                'timing_controller': asdict(TimingConfig()), 'kinematics_sha256': file_hash(kin_path),
                'actor_path': str(actor_path.resolve()), 'actor_sha256': actor_sha,
                'labels': {'baseline': 'live ACT + frozen residual', 'learned': 'same policy + close before lift'},
                'schedule': plan, 'expected_episodes': len(plan), 'repeats': args.repeats,
                'policy_inputs': 'cameras and joint readings; timing uses robot FK, predicted commands and encoders only',
                'note': 'Pickup-only controller comparison on one board/move; repeated seeds are not independent reliability samples'}
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    results, resets, transitions = [], {}, 0
    write_video_index(args.output, results, manifest['labels'], manifest['objective'])
    started = time.perf_counter()
    print('LIVE_TIMING_EVALUATION_STARTED', flush=True)
    try:
        for item in plan:
            label, seed = item['label'], item['seed']
            env = envs[label]
            env.unwrapped.fixed_case = source['cases'][0]
            parent = args.output / f"repeat_{item['repeat']}" / source['cases'][0]['id']
            parent.mkdir(parents=True, exist_ok=True)
            result = evaluate(env, model, [seed], parent, label, record_video=True)[0]
            result.update(task_id=source['cases'][0]['id'], repeat=item['repeat'])
            options, count, timing = audit_timing_episode(parent / f'{label}_{seed}', result, manifest, label == 'learned')
            result['timing_summary'] = timing
            if seed in resets:
                assert resets[seed] == options
            resets[seed] = options
            transitions += count
            results.append(result)
            (args.output / 'results.json').write_text(json.dumps(results, indent=2))
            (args.output / 'summary.json').write_text(json.dumps({'per_task': {'scandinavian_capture': summarize(results)},
                                                               'completed_episodes': len(results), 'expected_episodes': len(plan)}, indent=2))
            write_video_index(args.output, results, manifest['labels'], manifest['objective'])
        assert len(results) == len(plan)
        assert file_hash(actor_path) == actor_sha
        assert file_hash(Path(policy.checkpoint) / 'model.safetensors') == policy.weights_sha256
        integrity = {'passed': True, 'episodes': len(results), 'transitions': transitions,
                     'weights_unchanged': True, 'matched_resets': True, 'no_recorded_trajectories_used': True,
                     'timing_and_frames_verified': True, 'elapsed_seconds': time.perf_counter() - started}
        (args.output / 'integrity.json').write_text(json.dumps(integrity, indent=2))
        (args.output / 'status.json').write_text(json.dumps({'status': 'complete'}))
        print('LIVE_TIMING_EVALUATION_COMPLETE', json.dumps(integrity), flush=True)
    except Exception as error:
        (args.output / 'status.json').write_text(json.dumps({'status': 'failed', 'error': str(error)}, indent=2))
        raise
    finally:
        for env in envs.values():
            env.close()


if __name__ == '__main__':
    main()
