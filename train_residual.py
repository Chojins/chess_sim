"""Short SAC residual experiment with paired, exploration-free evaluation.

The actor receives frozen ACT camera features, joints, the proposed action,
previous correction and phase. Simulator-only object state is reward-only.
"""
import argparse
from dataclasses import asdict
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import random
import subprocess
import time

import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback

from act_client import LegacyACT
from chess_env import ChessEnv
from residual_learning import LearningResidualEnv


def transition_record(info):
    telemetry = info['telemetry']
    return {'frame_id': telemetry['frame_id'], 'simulation_time': telemetry['simulation_time'],
            'state': telemetry['observation']['observation_state'],
            'placement_telemetry': {key: telemetry['observation'].get(key) for key in
                                    ('pieces', 'gripper_grasp_center_m', 'gripper_tip_proxy_m', 'measured_joint_deg', 'measured_finger_joint_deg', 'target_legacy_action')},
            'score': info['score'], 'residual': info['residual'], 'learning': info['learning'],
            **({'act_timing': info['act_timing']} if 'act_timing' in info else {})}


def initialize_actor(model):
    """Deterministic initial policy is zero correction; SDE explores coherently."""
    head = [m for m in model.actor.mu.modules() if isinstance(m, torch.nn.Linear)][-1]
    torch.nn.init.zeros_(head.weight)
    torch.nn.init.zeros_(head.bias)


def make_learner(env, device, seed, buffer_size=10000, learner_class=SAC):
    model = learner_class('MlpPolicy', env, device=device, seed=seed, learning_rate=1e-4,
                buffer_size=buffer_size, learning_starts=256, batch_size=64,
                gamma=.99, train_freq=1, gradient_steps=1, ent_coef='auto_0.01',
                use_sde=True, sde_sample_freq=12, use_sde_at_warmup=True,
                policy_kwargs={'net_arch': [128, 128], 'log_std_init': -1.}, verbose=0)
    initialize_actor(model)
    return model


class TrainingLog(BaseCallback):
    def __init__(self, output):
        super().__init__()
        self.output = output
        self.episodes = []
        self.episode_return = 0.
        self.stream = None

    def _on_training_start(self):
        self.stream = (self.output / 'training_steps.jsonl').open('w')

    def _on_step(self):
        info = self.locals['infos'][0]
        reward = float(self.locals['rewards'][0])
        self.episode_return += reward
        self.stream.write(json.dumps({'step': self.num_timesteps, 'reward': reward,
                                      'done': bool(self.locals['dones'][0]),
                                      'action': self.locals['actions'][0].tolist(),
                                      **transition_record(info)}) + '\n')
        if self.locals['dones'][0]:
            row = {'training_step': self.num_timesteps, 'return': self.episode_return, 'score': info['score']}
            self.episodes.append(row)
            (self.output / 'training_episodes.json').write_text(json.dumps(self.episodes, indent=2))
            print('TRAIN_EPISODE', json.dumps(row), flush=True)
            self.episode_return = 0.
        if self.num_timesteps % 250 == 0:
            self.model.save(self.output / 'residual_latest')
            self.stream.flush()
            print('TRAIN_STEP', self.num_timesteps, flush=True)
        return True

    def _on_training_end(self):
        if self.stream:
            self.stream.close()


def evaluate(env, model, seeds, output, label, record_video=False, profile=False, collect_provenance=None,
             record_learning_transitions=False):
    rows = []
    for seed in seeds:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        obs, initial = env.reset(seed=seed)
        path = output / f'{label}_{seed}'
        path.mkdir()
        (path / 'initial.json').write_text(json.dumps(initial, indent=2))
        np.savez_compressed(path / 'initial_images.npz', **env.unwrapped._observation())
        reward_sum, steps = 0., 0
        video = None
        collector, collection_error, info = None, None, None
        if collect_provenance is not None:
            from episode_collector import EpisodeCollector
            actor_digest = None
            if model is not None:
                digest = hashlib.sha256()
                for name, tensor in sorted(model.actor.state_dict().items()):
                    digest.update(name.encode())
                    digest.update(str(tensor.dtype).encode())
                    digest.update(str(tuple(tensor.shape)).encode())
                    digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
                actor_digest = digest.hexdigest()
            collector = EpisodeCollector(path / 'lossless_episode', initial,
                                          {**collect_provenance, 'label': label, 'seed': seed,
                                           'residual_actor_state_sha256': actor_digest,
                                           'residual_mode': 'frozen actor' if model is not None else 'zero correction'})
        if record_video:
            from evaluation_video import EpisodeVideo
            video = EpisodeVideo(path, env.unwrapped._observation())
        measurements = []
        learning_states, learning_actions, learning_rewards, learning_terminated, learning_truncated = [], [], [], [], []
        if record_learning_transitions:
            learning_states.append(np.asarray(obs, np.float32).copy())
        try:
            if video:
                video.write(env.unwrapped._observation())
            with (path / 'steps.jsonl').open('w') as stream:
                while True:
                    step_started = time.perf_counter()
                    before_images = env.unwrapped._observation() if collector else None
                    before_metadata = env.unwrapped.latest if collector else None
                    action = np.zeros(6, np.float32) if model is None else model.predict(obs, deterministic=True)[0]
                    predicted = time.perf_counter()
                    obs, reward, terminated, truncated, info = env.step(action)
                    stepped = time.perf_counter()
                    if record_learning_transitions:
                        learning_states.append(np.asarray(obs, np.float32).copy())
                        learning_actions.append(np.asarray(action, np.float32).copy())
                        learning_rewards.append(reward)
                        learning_terminated.append(terminated)
                        learning_truncated.append(truncated)
                    if collector:
                        collector.write(before_images, before_metadata, info)
                    if video:
                        video.write(env.unwrapped._observation())
                    if profile:
                        measurements.append({
                            **info.get("timing_ms", {}),
                            "actor_predict_ms": (predicted-step_started)*1000,
                            "env_step_ms": (stepped-predicted)*1000,
                            "video_frame_ms": (time.perf_counter()-stepped)*1000})
                    reward_sum += reward
                    steps += 1
                    stream.write(json.dumps({'step': steps, 'action': action.tolist(), 'reward': reward,
                                             'terminated': terminated, 'truncated': truncated,
                                             **transition_record(info)}) + '\n')
                    if terminated or truncated:
                        break
        except BaseException as error:
            collection_error = repr(error)
            raise
        finally:
            if video:
                video.close()
            if collector:
                collection = collector.finalize(info['score'] if info else {}, error=collection_error)
        row = {'seed': seed, 'label': label, 'steps': steps, 'return': reward_sum, 'score': info['score']}
        if record_learning_transitions:
            if len(learning_states) != steps + 1 or not (learning_terminated[-1] or learning_truncated[-1]):
                raise ValueError('Incomplete residual replay episode')
            replay_path = path / 'learning_transitions.npz'
            np.savez_compressed(replay_path, observations=np.asarray(learning_states, np.float32),
                                actions=np.asarray(learning_actions, np.float32), rewards=np.asarray(learning_rewards, np.float32),
                                terminated=np.asarray(learning_terminated, bool), truncated=np.asarray(learning_truncated, bool))
            row['learning_transitions'] = {'path': replay_path.name, 'frames': steps,
                                          'sha256': hashlib.sha256(replay_path.read_bytes()).hexdigest(),
                                          'purpose': 'exact learner observations and raw bounded actions for matched-objective SAC replay; not ACT targets'}
        if collector:
            row['collection'] = {'path': 'lossless_episode', 'accepted_for_act_training': collection['accepted_for_act_training'],
                                 'frames': collection['frames']}
        if video:
            row['video'] = video.verify()
            assert row['video']['frames'] == steps + 1
        if profile:
            (path / 'timing_steps.json').write_text(json.dumps(measurements))
            row['timing_ms'] = {key: {'mean': float(np.mean([m[key] for m in measurements if key in m])),
                                     'p95': float(np.percentile([m[key] for m in measurements if key in m], 95))}
                                for key in sorted({k for m in measurements for k in m})}
        rows.append(row)
        np.savez_compressed(path / 'terminal_images.npz', **env.unwrapped._observation())
        (path / 'result.json').write_text(json.dumps(row, indent=2))
        (output / f'{label}_results.json').write_text(json.dumps(rows, indent=2))
        print('EVALUATION', json.dumps(row), flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=3000)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--eval-seeds', nargs='+', type=int, default=[101, 102, 103])
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--url', default='http://127.0.0.1:8080')
    parser.add_argument('--checkpoint', type=Path, default=Path('/home/jacob/lerobot/outputs/train/chess_game_multi_stereo/checkpoints/last/pretrained_model'))
    args = parser.parse_args()
    if args.steps < 257 or args.seed < 0 or any(s < 0 or s == args.seed for s in args.eval_seeds):
        parser.error('Need >=257 training steps and nonnegative training/evaluation seeds that differ')
    if len(set(args.eval_seeds)) != len(args.eval_seeds):
        parser.error('Evaluation seeds must be unique')
    if args.device == 'cuda':
        # Fail before simulator resets or output creation if CUDA is unavailable.
        x = torch.ones(16, device='cuda')
        assert x.sum().item() == 16
    policy = LegacyACT(args.checkpoint, args.device, n_action_steps=100)
    import lerobot
    source_path = Path(lerobot.__file__).resolve().parent
    source_revision = subprocess.run(['git', '-C', str(source_path), 'rev-parse', 'HEAD'],
                                     capture_output=True, text=True, check=False).stdout.strip()
    base = ChessEnv(args.url, pawn_jitter_mm=.5)
    env = LearningResidualEnv(base, policy)
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = {'task_version': 'e2e4_v1', 'reward': 'e2e4_v1 + residual penalty + potential_v1',
        'gamma': .99, 'potential_weight': 2., 'task_config': asdict(base.task_config),
        'checkpoint': policy.checkpoint, 'weights_sha256': policy.weights_sha256,
        'steps': args.steps, 'training_seed': args.seed, 'evaluation_seeds': args.eval_seeds,
        'residual_bounds_deg': env.env.bounds.tolist(), 'action_window': 100,
        'actor_inputs': 'frozen camera features, measured joints, proposed action, previous correction, chunk phase, shoulder gate, episode phase',
        'device': args.device, 'torch': torch.__version__, 'cuda': torch.version.cuda,
        'packages': {name: version(name) for name in ('stable-baselines3', 'gymnasium', 'torchvision', 'numpy')},
        'lerobot_source': str(source_path), 'lerobot_revision': source_revision,
        'note': 'Short bring-up experiment; no reliability or sim-to-real claim'}
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    callback = TrainingLog(args.output)
    try:
        baseline = evaluate(env, None, args.eval_seeds, args.output, 'baseline')
        model = make_learner(env, args.device, args.seed)
        model.save(args.output / 'residual_initial')
        env.reset(seed=args.seed)  # SB3's first reset retains this seeded RNG stream.
        started = time.perf_counter()
        model.learn(total_timesteps=args.steps, callback=callback)
        model.save(args.output / 'residual_final')
        elapsed = time.perf_counter() - started
        learned = evaluate(env, model, args.eval_seeds, args.output, 'learned')
        with (Path(policy.checkpoint) / 'model.safetensors').open('rb') as stream:
            assert hashlib.file_digest(stream, 'sha256').hexdigest() == policy.weights_sha256
        summary = {'baseline_successes': sum(r['score']['success'] for r in baseline),
                   'learned_successes': sum(r['score']['success'] for r in learned),
                   'evaluation_episodes_each': len(args.eval_seeds), 'training_steps': model.num_timesteps,
                   'gradient_updates': model._n_updates, 'training_seconds': elapsed,
                   'act_checkpoint_unchanged': True}
        (args.output / 'summary.json').write_text(json.dumps(summary, indent=2))
        print('EXPERIMENT_RESULT', json.dumps(summary), flush=True)
    except Exception as error:
        (args.output / 'error.json').write_text(json.dumps({'error': str(error)}, indent=2))
        raise
    finally:
        if callback.stream and not callback.stream.closed:
            callback.stream.close()
        env.close()


if __name__ == '__main__':
    main()
