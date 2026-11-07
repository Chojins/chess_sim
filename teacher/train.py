"""Short, independently controlled state-teacher SAC experiment.

This is single-environment bring-up, not the paper's parallel SGS pipeline.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor

from teacher.environment import StateTeacherEnv, TeacherConfig


class PriorSAC(SAC):
    """Optional training-only BC anchor; no imported demonstration transitions."""
    def configure_prior(self, data, weight, seed):
        train = ~data['validation']
        self.prior_observations = torch.as_tensor(data['observations'][train], device=self.device)
        self.prior_actions = torch.as_tensor(data['actions'][train], device=self.device)
        self.prior_rng = np.random.default_rng(seed)
        self.prior_weight = weight

    def _sample_action(self, learning_starts, action_noise=None, n_envs=1):
        # Populate fresh replay with the BC-initialized actor rather than
        # replacing its first 256 commands with uniformly random poses.
        if hasattr(self, 'prior_observations'):
            learning_starts = 0
        return super()._sample_action(learning_starts, action_noise, n_envs)

    def train(self, gradient_steps, batch_size=64):
        super().train(gradient_steps, batch_size)
        if not hasattr(self, 'prior_observations') or self.prior_weight == 0:
            return
        for _ in range(gradient_steps):
            indices = self.prior_rng.integers(len(self.prior_observations), size=batch_size)
            predicted = self.actor(self.prior_observations[indices], deterministic=True)
            loss = (predicted-self.prior_actions[indices]).square().mean()
            self.actor.optimizer.zero_grad()
            (self.prior_weight*loss).backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.)
            self.actor.optimizer.step()
        self.logger.record('train/initial_prior_loss', loss.item())

    def _excluded_save_params(self):
        return super()._excluded_save_params()+['prior_observations', 'prior_actions', 'prior_rng']


def actor_digest(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.actor.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def evaluate(env, model, seeds, directory):
    directory.mkdir(parents=True, exist_ok=False)
    rows = []
    for seed in seeds:
        obs, _ = env.reset(seed=seed)
        commands, rewards, grasp_errors = [], [], []
        with (directory/f'seed_{seed}.jsonl').open('x') as stream:
            for step in range(env.config.max_steps):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                commands.append(info['teacher']['command_joint_deg'])
                rewards.append(reward)
                actual=info['telemetry']['observation']
                target_index=(env.task.capture_index if env.task.capture_index is not None and not env.task.capture_bonus
                              else env.task.source_index)
                grasp_errors.append(float(np.linalg.norm(env.encoder.point(actual)-
                    actual['pieces']['piece_grasp_targets_m'][target_index])))
                stream.write(json.dumps({'step': step+1, 'action': action.tolist(),
                    'reward': reward, 'score': info['score'], 'teacher': info['teacher'],
                    'observation': info['telemetry']['observation']})+'\n')
                if terminated or truncated:
                    break
        q = np.asarray(commands)
        rows.append({'seed': seed, 'steps': len(q), 'return': float(sum(rewards)), 'score': info['score'],
                     'minimum_grasp_error_mm':min(grasp_errors)*1000,
                     'command_path_length_deg': float(np.linalg.norm(np.diff(q[:, :5], axis=0), axis=1).sum()),
                     'max_command_velocity_deg_s': np.max(np.abs(np.diff(q, axis=0))*30, axis=0).tolist() if len(q)>1 else [0.]*6})
    (directory/'results.json').write_text(json.dumps(rows, indent=2))
    return rows


class Progress(BaseCallback):
    def __init__(self, directory, requested):
        super().__init__()
        self.directory, self.requested = directory, requested
        self.started = time.perf_counter()
        self.episodes = []

    def _on_step(self):
        for info in self.locals['infos']:
            if 'episode' in info:
                self.episodes.append({'steps': self.num_timesteps, **info['episode'], 'score': info['score']})
        if self.num_timesteps % 100 == 0:
            status = {'stage': 'training', 'steps': self.num_timesteps, 'requested': self.requested,
                      'elapsed_seconds': time.perf_counter()-self.started, 'episodes': self.episodes}
            (self.directory/'status.json').write_text(json.dumps(status, indent=2))
            print('TEACHER_PROGRESS', json.dumps({k:v for k,v in status.items() if k!='episodes'}), flush=True)
        return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--url', default='http://127.0.0.1:8081')
    parser.add_argument('--steps', type=int, default=3000)
    parser.add_argument('--max-episode-steps', type=int, default=1200)
    parser.add_argument('--seed', type=int, default=30001)
    parser.add_argument('--eval-seeds', type=int, nargs='+', default=[31001, 31002])
    parser.add_argument('--bootstrap', type=Path, nargs='+', help='Audited trace directories; last whole episode is BC validation')
    parser.add_argument('--bc-updates', type=int, default=1500)
    parser.add_argument('--bc-weight', type=float, default=.1, help='Temporary warm-start retention; zero disables it')
    args = parser.parse_args()
    if (args.steps < 257 or args.seed < 0 or len(set(args.eval_seeds)) != len(args.eval_seeds)
            or not args.eval_seeds or min(args.eval_seeds) < 0 or args.seed in args.eval_seeds
            or args.bc_updates < 0 or not np.isfinite(args.bc_weight) or args.bc_weight < 0):
        parser.error('Invalid budget, seeds or BC configuration')
    config = TeacherConfig(max_steps=args.max_episode_steps)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'status.json').write_text(json.dumps({'stage':'connecting', 'steps_requested':args.steps}))
    torch.set_num_threads(4)
    env = None
    try:
        env = StateTeacherEnv(args.url, config=config)
        monitored = Monitor(env, filename=str(args.output/'episodes'))
        model = PriorSAC('MlpPolicy', monitored, learning_rate=1e-4, buffer_size=max(10000, args.steps),
                    learning_starts=256, batch_size=128, gamma=config.gamma, train_freq=1, gradient_steps=1,
                    ent_coef='auto_0.001', policy_kwargs={'net_arch':[256,256], 'log_std_init':-4.},
                    seed=args.seed, device='cpu', verbose=0)
        # Standard SAC uses a separate state-dependent log-std head.
        with torch.no_grad():
            model.actor.log_std.weight.zero_()
            model.actor.log_std.bias.fill_(-4.)
        model.set_logger(configure(str(args.output/'optimizer'), ['csv']))
        manifest = {**env.manifest(), 'seed': args.seed, 'evaluation_seeds': args.eval_seeds,
                    'steps_requested': args.steps, 'algorithm':'SAC', 'parallel_environments':1,
                    'sgs_sampling':False, 'rgb_distillation':False,
                    'status':'bring-up only; no claim of reliable or improved motion',
                    'source_hashes': {str(p):hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in sorted(Path('teacher').glob('*.py'))}}
        if args.bootstrap:
            from teacher.bootstrap import load_prior
            (args.output/'status.json').write_text(json.dumps({'stage':'auditing_bootstrap_sources'}))
            data, prior = load_prior(args.bootstrap, env.kinematics, env.anchors, config)
            manifest['bootstrap'] = prior
            manifest['bootstrap']['updates'] = args.bc_updates
            manifest['bootstrap']['retention_weight'] = args.bc_weight
            model.configure_prior(data, args.bc_weight, args.seed)
            rng = np.random.default_rng(args.seed)
            train = np.flatnonzero(~data['validation'])
            val_obs = torch.as_tensor(data['observations'][data['validation']])
            val_actions = torch.as_tensor(data['actions'][data['validation']])
            best_loss, best_actor = float('inf'), None
            (args.output/'status.json').write_text(json.dumps({'stage':'bootstrap', 'updates':args.bc_updates,
                'training_frames':len(train), 'validation_frames':len(val_obs)}))
            print('TEACHER_BOOTSTRAP', json.dumps({'updates':args.bc_updates, 'training_frames':len(train),
                                                 'validation_frames':len(val_obs)}), flush=True)
            for update in range(args.bc_updates+1):
                if update % 50 == 0:
                    with torch.no_grad():
                        loss = (model.actor(val_obs, deterministic=True)-val_actions).square().mean().item()
                    if loss < best_loss:
                        best_loss = loss
                        best_actor = {k:v.detach().clone() for k,v in model.actor.state_dict().items()}
                if update == args.bc_updates:
                    break
                selected = rng.choice(train, size=128, replace=True)
                pred = model.actor(torch.as_tensor(data['observations'][selected]), deterministic=True)
                loss = (pred-torch.as_tensor(data['actions'][selected])).square().mean()
                model.actor.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.actor.parameters(),1.)
                model.actor.optimizer.step()
            model.actor.load_state_dict(best_actor)
            model.actor.optimizer.state.clear()
            manifest['bootstrap']['selected_validation_mse'] = best_loss
        (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2))
        model.save(args.output/'teacher_initial')
        before_hash = actor_digest(model)
        (args.output/'status.json').write_text(json.dumps({'stage':'initial_evaluation'}))
        before = evaluate(env, model, args.eval_seeds, args.output/'before')
        # Reset training RNG after evaluation; the evaluation seeds never enter replay.
        monitored.reset(seed=args.seed)
        callback = Progress(args.output, args.steps)
        model.learn(args.steps, callback=callback, reset_num_timesteps=True)
        model.save(args.output/'teacher_final')
        model.save_replay_buffer(args.output/'replay_buffer.pkl')
        (args.output/'status.json').write_text(json.dumps({'stage':'final_evaluation'}))
        after = evaluate(env, model, args.eval_seeds, args.output/'after')
        status = {'stage':'pilot_complete', 'steps':model.num_timesteps, 'gradient_updates':model._n_updates,
                  'actor_changed':actor_digest(model)!=before_hash,
                  'before_successes':sum(bool(r['score']['success']) for r in before),
                  'after_successes':sum(bool(r['score']['success']) for r in after),
                  'evaluation_trials':len(after), 'ready_for_act_collection':False,
                  'next':'Assess full-move success and motion quality; build intermediate reset coverage and parallel training'}
        (args.output/'status.json').write_text(json.dumps(status, indent=2))
        print('TEACHER_PILOT_COMPLETE', json.dumps(status), flush=True)
    except BaseException as error:
        (args.output/'error.json').write_text(json.dumps({'error':repr(error)}, indent=2))
        (args.output/'status.json').write_text(json.dumps({'stage':'failed', 'error':repr(error)}, indent=2))
        raise
    finally:
        if env:
            env.close()


if __name__ == '__main__':
    main()
