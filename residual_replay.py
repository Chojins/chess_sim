"""Seed residual SAC with successful real simulator transitions, no oracle data."""
import json
from pathlib import Path
import numpy as np
from train_residual import evaluate


def collect_teacher_replay(env, model, cases, seeds, output, record_video=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    episodes = []
    for index, seed in enumerate(seeds):
        case = cases[index % len(cases)]
        env.unwrapped.fixed_case = case
        parent = output / case['id']
        parent.mkdir(exist_ok=True)
        row = evaluate(env, model, [seed], parent, 'teacher', record_video=record_video,
                       record_learning_transitions=True)[0]
        row['task_id'] = case['id']
        episodes.append(row)
        (output / 'results.json').write_text(json.dumps(episodes, indent=2))
    successful = [row for row in episodes if row['score']['success'] and row['score']['other_pieces_disturbed'] == 0]
    successful_seeds = {row['seed'] for row in successful}
    if len(successful_seeds) < 2:
        raise ValueError('Need successful teacher rollouts at two seeds for separate replay/retention validation')
    validation_seed = max(successful_seeds)
    observations, actions, weights, validation, episode_ids = [], [], [], [], []
    transitions, training_seeds = 0, []
    for number, row in enumerate(successful):
        path = output / row['task_id'] / f"teacher_{row['seed']}"
        # The environment and reward are unchanged from this collection to
        # model.learn. Fresh selection seeds never enter this replay buffer.
        with np.load(path / 'learning_transitions.npz', allow_pickle=False) as data:
            obs, raw, rewards = data['observations'], data['actions'], data['rewards']
            terminated, truncated = data['terminated'], data['truncated']
            if obs.shape != (row['steps'] + 1, env.observation_space.shape[0]) or raw.shape != (row['steps'], 6):
                raise ValueError('Teacher replay spaces differ from learner')
            if not terminated[-1] or truncated.any() or not np.isfinite(np.r_[obs.ravel(), raw.ravel(), rewards]).all():
                raise ValueError('Successful teacher replay is incomplete or nonfinite')
            is_validation = row['seed'] == validation_seed
            if not is_validation:
                for index in range(row['steps']):
                    model.replay_buffer.add(obs[index:index + 1], obs[index + 1:index + 2], raw[index:index + 1],
                                            rewards[index:index + 1], terminated[index:index + 1],
                                            [{'TimeLimit.truncated': False}])
                    transitions += 1
                training_seeds.append(row['seed'])
            observations.extend(obs[:-1])
            actions.extend(raw)
            weights.extend([1.] * len(raw))
            validation.extend([is_validation] * len(raw))
            episode_ids.extend([number] * len(raw))
    data = {'observations': np.asarray(observations, np.float32), 'actions': np.asarray(actions, np.float32),
            'weights': np.asarray(weights, np.float32), 'validation': np.asarray(validation, bool),
            'episode': np.asarray(episode_ids, np.int32)}
    np.savez_compressed(output / 'demonstrations.npz', **data)
    report = {'attempts': len(episodes), 'successful_episodes': len(successful),
              'replay_transitions': transitions, 'replay_seeds': training_seeds,
              'validation_seed': validation_seed, 'source_seeds': sorted(successful_seeds),
              'requested_seeds': seeds, 'observations': 'exact live frozen ACT features and real-observable proprioception',
              'labels': 'teacher raw residuals; no IK or object pose labels',
              'reward_source': 'exact matched learner reward from simulator transitions'}
    (output / 'replay.json').write_text(json.dumps(report, indent=2))
    env.unwrapped.fixed_case = None
    return data, report
