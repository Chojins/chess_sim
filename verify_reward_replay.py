"""Positive control: score the known half-speed recorded e2e4 through Gym."""
import argparse
import json
from pathlib import Path

import numpy as np

from chess_env import ChessEnv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('outputs/reward-replay'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    env = ChessEnv(max_steps=800)
    try:
        obs, initial = env.reset(seed=0)
        (args.output / 'initial.json').write_text(json.dumps(initial, indent=2))
        with np.load('outputs/replay/episode_000000.npz') as data:
            actions = np.repeat(data['states'], 2, axis=0)
        total_reward = 0.
        with (args.output / 'scores.jsonl').open('w') as stream:
            for index, action in enumerate(actions):
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                stream.write(json.dumps({'step': index + 1, 'reward': reward, **info['score']}) + '\n')
                if index % 100 == 0:
                    print('REWARD_REPLAY', index, info['score'], flush=True)
                if terminated or truncated:
                    break
        report = {'steps': index + 1, 'reward': total_reward, 'score': info['score'],
                  'passed': info['score']['success'], 'final': env.latest}
        (args.output / 'report.json').write_text(json.dumps(report, indent=2))
        np.savez_compressed(args.output / 'terminal_images.npz', **obs)
        if not report['passed']:
            raise RuntimeError('Recorded positive control did not pass the task reward')
        print('PASS: recorded grasp, transport, release and home recognised', flush=True)
    finally:
        env.close()


if __name__ == '__main__':
    main()
