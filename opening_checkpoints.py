"""Evaluate saved actors at episode boundaries without inserting eval into replay."""
import json
import random
import time

import numpy as np
import torch

from train_residual import TrainingLog


class OpeningCheckpoints(TrainingLog):
    def __init__(self, output, env, cases, seeds, interval, total, evaluate):
        super().__init__(output)
        self.raw_env, self.cases, self.seeds = env, cases, seeds
        self.interval, self.next_check, self.total = interval, interval, total
        self.evaluate = evaluate
        self.results = {}
        self.selections = []
        self.best_successes = -1
        self.best_label = None
        self.evaluation_seconds = 0.

    def consider(self, model, label, rows, step):
        successes = sum(row['score']['success'] for row in rows)
        self.selections.append({'label': label, 'step': step, 'successes': successes, 'episodes': len(rows)})
        if successes > self.best_successes:
            model.save(self.output / 'residual_best')
            self.best_successes, self.best_label = successes, label
        (self.output / 'checkpoint_selection.json').write_text(json.dumps({
            'criterion': 'most successes on selection seeds; ties retain earlier checkpoint',
            'best_label': self.best_label, 'results': self.selections,
            'note': 'Selection scores are not independent held-out reliability estimates.'}, indent=2))

    def _on_rollout_end(self):
        # SAC has stored the terminal transition and auto-reset the VecEnv by
        # this hook. Avoid interrupting episodes or mixing eval into replay.
        if not self.interval or self.num_timesteps < self.next_check or self.num_timesteps >= self.total:
            return
        if not bool(self.locals['dones'][0]):
            return
        rng = (random.getstate(), np.random.get_state(), torch.get_rng_state(),
               torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
        base = self.raw_env.unwrapped
        cursor = base.task_cursor
        env_rng = base.np_random.bit_generator.state.copy()
        label = f'checkpoint_{self.num_timesteps}'
        self.model.save(self.output / f'residual_{self.num_timesteps}')
        started = time.perf_counter()
        try:
            rows = self.evaluate(self.raw_env, self.model, self.cases, self.seeds, self.output, label)
            self.results[label] = rows
            self.consider(self.model, label, rows, self.num_timesteps)
        finally:
            base.fixed_case = None
            base.task_cursor = cursor - 1  # replay the unused auto-reset task
            base.np_random.bit_generator.state = env_rng
            self.model._last_obs = self.model.get_env().reset()
            self.model._last_episode_starts = np.ones((1,), dtype=bool)
            random.setstate(rng[0]); np.random.set_state(rng[1]); torch.set_rng_state(rng[2])
            if rng[3] is not None:
                torch.cuda.set_rng_state_all(rng[3])
            self.evaluation_seconds += time.perf_counter() - started
        self.next_check = (self.num_timesteps // self.interval + 1) * self.interval
