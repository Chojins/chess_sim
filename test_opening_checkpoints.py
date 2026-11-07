import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from opening_checkpoints import OpeningCheckpoints
from test_residual_learning import ToyEnv
from train_residual import make_learner


class CheckpointTests(unittest.TestCase):
    def test_evaluation_only_at_boundaries_and_training_resumes(self):
        class Env(ToyEnv):
            task_cursor = 0
            fixed_case = None
            def reset(self, **kwargs):
                result = super().reset(**kwargs)
                self.task_cursor += 1
                return result
        class Callback(OpeningCheckpoints):
            # The real TrainingLog requires chess telemetry, irrelevant here.
            def _on_training_start(self): pass
            def _on_step(self): return True
        torch.set_num_threads(2)
        env = Env()
        calls = []
        def evaluate(raw, model, cases, seeds, output, label):
            calls.append(model.num_timesteps)
            raw.reset(seed=99)
            raw.step(np.ones(6, np.float32))
            return [{'score': {'success': True}}]
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            cb = Callback(out, env, [], [99], 95, 320, evaluate)
            model = make_learner(env, 'cpu', 12, buffer_size=512)
            model.learn(320, callback=cb)
            self.assertEqual(calls, [100, 200, 300])
            self.assertEqual(model.replay_buffer.size(), 320)
            self.assertEqual(int(model.replay_buffer.dones[:320].sum()), 16)
            self.assertEqual(model._n_updates, 64)
            self.assertEqual(cb.best_label, 'checkpoint_100')
            self.assertTrue((out / 'residual_best.zip').exists())
            self.assertEqual(len(json.loads((out / 'checkpoint_selection.json').read_text())['results']), 3)
