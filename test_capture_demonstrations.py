"""Numerical and learner checks for the supervised residual path."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from stable_baselines3 import SAC
from capture_demonstrations import (BOUNDS, FILTER, DemonstrationSAC, JointKinematics,
                                    invert_filtered, supervised_loss, validate_semantics)
from residual_learning import filter_residual, proprio_features
from test_residual_learning import ToyEnv
from train_residual import make_learner


class DemonstrationTests(unittest.TestCase):
    def stability_model(self):
        from stable_baselines3.common.logger import configure
        model=make_learner(ToyEnv(),'cpu',42,learner_class=DemonstrationSAC)
        model.set_logger(configure(None,[]))
        data={'observations':np.zeros((8,8),np.float32),'actions':np.zeros((8,6),np.float32),
              'weights':np.ones(8,np.float32),'validation':np.arange(8)>=6}
        data['actions'][data['validation']]=1.  # Must not enter the retention budget.
        model.configure_demonstrations(data,5.,42)
        return model

    def test_critic_warmup_and_actor_learning_rate_are_separate(self):
        model=self.stability_model()
        model.configure_stability(.1,30,1.15)
        self.assertEqual(model.initial_retention_loss,0.)
        model._update_learning_rate([model.actor.optimizer,model.critic.optimizer])
        self.assertEqual(model.actor.optimizer.param_groups[0]['lr'],0.)
        self.assertEqual(model.critic.optimizer.param_groups[0]['lr'],1e-4)
        model._n_updates=30
        model._update_learning_rate([model.actor.optimizer,model.critic.optimizer])
        self.assertAlmostEqual(model.actor.optimizer.param_groups[0]['lr'],1e-5)
        self.assertEqual(model.critic.optimizer.param_groups[0]['lr'],1e-4)

    def test_actor_update_that_loses_demonstrations_is_rolled_back(self):
        model=self.stability_model()
        model.configure_stability(.1,0,1.15,retention_loss_floor=.0001)
        before=copy.deepcopy(model.actor.state_dict())
        def ruin(*args,**kwargs):
            head=[module for module in model.actor.mu.modules() if isinstance(module,torch.nn.Linear)][-1]
            with torch.no_grad():head.bias.fill_(10.)
        with patch.object(SAC,'train',side_effect=ruin):
            model.train(1,4)
        self.assertEqual(model.actor_update_rejections,1)
        for key,value in model.actor.state_dict().items():torch.testing.assert_close(value,before[key])

    def test_filter_inversion_respects_live_rate_and_gating(self):
        rng = np.random.default_rng(42)
        for gate in (.05, .3, .8, 1.):
            for _ in range(30):
                previous = rng.uniform(-1, 1, 6)
                original = rng.uniform(-1, 1, 6)
                applied = filter_residual(original, gate, previous, BOUNDS, FILTER['alpha'], FILTER['max_change_deg'])
                recovered = invert_filtered(applied, gate, previous)
                np.testing.assert_allclose(filter_residual(recovered, gate, previous, BOUNDS, FILTER['alpha'], FILTER['max_change_deg']), applied, atol=2e-6)
                self.assertTrue(np.all(np.abs(recovered) <= 1))
        with self.assertRaises(ValueError):
            invert_filtered(np.ones(6), 0, np.zeros(6))

    def test_row_vector_fk_has_correct_rotation_direction(self):
        identity = np.eye(4).tolist()
        specification = {'names': ['pan', 'lift', 'elbow', 'wrist', 'roll', 'jaw'], 'limits': [[-180, 180]] * 6,
                         'tool_link': 'tool', 'tool_local': [1, 0, 0],
                         'joints': [{'name': 'pan', 'parent': 'base', 'child': 'tool', 'f0': identity,
                                     'f1_inverse': identity, 'parent_world': identity, 'axis': [0, 0, 1]}]}
        np.testing.assert_allclose(JointKinematics(specification).point([90, 0, 0, 0, 0, 0]), [0, 1, 0], atol=1e-12)

    def test_same_features_at_chunk_boundary_and_real_calibration(self):
        mapping = {'scale': [-1, -1, 1, 1, -1, 1.16], 'legacy_home': [0, 180, 130, 86, -90, 0],
                   'sim_home_deg': [-2, 18, -50, 56, 90, -2.5]}
        state = np.array([1, 100, 132, 90, -88, 5.], np.float32)
        action = state + .3
        previous = np.arange(6) / 10
        features, gate = proprio_features(state, action, previous, 100, 800, mapping)
        q = np.array(mapping['sim_home_deg']) + np.array(mapping['scale']) * (state - mapping['legacy_home'])
        np.testing.assert_allclose(features[:6], q / 180, atol=1e-7)
        np.testing.assert_allclose(features[6:12], np.array(mapping['scale']) * (action - state) / 30, atol=1e-7)
        np.testing.assert_allclose(features[12:18], previous)
        self.assertEqual(features[18], 0)
        self.assertEqual(features[-1], .125)
        self.assertEqual(features[-2], gate)

    def test_reject_old_actor_normalization(self):
        manifest = {'capture_alignment': True, 'task_version': 'capture_pickup_v1', 'observation_size': 4122,
                    'action_window': 100, 'residual_filter': FILTER, 'residual_bounds_deg': BOUNDS.tolist()}
        validate_semantics(manifest)
        old = copy.deepcopy(manifest)
        old['residual_bounds_deg'] = [2, 2, 2, 2, 2, 3]
        with self.assertRaises(ValueError):
            validate_semantics(old)

    def test_anchor_updates_actor_and_saved_model_loads_as_standard_sac(self):
        torch.set_num_threads(2)
        model = make_learner(ToyEnv(), 'cpu', 42, learner_class=DemonstrationSAC)
        data = {'observations': np.zeros((16, 8), np.float32), 'actions': np.full((16, 6), .25, np.float32),
                'weights': np.ones(16, np.float32), 'validation': np.arange(16) > 11}
        model.configure_demonstrations(data, .5, 42)
        obs = torch.zeros((8, 8))
        target = torch.full((8, 6), .25)
        weight = torch.ones(8)
        initial = supervised_loss(model.actor, obs, target, weight).item()
        model.learn(total_timesteps=280)
        self.assertEqual(model.demo_updates, model._n_updates)
        self.assertGreater(model.demo_updates, 0)
        self.assertLess(supervised_loss(model.actor, obs, target, weight).item(), initial)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'model'
            model.save(path)
            restored = SAC.load(path, device='cpu')
            np.testing.assert_allclose(model.predict(np.zeros(8, np.float32), deterministic=True)[0],
                                       restored.predict(np.zeros(8, np.float32), deterministic=True)[0])
            self.assertFalse(hasattr(restored, 'demo_observations'))


if __name__ == '__main__':
    unittest.main()
