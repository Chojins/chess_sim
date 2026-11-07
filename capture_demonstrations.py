"""Offline bounded grasp labels and supervised anchoring for residual SAC.

The actor sees the same camera/proprioceptive features as live inference.
Simulator geometry is used only to create training labels, never actor inputs.
"""
import hashlib
import copy
import json
from pathlib import Path
import numpy as np
import torch
from stable_baselines3 import SAC
from residual_learning import filter_residual

BOUNDS = np.array([6, 6, 6, 6, 6, 3], float)
FILTER = {'alpha': .25, 'max_change_deg': [.4, .4, .4, .4, .4, .6]}


def file_hash(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def validate_semantics(manifest, task_versions=('capture_pickup_v1',)):
    if (not manifest.get('capture_alignment') or manifest.get('task_version') not in task_versions
            or manifest.get('observation_size') != 4122 or manifest.get('action_window') != 100
            or manifest.get('residual_filter') != FILTER
            or manifest.get('residual_bounds_deg') != BOUNDS.tolist()):
        raise ValueError('Warm-start requires compatible task, feature, bound and residual-filter semantics')


class JointKinematics:
    def __init__(self, specification):
        self.specification = specification
        self.limits = np.array(specification['limits'])

    def point(self, q):
        angles = dict(zip(self.specification['names'], q))
        angles['jaw2_hinge'] = -q[5]
        result = {}
        for joint in self.specification['joints']:
            axis = np.array(joint['axis'], float)
            x, y, z = axis
            cross = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
            angle = np.deg2rad(angles[joint['name']])
            rotation = np.eye(4)
            rotation[:3, :3] = (np.eye(3) + np.sin(angle) * cross + (1 - np.cos(angle)) * (cross @ cross)).T
            parent = result.get(joint['parent'], np.array(joint['parent_world']))
            result[joint['child']] = np.array(joint['f1_inverse']) @ rotation @ joint['f0'] @ parent
        return (np.r_[self.specification['tool_local'], 1.] @ result[self.specification['tool_link']])[:3]


def invert_filtered(target, gate, previous):
    if gate <= 1e-8:
        raise ValueError('Cannot label a correction when the gate is closed')
    raw = np.clip((previous + (target - previous) / FILTER['alpha']) / gate, -1., 1.)
    actual = filter_residual(raw, gate, previous, BOUNDS, FILTER['alpha'], FILTER['max_change_deg'])
    if not np.allclose(actual, target, atol=2e-6):
        raise ValueError('Requested label cannot be reached by the live filter')
    return raw.astype(np.float32)


def centred_label(kinematics, base_q, previous, gate, recorded_raw, target):
    """One-step reachable correction; preserve the demonstrated jaw command."""
    original = filter_residual(recorded_raw, gate, previous, BOUNDS, FILTER['alpha'], FILTER['max_change_deg'])
    low = filter_residual(-np.ones(6), gate, previous, BOUNDS, FILTER['alpha'], FILTER['max_change_deg']).astype(float)
    high = filter_residual(np.ones(6), gate, previous, BOUNDS, FILTER['alpha'], FILTER['max_change_deg']).astype(float)
    active = np.flatnonzero(high[:5] - low[:5] > 1e-7)
    if not len(active):
        return np.asarray(recorded_raw, np.float32), 0., 0.
    def point(applied):
        return kinematics.point(np.clip(base_q + applied * BOUNDS, kinematics.limits[:, 0], kinematics.limits[:, 1]))
    def error(values):
        applied = original.astype(float).copy()
        applied[active] = values
        return np.r_[(point(applied) - target) * 1000, .1 * (values - original[active])]
    # Projected damped Gauss-Newton over a very small, reachable joint box.
    fitted = original[active].astype(float).clip(low[active], high[active])
    for _ in range(12):
        current = error(fitted)
        jacobian = np.stack([(error(fitted + np.eye(len(active))[j] * 1e-4) - current) / 1e-4
                             for j in range(len(active))], axis=1)
        delta = np.linalg.solve(jacobian.T @ jacobian + np.eye(len(active)) * .01, -jacobian.T @ current)
        accepted = False
        for fraction in (1., .5, .25, .125):
            candidate = np.clip(fitted + delta * fraction, low[active], high[active])
            if np.sum(error(candidate) ** 2) < np.sum(current ** 2) - 1e-10:
                fitted, accepted = candidate, True
                break
        if not accepted:
            break
    # Partial relabelling keeps successful trajectories close to their source.
    adjusted = original.astype(float).copy()
    adjusted[active] = .5 * original[active] + .5 * fitted
    before = float(np.linalg.norm(point(original) - target))
    after = float(np.linalg.norm(point(adjusted) - target))
    if after > before + 1e-8:
        return np.asarray(recorded_raw, np.float32), before, before
    label = invert_filtered(adjusted, gate, np.asarray(previous))
    label[5] = recorded_raw[5]
    return label, before, after


def load_demonstrations(path):
    path = Path(path)
    manifest = json.loads(path.with_suffix('.json').read_text())
    validate_semantics(manifest, ('capture_pickup_v1', 'legal_move_v5'))
    if file_hash(path) != manifest['dataset_sha256']:
        raise ValueError('Demonstration data hash mismatch')
    with np.load(path, allow_pickle=False) as archive:
        data = {key: archive[key].copy() for key in ('observations', 'actions', 'weights', 'validation', 'episode')}
    n = len(data['observations'])
    if (data['observations'].shape != (n, 4122) or data['actions'].shape != (n, 6)
            or any(data[k].shape != (n,) for k in ('weights', 'validation', 'episode'))
            or not all(np.isfinite(data[k]).all() for k in ('observations', 'actions', 'weights'))
            or np.any(np.abs(data['actions']) > 1) or np.any(data['weights'] <= 0)
            or not data['validation'].any() or data['validation'].all()):
        raise ValueError('Invalid demonstration arrays or split')
    if set(data['episode'][data['validation']]) & set(data['episode'][~data['validation']]):
        raise ValueError('Demonstration split leaks an episode')
    return data, manifest


def supervised_loss(actor, observations, actions, weights):
    errors = (actor(observations, deterministic=True) - actions).square().mean(dim=1)
    return (errors * weights).sum() / weights.sum()


class DemonstrationSAC(SAC):
    """One additional supervised actor update after each SAC training call."""
    def configure_demonstrations(self, data, weight, seed):
        mask = ~data['validation']
        self.demo_observations = torch.as_tensor(data['observations'][mask], device=self.device)
        self.demo_actions = torch.as_tensor(data['actions'][mask], device=self.device)
        self.demo_weights = torch.as_tensor(data['weights'][mask], device=self.device)
        self.demo_weight = float(weight)
        self.demo_rng = np.random.default_rng(seed)
        self.demo_updates = 0

    def configure_stability(self, actor_lr_scale=1., critic_warmup_updates=0,
                            retention_loss_ratio=None, retention_loss_floor=.0025):
        if (not 0 < actor_lr_scale <= 1 or critic_warmup_updates < 0 or
                retention_loss_ratio is not None and retention_loss_ratio < 1 or
                not np.isfinite(retention_loss_floor) or retention_loss_floor <= 0):
            raise ValueError('Invalid actor stability settings')
        self.actor_lr_scale = float(actor_lr_scale)
        self.critic_warmup_updates = int(critic_warmup_updates)
        self.actor_update_rejections = 0
        with torch.no_grad():
            reference = supervised_loss(self.actor, self.demo_observations, self.demo_actions, self.demo_weights).item()
        self.retention_loss_budget = (max(reference*retention_loss_ratio, retention_loss_floor)
                                      if retention_loss_ratio is not None else None)
        self.initial_retention_loss = reference

    def _update_learning_rate(self, optimizers):
        super()._update_learning_rate(optimizers)
        if hasattr(self, 'actor_lr_scale'):
            scale = 0. if self._n_updates < self.critic_warmup_updates else self.actor_lr_scale
            for group in self.actor.optimizer.param_groups:
                group['lr'] *= scale
            self.logger.record('train/actor_learning_rate', self.actor.optimizer.param_groups[0]['lr'])

    def train(self, gradient_steps, batch_size=64):
        budget = getattr(self, 'retention_loss_budget', None)
        if budget is not None:
            actor_before = copy.deepcopy(self.actor.state_dict())
            optimizer_before = copy.deepcopy(self.actor.optimizer.state_dict())
        super().train(gradient_steps, batch_size)
        for _ in range(gradient_steps):
            indices = self.demo_rng.integers(len(self.demo_actions), size=batch_size)
            loss = supervised_loss(self.actor, self.demo_observations[indices], self.demo_actions[indices], self.demo_weights[indices])
            self.actor.optimizer.zero_grad()
            (self.demo_weight * loss).backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.)
            self.actor.optimizer.step()
            self.demo_updates += 1
            self.logger.record('train/demonstration_loss', loss.item())
        if budget is not None:
            with torch.no_grad():
                retained = supervised_loss(self.actor, self.demo_observations, self.demo_actions, self.demo_weights).item()
            if not np.isfinite(retained) or retained > budget:
                self.actor.load_state_dict(actor_before)
                self.actor.optimizer.load_state_dict(optimizer_before)
                self.actor_update_rejections += 1
            self.logger.record('train/retention_loss', retained)
            self.logger.record('train/actor_update_rejections', self.actor_update_rejections)

    def _excluded_save_params(self):
        return super()._excluded_save_params() + ['demo_observations', 'demo_actions', 'demo_weights', 'demo_rng']
