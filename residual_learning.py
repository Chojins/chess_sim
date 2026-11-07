"""Compact frozen ACT vision features and a grasp-focused residual wrapper."""
import time
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import torch.nn.functional as functional

from chess_env import ResidualACTEnv


def grasp_gate(joint_deg):
    """Smooth gate derived only from measured shoulder angle, not object state."""
    shoulder = float(joint_deg[1])
    return float(np.clip((shoulder - 60) / 20, 0, 1) * np.clip((140 - shoulder) / 20, 0, 1))


def approach_potential(gate, clearance_m):
    return float(gate * np.clip(1 - clearance_m / .03, 0, 1))


def reward_clearance(info):
    """Privileged shaping target: capture victim until release, then attacker."""
    observation = info['telemetry']['observation']
    geometry = observation.get('task_geometry', {})
    capture = geometry.get('capture_index')
    targets = observation['pieces'].get('piece_grasp_targets_m', observation['pieces'].get('piece_centers_m'))
    if capture is not None and info.get('score', {}).get('phase', 'remove_captured') == 'remove_captured':
        center = targets[capture]
    elif 'source_index' in geometry:
        center = targets[geometry['source_index']]
    else:
        center = observation['pieces']['pawn_center_m']
    # Legacy servers retain their historical tip-based shaping. New opening
    # runs require the explicit grasp reference in OpeningChessEnv.reset.
    point = observation.get('gripper_grasp_center_m', observation['gripper_tip_proxy_m'])
    return float(np.linalg.norm(np.asarray(point) - center))


def placement_potential(info):
    """Guide lifted pieces to their release location; no actor-only shortcuts.

    Actual piece position is scored, not the tool position or finger command.
    Capture victims use the tabletop zone until released, then the attacker
    uses its destination square. This signal is available only to the reward.
    """
    score = info.get('score', {})
    observation = info['telemetry']['observation']
    geometry = observation.get('task_geometry', {})
    victim = geometry.get('capture_index')
    removing = victim is not None and score.get('phase', 'remove_captured') == 'remove_captured'
    if removing:
        if not (score.get('capture') or {}).get('lifted', False):
            return 0.
        center = np.asarray(observation['pieces']['piece_centers_m'][victim])
        zone = geometry['capture_zone']
        xy = np.clip(center[:2], zone['min_xy_m'], zone['max_xy_m'])
        target = np.r_[xy, geometry['capture_rest_center_z_m']]
    else:
        if not score.get('lifted', False):
            return 0.
        pieces = observation['pieces']
        center = np.asarray(pieces['piece_centers_m'][geometry['source_index']]
                            if 'source_index' in geometry else pieces['pawn_center_m'])
        target = np.asarray(geometry['target_center_m'])
    delta = center - target
    if not np.isfinite(delta).all():
        raise ValueError('Non-finite placement telemetry')
    return float(np.exp(-np.sum((delta[:2] / .0225) ** 2) - (delta[2] / .020) ** 2))


def combined_potential(gate, info):
    return approach_potential(gate, reward_clearance(info)) + 2. * placement_potential(info)


def filter_residual(action, gate, previous, bounds, alpha=1., max_change_deg=None):
    """Smooth corrections only; the frozen ACT joint targets are unchanged."""
    desired = np.asarray(action, float) * gate
    previous = np.asarray(previous, float)
    delta = alpha * (desired - previous)
    if max_change_deg is not None:
        rate = np.asarray(max_change_deg, float) / np.asarray(bounds, float)
        delta = np.clip(delta, -rate, rate)
    return np.clip(previous + delta, -1., 1.).astype(np.float32)


def proprio_features(state, base_action, previous, step, max_steps, mapping, chunk_phase=None):
    """Shared live/offline features; all inputs are observable on the robot."""
    scale = np.asarray(mapping['scale'])
    q = np.asarray(mapping['sim_home_deg']) + scale * (np.asarray(state) - mapping['legacy_home'])
    gate = grasp_gate(q)
    phase = (step % 100) / 100 if chunk_phase is None else chunk_phase
    return np.r_[q / 180, scale * (np.asarray(base_action) - state) / 30,
                 previous, phase, gate, step / max_steps].astype(np.float32), gate



class FrozenACTFeatures:
    """Reuse the checkpoint backbone; no downloaded or trainable vision model."""
    def __init__(self, policy):
        self.policy = policy
        self.backbone = policy.policy.model.backbone
        self.backbone.requires_grad_(False)
        self.backbone.eval()
        self.output_dim = 2 * 512 * 2 * 2  # validated resnet18 checkpoint
        if policy.config['vision_backbone'] != 'resnet18':
            raise ValueError('This feature adapter is validated for resnet18')

    def __call__(self, obs):
        images = []
        for name, key in [('laptop_rgb', 'observation.images.laptop'), ('phone_rgb', 'observation.images.phone')]:
            value = torch.from_numpy(np.ascontiguousarray(obs[name])).to(self.policy.device).float().permute(2, 0, 1) / 255
            mean, std = self.policy.stats[key]
            images.append((value - mean) / (std + 1e-8))
        with torch.inference_mode():
            features = self.backbone(torch.stack(images))['feature_map']
            pooled = functional.adaptive_avg_pool2d(features, (2, 2)).flatten(1)
            pooled = functional.normalize(pooled, dim=1)
        result = pooled.flatten().cpu().numpy().copy()
        if result.shape != (self.output_dim,) or not np.isfinite(result).all():
            raise ValueError('Unexpected ACT feature shape or values')
        return result


class LearningResidualEnv(gym.Wrapper):
    """Real-observable actor inputs; privileged clearance is used only in reward.

    Potential shaping uses the learner's discount, with zero terminal potential
    for true terminations and retained potential for time-limit bootstrapping.
    Success/failure criteria are supplied by the underlying chess task.
    """
    def __init__(self, env, policy, gamma=.99, shaping_weight=2., encoder=None,
                 bounds_deg=(2,2,2,2,2,3), residual_alpha=1., max_residual_change_deg=None):
        super().__init__(ResidualACTEnv(env, policy, bounds_deg=bounds_deg))
        if not 0 < gamma <= 1 or shaping_weight < 0 or not np.isfinite(shaping_weight):
            raise ValueError('Invalid shaping parameters')
        self.encoder = encoder or FrozenACTFeatures(policy)
        if not np.isfinite(residual_alpha) or not 0 < residual_alpha <= 1:
            raise ValueError('Residual filter alpha must be within (0,1]')
        rate = None if max_residual_change_deg is None else np.asarray(max_residual_change_deg, float)
        if rate is not None and (rate.shape != (6,) or not np.isfinite(rate).all() or np.any(rate <= 0)):
            raise ValueError('Residual rate limits must be six positive degree values')
        self.residual_alpha, self.max_residual_change_deg = residual_alpha, rate
        self.gamma, self.shaping_weight = gamma, shaping_weight
        self.observation_space = spaces.Box(-np.inf, np.inf, (self.encoder.output_dim + 21,), np.float32)
        self.previous_action = np.zeros(6, np.float32)
        self.potential = 0.
        self.gate = 0.

    def _features(self, obs):
        base = self.env.env
        step,max_steps = self.env.steps,base.task_config.max_steps
        clock = getattr(self.env.policy,'actor_clock',None)
        if clock is not None:
            step,max_steps = clock(step,max_steps)
        proprio, self.gate = proprio_features(obs['state'], obs['base_action'], self.previous_action,
                                             step, max_steps,
                                             {'scale': base.scale, 'sim_home_deg': base.sim_home,
                                              'legacy_home': base.legacy_home}, obs['chunk_phase'])
        return np.r_[self.encoder(obs), proprio].astype(np.float32)

    def _potential(self, info):
        return combined_potential(self.gate, info)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.previous_action[:] = 0
        result = self._features(obs)
        self.potential = self._potential(info)
        return result, info

    def step(self, action):
        action = np.asarray(action, np.float32)
        if not self.action_space.contains(action):
            raise ValueError('Residual action must be within [-1, 1]')
        applied_gate = self.gate
        residual_paused = bool(getattr(self.env.policy,'residual_paused',False))
        applied = (self.previous_action.copy() if residual_paused else
                   filter_residual(action, applied_gate, self.previous_action, self.env.bounds,
                                   self.residual_alpha, self.max_residual_change_deg))
        obs, reward, terminated, truncated, info = self.env.step(applied)
        self.previous_action = applied.copy()
        features_started = time.perf_counter()
        result = self._features(obs)
        info.setdefault("timing_ms", {})["vision_features_ms"] = (time.perf_counter()-features_started)*1000
        next_potential = 0. if terminated else self._potential(info)
        shaping = self.shaping_weight * (self.gamma * next_potential - self.potential)
        self.potential = next_potential
        info['learning'] = {'placement_potential': placement_potential(info),
                            **({'residual_paused':True} if residual_paused else {}),
                            'approach_potential': approach_potential(self.gate, reward_clearance(info)),
                            'correction_gate': applied_gate,
                            'shaping_reward': shaping, 'applied_residual': applied.tolist(),
                            'task_reward': reward + info['residual']['penalty']}
        return result, reward + shaping, terminated, truncated, info
