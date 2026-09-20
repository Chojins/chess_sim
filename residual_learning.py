"""Compact frozen ACT vision features and a grasp-focused residual wrapper."""
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
    Success/failure criteria remain e2e4_v1 regardless of shaping.
    """
    def __init__(self, env, policy, gamma=.99, shaping_weight=2., encoder=None):
        super().__init__(ResidualACTEnv(env, policy))
        if not 0 < gamma <= 1 or shaping_weight < 0 or not np.isfinite(shaping_weight):
            raise ValueError('Invalid shaping parameters')
        self.encoder = encoder or FrozenACTFeatures(policy)
        self.gamma, self.shaping_weight = gamma, shaping_weight
        self.observation_space = spaces.Box(-np.inf, np.inf, (self.encoder.output_dim + 21,), np.float32)
        self.previous_action = np.zeros(6, np.float32)
        self.potential = 0.
        self.gate = 0.

    def _features(self, obs):
        base = self.env.env
        q = base.sim_home + base.scale * (obs['state'] - base.legacy_home)
        delta = base.scale * (obs['base_action'] - obs['state'])
        self.gate = grasp_gate(q)
        proprio = np.r_[q / 180, delta / 30, self.previous_action, obs['chunk_phase'],
                        self.gate, self.env.steps / base.task_config.max_steps]
        return np.r_[self.encoder(obs), proprio].astype(np.float32)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.previous_action[:] = 0
        result = self._features(obs)
        o = info['telemetry']['observation']
        clearance = np.linalg.norm(np.array(o['gripper_tip_proxy_m']) - o['pieces']['pawn_center_m'])
        self.potential = approach_potential(self.gate, clearance)
        return result, info

    def step(self, action):
        action = np.asarray(action, np.float32)
        if not self.action_space.contains(action):
            raise ValueError('Residual action must be within [-1, 1]')
        applied = action * self.gate
        obs, reward, terminated, truncated, info = self.env.step(applied)
        self.previous_action = applied.copy()
        result = self._features(obs)
        next_potential = 0. if terminated else approach_potential(self.gate, info['score']['clearance_m'])
        shaping = self.shaping_weight * (self.gamma * next_potential - self.potential)
        self.potential = next_potential
        info['learning'] = {'shaping_reward': shaping, 'applied_residual': applied.tolist(),
                            'task_reward': reward + info['residual']['penalty']}
        return result, reward + shaping, terminated, truncated, info
