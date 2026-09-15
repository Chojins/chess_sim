"""Gymnasium adapter for one externally running chess camera/control server.

Actor observations contain camera images and legacy joint readings only.
Privileged simulator measurements are returned in info for scoring/critics.
The caller owns the server. This is a single-client, partially observed env.
"""
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from act_client import command
from camera_client import PairedCameraClient
from chess_task import ChessTask, TaskConfig
from policy_overlay import highlight_move


class ChessEnv(gym.Env):
    metadata = {'render_modes': ['rgb_array'], 'render_fps': 30}

    def __init__(self, url='http://127.0.0.1:8080', calibration='calibration/sim_board.json',
                 max_steps=400, pawn_jitter_mm=0., render_mode=None):
        if render_mode not in (None, 'rgb_array'):
            raise ValueError('Supported render mode: rgb_array')
        if not np.isfinite(pawn_jitter_mm) or not 0 <= pawn_jitter_mm <= 2:
            raise ValueError('Pawn jitter must be within 0..2 mm')
        self.render_mode = render_mode
        self.client = PairedCameraClient(url)
        self.url, self.pawn_jitter_mm = url, pawn_jitter_mm
        self.calibration = json.loads(Path(calibration).read_text())['cameras']['red']
        self.task_config = TaskConfig(max_steps=max_steps)
        self.latest, self.images = self.client.read(color_order='rgb')
        state = self.latest.get('observation', {})
        if state.get('mode') != 'policy_control' or 'task_geometry' not in state:
            raise RuntimeError('Start the updated camera server with --control --contacts')
        limits = np.asarray(state['joint_limits_deg'])
        mapping = state['joint_map']
        self.scale = np.array(mapping['scale'])
        self.sim_home = np.array(mapping['sim_home_deg'])
        self.legacy_home = np.array(mapping['legacy_home'])
        bounds = self.legacy_home[:, None] + (limits - self.sim_home[:, None]) / self.scale[:, None]
        self.action_space = spaces.Box(bounds.min(1).astype(np.float32), bounds.max(1).astype(np.float32))
        self.observation_space = spaces.Dict({
            'state': spaces.Box(-np.inf, np.inf, (6,), np.float32),
            'laptop_rgb': spaces.Box(0, 255, self.images['red'].shape, np.uint8),
            'phone_rgb': spaces.Box(0, 255, self.images['blue'].shape, np.uint8)})
        self.task = None
        self.needs_reset = True

    def _observation(self):
        laptop = highlight_move(np.ascontiguousarray(self.images['red'][..., ::-1]),
                                self.calibration, 'e2e4')[..., ::-1].copy()
        return {'state': np.asarray(self.latest['observation']['observation_state'], np.float32),
                'laptop_rgb': laptop, 'phone_rgb': self.images['blue'].copy()}

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = {} if options is None else dict(options)
        if set(options) - {'pawn_offset_mm'}:
            raise ValueError('Unknown reset options')
        if 'pawn_offset_mm' not in options:
            options['pawn_offset_mm'] = self.np_random.uniform(-self.pawn_jitter_mm, self.pawn_jitter_mm, 2).tolist()
        self.needs_reset = True
        current, _ = self.client.read(color_order='rgb')
        response = command(self.url, '/reset', current['frame_id'], options=options)
        self.latest, self.images = self.client.read(color_order='rgb')
        if self.latest['frame_id'] != response['frame_id']:
            raise RuntimeError('Concurrent controller changed the reset observation')
        self.task = ChessTask(self.latest['observation'], self.task_config)
        self.needs_reset = False
        return self._observation(), {'telemetry': self.latest, 'reset_options': options, 'task_version': self.task.version}

    def step(self, action):
        if self.needs_reset:
            raise RuntimeError('reset() is required before step()')
        action = np.asarray(action, np.float32)
        if not self.action_space.contains(action) or not np.isfinite(action).all():
            raise ValueError('Action must be six finite in-range legacy joint targets')
        before = self.latest
        # A network/capture failure has uncertain execution state. Never retry
        # or manufacture a transition from the previous camera observation.
        self.needs_reset = True
        response = command(self.url, '/step', before['frame_id'], action)
        latest, images = self.client.read(color_order='rgb')
        if latest['frame_id'] != response['frame_id'] or latest['frame_id'] != before['frame_id'] + 1:
            raise RuntimeError('Concurrent controller or mismatched camera frame')
        if not np.isclose(latest['simulation_time'] - before['simulation_time'], 1 / 30, atol=1e-6):
            raise RuntimeError('Unexpected simulated step duration')
        self.latest, self.images = latest, images
        reward, terminated, truncated, score = self.task.update(latest['observation'])
        self.needs_reset = terminated or truncated
        return self._observation(), reward, terminated, truncated, {
            'telemetry': latest, 'score': score, 'task_version': self.task.version}

    def render(self):
        return np.concatenate([self.images['blue'], self.images['red']], axis=1).copy()

    def close(self):
        self.needs_reset = True  # server remains owned by its launching process


class ResidualACTEnv(gym.Wrapper):
    """Bounded corrections in physical joint degrees over a frozen ACT policy.

    Does not train a learner. The base action and chunk phase are observable;
    images plus a phase value remain a POMDP (a recurrent learner may help).
    """
    def __init__(self, env, policy, bounds_deg=(2, 2, 2, 2, 2, 3)):
        super().__init__(env)
        self.policy = policy
        self.bounds = np.asarray(bounds_deg, float)
        if self.bounds.shape != (6,) or not np.isfinite(self.bounds).all() or np.any(self.bounds <= 0):
            raise ValueError('Residual bounds must be six positive degree values')
        self.action_space = spaces.Box(-1., 1., (6,), np.float32)
        self.observation_space = spaces.Dict({**env.observation_space.spaces,
            'base_action': spaces.Box(-np.inf, np.inf, (6,), np.float32),
            'chunk_phase': spaces.Box(0., 1., (1,), np.float32)})
        self.steps = 0

    def _with_base(self, obs, terminal=False):
        self.base = np.zeros(6, np.float32) if terminal else self.policy.action(
            obs['state'], obs['laptop_rgb'], obs['phone_rgb'])
        return {**obs, 'base_action': self.base.copy(),
                'chunk_phase': np.array([self.steps % self.policy.config['n_action_steps'] /
                                          self.policy.config['n_action_steps']], np.float32)}

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.policy.reset()
        self.steps = 0
        return self._with_base(obs), info

    def step(self, action):
        if self.env.needs_reset:
            raise RuntimeError('reset() is required before step()')
        residual = np.asarray(action, np.float32)
        if not self.action_space.contains(residual):
            raise ValueError('Residual action must be within [-1, 1]')
        requested = self.base + residual * self.bounds / self.env.scale
        executed = np.clip(requested, self.env.action_space.low, self.env.action_space.high).astype(np.float32)
        obs, reward, terminated, truncated, info = self.env.step(executed)
        self.steps += 1
        penalty = .01 * float(np.mean(residual ** 2))
        info['residual'] = {'requested_legacy': requested.tolist(), 'executed_legacy': executed.tolist(),
                            'limited': bool(np.any((requested < self.env.action_space.low) |
                                                   (requested > self.env.action_space.high))), 'penalty': penalty}
        return self._with_base(obs, terminated), reward - penalty, terminated, truncated, info
