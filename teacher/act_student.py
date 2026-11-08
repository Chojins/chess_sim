"""RGB/joint ACT student, initialized from the preserved legacy ACT weights.

Real replay can supply its original two views. Simulation adds a separate
desired-board image. Both paths share every parameter and legacy joint unit.
"""
import json
from pathlib import Path
import numpy as np
import torch
from act_client import LegacyACT


CAMERAS = ('laptop', 'phone', 'goal')


class GoalACT(LegacyACT):
    def __init__(self, checkpoint, device='cpu', n_action_steps=50):
        # Load the original two-view model strictly before extending its inputs.
        super().__init__(checkpoint, device, n_action_steps)
        from lerobot.configs.types import FeatureType, PolicyFeature
        key = 'observation.images.goal'
        self.policy.config.input_features[key] = PolicyFeature(FeatureType.VISUAL, (3, 480, 640))
        self.config['input_features'][key] = {'type': 'VISUAL', 'shape': [3, 480, 640]}
        self.stats[key] = (
            torch.tensor([.485, .456, .406], device=device).reshape(3, 1, 1),
            torch.tensor([.229, .224, .225], device=device).reshape(3, 1, 1))

    def batch(self, samples):
        """Normalize explicit public inputs; reject mixed camera counts in a batch."""
        count = {len(s['images']) for s in samples}
        if not samples or len(count) != 1 or next(iter(count)) not in (2, 3):
            raise ValueError('Expected one homogeneous two- or three-view batch')
        state = np.stack([s['state'] for s in samples]).astype(np.float32)
        if state.shape != (len(samples), 6) or not np.isfinite(state).all():
            raise ValueError('Expected six finite legacy joint values')
        mean, std = self.stats['observation.state']
        result = {'observation.state': (torch.as_tensor(state, device=self.device) - mean) / (std + 1e-8)}
        images = []
        for i, name in enumerate(CAMERAS[:next(iter(count))]):
            rgb = np.stack([s['images'][i] for s in samples])
            if rgb.dtype != np.uint8 or rgb.shape != (len(samples), 480, 640, 3):
                raise ValueError('Expected actual uint8 RGB images at 640 x 480')
            value = torch.as_tensor(rgb, device=self.device).permute(0, 3, 1, 2).float() / 255
            mean, std = self.stats['observation.images.' + name]
            images.append((value - mean) / (std + 1e-8))
        result['observation.images'] = images
        if 'actions' in samples[0]:
            actions = np.stack([s['actions'] for s in samples]).astype(np.float32)
            padding = np.stack([s['is_pad'] for s in samples])
            if (actions.shape != (len(samples), self.config['chunk_size'], 6)
                    or padding.shape != actions.shape[:2] or padding.dtype != np.bool_
                    or not np.isfinite(actions).all()):
                raise ValueError('Invalid episode-bounded action chunks')
            mean, std = self.action_stats
            result['action'] = (torch.as_tensor(actions, device=self.device) - mean) / (std + 1e-8)
            result['action_is_pad'] = torch.as_tensor(padding, device=self.device)
        return result

    def loss(self, samples):
        # ACTPolicy.forward assumes all configured cameras are always present.
        # Calling the shared model permits truthful two-view real replay without
        # inventing an unknown desired board for old demonstrations.
        batch = self.batch(samples)
        prediction, (mu, logvar) = self.policy.model(batch)
        valid = (~batch['action_is_pad']).unsqueeze(-1)
        l1 = ((prediction - batch['action']).abs() * valid).sum() / (valid.sum() * 6).clamp_min(1)
        kl = torch.zeros((), device=self.device)
        if mu is not None:
            kl = (-.5 * (1 + logvar - mu.square() - logvar.exp())).sum(-1).mean()
        loss = l1 + self.policy.config.kl_weight * kl
        return loss, {'l1': float(l1.detach()), 'kl': float(kl.detach())}

    def action(self, state, laptop_rgb, phone_rgb, goal_rgb):
        batch = self.batch([{'state': state, 'images': (laptop_rgb, phone_rgb, goal_rgb)}])
        self.policy.eval()
        with torch.inference_mode():
            # Manage the original ACT queue while using the explicit image list.
            if not self.policy._action_queue:
                chunk = self.policy.model(batch)[0][:, :self.config['n_action_steps']]
                self.policy._action_queue.extend(chunk.transpose(0, 1))
            action = self.policy._action_queue.popleft()
            mean, std = self.action_stats
            result = (action * std + mean)[0].float().cpu().numpy()
        if result.shape != (6,) or not np.isfinite(result).all():
            raise ValueError('Invalid student action')
        return result

    def save_copy(self, directory):
        from safetensors.torch import save_file
        directory = Path(directory); directory.mkdir(parents=True, exist_ok=False)
        weights = {k: v.detach().cpu().contiguous() for k, v in self.policy.state_dict().items()}
        for feature, (mean, std) in self.stats.items():
            prefix = 'normalize_inputs.buffer_' + feature.replace('.', '_')
            weights[prefix + '.mean'] = mean.detach().cpu().contiguous()
            weights[prefix + '.std'] = std.detach().cpu().contiguous()
        for prefix in ('normalize_targets.buffer_action', 'unnormalize_outputs.buffer_action'):
            for suffix, value in zip(('.mean', '.std'), self.action_stats):
                # Separate tensors: safetensors deliberately rejects shared storage.
                weights[prefix + suffix] = value.detach().cpu().clone().contiguous()
        save_file(weights, str(directory / 'model.safetensors'))
        (directory / 'config.json').write_text(json.dumps({'type': 'act', **self.config}, indent=2))


def limited_joint_target(desired_legacy, current_deg, mapping, low, high, max_velocity, fps=30):
    """Uniform joint slew: all six targets approach the prediction together."""
    desired = np.asarray(desired_legacy, float); current = np.asarray(current_deg, float)
    if desired.shape != (6,) or not np.isfinite(desired).all():
        raise ValueError('Expected six finite legacy ACT targets')
    physical = np.asarray(mapping['sim_home_deg']) + np.asarray(mapping['scale']) * (
        desired - np.asarray(mapping['legacy_home']))
    delta = np.clip(physical, low, high) - current
    delta /= max(1., float(np.max(np.abs(delta) / (np.asarray(max_velocity) / fps))))
    return np.clip(current + delta, low, high)
