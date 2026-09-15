"""Run a legacy ACT checkpoint against the synchronous chess simulator.

Records diagnostic simulated transitions, not expert demonstrations or a
LeRobot dataset. Only localhost simulation endpoints are used; no robot SDK.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np

from camera_client import PairedCameraClient
from policy_overlay import highlight_move


class LegacyACT:
    def __init__(self, checkpoint, device='cpu', n_action_steps=None):
        import draccus
        import torch
        from safetensors.torch import load_file
        from lerobot.policies.act.configuration_act import ACTConfig
        from lerobot.policies.act.modeling_act import ACTPolicy
        self.torch, self.device = torch, device
        torch.set_num_threads(4)
        checkpoint = Path(checkpoint).resolve()
        config = json.loads((checkpoint / 'config.json').read_text())
        if config.pop('type') != 'act':
            raise ValueError('Expected an ACT checkpoint')
        if any(mode != 'MEAN_STD' for mode in config['normalization_mapping'].values()):
            raise ValueError('This compatibility adapter requires MEAN_STD normalization')
        config['device'] = device
        config['pretrained_backbone_weights'] = None  # checkpoint supplies all weights; no download
        if n_action_steps is not None:
            config['n_action_steps'] = n_action_steps
        policy_config = draccus.decode(ACTConfig, config)
        self.policy = ACTPolicy(policy_config)
        weights = load_file(str(checkpoint / 'model.safetensors'))
        self.policy.load_state_dict({k: v for k, v in weights.items() if k.startswith('model.')}, strict=True)
        self.policy.to(device).eval()
        self.stats = {}
        expected = set()
        for feature in config['input_features']:
            prefix = 'normalize_inputs.buffer_' + feature.replace('.', '_')
            self.stats[feature] = self._stats(weights, prefix)
            expected.update([prefix + '.mean', prefix + '.std'])
        self.action_stats = self._stats(weights, 'unnormalize_outputs.buffer_action')
        expected.update(['unnormalize_outputs.buffer_action.mean', 'unnormalize_outputs.buffer_action.std',
                         'normalize_targets.buffer_action.mean', 'normalize_targets.buffer_action.std'])
        unknown = {k for k in weights if not k.startswith('model.')} - expected
        if unknown:
            raise ValueError('Unrecognized checkpoint buffers: ' + ', '.join(sorted(unknown)))
        self.config = config
        self.checkpoint = str(checkpoint)
        with (checkpoint / 'model.safetensors').open('rb') as stream:
            self.weights_sha256 = hashlib.file_digest(stream, 'sha256').hexdigest()

    def _stats(self, weights, prefix):
        mean, std = [weights[prefix + suffix].to(self.device) for suffix in ('.mean', '.std')]
        if not self.torch.isfinite(mean).all() or not self.torch.isfinite(std).all() or (std < 0).any():
            raise ValueError('Invalid saved normalization statistics')
        return mean, std

    def reset(self):
        self.policy.reset()

    def action(self, state, laptop_rgb, phone_rgb):
        torch = self.torch
        arrays = {'observation.state': np.asarray(state, np.float32),
                  'observation.images.laptop': laptop_rgb,
                  'observation.images.phone': phone_rgb}
        batch = {}
        for key, array in arrays.items():
            value = torch.from_numpy(np.ascontiguousarray(array)).to(self.device).float()
            if value.ndim == 3:
                value = value.permute(2, 0, 1) / 255.0
            if list(value.shape) != self.config['input_features'][key]['shape']:
                raise ValueError(f'Wrong policy feature shape: {key}: {value.shape}')
            mean, std = self.stats[key]
            batch[key] = ((value - mean) / (std + 1e-8)).unsqueeze(0)
        with torch.inference_mode():
            normalized = self.policy.select_action(batch)
            mean, std = self.action_stats
            action = normalized * std + mean
        result = action[0].cpu().numpy()
        if result.shape != (6,) or not np.isfinite(result).all():
            raise ValueError('Invalid ACT output')
        return result


class HomeReturnMonitor:
    """End a move after leaving home and returning for ten consecutive steps."""
    def __init__(self, home_deg):
        self.home = np.asarray(home_deg)[:5]
        self.departed = False
        self.stable_steps = 0

    def update(self, joint_deg):
        error = float(np.max(np.abs(np.asarray(joint_deg)[:5] - self.home)))
        self.departed |= error > 15
        self.stable_steps = self.stable_steps + 1 if self.departed and error < 5 else 0
        return self.stable_steps >= 10


def command(url, path, frame_id, action=None):
    payload = {'frame_id': frame_id}
    if action is not None:
        payload['action'] = np.asarray(action).tolist()
    req = Request(url.rstrip('/') + path, data=json.dumps(payload).encode(),
                  headers={'Content-Type': 'application/json'}, method='POST')
    try:
        with urlopen(req, timeout=35) as response:
            return json.load(response)
    except HTTPError as error:
        raise RuntimeError(f'Simulator rejected {path}: {error.read().decode()}') from error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=Path('/home/jacob/lerobot/outputs/train/chess_game_multi_stereo/checkpoints/last/pretrained_model'))
    parser.add_argument('--url', default='http://127.0.0.1:8080')
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    parser.add_argument('--steps', type=int, default=400)
    parser.add_argument('--episodes', type=int, default=1)
    parser.add_argument('--move', default='e2e4')
    parser.add_argument('--n-action-steps', type=int)
    parser.add_argument('--calibration', type=Path, default=Path('calibration/sim_board.json'))
    parser.add_argument('--output', type=Path, default=Path('outputs/act') / time.strftime('%Y%m%d-%H%M%S'))
    args = parser.parse_args()
    if args.steps < 1 or args.episodes < 1:
        parser.error('steps and episodes must be positive')
    if args.move != 'e2e4':
        parser.error('The current reset board and success measurement support blue e2e4; other tasks need scene setup')
    args.output.mkdir(parents=True, exist_ok=False)
    policy = LegacyACT(args.checkpoint, args.device, args.n_action_steps)
    print('ACT_LOADED', policy.checkpoint, flush=True)
    calibration = json.loads(args.calibration.read_text())['cameras']['red']
    client = PairedCameraClient(args.url)
    results = []
    for episode in range(args.episodes):
        path = args.output / f'episode_{episode:06d}'
        path.mkdir()
        metadata, images = client.read(color_order='rgb')
        if 'observation' not in metadata:
            raise RuntimeError('Start camera_server with --control')
        command(args.url, '/reset', metadata['frame_id'])
        metadata, images = client.read(color_order='rgb')
        home_deg = np.asarray(metadata['observation']['measured_joint_deg'])
        home_monitor = HomeReturnMonitor(home_deg)
        policy.reset()
        completed, failure, termination = 0, None, 'step_limit'
        manifest = {'format': 'chess_sim_transitions_v1', 'source': 'ACT simulation, not expert demonstrations',
                    'checkpoint': policy.checkpoint, 'weights_sha256': policy.weights_sha256,
                    'policy_config': policy.config, 'move': args.move, 'device': args.device,
                    'initial_observation': metadata, 'camera_mapping': {'laptop': 'red highlighted', 'phone': 'blue raw'}}
        (path / 'manifest.json').write_text(json.dumps(manifest, indent=2))
        try:
            for step in range(args.steps):
                action = None
                obs = metadata['observation']
                laptop = highlight_move(np.ascontiguousarray(images['red'][..., ::-1]), calibration, args.move)[..., ::-1].copy()
                started = time.perf_counter()
                action = policy.action(obs['observation_state'], laptop, images['blue'])
                elapsed = time.perf_counter() - started
                response = command(args.url, '/step', metadata['frame_id'], action)
                next_meta, next_images = client.read(color_order='rgb')
                if next_meta['frame_id'] != response['frame_id']:
                    raise RuntimeError('Another controller advanced the simulator during this transition')
                np.savez_compressed(path / f'frame_{step:06d}.npz',
                    observation_state=np.asarray(obs['observation_state'], np.float32),
                    laptop_rgb=laptop, phone_rgb=images['blue'], red_raw_rgb=images['red'],
                    action_legacy=action, next_observation_state=np.asarray(next_meta['observation']['observation_state'], np.float32),
                    frame_id=metadata['frame_id'], next_frame_id=next_meta['frame_id'],
                    simulation_time=metadata['simulation_time'], next_simulation_time=next_meta['simulation_time'],
                    inference_seconds=elapsed)
                metadata, images = next_meta, next_images
                completed += 1
                if home_monitor.update(metadata['observation']['measured_joint_deg']):
                    termination = 'returned_home'
                    break
                if step % 50 == 0:
                    print(json.dumps({'episode': episode, 'step': step, 'inference_s': elapsed,
                                      'action': action.tolist()}), flush=True)
        except (RuntimeError, ValueError, OSError) as error:
            failure = str(error)
            termination = 'error'
            (path / 'failure.json').write_text(json.dumps({'error': failure, 'observation': metadata,
                'requested_action': None if action is None else action.tolist()}, indent=2))
            print('ROLLOUT_STOPPED', failure, flush=True)
        obs = metadata['observation']
        pieces = obs.get('pieces', {})
        home_error = float(np.abs(np.asarray(obs['measured_joint_deg'])[:5] - home_deg[:5]).max())
        success = (pieces.get('pawn_lift_max_m', 0) > 0.005 and
                   pieces.get('pawn_target_xy_error_m', float('inf')) < 0.005 and
                   pieces.get('pawn_upright_cosine', 0) > 0.95 and
                   abs(pieces.get('pawn_height_change_m', float('inf'))) < 0.003 and
                   pieces.get('other_pieces_displaced_over_3mm', 1) == 0 and home_error < 5 and not failure)
        result = {'episode': episode, 'steps': completed, 'failure': failure, 'success': bool(success), 'termination': termination,
                  'home_error_max_deg': home_error, 'final_observation': metadata}
        (path / 'result.json').write_text(json.dumps(result, indent=2))
        # Save the terminal images/state so the final recorded transition is recoverable.
        np.savez_compressed(path / 'terminal.npz', blue_rgb=images['blue'], red_rgb=images['red'],
                            observation_json=np.array(json.dumps(metadata)))
        results.append(result)
        (args.output / 'results.json').write_text(json.dumps(results, indent=2))
    print('ROLLOUT_RESULTS', str(args.output / 'results.json'), flush=True)


if __name__ == '__main__':
    main()
