"""Check ACT CPU parity, frozen visual features and optional CUDA compute."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from act_client import LegacyACT
from residual_learning import FrozenACTFeatures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--output', type=Path, default=Path('outputs/rl-runtime-check.json'))
    args = parser.parse_args()
    device = 'cuda' if args.cuda else 'cpu'
    if args.cuda:
        x = torch.ones((32, 32), device='cuda')
        assert (x @ x).sum().item() == 32768
    policy = LegacyACT('/home/jacob/lerobot/outputs/train/chess_game_multi_stereo/checkpoints/last/pretrained_model', device)
    encoder = FrozenACTFeatures(policy)
    errors = []
    for seed in (11, 23, 37):
        directory = Path(f'outputs/act-window-pilot/seed_{seed}_window_100')
        with np.load(directory / 'initial_images.npz') as file:
            obs = {key: file[key] for key in file.files}
        policy.reset()
        action = policy.action(obs['state'], obs['laptop_rgb'], obs['phone_rgb'])
        expected = json.loads((directory / 'steps.jsonl').read_text().splitlines()[0])['action_legacy']
        errors.append(float(np.abs(action - expected).max()))
        # Different PyTorch/CUDA kernels can vary slightly, but unit errors or
        # a changed normalizer/checkpoint should fail this check decisively.
        np.testing.assert_allclose(action, expected, atol=.01, rtol=1e-4)
        features = encoder(obs)
        assert features.shape == (4096,) and np.isfinite(features).all()
        assert not any(p.requires_grad for p in encoder.backbone.parameters())
    result = {'device': device, 'torch': torch.__version__, 'cuda_build': torch.version.cuda,
              'first_action_max_errors_legacy_units': errors, 'frozen_visual_feature_size': 4096,
              'passed': True, 'weights_sha256': policy.weights_sha256}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
