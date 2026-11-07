"""Warm-start the residual actor from verified supervised corrections."""
import argparse
import copy
import json
from pathlib import Path
import numpy as np
import torch
from stable_baselines3 import SAC
from capture_demonstrations import file_hash, load_demonstrations, supervised_loss


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--demonstrations', type=Path, required=True)
    parser.add_argument('--updates', type=int, default=1200)
    args = parser.parse_args()
    if args.updates < 1:
        parser.error('Updates must be positive')
    torch.set_num_threads(2)
    torch.manual_seed(1590)
    data, manifest = load_demonstrations(args.demonstrations)
    source = Path(manifest['source_model'])
    if file_hash(source) != manifest['source_model_sha256']:
        raise ValueError('Source checkpoint changed')
    output = args.demonstrations.parent / 'pretraining'
    output.mkdir(exist_ok=False)
    model = SAC.load(source, device='cuda')
    model.actor.optimizer = torch.optim.Adam(model.actor.parameters(), lr=1e-4)
    observations = torch.as_tensor(data['observations'], device=model.device)
    actions = torch.as_tensor(data['actions'], device=model.device)
    weights = torch.as_tensor(data['weights'], device=model.device)
    train = np.flatnonzero(~data['validation'])
    validation = np.flatnonzero(data['validation'])
    rng = np.random.default_rng(1590)
    def assess(indices):
        with torch.no_grad():
            return supervised_loss(model.actor, observations[indices], actions[indices], weights[indices]).item()
    initial = {'training_loss': assess(train), 'validation_loss': assess(validation)}
    best_loss = initial['validation_loss']
    best = copy.deepcopy(model.actor.state_dict())
    best_step = 0
    history = []
    for step in range(1, args.updates + 1):
        indices = rng.choice(train, size=64, replace=True)
        loss = supervised_loss(model.actor, observations[indices], actions[indices], weights[indices])
        model.actor.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.actor.parameters(), 1.)
        model.actor.optimizer.step()
        if step % 50 == 0 or step == args.updates:
            row = {'update': step, 'training_loss': assess(train), 'validation_loss': assess(validation)}
            history.append(row)
            if row['validation_loss'] < best_loss:
                best_loss, best_step = row['validation_loss'], step
                best = copy.deepcopy(model.actor.state_dict())
            print('SUPERVISED_UPDATE', json.dumps(row), flush=True)
    if best_step == 0:
        raise RuntimeError('Supervision did not improve validation fit; RL was not launched')
    model.actor.load_state_dict(best)
    model.save(output / 'demo_pretrained')
    if file_hash(source) != manifest['source_model_sha256']:
        raise ValueError('Source checkpoint changed')
    summary = {'initial': initial, 'selected_update': best_step, 'selected_validation_loss': best_loss,
               'updates': args.updates, 'history': history, 'note': 'Validation fit only, not physical pickup reliability'}
    (output / 'summary.json').write_text(json.dumps(summary, indent=2))
    (output / 'manifest.json').write_text(json.dumps({**manifest, 'supervised_pretraining': summary,
                                                    'pretrained_sha256': file_hash(output / 'demo_pretrained.zip')}, indent=2))
    print('SUPERVISED_PRETRAINING_READY', json.dumps(summary['initial']), best_loss, flush=True)


if __name__ == '__main__':
    main()
