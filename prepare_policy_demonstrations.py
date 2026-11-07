"""Recover successful selected residual rollouts for supervised retention.

These labels are the teacher's actual bounded residual actions. No IK labels
or object positions enter the actor. Pickup episodes are residual supervision
only and are never exported as complete ACT demonstrations.
"""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
import torch
from act_client import LegacyACT
from capture_demonstrations import BOUNDS, FILTER, file_hash, validate_semantics
from residual_learning import FrozenACTFeatures, filter_residual, proprio_features
from train_openings import move_features


def source_episodes(pilot, heldout):
    manifest = json.loads((pilot / 'manifest.json').read_text())
    selection = json.loads((pilot / 'checkpoint_selection.json').read_text())
    if not json.loads((pilot / 'integrity.json').read_text())['passed']:
        raise ValueError('Source training integrity audit did not pass')
    paths = list(pilot.glob(f"*/{selection['best_label']}_*/result.json"))
    if heldout:
        recorded = json.loads((heldout / 'manifest.json').read_text())
        if (not json.loads((heldout / 'integrity.json').read_text())['passed'] or
                recorded['model_hashes']['learned'] != file_hash(pilot / 'residual_best.zip')):
            raise ValueError('Held-out recordings do not match the selected source actor')
        paths += list(heldout.glob('repeat_*/*/learned_*/result.json'))
    sources = []
    for result_path in sorted(paths):
        result = json.loads(result_path.read_text())
        if result['score']['success'] and result['score']['other_pieces_disturbed'] == 0:
            sources.append((result_path.parent, result))
    if len({result['seed'] for _, result in sources}) < 3:
        raise ValueError('Need successful source episodes at three distinct seeds for retention split')
    return manifest, sources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot', type=Path, required=True)
    parser.add_argument('--heldout', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    manifest, sources = source_episodes(args.pilot, args.heldout)
    validate_semantics(manifest, ('capture_pickup_v1', 'legal_move_v5'))
    if not manifest.get('live_timing'):
        raise ValueError('Retention source requires the explicit live timing deployment manifest')
    cases = {case['id']: case for case in manifest['cases']}
    validation_seed = max(result['seed'] for _, result in sources)
    torch.set_num_threads(2)
    policy = LegacyACT(manifest['checkpoint'], 'cuda', n_action_steps=manifest['action_window'])
    if policy.weights_sha256 != manifest['weights_sha256']:
        raise ValueError('Source ACT weights changed')
    encoder = FrozenACTFeatures(policy)
    args.output.mkdir(parents=True, exist_ok=False)
    arrays = {key: [] for key in ('observations', 'actions', 'weights', 'validation', 'episode')}
    episodes = []
    for number, (path, result) in enumerate(sources):
        initial = json.loads((path / 'initial.json').read_text())
        before = initial['telemetry']['observation']
        mapping = before['joint_map']
        max_steps = initial.get('task_config', {}).get('max_steps', 800 if cases[path.parent.name]['capture'] else 400)
        rows = [json.loads(line) for line in (path / 'steps.jsonl').read_text().splitlines()]
        video = cv2.VideoCapture(str(path / 'cameras.mp4'))
        with np.load(path / 'initial_images.npz') as images:
            reset_images = {key: images[key].copy() for key in ('laptop_rgb', 'phone_rgb')}
        previous = np.zeros(6, np.float32)
        for i, row in enumerate(rows):
            ok, frame = video.read()
            if not ok or frame.shape != (480, 1280, 3):
                raise ValueError('Missing before-action camera frame')
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            images = reset_images if i == 0 else {'laptop_rgb': rgb[:, :640], 'phone_rgb': rgb[:, 640:]}
            timing = row['act_timing']
            base = np.array(timing['act_base_legacy'], np.float32)
            state = np.array(before['observation_state'] if i == 0 else rows[i - 1]['state'], np.float32)
            proprio, gate = proprio_features(state, base, previous, i, max_steps, mapping, timing['native_chunk_phase'])
            if abs(gate - row['learning']['correction_gate']) > 1e-5:
                raise ValueError('Recovered retention features do not match live gate')
            raw = np.array(row['action'], np.float32)
            applied = filter_residual(raw, gate, previous, BOUNDS, FILTER['alpha'], FILTER['max_change_deg'])
            np.testing.assert_allclose(applied, row['learning']['applied_residual'], atol=2e-6)
            np.testing.assert_allclose(base + applied * BOUNDS / mapping['scale'], row['residual']['requested_legacy'], atol=2e-5)
            arrays['observations'].append(np.r_[encoder(images), proprio, move_features(cases[path.parent.name])].astype(np.float32))
            arrays['actions'].append(raw)
            arrays['weights'].append(4. if timing['phase'] == 'arm_settle' else 1.)
            arrays['validation'].append(result['seed'] == validation_seed)
            arrays['episode'].append(number)
            previous = applied
            before = row['placement_telemetry']
        if not video.read()[0] or video.read()[0]:
            raise ValueError('Recording must have exactly one terminal frame')
        video.release()
        episodes.append({'path': str(path.resolve()), 'seed': result['seed'], 'frames': len(rows),
                         'validation': result['seed'] == validation_seed,
                         'hashes': {name: file_hash(path / name) for name in ('initial.json', 'initial_images.npz', 'steps.jsonl', 'cameras.mp4', 'result.json')}})
        print('RETENTION_EPISODE', json.dumps(episodes[-1]), flush=True)
    dataset = args.output / 'demonstrations.npz'
    np.savez_compressed(dataset, observations=np.asarray(arrays['observations'], np.float32),
                        actions=np.asarray(arrays['actions'], np.float32), weights=np.asarray(arrays['weights'], np.float32),
                        validation=np.asarray(arrays['validation'], bool), episode=np.asarray(arrays['episode'], np.int32))
    metadata = {**manifest, 'dataset_sha256': file_hash(dataset), 'frames': len(arrays['actions']),
                'source_model': str((args.pilot / 'residual_best.zip').resolve()),
                'source_model_sha256': file_hash(args.pilot / 'residual_best.zip'),
                'episodes': episodes, 'source_seeds': sorted({result['seed'] for _, result in sources}),
                'label_method': 'successful selected teacher residual actions, unchanged; no geometry relabelling',
                'split': 'whole seeds; largest successful seed is validation; all sources retired from future reliability tests',
                'image_source': 'lossless reset; subsequent frames decoded from H264 review recordings',
                'privileged_training_labels_only': False, 'actor_geometry_inputs': False,
                'purpose': 'retain successful residual behaviour while complete-move SAC explores later phases; not ACT dataset'}
    dataset.with_suffix('.json').write_text(json.dumps(metadata, indent=2))
    print('RETENTION_READY', json.dumps({'episodes': len(episodes), 'frames': metadata['frames'], 'validation_seed': validation_seed}), flush=True)


if __name__ == '__main__':
    main()
