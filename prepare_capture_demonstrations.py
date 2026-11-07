"""Recover successful ordinary rollouts as supervised residual training data."""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
import torch
from act_client import LegacyACT
from capture_demonstrations import JointKinematics, BOUNDS, FILTER, centred_label, file_hash, validate_semantics
from residual_learning import FrozenACTFeatures, filter_residual, proprio_features
from train_openings import move_features


SOURCES = [
    ('outputs/capture-alignment-rl-v9/scandinavian_capture/checkpoint_2031_1401', False),
    ('outputs/capture-alignment-rl-v9/scandinavian_capture/checkpoint_2031_1402', False),
    ('outputs/capture-alignment-rl-v9/scandinavian_capture/checkpoint_2031_1403', True),
    ('outputs/capture-heldout-v9/repeat_0/scandinavian_capture/learned_1502', False),
    ('outputs/capture-heldout-v9/repeat_1/scandinavian_capture/learned_1502', False),
    ('outputs/capture-heldout-v9/repeat_1/scandinavian_capture/learned_1503', True),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--kinematics', type=Path, required=True)
    parser.add_argument('--validate-only', action='store_true', help='Audit all source frames and reachable labels without GPU inference or output files')
    args = parser.parse_args()
    if not args.validate_only:
        args.output.mkdir(parents=True, exist_ok=False)
    original = json.loads(Path('outputs/capture-alignment-rl-v9/manifest.json').read_text())
    validate_semantics(original)
    source_model = Path('outputs/capture-alignment-rl-v9/residual_2031.zip')
    kin = JointKinematics(json.loads(args.kinematics.read_text()))
    torch.set_num_threads(2)
    if args.validate_only:
        encoder = lambda images: np.zeros(4096, np.float32)
    else:
        policy = LegacyACT(original['checkpoint'], 'cuda', n_action_steps=100)
        if policy.weights_sha256 != original['weights_sha256']:
            raise ValueError('ACT weights changed')
        encoder = FrozenACTFeatures(policy)
    arrays = {k: [] for k in ('observations', 'actions', 'weights', 'validation', 'episode')}
    episodes = []
    maximum_fk_error, labelled = 0., []
    print('DEMONSTRATION_PREPARATION_STARTED', flush=True)
    for episode_id, (directory, validation) in enumerate(SOURCES):
        path = Path(directory)
        result = json.loads((path / 'result.json').read_text())
        if not result['score']['success'] or result['score']['other_pieces_disturbed']:
            raise ValueError('Only successful, undisturbed pickups are eligible')
        initial = json.loads((path / 'initial.json').read_text())['telemetry']['observation']
        mapping = initial['joint_map']
        scale = np.array(mapping['scale'])
        rows = [json.loads(line) for line in (path / 'steps.jsonl').read_text().splitlines()]
        video = cv2.VideoCapture(str(path / 'cameras.mp4'))
        with np.load(path / 'initial_images.npz', allow_pickle=False) as images:
            reset_images = {k: images[k].copy() for k in ('laptop_rgb', 'phone_rgb')}
        before = initial
        state, previous = np.array(initial['observation_state'], np.float32), np.zeros(6, np.float32)
        victim = initial['task_geometry']['capture_index']
        initial_z = initial['pieces']['piece_centers_m'][victim][2]
        for i, row in enumerate(rows):
            ok, frame = video.read()
            if not ok or frame.shape != (480, 1280, 3):
                raise ValueError(f'{path}: video lacks before-action frame {i}')
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            images = reset_images if i == 0 else {'laptop_rgb': rgb[:, :640], 'phone_rgb': rgb[:, 640:]}
            applied = np.array(row['learning']['applied_residual'])
            base = (np.array(row['residual']['requested_legacy']) - applied * BOUNDS / scale).astype(np.float32)
            proprio, gate = proprio_features(state, base, previous, i, 800, mapping)
            if abs(gate - row['learning']['correction_gate']) > 1e-5:
                raise ValueError('Offline gate does not match live actor observation')
            raw = np.array(row['action'], np.float32)
            np.testing.assert_allclose(filter_residual(raw, gate, previous, BOUNDS, FILTER['alpha'], FILTER['max_change_deg']), applied, atol=2e-6)
            fk_error = np.linalg.norm(kin.point(before['measured_joint_deg']) - before['gripper_grasp_center_m'])
            maximum_fk_error = max(maximum_fk_error, float(fk_error))
            if fk_error > 2e-6:
                raise ValueError(f'Offline FK differs from measured tool by {fk_error} metres')
            target = np.array(before['pieces']['piece_grasp_targets_m'][victim])
            distance = np.linalg.norm(np.array(before['gripper_grasp_center_m']) - target)
            lifted = before['pieces']['piece_centers_m'][victim][2] - initial_z > .005
            weight = 2. if lifted else 1.
            label = raw.copy()
            if gate > .3 and not lifted and distance < .025 and before['pieces']['piece_upright_cosines'][victim] > .95:
                base_q = np.array(mapping['sim_home_deg']) + scale * (base - mapping['legacy_home'])
                label, error_before, error_after = centred_label(kin, base_q, previous, gate, raw, target)
                labelled.append({'episode': episode_id, 'step': i + 1, 'before_mm': error_before * 1000, 'after_mm': error_after * 1000})
                weight = 4.
            if gate == 0:
                label[:] = 0
                weight = .25
            arrays['observations'].append(np.r_[encoder(images), proprio, move_features(original['cases'][0])].astype(np.float32))
            arrays['actions'].append(label)
            arrays['weights'].append(weight)
            arrays['validation'].append(validation)
            arrays['episode'].append(episode_id)
            before = row['placement_telemetry']
            state = np.array(row['state'], np.float32)
            previous = applied.astype(np.float32)
        if not video.read()[0] or video.read()[0]:
            raise ValueError('Video must have exactly one terminal frame after the action observations')
        video.release()
        episodes.append({'path': str(path.resolve()), 'seed': result['seed'], 'validation': validation, 'frames': len(rows),
                         'hashes': {name: file_hash(path / name) for name in ('initial.json', 'initial_images.npz', 'steps.jsonl', 'cameras.mp4', 'result.json')}})
        print('DEMONSTRATION_EPISODE', json.dumps(episodes[-1]), flush=True)
    if not labelled:
        raise ValueError('No eligible near-grasp labels')
    if args.validate_only:
        print('DEMONSTRATIONS_VALIDATED', json.dumps({'frames': len(arrays['actions']), 'labelled_frames': len(labelled),
                                                     'maximum_fk_error_m': maximum_fk_error,
                                                     'mean_label_improvement_mm': float(np.mean([r['before_mm'] - r['after_mm'] for r in labelled]))}), flush=True)
        return
    dataset = args.output / 'demonstrations.npz'
    np.savez_compressed(dataset, observations=np.array(arrays['observations'], np.float32), actions=np.array(arrays['actions'], np.float32),
                        weights=np.array(arrays['weights'], np.float32), validation=np.array(arrays['validation'], bool), episode=np.array(arrays['episode'], np.int32))
    manifest = {**original, 'dataset_sha256': file_hash(dataset), 'source_model': str(source_model.resolve()),
                'source_model_sha256': file_hash(source_model), 'episodes': episodes,
                'source_seeds': sorted({e['seed'] for e in episodes}), 'frames': len(arrays['actions']),
                'kinematics_sha256': file_hash(args.kinematics), 'maximum_logged_fk_error_m': maximum_fk_error,
                'labelled_frames': len(labelled), 'label_statistics': labelled,
                'label_method': 'successful unpaused ACT+residual rollouts; near-grasp arms half blended toward one-step reachable centred FK target; demonstrated jaw/lift retained',
                'image_source': 'lossless reset; subsequent frames decoded from lossy H264 recordings',
                'split': 'whole episodes; seed 1403/1503 validation; 1401/1402/1502 training; all source seeds retired from reliability testing',
                'privileged_training_labels_only': True, 'actor_geometry_inputs': False}
    dataset.with_suffix('.json').write_text(json.dumps(manifest, indent=2))
    print('DEMONSTRATIONS_READY', json.dumps({'frames': manifest['frames'], 'labelled_frames': len(labelled), 'maximum_fk_error_m': maximum_fk_error}), flush=True)


if __name__ == '__main__':
    main()
