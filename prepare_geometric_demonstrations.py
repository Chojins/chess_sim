"""Build actor supervision from independently audited complete teacher moves."""
import argparse
import json
from pathlib import Path
import numpy as np
from capture_demonstrations import file_hash
from verify_geometric_teacher import audit


def episode_quality(path, result):
    """Return an exclusion reason, or None for a centred complete teacher move."""
    path = Path(path)
    capture = result['score'].get('capture')
    if not result['score']['success']:
        return result['score']['reason']
    if capture and not capture['upright_stable']:
        return 'toppled_capture'
    rows = [json.loads(line) for line in (path/'steps.jsonl').read_text().splitlines()]
    decisions = [json.loads(line) for line in (path.parent/f"teacher_{result['seed']}_decisions.jsonl").read_text().splitlines()]
    for i in range(1, len(rows)):
        if rows[i]['act_timing']['phase'] == 'close_jaw' and rows[i-1]['act_timing']['phase'] == 'arm_settle':
            decision = decisions[i-1]
            if decision['role'] != 'align_opening' or np.linalg.norm(decision['alignment_error_mm']) > 2.:
                return 'closing_without_verified_centred_approach'
    return None


def prepare(roots, output):
    output = Path(output)
    if output.exists():
        raise ValueError('Demonstration output already exists')
    source, episodes, rejected, joint_map = None, [], [], None
    source_seeds = []
    for root in map(Path, roots):
        audit(root)
        manifest = json.loads((root/'manifest.json').read_text())
        if manifest['objective'] != 'complete move' or manifest['task_version'] != 'legal_move_v5':
            raise ValueError('Pickup-only rollouts cannot seed complete-move supervision/replay')
        if source is None:
            source = manifest
        else:
            for key in ('weights_sha256', 'grasp_reference', 'live_timing', 'capture_home_retry',
                        'residual_bounds_deg', 'residual_filter', 'gamma', 'potential_weight', 'cases', 'teacher'):
                if source[key] != manifest[key]:
                    raise ValueError('Teacher collections have different deployment/reward semantics')
        if set(source_seeds) & set(manifest['source_seeds']):
            raise ValueError('Teacher collection seed overlap')
        source_seeds.extend(manifest['source_seeds'])
        for result in json.loads((root/'results.json').read_text()):
            path = root/result['task_id']/f"teacher_{result['seed']}"
            initial = json.loads((path/'initial.json').read_text())
            current_map = initial['telemetry']['observation']['joint_map']
            if joint_map is None:
                joint_map = current_map
            elif current_map != joint_map:
                raise ValueError('Teacher episodes use different robot calibration')
            reason = episode_quality(path, result)
            if reason is not None:
                rejected.append({'path': str(path), 'reason': reason})
                continue
            replay = path/result['learning_transitions']['path']
            episodes.append({'path': str(path.resolve()), 'seed': result['seed'], 'task_id': result['task_id'],
                'transitions': result['steps'], 'replay_sha256': file_hash(replay), 'replay_file': replay.name,
                'trace_sha256': file_hash(path/'steps.jsonl'), 'decisions': str((path.parent/f"teacher_{result['seed']}_decisions.jsonl").resolve()),
                'decisions_sha256': file_hash(path.parent/f"teacher_{result['seed']}_decisions.jsonl")})
    if len({row['seed'] for row in episodes}) < 2:
        raise ValueError(f'Need at least two upright complete, centred teacher episodes; got {len(episodes)}')
    validation_seed = max(row['seed'] for row in episodes)
    arrays = {key: [] for key in ('observations', 'actions', 'weights', 'validation', 'episode')}
    for number, episode in enumerate(episodes):
        path = Path(episode['path'])
        decisions = [json.loads(line) for line in Path(episode['decisions']).read_text().splitlines()]
        with np.load(path/episode['replay_file'], allow_pickle=False) as data:
            arrays['observations'].extend(data['observations'][:-1])
            arrays['actions'].extend(data['actions'])
        arrays['weights'].extend([5. if row['role']=='align_opening' else 2. if row['phase'] in ('close_jaw', 'closed_settle') else 1. for row in decisions])
        arrays['validation'].extend([episode['seed']==validation_seed]*episode['transitions'])
        arrays['episode'].extend([number]*episode['transitions'])
    arrays = {key: np.asarray(values, dtype=bool if key=='validation' else np.int32 if key=='episode' else np.float32)
              for key, values in arrays.items()}
    output.mkdir(parents=True, exist_ok=False)
    dataset = output/'demonstrations.npz'
    np.savez_compressed(dataset, **arrays)
    # The untrained zero-mean source initializes only the supervised actor;
    # labels are the explicitly privileged geometric teacher, not its outputs.
    manifest = {**source, 'source_seeds': sorted(source_seeds), 'dataset_sha256': file_hash(dataset),
        'joint_map': joint_map,
        'episodes': episodes, 'validation_seed': validation_seed, 'rejected_episodes': rejected,
        'frames': len(arrays['actions']), 'purpose': 'centred complete-move teacher actions for BC and matched-reward replay',
        'privileged_actor_inputs': False, 'collection_roots': [str(Path(root).resolve()) for root in roots]}
    dataset.with_suffix('.json').write_text(json.dumps(manifest, indent=2))
    return {'episodes': len(episodes), 'training_episodes': len(episodes)-1, 'frames': len(arrays['actions']),
            'validation_seed': validation_seed, 'dataset': str(dataset), 'rejected_episodes': len(rejected)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path, nargs='+')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.directory, args.output)))
