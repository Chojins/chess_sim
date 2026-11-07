"""Load audited full-move geometric demonstrations into SAC's initial replay."""
import json
from pathlib import Path
import numpy as np
from capture_demonstrations import file_hash, load_demonstrations


def matched_semantics(source, deployment):
    if (source.get('capture_transport') != deployment.get('capture_transport') or
            source.get('capture_step_limit',800) != deployment.get('capture_step_limit',800)):
        raise ValueError('Demonstration replay differs in capture controller or step budget')
    for key in ('task_version', 'objective', 'weights_sha256', 'grasp_reference',
                'observation_size', 'action_window', 'capture_alignment',
                'capture_home_retry', 'residual_bounds_deg', 'residual_filter',
                'gamma', 'potential_weight', 'cases', 'joint_map'):
        if source.get(key) != deployment.get(key):
            raise ValueError(f'Demonstration replay differs in {key}')
    for key in ('config', 'kinematics_sha256'):
        if source['live_timing'][key] != deployment['live_timing'][key]:
            raise ValueError(f'Demonstration replay differs in timing {key}')
    if source['objective'] != 'complete move' or source['task_version'] != 'legal_move_v5':
        raise ValueError('Only complete-move rewards can prefill complete-move replay')


def prefill(path, deployment, model=None):
    from verify_geometric_teacher import audit
    data, metadata = load_demonstrations(path)
    matched_semantics(metadata, deployment)
    if {deployment['training_seed'], *deployment['evaluation_seeds']} & set(metadata['source_seeds']):
        raise ValueError('Demonstration sources overlap online training or selection seeds')
    for root in metadata['collection_roots']:
        audit(root)
    validation_seed = metadata['validation_seed']
    count, replay_seeds, start = 0, [], 0
    for number, episode in enumerate(metadata['episodes']):
        directory = Path(episode['path'])
        replay = directory/episode['replay_file']
        if (file_hash(replay) != episode['replay_sha256'] or
                file_hash(directory/'steps.jsonl') != episode['trace_sha256'] or
                file_hash(episode['decisions']) != episode['decisions_sha256']):
            raise ValueError('Demonstration episode provenance changed')
        result = json.loads((directory/'result.json').read_text())
        initial = json.loads((directory/'initial.json').read_text())
        if initial['telemetry']['observation']['joint_map'] != deployment['joint_map']:
            raise ValueError('Demonstration replay uses different robot calibration')
        capture = result['score'].get('capture')
        if (not result['score']['success'] or result['score']['other_pieces_disturbed'] or
                capture and not capture['upright_stable']):
            raise ValueError('Replay contains an unsuccessful or toppled complete move')
        stop = start + episode['transitions']
        validation = episode['seed'] == validation_seed
        np.testing.assert_equal(data['validation'][start:stop], np.full(stop-start, validation))
        np.testing.assert_equal(data['episode'][start:stop], np.full(stop-start, number))
        with np.load(replay, allow_pickle=False) as transitions:
            obs, actions, rewards = (transitions[key] for key in ('observations', 'actions', 'rewards'))
            terminal, truncated = transitions['terminated'], transitions['truncated']
            if (obs.shape != (stop-start+1, deployment['observation_size']) or
                    actions.shape != (stop-start, 6) or not terminal[-1] or terminal[:-1].any() or
                    truncated.any() or not np.isfinite(np.r_[obs.ravel(), actions.ravel(), rewards]).all()):
                raise ValueError('Incomplete or invalid complete-move replay')
            np.testing.assert_equal(data['observations'][start:stop], obs[:-1])
            np.testing.assert_equal(data['actions'][start:stop], actions)
            if not validation:
                if model is not None:
                    if model.replay_buffer.size()+len(actions) >= model.replay_buffer.buffer_size:
                        raise ValueError('Demonstrations exceed the initial replay capacity')
                    for i in range(len(actions)):
                        model.replay_buffer.add(obs[i:i+1], obs[i+1:i+2], actions[i:i+1],
                                                rewards[i:i+1], terminal[i:i+1],
                                                [{'TimeLimit.truncated': False}])
                count += len(actions)
                replay_seeds.append(episode['seed'])
        start = stop
    if start != len(data['actions']) or not count or validation_seed in replay_seeds:
        raise ValueError('Replay split is incomplete or leaks validation')
    return {'kind': 'geometric_teacher', 'replay_transitions': count,
            'replay_seeds': sorted(set(replay_seeds)), 'validation_seed': validation_seed,
            'requested_seeds': metadata['source_seeds'], 'successful_episodes': len(metadata['episodes']),
            'dataset': str(Path(path).resolve()), 'dataset_sha256': metadata['dataset_sha256'],
            'rewards': 'independently recomputed matched full-move rewards',
            'privileged_actor_inputs': False}
