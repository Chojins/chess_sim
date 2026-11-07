"""Export audited successful simulation episodes as a local LeRobot dataset.

Keeps lossless RGB images, camera names and legacy joint units expected by the
existing ACT. This command never pushes data to Hugging Face.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from verify_collected_episodes import audit_episode, digest, read_json


def export_episodes(index, output, repo_id):
    index, output = Path(index), Path(output)
    if output.exists():
        raise ValueError('Dataset output already exists')
    report = read_json(index)
    index_hash = digest(index)
    if report.get('format') != 'chess_sim_audited_episodes_v1' or not report.get('episodes'):
        raise ValueError('Expected a nonempty independently audited episode index')
    episodes = report['episodes']
    calibration = episodes[0]['provenance']['calibration']
    act_hash = episodes[0]['provenance']['act_sha256']
    for episode in episodes:
        path = Path(episode['path'])
        if episode['provenance']['calibration'] != calibration or episode['provenance']['act_sha256'] != act_hash:
            raise ValueError('Mixed calibration/ACT semantics require a separate dataset')
        for filename, expected in episode['file_sha256'].items():
            if Path(filename).name != filename or digest(path / filename) != expected:
                raise ValueError('Indexed episode file changed or filename is invalid')
        if digest(path.parent / 'steps.jsonl') != episode['trace_sha256']:
            raise ValueError('Indexed physics trace changed')
        if audit_episode(path) != episode:
            raise ValueError('Episode no longer matches its audited index')
    # Dataset dependencies are deliberately optional for the live RL process.
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    names = calibration['legacy_names']
    if len(names) != 6:
        raise ValueError('Expected six legacy joint names')
    features = {'observation.state': {'dtype': 'float32', 'shape': (6,), 'names': names},
                'action': {'dtype': 'float32', 'shape': (6,), 'names': names}}
    for camera in ('laptop', 'phone'):
        features[f'observation.images.{camera}'] = {'dtype': 'image', 'shape': (480, 640, 3),
                                                    'names': ['height', 'width', 'channels']}
    dataset = LeRobotDataset.create(repo_id=repo_id, fps=30, features=features, root=output,
                                    robot_type='so100_legacy_sim', use_videos=False, image_writer_threads=4)
    try:
        for episode in episodes:
            path = Path(episode['path'])
            task = f"Chess move {episode['move']}" + ('; captured piece to acting robot right-side tabletop' if episode['capture'] else '')
            for number in range(episode['frames']):
                with np.load(path / f'frame_{number:06d}.npz', allow_pickle=False) as frame:
                    dataset.add_frame({'observation.state': frame['observation_state'].copy(),
                                       'action': frame['action_legacy'].copy(),
                                       'observation.images.laptop': frame['laptop_rgb'].copy(),
                                       'observation.images.phone': frame['phone_rgb'].copy(),
                                       'task': task})
            dataset.save_episode()
    finally:
        dataset.finalize()
    if dataset.meta.total_episodes != len(episodes) or dataset.meta.total_frames != report['frames']:
        raise ValueError('Exported dataset frame/episode counts differ from audit')
    if digest(index) != index_hash:
        raise ValueError('Accepted episode index changed during export; wait for collection to finish')
    provenance = {'index_path': str(index.resolve()), 'index_sha256': index_hash,
                  'episodes': episodes, 'calibration': calibration, 'act_sha256': act_hash,
                  'calibration_conversion': None, 'uploaded': False,
                  'note': 'Automatically verified simulation teacher data; retains source legacy units and lossless RGB camera images'}
    (output / 'simulation_provenance.json').write_text(json.dumps(provenance, indent=2))
    from verify_act_dataset import verify_dataset
    roundtrip=verify_dataset(output,repo_id)
    return {'dataset': str(output.resolve()), 'episodes': len(episodes), 'frames': report['frames'],
            'image_encoding': 'lossless PNG', 'uploaded': False,
            'roundtrip_sampled_frames':roundtrip['sampled_frames_checked']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repo-id', default='local/chess-sim-verified')
    args = parser.parse_args()
    print(json.dumps(export_episodes(args.index, args.output, args.repo_id)))


if __name__ == '__main__':
    main()
