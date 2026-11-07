"""Read sampled exported ACT frames back and compare with lossless sources."""
import argparse
import json
from pathlib import Path
import numpy as np
from verify_collected_episodes import digest,read_json


def verify_dataset(root,repo_id):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    root=Path(root)
    provenance=read_json(root/'simulation_provenance.json')
    index=Path(provenance['index_path'])
    if digest(index)!=provenance['index_sha256']:
        raise ValueError('Source episode index changed')
    source=read_json(index)
    data=LeRobotDataset(repo_id,root=root)
    if data.fps!=30 or data.num_episodes!=len(source['episodes']) or len(data)!=source['frames']:
        raise ValueError('Dataset counts/fps differ from verified sources')
    offset=checked=0
    for number,episode in enumerate(source['episodes']):
        for frame in sorted({0,episode['frames']//2,episode['frames']-1}):
            actual=data[offset+frame]
            with np.load(Path(episode['path'])/f'frame_{frame:06d}.npz',allow_pickle=False) as expected:
                np.testing.assert_array_equal(actual['observation.state'].numpy(),expected['observation_state'])
                np.testing.assert_array_equal(actual['action'].numpy(),expected['action_legacy'])
                for camera,name in [('laptop','laptop_rgb'),('phone','phone_rgb')]:
                    pixels=actual[f'observation.images.{camera}'].permute(1,2,0).numpy()
                    np.testing.assert_array_equal(np.rint(pixels*255).astype(np.uint8),expected[name])
            if int(actual['episode_index'])!=number or int(actual['frame_index'])!=frame:
                raise ValueError('Dataset frame/episode ordering changed')
            np.testing.assert_allclose(float(actual['timestamp']),frame/30,atol=3e-6,rtol=0)
            checked+=1
        offset+=episode['frames']
    report={'passed':True,'episodes':data.num_episodes,'frames':len(data),'sampled_frames_checked':checked,
            'sampled_rgb_pixels_exact':True,'sampled_legacy_states_and_executed_actions_exact':True,
            'fps':data.fps,'source_index_sha256':provenance['index_sha256'],'uploaded':False,
            'scope':'Sampled exported frames read through the real LeRobot loader; source trajectories were independently audited in full'}
    (root/'roundtrip_integrity.json').write_text(json.dumps(report,indent=2))
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--repo-id',required=True)
    args=parser.parse_args();print(json.dumps(verify_dataset(args.root,args.repo_id)))
