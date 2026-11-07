"""Convert independently audited neural-teacher frames into a local ACT dataset."""
import argparse
import json
from pathlib import Path
import numpy as np
from teacher.episode_data import audit_episode,digest


def export(index,output,repo_id):
    index=Path(index);output=Path(output)
    if output.exists(): raise ValueError('Refusing to overwrite dataset output')
    report=json.loads(index.read_text());index_hash=digest(index)
    if report.get('format')!='chess_sim_neural_teacher_audited_v1' or not report.get('episodes'):
        raise ValueError('Need a nonempty independently audited neural-teacher index')
    first=report['episodes'][0]['provenance'];calibration=first['calibration'];actor_hash=first['teacher_checkpoint_sha256']
    for episode in report['episodes']:
        if episode['provenance']['calibration']!=calibration or episode['provenance']['teacher_checkpoint_sha256']!=actor_hash:
            raise ValueError('Mixed teacher/calibration semantics require a separate dataset')
        if audit_episode(episode['episode_path'])!=episode: raise ValueError('Indexed episode changed')
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    names=calibration['legacy_names']
    if len(names)!=6: raise ValueError('Expected six source legacy joint names')
    features={k:{'dtype':'float32','shape':(6,),'names':names} for k in ('observation.state','action')}
    for camera in ('laptop','phone'):
        features[f'observation.images.{camera}']={'dtype':'image','shape':(480,640,3),'names':['height','width','channels']}
    dataset=LeRobotDataset.create(repo_id=repo_id,fps=30,features=features,root=output,
        robot_type='so100_legacy_sim',use_videos=False,image_writer_threads=4)
    try:
        for episode in report['episodes']:
            for n in range(episode['frames']):
                with np.load(Path(episode['path'])/f'frame_{n:06d}.npz',allow_pickle=False) as f:
                    dataset.add_frame({'observation.state':f['observation_state'].copy(),'action':f['action_legacy'].copy(),
                        'observation.images.laptop':f['laptop_rgb'].copy(),'observation.images.phone':f['phone_rgb'].copy(),
                        'task':f'Chess move {episode["move"]}'})
            dataset.save_episode()
    finally: dataset.finalize()
    if dataset.meta.total_frames!=report['frames'] or dataset.meta.total_episodes!=len(report['episodes']):
        raise ValueError('Dataset counts differ from accepted source')
    if digest(index)!=index_hash: raise ValueError('Accepted index changed during export')
    provenance={'index_path':str(index.resolve()),'index_sha256':index_hash,'teacher_checkpoint_sha256':actor_hash,
        'episodes':report['episodes'],'calibration':calibration,'calibration_conversion':None,'uploaded':False,
        'note':'Qualified state-based neural teacher, executed legacy targets and lossless RGB. No RGB student trained yet.'}
    (output/'simulation_provenance.json').write_text(json.dumps(provenance,indent=2))
    from verify_act_dataset import verify_dataset
    integrity=verify_dataset(output,repo_id)
    return {'dataset':str(output.resolve()),'episodes':len(report['episodes']),'frames':report['frames'],
        'roundtrip_sampled_frames':integrity['sampled_frames_checked'],'uploaded':False}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--index',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--repo-id',default='local/chess-neural-teacher-pilot')
    args=p.parse_args();print(json.dumps(export(args.index,args.output,args.repo_id)))
