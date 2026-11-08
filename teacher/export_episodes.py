"""Convert independently audited neural-teacher frames into a local ACT dataset."""
import argparse
import json
from pathlib import Path
import numpy as np
from teacher.episode_data import audit_episode,digest


def export(index,output,repo_id,goal_input=False):
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
    if goal_input:
        features['observation.images.goal']={'dtype':'image','shape':(480,640,3),'names':['height','width','channels']}
    dataset=LeRobotDataset.create(repo_id=repo_id,fps=30,features=features,root=output,
        robot_type='so100_legacy_sim',use_videos=False,image_writer_threads=4)
    goals=[]
    try:
        for episode_number,episode in enumerate(report['episodes']):
            goal=None
            if goal_input:
                from PIL import Image
                from teacher.goal_image import render_goal
                svg=Path(episode['episode_path'])/'desired_board.svg';goal=render_goal(svg)
                folder=output/'goal_images';folder.mkdir(exist_ok=True);path=folder/f'episode_{episode_number:06d}.png'
                Image.fromarray(goal).save(path)
                goals.append({'episode_index':episode_number,'path':str(path.relative_to(output)),
                    'sha256':digest(path),'source_svg_sha256':digest(svg)})
            for n in range(episode['frames']):
                with np.load(Path(episode['path'])/f'frame_{n:06d}.npz',allow_pickle=False) as f:
                    frame={'observation.state':f['observation_state'].copy(),'action':f['action_legacy'].copy(),
                        'observation.images.laptop':f['laptop_rgb'].copy(),'observation.images.phone':f['phone_rgb'].copy(),
                        'task':f'Chess move {episode["move"]}'}
                    if goal is not None: frame['observation.images.goal']=goal.copy()
                    dataset.add_frame(frame)
            dataset.save_episode()
    finally: dataset.finalize()
    if dataset.meta.total_frames!=report['frames'] or dataset.meta.total_episodes!=len(report['episodes']):
        raise ValueError('Dataset counts differ from accepted source')
    if digest(index)!=index_hash: raise ValueError('Accepted index changed during export')
    provenance={'index_path':str(index.resolve()),'index_sha256':index_hash,'teacher_checkpoint_sha256':actor_hash,
        'episodes':report['episodes'],'calibration':calibration,'calibration_conversion':None,'uploaded':False,
        'note':'Qualified state-based neural teacher, executed legacy targets and lossless RGB. No RGB student trained yet.',
        'goal_image_input':goal_input,'goal_images':goals}
    (output/'simulation_provenance.json').write_text(json.dumps(provenance,indent=2))
    from verify_act_dataset import verify_dataset
    integrity=verify_dataset(output,repo_id)
    if goal_input:
        from PIL import Image
        decoded=LeRobotDataset(repo_id,root=output);offset=checked=0
        for episode,goal in zip(report['episodes'],goals):
            path=output/goal['path']
            if digest(path)!=goal['sha256']: raise ValueError('Derived goal image changed')
            expected=np.asarray(Image.open(path).convert('RGB'))
            for n in sorted({0,episode['frames']//2,episode['frames']-1}):
                actual=decoded[offset+n]['observation.images.goal'].permute(1,2,0).numpy()
                np.testing.assert_array_equal(np.rint(actual*255).astype(np.uint8),expected);checked+=1
            offset+=episode['frames']
        (output/'goal_roundtrip_integrity.json').write_text(json.dumps({'passed':True,'sampled_frames_checked':checked,
            'goal_rgb_pixels_exact':True,'goal_is_separate_from_live_cameras':True},indent=2))
    return {'dataset':str(output.resolve()),'episodes':len(report['episodes']),'frames':report['frames'],
        'roundtrip_sampled_frames':integrity['sampled_frames_checked'],'uploaded':False}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--index',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--repo-id',default='local/chess-neural-teacher-pilot')
    p.add_argument('--goal-input',action='store_true',help='Include the desired-board RGB diagram as a third input')
    args=p.parse_args();print(json.dumps(export(args.index,args.output,args.repo_id,args.goal_input)))
