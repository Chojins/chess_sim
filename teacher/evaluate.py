"""Evaluate a frozen teacher, optionally recording actual paired simulator cameras.

Camera recording still uses privileged state for control. It is not an RGB
student, and the recording is not an ACT training export.
"""
import argparse
import hashlib
import html
import json
from pathlib import Path

import numpy as np
import torch

from act_client import command
from camera_client import PairedCameraClient
from evaluation_video import EpisodeVideo
from policy_overlay import highlight_move
from teacher.environment import StateTeacherEnv,TeacherConfig
from teacher.train import PriorSAC,evaluate


class CameraStateClient:
    """Use the unchanged camera/control server with the teacher state contract."""
    def __init__(self,url):
        self.url=url
        self.client=PairedCameraClient(url,timeout=30)
        self.uncertain=True
        self.latest=None
        self.images=None

    def read(self):
        metadata,self.images=self.client.read(color_order='rgb')
        if metadata['observation']['mode']!='policy_control':
            raise RuntimeError('Camera server must run with --control --contacts --magnets')
        self.latest={'sequence':metadata['frame_id'],'simulation_time':metadata['simulation_time'],
                     'observation':metadata['observation'],'camera_render_products':2}
        return self.latest

    def _command(self,path,action=None,options=None):
        before=self.latest
        self.uncertain=True
        response=command(self.url,path,before['sequence'],action=action,options=options)
        latest=self.read()
        if latest['sequence']!=response['frame_id'] or latest['sequence']!=before['sequence']+1:
            raise RuntimeError('Camera observation does not match executed teacher command')
        if path=='/step' and not np.isclose(latest['simulation_time']-before['simulation_time'],1/30,atol=1e-6):
            raise RuntimeError('Lost camera/control timestep')
        self.uncertain=False
        return latest

    def reset(self,options):
        self.read()
        return self._command('/reset',options=options)

    def step(self,action):
        if self.uncertain:
            raise RuntimeError('reset() required after uncertain camera command')
        return self._command('/step',action=action)

    def close(self):
        pass


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot',type=Path,required=True)
    parser.add_argument('--checkpoint',default='teacher_bc.zip')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seeds',nargs='+',type=int,required=True)
    parser.add_argument('--url',default='http://127.0.0.1:8081')
    parser.add_argument('--record-video',action='store_true',help='Requires a camera/control server, normally port 8080')
    args=parser.parse_args()
    if not args.seeds or len(set(args.seeds))!=len(args.seeds) or min(args.seeds)<0:
        parser.error('Need unique nonnegative seeds')
    args.output.mkdir(parents=True,exist_ok=False)
    manifest=json.loads((args.pilot/'manifest.json').read_text())
    checkpoint=args.pilot/args.checkpoint
    torch.set_num_threads(2)
    model=PriorSAC.load(checkpoint,device='cpu')
    client=CameraStateClient(args.url) if args.record_video else None
    env=StateTeacherEnv(args.url,client=client,case=manifest['case'],config=TeacherConfig(**manifest['config']))
    env_manifest=env.manifest()
    if env_manifest['joint_map']!=manifest['joint_map'] or env.observation_space.shape!=model.observation_space.shape:
        raise ValueError('Teacher checkpoint and evaluation environment differ')
    provenance={'pilot':str(args.pilot.resolve()),'checkpoint':str(checkpoint.resolve()),
                'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                'mode':'frozen privileged-state teacher evaluation','rgb_student':False,'seeds':args.seeds,
                'environment':env_manifest,'record_video':args.record_video}
    (args.output/'manifest.json').write_text(json.dumps(provenance,indent=2))
    try:
        if not args.record_video:
            results=evaluate(env,model,args.seeds,args.output/'trials')
        else:
            calibration=json.loads(Path('calibration/sim_board.json').read_text())['cameras']['red']
            def pair():
                fixed=highlight_move(np.ascontiguousarray(client.images['red'][...,::-1]),calibration,manifest['case']['move'])
                return {'laptop_rgb':np.ascontiguousarray(fixed[...,::-1]),'phone_rgb':client.images['blue']}
            results=[]
            for seed in args.seeds:
                directory=args.output/f'seed_{seed}'
                directory.mkdir()
                obs,initial=env.reset(seed=seed)
                (directory/'initial.json').write_text(json.dumps(initial,indent=2))
                video=EpisodeVideo(directory,pair())
                video.write(pair())
                try:
                    with (directory/'steps.jsonl').open('x') as stream:
                        for step in range(env.config.max_steps):
                            action,_=model.predict(obs,deterministic=True)
                            obs,reward,terminated,truncated,info=env.step(action)
                            stream.write(json.dumps({'step':step+1,'action':action.tolist(),'reward':reward,
                                'terminated':terminated,'truncated':truncated,'score':info['score'],
                                'teacher':info['teacher'],'observation':info['telemetry']['observation']})+'\n')
                            video.write(pair())
                            if terminated or truncated:
                                break
                finally:
                    video.close()
                row={'seed':seed,'steps':step+1,'score':info['score'],'video':video.verify()}
                results.append(row)
                (directory/'result.json').write_text(json.dumps(row,indent=2))
            body=['<!doctype html><meta charset="utf-8"><title>State teacher review</title>',
                  '<h1>Learned state teacher: e2–e4</h1>',
                  '<p>Actual simulator camera recordings. The policy uses privileged state, not RGB. '
                  'Left: fixed red camera; right: blue wrist camera. Playback is simulated 30 Hz.</p>']
            for row in results:
                label=f"Seed {row['seed']}: {row['score']['reason']}"
                body.append(f'<h2>{html.escape(label)}</h2><video controls width="1200" '
                            f'src="seed_{row["seed"]}/cameras.mp4"></video>')
            (args.output/'videos.html').write_text('\n'.join(body))
        report={'trials':len(results),'complete_moves':sum(bool(r['score']['success']) for r in results),
                'pickups':sum(bool(r['score']['lifted']) for r in results),'results':results}
        (args.output/'results.json').write_text(json.dumps(report,indent=2))
        print(json.dumps({k:v for k,v in report.items() if k!='results'}),flush=True)
    finally:
        env.close()


if __name__=='__main__':
    main()
