"""Compare a learned action interface on the same audited physical references."""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from stable_baselines3.common.logger import configure

from teacher.environment import StateTeacherEnv,TeacherConfig
from teacher.imitation import StandardizedFeatures,load_demonstrations,fit_actor
from teacher.reference import ORDINARY_CASE
from teacher.train import PriorSAC,evaluate


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--updates',type=int,default=6000)
    parser.add_argument('--seed',type=int,default=38000)
    parser.add_argument('--eval-seeds',nargs='+',type=int,default=[38001,38002])
    parser.add_argument('--action-mode',choices=['absolute','delta','arm_delta_gripper_position'],default='absolute')
    parser.add_argument('--url',default='http://127.0.0.1:8081')
    parser.add_argument('--state-noise-deg',type=float,default=0.)
    args=parser.parse_args()
    if args.updates<1 or args.seed<0 or len(set(args.eval_seeds))!=len(args.eval_seeds):
        parser.error('Invalid updates or seeds')
    args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2)
    source=json.loads((args.source/'manifest.json').read_text())
    source_config=TeacherConfig(**source['config'])
    config=replace(source_config,action_mode=args.action_mode)
    env=None
    try:
        env=StateTeacherEnv(args.url,case=ORDINARY_CASE,config=config)
        paths=[Path(s['path']) for s in source['sources']]
        validation=[i for i,s in enumerate(source['sources']) if s['validation']]
        data,sources=load_demonstrations(paths,validation,env.kinematics,env.anchors,source_config)
        targets=[]
        for path in paths:
            with np.load(path/'demonstration.npz',allow_pickle=False) as archive:
                targets.append(archive['physical_targets'].copy())
        targets=np.concatenate(targets)
        if args.action_mode=='absolute':
            data['actions']=(2*(targets-env.low)/(env.high-env.low)-1).clip(-1,1).astype(np.float32)
        elif args.action_mode=='arm_delta_gripper_position' and source_config.action_mode=='delta':
            data['actions'][:,5]=(2*(targets[:,5]-env.low[5])/(env.high[5]-env.low[5])-1).clip(-1,1)
        elif source_config.action_mode!='delta':
            raise ValueError('Delta relabelling requires an original delta dataset')
        train=data['observations'][~data['validation']]
        mean,scale=train.mean(0),np.maximum(train.std(0),.05)
        model=PriorSAC('MlpPolicy',env,seed=args.seed,device='cpu',learning_rate=3e-4,
            buffer_size=10000,learning_starts=256,batch_size=128,gamma=config.gamma,ent_coef=1e-5,
            policy_kwargs={'net_arch':[256,256],'features_extractor_class':StandardizedFeatures,
                'features_extractor_kwargs':{'mean':mean.tolist(),'scale':scale.tolist(),
                                            'time_max':float(train[:,-1].max())}},verbose=0)
        with torch.no_grad():
            model.actor.log_std.weight.zero_();model.actor.log_std.bias.fill_(-5.)
        model.set_logger(configure(str(args.output/'optimizer'),['csv']))
        manifest={**env.manifest(),'algorithm':'state-policy BC action-interface comparison',
            'sources':sources,'training_seed':args.seed,'evaluation_seeds':args.eval_seeds,
            'relabelled_actions':'same physical next command from each audited transition; no old rewards/replay imported',
            'source_hashes':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in Path('teacher').glob('*.py')}}
        (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2))
        fit_actor(model,data,env,args.updates,args.seed,args.output,state_noise_deg=args.state_noise_deg)
        model.save(args.output/'teacher_bc')
        (args.output/'status.json').write_text(json.dumps({'stage':'closed_loop_evaluation','rl_started':False}))
        results=evaluate(env,model,args.eval_seeds,args.output/'evaluation')
        status={'stage':'imitation_evaluated','complete_moves':sum(bool(r['score']['success']) for r in results),
            'pickups':sum(bool(r['score']['lifted']) for r in results),'trials':len(results),
            'rl_started':False,'ready_for_act_collection':False}
        (args.output/'status.json').write_text(json.dumps(status,indent=2))
        print('ORDINARY_REFIT_COMPLETE',json.dumps(status),flush=True)
    except BaseException as error:
        (args.output/'status.json').write_text(json.dumps({'stage':'failed','error':repr(error)},indent=2))
        raise
    finally:
        if env:
            env.close()


if __name__=='__main__':
    main()
