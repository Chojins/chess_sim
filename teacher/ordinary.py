"""Establish a standalone learned e2e4 policy before permitting further RL."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from stable_baselines3.common.logger import configure

from teacher.environment import StateTeacherEnv,TeacherConfig
from teacher.imitation import StandardizedFeatures,load_demonstrations,fit_actor
from teacher.reference import ORDINARY_CASE,record_reference
from teacher.train import PriorSAC,evaluate


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--url',default='http://127.0.0.1:8081')
    parser.add_argument('--updates',type=int,default=6000)
    parser.add_argument('--seed',type=int,default=34000)
    parser.add_argument('--eval-seeds',nargs='+',type=int,default=[36001,36002])
    args=parser.parse_args()
    if args.updates<1 or args.seed<0 or len(set(args.eval_seeds))!=len(args.eval_seeds):
        parser.error('Invalid update count or seeds')
    output=args.output
    output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2)
    config=TeacherConfig(max_steps=600,action_mode='delta',acceleration_weight=.0005,acceleration_cost_cap=.005)
    env=None
    try:
        (output/'status.json').write_text(json.dumps({'stage':'collecting_retimed_references'}))
        env=StateTeacherEnv(args.url,case=ORDINARY_CASE,config=config)
        references=[]
        for index,source_seed in enumerate([7501,7503,7504,7502]):
            source=Path(f'outputs/hybrid-readiness-v29/ordinary/repeat_0/initial_e2e4/learned_{source_seed}')
            path=output/f'reference_{index}'
            result=record_reference(env,source,path,args.seed+index+1)
            print('ORDINARY_REFERENCE',json.dumps({'index':index,'complete':result['complete'],'frames':result['frames'],
                                                  'reason':result['score']['reason']}),flush=True)
            if not result['complete']:
                (output/'status.json').write_text(json.dumps({'stage':'reference_failed','reference':index,
                    'score':result['score'],'rl_started':False},indent=2))
                return
            references.append(path)
        data,sources=load_demonstrations(references,[3],env.kinematics,env.anchors,config)
        train=data['observations'][~data['validation']]
        mean=train.mean(0);scale=np.maximum(train.std(0),.05)
        model=PriorSAC('MlpPolicy',env,seed=args.seed,device='cpu',learning_rate=3e-4,
                      buffer_size=10000,learning_starts=256,batch_size=128,gamma=config.gamma,
                      ent_coef=1e-5,policy_kwargs={'net_arch':[256,256],
                          'features_extractor_class':StandardizedFeatures,
                          'features_extractor_kwargs':{'mean':mean.tolist(),'scale':scale.tolist()}},verbose=0)
        with torch.no_grad():
            model.actor.log_std.weight.zero_();model.actor.log_std.bias.fill_(-5.)
        model.set_logger(configure(str(output/'optimizer'),['csv']))
        manifest={**env.manifest(),'algorithm':'state-policy behavioral cloning before SAC',
                  'sources':sources,'training_seed':args.seed,'evaluation_seeds':args.eval_seeds,
                  'normalization_statistics':'training episodes only; scale floor 0.05',
                  'source_hashes':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in Path('teacher').glob('*.py')}}
        (output/'manifest.json').write_text(json.dumps(manifest,indent=2))
        fit_actor(model,data,env,args.updates,args.seed,output)
        model.save(output/'teacher_bc')
        (output/'status.json').write_text(json.dumps({'stage':'closed_loop_evaluation','rl_started':False}))
        results=evaluate(env,model,args.eval_seeds,output/'evaluation')
        successes=sum(bool(r['score']['success']) for r in results)
        pickups=sum(bool(r['score']['lifted']) for r in results)
        status={'stage':'imitation_evaluated','complete_moves':successes,'pickups':pickups,'trials':len(results),
                'rl_started':False,'ready_for_act_collection':False,
                'next':('Review motion and run a fresh qualification before RL refinement' if successes==len(results)
                        else 'Fix the learned closed-loop approach/grasp before RL')}
        (output/'status.json').write_text(json.dumps(status,indent=2))
        print('ORDINARY_IMITATION_COMPLETE',json.dumps(status),flush=True)
    except BaseException as error:
        (output/'status.json').write_text(json.dumps({'stage':'failed','error':repr(error),'rl_started':False},indent=2))
        raise
    finally:
        if env:
            env.close()


if __name__=='__main__':
    main()
