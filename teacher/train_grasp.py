"""Run reference feasibility checks and short-horizon state-based SAC.

All reported successes are near-grasp pickup diagnostics, not chess moves.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor

from teacher.near_grasp import NearGraspEnv,LiftConfig
from teacher.train import actor_digest


def save_status(path,**values):
    temporary=Path(path).with_suffix('.tmp')
    temporary.write_text(json.dumps(values,indent=2))
    temporary.replace(path)


def run_trial(env,policy,entry,seed,path,reference=None,capture=None):
    path.mkdir(parents=True,exist_ok=False)
    env.fixed_entry=entry
    obs,initial=env.reset(seed=seed)
    (path/'initial.json').write_text(json.dumps(initial,indent=2))
    if capture is not None: capture(env.base.latest)
    result=None;step=0
    def advance(action,stream):
        nonlocal obs,result,step
        obs,reward,terminated,truncated,info=env.step(action)
        step+=1
        stream.write(json.dumps({'step':step,'action':np.asarray(action).tolist(),'reward':reward,
            'terminated':terminated,'truncated':truncated,'score':info['score'],
            'teacher':info['teacher'],'reward_components':info['reward_components'],
            'observation':info['telemetry']['observation']})+'\n')
        if capture is not None: capture(env.base.latest)
        if terminated or truncated:
            result={'reset_name':entry['name'],'reset_family':entry.get('family','open_approach'),
                    'seed':seed,'steps':step,'score':info['score']}
            if 'case' in entry: result['case']=entry['case']
        return result is not None
    with (path/'steps.jsonl').open('x') as stream:
        if reference is None:
            while result is None:
                action,_=policy.predict(obs,deterministic=True)
                advance(action,stream)
        else:
            previous=env.base.command.copy()
            for row in reference[entry['prefix_step']:]:
                target=np.clip(np.asarray(row['teacher']['command_joint_deg']),env.base.low,env.base.high)
                count=max(1,int(np.ceil(np.max(np.abs(target-previous)*30/env.base.config.max_velocity_deg_s))))
                for fraction in np.linspace(0,1,count+1)[1:]:
                    goal=previous+(target-previous)*fraction
                    if advance(env.reference_action(goal) if hasattr(env,'reference_action') else env.base.normalized_target(goal),stream):
                        break
                if result:
                    break
                previous=target
            while result is None:
                advance(env.reference_action(previous) if hasattr(env,'reference_action') else env.base.normalized_target(previous),stream)
    (path/'result.json').write_text(json.dumps(result,indent=2))
    return result


def evaluate_bank(env,model,entries,directory,seed,reference=None):
    directory.mkdir(parents=True,exist_ok=False)
    rows=[]
    for index,entry in enumerate(entries):
        row=run_trial(env,model,entry,seed+index,directory/entry['name'],reference=reference)
        rows.append(row)
    (directory/'results.json').write_text(json.dumps(rows,indent=2))
    return rows


class GraspProgress(BaseCallback):
    def __init__(self,output,steps):
        super().__init__();self.output=output;self.steps=steps
        self.started=time.perf_counter();self.successes=self.episodes=0

    def _on_step(self):
        for done,info in zip(self.locals['dones'],self.locals['infos']):
            if done:
                self.episodes+=1;self.successes+=bool(info['score']['success'])
                row={'steps':self.num_timesteps,'reset_name':info['reset_name'],'score':info['score'],
                     'episode':info.get('episode')}
                with (self.output/'training_episodes.jsonl').open('a') as stream:
                    stream.write(json.dumps(row)+'\n')
        if self.num_timesteps%200==0:
            values={'stage':'rl_training','steps':self.num_timesteps,'requested_steps':self.steps,
                'episodes':self.episodes,'training_pickups':self.successes,
                'elapsed_seconds':time.perf_counter()-self.started,'objective':'grasp_lift_only',
                'complete_chess_moves_certified':0,'ready_for_act_collection':False}
            save_status(self.output/'status.json',**values)
            print('GRASP_RL_PROGRESS',json.dumps(values),flush=True)
        return True


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bank',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--url',default='http://127.0.0.1:8081')
    parser.add_argument('--steps',type=int,default=12000)
    parser.add_argument('--seed',type=int,default=52001)
    parser.add_argument('--horizon',type=int,default=128)
    parser.add_argument('--reference-only',action='store_true')
    parser.add_argument('--reference-proof',type=Path,help='Reuse a matching, independently re-audited reference-feasibility run')
    parser.add_argument('--skip-reference-check',action='store_true',help='Only for a short integration smoke test')
    args=parser.parse_args()
    if args.seed<0 or args.steps<257:
        parser.error('Need >=257 training steps and a nonnegative seed')
    args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2)
    env=None
    try:
        env=NearGraspEnv(args.bank,args.url,lift_config=LiftConfig(horizon=args.horizon))
        evaluation=[row for row in env.bank['entries'] if row['split']=='evaluation']
        if not evaluation:
            raise ValueError('A separate held-out reset set is required')
        train_names={row['name'] for row in env.entries}
        if train_names & {row['name'] for row in evaluation}:
            raise ValueError('Evaluation reset leaked into training')
        manifest={'objective':'grasp_lift_only','complete_chess_move':False,'case':env.bank['case'],
            'bank':str(args.bank.resolve()),'bank_sha256':hashlib.sha256(args.bank.read_bytes()).hexdigest(),
            'training_reset_names':sorted(train_names),'evaluation_reset_names':[r['name'] for r in evaluation],
            'observation_schema':env.schema,'observation_size':env.observation_space.shape[0],
            'action_config':asdict(env.base.config),'lift_config':asdict(env.lift_config),
            'joint_map':env.base.mapping,'steps':args.steps,'seed':args.seed,'algorithm':'SAC',
            'initialization':'new actor/critics; no ACT, waypoint controller or failed BC checkpoint',
            'source_hashes':{str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in Path('teacher').glob('*.py')}}
        (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2))
        if args.reference_proof:
            from teacher.verify_grasp import audit_trial
            proof=json.loads((args.reference_proof/'manifest.json').read_text())
            for key in ('bank_sha256','action_config','lift_config','objective'):
                if proof[key]!=json.loads(json.dumps(manifest[key])):
                    raise ValueError(f'Reference proof differs from training contract: {key}')
            checked=[audit_trial(path.parent,proof) for path in sorted(args.reference_proof.glob('reference/*/result.json'))]
            families={row.get('family','open_approach') for row in env.entries}
            if (not checked or not all(row['success'] for row in checked)
                    or {row['reset_family'] for row in checked}!=families):
                raise ValueError('Reference proof does not pass both reset families')
            manifest['reference_feasibility_proof']={'path':str(args.reference_proof.resolve()),
                'trials':len(checked),'transitions':sum(row['steps'] for row in checked),'recomputed':True}
            (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2))
        elif not args.skip_reference_check:
            save_status(args.output/'status.json',stage='checking_reference_feasibility')
            reference=[json.loads(s) for s in (Path(env.bank['reference'])/'steps.jsonl').read_text().splitlines()]
            # Use training states for the reference check; held-out reset
            # continuations never supply learner transitions or gradients.
            checks=evaluate_bank(env,None,env.entries[:3],args.output/'reference',args.seed+1000,reference=reference)
            if not all(row['score']['success'] for row in checks):
                save_status(args.output/'status.json',stage='reference_check_failed',results=checks,rl_started=False)
                return
        if args.reference_only:
            save_status(args.output/'status.json',stage='reference_feasibility_passed',rl_started=False)
            return
        env.fixed_entry=None
        monitor=Monitor(env,filename=str(args.output/'episodes'))
        model=SAC('MlpPolicy',monitor,seed=args.seed,device='cpu',learning_rate=3e-4,
                  buffer_size=max(20000,args.steps),learning_starts=512,batch_size=128,
                  gamma=.99,train_freq=1,gradient_steps=1,ent_coef='auto_0.001',
                  policy_kwargs={'net_arch':[128,128]},verbose=0)
        with torch.no_grad():
            model.actor.log_std.weight.zero_();model.actor.log_std.bias.fill_(-1.5)
        model.set_logger(configure(str(args.output/'optimizer'),['csv']))
        model.save(args.output/'grasp_initial')
        initial_actor=actor_digest(model)
        save_status(args.output/'status.json',stage='initial_evaluation',rl_started=False)
        before=evaluate_bank(env,model,evaluation,args.output/'before',args.seed+2000)
        env.fixed_entry=None
        monitor.reset(seed=args.seed)
        progress=GraspProgress(args.output,args.steps)
        model.learn(args.steps,callback=progress)
        model.save(args.output/'grasp_final');model.save_replay_buffer(args.output/'replay_buffer.pkl')
        save_status(args.output/'status.json',stage='final_evaluation',rl_started=True,steps=model.num_timesteps)
        after=evaluate_bank(env,model,evaluation,args.output/'after',args.seed+2000)
        by_family={}
        for family in sorted({r.get('family','open_approach') for r in evaluation}):
            by_family[family]={key:{'successes':sum(bool(r['score']['success']) for r in rows if r['reset_family']==family),
                'trials':sum(r['reset_family']==family for r in rows)} for key,rows in [('before',before),('after',after)]}
        status={'stage':'grasp_rl_complete','steps':model.num_timesteps,'gradient_updates':model._n_updates,
            'before_pickups':sum(bool(r['score']['success']) for r in before),
            'after_pickups':sum(bool(r['score']['success']) for r in after),'evaluation_trials':len(after),
            'training_episodes':progress.episodes,'training_pickups':progress.successes,
            'per_reset_family':by_family,'actor_weights_changed':actor_digest(model)!=initial_actor,
            'objective':'grasp_lift_only','complete_chess_moves_certified':0,'ready_for_act_collection':False}
        save_status(args.output/'status.json',**status)
        print('GRASP_RL_COMPLETE',json.dumps(status),flush=True)
    except BaseException as error:
        save_status(args.output/'status.json',stage='failed',error=repr(error))
        raise
    finally:
        if env:
            env.close()


if __name__=='__main__':
    main()
