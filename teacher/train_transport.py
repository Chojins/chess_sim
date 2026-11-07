"""Checkpointed state SAC for held-pawn transport and release, with frozen grasp."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback,CallbackList
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor

from teacher.conservative_grasp import family_counts
from teacher.grasp_checkpoints import ConservativeSAC,GraspCheckpoints,GraspSelection,file_hash
from teacher.place_reference import run_reference
from teacher.train_grasp import evaluate_bank,save_status
from teacher.transport_place import TransportPlaceEnv,PlaceConfig
from teacher.verify_transport import audit_trial


def audited_evaluation(env,model,entries,directory,seed,manifest):
    rows=evaluate_bank(env,model,entries,directory,seed)
    checked=[audit_trial(directory/entry['name'],manifest) for entry in entries]
    if [r['score']['success'] for r in rows]!=[r['success'] for r in checked]:
        raise ValueError('Placement results differ from independently rescored physics')
    (directory/'integrity.json').write_text(json.dumps({'passed':True,'trials':len(checked),
        'transitions':sum(r['steps'] for r in checked),'complete_chess_moves_certified':0},indent=2))
    return rows


class PlaceProgress(BaseCallback):
    def __init__(self,output,total):
        super().__init__();self.output=output;self.total=total
        self.started=time.perf_counter();self.episodes=self.successes=0
    def _on_step(self):
        for done,info in zip(self.locals['dones'],self.locals['infos']):
            if done:
                self.episodes+=1;self.successes+=bool(info['score']['success'])
                with (self.output/'training_episodes.jsonl').open('a') as stream:
                    stream.write(json.dumps({'steps':self.num_timesteps,'reset_name':info['reset_name'],
                        'score':info['score'],'episode':info.get('episode')})+'\n')
        if self.num_timesteps%200==0:
            status={'stage':'rl_training','steps':self.num_timesteps,'requested_steps':self.total,
                'episodes':self.episodes,'training_placements':self.successes,
                'elapsed_seconds':time.perf_counter()-self.started,'objective':'transport_place_only',
                'complete_chess_moves_certified':0,'ready_for_act_collection':False}
            save_status(self.output/'status.json',**status)
            print('PLACE_RL_PROGRESS',json.dumps(status),flush=True)
        return True


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bank',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--url',default='http://127.0.0.1:8081')
    parser.add_argument('--steps',type=int,default=12000)
    parser.add_argument('--seed',type=int,default=75001)
    parser.add_argument('--horizon',type=int,default=192)
    parser.add_argument('--checkpoint-every',type=int,default=2000)
    parser.add_argument('--critic-warmup',type=int,default=512)
    parser.add_argument('--reference-only',action='store_true')
    parser.add_argument('--reference-proof',type=Path)
    parser.add_argument('--smoke',action='store_true',help='Integration check; not a qualification run')
    args=parser.parse_args()
    if args.steps<513 or args.critic_warmup<0 or args.seed<0 or args.checkpoint_every<1:
        parser.error('Invalid training budget')
    args.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2);env=None
    try:
        save_status(args.output/'status.json',stage='preparing')
        env=TransportPlaceEnv(args.bank,args.url,config=PlaceConfig(horizon=args.horizon))
        validation=[r for r in env.bank['entries'] if r['split']=='validation']
        final_entries=[r for r in env.bank['entries'] if r['split']=='test']
        groups=[env.entries,validation,final_entries]
        if not all(groups): raise ValueError('Require train, validation and unseen final-test states')
        families={r['family'] for r in env.entries}
        names=[{r['name'] for r in group} for group in groups]
        seeds=[{r['seed'] for r in group} for group in groups]
        for i,group in enumerate(groups):
            if len(names[i])!=len(group) or {r['family'] for r in group}!=families:
                raise ValueError('Every split must have unique names and cover all reset families')
            for j in range(i):
                if names[i]&names[j] or seeds[i]&seeds[j]: raise ValueError('Reset seed leakage across partitions')
        frozen=Path(env.bank['grasp_checkpoint']);grasp_hash=env.bank['grasp_checkpoint_sha256']
        if file_hash(frozen)!=grasp_hash: raise ValueError('Qualified grasp checkpoint changed')
        manifest={'objective':'transport_place_only','complete_chess_move':False,'case':env.bank['case'],
            'bank':str(args.bank.resolve()),'bank_sha256':file_hash(args.bank),
            'action_config':asdict(env.base.config),'place_config':asdict(env.place_config),
            'observation_schema':env.schema,'observation_size':env.observation_space.shape[0],
            'joint_map':env.base.mapping,'training_reset_names':sorted(names[0]),
            'validation_reset_names':sorted(names[1]),'final_test_reset_names':sorted(names[2]),
            'steps':args.steps,'seed':args.seed,'algorithm':'SAC','initialization':'fresh actor, critics and replay',
            'actor_learning_rate':1e-4,'critic_learning_rate':3e-4,'fixed_entropy_coefficient':.002,
            'critic_warmup_updates':args.critic_warmup,'checkpoint_interval':args.checkpoint_every,
            'grasp_checkpoint_sha256':grasp_hash,'grasp_actor_frozen':True,
            'final_test_used_for_selection':False,'smoke_test':args.smoke,
            'source_hashes':{str(p):file_hash(p) for p in Path('teacher').glob('*.py')}}
        source=args.output/'source';source.mkdir()
        for path in Path('teacher').glob('*.py'): (source/path.name).write_bytes(path.read_bytes())
        (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2))
        if args.reference_proof:
            proof=json.loads((args.reference_proof/'manifest.json').read_text())
            for key in ('bank_sha256','action_config','place_config','objective'):
                if proof[key]!=json.loads(json.dumps(manifest[key])): raise ValueError('Reference contract changed: '+key)
            checked=[audit_trial(p.parent,manifest) for p in args.reference_proof.glob('reference/*/result.json')]
            if not checked or {r['reset_family'] for r in checked}!=families or not all(r['success'] for r in checked):
                raise ValueError('Reference feasibility is incomplete')
        elif not args.smoke:
            save_status(args.output/'status.json',stage='reference_feasibility',rl_started=False)
            checks=[]
            for family in sorted(families):
                entry=next(r for r in env.entries if r['family']==family)
                path=args.output/'reference'/entry['name']
                result=run_reference(env,entry,args.seed+1000,path)
                audit_trial(path,manifest);checks.append(result)
            (args.output/'reference'/'results.json').write_text(json.dumps(checks,indent=2))
            if not all(r['score']['success'] for r in checks):
                save_status(args.output/'status.json',stage='reference_check_failed',results=checks,rl_started=False)
                return
        if args.reference_only:
            save_status(args.output/'status.json',stage='reference_feasibility_passed',rl_started=False);return
        env.fixed_entry=None;monitor=Monitor(env,filename=str(args.output/'episodes'))
        model=ConservativeSAC('MlpPolicy',monitor,seed=args.seed,device='cpu',learning_rate=3e-4,
            actor_lr_scale=1/3,critic_warmup_updates=args.critic_warmup,buffer_size=max(20000,args.steps),
            learning_starts=512,batch_size=128,gamma=.99,train_freq=1,gradient_steps=1,
            ent_coef=.002,policy_kwargs={'net_arch':[128,128]},verbose=0)
        with torch.no_grad():
            model.actor.log_std.weight.zero_();model.actor.log_std.bias.fill_(-1.5)
        model.set_logger(configure(str(args.output/'optimizer'),['csv']));model.save(args.output/'place_initial')
        selector=GraspSelection(args.output,validation,prefix='place')
        save_status(args.output/'status.json',stage='initial_validation',rl_started=False)
        initial=audited_evaluation(env,model,validation,args.output/'validation_initial',args.seed+2000,manifest)
        selector.consider(args.output/'place_initial.zip','initial',0,initial)
        def checkpoint_eval(raw,actor,label,step):
            save_status(args.output/'status.json',stage='checkpoint_validation',steps=step,best_validation=selector.best)
            return audited_evaluation(raw,actor,validation,args.output/f'validation_{step}',args.seed+2000,manifest)
        checkpoint=GraspCheckpoints(args.output,env,selector,args.checkpoint_every,args.steps,checkpoint_eval)
        progress=PlaceProgress(args.output,args.steps)
        env.fixed_entry=None;monitor.reset(seed=args.seed)
        model.learn(args.steps,callback=CallbackList([progress,checkpoint]))
        model.save(args.output/'place_final');model.save_replay_buffer(args.output/'replay_buffer.pkl')
        save_status(args.output/'status.json',stage='final_validation',steps=model.num_timesteps)
        final=audited_evaluation(env,model,validation,args.output/'validation_final',args.seed+2000,manifest)
        if selector.consider(args.output/'place_final.zip','final',model.num_timesteps,final):
            model.save_replay_buffer(args.output/'place_best_replay.pkl')
        selected_hash=file_hash(args.output/'place_best.zip')
        selected=ConservativeSAC.load(args.output/'place_best.zip',device='cpu')
        save_status(args.output/'status.json',stage='selected_policy_final_test',selected=selector.best)
        held=audited_evaluation(env,selected,final_entries,args.output/'after',args.seed+3000,manifest)
        if file_hash(args.output/'place_best.zip')!=selected_hash or file_hash(frozen)!=grasp_hash:
            raise ValueError('Frozen policy changed during evaluation')
        status={'stage':'transport_place_rl_complete','steps':model.num_timesteps,'gradient_updates':model._n_updates,
            'training_episodes':progress.episodes,'training_placements':progress.successes,
            'selected_checkpoint':selector.best,'selected_checkpoint_sha256':selected_hash,
            'final_policy_validation':family_counts(final),'selected_policy_final_test':family_counts(held),
            'final_test_successes':sum(bool(r['score']['success']) for r in held),'final_test_trials':len(held),
            'checkpoint_validation_seconds':checkpoint.validation_seconds,'objective':'transport_place_only',
            'complete_chess_moves_certified':0,'ready_for_act_collection':False}
        save_status(args.output/'status.json',**status)
        print('TRANSPORT_PLACE_RL_COMPLETE',json.dumps(status),flush=True)
    except BaseException as error:
        save_status(args.output/'status.json',stage='failed',error=repr(error));raise
    finally:
        if env: env.close()


if __name__=='__main__': main()
