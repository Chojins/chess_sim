"""Unified move SAC with training-only SGS-style resets and audited selection."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import time
import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback,CallbackList
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor

from teacher.conservative_grasp import family_counts
from teacher.grasp_checkpoints import ConservativeSAC,GraspCheckpoints,GraspSelection,file_hash
from teacher.move_teacher import MoveEnv,MoveConfig
from teacher.train_grasp import evaluate_bank,run_trial,save_status
from teacher.verify_move import audit_trial


def audit_policy_trial(path,manifest):
    if manifest.get('policy_action_space'):
        from teacher.verify_task_space import audit_trial as audit_cartesian
        return audit_cartesian(path,manifest)
    return audit_trial(path,manifest)


def audited_evaluation(env,actor,entries,directory,seed,manifest):
    sampler_before=env.sampler.state()
    rows=evaluate_bank(env,actor,entries,directory,seed)
    checked=[audit_policy_trial(directory/e['name'],manifest) for e in entries]
    if env.sampler.state()!=sampler_before: raise ValueError('Evaluation changed training sampler history')
    if [r['score']['success'] for r in rows]!=[r['success'] for r in checked]: raise ValueError('Raw-physics audit mismatch')
    (directory/'integrity.json').write_text(json.dumps({'passed':True,'trials':len(checked),
        'transitions':sum(r['steps'] for r in checked),'sampler_unchanged':True},indent=2))
    return rows


class MoveProgress(BaseCallback):
    def __init__(self,output,total,raw):
        super().__init__();self.output=output;self.total=total;self.raw=raw
        self.started=time.perf_counter();self.episodes=self.successes=self.home_successes=0
    def _on_step(self):
        for done,info in zip(self.locals['dones'],self.locals['infos']):
            if done:
                self.episodes+=1;self.successes+=bool(info['score']['success'])
                self.home_successes+=bool(info['score']['success'] and info['score']['home_start'])
                with (self.output/'training_episodes.jsonl').open('a') as stream:
                    stream.write(json.dumps({'steps':self.num_timesteps,'reset_name':info['reset_name'],
                        'score':info['score'],'episode':info.get('episode')})+'\n')
        if self.num_timesteps%200==0:
            status={'stage':'rl_training','steps':self.num_timesteps,'requested_steps':self.total,
                'episodes':self.episodes,'training_goal_completions':self.successes,'training_home_start_completions':self.home_successes,
                'elapsed_seconds':time.perf_counter()-self.started,'objective':'unified_e2e4','ready_for_act_collection':False}
            save_status(self.output/'status.json',**status)
            save_status(self.output/'sampler.json',**self.raw.sampler.state())
            print('UNIFIED_RL_PROGRESS',json.dumps(status),flush=True)
        return True


def reference_checks(env,entries,directory,manifest,seed,previous=None):
    directory.mkdir(parents=True,exist_ok=False);rows=[]
    imported=[]
    if previous:
        old=json.loads((previous/'manifest.json').read_text())
        for key in ('objective','action_config','move_config'):
            if old[key]!=json.loads(json.dumps(manifest[key])): raise ValueError('Cannot import a different reference contract')
        if old.get('policy_action_space')!=manifest.get('policy_action_space') or old.get('cartesian_config')!=manifest.get('cartesian_config'):
            raise ValueError('Cannot import a different policy control interface')
        old_entries={r['name']:r for r in json.loads(Path(old['bank']).read_text())['entries']}
    prefix=[json.loads(s) for s in (Path(env.bank['reference'])/'steps.jsonl').read_text().splitlines()][:140]
    for index,entry in enumerate(entries):
        old_path=previous/'reference'/entry['name'] if previous else None
        if old_path and (old_path/'result.json').exists():
            old_entry=old_entries[entry['name']]
            for key in ('sha256','horizon','family'):
                if old_entry[key]!=entry[key]: raise ValueError('Imported reset changed')
            shutil.copytree(old_path,directory/entry['name'])
            audit_policy_trial(directory/entry['name'],manifest)
            rows.append(json.loads((old_path/'result.json').read_text()));imported.append(entry['name']);continue
        trace=Path(env.bank.get('continuation_directory',Path(manifest['bank']).parent))/f'seed_{entry["seed"]}.jsonl'
        if file_hash(trace)!=entry['continuation_sha256']: raise ValueError('Reference continuation changed')
        continuation=[{'teacher':{'command_joint_deg':json.loads(s)['command_joint_deg']}} for s in trace.read_text().splitlines()]
        e=dict(entry,prefix_step=0 if entry.get('reference_hold_only') else entry['physical_step'])
        row=run_trial(env,None,e,seed+index,directory/e['name'],reference=[] if entry.get('reference_hold_only') else prefix+continuation)
        audit_policy_trial(directory/e['name'],manifest);rows.append(row)
    (directory/'results.json').write_text(json.dumps(rows,indent=2))
    (directory/'integrity.json').write_text(json.dumps({'passed':True,'trials':len(rows),
        'reused_trials_reaudited':imported,'source_proof':str(previous) if previous else None},indent=2))
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bank',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--url',default='http://127.0.0.1:8081');parser.add_argument('--steps',type=int,default=16000)
    parser.add_argument('--seed',type=int,default=84001);parser.add_argument('--checkpoint-every',type=int,default=4000)
    parser.add_argument('--critic-warmup',type=int,default=512);parser.add_argument('--horizon',type=int,default=800)
    parser.add_argument('--gamma',type=float,default=.99)
    parser.add_argument('--task-space',action='store_true',help='Generic differential IK plus open/close jaw instead of joint increments')
    parser.add_argument('--warm-start',type=Path,help='BC actor checkpoint; critics and replay must still be fresh')
    parser.add_argument('--reference-data',type=Path,help='Audited training-only RL replay/prior data, required with warm-start')
    parser.add_argument('--prior-weight',type=float,default=.5)
    parser.add_argument('--prior-floor',type=float,default=.1)
    parser.add_argument('--online-oracle-weight',type=float,default=0.,help='Training-only corrective loss on real replay observations')
    parser.add_argument('--prefer-faster',action='store_true',help='With perfect family success, rank home completion time in simulated-second buckets')
    parser.add_argument('--bounded-exploration',action='store_true',help='Keep warm-start Gaussian task-space exploration std between exp(-5) and exp(-2.5)')
    parser.add_argument('--reference-only',action='store_true');parser.add_argument('--reference-proof',type=Path)
    parser.add_argument('--extend-reference-proof',type=Path,help='Reference-only: re-audit unchanged trials and check newly added reset families')
    parser.add_argument('--smoke',action='store_true');parser.add_argument('--uniform',action='store_true')
    args=parser.parse_args()
    if args.steps<513 or args.seed<0 or args.checkpoint_every<1 or args.critic_warmup<0: parser.error('Invalid run settings')
    if not 0<args.gamma<1 or bool(args.warm_start)!=bool(args.reference_data): parser.error('Invalid discount or incomplete warm-start/reference-data pair')
    if (not np.isfinite([args.prior_weight,args.prior_floor,args.online_oracle_weight]).all() or args.prior_weight<0
            or args.online_oracle_weight<0 or not 0<=args.prior_floor<=1 or (args.online_oracle_weight and not args.warm_start)):
        parser.error('Invalid reference/corrective regularization')
    if args.bounded_exploration and not (args.warm_start and args.task_space):
        parser.error('Bounded task-space exploration requires a fitted Cartesian warm-start')
    if (args.extend_reference_proof and (not args.reference_only or args.reference_proof or args.smoke)) or (args.reference_only and args.smoke):
        parser.error('Reference proof extension requires reference-only mode without smoke or reference-proof')
    args.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2);env=None
    try:
        save_status(args.output/'status.json',stage='preparing')
        if args.task_space:
            from teacher.task_space import TaskSpaceMoveEnv
            env=TaskSpaceMoveEnv(args.bank,args.url,config=MoveConfig(horizon=args.horizon),adaptive=not args.uniform,gamma=args.gamma)
        else:
            env=MoveEnv(args.bank,args.url,config=MoveConfig(horizon=args.horizon),adaptive=not args.uniform,gamma=args.gamma)
        validation=[r for r in env.bank['entries'] if r['split']=='validation'];test=[r for r in env.bank['entries'] if r['split']=='test']
        groups=[env.entries,validation,test];families={r['family'] for r in env.entries}
        if not all(groups): raise ValueError('Need separate training, validation and final test sets')
        for i,group in enumerate(groups):
            if len({r['name'] for r in group})!=len(group) or {r['family'] for r in group}!=families:
                raise ValueError('Every split must have unique names and cover all reset families')
            for other in groups[:i]:
                if {r['seed'] for r in group}&{r['seed'] for r in other}: raise ValueError('Reset seed leakage')
        frozen=Path(env.bank['grasp_checkpoint']) if env.bank.get('grasp_checkpoint') else None
        frozen_hash=env.bank.get('grasp_checkpoint_sha256')
        if frozen is not None:
            if file_hash(frozen)!=frozen_hash: raise ValueError('Frozen grasp changed')
        elif env.bank.get('reset_builder')!='analytic_geometry_oracle':
            raise ValueError('Reset bank has no recognized physical provenance')
        manifest={'objective':'unified_e2e4','bank':str(args.bank.resolve()),'bank_sha256':file_hash(args.bank),
            'case':env.bank['case'],'action_config':asdict(env.base.config),'move_config':asdict(env.move_config),
            'observation_schema':env.schema,'observation_size':env.observation_space.shape[0],'joint_map':env.base.mapping,
            'training_reset_names':[r['name'] for r in env.entries],'validation_reset_names':[r['name'] for r in validation],
            'final_test_reset_names':[r['name'] for r in test],'algorithm':'SAC','steps':args.steps,'seed':args.seed,
            'initialization':'fresh unified actor/critics/replay; references only generate resets and feasibility checks',
            'adaptive_sampling':not args.uniform,'sampler_config':{k:v for k,v in env.sampler.state().items() if k not in ('history','probabilities')},
            'sampling_scope':'training outcomes only; validation and final test are fixed and never update sampler',
            'actor_learning_rate':1e-4,'critic_learning_rate':3e-4,'fixed_entropy_coefficient':.002,
            'critic_warmup_updates':args.critic_warmup,'checkpoint_interval':args.checkpoint_every,
            'grasp_checkpoint_sha256':frozen_hash,'smoke_test':args.smoke,'final_test_used_for_selection':False,
            'prefer_faster_after_perfect_success':args.prefer_faster,
            'exploration_log_std_bounds':[-5.,-2.5] if args.bounded_exploration else [-20.,2.],
            'qualification_scope':'Only home-start evaluations measure full execution from home; late resets measure continuation goals',
            'source_hashes':{str(p):file_hash(p) for p in Path('teacher').glob('*.py')}}
        if args.task_space:
            manifest.update(policy_action_space=env.policy_action_space,cartesian_config=asdict(env.cartesian_config))
        if args.warm_start:
            warm_status=json.loads((args.warm_start.parent/'status.json').read_text())
            if warm_status.get('critics_trained') or warm_status.get('replay_size')!=0:
                raise ValueError('Warm-start must have untrained critics and empty replay')
            manifest.update(initialization='fitted actor, fresh critics; training-only reference replay and decaying prior',
                warm_start=str(args.warm_start.resolve()),warm_start_sha256=file_hash(args.warm_start),
                actor_learning_rate=1e-5,critic_learning_rate=3e-4)
        source=args.output/'source';source.mkdir()
        for p in Path('teacher').glob('*.py'): (source/p.name).write_bytes(p.read_bytes())
        (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2))
        if args.reference_proof:
            proof_path=args.reference_proof/'policy_manifest.json'
            if not proof_path.exists(): proof_path=args.reference_proof/'manifest.json'
            proof=json.loads(proof_path.read_text())
            for key in ('objective','bank_sha256','move_config'):
                if proof[key]!=json.loads(json.dumps(manifest[key])): raise ValueError('Reference contract mismatch: '+key)
            if proof.get('policy_action_space')!=manifest.get('policy_action_space') or proof.get('cartesian_config')!=manifest.get('cartesian_config'):
                raise ValueError('Reference policy control interface changed')
            if {k:v for k,v in proof['action_config'].items() if k!='gamma'}!={k:v for k,v in json.loads(json.dumps(manifest['action_config'])).items() if k!='gamma'}:
                raise ValueError('Reference physical action contract changed')
            proof_trials=list(args.reference_proof.glob('reference/*/result.json'))
            if not proof_trials: proof_trials=list(args.reference_proof.glob('trials/*/result.json'))
            checked=[audit_policy_trial(p.parent,proof) for p in proof_trials]
            if not checked or {r['reset_family'] for r in checked}!=families or not all(r['success'] for r in checked):
                raise ValueError('Incomplete unified reference proof')
            manifest['reference_feasibility']={'path':str(args.reference_proof.resolve()),'trials':len(checked),
                'source_gamma':proof['action_config']['gamma'],'target_gamma':args.gamma,
                'note':'Raw source rewards re-audited under original discount; physical action limits, reset identities and completion goal identical.'}
            (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2))
        elif not args.smoke:
            save_status(args.output/'status.json',stage='reference_feasibility',rl_started=False)
            entries=[next(r for r in env.entries if r['family']==family) for family in sorted(families)]
            checked=reference_checks(env,entries,args.output/'reference',manifest,args.seed+1000,args.extend_reference_proof)
            if not all(r['score']['success'] for r in checked):
                save_status(args.output/'status.json',stage='reference_check_failed',results=checked,rl_started=False);return
        if args.reference_only:
            save_status(args.output/'status.json',stage='reference_feasibility_passed',rl_started=False);return
        env.fixed_entry=None;monitor=Monitor(env,filename=str(args.output/'episodes'))
        if args.warm_start:
            from teacher.refinement import RefinementSAC
            load_kwargs={}
            if args.bounded_exploration:
                from teacher.bounded_exploration import BoundedSACPolicy
                load_kwargs['policy_class']=BoundedSACPolicy
            model=RefinementSAC.load(args.warm_start,env=monitor,device='cpu',seed=args.seed,learning_rate=3e-4,
                actor_lr_scale=1/30,critic_warmup_updates=args.critic_warmup,gamma=args.gamma,learning_starts=0,**load_kwargs)
            model.set_random_seed(args.seed)
            manifest['reference_replay']=model.configure_reference(args.reference_data,env,args.seed+9000,
                weight=args.prior_weight,floor_fraction=args.prior_floor)
            manifest['training_only_corrective_oracle']=model.configure_online_oracle(env,args.online_oracle_weight)
            (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2))
        else:
            model=ConservativeSAC('MlpPolicy',monitor,seed=args.seed,device='cpu',learning_rate=3e-4,
                actor_lr_scale=1/3,critic_warmup_updates=args.critic_warmup,buffer_size=max(30000,args.steps),learning_starts=512,
                batch_size=128,gamma=args.gamma,train_freq=1,gradient_steps=1,ent_coef=.002,policy_kwargs={'net_arch':[128,128]},verbose=0)
            with torch.no_grad(): model.actor.log_std.weight.zero_();model.actor.log_std.bias.fill_(-1.5)
        model.set_logger(configure(str(args.output/'optimizer'),['csv']));model.save(args.output/'move_initial')
        selector=GraspSelection(args.output,validation,prefix='move',complete_moves=True,prefer_faster=args.prefer_faster)
        save_status(args.output/'status.json',stage='initial_validation',rl_started=False)
        initial=audited_evaluation(env,model,validation,args.output/'validation_initial',args.seed+2000,manifest)
        selector.consider(args.output/'move_initial.zip','initial',0,initial)
        if args.warm_start: model.save_replay_buffer(args.output/'move_best_replay.pkl')
        def checkpoint_eval(raw,actor,label,step):
            save_status(args.output/'status.json',stage='checkpoint_validation',steps=step,best_validation=selector.best)
            rows=audited_evaluation(raw,actor,validation,args.output/f'validation_{step}',args.seed+2000,manifest)
            save_status(args.output/f'sampler_{step}.json',**raw.sampler.state());return rows
        checkpoint=GraspCheckpoints(args.output,env,selector,args.checkpoint_every,args.steps,checkpoint_eval)
        progress=MoveProgress(args.output,args.steps,env)
        env.fixed_entry=None;monitor.reset(seed=args.seed);model.learn(args.steps,callback=CallbackList([progress,checkpoint]))
        model.save(args.output/'move_final');model.save_replay_buffer(args.output/'replay_buffer.pkl')
        save_status(args.output/'status.json',stage='final_validation',steps=model.num_timesteps)
        final=audited_evaluation(env,model,validation,args.output/'validation_final',args.seed+2000,manifest)
        if selector.consider(args.output/'move_final.zip','final',model.num_timesteps,final): model.save_replay_buffer(args.output/'move_best_replay.pkl')
        chosen_hash=file_hash(args.output/'move_best.zip');chosen=ConservativeSAC.load(args.output/'move_best.zip',device='cpu')
        save_status(args.output/'status.json',stage='selected_policy_final_test',selected=selector.best)
        held=audited_evaluation(env,chosen,test,args.output/'after',args.seed+3000,manifest)
        if file_hash(args.output/'move_best.zip')!=chosen_hash or (frozen is not None and file_hash(frozen)!=frozen_hash):
            raise ValueError('Frozen policy changed')
        status={'stage':'unified_move_rl_complete','steps':model.num_timesteps,'gradient_updates':model._n_updates,
            'training_episodes':progress.episodes,'training_goal_completions':progress.successes,
            'training_home_start_completions':progress.home_successes,'selected_checkpoint':selector.best,
            'selected_checkpoint_sha256':chosen_hash,'final_policy_validation':family_counts(final),
            'selected_policy_final_test':family_counts(held),'final_test_successes':sum(r['score']['success'] for r in held),
            'final_test_trials':len(held),'home_start_test_successes':sum(r['score']['success'] for r in held if r['score']['home_start']),
            'home_start_test_trials':sum(r['score']['home_start'] for r in held),'checkpoint_validation_seconds':checkpoint.validation_seconds,
            'objective':'unified_e2e4','ready_for_act_collection':False,'adaptive_sampling':not args.uniform}
        save_status(args.output/'sampler.json',**env.sampler.state());save_status(args.output/'status.json',**status)
        print('UNIFIED_MOVE_RL_COMPLETE',json.dumps(status),flush=True)
    except BaseException as error:
        save_status(args.output/'status.json',stage='failed',error=repr(error));raise
    finally:
        if env: env.close()


if __name__=='__main__': main()
