"""Train with fixed entropy, slower actor updates and validated checkpoints.

Validation selects the actor; a separate final test is used only after selection.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor

from teacher.grasp_checkpoints import ConservativeSAC,GraspCheckpoints,GraspSelection,file_hash,policy_success
from teacher.near_grasp import NearGraspEnv,LiftConfig
from teacher.train_grasp import GraspProgress,evaluate_bank,save_status
from teacher.verify_grasp import audit_trial


def audited_evaluation(env,model,entries,directory,seed,manifest,reference=None):
    results=evaluate_bank(env,model,entries,directory,seed,reference=reference)
    audits=[audit_trial(directory/entry['name'],manifest) for entry in entries]
    if [row['score']['success'] for row in results]!=[row['success'] for row in audits]:
        raise ValueError('Validation results differ from independently re-scored physics')
    (directory/'integrity.json').write_text(json.dumps({'passed':True,'trials':len(audits),
        'transitions':sum(row['steps'] for row in audits),'complete_chess_moves_certified':0},indent=2))
    return results


def family_counts(rows):
    counts={}
    for row in rows:
        group=counts.setdefault(row['reset_family'],{'successes':0,'trials':0})
        group['trials']+=1;group['successes']+=policy_success(row)
    return counts


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bank',type=Path,required=True)
    parser.add_argument('--validation-bank',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--url',default='http://127.0.0.1:8081')
    parser.add_argument('--steps',type=int,default=12000)
    parser.add_argument('--seed',type=int,default=63001)
    parser.add_argument('--checkpoint-every',type=int,default=1000)
    parser.add_argument('--actor-lr',type=float,default=1e-4)
    parser.add_argument('--critic-lr',type=float,default=3e-4)
    parser.add_argument('--entropy',type=float,default=.002)
    parser.add_argument('--critic-warmup',type=int,default=512)
    parser.add_argument('--horizon',type=int,default=128)
    parser.add_argument('--smoke',action='store_true',help='Integration check only; skip physical reference feasibility already checked separately')
    args=parser.parse_args()
    if (args.steps<257 or args.seed<0 or args.checkpoint_every<1 or args.critic_warmup<0
            or not all(np.isfinite(v) and v>0 for v in [args.actor_lr,args.critic_lr,args.entropy])
            or args.actor_lr>args.critic_lr):
        parser.error('Invalid training budget or conservative optimizer settings')
    args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2)
    env=None
    try:
        save_status(args.output/'status.json',stage='preparing',objective='grasp_lift_only')
        env=NearGraspEnv(args.bank,args.url,lift_config=LiftConfig(horizon=args.horizon))
        final_entries=[row for row in env.bank['entries'] if row['split']=='evaluation']
        validation_bank=json.loads(args.validation_bank.read_text())
        validation=[row for row in validation_bank['entries'] if row['split']=='evaluation']
        train_names={row['name'] for row in env.entries}
        val_names={row['name'] for row in validation}
        test_names={row['name'] for row in final_entries}
        if (not validation or not final_entries or train_names&val_names or train_names&test_names or val_names&test_names
                or len(val_names)!=len(validation) or len(test_names)!=len(final_entries)
                or validation_bank['case']!=env.bank['case']):
            raise ValueError('Training, validation and final-test resets must be disjoint and share the command')
        expected_families={row.get('family','open_approach') for row in env.entries}
        for entries in [validation,final_entries]:
            if {r.get('family','open_approach') for r in entries}!=expected_families:
                raise ValueError('Validation and final test must cover every training reset family')
        manifest={'objective':'grasp_lift_only','complete_chess_move':False,'case':env.bank['case'],
            'algorithm':'SAC','bank':str(args.bank.resolve()),'bank_sha256':file_hash(args.bank),
            'validation_bank':str(args.validation_bank.resolve()),'validation_bank_sha256':file_hash(args.validation_bank),
            'training_reset_names':sorted(train_names),'validation_reset_names':sorted(val_names),
            'evaluation_reset_names':sorted(test_names),'action_config':asdict(env.base.config),
            'lift_config':asdict(env.lift_config),'observation_schema':env.schema,
            'observation_size':env.observation_space.shape[0],'joint_map':env.base.mapping,
            'steps':args.steps,'seed':args.seed,'checkpoint_interval':args.checkpoint_every,
            'actor_learning_rate':args.actor_lr,'critic_learning_rate':args.critic_lr,
            'fixed_entropy_coefficient':args.entropy,'critic_warmup_updates':args.critic_warmup,
            'initialization':'fresh actor, critics and replay; v36 failed checkpoint is not reused',
            'selection':'episode-boundary deterministic validation; worst-family success, total success, earlier tie',
            'final_test_used_for_selection':False,'smoke_test':args.smoke,
            'source_hashes':{str(p):file_hash(p) for p in Path('teacher').glob('*.py')}}
        # Freeze the source text alongside hashes before the learner starts.
        source=args.output/'source';source.mkdir()
        for path in Path('teacher').glob('*.py'):
            (source/path.name).write_bytes(path.read_bytes())
        (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2))
        if not args.smoke:
            save_status(args.output/'status.json',stage='reference_feasibility')
            reference=[json.loads(s) for s in (Path(env.bank['reference'])/'steps.jsonl').read_text().splitlines()]
            checks=audited_evaluation(env,None,env.entries[:3],args.output/'reference',args.seed+1000,manifest,reference)
            if not all(row['score']['success'] for row in checks):
                save_status(args.output/'status.json',stage='reference_check_failed',results=checks,rl_started=False)
                return
        env.fixed_entry=None
        monitor=Monitor(env,filename=str(args.output/'episodes'))
        model=ConservativeSAC('MlpPolicy',monitor,seed=args.seed,device='cpu',learning_rate=args.critic_lr,
            actor_lr_scale=args.actor_lr/args.critic_lr,critic_warmup_updates=args.critic_warmup,
            buffer_size=max(20000,args.steps),learning_starts=512,batch_size=128,gamma=.99,
            train_freq=1,gradient_steps=1,ent_coef=args.entropy,policy_kwargs={'net_arch':[128,128]},verbose=0)
        with torch.no_grad():
            model.actor.log_std.weight.zero_();model.actor.log_std.bias.fill_(-1.5)
        model.set_logger(configure(str(args.output/'optimizer'),['csv']))
        model.save(args.output/'grasp_initial')
        selector=GraspSelection(args.output,validation)
        save_status(args.output/'status.json',stage='initial_validation',rl_started=False)
        initial=audited_evaluation(env,model,validation,args.output/'validation_initial',args.seed+2000,manifest)
        selector.consider(args.output/'grasp_initial.zip','initial',0,initial)
        def checkpoint_eval(raw,actor,label,step):
            save_status(args.output/'status.json',stage='checkpoint_validation',steps=step,label=label,
                        best_validation=selector.best,final_test_used_for_selection=False)
            return audited_evaluation(raw,actor,validation,args.output/f'validation_{step}',args.seed+2000,manifest)
        checkpoint=GraspCheckpoints(args.output,env,selector,args.checkpoint_every,args.steps,checkpoint_eval)
        progress=GraspProgress(args.output,args.steps)
        env.fixed_entry=None
        monitor.reset(seed=args.seed)
        model.learn(args.steps,callback=CallbackList([progress,checkpoint]))
        model.save(args.output/'grasp_final');model.save_replay_buffer(args.output/'replay_buffer.pkl')
        save_status(args.output/'status.json',stage='final_validation',steps=model.num_timesteps)
        final=audited_evaluation(env,model,validation,args.output/'validation_final',args.seed+2000,manifest)
        if selector.consider(args.output/'grasp_final.zip','final',model.num_timesteps,final):
            model.save_replay_buffer(args.output/'grasp_best_replay.pkl')
        # Freeze the selection before touching the fresh final-test reset set.
        selected_hash=file_hash(args.output/'grasp_best.zip')
        chosen=ConservativeSAC.load(args.output/'grasp_best.zip',device='cpu')
        save_status(args.output/'status.json',stage='selected_policy_final_test',selected=selector.best)
        held=audited_evaluation(env,chosen,final_entries,args.output/'after',args.seed+3000,manifest)
        if file_hash(args.output/'grasp_best.zip')!=selected_hash:
            raise ValueError('Selected checkpoint changed during final testing')
        status={'stage':'checkpointed_grasp_rl_complete','steps':model.num_timesteps,'gradient_updates':model._n_updates,
            'training_episodes':progress.episodes,'training_lifts':progress.successes,
            'selected_checkpoint':selector.best,'selected_checkpoint_sha256':selected_hash,
            'final_policy_validation':family_counts(final),'selected_policy_final_test':family_counts(held),
            'final_test_successes':sum(bool(r['score']['success']) for r in held),'final_test_trials':len(held),
            'checkpoint_validation_seconds':checkpoint.validation_seconds,'objective':'grasp_lift_only',
            'complete_chess_moves_certified':0,'ready_for_act_collection':False}
        save_status(args.output/'status.json',**status)
        print('CHECKPOINTED_GRASP_RL_COMPLETE',json.dumps(status),flush=True)
    except BaseException as error:
        save_status(args.output/'status.json',stage='failed',error=repr(error))
        raise
    finally:
        if env:
            env.close()


if __name__=='__main__':
    main()
