"""Conservative SAC updates and episode-boundary, validation-only selection."""
import copy
import hashlib
import json
from pathlib import Path
import random
import shutil
import time

import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback

from teacher.train import actor_digest


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def policy_success(row):
    return bool(row['score']['success'] and row.get('robot_collision_audit',{}).get('contact_free',True)
        and row.get('camera_clearance_audit',{}).get('passed',True))


class ConservativeSAC(SAC):
    def __init__(self,*args,actor_lr_scale=1/3,critic_warmup_updates=512,**kwargs):
        if (not np.isfinite(actor_lr_scale) or not 0<actor_lr_scale<=1
                or type(critic_warmup_updates) is not int or critic_warmup_updates<0):
            raise ValueError('Invalid conservative actor settings')
        self.actor_lr_scale=actor_lr_scale
        self.critic_warmup_updates=critic_warmup_updates
        super().__init__(*args,**kwargs)

    def _update_learning_rate(self,optimizers):
        super()._update_learning_rate(optimizers)
        scale=0. if self._n_updates<self.critic_warmup_updates else self.actor_lr_scale
        for group in self.actor.optimizer.param_groups:
            group['lr']*=scale
        self.logger.record('train/actor_learning_rate',self.actor.optimizer.param_groups[0]['lr'])


class GraspSelection:
    def __init__(self,output,entries,prefix='grasp',complete_moves=False,prefer_faster=False,
                 objective='unified_e2e4',group_by_case=False):
        self.output=Path(output)
        if not prefix.isidentifier(): raise ValueError('Checkpoint prefix must be an identifier')
        self.prefix=prefix
        self.complete_moves=complete_moves
        self.prefer_faster=prefer_faster
        self.objective=objective;self.group_by_case=group_by_case
        self.cases={e['name']:e['case'] for e in entries} if group_by_case else {}
        self.expected={entry['name']:entry.get('family','open_approach') for entry in entries}
        if len(self.expected)!=len(entries) or not entries:
            raise ValueError('Validation reset identities must be nonempty and unique')
        if prefer_faster and (not complete_moves or 'home' not in self.expected.values()):
            raise ValueError('Efficiency selection requires complete moves and a home validation family')
        self.rows=[];self.best_rank=None;self.best=None

    def consider(self,checkpoint,label,step,results):
        names=[row['reset_name'] for row in results]
        if len(set(names))!=len(names) or set(names)!=set(self.expected):
            raise ValueError('Checkpoint evaluation does not cover the fixed validation set')
        counts={}
        for row in results:
            family=self.expected[row['reset_name']]
            if (row['reset_family']!=family
                    or (bool(row['score'].get('complete_chess_move',False))
                        != (bool(row['score']['success']) if self.complete_moves else False))
                    or (self.complete_moves and row['score'].get('objective')!=self.objective)):
                raise ValueError('Validation task or reset family changed')
            if self.group_by_case and row.get('case')!=self.cases[row['reset_name']]: raise ValueError('Validation goal changed')
            group=f'{self.cases[row["reset_name"]]["id"]}/{family}' if self.group_by_case else family
            count=counts.setdefault(group,{'successes':0,'trials':0})
            count['trials']+=1;count['successes']+=policy_success(row)
        successes=sum(c['successes'] for c in counts.values())
        rank=(min(c['successes']/c['trials'] for c in counts.values()),successes)
        if self.prefer_faster:
            if any(type(r.get('steps')) is not int or r['steps']<1 for r in results):
                raise ValueError('Efficiency selection requires valid physical step counts')
            mean_home=float(np.mean([r['steps'] for r in results if r['reset_family']=='home']))
            # Only after perfect family success, prefer a whole simulated-second
            # improvement. Tiny timing changes retain the earlier actor.
            rank+=(-int(np.ceil(mean_home/30)) if successes==len(results) else 0,)
        record={'label':label,'step':int(step),'checkpoint':str(Path(checkpoint).resolve()),
                'checkpoint_sha256':file_hash(checkpoint),'successes':successes,'trials':len(results),
                'per_family':counts,'rank':list(rank)}
        if self.prefer_faster: record['mean_home_completion_steps']=mean_home
        self.rows.append(record)
        improved=self.best_rank is None or rank>self.best_rank
        if improved:
            shutil.copyfile(checkpoint,self.output/f'{self.prefix}_best.zip')
            self.best_rank=rank;self.best=copy.deepcopy(record)
        criterion='maximize worst-family success fraction, then total successes; ties retain earlier checkpoint'
        if self.prefer_faster:
            criterion='maximize worst-family success fraction, then total successes; only with perfect success prefer lower ceil(mean home steps / 30); ties retain earlier checkpoint'
        report={'criterion':criterion,
                'validation_only':True,'final_test_used_for_selection':False,
                'best':self.best,'checkpoints':self.rows,
                'best_file_sha256':file_hash(self.output/f'{self.prefix}_best.zip')}
        (self.output/'checkpoint_selection.json').write_text(json.dumps(report,indent=2))
        return improved


class GraspCheckpoints(BaseCallback):
    """Evaluate only after SAC stores a terminal transition and auto-resets.

The unused training reset is restored afterwards. Evaluation neither appends
replay entries nor consumes the learner's RNG stream.
"""
    def __init__(self,output,env,selector,interval,total,evaluate):
        super().__init__()
        if interval<1:
            raise ValueError('Checkpoint interval must be positive')
        self.output=Path(output);self.raw_env=env;self.selector=selector
        self.interval=interval;self.next_check=interval;self.total=total;self.evaluate=evaluate
        self.validation_seconds=0.
        self.training_names={row['name'] for row in env.entries}

    def _on_step(self):
        return True

    def _on_rollout_end(self):
        if self.num_timesteps<self.next_check or self.num_timesteps>=self.total:
            return
        if not bool(self.locals['dones'][0]):
            return
        if self.model.n_envs!=1:
            raise ValueError('Checkpoint evaluation currently requires one controlling environment')
        unused=copy.deepcopy(self.raw_env.entry)
        if unused['name'] not in self.training_names or self.raw_env.fixed_entry is not None:
            raise RuntimeError('Learner was not using its training reset distribution')
        env_rng=copy.deepcopy(self.raw_env.np_random.bit_generator.state)
        rng=(random.getstate(),np.random.get_state(),torch.get_rng_state(),
             torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None)
        replay=self.model.replay_buffer
        replay_identity=(replay.pos,replay.full,replay.size())
        before_actor=actor_digest(self.model)
        label=f'checkpoint_{self.num_timesteps}'
        checkpoint=self.output/f'{self.selector.prefix}_{self.num_timesteps}.zip'
        self.model.save(checkpoint)
        started=time.perf_counter()
        try:
            rows=self.evaluate(self.raw_env,self.model,label,self.num_timesteps)
            if (replay.pos,replay.full,replay.size())!=replay_identity or actor_digest(self.model)!=before_actor:
                raise RuntimeError('Validation changed learner weights or replay')
            if self.selector.consider(checkpoint,label,self.num_timesteps,rows):
                self.model.save_replay_buffer(self.output/f'{self.selector.prefix}_best_replay.pkl')
            print(self.selector.prefix.upper()+'_CHECKPOINT',json.dumps(self.selector.rows[-1]),flush=True)
        finally:
            self.raw_env.fixed_entry=unused
            try:
                self.model._last_obs=self.model.get_env().reset()
                self.model._last_episode_starts=np.ones((1,),dtype=bool)
            finally:
                self.raw_env.fixed_entry=None
                self.raw_env.np_random.bit_generator.state=env_rng
                random.setstate(rng[0]);np.random.set_state(rng[1]);torch.set_rng_state(rng[2])
                if rng[3] is not None:
                    torch.cuda.set_rng_state_all(rng[3])
                self.validation_seconds+=time.perf_counter()-started
        self.next_check=(self.num_timesteps//self.interval+1)*self.interval
