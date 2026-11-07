"""Collect train-only on-policy states for corrective imitation (not ACT data)."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from teacher.grasp_checkpoints import ConservativeSAC,file_hash
from teacher.reactive_oracle import ReactiveOracle
from teacher.review_policy import make_env
from teacher.train_grasp import evaluate_bank,save_status


class TrainingMixture:
    def __init__(self,actor,oracle,beta,seed):
        self.actor=actor;self.oracle=oracle;self.beta=beta;self.rng=np.random.default_rng(seed)
        self.interventions=0;self.steps=0
    def predict(self,observation,deterministic=True):
        self.steps+=1
        if self.rng.random()<self.beta:
            self.interventions+=1
            return self.oracle.predict(observation)
        return self.actor.predict(observation,deterministic=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--checkpoint',default='move_bc.zip')
    p.add_argument('--output',type=Path,required=True);p.add_argument('--url',default='http://127.0.0.1:8081')
    p.add_argument('--bank',type=Path,help='Use a compatible new reset bank without changing the frozen actor')
    p.add_argument('--families',nargs='+',default=['home','approach','open_grasp'])
    p.add_argument('--maximum-trials',type=int,default=9);p.add_argument('--beta',type=float,default=0.)
    p.add_argument('--seed',type=int,default=99001)
    args=p.parse_args()
    if not 0<=args.beta<=1 or args.maximum_trials<1: p.error('Invalid collection budget/mixture')
    torch.set_num_threads(2);args.output.mkdir(parents=True,exist_ok=False);env=None
    source=json.loads((args.run/'manifest.json').read_text());checkpoint=args.run/args.checkpoint;digest=file_hash(checkpoint)
    if args.bank:
        bank=json.loads(args.bank.read_text())
        if bank['case']!=source['case']: raise ValueError('New reset bank task differs')
        source={**source,'bank':str(args.bank.resolve()),'bank_sha256':file_hash(args.bank)}
    try:
        env,audit=make_env(source,args.url,'train')
        entries=[r for r in env.entries if r['family'] in args.families][:args.maximum_trials]
        if not entries or any(e['split']!='train' for e in entries): raise ValueError('Need training-only resets')
        model=ConservativeSAC.load(checkpoint,device='cpu')
        mixture=TrainingMixture(model,ReactiveOracle(env.cartesian_config),args.beta,args.seed)
        manifest={**source,'mode':'training_state_correction_collection','training_enabled':False,
            'teacher_label':'analytic_neural_training_mixture' if args.beta else 'frozen_neural_actor',
            'learned_policy':not bool(args.beta),'source_checkpoint':str(checkpoint.resolve()),
            'source_checkpoint_sha256':digest,'oracle_intervention_probability':args.beta,
            'accepted_for_act_training':False,'ready_for_act_collection':False}
        (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2))
        save_status(args.output/'status.json',stage='collecting_training_corrections',beta=args.beta)
        results=evaluate_bank(env,mixture,entries,args.output/'trials',args.seed)
        for entry in entries: audit(args.output/'trials'/entry['name'],manifest)
        if file_hash(checkpoint)!=digest: raise ValueError('Frozen actor changed during collection')
        save_status(args.output/'status.json',stage='training_corrections_complete',trials=len(results),
            successes=sum(r['score']['success'] for r in results),
            home_successes=sum(r['score']['success'] for r in results if r['reset_family']=='home'),
            steps=mixture.steps,oracle_interventions=mixture.interventions,results=results,
            ready_for_act_collection=False)
    except BaseException as error:
        save_status(args.output/'status.json',stage='failed',error=repr(error));raise
    finally:
        if env: env.close()


if __name__=='__main__': main()
