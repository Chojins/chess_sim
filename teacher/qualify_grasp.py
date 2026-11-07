"""Qualify one frozen grasp/lift actor on newly reached physical reset states.

This process cannot train or select between actors. It does not certify a
complete chess move, home-start execution, RGB deployment, or real hardware.
"""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

import torch

from teacher.conservative_grasp import audited_evaluation,family_counts
from teacher.grasp_checkpoints import ConservativeSAC,file_hash
from teacher.near_grasp import NearGraspEnv,LiftConfig
from teacher.train_grasp import save_status


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot',type=Path,required=True)
    parser.add_argument('--checkpoint',default='grasp_best.zip')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--url',default='http://127.0.0.1:8081')
    parser.add_argument('--seeds',type=int,nargs='+',required=True)
    parser.add_argument('--minimum-trials-per-family',type=int,default=10)
    parser.add_argument('--minimum-success-fraction',type=float,default=.9)
    args=parser.parse_args()
    if (len(set(args.seeds))!=len(args.seeds) or not args.seeds or min(args.seeds)<0
            or args.minimum_trials_per_family<1 or not 0<args.minimum_success_fraction<=1):
        parser.error('Need unique seeds and valid qualification thresholds')
    args.output.mkdir(parents=True,exist_ok=False)
    source=json.loads((args.pilot/'manifest.json').read_text())
    checkpoint=args.pilot/args.checkpoint
    frozen=args.output/'frozen_actor.zip'
    shutil.copyfile(checkpoint,frozen)
    actor_hash=file_hash(frozen)
    (args.output/'source_manifest.json').write_text(json.dumps(source,indent=2))
    source_dir=args.output/'source';source_dir.mkdir()
    for path in Path('teacher').glob('*.py'):
        (source_dir/path.name).write_bytes(path.read_bytes())
    env=None
    try:
        save_status(args.output/'status.json',stage='preparing_fresh_resets',checkpoint_sha256=actor_hash,
                    requested_seeds=args.seeds,training_enabled=False)
        bank_dir=args.output/'reset_set'
        subprocess.run([sys.executable,'-m','teacher.build_reset_bank','--output',str(bank_dir),
            '--url',args.url,'--prefix-steps','140','155','--allow-contact-starts','--evaluation-only',
            '--eval-seeds',*map(str,args.seeds)],check=True)
        bank_path=bank_dir/'manifest.json'
        torch.set_num_threads(2)
        env=NearGraspEnv(bank_path,args.url,split='evaluation',lift_config=LiftConfig(**source['lift_config']))
        # Same actor inputs, action limits, task and calibration as the selected
        # source. Only physical reset seed membership changes.
        from dataclasses import asdict
        if json.loads(json.dumps(asdict(env.base.config)))!=source['action_config'] or env.base.mapping!=source['joint_map']:
            raise ValueError('Frozen policy and qualification control contracts differ')
        model=ConservativeSAC.load(frozen,device='cpu')
        if model.observation_space.shape!=env.observation_space.shape:
            raise ValueError('Frozen actor observation shape differs')
        manifest={**source,'mode':'frozen_grasp_lift_qualification','bank':str(bank_path.resolve()),
            'bank_sha256':file_hash(bank_path),'source_pilot':str(args.pilot.resolve()),
            'checkpoint':str(frozen.resolve()),'checkpoint_sha256':actor_hash,
            'qualification_seeds':args.seeds,'evaluation_reset_names':[r['name'] for r in env.entries],
            'training_enabled':False,'checkpoint_selection_enabled':False,
            'minimum_trials_per_family':args.minimum_trials_per_family,
            'minimum_success_fraction':args.minimum_success_fraction,'smoke_test':False}
        (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2))
        save_status(args.output/'status.json',stage='frozen_policy_evaluation',trials=len(env.entries),training_enabled=False)
        rows=audited_evaluation(env,model,env.entries,args.output/'evaluation',71001,manifest)
        counts=family_counts(rows)
        for value in counts.values():
            value['success_fraction']=value['successes']/value['trials']
            value['passed']=(value['trials']>=args.minimum_trials_per_family
                             and value['success_fraction']>=args.minimum_success_fraction)
        if file_hash(frozen)!=actor_hash or file_hash(checkpoint)!=actor_hash:
            raise ValueError('Frozen checkpoint changed during qualification')
        status={'stage':'grasp_qualification_complete','passed':all(r['passed'] for r in counts.values()),
            'per_reset_family':counts,'successes':sum(bool(r['score']['success']) for r in rows),
            'trials':len(rows),'checkpoint_sha256':actor_hash,'training_enabled':False,
            'objective':'grasp_lift_only','complete_chess_moves_certified':0,'ready_for_act_collection':False,
            'scope':'Blue e2e4 near-grasp and pre-lift contact states only; not home-start or complete moves',
            'note':'Two prefixes share each reset seed; these are descriptive bring-up fractions, not independent-sample confidence guarantees.'}
        save_status(args.output/'status.json',**status)
        print('GRASP_QUALIFICATION_COMPLETE',json.dumps(status),flush=True)
    except BaseException as error:
        save_status(args.output/'status.json',stage='failed',error=repr(error),training_enabled=False)
        raise
    finally:
        if env:
            env.close()


if __name__=='__main__':
    main()
