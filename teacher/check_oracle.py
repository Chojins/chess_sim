"""Physical calibration check of an ANALYTIC oracle; never learned-policy success."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import numpy as np
from teacher.task_space import TaskSpaceMoveEnv
from teacher.move_teacher import MoveConfig
from teacher.reactive_oracle import ReactiveOracle
from teacher.train_grasp import evaluate_bank,save_status
from teacher.verify_task_space import audit_trial
from teacher.grasp_checkpoints import file_hash


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bank',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--url',default='http://127.0.0.1:8081');parser.add_argument('--split',default='train')
    parser.add_argument('--families',nargs='+',default=['home']);parser.add_argument('--maximum-trials',type=int,default=1)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False);env=None
    try:
        env=TaskSpaceMoveEnv(args.bank,args.url,split=args.split,config=MoveConfig(),adaptive=False,gamma=.997)
        entries=[r for r in env.entries if r['family'] in args.families][:args.maximum_trials]
        if not entries: raise ValueError('No oracle calibration starts')
        oracle=ReactiveOracle(env.cartesian_config)
        manifest={'objective':'unified_e2e4','teacher_label':'analytic_geometry_oracle','learned_policy':False,
            'bank':str(args.bank.resolve()),'bank_sha256':file_hash(args.bank),'case':env.bank['case'],
            'action_config':asdict(env.base.config),'move_config':asdict(env.move_config),'joint_map':env.base.mapping,
            'observation_schema':env.schema,'observation_size':env.observation_space.shape[0],
            'policy_action_space':env.policy_action_space,'cartesian_config':asdict(env.cartesian_config),
            'oracle_config':asdict(oracle.config),'oracle_version':oracle.version,
            'source_hashes':{str(p):file_hash(p) for p in Path('teacher').glob('*.py')},
            'training_enabled':False,'ready_for_act_collection':False}
        (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2))
        save_status(args.output/'status.json',stage='analytic_calibration_probe',learned_policy=False)
        results=evaluate_bank(env,oracle,entries,args.output/'trials',97001)
        for entry in entries: audit_trial(args.output/'trials'/entry['name'],manifest)
        save_status(args.output/'status.json',stage='analytic_calibration_probe_complete',learned_policy=False,
            successes=sum(r['score']['success'] for r in results),trials=len(results),results=results,ready_for_act_collection=False)
    except BaseException as error:
        save_status(args.output/'status.json',stage='failed',error=repr(error),learned_policy=False);raise
    finally:
        if env: env.close()


if __name__=='__main__': main()
