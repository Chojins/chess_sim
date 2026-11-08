"""Build goal-labelled fresh home resets with disjoint episode partitions."""
import argparse
import json
from pathlib import Path
import numpy as np
from teacher.grasp_checkpoints import file_hash
from teacher.ordinary_moves import ordinary_case,validate_case_geometry
from teacher.train_grasp import save_status
from teacher.transport import StateClient


def selected_cases(moves=None,history=None,cases_file=None):
    if cases_file:
        records=json.loads(Path(cases_file).read_text())
        cases=[ordinary_case(r.get('history',[]),r['move']) for r in records]
    else: cases=[ordinary_case(history or [],m) for m in (moves or ['e2e4','e2e3','d2d4','c2c4'])]
    if not cases or len({c['id'] for c in cases})!=len(cases): raise ValueError('Need nonempty unique ordinary cases')
    return cases


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--name-prefix',required=True)
    p.add_argument('--url',default='http://127.0.0.1:8081');p.add_argument('--cases-file',type=Path)
    p.add_argument('--moves',nargs='+');p.add_argument('--history',nargs='*',default=[])
    p.add_argument('--train-seeds',nargs='+',type=int,default=[121003,121004,121005])
    p.add_argument('--validation-seeds',nargs='+',type=int,default=[122001])
    p.add_argument('--test-seeds',nargs='+',type=int,default=[123001])
    args=p.parse_args();cases=selected_cases(args.moves,args.history,args.cases_file)
    partitions={'train':args.train_seeds,'validation':args.validation_seeds,'test':args.test_seeds}
    seeds=sum(partitions.values(),[])
    if min(seeds)<0 or len(set(seeds))!=len(seeds): p.error('Seed partitions must be nonnegative and disjoint')
    args.output.mkdir(parents=True,exist_ok=False);client=StateClient(args.url);entries=[]
    try:
        for split,group in partitions.items():
            for case in cases:
                for seed in group:
                    offset=np.random.default_rng(seed).uniform(-.5,.5,2)
                    latest=client.reset({'history':case['history'],'move':case['move'],'pawn_offset_mm':offset.tolist()})
                    validate_case_geometry(case,latest['observation']['task_geometry'])
                    history={'case':case,'seed':seed,'split':split,'family':'home','held_start':False,'move':case['move'],
                        'prior_lift_verified':False,'prior_max_lift_m':0.,'prior_release_verified':False,'release_intent':None,
                        'pawn_offset_mm':offset.tolist()}
                    name=f'{args.name_prefix}_{case["id"]}_{seed}'
                    saved=client.save_reset(name,history)['reset_context'];context=client.restore_reset(name)['reset_context']
                    if context['checkpoint_sha256']!=saved['sha256'] or max(context['parity'].values())>1e-5:
                        raise ValueError('Goal reset parity failed')
                    entries.append({**saved,**history,'horizon':800,'restore_parity':context['parity']})
                    save_status(args.output/'status.json',stage='building_ordinary_home_bank',checkpoints=len(entries),case=case,split=split)
        bank={'format':'ordinary_move_home_bank_v1','case':cases[0],'cases':cases,'entries':entries,'partitions':partitions,
            'reset_builder':'fresh_ordinary_home_physics','source':'Actual reset states only; no policy qualification implied',
            'goal':'Requested piece placed/released upright on its legal target, arm returned home',
            'scope':'Blue ordinary pawn/knight goals, current calibration; no captures or red control'}
        path=args.output/'manifest.json';path.write_text(json.dumps(bank,indent=2))
        save_status(args.output/'status.json',stage='ordinary_home_bank_ready',checkpoints=len(entries),bank_sha256=file_hash(path))
    except BaseException as error:
        save_status(args.output/'status.json',stage='failed',error=repr(error));raise
    finally: client.close()


if __name__=='__main__': main()
