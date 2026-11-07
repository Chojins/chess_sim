"""Add physical near-completion starts so the shared goal has easy examples."""
import argparse
import copy
import json
from pathlib import Path
import numpy as np
from teacher.move_teacher import MoveEnv


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bank',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--url',default='http://127.0.0.1:8081');args=parser.parse_args()
    if args.output.exists(): raise ValueError('Refusing to overwrite reset index')
    bank=json.loads(args.bank.read_text());bank['continuation_directory']=str(args.bank.parent.resolve())
    env=MoveEnv(args.bank,args.url)
    try:
        for entry in [r for r in bank['entries'] if r['family']=='near_home']:
            env.fixed_entry=entry;env.reset(seed=entry['seed']);trace=[]
            home=np.r_[env.task.home,5.4]
            while np.max(np.abs(np.asarray(env.base.actual['measured_joint_deg'])[:5]-env.task.home))>=3.5:
                action=env.base.normalized_target(home)
                obs,reward,done,truncated,info=env.step(action)
                trace.append({'action':action.tolist(),'observation':env.base.actual,'score':info['score']})
                if done or truncated: raise ValueError('Boundary must precede completion')
            history={**{k:entry[k] for k in ('move','seed','split','grasp_checkpoint_sha256','reference_sha256')},
                'family':'home_settle','held_start':False,'prior_lift_verified':True,'prior_max_lift_m':env.task.max_lift,
                'prior_release_verified':True,'release_intent':env.task.release_location,'parent_reset':entry['name']}
            saved=env.base.client.save_reset(entry['name'].replace('near_home','home_settle'),history)['reset_context']
            restored=env.base.client.restore_reset(saved['name'])['reset_context']
            if max(restored['parity'].values())>1e-5: raise ValueError('Boundary reset parity failed')
            boundary={**saved,**history,'physical_step':0,'prefix_step':0,'reference_hold_only':True,
                'horizon':96,'continuation_sha256':entry['continuation_sha256'],'restore_parity':restored['parity']}
            bank['entries'].append(boundary)
            trace_path=args.output.parent/f'home_boundary_{entry["seed"]}.json'
            trace_path.write_text(json.dumps({'parent':entry['name'],'steps':trace},indent=2))
            print('HOME_BOUNDARY_SAVED',saved['name'],flush=True)
        args.output.write_text(json.dumps(bank,indent=2))
    finally: env.close()


if __name__=='__main__': main()
