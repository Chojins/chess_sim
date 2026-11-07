"""Index physical open-grasp starts under the shared completion objective."""
import argparse
import json
from pathlib import Path
import numpy as np
from teacher.transport import StateClient


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bank',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--url',default='http://127.0.0.1:8081');args=parser.parse_args()
    if args.output.exists(): raise ValueError('Refusing to overwrite reset index')
    bank=json.loads(args.bank.read_text());client=StateClient(args.url)
    try:
        for parent in [r for r in bank['entries'] if r['family']=='held_source']:
            name=parent['name'].replace('held_source','grasp_start')
            latest=client.restore_reset(name);context=latest['reset_context'];actual=latest['observation']
            i=actual['task_geometry']['source_index'];baseline=context['baseline_observation']
            height=actual['pieces']['piece_centers_m'][i][2]-baseline['pieces']['piece_centers_m'][i][2]
            if abs(height)>.002 or actual['pieces']['piece_upright_cosines'][i]<.98:
                raise ValueError('Open grasp must precede lift')
            history={k:parent[k] for k in ('seed','split','move','grasp_checkpoint_sha256','reference_sha256')}
            history.update(family='open_grasp',held_start=False,prior_lift_verified=False,prior_max_lift_m=0.,
                prior_release_verified=False,release_intent=None,parent_reset=name)
            saved=client.save_reset(parent['name'].replace('held_source','open_grasp'),history)['reset_context']
            restored=client.restore_reset(saved['name'])['reset_context']
            if max(restored['parity'].values())>1e-5: raise ValueError('Open grasp reset parity failed')
            bank['entries'].append({**saved,**history,'physical_step':140,'prefix_step':140,
                'horizon':min(800,parent['horizon']+100),'continuation_sha256':parent['continuation_sha256'],
                'restore_parity':restored['parity']})
            print('OPEN_GRASP_SAVED',saved['name'],flush=True)
        args.output.write_text(json.dumps(bank,indent=2))
    finally: client.close()


if __name__=='__main__': main()
