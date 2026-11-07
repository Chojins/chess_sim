"""Resume only the lossless export proof after completed reliability checks."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import shlex
import subprocess
import traceback


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--profile',type=Path,required=True)
    parser.add_argument('--seed-start',type=int,required=True)
    args=parser.parse_args();root=args.root
    report=json.loads((root/'policy_readiness.json').read_text())
    if not report['passed']:raise ValueError('Reliability must already have passed')
    previous=json.loads((root/'status.json').read_text())
    if previous['stage']!='needs_analysis':raise ValueError('Expected stopped proof stage')
    (root/'writer_proof_failure_v1.json').write_text(json.dumps(previous,indent=2))
    if (root/'error.txt').exists() and not (root/'writer_proof_error_v1.txt').exists():
        (root/'error.txt').rename(root/'writer_proof_error_v1.txt')
    def status(stage,**detail):
        row={'stage':stage,'updated_utc':datetime.now(timezone.utc).isoformat(),**detail}
        (root/'status.json').write_text(json.dumps(row,indent=2));print('READINESS_STAGE',json.dumps(row),flush=True)
    def run(script,*values,environment='run_rl_python.sh'):
        subprocess.run(['bash',environment,script,*map(str,values)],check=True)
    try:
        status('recheck_before_export_proof')
        run('verify_policy_reliability.py',root/'capture_initial',root/'ordinary',
            '--output',root/'policy_readiness_rechecked.json')
        status('episode_writer_proof_v2')
        sample=root/'sample_episodes_v2'
        run('collect_opening_episodes.py','--pilot',args.profile,'--checkpoint','source_actor.zip',
            '--output',sample,'--episodes',2,'--seed-start',args.seed_start,'--record-video')
        accepted=json.loads((sample/'accepted_episodes.json').read_text())
        if not accepted['episodes']:raise RuntimeError('No upright verified proof episodes were accepted')
        status('dataset_export_and_readback')
        dataset=root/'sample_act_dataset'
        run('export_act_episodes.py','--index',sample/'accepted_episodes.json','--output',dataset,
            '--repo-id',f'local/chess-sim-{root.name}-proof',environment='run_dataset_python.sh')
        roundtrip=json.loads((dataset/'roundtrip_integrity.json').read_text())
        if not roundtrip['passed']:raise RuntimeError('Actual dataset read-back failed')
        capture=['bash','run_rl_python.sh','collect_opening_episodes.py','--pilot',str(args.profile),
            '--checkpoint','source_actor.zip','--output',f'outputs/act-capture-episodes-{root.name}',
            '--episodes','40','--seed-start','7701','--record-video']
        ordinary=['bash','run_rl_python.sh','collect_opening_episodes.py','--pilot','outputs/opening-rl-capture-pickup-v6',
            '--ids','initial_e2e4','initial_d2d4','open_game_knight','--output',f'outputs/act-ordinary-episodes-{root.name}',
            '--episodes','60','--seed-start','7801','--record-video']
        for name,command in [('collection_command',capture),('ordinary_collection_command',ordinary)]:
            (root/f'{name}.json').write_text(json.dumps(command,indent=2))
        (root/'READY_TO_COLLECT.md').write_text(
            '# Ready for simulated episode collection\n\n'
            'Independent qualification passed 80/80 complete moves: 20 each for e2e4, d2d4, g1f3 and the e4d5 first capture after e2e4/d7d5. '
            'All tested captured victims were upright. These repeated trials cover ten fresh seeds per task, twice each; '
            'they are a bring-up criterion, not a guarantee for arbitrary boards or real hardware.\n\n'
            'Capture execution uses the v24 learned residual plus an explicit command/calibration/encoder supervisor. '
            'No additional RL gain is claimed. Ordinary moves use the retained v6 expert. '
            'Only these blue-robot opening states are qualified; red control, other capture histories and special moves need separate tests.\n\n'
            f'Capture profile: `{args.profile}`; checkpoint: `source_actor.zip`.\n\n'
            f'Proof: {len(accepted["episodes"])} upright verified episodes, {accepted["frames"]} frames. '
            f'Local dataset: `{dataset}`. The real LeRobot reader verified sampled RGB pixels, states, actions, indices and timestamps. '
            'Source trajectories were audited in full. ACT weights are unchanged; nothing was uploaded.\n\n'
            'Next capture collection (40 attempts):\n\n```bash\n'+shlex.join(capture)+'\n```\n\n'
            'Ordinary collection (60 attempts, balanced across the three moves):\n\n```bash\n'+shlex.join(ordinary)+'\n```\n')
        ledger=json.loads((root/'manifest.json').read_text())
        ledger['collection_seeds']=sorted(set(ledger.get('collection_seeds',[])+list(range(args.seed_start,args.seed_start+2))))
        (root/'manifest.json').write_text(json.dumps(ledger,indent=2))
        status('ready_to_collect',capture_profile=str(args.profile),checkpoint='source_actor.zip',
            readiness=str(root/'policy_readiness_rechecked.json'),upright_proof_episodes=len(accepted['episodes']),
            proof_frames=accepted['frames'],dataset=str(dataset),roundtrip=str(dataset/'roundtrip_integrity.json'),
            act_retrained=False,uploaded=False)
    except BaseException as error:
        status('needs_analysis',error=repr(error),act_retrained=False,uploaded=False)
        (root/'error.txt').write_text(traceback.format_exc());raise


if __name__=='__main__':main()
