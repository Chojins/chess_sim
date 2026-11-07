"""Qualify the explicit capture supervisor plus learned residual for ACT data.

All work is local simulation. Fresh capture tests precede ordinary regressions.
If the frozen candidate fails, one conservative, independently audited SAC run
uses newly collected transitions from this controller, never old-MDP replay.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import subprocess
import time
import traceback


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--profile',type=Path,default=Path('outputs/capture-transport-profile-v28'))
    parser.add_argument('--probe',type=Path,default=Path('outputs/capture-transport-probe-v28'))
    parser.add_argument('--seed-base',type=int,default=5300,help='Fresh seed range: base+1 captures, +201 replay, +300 training, +401 held-out, +501 ordinary, +601 proof')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    seed_base=args.seed_base
    def retire_completed_stage_seeds():
        # Nested stages must be visible to existing top-level freshness checks.
        # Record only stages already created; reserving future seeds here
        # would make their own evaluation commands reject them as reused.
        ledger={'mode':'pipeline_seed_retirement_ledger','note':'Provenance bookkeeping; not a training dataset'}
        for key in ('source_seeds','evaluation_seeds','collection_seeds'):
            ledger[key]=[]
        for path in args.output.rglob('manifest.json'):
            if path==args.output/'manifest.json':continue
            manifest=json.loads(path.read_text())
            for key in ('source_seeds','evaluation_seeds','collection_seeds'):
                ledger[key].extend(manifest.get(key,[]))
            if manifest.get('training_seed') is not None:
                ledger['source_seeds'].append(manifest['training_seed'])
        for key in ('source_seeds','evaluation_seeds','collection_seeds'):
            ledger[key]=sorted(set(ledger[key]))
        (args.output/'manifest.json').write_text(json.dumps(ledger,indent=2))
    def status(stage,**details):
        retire_completed_stage_seeds()
        row={'stage':stage,'updated_utc':datetime.now(timezone.utc).isoformat(),**details}
        (args.output/'status.json').write_text(json.dumps(row,indent=2))
        print('READINESS_STAGE',json.dumps(row),flush=True)
    def run(script,*values):
        command=['bash','run_rl_python.sh',script,*map(str,values)]
        print('RUN',json.dumps(command),flush=True);subprocess.run(command,check=True)
    def wait_finished(root):
        deadline=time.monotonic()+7200
        while not (root/'status.json').exists():
            if time.monotonic()>deadline:raise RuntimeError(f'Timed out waiting for {root}')
            time.sleep(5)
        state=json.loads((root/'status.json').read_text())
        if state['status']!='complete':raise RuntimeError(f'Previous stage failed: {state}')
    def physical_audit(root):
        from verify_policy_reliability import recompute
        from evaluate_openings import audit_episode
        manifest=json.loads((root/'manifest.json').read_text())
        rows=json.loads((root/'results.json').read_text())
        if not json.loads((root/'integrity.json').read_text())['passed']:
            raise ValueError('Frozen evaluation audit failed')
        moves={case['id']:case['move'] for case in manifest['cases']}
        for row in rows:
            path=root/f"repeat_{row['repeat']}"/row['task_id']/f"{row['label']}_{row['seed']}"
            audit_episode(path,row,moves[row['task_id']],manifest);recompute(path)
        return manifest,rows
    def capture_test(pilot,checkpoint,root,seed):
        run('evaluate_openings.py','--output',root,'--pilot',pilot,'--learned-checkpoint',checkpoint,
            '--only-policy','learned','--ids','scandinavian_capture','--seeds',*range(seed,seed+10),
            '--repeats',2,'--record-video','--comparison-note',
            'Twenty fresh full captures with explicit calibrated victim transport and learned alignment; no pure-RL improvement claim')
        from verify_policy_reliability import assess
        manifest,rows=physical_audit(root)
        report=assess(manifest,rows)
        (root/'capture_readiness.json').write_text(json.dumps(report,indent=2))
        return all(row['passed'] for row in report['per_task'].values())
    try:
        status('waiting_for_physical_probe')
        wait_finished(args.probe)
        _,probe=physical_audit(args.probe)
        pilot,checkpoint=args.profile,'source_actor.zip'
        captures=args.output/'capture_initial'
        # A two-of-three bring-up screen is not the final reliability gate.
        initial_pass=False
        if sum(row['score']['success'] for row in probe)>=2:
            status('capture_reliability_initial')
            initial_pass=capture_test(pilot,checkpoint,captures,seed_base+1)
        if not initial_pass:
            training=args.output/'stable_rl'
            status('conservative_residual_rl',reason='Frozen candidate below capture threshold; fresh-controller replay only')
            run('train_openings.py','--output',training,'--ids','scandinavian_capture',
                '--capture-alignment','--capture-home-retry','--live-timing','--adaptive-alignment',
                '--alignment-steps',32,'--arm-settle-limit',90,'--latch-alignment',
                '--board-grasps-only','--release-aware','--capture-transport','--capture-step-limit',1600,
                '--grasp-interventions',4,'--initialize-from',pilot/checkpoint,
                '--replay-seeds',*range(seed_base+201,seed_base+207),'--bc-weight',5,'--exploration-log-std',-4,
                '--actor-lr-scale',.1,'--critic-warmup-updates',1000,'--retention-loss-ratio',1.15,
                '--steps',6000,'--eval-every',2000,'--seed',seed_base+300,'--eval-seeds',seed_base+301,seed_base+302,'--record-evaluations')
            run('verify_opening_rl.py',training)
            pilot,checkpoint=training,'residual_best.zip'
            captures=args.output/'capture_trained'
            status('capture_reliability_after_rl')
            if not capture_test(pilot,checkpoint,captures,seed_base+401):
                raise RuntimeError('Capture reliability remains below 90%; preserve traces and diagnose, without collecting ACT data')
        ready=args.output/'capture_passed.status';ready.write_text('exit_code=0\n')
        ordinary=args.output/'ordinary'
        status('ordinary_regression')
        subprocess.run(['bash','run_ordinary_reliability.sh',str(ordinary),str(ready),str(seed_base+501)],check=True)
        run('verify_policy_reliability.py',captures,ordinary,'--output',args.output/'policy_readiness.json')
        status('episode_writer_proof')
        sample=args.output/'sample_episodes'
        run('collect_opening_episodes.py','--pilot',pilot,'--checkpoint',checkpoint,'--output',sample,
            '--episodes',2,'--seed-start',seed_base+601,'--record-video')
        accepted=json.loads((sample/'accepted_episodes.json').read_text())
        if not accepted['episodes']:
            raise RuntimeError('Reliability passed, but no upright proof episode was accepted; diagnose export quality')
        dataset=args.output/'sample_act_dataset'
        subprocess.run(['bash','run_dataset_python.sh','export_act_episodes.py','--index',str(sample/'accepted_episodes.json'),
            '--output',str(dataset),'--repo-id',f'local/chess-sim-{args.output.name}-proof'],check=True)
        command=['bash','run_rl_python.sh','collect_opening_episodes.py','--pilot',str(pilot),
                 '--checkpoint',checkpoint,'--output',f'outputs/act-capture-episodes-{args.output.name}',
                 '--episodes','40','--seed-start',str(seed_base+701),'--record-video']
        (args.output/'collection_command.json').write_text(json.dumps(command,indent=2))
        (args.output/'READY_TO_COLLECT.md').write_text(
            '# Ready for local episode collection\n\n'
            'The attached readiness report verifies >=90% complete-move success on >=20 fresh trials per tested task. '
            'Capture execution uses an explicit command/calibration/encoder supervisor plus the learned residual; '
            'this is not a claim that RL learned the entire transfer. The capture profile supports this tested first-capture opening only. '
            'Ordinary moves use the retained v6 expert; red remains at home. Other board states, special moves and real hardware remain unqualified.\n\n'
            f'Selected capture profile: `{pilot}`; checkpoint: `{checkpoint}`.\n\n'
            f'Upright verified sample episodes: {len(accepted["episodes"])}. Local ACT dataset: `{dataset}`. '
            'Existing ACT weights are unchanged and nothing was uploaded.\n\n'
            'Next capture collection command:\n\n```bash\n'+shlex.join(command)+'\n```\n')
        status('ready_to_collect',capture_profile=str(pilot),checkpoint=checkpoint,
            readiness=str(args.output/'policy_readiness.json'),upright_proof_episodes=len(accepted['episodes']),
            dataset=str(dataset),act_retrained=False,uploaded=False)
    except BaseException as error:
        status('needs_analysis',error=repr(error),act_retrained=False,uploaded=False)
        (args.output/'error.txt').write_text(traceback.format_exc());raise


if __name__=='__main__':main()
