"""Sequential simulation-only teacher, residual-training and reliability stages.

Run with run_rl_python.sh. Every later stage depends on audited earlier results;
neither ACT weights nor hardware are modified. Inspect pipeline/status.json.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time
import traceback


def screening_readiness(results):
    """Training requires legal completions and stable RHS removals.

    Uprightness remains a reward preference and ACT-export requirement, as
    requested by the user; a toppled off-board victim is not a chess failure.
    This small screening test does not establish policy reliability.
    """
    identities = {(row['seed'], row['repeat'], row['task_id']) for row in results}
    if len(identities) != len(results):
        raise ValueError('Duplicate physical screening trials')
    successes = sum(bool(row['score']['success']) for row in results)
    stable = upright = 0
    for row in results:
        capture = row['score'].get('capture') or {}
        removed = bool(capture.get('lifted') and capture.get('inside_actor_rhs') and capture.get('released_stable'))
        stable += removed
        upright += bool(removed and capture.get('upright_stable'))
    return {'criterion_version': 2, 'trials': len(results), 'complete_successes': successes,
            'stable_rhs_removals': stable, 'upright_removals': upright,
            'passed': len(results) == 5 and successes >= 2 and stable >= 4,
            'purpose': 'training readiness only; no reliability claim',
            'uprightness': 'reward preference; still required for exported ACT episodes'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    resume = parser.add_mutually_exclusive_group()
    resume.add_argument('--resume-screen', action='store_true', help='Resume completed, hashed supervised initialization after a pre-screen failure')
    resume.add_argument('--resume-rl', action='store_true', help='Re-audit completed screening trials and resume at RL without repeating them')
    args = parser.parse_args()
    resuming = args.resume_screen or args.resume_rl
    args.output.mkdir(parents=True, exist_ok=resuming)
    status_file = args.output/'status.json'
    if args.resume_screen:
        previous = json.loads(status_file.read_text())
        if previous['stage'] != 'needs_analysis' or Path('outputs/geometric-supervised-screen-v24').exists():
            raise ValueError('Screen resume requires a stopped pipeline before any screening trials')
        (args.output/'pre_screen_failure.json').write_text(json.dumps(previous, indent=2))
    if args.resume_rl:
        previous = json.loads(status_file.read_text())
        if previous['stage'] != 'needs_analysis' or Path('outputs/geometric-residual-rl-v25').exists():
            raise ValueError('RL resume requires a stopped pipeline before training starts')
        (args.output/'pre_rl_screen_failure.json').write_text(json.dumps(previous, indent=2))
    def status(stage, **details):
        row = {'stage': stage, 'updated_utc': datetime.now(timezone.utc).isoformat(), **details}
        status_file.write_text(json.dumps(row, indent=2))
        print('PIPELINE_STAGE', json.dumps(row), flush=True)
    def run(script, *arguments):
        command = ['bash', 'run_rl_python.sh', script, *map(str, arguments)]
        print('RUN', json.dumps(command), flush=True)
        subprocess.run(command, check=True)
    def wait_for_collection(root):
        deadline = time.monotonic()+3600
        while not (root/'status.json').exists():
            if time.monotonic() > deadline:
                raise RuntimeError(f'Teacher collection did not finish within an hour: {root}')
            time.sleep(5)
        result = json.loads((root/'status.json').read_text())
        if result.get('status') != 'complete' or not result.get('act_checkpoint_unchanged'):
            raise RuntimeError(f'Teacher failed: {root}: {result}')
    try:
        demos = Path('outputs/geometric-demonstrations-v24')
        if resuming:
            from capture_demonstrations import file_hash, load_demonstrations
            load_demonstrations(demos/'demonstrations.npz')
            prior = json.loads((demos/'pretraining/manifest.json').read_text())
            if file_hash(demos/'pretraining/demo_pretrained.zip') != prior['pretrained_sha256']:
                raise ValueError('Supervised checkpoint changed before resume')
        else:
            from prepare_geometric_demonstrations import episode_quality
            roots = [Path('outputs/geometric-captures-v23')]
            status('teacher_collection', roots=list(map(str, roots)))
            wait_for_collection(roots[0])
            eligible = 0
            for batch in range(3):
                root = roots[-1]
                run('verify_geometric_teacher.py', root)
                for result in json.loads((root/'results.json').read_text()):
                    directory = root/result['task_id']/f"teacher_{result['seed']}"
                    eligible += episode_quality(directory, result) is None
                status('teacher_collection', eligible_complete_episodes=eligible, roots=list(map(str, roots)))
                if eligible >= 4:
                    break
                if batch == 2:
                    raise RuntimeError(f'Only {eligible} complete centred teacher episodes after 18 attempts; diagnose transport before training')
                root = Path(f'outputs/geometric-captures-v23{chr(ord("b")+batch)}')
                roots.append(root)
                start = 3811+batch*10
                run('collect_geometric_grasps.py', '--output', root, '--seeds', *range(start, start+6), '--release-aware')
            status('prepare_supervision', teacher_episodes=eligible)
            run('prepare_geometric_demonstrations.py', *roots, '--output', demos)
            status('supervised_initialization')
            run('pretrain_capture_residual.py', '--demonstrations', demos/'demonstrations.npz', '--updates', 2000)
        pilot = demos/'pretraining'
        screening = Path('outputs/geometric-supervised-screen-v24')
        if not args.resume_rl:
            status('physical_screen')
            run('evaluate_openings.py', '--output', screening, '--pilot', pilot,
                '--learned-checkpoint', 'demo_pretrained.zip', '--only-policy', 'learned',
                '--ids', 'scandinavian_capture', '--seeds', *range(3901,3906), '--repeats', 1,
                '--record-video', '--learned-label', 'supervised residual from centred complete-move teacher',
                '--comparison-note', 'Five fresh physical screening trials, not reliability certification')
        else:
            status('recheck_physical_screen', reason='Toppling is a soft preference, not a hard capture failure')
        from verify_policy_reliability import recompute, assess
        results = json.loads((screening/'results.json').read_text())
        from capture_demonstrations import file_hash
        from evaluate_openings import audit_episode
        screen_manifest = json.loads((screening/'manifest.json').read_text())
        if (not json.loads((screening/'integrity.json').read_text())['passed'] or
                len(results) != screen_manifest['expected_episodes'] or
                file_hash(pilot/'demo_pretrained.zip') != screen_manifest['model_hashes']['learned']):
            raise ValueError('Physical screening is incomplete or the actor changed')
        for result in results:
            directory = screening/f"repeat_{result['repeat']}"/result['task_id']/f"learned_{result['seed']}"
            audit_episode(directory, result, 'e4d5', screen_manifest)
            recompute(directory)
        screen = screening_readiness(results)
        if args.resume_rl:
            prior_screen = screening/'screening.json'
            if prior_screen.exists():
                (screening/'screening_upright_gate_v1.json').write_text(prior_screen.read_text())
        (screening/'screening.json').write_text(json.dumps(screen, indent=2))
        if not screen['passed']:
            raise RuntimeError(f'Supervised actor failed physical screening: {screen}; do not launch blind RL')
        training = Path('outputs/geometric-residual-rl-v25')
        status('residual_rl_training', physical_screen=screen)
        run('train_openings.py', '--output', training, '--ids', 'scandinavian_capture',
            '--capture-alignment', '--capture-home-retry', '--live-timing', '--adaptive-alignment',
            '--alignment-steps', 32, '--arm-settle-limit', 90, '--latch-alignment',
            '--board-grasps-only', '--release-aware', '--grasp-interventions', 4,
            '--initialize-from', pilot/'demo_pretrained.zip', '--demonstrations', demos/'demonstrations.npz',
            '--prefill-demonstration-replay', '--bc-weight', .5, '--exploration-log-std', -4,
            '--steps', 4000, '--eval-every', 2000, '--seed', 4000, '--eval-seeds', 4001, 4002,
            '--record-evaluations')
        run('verify_opening_rl.py', training)
        captures = Path('outputs/geometric-capture-reliability-v25')
        status('capture_reliability')
        run('evaluate_openings.py', '--output', captures, '--pilot', training, '--only-policy', 'learned',
            '--ids', 'scandinavian_capture', '--seeds', *range(4101,4111), '--repeats', 2, '--record-video',
            '--comparison-note', 'Twenty fresh complete captures; source retained on selection ties is not an RL improvement')
        report = assess(json.loads((captures/'manifest.json').read_text()), json.loads((captures/'results.json').read_text()))
        (captures/'capture_readiness.json').write_text(json.dumps(report, indent=2))
        if not all(row['passed'] for row in report['per_task'].values()):
            raise RuntimeError('Complete-capture reliability below 90%; no ACT episodes collected')
        ready = args.output/'capture_ready.status'; ready.write_text('exit_code=0\n')
        ordinary = Path('outputs/geometric-ordinary-reliability-v25')
        status('ordinary_regression')
        subprocess.run(['bash','run_ordinary_reliability.sh',str(ordinary),str(ready), '4201'], check=True)
        run('verify_policy_reliability.py', captures, ordinary, '--output', args.output/'policy_readiness.json')
        status('verified_episode_collection')
        run('collect_opening_episodes.py', '--pilot', training, '--output', 'outputs/geometric-act-capture-episodes-v25',
            '--episodes', 20, '--seed-start', 4301, '--record-video')
        run('collect_opening_episodes.py', '--pilot', 'outputs/opening-rl-capture-pickup-v6',
            '--ids', 'initial_e2e4', 'initial_d2d4', 'open_game_knight',
            '--output', 'outputs/geometric-act-ordinary-episodes-v25', '--episodes', 20,
            '--seed-start', 4401, '--record-video')
        for kind in ['capture','ordinary']:
            subprocess.run(['bash','run_dataset_python.sh','export_act_episodes.py',
                '--index',f'outputs/geometric-act-{kind}-episodes-v25/accepted_episodes.json',
                '--output',f'outputs/geometric-act-{kind}-dataset-v25','--repo-id',f'local/chess-sim-{kind}-v25'], check=True)
        status('complete', act_retrained=False, uploaded=False,
               readiness=str(args.output/'policy_readiness.json'))
    except BaseException as error:
        status('needs_analysis', error=repr(error), act_retrained=False, uploaded=False)
        (args.output/'error.txt').write_text(traceback.format_exc())
        raise


if __name__ == '__main__':
    main()
