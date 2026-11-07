"""Collect live ACT rollouts corrected by a privileged, bounded grasp teacher."""
import argparse
from dataclasses import asdict
import json
import shutil
from pathlib import Path
import numpy as np
import torch
from act_client import LegacyACT
from capture_demonstrations import BOUNDS, FILTER, JointKinematics, file_hash
from evaluate_openings import audit_episode
from evaluation_video import write_training_video_index
from geometric_grasp_teacher import GeometricGraspTeacher
from grasp_geometry import GRASP_REFERENCE
from live_act_timing import CloseBeforeLiftACT, TimingConfig, board_pickup_bounds
from train_openings import OpeningChessEnv, OpeningLearningEnv
from train_residual import evaluate, make_learner
from verify_policy_reliability import recompute


class RecordedTeacher(GeometricGraspTeacher):
    def predict(self, observation, deterministic=True):
        action, state = super().predict(observation, deterministic)
        self.stream.write(json.dumps({'frame_id': self.env.unwrapped.latest['frame_id'],
                                      **self.last_decision})+'\n')
        return action, state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seeds', type=int, nargs='+', required=True)
    parser.add_argument('--pickup-only', action='store_true')
    parser.add_argument('--release-aware', action='store_true', help='Use encoder/FK history to distinguish board release from pickup')
    parser.add_argument('--ids', nargs='+', default=['scandinavian_capture'])
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds) or any(s < 0 for s in args.seeds):
        parser.error('Seeds must be unique and nonnegative')
    used = set()
    for path in Path('outputs').glob('*/manifest.json'):
        prior = json.loads(path.read_text())
        for key in ('evaluation_seeds', 'source_seeds', 'seeds', 'collection_seeds'):
            used.update(prior.get(key, []))
        if prior.get('training_seed') is not None:
            used.add(prior['training_seed'])
    if used & set(args.seeds):
        parser.error('Teacher seeds overlap a previous experiment')
    cases_by_id = {case['id']: case for case in json.loads(Path('outputs/openings/curriculum.json').read_text())['tasks']}
    if any(name not in cases_by_id or not cases_by_id[name]['executable'] for name in args.ids):
        parser.error('Select executable blue tasks')
    cases = [cases_by_id[name] for name in args.ids]
    if args.release_aware and len({bool(case['capture']) for case in cases}) != 1:
        parser.error('Release-aware collection needs one command type per controller profile')
    if args.pickup_only and any(not case['capture'] for case in cases):
        parser.error('Pickup diagnostic requires captures')
    kin_path = Path('outputs/capture-supervised-kinematics-v10.json')
    spec = json.loads(kin_path.read_text())
    reference = json.loads(Path('outputs/grasp-pose-review-v18/reference.json').read_text())
    if spec['usd_sha256'] != file_hash('Sim_Stage.usd') or reference['usd_sha256'] != spec['usd_sha256'] or reference['grasp_reference'] != GRASP_REFERENCE:
        raise ValueError('Approved geometry changed')
    policy = LegacyACT('/home/jacob/lerobot/outputs/train/chess_game_multi_stereo/checkpoints/last/pretrained_model', 'cuda', n_action_steps=100)
    base = OpeningChessEnv(cases, pickup_only=args.pickup_only, capture_home_retry=not args.pickup_only)
    config = TimingConfig(guard_all_closures=True, minimum_arm_adaptation_steps=32,
        latch_after_adaptation=True, board_pickup_bounds_xy_m=board_pickup_bounds(base.latest['observation']['physics_settings']['contacts']['square_centers_m']),
        alignment_open_jaw_deg=10., maximum_arm_settle_steps=90,
        suppress_board_release_closures=args.release_aware,
        required_tabletop_releases=int(args.release_aware and bool(cases[0]['capture'])),
        confirm_settled_contact=args.release_aware,
        maximum_interventions=1 if args.pickup_only else 4)
    timed = CloseBeforeLiftACT(policy, base.latest['observation']['joint_map'], JointKinematics(spec), config)
    env = OpeningLearningEnv(base, timed, capture_alignment=True, bounds_deg=BOUNDS,
                            residual_alpha=FILTER['alpha'], max_residual_change_deg=FILTER['max_change_deg'])
    teacher = RecordedTeacher(env, spec, reference['anchors_jaw_local_m'])
    args.output.mkdir(parents=True, exist_ok=False)
    code = args.output/'code'; code.mkdir()
    for name in ('geometric_grasp_teacher.py', 'show_centered_grasp.py', 'review_grasp_geometry.py', 'live_act_timing.py'):
        shutil.copy2(name, code/name)
    manifest = {'mode': 'privileged_geometric_teacher_collection',
        'objective': 'pickup-only diagnostic; not full capture success' if args.pickup_only else 'complete move',
        'task_version': 'capture_pickup_v1' if args.pickup_only else 'legal_move_v5',
        'reward': 'matched opening reward plus centred grasp shaping and closing cost',
        'cases': cases, 'source_seeds': args.seeds, 'evaluation_seeds': [], 'training_seed': 3650,
        'capture_home_retry': not args.pickup_only, 'pickup_limit_steps': 240 if args.pickup_only else None,
        'checkpoint': policy.checkpoint, 'weights_sha256': policy.weights_sha256,
        'grasp_reference': GRASP_REFERENCE, 'capture_alignment': True, 'observation_size': 4122,
        'residual_bounds_deg': BOUNDS.tolist(), 'residual_filter': FILTER, 'gamma': .99, 'potential_weight': 2.,
        'action_window': 100, 'live_timing': {'config': config.to_dict(), 'kinematics_path': str(kin_path.resolve()),
            'kinematics_sha256': file_hash(kin_path)},
        'teacher': {'privileged': True, 'reference': 'human-approved opposing surface midpoint at current opening',
            'surface_anchors_jaw_local_m': reference['anchors_jaw_local_m'],
            'actor_privileged_inputs': False,
            'code_sha256': {name: file_hash(name) for name in ('geometric_grasp_teacher.py', 'show_centered_grasp.py', 'review_grasp_geometry.py')}},
        'note': 'Teacher corrections use simulator piece targets; no learned-policy reliability claim'}
    source = make_learner(env, 'cuda', 3650)
    with torch.no_grad():source.actor.log_std.fill_(-4.)
    source.save(args.output/'source_model.zip')
    manifest.update(source_model=str((args.output/'source_model.zip').resolve()),
                    source_model_sha256=file_hash(args.output/'source_model.zip'))
    del source
    (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2))
    write_training_video_index(args.output)
    results = []
    try:
        for index, seed in enumerate(args.seeds):
            case = cases[index % len(cases)]
            base.fixed_case = case
            parent = args.output/case['id']; parent.mkdir(exist_ok=True)
            teacher.reset()
            with (parent/f'teacher_{seed}_decisions.jsonl').open('x') as stream:
                teacher.stream = stream
                result = evaluate(env, teacher, [seed], parent, 'teacher', record_video=True, record_learning_transitions=True)[0]
            result['task_id'] = case['id']
            directory = parent/f'teacher_{seed}'
            audit_episode(directory, result, case['move'], manifest)
            if not args.pickup_only:
                recompute(directory)
            results.append(result)
            (args.output/'results.json').write_text(json.dumps(results, indent=2))
            summary = {'attempts': len(results), 'expected_attempts': len(args.seeds),
                'successes': sum(row['score']['success'] for row in results),
                'objective': manifest['objective'], 'privileged_teacher': True,
                'complete_upright_successes': sum(row['score']['success'] and bool((row['score'].get('capture') or {}).get('upright_stable')) for row in results)}
            (args.output/'summary.json').write_text(json.dumps(summary, indent=2))
            write_training_video_index(args.output)
        (args.output/'status.json').write_text(json.dumps({'status': 'complete', 'act_checkpoint_unchanged': file_hash(Path(policy.checkpoint)/'model.safetensors') == policy.weights_sha256}))
        print(json.dumps(summary), flush=True)
    except BaseException as error:
        (args.output/'status.json').write_text(json.dumps({'status': 'failed', 'error': repr(error)}))
        raise
    finally:
        env.close()


if __name__ == '__main__':
    main()
