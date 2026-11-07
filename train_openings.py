"""Bounded multi-move residual SAC pilot over frozen ACT; no calibration fitting."""
import argparse
from dataclasses import asdict, replace
from functools import partial
import hashlib
import json
from pathlib import Path
import time

from stable_baselines3 import SAC
from gymnasium import spaces
import numpy as np
import torch

from act_client import LegacyACT
from grasp_geometry import GRASP_REFERENCE
from chess_env import ChessEnv
from residual_learning import LearningResidualEnv
from train_residual import evaluate, make_learner
from opening_checkpoints import OpeningCheckpoints


def move_features(case):
    move = case['move']
    coordinates = [(ord(move[0]) - ord('a')) / 7, (int(move[1]) - 1) / 7,
                   (ord(move[2]) - ord('a')) / 7, (int(move[3]) - 1) / 7]
    return np.array([*coordinates, float(case['capture'])], np.float32)


class OpeningChessEnv(ChessEnv):
    def __init__(self, cases, pickup_only=False, capture_home_retry=False, capture_step_limit=800):
        if type(capture_step_limit) is not int or capture_step_limit < 800:
            raise ValueError('Capture step budget must be an integer >=800')
        self.capture_step_limit = capture_step_limit
        self.pickup_only = pickup_only
        super().__init__(pawn_jitter_mm=.5)
        self.task_config = replace(self.task_config, capture_home_retry=capture_home_retry)
        self.cases, self.fixed_case, self.current_case = cases, None, cases[0]
        self.task_cursor = 0

    def reset(self, *, seed=None, options=None):
        if options:
            raise ValueError('Use fixed_case to select an evaluation task')
        if self.fixed_case is None:
            if seed is not None:
                self.task_cursor = 0
            self.current_case = self.cases[self.task_cursor % len(self.cases)]
            self.task_cursor += 1
        else:
            self.current_case = self.fixed_case
        case = self.current_case
        self.task_config = replace(self.task_config, max_steps=getattr(self,'capture_step_limit',800) if case['capture'] else 400)
        result = super().reset(seed=seed, options={'history': case['history'], 'move': case['move']})
        reference = result[1]['telemetry']['observation'].get('grasp_reference', {})
        if reference != GRASP_REFERENCE or 'piece_grasp_targets_m' not in result[1]['telemetry']['observation']['pieces']:
            raise RuntimeError('Restart the simulator with the current grasp reference and piece targets')
        if getattr(self, 'pickup_only', False):
            from capture_pickup import CapturePickupTask
            self.task = CapturePickupTask(result[1]['telemetry']['observation'], self.task_config)
            result[1]['task_version'] = self.task.version
        return result


class OpeningLearningEnv(LearningResidualEnv):
    def __init__(self, env, policy, capture_alignment=False, **kwargs):
        self.capture_alignment = capture_alignment
        super().__init__(env, policy, **kwargs)
        self.observation_space = spaces.Box(-np.inf, np.inf, (self.encoder.output_dim + 26,), np.float32)

    def _potential(self, info):
        if getattr(self.unwrapped, 'pickup_only', False):
            from capture_pickup import pickup_potential
            potential = pickup_potential(self.gate, info)
            if self.capture_alignment:
                from capture_alignment import capture_alignment
                potential += self.gate * capture_alignment(info)[0]
            return potential
        potential = super()._potential(info)
        if self.capture_alignment:
            from capture_alignment import capture_alignment
            potential += self.gate * capture_alignment(info)[0]
        return potential

    def step(self, action):
        previous = self.unwrapped.latest['observation']
        obs, reward, terminated, truncated, info = super().step(action)
        if self.capture_alignment:
            from capture_alignment import closing_alignment_penalty
            penalty, closing, quality = closing_alignment_penalty(previous, info, info['learning']['correction_gate'])
            reward += penalty
            info['learning'].update(capture_alignment_quality=quality,
                                    closing_alignment_penalty=penalty, commanded_closing_deg=closing)
        return obs, reward, terminated, truncated, info

    def _features(self, obs):
        # Move command is provided by the game coordinator, not simulator truth.
        return np.r_[super()._features(obs), move_features(self.unwrapped.current_case)].astype(np.float32)


def evaluate_cases(env, model, cases, seeds, output, label, record_video=False, collect_provenance=None):
    results = []
    for case in cases:
        env.unwrapped.fixed_case = case
        path = output / case['id']
        path.mkdir(exist_ok=True)
        rows = evaluate(env, model, seeds, path, label, record_video=record_video, collect_provenance=collect_provenance)
        for row in rows:
            row['task_id'] = case['id']
        results.extend(rows)
        (output / f'{label}_results.json').write_text(json.dumps(results, indent=2))
        if record_video:
            from evaluation_video import write_training_video_index
            write_training_video_index(output)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--curriculum', type=Path, default=Path('outputs/openings/curriculum.json'))
    parser.add_argument('--initialize-from', type=Path, help='Warm-start actor; fresh critics and replay for the revised reward')
    parser.add_argument('--eval-every', type=int, default=0)
    parser.add_argument('--pickup-only', action='store_true', help='Diagnostic objective; these checkpoints are not full-move policies')
    parser.add_argument('--capture-home-retry', action='store_true', help='Allow intermediate home/retry poses before the attacker is grasped, within the full capture budget')
    parser.add_argument('--capture-alignment', action='store_true', help='Wider smoothed corrections and centred-closing reward')
    parser.add_argument('--demonstrations', type=Path, help='Verified residual demonstrations; maintains an actor loss during SAC')
    parser.add_argument('--bc-weight', type=float, default=.5)
    parser.add_argument('--replay-seeds', type=int, nargs='+', help='Separate teacher-training seeds: exact successful transitions seed SAC replay and retention supervision')
    parser.add_argument('--prefill-demonstration-replay', action='store_true', help='Seed replay from independently verified complete demonstration transitions; excludes validation')
    parser.add_argument('--exploration-log-std', type=float, help='Set SDE exploration variance before training; preserves the deterministic actor mean')
    parser.add_argument('--actor-lr-scale', type=float, default=1., help='Actor learning-rate multiplier; critics retain their base learning rate')
    parser.add_argument('--critic-warmup-updates', type=int, default=0, help='Fit critics before permitting actor optimizer steps')
    parser.add_argument('--retention-loss-ratio', type=float, help='Reject actor updates exceeding this multiple of initial training-demonstration loss (minimum budget .0025)')
    parser.add_argument('--capture-transport', action='store_true', help='Explicit command/calibration/encoder-only victim transport supervisor')
    parser.add_argument('--capture-step-limit', type=int, default=800)
    parser.add_argument('--live-timing', action='store_true', help='Opt-in cached ACT close-before-lift controller; allows full-move alignment training')
    parser.add_argument('--adaptive-alignment', action='store_true', help='Allow residual arm correction during an open-jaw pause before locking the grasp pose')
    parser.add_argument('--alignment-steps', type=int, help='Open-jaw adaptation frames; default 18')
    parser.add_argument('--arm-settle-limit', type=int, help='Maximum adaptation/settling frames; default 60 with adaptation, 30 otherwise')
    parser.add_argument('--latch-alignment', action='store_true', help='Lock the corrected approach after adaptation so exploration cannot prevent encoder settling')
    parser.add_argument('--board-grasps-only', action='store_true', help='Restrict timing guards to the static calibrated square footprint; leave tabletop release motion alone')
    parser.add_argument('--release-aware', action='store_true', help='Do not guard on-board placement followed by empty-jaw closure; uses encoder/FK history')
    parser.add_argument('--grasp-interventions', type=int, help='Maximum guarded pickups including retries; default one for diagnostics, two for complete captures')
    parser.add_argument('--kinematics', type=Path, default=Path('outputs/capture-supervised-kinematics-v10.json'))
    parser.add_argument('--collect-evaluations', action='store_true', help='Save lossless before-action camera/state/executed-action bundles during deterministic evaluation')
    parser.add_argument('--record-evaluations', action='store_true')
    parser.add_argument('--capture-repeats', type=int, default=1, help='Training occurrences of each capture per cycle; evaluation stays balanced')
    parser.add_argument('--steps', type=int, default=6000)
    parser.add_argument('--seed', type=int, default=501)
    parser.add_argument('--eval-seeds', type=int, nargs='+', default=[601, 602])
    parser.add_argument('--ids', nargs='+', default=['initial_e2e4', 'initial_d2d4', 'open_game_knight', 'scandinavian_capture'])
    args = parser.parse_args()
    if args.capture_repeats < 1:
        parser.error('Capture repeats must be positive')
    if args.eval_every < 0:
        parser.error("Evaluation interval must be nonnegative")
    if args.steps < 257 or args.seed < 0 or any(s < 0 or s == args.seed for s in args.eval_seeds) or len(set(args.eval_seeds)) != len(args.eval_seeds):
        parser.error('Need >=257 steps and unique held-out nonnegative evaluation seeds')
    tasks = {t['id']: t for t in json.loads(args.curriculum.read_text())['tasks']}
    if len(set(args.ids)) != len(args.ids) or any(name not in tasks or not tasks[name]['executable'] for name in args.ids):
        parser.error('Task IDs must be unique executable blue tasks')
    cases = [tasks[name] for name in args.ids]
    if args.pickup_only and any(not c['capture'] for c in cases):
        parser.error('Pickup-only requires capture task IDs')
    if args.capture_home_retry and args.pickup_only:
        parser.error('Intermediate capture home poses are a complete-move setting')
    if args.capture_step_limit < 800 or args.capture_transport and (len(cases) != 1 or not cases[0]['capture'] or args.pickup_only or not args.release_aware):
        parser.error('Transport needs one complete capture case, release-aware timing, and >=800 capture steps')
    if (not np.isfinite(args.actor_lr_scale) or not 0 < args.actor_lr_scale <= 1 or args.critic_warmup_updates < 0 or
            args.retention_loss_ratio is not None and (not np.isfinite(args.retention_loss_ratio) or args.retention_loss_ratio < 1)):
        parser.error('Invalid actor stability settings')
    stable_actor = args.actor_lr_scale != 1 or args.critic_warmup_updates or args.retention_loss_ratio is not None
    if stable_actor and not (args.demonstrations or args.replay_seeds):
        parser.error('Actor stability requires training-only demonstration anchors')
    if args.capture_alignment and not (args.pickup_only or args.live_timing):
        parser.error('Capture alignment requires pickup-only or explicit live timing for complete moves')
    if args.live_timing and not args.capture_alignment:
        parser.error('Live timing requires the matched wider bounds/filter from capture alignment')
    if args.adaptive_alignment and not args.live_timing:
        parser.error('Adaptive alignment requires the live timing controller')
    if args.latch_alignment and not args.adaptive_alignment:
        parser.error('Latched alignment requires adaptive alignment')
    if args.board_grasps_only and not args.live_timing:
        parser.error('Board grasp restriction requires live timing')
    if args.release_aware and not (args.board_grasps_only and args.live_timing):
        parser.error('Release discrimination requires live timing and calibrated board bounds')
    if args.release_aware and len({bool(case['capture']) for case in cases}) != 1:
        parser.error('Use a separate controller profile for capture versus ordinary moves')
    if args.alignment_steps is not None and (not args.adaptive_alignment or args.alignment_steps < 1):
        parser.error('Positive alignment frames require adaptive alignment')
    adaptation_steps = (args.alignment_steps or 18) if args.adaptive_alignment else 0
    settle_limit = args.arm_settle_limit or (60 if args.adaptive_alignment else 30)
    if args.arm_settle_limit is not None and (not args.live_timing or settle_limit <= adaptation_steps or args.arm_settle_limit < 1):
        parser.error('Live arm settling limit must exceed adaptation frames')
    if args.prefill_demonstration_replay and (not args.demonstrations or args.replay_seeds or args.pickup_only or not args.live_timing):
        parser.error('Demonstration replay requires full-move demonstrations and live timing, without replay-seeds')
    if args.grasp_interventions is not None and (not args.live_timing or not 1 <= args.grasp_interventions <= 8):
        parser.error('Need live timing and between one and eight grasp interventions')
    if args.exploration_log_std is not None and (not np.isfinite(args.exploration_log_std) or args.exploration_log_std >= 0):
        parser.error('Exploration setting requires a finite negative log standard deviation')
    if args.replay_seeds:
        if (not args.capture_alignment or not args.live_timing or args.demonstrations or
                not np.isfinite(args.bc_weight) or args.bc_weight <= 0 or
                len(set(args.replay_seeds)) != len(args.replay_seeds) or len(args.replay_seeds) < 2 or
                any(seed < 0 for seed in args.replay_seeds) or
                set(args.replay_seeds) & {args.seed, *args.eval_seeds}):
            parser.error('Teacher replay requires matched live timing, separate unique seeds, and no other demonstration dataset')
    demo_data, demo_manifest = None, None
    if args.demonstrations:
        if not args.capture_alignment or not args.initialize_from or not np.isfinite(args.bc_weight) or args.bc_weight <= 0:
            parser.error('Demonstrations require a matched capture-alignment warm-start and a positive BC weight')
        from capture_demonstrations import load_demonstrations
        demo_data, demo_manifest = load_demonstrations(args.demonstrations)
        if {args.seed, *args.eval_seeds} & set(demo_manifest['source_seeds']):
            parser.error('Training/selection seeds must differ from demonstration sources')
    source_manifest = None
    if args.initialize_from and args.capture_alignment:
        from capture_demonstrations import validate_semantics
        source_manifest = json.loads((args.initialize_from.parent / 'manifest.json').read_text())
        validate_semantics(source_manifest, ('capture_pickup_v1', 'legal_move_v5') if args.live_timing else ('capture_pickup_v1',))
    assert torch.ones(16, device='cuda').sum().item() == 16
    policy = LegacyACT('/home/jacob/lerobot/outputs/train/chess_game_multi_stereo/checkpoints/last/pretrained_model', 'cuda', n_action_steps=100)
    for source in (source_manifest, demo_manifest):
        if source and (source['weights_sha256'] != policy.weights_sha256 or source['grasp_reference'] != GRASP_REFERENCE):
            raise ValueError('Supervised/source actor uses different ACT weights or grasp semantics')
    training_cases = [case for case in cases for _ in range(args.capture_repeats if case['capture'] else 1)]
    settings = {'bounds_deg':(6,6,6,6,6,3), 'residual_alpha':.25,
                'max_residual_change_deg':(.4,.4,.4,.4,.4,.6)} if args.capture_alignment else {}
    base = OpeningChessEnv(training_cases, pickup_only=args.pickup_only, capture_home_retry=args.capture_home_retry,
                           capture_step_limit=args.capture_step_limit)
    transport_settings = None
    timing_settings = None
    if args.live_timing:
        from capture_demonstrations import JointKinematics, file_hash
        from live_act_timing import CloseBeforeLiftACT, TimingConfig, board_pickup_bounds
        specification = json.loads(args.kinematics.read_text())
        if specification['usd_sha256'] != file_hash('Sim_Stage.usd'):
            raise ValueError('USD changed since robot kinematics export')
        config = TimingConfig(maximum_interventions=args.grasp_interventions if args.grasp_interventions is not None else (1 if args.pickup_only else 2),
                              guard_all_closures=args.adaptive_alignment,
                              minimum_arm_adaptation_steps=adaptation_steps,
                              latch_after_adaptation=args.latch_alignment,
                              board_pickup_bounds_xy_m=board_pickup_bounds(base.latest['observation']['physics_settings']['contacts']['square_centers_m']) if args.board_grasps_only else None,
                              maximum_arm_settle_steps=settle_limit,
                              suppress_board_release_closures=args.release_aware,
                              required_tabletop_releases=int(args.release_aware and bool(cases[0]['capture'])),
                              confirm_settled_contact=args.release_aware,
                              alignment_open_jaw_deg=10. if args.adaptive_alignment else None)
        policy = CloseBeforeLiftACT(policy, base.latest['observation']['joint_map'], JointKinematics(specification), config)
        timing_settings = {'config': config.to_dict(), 'kinematics_path': str(args.kinematics.resolve()),
                           'kinematics_sha256': file_hash(args.kinematics),
                           'control_inputs': 'cached ACT commands, robot FK, measured arm and finger joints; no piece poses',
                           'role': 'separate timing controller; the learned residual checkpoint requires this deployment wrapper'}
        if args.capture_transport:
            from capture_transport import CaptureTransportACT, settings_from_command
            transport_config = settings_from_command(base, cases[0])
            policy = CaptureTransportACT(policy, transport_config)
            transport_settings = {'config':transport_config.to_dict(), 'code_sha256':file_hash('capture_transport.py'),
                                  'role':'explicit command/calibration/encoder supervisor; not learned transport'}
    env = OpeningLearningEnv(base, policy,
                             capture_alignment=args.capture_alignment, **settings)
    args.output.mkdir(parents=True, exist_ok=False)
    if args.record_evaluations:
        from evaluation_video import write_training_video_index
        write_training_video_index(args.output)
    if demo_data is not None or args.replay_seeds:
        from capture_demonstrations import DemonstrationSAC
        model = make_learner(env, 'cuda', args.seed, learner_class=DemonstrationSAC)
        if demo_data is not None:
            model.configure_demonstrations(demo_data, args.bc_weight, args.seed)
    else:
        model = make_learner(env, 'cuda', args.seed)
    source_hash = None
    if args.initialize_from:
        teacher = SAC.load(args.initialize_from, device='cpu')
        if teacher.observation_space != model.observation_space or teacher.action_space != model.action_space:
            raise ValueError('Checkpoint spaces do not match opening learner')
        model.actor.load_state_dict(teacher.actor.state_dict())
        source_hash = hashlib.sha256(args.initialize_from.read_bytes()).hexdigest()
        del teacher
    if args.exploration_log_std is not None:
        with torch.no_grad():
            model.actor.log_std.fill_(args.exploration_log_std)
    replay_report = None
    if args.replay_seeds:
        from residual_replay import collect_teacher_replay
        from capture_demonstrations import file_hash
        frozen_teacher = args.output / 'replay_teacher.zip'
        model.save(frozen_teacher)
        demo_data, replay_report = collect_teacher_replay(env, model, cases, args.replay_seeds,
                                                        args.output / 'teacher_replay', args.record_evaluations)
        replay_report.update(teacher_checkpoint=str(frozen_teacher.resolve()),
                             teacher_sha256=file_hash(frozen_teacher))
        (args.output / 'teacher_replay' / 'replay.json').write_text(json.dumps(replay_report, indent=2))
        args.demonstrations = args.output / 'teacher_replay' / 'demonstrations.npz'
        demo_manifest = {'capture_alignment': args.capture_alignment,
                         'task_version': 'capture_pickup_v1' if args.pickup_only else 'legal_move_v5',
                         'objective': 'pickup-only diagnostic; not full capture success' if args.pickup_only else 'complete move',
                         'observation_size': env.observation_space.shape[0], 'action_window': 100,
                         'residual_bounds_deg': env.env.bounds.tolist(),
                         'residual_filter': {'alpha': env.residual_alpha,
                                             'max_change_deg': env.max_residual_change_deg.tolist() if env.max_residual_change_deg is not None else None},
                         'weights_sha256': policy.weights_sha256, 'grasp_reference': GRASP_REFERENCE,
                         'live_timing': timing_settings, 'capture_home_retry': args.capture_home_retry,
                         'capture_transport': transport_settings, 'capture_step_limit': args.capture_step_limit,
                         'source_semantics': source_manifest,
                         'dataset_sha256': file_hash(args.demonstrations),
                         'source_seeds': args.replay_seeds, 'frames': len(demo_data['actions']),
                         'episodes': replay_report, 'purpose': 'exact live teacher retention and matched-objective successful replay'}
        args.demonstrations.with_suffix('.json').write_text(json.dumps(demo_manifest, indent=2))
        model.configure_demonstrations(demo_data, args.bc_weight, args.seed)
    if stable_actor:
        model.configure_stability(args.actor_lr_scale, args.critic_warmup_updates, args.retention_loss_ratio)
    manifest = {'cases': cases, 'training_seed': args.seed, 'evaluation_seeds': args.eval_seeds,
                'steps': args.steps, 'evaluation_interval': args.eval_every,
                'evaluation_schedule': 'first completed training episode at or after each interval',
                'evaluation_use': 'checkpoint selection; not independent held-out testing', 'checkpoint': policy.checkpoint, 'weights_sha256': policy.weights_sha256,
                'task_version': 'capture_pickup_v1' if args.pickup_only else 'legal_move_v5',
                'reward': ('capture_pickup_v1 + pickup_potential_v1 + centred_grasp_potential_and_closing_cost_v1' if args.capture_alignment and args.pickup_only else
                           'legal_move_v5 + approach_plus_placement_potential_v4 + centred_victim_grasp_and_closing_cost_v1' if args.capture_alignment else
                           'capture_pickup_v1 + pickup_potential_v1' if args.pickup_only else
                           'legal_move_v5 + residual penalty + approach_plus_placement_potential_v4'),
                'objective': 'pickup-only diagnostic; not full capture success' if args.pickup_only else 'complete move',
                'pickup_limit_steps': 240 if args.pickup_only else None,
                'capture_home_retry': args.capture_home_retry,
                'capture_step_limit': args.capture_step_limit,
                'capture_transport': transport_settings,
                'actor_time_reference': '800-step actor budget excluding supervised transport; residual history paused' if transport_settings else None,
                'actor_stability': {'actor_lr_scale':args.actor_lr_scale, 'critic_warmup_updates':args.critic_warmup_updates,
                    'retention_loss_ratio':args.retention_loss_ratio,
                    'training_loss_budget':getattr(model,'retention_loss_budget',None),
                    'initial_training_loss':getattr(model,'initial_retention_loss',None)} if stable_actor else None,
                'joint_map': base.latest['observation']['joint_map'],
                'record_evaluations': args.record_evaluations,
                'collect_evaluations': args.collect_evaluations,
                'grasp_reference': GRASP_REFERENCE,
                'placement_potential': {'weight_inside_potential': 2., 'xy_scale_m': .0225, 'z_scale_m': .020},
                'gamma': .99, 'potential_weight': 2., 'residual_bounds_deg': env.env.bounds.tolist(),
                'residual_filter': {'alpha':env.residual_alpha,
                                    'max_change_deg':env.max_residual_change_deg.tolist() if env.max_residual_change_deg is not None else None},
                'capture_alignment': args.capture_alignment,
                'live_timing': timing_settings,
                'capture_alignment_settings': {'scales_m':[.008,.008,.005], 'potential_weight_inside':1.,
                                               'misaligned_full_closure_cost':2., 'closing_span_deg':12.5,
                                               'cost_active_within_m':.04, 'cost_only_before_victim_lift':True} if args.capture_alignment else None,
                'actor_features': 'frozen ACT vision + proprioception + from/to coordinates + capture flag; no privileged capture phase',
                'observation_size': env.observation_space.shape[0], 'action_window': 100,
                'calibration_randomization': False, 'pawn_jitter_mm': .5,
                'schedule': 'round robin training occurrences; balanced evaluation; fresh critics, optimizers and replay',
                'training_task_cycle': [case['id'] for case in training_cases],
                'capture_repeats': args.capture_repeats,
                'initialize_from': str(args.initialize_from.resolve()) if args.initialize_from else None,
                'actor_initialization': 'source actor' if args.initialize_from else 'fresh actor with exactly zero deterministic residual',
                'source_sha256': source_hash,
                'exploration_log_std_reset': args.exploration_log_std,
                'teacher_replay': replay_report,
                'source_seeds': args.replay_seeds or [],
                'demonstrations': {'path': str(args.demonstrations.resolve()), 'sha256': demo_manifest['dataset_sha256'],
                                   'source_seeds': demo_manifest['source_seeds'], 'bc_weight': args.bc_weight,
                                   'actor_loss': 'weighted deterministic residual MSE; one extra actor update per SAC update',
                                   'privileged_actor_inputs': False} if demo_manifest else None,
                'evaluation_labels': ['baseline', 'before', 'learned'] if args.initialize_from or args.capture_alignment else ['baseline', 'learned'],
                'torch': torch.__version__, 'cuda': torch.version.cuda}
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    if args.prefill_demonstration_replay:
        from geometric_replay import prefill
        replay_report = prefill(args.demonstrations, manifest, model)
        manifest['teacher_replay'] = replay_report
        manifest['source_seeds'] = demo_manifest['source_seeds']
        (args.output/'demonstration_replay.json').write_text(json.dumps(replay_report, indent=2))
        (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2))
    provenance = {'act_sha256': policy.weights_sha256, 'pilot_output': str(args.output.resolve()),
                  'calibration': base.latest['observation']['joint_map'], 'grasp_reference': GRASP_REFERENCE,
                  'residual_bounds_deg': env.env.bounds.tolist(), 'residual_filter': manifest['residual_filter'],
                  'live_timing': timing_settings, 'task_version': manifest['task_version'],
                  'policy_mode': 'deterministic evaluation; exploration disabled'} if args.collect_evaluations else None
    evaluate_fn = partial(evaluate_cases, record_video=args.record_evaluations, collect_provenance=provenance)
    callback = OpeningCheckpoints(args.output, env, cases, args.eval_seeds, args.eval_every, args.steps, evaluate_fn)
    try:
        baseline = evaluate_fn(env, None, cases, args.eval_seeds, args.output, 'baseline')
        before = evaluate_fn(env, model, cases, args.eval_seeds, args.output, 'before') if args.initialize_from or args.capture_alignment else None
        if before is not None:
            callback.consider(model, 'before', before, 0)
        env.unwrapped.fixed_case = None
        env.unwrapped.task_cursor = 0
        model.save(args.output / 'residual_initial')
        started = time.perf_counter()
        model.learn(total_timesteps=args.steps, callback=callback)
        elapsed = time.perf_counter() - started - callback.evaluation_seconds
        model.save(args.output / 'residual_final')
        learned = evaluate_fn(env, model, cases, args.eval_seeds, args.output, 'learned')
        callback.consider(model, 'learned', learned, model.num_timesteps)
        manifest['evaluation_labels'] += list(callback.results)
        (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2))
        with (Path(policy.checkpoint) / 'model.safetensors').open('rb') as stream:
            assert hashlib.file_digest(stream, 'sha256').hexdigest() == policy.weights_sha256
        results = {case['id']: {label: {'successes': sum(r['score']['success'] for r in rows if r['task_id'] == case['id']),
                                       'episodes': len(args.eval_seeds)}
                               for label, rows in ([('baseline', baseline)] + ([('before', before)] if before is not None else []) + [('learned', learned)] + list(callback.results.items()))} for case in cases}
        summary = {'per_task': results, 'training_steps': model.num_timesteps, 'gradient_updates': model._n_updates,
                   'training_seconds': elapsed, 'intermediate_evaluation_seconds': callback.evaluation_seconds,
                   'best_checkpoint_label': callback.best_label, 'act_checkpoint_unchanged': True,
                   'demonstration_updates': getattr(model, 'demo_updates', 0),
                   'initial_successful_replay_transitions': replay_report['replay_transitions'] if replay_report else 0,
                   'actor_update_rejections':getattr(model,'actor_update_rejections',0),
                   'note': f'Small pilot on {len(cases)} positions; not a generalization or sim-to-real reliability estimate'}
        (args.output / 'summary.json').write_text(json.dumps(summary, indent=2))
        print('OPENING_RL_RESULT', json.dumps(summary), flush=True)
    except Exception as error:
        (args.output / 'error.json').write_text(json.dumps({'error': str(error)}, indent=2))
        raise
    finally:
        if callback.stream and not callback.stream.closed:
            callback.stream.close()
        env.close()


if __name__ == '__main__':
    main()
