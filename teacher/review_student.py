"""Camera-only ACT rollout with privileged physics used solely for scoring."""
import argparse
import html
import json
from pathlib import Path
import time
import numpy as np
import torch
import chess
from camera_client import PairedCameraClient
from evaluation_video import EpisodeVideo
from policy_overlay import highlight_move
from teacher.act_student import GoalACT, limited_joint_target
from teacher.episode_data import digest
from teacher.goal_image import render_goal
from teacher.goal_spec import save_goal
from teacher.move_teacher import MoveEnv
from teacher.review_policy import make_env, matched_images, motion_metrics
from teacher.verify_move import audit_trial
from teacher.robot_clearance import append_clearance_audit
from teacher.grasp_checkpoints import policy_success


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--teacher-run', type=Path, required=True, help='Immutable task/reset contract, not runtime proposals')
    p.add_argument('--output', type=Path, required=True); p.add_argument('--url', default='http://127.0.0.1:8082')
    p.add_argument('--split', choices=['validation','test'], default='validation')
    p.add_argument('--device', choices=['cpu','cuda'], default='cuda'); p.add_argument('--n-action-steps', type=int, default=50)
    p.add_argument('--all-seeds', action='store_true')
    args = p.parse_args(); args.output.mkdir(parents=True, exist_ok=False); env = None
    try:
        student = GoalACT(args.checkpoint, args.device, args.n_action_steps)
        source = json.loads((args.teacher_run/'manifest.json').read_text())
        env, _ = make_env(source, args.url, args.split)
        entries = [entry for entry in env.entries if entry['family']=='home']
        if not entries: raise ValueError('RGB student review requires full home-start trials')
        if not args.all_seeds:
            first = {}
            for entry in entries: first.setdefault(entry.get('case', source['case'])['id'], entry)
            entries = list(first.values())
        manifest = {**source, 'mode': 'RGB_ACT_student_review', 'rgb_student': True,
            'runtime_teacher': False, 'runtime_oracle': False, 'training_enabled': False,
            'policy_action_space': None, 'runtime_action_space': 'absolute legacy ACT targets with uniform joint slew',
            'checkpoint': str(args.checkpoint.resolve()), 'checkpoint_sha256': digest(args.checkpoint/'model.safetensors'),
            'student_inputs': ['six measured legacy joint values','fixed RGB with requested move highlight','wrist RGB','desired-board RGB'],
            'privileged_state_usage': 'environment completion/reward audit only; never passed to the ACT model',
            'n_action_steps': args.n_action_steps, 'ready_for_robot': False}
        (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2))
        calibration = json.loads(Path('calibration/sim_board.json').read_text())['cameras']['red']
        camera = PairedCameraClient(args.url, timeout=30); results = []
        for entry in entries:
            directory = args.output/entry['name']; directory.mkdir(); env.fixed_entry = entry
            _, initial = env.reset(seed=entry['seed']); student.reset()
            (directory/'initial.json').write_text(json.dumps(initial, indent=2))
            case = entry.get('case', source['case']); board = chess.Board()
            for move in case['history']: board.push_uci(move)
            save_goal(directory, board.fen(), case['move']); goal = render_goal(directory/'desired_board.svg')
            def pair():
                rgb = matched_images(camera, env.base.latest)
                fixed = highlight_move(np.ascontiguousarray(rgb['red'][...,::-1]), calibration, case['move'])
                return {'laptop_rgb': np.ascontiguousarray(fixed[...,::-1]), 'phone_rgb': rgb['blue']}
            images = pair(); video = EpisodeVideo(directory, images); video.write(images); rows = []
            try:
                with (directory/'steps.jsonl').open('x') as stream:
                    while True:
                        state = env.base.actual['observation_state']
                        started = time.perf_counter()
                        desired = student.action(state, images['laptop_rgb'], images['phone_rgb'], goal)
                        elapsed = time.perf_counter()-started
                        target = limited_joint_target(desired, env.base.command, env.base.mapping,
                            env.base.low, env.base.high, env.base.config.max_velocity_deg_s)
                        action = env.base.normalized_target(target)
                        # The ACT predicts joints directly. Bypass Cartesian IK,
                        # retaining the exact same independent completion scorer.
                        _, reward, done, truncated, info = MoveEnv.step(env, action)
                        info['teacher'].update(privileged_actor=False, rgb_student=True)
                        row = {'step':len(rows)+1,'action':action.tolist(),'reward':reward,'terminated':done,'truncated':truncated,
                            'score':info['score'],'teacher':info['teacher'],'reward_components':info['reward_components'],
                            'observation':info['telemetry']['observation'],'student':{'desired_legacy':desired.tolist(),
                                'inference_seconds':elapsed,'input_sequence':info['telemetry']['sequence']-1}}
                        stream.write(json.dumps(row)+'\n'); rows.append(row); images = pair(); video.write(images)
                        if done or truncated: break
            finally: video.close()
            result = {'reset_name':entry['name'],'reset_family':entry['family'],'seed':entry['seed'],'case':case,
                'steps':len(rows),'score':rows[-1]['score'],'video':video.verify(),
                'motion':motion_metrics(rows, initial['telemetry']['observation']), 'accepted_for_act_training':False}
            (directory/'result.json').write_text(json.dumps(result, indent=2))
            checked=append_clearance_audit(directory,manifest,audit_trial(directory,manifest))
            for key in ('robot_collision_audit','camera_clearance_audit'):
                if key in checked:result[key]=checked[key]
            (directory/'result.json').write_text(json.dumps(result,indent=2))
            results.append(result)
            (args.output/'results.json').write_text(json.dumps({'trials':len(results),
                'successes':sum(policy_success(r) for r in results),
                'physical_goal_completions':sum(r['score']['success'] for r in results),
                'runtime_teacher':False,'ready_for_robot':False,'results':results},indent=2))
            print('RGB_STUDENT_TRIAL', json.dumps(result), flush=True)
        if digest(args.checkpoint/'model.safetensors') != manifest['checkpoint_sha256']:
            raise ValueError('Frozen RGB student changed during evaluation')
        body = ['<!doctype html><meta charset="utf-8"><title>RGB ACT student review</title>',
            '<h1>RGB ACT student</h1><p>RGB and joints only. No state teacher actions or analytic interventions.</p>']
        for result in results:
            body.append(f'<h2>{html.escape(result["case"]["move"])}: {html.escape(result["score"]["reason"])}</h2>'
                f'<video controls preload="metadata" width="1200" src="{html.escape(result["reset_name"])}/cameras.mp4"></video>')
        (args.output/'videos.html').write_text('\n'.join(body))
    finally:
        if env: env.close()


if __name__ == '__main__': main()
