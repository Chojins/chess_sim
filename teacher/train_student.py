"""Fine-tune a new local RGB ACT copy with balanced sim and real replay."""
import argparse
import json
from pathlib import Path
import random
import time
import numpy as np
import torch
from teacher.act_student import GoalACT
from teacher.episode_data import digest
from teacher.student_data import SimChunks, RealChunks, canonical_joint_names


def real_replay_retained(checked,initial,relative_increase,absolute_slack):
    if relative_increase is None:return True
    return checked['real_l1']<=initial['real_l1']*(1+relative_increase)+absolute_slack


def validation(student, sim, real, seed):
    rng = np.random.default_rng(seed); student.policy.eval(); rows = []
    # Fixed episode/phase samples; no gradients and no VAE posterior at inference.
    with torch.inference_mode():
        for source, data in (('sim', sim), ('real', real)):
            for episode in data.splits['validation']:
                if source == 'sim': length = episode['frames']
                else: length = len(data.cache[episode][0])
                for fraction in (0., .25, .5, .75):
                    sample = data.sample(rng, 'validation', int(fraction*(length-1)), episode)
                    _, metrics = student.loss([sample])
                    rows.append({'source': source, 'episode': sample['episode'], 'frame': sample['frame'],
                        'move': sample.get('move'), **metrics})
    return {'sim_l1': float(np.mean([r['l1'] for r in rows if r['source']=='sim'])),
            'real_l1': float(np.mean([r['l1'] for r in rows if r['source']=='real'])), 'samples': rows}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, default=Path('/home/jacob/lerobot/outputs/train/chess_game_multi_stereo/checkpoints/1280000/pretrained_model'))
    p.add_argument('--index', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--real-data', type=Path, default=Path('/home/jacob/.cache/huggingface/lerobot/Chojins/chess_game_001_blue_stereo'))
    p.add_argument('--updates', type=int, default=1000); p.add_argument('--batch-size', type=int, default=1)
    p.add_argument('--learning-rate', type=float, default=1e-5); p.add_argument('--checkpoint-every', type=int, default=250)
    p.add_argument('--real-episodes', type=int, default=32); p.add_argument('--seed', type=int, default=126001)
    p.add_argument('--real-sampling',choices=['prefix','spread'],default='prefix')
    p.add_argument('--max-real-l1-increase',type=float,help='Relative held-out real replay loss increase allowed for selecting a sim-improved checkpoint')
    p.add_argument('--real-l1-absolute-slack',type=float,default=.01)
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cuda'); p.add_argument('--n-action-steps', type=int, default=50)
    args = p.parse_args()
    if min(args.updates, args.batch_size, args.checkpoint_every, args.real_episodes, args.n_action_steps)<1 or not 0<args.learning_rate<=1e-4:
        p.error('Invalid training settings')
    if (not np.isfinite(args.real_l1_absolute_slack) or args.real_l1_absolute_slack<0 or (args.max_real_l1_increase is not None
            and (not np.isfinite(args.max_real_l1_increase) or args.max_real_l1_increase<0))):p.error('Invalid real replay retention gate')
    args.output.mkdir(parents=True, exist_ok=False)
    status = args.output/'status.json'; manifest_path = args.output/'manifest.json'
    status.write_text(json.dumps({'stage': 'loading_and_auditing_sources'}))
    torch.set_num_threads(4); torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    rng = np.random.default_rng(args.seed); started = time.perf_counter()
    try:
        student = GoalACT(args.checkpoint, args.device, args.n_action_steps)
        source_hash = digest(args.checkpoint/'model.safetensors')
        sim = SimChunks(args.index, student.config['chunk_size'])
        real = RealChunks(args.real_data, student.config['chunk_size'], args.real_episodes,sampling=args.real_sampling)
        # The sim labels and real replay must use exactly the original joint order.
        source_info = canonical_joint_names(real.info['features']['observation.state']['names'])
        accepted = json.loads(args.index.read_text())['episodes']
        if (canonical_joint_names(real.info['features']['action']['names']) != source_info
                or any(canonical_joint_names(e['provenance']['calibration']['legacy_names']) != source_info for e in accepted)):
            raise ValueError('Simulation and real legacy joint order differ')
        manifest = {'format': 'chess_rgb_goal_act_finetune_v1', 'source_checkpoint': str(args.checkpoint.resolve()),
            'source_checkpoint_sha256': source_hash, 'accepted_index': str(args.index.resolve()),
            'accepted_index_sha256': sim.digest, 'real_data': str(args.real_data.resolve()),
            'sim_splits': {k: [e['episode_path'] for e in v] for k,v in sim.splits.items()},
            'real_splits': real.splits,'real_sampling':args.real_sampling,'moves': sorted({e['move'] for e in accepted}),
            'real_retention_gate':{'relative_l1_increase':args.max_real_l1_increase,'absolute_l1_slack':args.real_l1_absolute_slack},
            'inputs': ['measured legacy joints', 'highlighted fixed RGB', 'wrist RGB', 'separate desired-board RGB'],
            'real_replay_views': 2, 'sim_views': 3, 'training_mix': 'one equal-weight sim loss and real loss per AdamW step',
            'action_labels': 'executed absolute legacy targets; full chunks never cross episode boundaries',
            'goal_normalization': 'ImageNet mean/std; live RGB and state/action retain original checkpoint normalization',
            'joint_namespace_alias': 'main_ prefix only; same order and legacy units, no numerical conversion',
            'validation': 'whole episodes held out; fixed phases; deterministic zero-latent inference',
            'runtime_teacher_or_privileged_inputs': False, 'uploaded': False,
            'seed': args.seed, 'updates': args.updates, 'batch_size_per_domain': args.batch_size,
            'learning_rate': args.learning_rate, 'n_action_steps': args.n_action_steps,
            'source_hashes': {str(f): digest(f) for f in (Path('teacher/act_student.py'), Path('teacher/student_data.py'), Path(__file__))}}
        manifest_path.write_text(json.dumps(manifest, indent=2))
        amp = args.device=='cuda' and torch.cuda.is_bf16_supported()
        optimizer = torch.optim.AdamW(student.policy.parameters(), lr=args.learning_rate, weight_decay=1e-4)
        initial = validation(student, sim, real, args.seed+1)
        (args.output/'validation_initial.json').write_text(json.dumps(initial, indent=2))
        student.save_copy(args.output/'initial')
        best = initial['sim_l1']; selected = 'initial'; completed = 0
        with (args.output/'metrics.jsonl').open('x') as stream:
            for update in range(1, args.updates+1):
                student.policy.train(); optimizer.zero_grad(set_to_none=True); metrics = {}
                # Separate microbatches keep the three-view transformer within VRAM.
                for name, data in (('sim', sim), ('real', real)):
                    samples = [data.sample(rng) for _ in range(args.batch_size)]
                    with torch.autocast(device_type=args.device, dtype=torch.bfloat16, enabled=amp):
                        loss, terms = student.loss(samples)
                    if not torch.isfinite(loss): raise ValueError('Non-finite ACT loss')
                    (.5*loss).backward(); metrics[name] = terms
                grad = torch.nn.utils.clip_grad_norm_(student.policy.parameters(), 10., error_if_nonfinite=True)
                optimizer.step(); completed = update
                row = {'update': update, 'elapsed_seconds': time.perf_counter()-started, 'gradient_norm': float(grad), **metrics}
                stream.write(json.dumps(row)+'\n'); stream.flush()
                if update % 10 == 0:
                    progress = {'stage': 'rgb_act_finetuning', 'updates': update, 'requested_updates': args.updates,
                        'elapsed_seconds': row['elapsed_seconds'], 'selected': selected, 'closed_loop_evaluated': False}
                    status.write_text(json.dumps(progress, indent=2)); print('RGB_ACT_PROGRESS', json.dumps(progress), flush=True)
                if update % args.checkpoint_every == 0 or update == args.updates:
                    checked = validation(student, sim, real, args.seed+1)
                    folder = f'checkpoint_{update:06d}'; student.save_copy(args.output/folder)
                    (args.output/f'validation_{update:06d}.json').write_text(json.dumps(checked, indent=2))
                    retained=real_replay_retained(checked,initial,args.max_real_l1_increase,args.real_l1_absolute_slack)
                    if checked['sim_l1'] < best and retained:
                        best = checked['sim_l1']; selected = folder
                    (args.output/'selection.json').write_text(json.dumps({'checkpoint': selected,
                        'validation_sim_l1': best,'latest_real_replay_retained':retained,
                        'maximum_real_l1':None if args.max_real_l1_increase is None else initial['real_l1']*(1+args.max_real_l1_increase)+args.real_l1_absolute_slack,
                        'criterion': 'lowest held-out sim episode RGB action L1 satisfying the declared real replay retention gate; closed-loop evaluation still required'}, indent=2))
                    if digest(args.checkpoint/'model.safetensors') != source_hash or digest(args.index) != sim.digest:
                        raise ValueError('Immutable training source changed')
        status.write_text(json.dumps({'stage': 'rgb_act_fit_complete', 'updates': completed, 'selected': selected,
            'validation_sim_l1': best, 'closed_loop_evaluated': False, 'ready_for_robot': False,
            'elapsed_seconds': time.perf_counter()-started}, indent=2))
    except BaseException as error:
        status.write_text(json.dumps({'stage': 'failed', 'error': repr(error)}, indent=2)); raise


if __name__ == '__main__': main()
