"""Check a selected RGB ACT copy reloads identically on whole-episode holdouts."""
import argparse
import gc
import json
from pathlib import Path
import numpy as np
import torch
from teacher.act_student import GoalACT
from teacher.episode_data import digest
from teacher.student_data import SimChunks,RealChunks


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    args=p.parse_args();manifest=json.loads((args.run/'manifest.json').read_text())
    selected=json.loads((args.run/'selection.json').read_text())['checkpoint'];checkpoint=args.run/selected
    if digest(manifest['accepted_index'])!=manifest['accepted_index_sha256']:raise ValueError('Training index changed')
    if digest(Path(manifest['source_checkpoint'])/'model.safetensors')!=manifest['source_checkpoint_sha256']:
        raise ValueError('Original ACT changed')
    model=GoalACT(checkpoint,args.device,manifest['n_action_steps']);model.policy.eval()
    sim=SimChunks(manifest['accepted_index'],model.config['chunk_size'],audit=False)
    real=RealChunks(manifest['real_data'],model.config['chunk_size'],len(manifest['real_splits']['train']),
        sampling=manifest.get('real_sampling','prefix'))
    if real.splits!=manifest['real_splits']:raise ValueError('Real replay split changed')
    samples=[];rng=np.random.default_rng(142102)
    for episode in sim.splits['validation']:
        for frame in sorted({0,episode['frames']//2,episode['frames']-1}):
            path=Path(episode['path'])/f'frame_{frame:06d}.npz'
            if digest(path)!=episode['file_sha256'][str(path.relative_to(episode['episode_path']))]:
                raise ValueError('Held-out RGB frame changed')
            samples.append(sim.sample(rng,'validation',frame,episode))
    for episode in real.splits['validation']:
        samples.append(real.sample(rng,'validation',0,episode))
    with torch.inference_mode():expected=[model.policy.model(model.batch([s]))[0].cpu() for s in samples]
    copy=args.run/'selected_roundtrip';model.save_copy(copy);del model;gc.collect()
    if args.device=='cuda':torch.cuda.empty_cache()
    restored=GoalACT(copy,args.device,manifest['n_action_steps']);restored.policy.eval();maximum=0.
    with torch.inference_mode():
        for sample,before in zip(samples,expected):
            after=restored.policy.model(restored.batch([sample]))[0].cpu()
            torch.testing.assert_close(after,before,rtol=0,atol=0)
            maximum=max(maximum,float((after-before).abs().max()))
    report={'passed':True,'selected_checkpoint':selected,'checkpoint_sha256':digest(checkpoint/'model.safetensors'),
        'samples':len(samples),'sim_holdout_samples':sum(len(s['images'])==3 for s in samples),
        'real_holdout_samples':sum(len(s['images'])==2 for s in samples),'maximum_reload_error':maximum,
        'original_ACT_unchanged':True,'closed_loop_success_established':False}
    (args.run/'roundtrip_integrity.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)


if __name__=='__main__':main()
