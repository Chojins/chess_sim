"""Create an explicitly hybrid capture profile; source actor remains frozen."""
import argparse
import json
from pathlib import Path
import shutil
from capture_demonstrations import file_hash
from capture_transport import settings_from_command
from train_openings import OpeningChessEnv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source', type=Path, default=Path('outputs/geometric-demonstrations-v24/pretraining'))
    args = parser.parse_args()
    source = json.loads((args.source/'manifest.json').read_text())
    if len(source['cases']) != 1 or not source['cases'][0]['capture']:
        raise ValueError('One first-capture command per profile')
    actor = args.source/'demo_pretrained.zip'
    if file_hash(actor) != source['pretrained_sha256']:
        raise ValueError('Source actor changed')
    base = OpeningChessEnv(source['cases'], capture_home_retry=True, capture_step_limit=1600)
    try:
        if base.latest['observation']['joint_map'] != source['joint_map']:
            raise ValueError('Robot calibration changed')
        config = settings_from_command(base, source['cases'][0])
        args.output.mkdir(parents=True, exist_ok=False)
        shutil.copy2(actor, args.output/'source_actor.zip')
        shutil.copy2('capture_transport.py', args.output/'capture_transport.py')
        manifest = {**source, 'mode': 'hybrid_capture_profile', 'capture_step_limit':1600,
                    'actor_time_reference':'original 800-step budget, excluding supervisor steps; residual history paused during transport',
                    'source_actor': str(actor.resolve()), 'source_actor_sha256':file_hash(actor),
                    'capture_transport': {'config':config.to_dict(), 'code_sha256':file_hash('capture_transport.py'),
                        'role':'explicit supervisor, separate from learned residual',
                        'control_inputs':'chess command, static board/RHS calibration and measured joints; no piece poses'},
                    'note':'Longer capture step budget accounts for physical transport; requires fresh evaluation. No RL improvement claimed.'}
        (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2))
        print(json.dumps({'profile':str(args.output),'config':config.to_dict()}))
    finally:
        base.close()


if __name__ == '__main__':
    main()
