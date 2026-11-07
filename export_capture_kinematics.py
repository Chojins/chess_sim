"""Export immutable USD joint frames for offline NumPy grasp labelling."""
import argparse
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
from pxr import Usd, UsdGeom
from episode_replay import EpisodeReplay
from grasp_geometry import GRASP_LOCAL_M


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    stage = Usd.Stage.Open('Sim_Stage.usd')
    stage.SetEditTarget(stage.GetSessionLayer())
    with TemporaryDirectory(prefix='chess-fk-export-') as directory:
        fk = EpisodeReplay(stage, 'outputs/replay/episode_000000.npz',
                           'calibration/legacy_blue_joint_map_physics.json', directory, kinematic=False)
        cache = UsdGeom.XformCache()
        joints = []
        for name, prim, parent, child, f0, f1, axis in fk.joints:
            joints.append({'name': name, 'parent': str(parent), 'child': str(child),
                           'f0': np.array(f0).tolist(), 'f1_inverse': np.array(f1.GetInverse()).tolist(),
                           'axis': list(axis), 'parent_world': np.array(cache.GetLocalToWorldTransform(stage.GetPrimAtPath(parent))).tolist()})
        result = {'joints': joints, 'names': fk.mapping.sim_names,
                  'tool_link': fk.root + '/gripper', 'tool_local': list(GRASP_LOCAL_M),
                  'limits': [[j[1].GetAttribute('physics:lowerLimit').Get(), j[1].GetAttribute('physics:upperLimit').Get()] for j in fk.joints[:6]],
                  'usd_sha256': hashlib.sha256(Path('Sim_Stage.usd').read_bytes()).hexdigest()}
        with args.output.open('x') as stream:
            json.dump(result, stream, indent=2)
    print('KINEMATICS_EXPORTED', str(args.output), flush=True)


if __name__ == '__main__':
    main()
