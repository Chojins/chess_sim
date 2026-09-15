"""Check independent recorded moves against board squares without refitting."""
import json
from pathlib import Path
import numpy as np
from pxr import Usd, UsdGeom, Gf
from episode_replay import EpisodeReplay, JointMap


def main():
    output = Path('outputs/replay/cross-episode')
    output.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.Open('Sim_Stage.usd')
    replay = EpisodeReplay(stage, 'outputs/replay/episode_000000.npz',
                           'calibration/legacy_blue_joint_map_board_fit.json', output, kinematic=False)
    bbox = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ['default', 'render'])
    centers = np.array([bbox.ComputeWorldBound(p).ComputeAlignedRange().GetMidpoint()
                       for p in Usd.PrimRange(stage.GetPrimAtPath('/World/board_assembly'))
                       if p.GetName().startswith('tn__boardsquare_')]) * 1000
    xs, ys = [np.unique(np.round(centers[:, k], 3)) for k in (0, 1)]
    limits = np.array([[j[1].GetAttribute('physics:lowerLimit').Get(), j[1].GetAttribute('physics:upperLimit').Get()]
                       for j in replay.joints[:6]])
    checks = [(1, 'g1f3', [(150, 'g1'), (220, 'f3')]), (2, 'd2d4', [(180, 'd2'), (230, 'd4')])]
    report = {'method': 'Independent episodes; contact frames and move labels inspected in recorded laptop video. '
                        'Estimated distal fingertip XY compared with board-square centres; not physical replay success.',
              'maps': {}}
    for name in ['legacy_blue_joint_map', 'legacy_blue_joint_map_board_fit', 'legacy_blue_joint_map_contact_height',
                 'legacy_blue_joint_map_physics']:
        mapping = JointMap('calibration/' + name + '.json')
        results = []
        for episode, move, contacts in checks:
            with np.load(f'outputs/replay/episode_{episode:06d}.npz') as data:
                states, actions = mapping.to_sim_deg(data['states']), mapping.to_sim_deg(data['actions'])
                if data['joint_names'].tolist() != mapping.names:
                    raise ValueError('Unexpected joint order')
            row = {'episode': episode, 'move': move, 'frames': len(states),
                   'state_limit_violations': int(((states < limits[:, 0]) | (states > limits[:, 1])).sum()),
                   'action_limit_violations': int(((actions < limits[:, 0]) | (actions > limits[:, 1])).sum()), 'contacts': []}
            for frame, square in contacts:
                q = dict(zip(mapping.sim_names, states[frame]))
                q['jaw2_hinge'] = -q['jaw1_hinge']
                tip = np.array(replay.forward(q)[replay.root + '/gripper'].Transform(Gf.Vec3d(0, -0.08112725, 0))) * 1000
                target = np.array([xs[ord(square[0]) - ord('a')], ys[int(square[1]) - 1]])
                row['contacts'].append({'frame': frame, 'square': square, 'tip_mm': tip.tolist(),
                                        'target_xy_mm': target.tolist(), 'xy_error_mm': float(np.linalg.norm(tip[:2] - target))})
            results.append(row)
        report['maps'][name] = results
    (output / 'report.json').write_text(json.dumps(report, indent=2))
    for name, results in report['maps'].items():
        print(name)
        for row in results:
            print(row['move'], 'errors mm', [round(c['xy_error_mm'], 2) for c in row['contacts']],
                  'state/action limit violations', row['state_limit_violations'], row['action_limit_violations'])


if __name__ == '__main__':
    main()
