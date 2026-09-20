"""Offline FK comparison of successful replay and failed ACT grasp traces.

Run with ./run_usd_python.sh compare_grasps.py. No physics is advanced.
"""
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from pxr import Gf, Usd

from episode_replay import EpisodeReplay


def main():
    output = Path('outputs/grasp-comparison')
    output.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.Open('Sim_Stage.usd')
    stage.SetEditTarget(stage.GetSessionLayer())
    fk = EpisodeReplay(stage, 'outputs/replay/episode_000000.npz',
                       'calibration/legacy_blue_joint_map_physics.json', output, kinematic=False)
    reference = [json.loads(line) for line in Path('outputs/magnets/verification-final/physics_samples.jsonl').read_text().splitlines()]
    trials = {'recorded_half_speed': [(r['measured_joint_deg'], r['sim_joint_deg'], r['pieces']['pawn_center_m']) for r in reference]}
    for path in sorted(Path('outputs/act-window-pilot').glob('seed_*_window_100')):
        initial = json.loads((path / 'initial.json').read_text())['telemetry']['observation']
        source = initial['task_geometry']['source_center_m']
        # ACT failures barely lift the pawn. Use the source for relative tool
        # coordinates; exact clearance to the moving pawn is in the step log.
        trials[path.name] = [(fk.mapping.to_sim_deg(r['next_state']), fk.mapping.to_sim_deg(r['action_legacy']), source)
                            for r in map(json.loads, (path / 'steps.jsonl').read_text().splitlines())]
    report = {}
    for name, rows in trials.items():
        samples = []
        opened = False
        closure = None
        for index, (q, target, pawn) in enumerate(rows):
            angles = dict(zip(fk.mapping.sim_names, q))
            angles['jaw2_hinge'] = -q[5]
            tip = np.array(fk.forward(angles)['/World/so100_blue/gripper'].Transform(Gf.Vec3d(0, -.08112725, 0)))
            sample = {'step': index, 'joint_deg': list(map(float, q)),
                      'target_deg': list(map(float, target)), 'tip_m': tip.tolist(),
                      'tool_relative_to_pawn_mm': ((tip - pawn) * 1000).tolist()}
            samples.append(sample)
            opened |= target[5] > 4
            if opened and closure is None and target[5] < 0 and np.linalg.norm(sample['tool_relative_to_pawn_mm']) < 30:
                closure = sample
        closest = min(samples, key=lambda s: np.linalg.norm(s['tool_relative_to_pawn_mm']))
        report[name] = {'near_pawn_close_command': closure, 'closest_tool': closest}
        (output / f'{name}.json').write_text(json.dumps(samples, indent=2))
    desired = np.array(report['recorded_half_speed']['near_pawn_close_command']['tool_relative_to_pawn_mm']) / 1000
    for name, entry in report.items():
        if name == 'recorded_half_speed' or entry['near_pawn_close_command'] is None:
            continue
        sample = entry['near_pawn_close_command']
        q = np.array(sample['joint_deg'])
        pawn = np.array(sample['tip_m']) - np.array(sample['tool_relative_to_pawn_mm']) / 1000
        def error(delta):
            corrected = q.copy()
            corrected[:5] += delta
            angles = dict(zip(fk.mapping.sim_names, corrected))
            angles['jaw2_hinge'] = -corrected[5]
            point = np.array(fk.forward(angles)['/World/so100_blue/gripper'].Transform(Gf.Vec3d(0, -.08112725, 0)))
            return (point - pawn - desired) * 1000
        fit = least_squares(error, np.zeros(5), bounds=(-np.ones(5) * 2, np.ones(5) * 2))
        entry['kinematic_feasibility_only'] = {'delta_joint_deg': fit.x.tolist(),
                                               'remaining_tip_error_mm': float(np.linalg.norm(error(fit.x)))}
    (output / 'comparison.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
