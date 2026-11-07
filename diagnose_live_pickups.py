"""Summarize completed live capture traces and plot alignment through closure."""
import argparse
import json
from pathlib import Path
import numpy as np


def summarize_episode(path):
    initial = json.loads((path / 'initial.json').read_text())['telemetry']['observation']
    victim = initial['task_geometry']['capture_index']
    rows = [json.loads(line) for line in (path / 'steps.jsonl').read_text().splitlines()]
    if victim is None or not rows:
        return None
    paused = [row for row in rows if (row.get('act_timing') or {}).get('phase') == 'arm_settle']
    closed = [row for row in rows if (row.get('act_timing') or {}).get('phase') == 'closed_settle']
    def errors(row):
        observation = row['placement_telemetry']
        delta = np.asarray(observation['gripper_grasp_center_m']) - observation['pieces']['piece_grasp_targets_m'][victim]
        return (delta * 1000).tolist()
    heights = [row['placement_telemetry']['pieces']['piece_centers_m'][victim][2] -
               initial['pieces']['piece_centers_m'][victim][2] for row in rows]
    fingers = closed[-1]['placement_telemetry']['measured_finger_joint_deg'] if closed else None
    return {'path': str(path.resolve()), 'label': path.name, 'trial_group': path.parent.parent.name,
            'success': rows[-1]['score']['success'],
            'reason': rows[-1]['score']['reason'], 'steps': len(rows),
            'objective': rows[-1]['score'].get('objective', 'complete move'),
            'paused_alignment_frames': len(paused),
            'first_alignment_error_xyz_mm': errors(paused[0]) if paused else None,
            'final_alignment_error_xyz_mm': errors(paused[-1]) if paused else None,
            'settled_closure_error_xyz_mm': errors(closed[-1]) if closed else None,
            'settled_finger_angles_deg': fingers,
            'maximum_victim_lift_mm': float(max(heights) * 1000),
            'maximum_held_streak': max(row['score'].get('hold_streak', 0) for row in rows),
            'protected_piece_disturbance': rows[-1]['score'].get('other_pieces_disturbed')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    results = []
    for path in sorted(args.root.rglob('result.json')):
        if not (path.parent / 'steps.jsonl').exists():
            continue
        result = summarize_episode(path.parent)
        if result is not None:
            results.append(result)
    if not results:
        raise ValueError('No completed capture episodes')
    (args.output / 'alignment.json').write_text(json.dumps(results, indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plot_rows = [row for row in results if row['first_alignment_error_xyz_mm'] is not None]
    if plot_rows:
        figure, axes = plt.subplots(1, 3, figsize=(13, max(4, len(plot_rows) * .24)), sharey=True)
        positions = np.arange(len(plot_rows))
        for axis_index, axis in enumerate(axes):
            for i, row in enumerate(plot_rows):
                first = row['first_alignment_error_xyz_mm'][axis_index]
                final = row['final_alignment_error_xyz_mm'][axis_index]
                axis.plot([first, final], [i, i], color='green' if row['success'] else 'firebrick', alpha=.7)
                axis.plot(first, i, 'o', color='gray', markersize=4)
                axis.plot(final, i, 'x', color='green' if row['success'] else 'firebrick')
            axis.axvline(0, color='black', linewidth=.7)
            axis.set_xlabel(f"{'XYZ'[axis_index]} grasp error (mm)")
            axis.grid(axis='x', alpha=.2)
        axes[0].set_yticks(positions, [f"{i+1}: {row['trial_group']} / {row['label']}" for i, row in enumerate(plot_rows)])
        axes[0].invert_yaxis()
        figure.suptitle('Open-jaw alignment: gray=start, cross=end; green=task success, red=failure')
        figure.tight_layout()
        figure.savefig(args.output / 'alignment.png', dpi=160)
        plt.close(figure)
    print(json.dumps({'episodes': len(results), 'successes': sum(row['success'] for row in results),
                      'intervened_episodes': len(plot_rows), 'output': str(args.output),
                      'note': 'Trace diagnosis only; diagnostic pickup success is not complete capture success'}))


if __name__ == '__main__':
    main()
