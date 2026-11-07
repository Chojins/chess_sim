"""Map the user's numbered video review to commands, jaw motion and height."""
import argparse
import json
from pathlib import Path
import re
import numpy as np
from capture_demonstrations import JointKinematics, file_hash
from capture_timing import closure_window


USER_SUCCESS = {2, 5, 6, 15, 16, 17, 18}
PROBE_NUMBERS = (9, 13, 20, 2, 4)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    root = Path('outputs/capture-supervised-heldout-v10')
    kin_path = Path('outputs/capture-supervised-kinematics-v10.json')
    kin = JointKinematics(json.loads(kin_path.read_text()))
    links = re.findall(r'href="([^"]+cameras.mp4)"', (root / 'videos.html').read_text())
    if len(links) != 20:
        raise ValueError('Expected the twenty videos reviewed by the user')
    audits, cases, plots = [], [], []
    for number, link in enumerate(links, 1):
        path = root / Path(link).parent
        initial = json.loads((path / 'initial.json').read_text())
        observation = initial['telemetry']['observation']
        mapping = observation['joint_map']
        rows = [json.loads(line) for line in (path / 'steps.jsonl').read_text().splitlines()]
        commands = np.array([np.array(mapping['sim_home_deg']) + np.array(mapping['scale']) *
                             (np.array(row['residual']['executed_legacy']) - mapping['legacy_home']) for row in rows])
        hold, close, command_z = closure_window(commands, kin)
        victim = observation['task_geometry']['capture_index']
        target = np.array(rows[hold]['placement_telemetry']['pieces']['piece_grasp_targets_m'][victim])
        point = np.array(rows[hold]['placement_telemetry']['gripper_grasp_center_m'])
        measured = np.array([row['placement_telemetry']['measured_joint_deg'] for row in rows])
        measured_z = np.array([row['placement_telemetry']['gripper_grasp_center_m'][2] for row in rows])
        lift = next((j for j in range(hold + 1, close + 1) if command_z[j] >= command_z[hold] + .005), None)
        audit = {'video_number': number, 'source': str(path.resolve()), 'user_success': number in USER_SUCCESS,
                 'logged_success': rows[-1]['score']['success'], 'approach_step': hold + 1, 'closure_step': close + 1,
                 'command_jaw_at_approach_deg': float(commands[hold, 5]),
                 'measured_error_at_approach_mm': ((point - target) * 1000).tolist(),
                 'command_error_at_approach_mm': ((kin.point(commands[hold]) - target) * 1000).tolist(),
                 'lift_before_closure': lift is not None, 'lift_step': lift + 1 if lift is not None else None,
                 'closure_delay_after_lift_s': (close - lift) / 30 if lift is not None else 0.,
                 'command_jaw_at_lift_deg': float(commands[lift, 5]) if lift is not None else None}
        audits.append(audit)
        if number in PROBE_NUMBERS:
            cases.append({'id': f'video_{number:02d}', 'video_number': number, 'source': str(path.resolve()),
                          'source_hashes': {name: file_hash(path / name) for name in ('initial.json', 'steps.jsonl')},
                          'reset_options': initial['reset_options'], 'joint_map': mapping,
                          'expected_initial_joint_deg': observation['measured_joint_deg'],
                          'commands_deg': commands.tolist(), 'hold_index': hold, 'close_index': close,
                          'original_success': audit['logged_success'], 'geometry_audit': audit})
            plots.append((number, commands, measured, command_z, measured_z, target[2], hold, close))
    report = {'episodes': audits, 'user_notes_match_logged_outcomes': all(r['user_success'] == r['logged_success'] for r in audits),
              'failures': sum(not r['logged_success'] for r in audits),
              'failures_with_commanded_lift_before_closure': sum(not r['logged_success'] and r['lift_before_closure'] for r in audits),
              'note': 'Video 4 was described as gripping too high; remaining failures as lifting before gripping. Encoder and command audit, not contact classification.'}
    (args.output / 'timing_audit.json').write_text(json.dumps(report, indent=2))
    plan = {'version': 'recorded_close_before_lift_probe_v1', 'cases': cases,
            'kinematics_sha256': file_hash(kin_path), 'grasp_geometry_changed': False, 'rl_training': False,
            'source': 'frozen recorded executed commands; no live ACT or residual inference',
            'trigger': 'offline lowest commanded approach height before first closing command; robot kinematics and commanded joints only',
            'execution': 'hold arm at selected recorded command while playing subsequent jaw commands, wait >=12 frames and 5 encoder-settled frames, then resume arm path with closed jaw',
            'maximum_settle_steps': 45, 'minimum_settle_steps': 12, 'stable_frames': 5,
            'objective': 'timing diagnostic; not a learned policy or complete capture'}
    (args.output / 'plan.json').write_text(json.dumps(plan, indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(plots), 2, figsize=(12, 3 * len(plots)))
    for row, (number, commands, measured, cz, mz, target_z, hold, close) in enumerate(plots):
        t = np.arange(len(cz)) / 30
        axes[row, 0].plot(t, (cz - target_z) * 1000, label='commanded tool height')
        axes[row, 0].plot(t, (mz - target_z) * 1000, label='measured tool height')
        axes[row, 0].axhline(0, color='gray', linestyle=':')
        axes[row, 0].set(ylabel=f'Video {number}: height error (mm)', ylim=(-10, 110))
        axes[row, 1].plot(t, commands[:, 5], label='jaw command')
        axes[row, 1].plot(t, measured[:, 5], label='jaw measured')
        axes[row, 1].set(ylabel='Jaw angle (degrees)')
        for ax in axes[row]:
            ax.axvline(hold / 30, color='orange', linestyle='--', label='approach low point')
            ax.axvline(close / 30, color='green', linestyle='--', label='closing command <=1 degree')
            ax.set(xlabel='Simulation seconds', xlim=(max(0, hold / 30 - 1), min(t[-1], close / 30 + .7)))
            ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.output / 'timing_audit.png', dpi=150)
    plt.close(fig)
    print('TIMING_AUDIT', json.dumps({k: report[k] for k in report if k != 'episodes'}), flush=True)


if __name__ == '__main__':
    main()
