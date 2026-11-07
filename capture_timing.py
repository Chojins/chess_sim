"""Recorded-command timing diagnostic, separate from ACT/residual execution."""
import numpy as np


def closure_window(commands, kinematics):
    """Select lowest approach command before closure using robot motion only."""
    commands = np.asarray(commands, float)
    if commands.ndim != 2 or commands.shape[1] != 6 or not np.isfinite(commands).all():
        raise ValueError('Expected finite six-joint commands')
    open_indices = np.flatnonzero(commands[:, 5] >= 6.)
    if not len(open_indices):
        raise ValueError('No open-jaw approach')
    opened = int(open_indices[0])
    closed_indices = np.flatnonzero((np.arange(len(commands)) > opened) & (commands[:, 5] <= 1.))
    if not len(closed_indices):
        raise ValueError('No recorded closing command')
    closed = int(closed_indices[0])
    candidates = np.arange(opened, closed)
    candidates = candidates[(commands[candidates, 1] > 60) & (commands[candidates, 1] < 140)
                            & (commands[candidates, 5] > 4.)]
    if not len(candidates):
        raise ValueError('No approach in the shoulder working range')
    z = np.array([kinematics.point(q)[2] for q in commands])
    hold = int(candidates[np.argmin(z[candidates])])
    return hold, closed, z


def settled_fingers(current, previous, held_arm, tolerance_deg=.8):
    """Encoder-only settling; this does not certify that a piece was grasped."""
    joints = np.asarray(current['measured_joint_deg'])
    fingers = current['measured_finger_joint_deg']
    prior = previous['measured_finger_joint_deg']
    arm = float(np.max(np.abs(joints[:5] - np.asarray(held_arm)[:5])))
    narrowed = max(abs(float(fingers[name])) for name in ('jaw1_hinge', 'jaw2_hinge')) <= 6.
    stable = max(abs(float(fingers[name]) - float(prior[name])) for name in fingers) <= .25
    return bool(arm <= tolerance_deg and narrowed and stable)
