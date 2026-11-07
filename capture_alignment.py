"""Reward-only grasp shaping for capture pickup and complete-move training."""
import numpy as np


def capture_alignment(info):
    observation = info['telemetry']['observation']
    index = observation.get('task_geometry', {}).get('capture_index')
    if index is None or info.get('score', {}).get('phase', 'remove_captured') != 'remove_captured':
        return 0., float('inf')
    target = np.asarray(observation['pieces']['piece_grasp_targets_m'][index])
    delta = np.asarray(observation['gripper_grasp_center_m']) - target
    if not np.isfinite(delta).all():
        raise ValueError('Non-finite capture grasp geometry')
    # A tighter vertical scale distinguishes neck-centred contact from the
    # observed 12-16mm high grasps. Targets retain the user's 17mm pawn height.
    quality = float(np.exp(-.5*np.sum((delta / [.008,.008,.005])**2)))
    upright = observation['pieces']['piece_upright_cosines'][index] > .95
    return quality if upright else 0., float(np.linalg.norm(delta))


def closing_alignment_penalty(previous_observation, info, gate, weight=2.):
    """Charge misaligned commanded closure, not finger contact/chatter.

    Reward-only simulator geometry; the actor gets no object pose input.
    A full 12.5-degree misaligned closure costs at most 2.
    A nearby open/holding jaw and post-lift motion incur no closing cost.
    """
    quality, distance = capture_alignment(info)
    score = info.get('score', {})
    if distance >= .04 or gate <= 0 or (score.get('capture') or {}).get('lifted', False):
        return 0., 0., quality
    current = info['telemetry']['observation']
    scale = current['joint_map']['scale'][5]
    before = previous_observation['target_legacy_action'][5]
    after = current['target_legacy_action'][5]
    closing_deg = max(0., (before-after)*scale)
    penalty = -weight * gate * min(closing_deg/12.5, 1.) * (1-quality)
    return float(penalty), float(closing_deg), quality
