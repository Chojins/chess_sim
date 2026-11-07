"""User-specified grasp geometry, in metres; independent of the learner."""
import numpy as np

FINGER_FRONT_M = 0.08212725  # zero-angle imported finger mesh, along local -Y
FINGER_SETBACK_M = 0.007
GRASP_LOCAL_M = (0., -(FINGER_FRONT_M - FINGER_SETBACK_M), 0.)
GRASP_HEIGHTS_M = dict(pawn=.017, knight=.017, rook=.019, bishop=.027, king=.032, queen=.030)
GRASP_REFERENCE = {
    'version': 'finger_setback7mm_piece_heights_v1',
    'local_point_m': list(GRASP_LOCAL_M),
    'finger_setback_m': FINGER_SETBACK_M,
    'finger_reference': 'zero_hinge_angle_front; fixed axial reference',
    'piece_grasp_heights_m': GRASP_HEIGHTS_M,
}


def piece_grasp_target(path, bottom, top):
    """Target in the authored upright world bounds; caller attaches it to body."""
    kind = path.rsplit('/', 1)[-1].split('_')[0]
    height = GRASP_HEIGHTS_M[kind]
    bottom, top = np.asarray(bottom, float), np.asarray(top, float)
    if not np.isfinite([*bottom, *top]).all() or height > top[2] - bottom[2]:
        raise ValueError(f'Grasp height outside piece bounds: {path}')
    point = (bottom + top) / 2
    point[2] = bottom[2] + height
    return point
