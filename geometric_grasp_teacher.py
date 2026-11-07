"""Privileged grasp teacher for generating residual supervision in simulation.

Object poses enter this teacher only. The learner's observation vector remains
frozen ACT images, joints, proposed actions and the commanded chess move.
"""
import numpy as np
from capture_demonstrations import BOUNDS, FILTER, invert_filtered
from residual_learning import filter_residual
from review_grasp_geometry import opening_center
from show_centered_grasp import fit_pose


class GeometricGraspTeacher:
    def __init__(self, env, specification, anchors):
        self.env, self.spec, self.anchors = env, specification, anchors
        np.testing.assert_equal(env.env.bounds, BOUNDS)
        if env.residual_alpha != FILTER['alpha'] or env.max_residual_change_deg.tolist() != FILTER['max_change_deg']:
            raise ValueError('Teacher and inverse residual filter differ')
        self.reset()

    def reset(self):
        self.hold = None
        self.hold_native = None
        self.hold_height = None
        self.taper_start = None
        self.active_guard = None
        self.last_decision = None

    def predict(self, observation, deterministic=True):
        base = self.env.unwrapped
        timing = self.env.env.policy
        actual = base.latest['observation']
        q = np.asarray(actual['measured_joint_deg'])
        proposal = base.sim_home + base.scale*(self.env.env.base-base.legacy_home)
        gate, previous = self.env.gate, self.env.previous_action
        desired = np.zeros(6)
        point = opening_center(self.spec, self.anchors, q, actual['measured_finger_joint_deg']['jaw2_hinge'])
        decision = {'phase': timing.phase, 'native_index': timing.native_index,
                    'interventions': timing.interventions, 'opening_center_m': point.tolist(),
                    'privileged': True, 'role': 'pass_through'}
        if timing.phase == 'arm_settle':
            task = base.task
            index = task.capture_index if task.capture_index is not None and not task.capture_bonus else task.source_index
            target = np.asarray(actual['pieces']['piece_grasp_targets_m'][index])
            decision.update(target_index=index, target_m=target.tolist(),
                            alignment_error_mm=((point-target)*1000).tolist())
            if self.active_guard != timing.interventions:
                self.active_guard = timing.interventions
                self.hold = None
                self.taper_start = None
                self.hold_native = timing.native_index
                self.hold_height = target[2]
            if np.linalg.norm(point-target) < .045 and gate > .05:
                nominal = proposal.copy(); nominal[5] = q[5]
                try:
                    fitted, _, _ = fit_pose(self.spec, self.anchors, nominal, target,
                                            np.asarray(actual['joint_limits_deg']))
                except RuntimeError:
                    decision['role'] = 'unreachable_inside_residual_bounds'
                else:
                    # Encoder feedback compensates drive sag; corrections are
                    # still projected through the learner's exact rate filter.
                    command = base.sim_home + base.scale*(np.asarray(actual['target_legacy_action'])-base.legacy_home)
                    desired[:5] = (command[:5]+.5*(fitted[:5]-q[:5])-proposal[:5])/BOUNDS[:5]
                    decision.update(role='align_opening', fitted_joint_deg=fitted.tolist())
            else:
                decision['role'] = 'skip_guard_far_from_active_piece'
        elif self.hold is not None and timing.phase in ('close_jaw', 'closed_settle'):
            desired[:5] = self.hold[:5]
            decision['role'] = 'retain_grasp_arm_offset'
        elif self.hold is not None and timing.native_index > self.hold_native:
            if self.taper_start is None and point[2] >= self.hold_height+.020:
                self.taper_start = timing.native_index
            fraction = 1. if self.taper_start is None else max(0., 1.-(timing.native_index-self.taper_start)/40.)
            desired[:5] = self.hold[:5]*fraction
            decision.update(role='smooth_grasp_offset_into_lift', taper_fraction=fraction)
        # Match the force-producing closed command used in the approved lift
        # probe. A forecast opening is passed through for normal release.
        if timing.interventions and timing.phase != 'arm_settle' and proposal[5] <= 4.:
            desired[5] = (-2.5-proposal[5])/BOUNDS[5]
        low = filter_residual(-np.ones(6), gate, previous, BOUNDS, FILTER['alpha'], FILTER['max_change_deg'])
        high = filter_residual(np.ones(6), gate, previous, BOUNDS, FILTER['alpha'], FILTER['max_change_deg'])
        reachable = np.clip(desired, low, high)
        raw = np.zeros(6, np.float32) if gate <= 1e-8 else invert_filtered(reachable, gate, previous)
        if timing.phase == 'arm_settle':
            self.hold = reachable.copy()
        decision.update(action=raw.tolist(), reachable_residual=reachable.tolist(), gate=gate)
        self.last_decision = decision
        return raw, None
