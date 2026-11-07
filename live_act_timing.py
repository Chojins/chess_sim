"""Opt-in close-before-lift control around frozen live ACT and residuals.

Robot joint commands, FK and encoder feedback are the only control inputs.
There is no piece pose lookup, recorded trajectory, IK, or weight update.
"""
from dataclasses import asdict, dataclass
import numpy as np
from capture_timing import settled_fingers


@dataclass(frozen=True)
class TimingConfig:
    guard_all_closures: bool = False
    minimum_arm_adaptation_steps: int = 0
    latch_after_adaptation: bool = False
    board_pickup_bounds_xy_m: tuple | None = None
    alignment_open_jaw_deg: float | None = None
    suppress_board_release_closures: bool = False
    required_tabletop_releases: int = 0
    confirm_settled_contact: bool = False
    minimum_lift_m: float = .005
    maximum_approach_z_m: float = .120
    open_jaw_deg: float = 4.
    close_command_deg: float = 1.
    arm_tolerance_deg: float = .8
    arm_stable_frames: int = 3
    maximum_arm_settle_steps: int = 30
    minimum_closed_settle_steps: int = 12
    maximum_closed_settle_steps: int = 45
    closed_stable_frames: int = 5
    maximum_interventions: int = 1

    def to_dict(self):
        settings = asdict(self)
        # Older deployed manifests predate this opt-in protocol. Keep their
        # serialized settings identical when it is disabled.
        if not self.latch_after_adaptation:
            settings.pop('latch_after_adaptation')
        if not self.suppress_board_release_closures:
            settings.pop('suppress_board_release_closures')
        if not self.required_tabletop_releases:
            settings.pop('required_tabletop_releases')
        if not self.confirm_settled_contact:
            settings.pop('confirm_settled_contact')
        if self.board_pickup_bounds_xy_m is None:
            settings.pop('board_pickup_bounds_xy_m')
        else:
            settings['board_pickup_bounds_xy_m'] = list(self.board_pickup_bounds_xy_m)
        return settings


class GraspHistory:
    """Recognize held-object releases using joint feedback and calibrated FK.

    A release followed by empty-jaw closure must not become another guarded
    pickup. Contact requires both fingers to stop short of a closed command;
    lift and release are confirmed independently. No object state is read.
    """
    def __init__(self, kinematics, config):
        if config.board_pickup_bounds_xy_m is None:
            raise ValueError('Release discrimination needs calibrated board bounds')
        self.kinematics, self.config = kinematics, config
        self.bounds = np.asarray(config.board_pickup_bounds_xy_m).reshape(2, 2)
        self.guard = 0
        self.origin_z = None
        self.contact_frames = 0
        self.holding = self.lifted = self.board_release = False
        self.releases = 0
        self.tabletop_releases = 0
        self.previous_fingers = None
        self.empty_frames = 0

    def inside(self, point):
        return bool(np.all(point[:2] >= self.bounds[0]) and np.all(point[:2] <= self.bounds[1]))

    def state(self):
        result = dict(guard=self.guard, origin_z_m=self.origin_z,
                    contact_frames=self.contact_frames, holding=self.holding,
                    lifted=self.lifted, board_release=self.board_release,
                    releases=self.releases)
        if self.config.required_tabletop_releases:
            result['tabletop_releases'] = self.tabletop_releases
        if self.config.confirm_settled_contact:
            result['empty_frames'] = self.empty_frames
        return result

    def permits(self, forecast):
        if self.board_release:
            return False
        if self.holding and self.lifted:
            opening = np.flatnonzero(forecast[:, 5] > self.config.open_jaw_deg)
            if len(opening) and self.inside(self.kinematics.point(forecast[opening[0]])):
                return False
        return True

    def observe(self, q, fingers, command, phase, guard):
        point = self.kinematics.point(q)
        finger_angles = np.array([fingers['jaw1_hinge'], -fingers['jaw2_hinge']])
        settled = self.previous_fingers is not None and np.max(np.abs(finger_angles-self.previous_fingers)) <= .2
        self.previous_fingers = finger_angles
        if phase == 'arm_settle':
            if guard != self.guard:
                self.guard, self.contact_frames = guard, 0
                self.empty_frames = 0
                self.holding = self.lifted = False
            self.origin_z = float(point[2])
            return
        if self.origin_z is None:
            return
        jaw1, jaw2 = fingers['jaw1_hinge'], -fingers['jaw2_hinge']
        contact = (command[5] <= self.config.close_command_deg and
                   min(jaw1, jaw2) > .5 and min(jaw1, jaw2) - command[5] > 2.)
        if self.config.confirm_settled_contact:
            contact = contact and settled and phase != 'close_jaw'
            empty = command[5] <= self.config.close_command_deg and max(jaw1, jaw2) <= .5
            self.empty_frames = self.empty_frames + 1 if empty else 0
            if self.empty_frames >= 3:
                self.holding = self.lifted = False
        self.contact_frames = self.contact_frames + 1 if contact else 0
        if self.contact_frames >= 3:
            self.holding = True
        if self.holding and point[2] >= self.origin_z + self.config.minimum_lift_m:
            self.lifted = True
        if self.holding and command[5] > self.config.open_jaw_deg and min(jaw1, jaw2) >= 6.:
            if self.lifted:
                self.releases += 1
                inside = self.inside(point)
                if not inside:
                    self.tabletop_releases += 1
                self.board_release = self.board_release or (inside and self.tabletop_releases >= self.config.required_tabletop_releases)
            self.holding = self.lifted = False
            self.contact_frames = 0


def board_pickup_bounds(square_centers):
    """Static calibrated square footprint; never inspects piece poses."""
    squares = np.asarray(square_centers, float)
    if squares.shape != (64, 3) or not np.isfinite(squares).all():
        raise ValueError('Need 64 finite calibrated square centres')
    distances = np.linalg.norm(squares[:, None, :2] - squares[None, :, :2], axis=-1)
    np.fill_diagonal(distances, np.inf)
    pitch = float(np.median(distances.min(axis=1)))
    if pitch <= 0:
        raise ValueError('Duplicate square centres')
    return tuple(np.r_[squares[:, :2].min(axis=0) - pitch/2,
                       squares[:, :2].max(axis=0) + pitch/2])


def early_lift_plan(commands_deg, kinematics, config=TimingConfig()):
    """Look only within already predicted commands; no rollout or object truth."""
    commands = np.asarray(commands_deg, float)
    if commands.ndim != 2 or commands.shape[1] != 6 or not np.isfinite(commands).all():
        raise ValueError('Invalid ACT command forecast')
    opening = np.flatnonzero(commands[:, 5] > config.open_jaw_deg)
    if not len(opening):
        return None
    first_open = int(opening[0])
    closing = np.flatnonzero((np.arange(len(commands)) > first_open) &
                            (commands[:, 5] <= config.close_command_deg))
    if not len(closing):
        return None
    close = int(closing[0])
    candidates = np.arange(first_open, close)
    candidates = candidates[(commands[candidates, 5] > config.open_jaw_deg) &
                            (commands[candidates, 1] > 60) & (commands[candidates, 1] < 140)]
    if not len(candidates):
        return None
    # FK stops before closure, so it need not inspect transport commands.
    points = np.array([kinematics.point(q) for q in commands[:close + 1]])
    z = points[:, 2]
    if config.board_pickup_bounds_xy_m is not None:
        bounds = np.asarray(config.board_pickup_bounds_xy_m, float).reshape(2, 2)
        if not np.isfinite(bounds).all() or np.any(bounds[1] <= bounds[0]):
            raise ValueError('Invalid board pickup bounds')
        inside = np.all((points[:, :2] >= bounds[0]) & (points[:, :2] <= bounds[1]), axis=1)
        # A release followed by empty-jaw closure on the tabletop is not a
        # board pickup, even if its forecast briefly crosses the board edge.
        if not inside[close]:
            return None
        candidates = candidates[inside[candidates]]
        if not len(candidates):
            return None
    hold = int(candidates[np.argmin(z[candidates])])
    if z[hold] > config.maximum_approach_z_m:
        return None
    lifts = np.flatnonzero(z[hold + 1:close] >= z[hold] + config.minimum_lift_m)
    if not len(lifts) and not config.guard_all_closures:
        return None
    result = {'hold_index': hold, 'close_index': close,
              'lift_index': hold + 1 + int(lifts[0]) if len(lifts) else None, 'approach_z_m': float(z[hold])}
    if config.board_pickup_bounds_xy_m is not None:
        result.update(approach_xy_m=points[hold, :2].tolist(), closing_xy_m=points[close, :2].tolist())
    return result


class CloseBeforeLiftACT:
    """Separate arm/jaw cursors temporarily, leaving the ACT cache untouched."""
    def __init__(self, policy, mapping, kinematics, config=TimingConfig()):
        if policy.config.get('temporal_ensemble_coeff') is not None:
            raise ValueError('Timing control requires cached ACT without temporal ensembling')
        self.source = policy
        self.mapping = mapping
        self.kinematics = kinematics
        self.timing_config = config
        self.scale = np.array(mapping['scale'], float)
        self.home = np.array(mapping['legacy_home'], float)
        self.sim_home = np.array(mapping['sim_home_deg'], float)
        self.reset()

    def __getattr__(self, name):
        return getattr(self.source, name)

    def reset(self):
        self.source.reset()
        self.phase = 'normal'
        self.failure = None
        self.native_count = 0
        self.native_index = -1
        self.cache_origin = 0
        self.interventions = 0
        self.plan = None
        self.held_raw = None
        self.anchor = None
        self.open_executed_jaw = None
        self.last_arm_goal = None
        self.jaw_sequence = []
        self.jaw_index = 0
        self.closed_raw_jaw = None
        self.closed_executed_jaw = None
        self.catchup_until = -1
        self.last_offset = np.zeros(6)
        self.last_base = None
        self.phase_steps = 0
        self.streak = 0
        self.last_fingers = None
        self.current_metadata = None
        self.grasp_history = (GraspHistory(self.kinematics, self.timing_config)
                              if self.timing_config.suppress_board_release_closures else None)

    @property
    def chunk_phase(self):
        return ((self.native_index-self.cache_origin) % self.config['n_action_steps']) / self.config['n_action_steps']

    def resync_after_supervisor(self):
        """Replan from fresh cameras while preserving grasp/release history."""
        self.source.reset()
        self.cache_origin = self.native_count
        self.plan = self.held_raw = self.anchor = None
        self.catchup_until = -1
        self.phase = 'normal'
        self.phase_steps = self.streak = 0
        self.last_offset = np.zeros(6)

    def _physical(self, legacy):
        return np.clip(self.sim_home + self.scale * (legacy - self.home),
                       self.kinematics.limits[:, 0], self.kinematics.limits[:, 1])

    def action(self, state, laptop_rgb, phone_rgb):
        if self.phase in ('arm_settle', 'close_jaw', 'closed_settle', 'failed'):
            base = self.held_raw.copy()
            if self.phase == 'arm_settle' and self.timing_config.alignment_open_jaw_deg is not None:
                base[5] = self.home[5] + (self.timing_config.alignment_open_jaw_deg - self.sim_home[5]) / self.scale[5]
            if self.phase == 'close_jaw':
                base[5] = self.jaw_sequence[self.jaw_index]
            elif self.phase == 'closed_settle':
                base[5] = self.closed_raw_jaw
            self.last_base = base.astype(np.float32)
            return self.last_base.copy()
        raw = self.source.action(state, laptop_rgb, phone_rgb).copy()
        self.native_index = self.native_count
        self.native_count += 1
        if self.plan is None and self.interventions < self.timing_config.maximum_interventions:
            # Only inspect the cache at a chunk boundary. Its contents and
            # the offset estimate then stay fixed until the selected command.
            if (self.native_index-self.cache_origin) % self.config['n_action_steps'] == 0:
                pending = self.source.pending_actions()
                forecast = np.vstack([raw, pending])
                physical = self._physical(forecast + self.last_offset)
                allowed = self.grasp_history is None or self.grasp_history.permits(physical)
                candidate = early_lift_plan(physical, self.kinematics, self.timing_config) if allowed else None
                if candidate:
                    hold, close = candidate['hold_index'], candidate['close_index']
                    self.plan = {**candidate, 'hold_native_index': self.native_index + hold,
                                 'close_native_index': self.native_index + close}
                    self.jaw_sequence = forecast[hold + 1:close + 1, 5].copy()
        if self.plan is not None and self.native_index == self.plan['hold_native_index']:
            self.held_raw = raw.copy()
            self.anchor = None
            self.open_executed_jaw = None
            self.last_arm_goal = None
            self.phase = 'arm_settle'
            self.phase_steps, self.streak, self.jaw_index = 0, 0, 0
            self.interventions += 1
        elif self.native_index <= self.catchup_until:
            raw[5] = self.closed_raw_jaw
            self.phase = 'resume_closed'
        else:
            self.phase = 'normal'
        self.last_base = raw.astype(np.float32)
        return self.last_base.copy()

    def constrain_action(self, requested):
        """Hold the combined arm target exactly despite residual drift."""
        action = np.asarray(requested, float).copy()
        if self.phase in ('arm_settle', 'close_jaw', 'closed_settle', 'failed'):
            if self.anchor is None:
                self.anchor = action.copy()
            adapting = (self.phase == 'arm_settle' and
                        self.timing_config.minimum_arm_adaptation_steps > 0 and
                        (not self.timing_config.latch_after_adaptation or
                         self.phase_steps < self.timing_config.minimum_arm_adaptation_steps))
            if adapting:
                self.anchor[:5] = action[:5]
            else:
                action[:5] = self.anchor[:5]
            if self.phase == 'arm_settle':
                if self.open_executed_jaw is None:
                    opening = self.timing_config.alignment_open_jaw_deg
                    self.open_executed_jaw = self.anchor[5] if opening is None else self.home[5] + (opening - self.sim_home[5]) / self.scale[5]
                action[5] = self.open_executed_jaw
            elif self.phase in ('closed_settle', 'failed') and self.closed_executed_jaw is not None:
                action[5] = self.closed_executed_jaw
        self.current_metadata = {'phase': self.phase, 'native_index': self.native_index,
                                 'native_chunk_phase': self.chunk_phase, 'interventions': self.interventions,
                                 'act_base_legacy': self.last_base.tolist(),
                                 'arm_anchor_legacy': self.anchor[:5].tolist() if self.anchor is not None else None,
                                 'jaw_cursor': self.jaw_index if self.phase == 'close_jaw' else None,
                                 'phase_step': self.phase_steps + 1, 'plan': self.plan,
                                 'failure': self.failure, 'timing_config': self.timing_config.to_dict()}
        if self.grasp_history is not None:
            self.current_metadata['grasp_history'] = self.grasp_history.state()
        if self.cache_origin:
            self.current_metadata['cache_origin'] = self.cache_origin
        return action, dict(self.current_metadata)

    def observe_execution(self, requested, executed, measured_joint_deg, measured_finger_joint_deg):
        """Consume encoder feedback only, never simulator object state."""
        self.last_offset = np.asarray(requested) - self.last_base
        measured = np.asarray(measured_joint_deg)
        if self.grasp_history is not None:
            self.grasp_history.observe(measured, measured_finger_joint_deg,
                                       self._physical(np.asarray(executed)), self.phase, self.interventions)
        current = {'measured_joint_deg': measured,
                   'measured_finger_joint_deg': dict(measured_finger_joint_deg)}
        self.phase_steps += 1
        if self.phase == 'arm_settle':
            goal = self._physical(self.anchor)
            ready = np.max(np.abs(measured[:5] - goal[:5])) <= self.timing_config.arm_tolerance_deg
            if self.timing_config.minimum_arm_adaptation_steps:
                ready = ready and self.last_arm_goal is not None and np.max(np.abs(goal[:5] - self.last_arm_goal)) <= .25
            fingers = measured_finger_joint_deg
            ready = ready and fingers['jaw1_hinge'] >= 6. and fingers['jaw2_hinge'] <= -6.
            self.streak = self.streak + 1 if ready else 0
            self.last_arm_goal = goal[:5].copy()
            if (self.streak >= self.timing_config.arm_stable_frames and
                    self.phase_steps >= self.timing_config.minimum_arm_adaptation_steps):
                self.phase, self.phase_steps, self.streak = 'close_jaw', 0, 0
            elif self.phase_steps >= self.timing_config.maximum_arm_settle_steps:
                self.failure, self.phase = 'timing_arm_settle_timeout', 'failed'
        elif self.phase == 'close_jaw':
            self.jaw_index += 1
            if self.jaw_index == len(self.jaw_sequence):
                self.closed_raw_jaw = float(self.last_base[5])
                self.closed_executed_jaw = float(executed[5])
                self.phase, self.phase_steps, self.streak = 'closed_settle', 0, 0
        elif self.phase == 'closed_settle':
            ready = self.last_fingers is not None and settled_fingers(current,
                      {'measured_finger_joint_deg': self.last_fingers}, self._physical(self.anchor), self.timing_config.arm_tolerance_deg)
            self.streak = self.streak + 1 if ready else 0
            if (self.phase_steps >= self.timing_config.minimum_closed_settle_steps and
                    self.streak >= self.timing_config.closed_stable_frames):
                self.catchup_until = self.plan['close_native_index']
                self.plan = None
                self.phase, self.phase_steps, self.streak = 'normal', 0, 0
            elif self.phase_steps >= self.timing_config.maximum_closed_settle_steps:
                self.failure, self.phase = 'timing_finger_settle_timeout', 'failed'
        self.last_fingers = dict(measured_finger_joint_deg)
