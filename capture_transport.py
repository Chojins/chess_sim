"""Command/calibration/encoder-only victim transport around frozen ACT.

This explicit supervisor is separate from the learned residual. It never reads
piece poses, rewards, success flags or simulator capture-phase flags.
"""
from dataclasses import dataclass, asdict
import numpy as np
from review_grasp_geometry import joint_transforms


@dataclass(frozen=True)
class TransportConfig:
    zone_min_xy_m: tuple
    zone_max_xy_m: tuple
    pickup_xy_m: tuple
    board_top_m: float
    surface_z_m: float = 0.
    transit_z_m: float = .105
    joint_step_deg: float = .8
    closed_jaw_deg: float = -2.5
    open_jaw_deg: float = 10.
    release_steps: int = 36
    settle_steps: int = 10
    settle_timeout: int = 100
    actor_step_budget: int = 800
    return_home: bool = False
    attacker_xy_m: tuple | None = None

    def to_dict(self):
        return {key: list(value) if isinstance(value, tuple) else value for key, value in asdict(self).items()}


def rotation(spec, q):
    return joint_transforms(spec, q)[spec['tool_link']][:3, :3]


def pickup_plan_allowed(plan, pickup_xy, radius=.018):
    """A destination release/closure is not another source-square pickup."""
    return all(np.linalg.norm(np.asarray(plan[key])-pickup_xy) <= radius
               for key in ('approach_xy_m','closing_xy_m'))


def fit_upright(kin, start, target, local_up):
    """Five-axis IK: tool position and payload's initial vertical direction."""
    q = np.asarray(start, float).copy()
    target = np.asarray(target, float)
    def error(values):
        up = local_up @ rotation(kin.specification, values)
        return np.r_[(kin.point(values)-target)*1000., (up-[0., 0., 1.])*60.]
    for _ in range(180):
        current = error(q)
        if np.linalg.norm(current[:3]) < .08 and np.linalg.norm(current[3:]) < .05:
            return q
        jac = np.stack([(error(q+np.eye(6)[i]*.001)-current)/.001 for i in range(5)], axis=1)
        delta = -np.linalg.solve(jac.T@jac+.02*np.eye(5), jac.T@current)
        for fraction in (1., .5, .25, .125):
            candidate = q.copy()
            candidate[:5] = np.clip(q[:5]+np.clip(delta, -2., 2.)*fraction, kin.limits[:5, 0], kin.limits[:5, 1])
            if np.linalg.norm(error(candidate)) < np.linalg.norm(current):
                q = candidate
                break
        else:
            break
    final = error(q)
    if np.linalg.norm(final[:3]) > .5 or np.linalg.norm(final[3:]) > .5:
        raise ValueError(f'Transport waypoint unreachable: position {np.linalg.norm(final[:3]):.2f} mm, up {np.linalg.norm(final[3:]):.2f}')
    return q


def build_route(kin, entry, grasp_height, config, home=None):
    """Continuous lift, RHS placement, open retreat and empty return."""
    if not .005 < grasp_height < .050:
        raise ValueError('Invalid encoder-derived grasp height')
    entry = np.asarray(entry, float).copy()
    entry[5] = config.closed_jaw_deg
    origin = kin.point(entry)
    up = np.array([0., 0., 1.]) @ rotation(kin.specification, entry).T
    lo, hi = np.asarray(config.zone_min_xy_m), np.asarray(config.zone_max_xy_m)
    if np.any(hi <= lo):
        raise ValueError('Invalid calibrated RHS bounds')
    # Current profile handles the first capture; no unknown occupied tabletop
    # slots are guessed. Stay 35 mm beyond the board edge, within the RHS area.
    drop_xy = np.array([lo[0]+min(.035, (hi[0]-lo[0])/2), (lo[1]+hi[1])/2])
    high = max(config.transit_z_m, origin[2])
    places = [('lift', np.r_[origin[:2], high]),
              ('transfer', np.r_[drop_xy, high]),
              ('lower', np.r_[drop_xy, config.surface_z_m+grasp_height+.001]),
              ('retreat', np.r_[drop_xy, high]),
              ('return_high', np.r_[origin[:2], high]),
              ('return', origin)]
    if config.return_home:
        if home is None:
            raise ValueError('Home return needs the measured reset joint pose')
        places = places[:-1]
    route, q, point = [], entry.copy(), origin.copy()
    for name, target in places:
        if name == 'retreat':
            route.append(('release', q.copy()))
            q[5] = config.open_jaw_deg
        segments = max(1, int(np.ceil(np.linalg.norm(target-point)/.010)))
        beginning = point.copy()
        for fraction in np.linspace(0., 1., segments+1)[1:]:
            position = beginning+(target-beginning)*fraction
            fitted = fit_upright(kin, q, position, up)
            steps = max(1, int(np.ceil(np.max(np.abs(fitted[:5]-q[:5]))/config.joint_step_deg)))
            for blend in np.linspace(0., 1., steps+1)[1:]:
                route.append((name, q+(fitted-q)*blend))
            q = fitted
        route.append(('settle_'+name, q.copy()))
        point = target
    if config.return_home:
        goal = np.asarray(home,float).copy();goal[5] = config.open_jaw_deg
        if np.any(goal < kin.limits[:,0]) or np.any(goal > kin.limits[:,1]):
            raise ValueError('Home return target outside limits')
        steps = max(1,int(np.ceil(np.max(np.abs(goal[:5]-q[:5]))/config.joint_step_deg)))
        for fraction in np.linspace(0.,1.,steps+1)[1:]:
            candidate = q+(goal-q)*fraction
            point = kin.point(candidate)
            # Empty return stays clear of the board's tall pieces.
            if -.09 < point[0] < .095 and -.28 < point[1] < -.095 and point[2] < .085:
                raise ValueError('Empty home return crosses below board clearance')
            route.append(('home',candidate))
        route.append(('settle_home',goal))
    return route, up, drop_xy


class CaptureTransportACT:
    def __init__(self, timed, settings):
        if timed.grasp_history is None:
            raise ValueError('Capture transport requires encoder-based grasp history')
        self.timed, self.kin, self.settings = timed, timed.kinematics, settings
        self.reset()

    def __getattr__(self, key):
        return getattr(self.timed, key)

    def reset(self):
        self.timed.reset()
        self.active = self.used = False
        self.transport_steps = 0
        self.route = []
        self.cursor = self.phase_steps = self.streak = 0
        self.grasp_height = None
        self.compensation = np.zeros(5)
        self.last_nominal = self.last_command = None
        self.local_up = None
        self.home_q = None
        self.rejected_pickup_plan = None

    def action(self, *args):
        if self.home_q is None:
            self.home_q = self.timed._physical(args[0])
        if self.active:
            return self.timed.last_base.copy()
        raw = self.timed.action(*args)
        self.rejected_pickup_plan = None
        if self.pickup_filter_active and self.timed.plan is not None:
            if not pickup_plan_allowed(self.timed.plan,np.asarray(self.settings.attacker_xy_m)):
                self.rejected_pickup_plan = self.timed.plan
                self.timed.plan = None
                if self.timed.phase == 'arm_settle':
                    self.timed.interventions -= 1
                    self.timed.phase = 'normal'
                    self.timed.held_raw = self.timed.anchor = None
        return raw

    @property
    def residual_paused(self):
        return self.active

    @property
    def pickup_filter_active(self):
        return (self.settings.attacker_xy_m is not None and
                (self.used or self.timed.grasp_history.tabletop_releases >= 1))

    def actor_clock(self, step, max_steps):
        return max(0,step-self.transport_steps), self.settings.actor_step_budget

    def constrain_action(self, requested):
        executed, meta = self.timed.constrain_action(requested)
        meta['capture_transport_steps'] = self.transport_steps
        meta['actor_step_budget'] = self.settings.actor_step_budget
        if self.pickup_filter_active:
            meta['commanded_pickup_xy_m'] = list(self.settings.attacker_xy_m)
        if self.rejected_pickup_plan is not None:
            meta['rejected_destination_pickup_plan'] = self.rejected_pickup_plan
        if self.active:
            phase, nominal = self.route[self.cursor]
            goal = nominal.copy()
            if phase == 'release':
                blend = min(1., (self.phase_steps+1)/self.settings.release_steps)
                goal[5] = self.settings.closed_jaw_deg+(self.settings.open_jaw_deg-self.settings.closed_jaw_deg)*blend
            goal[:5] += self.compensation
            goal = np.clip(goal, self.kin.limits[:, 0], self.kin.limits[:, 1])
            executed = self.timed.home+(goal-self.timed.sim_home)/self.timed.scale
            self.last_nominal, self.last_command = nominal.copy(), goal.copy()
            meta['capture_transport'] = {'phase': phase, 'cursor': self.cursor, 'phase_step': self.phase_steps+1,
                'nominal_joint_deg': nominal.tolist(), 'compensation_deg': self.compensation.tolist(),
                'command_joint_deg': goal.tolist(), 'local_up': self.local_up.tolist(),
                'entry_joint_deg': self.entry_q.tolist(),
                'home_joint_deg': self.home_q.tolist(),
                'drop_xy_m': self.drop_xy.tolist(), 'grasp_height_m': self.grasp_height,
                'settings': self.settings.to_dict(), 'control_inputs': 'command, static calibration and measured joints only'}
        return executed, meta

    def observe_execution(self, requested, executed, measured, fingers):
        measured = np.asarray(measured, float)
        if not self.active:
            phase = self.timed.phase
            if phase == 'closed_settle' and self.timed.interventions == 1:
                self.grasp_height = float(self.kin.point(measured)[2]-self.settings.board_top_m)
            self.timed.observe_execution(requested, executed, measured, fingers)
            history = self.timed.grasp_history
            if (not self.used and self.timed.interventions == 1 and history.holding and history.lifted
                    and self.timed.phase in ('normal', 'resume_closed') and self.grasp_height is not None):
                if np.linalg.norm(self.kin.point(measured)[:2]-self.settings.pickup_xy_m) > .018:
                    return
                try:
                    self.entry_q = measured.copy()
                    self.route, self.local_up, self.drop_xy = build_route(self.kin, measured, self.grasp_height, self.settings,self.home_q)
                except ValueError as error:
                    self.timed.failure = 'capture_transport_unreachable'
                    self.timed.phase = 'failed'
                    self.planning_error = str(error)
                    return
                self.active = self.used = True
            return
        phase, nominal = self.route[self.cursor]
        self.transport_steps += 1
        self.timed.grasp_history.observe(measured, fingers, self.timed._physical(executed), 'normal', self.timed.interventions)
        self.timed.last_fingers = dict(fingers)
        self.phase_steps += 1
        if phase.startswith('settle_'):
            self.compensation = np.clip(self.compensation+.3*(nominal[:5]-measured[:5]), -3., 3.)
            error = np.linalg.norm(self.kin.point(measured)-self.kin.point(nominal))
            up = self.local_up @ rotation(self.kin.specification, measured)
            ready = error < .002 and (phase == 'settle_home' or up[2] > np.cos(np.deg2rad(2.5)))
            self.streak = self.streak+1 if ready else 0
            advance = self.phase_steps >= self.settings.settle_steps and self.streak >= 3
            if self.phase_steps >= self.settings.settle_timeout:
                self.timed.failure, self.timed.phase = 'capture_transport_settle_timeout', 'failed'
                self.active = False
                return
        elif phase == 'release':
            opened = fingers['jaw1_hinge'] >= 6 and fingers['jaw2_hinge'] <= -6
            self.streak = self.streak+1 if opened else 0
            advance = self.phase_steps >= self.settings.release_steps+10 and self.streak >= 5
        else:
            advance = True
        if advance:
            self.cursor += 1
            self.phase_steps = self.streak = 0
            if self.cursor == len(self.route):
                self.active = False
                self.timed.resync_after_supervisor()


def settings_from_command(base, case):
    import chess
    board = chess.Board()
    for text in case['history']:
        board.push_uci(text)
    move = board.parse_uci(case['move'])
    if not board.is_capture(move) or not board.turn:
        raise ValueError('Transport profile currently supports blue captures only')
    if len(board.pieces(chess.PAWN, chess.BLACK)) + sum(len(board.pieces(kind, chess.BLACK)) for kind in range(2,7)) != 16:
        raise ValueError('First-capture profile cannot assume an unoccupied RHS after earlier captures')
    square = move.to_square - 8 if board.is_en_passant(move) else move.to_square
    observation = base.latest['observation']
    contact = observation['physics_settings']['contacts']
    sites = sorted(np.asarray(contact['square_centers_m']), key=lambda p: p[1])
    squares = [p for start in range(0, 64, 8) for p in sorted(sites[start:start+8], key=lambda p: p[0])]
    bounds = np.asarray(contact['board_frame_bounds_m'])
    middle = float((bounds[0, 1]+bounds[1, 1])/2)
    return TransportConfig((float(bounds[1, 0]), middle-.075),
                           (float(bounds[1, 0]+.120), middle+.075), tuple(squares[square][:2]),
                           contact['board_top_m'], 0.,return_home=True,attacker_xy_m=tuple(squares[move.from_square][:2]))
