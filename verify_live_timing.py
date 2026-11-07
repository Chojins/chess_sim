"""Offline checks of live ACT timing, residual bounds and closure ordering."""
import numpy as np


class TimingAudit:
    def __init__(self, mapping, bounds, settings, window=100, kinematics=None, transport_settings=None):
        self.mapping, self.bounds = mapping, np.asarray(bounds)
        self.settings = settings
        self.window = window
        self.previous = None
        self.interventions = self.paused_steps = 0
        self.kinematics, self.transport_settings = kinematics, transport_settings
        self.transport_route = None
        self.transport_steps = 0
        self.grasp_history = None
        if settings.get('suppress_board_release_closures'):
            from live_act_timing import GraspHistory, TimingConfig
            if kinematics is None:
                raise ValueError('Release-history audit needs the saved robot kinematics')
            self.grasp_history = GraspHistory(kinematics, TimingConfig(**settings))

    def check(self, row):
        timing = row['act_timing']
        if self.transport_settings and 'actor_step_budget' in self.transport_settings['config']:
            assert timing['capture_transport_steps'] == self.transport_steps
            assert timing['actor_step_budget'] == self.transport_settings['config']['actor_step_budget']
            if (self.transport_settings['config'].get('attacker_xy_m') is not None and
                    (timing['capture_transport_steps']>0 or timing.get('grasp_history',{}).get('tabletop_releases',0)>0)):
                assert timing.get('commanded_pickup_xy_m') is not None
        if timing.get('commanded_pickup_xy_m') is not None:
            from capture_transport import pickup_plan_allowed
            target=self.transport_settings['config']['attacker_xy_m']
            np.testing.assert_equal(timing['commanded_pickup_xy_m'],target)
            if timing['plan'] is not None:
                assert pickup_plan_allowed(timing['plan'],np.asarray(target))
            if timing.get('rejected_destination_pickup_plan') is not None:
                assert not pickup_plan_allowed(timing['rejected_destination_pickup_plan'],np.asarray(target))
        if self.grasp_history is not None:
            assert timing['grasp_history'] == self.grasp_history.state()
            after = row['placement_telemetry']
            command = np.asarray(self.mapping['sim_home_deg']) + np.asarray(self.mapping['scale'])*(np.asarray(after['target_legacy_action'])-self.mapping['legacy_home'])
            command = np.clip(command, self.grasp_history.kinematics.limits[:, 0], self.grasp_history.kinematics.limits[:, 1])
            self.grasp_history.observe(after['measured_joint_deg'], after['measured_finger_joint_deg'],
                                       command, timing['phase'], timing['interventions'])
        assert timing['timing_config'] == self.settings
        board_bounds = self.settings.get('board_pickup_bounds_xy_m')
        if board_bounds is not None and timing['plan'] is not None:
            bounds = np.asarray(board_bounds).reshape(2, 2)
            for key in ('approach_xy_m', 'closing_xy_m'):
                point = np.asarray(timing['plan'][key])
                assert np.all(point >= bounds[0]) and np.all(point <= bounds[1])
        base = np.asarray(timing['act_base_legacy'])
        requested = base + np.asarray(row['learning']['applied_residual']) * self.bounds / self.mapping['scale']
        np.testing.assert_allclose(requested, row['residual']['requested_legacy'], atol=2e-5)
        timed = np.asarray(row['residual']['timed_legacy'])
        phase = timing['phase']
        paused = ('arm_settle', 'close_jaw', 'closed_settle', 'failed')
        transport = timing.get('capture_transport')
        if transport:
            if not self.transport_settings or self.kinematics is None:
                raise ValueError('Unconfigured capture transport in trace')
            assert transport['settings'] == self.transport_settings['config']
            if self.transport_route is None:
                from capture_transport import TransportConfig, build_route
                self.transport_route, self.transport_up, self.transport_xy = build_route(self.kinematics,
                    transport['entry_joint_deg'], transport['grasp_height_m'], TransportConfig(**transport['settings']),transport.get('home_joint_deg'))
            self.transport_steps += 1
            name, nominal = self.transport_route[transport['cursor']]
            assert name == transport['phase']
            np.testing.assert_allclose(nominal, transport['nominal_joint_deg'], atol=1e-7)
            np.testing.assert_allclose(self.transport_up, transport['local_up'], atol=1e-7)
            np.testing.assert_allclose(self.transport_xy, transport['drop_xy_m'], atol=1e-7)
            goal = nominal.copy()
            if name == 'release':
                cfg = transport['settings']
                fraction = min(1., transport['phase_step']/cfg['release_steps'])
                goal[5] = cfg['closed_jaw_deg']+(cfg['open_jaw_deg']-cfg['closed_jaw_deg'])*fraction
            compensation = np.asarray(transport['compensation_deg'])
            assert compensation.shape == (5,) and np.max(np.abs(compensation)) <= 3.000001
            goal[:5] += compensation
            goal = np.clip(goal, self.kinematics.limits[:, 0], self.kinematics.limits[:, 1])
            physical = np.asarray(self.mapping['sim_home_deg'])+np.asarray(self.mapping['scale'])*(timed-self.mapping['legacy_home'])
            np.testing.assert_allclose(goal, physical, atol=1e-6)
            np.testing.assert_allclose(goal, transport['command_joint_deg'], atol=1e-7)
        elif phase in paused:
            np.testing.assert_allclose(timed[:5], timing['arm_anchor_legacy'], atol=1e-7)
            self.paused_steps += 1
            if phase == 'arm_settle':
                adapting = (self.settings.get('minimum_arm_adaptation_steps', 0) and
                            (not self.settings.get('latch_after_adaptation', False) or
                             timing['phase_step'] <= self.settings['minimum_arm_adaptation_steps']))
                if adapting:
                    np.testing.assert_allclose(timed[:5], requested[:5], atol=1e-7)
                jaw = self.settings.get('alignment_open_jaw_deg')
                if jaw is not None:
                    physical_jaw = self.mapping['sim_home_deg'][5] + self.mapping['scale'][5] * (timed[5] - self.mapping['legacy_home'][5])
                    np.testing.assert_allclose(physical_jaw, jaw, atol=1e-7)
        else:
            np.testing.assert_allclose(timed, requested, atol=1e-7)
        np.testing.assert_allclose(timing['native_chunk_phase'], (timing['native_index']-timing.get('cache_origin',0)) % self.window / self.window)
        if self.previous:
            previous = self.previous
            assert timing['native_index'] - previous['native_index'] in (0, 1)
            if phase in paused and previous['phase'] in paused:
                assert timing['native_index'] == previous['native_index']
                latched = (self.settings.get('latch_after_adaptation', False) and
                           phase == 'arm_settle' and
                           timing['phase_step'] > self.settings['minimum_arm_adaptation_steps'])
                if phase != 'arm_settle' or latched:
                    np.testing.assert_allclose(timing['arm_anchor_legacy'], previous['arm_anchor_legacy'], atol=1e-7)
            if phase == 'close_jaw' and previous['phase'] == 'arm_settle':
                assert previous['phase_step'] >= self.settings.get('minimum_arm_adaptation_steps', 0)
            if phase == 'resume_closed' and previous['phase'] == 'closed_settle':
                assert previous['phase_step'] >= self.settings['minimum_closed_settle_steps']
        if timing['failure']:
            assert row.get('terminated', row.get('done', False)) and row['score']['reason'] == timing['failure']
        self.interventions = max(self.interventions, timing['interventions'])
        assert self.interventions <= self.settings['maximum_interventions']
        self.previous = timing
