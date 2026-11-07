"""Tests for live cached ACT timing without simulator object observations."""
from collections import deque
from dataclasses import replace
from types import SimpleNamespace
import unittest
import numpy as np
import torch
from act_client import LegacyACT
from live_act_timing import CloseBeforeLiftACT, TimingConfig, GraspHistory, early_lift_plan, board_pickup_bounds


class ToyKinematics:
    limits = np.tile([-180., 180.], (6, 1))
    def point(self, q):
        return np.array([0, 0, q[2]])


class CachedPolicy:
    config = {'n_action_steps': 8, 'temporal_ensemble_coeff': None}
    def __init__(self, commands):
        self.commands = commands
        self.reset()
    def reset(self):
        self.cursor = 0
    def action(self, *_):
        result = self.commands[min(self.cursor, len(self.commands) - 1)].astype(np.float32).copy()
        self.cursor += 1
        return result
    def pending_actions(self):
        end = ((self.cursor - 1) // 8 + 1) * 8
        return self.commands[self.cursor:end].astype(np.float32).copy()


class LiveTimingTests(unittest.TestCase):
    def test_supervisor_resync_replans_without_losing_release_history(self):
        wrapper=self.wrapper(maximum_interventions=4,suppress_board_release_closures=True,
                             board_pickup_bounds_xy_m=(-.09,-.09,.09,.09),required_tabletop_releases=1)
        wrapper.native_count,wrapper.native_index=13,12
        wrapper.interventions=1
        history=wrapper.grasp_history;history.tabletop_releases=1
        wrapper.resync_after_supervisor()
        wrapper.action(None,None,None)
        self.assertIs(wrapper.grasp_history,history)
        self.assertEqual(history.tabletop_releases,1)
        self.assertEqual(wrapper.interventions,1)
        self.assertEqual(wrapper.native_index,13)
        self.assertEqual(wrapper.chunk_phase,0.)
        self.assertEqual(wrapper.plan['hold_native_index'],16)

    def commands(self):
        commands = np.zeros((8, 6))
        commands[:, 0] = np.arange(8)
        commands[:, 1] = 100
        commands[:, 2] = [.1, .08, .04, .03, .04, .05, .07, .09]
        commands[:, 5] = [8, 9, 10, 10, 8, 4, 0, -1]
        return commands

    def wrapper(self, **config):
        mapping = {'scale': [1] * 6, 'legacy_home': [0] * 6, 'sim_home_deg': [0] * 6}
        defaults = TimingConfig(arm_stable_frames=2, maximum_arm_settle_steps=4,
                                minimum_closed_settle_steps=2, maximum_closed_settle_steps=5,
                                closed_stable_frames=2)
        return CloseBeforeLiftACT(CachedPolicy(self.commands()), mapping, ToyKinematics(), replace(defaults, **config))

    def test_peek_denormalizes_without_depleting_queue(self):
        policy = LegacyACT.__new__(LegacyACT)
        policy.config = {'temporal_ensemble_coeff': None}
        policy.torch = torch
        queued = deque([torch.full((1, 6), .25), torch.full((1, 6), .5)])
        policy.policy = SimpleNamespace(_action_queue=queued)
        policy.action_stats = (torch.ones(6), torch.full((6,), 2.))
        np.testing.assert_allclose(policy.pending_actions(), [[1.5] * 6, [2.] * 6])
        self.assertEqual(len(queued), 2)
        np.testing.assert_allclose(queued[0].numpy(), [[.25] * 6])
        queued.clear()
        self.assertEqual(policy.pending_actions().shape, (0, 6))

    def test_guard_does_not_change_already_correct_close_then_lift_order(self):
        commands = self.commands()
        commands[:, 2] = [.1, .08, .04, .03, .03, .03, .03, .09]
        self.assertIsNone(early_lift_plan(commands, ToyKinematics()))
        commands[:, 5] = 10
        self.assertIsNone(early_lift_plan(commands, ToyKinematics()))

    def test_pause_closes_then_settles_before_consuming_next_arm_command(self):
        wrapper = self.wrapper()
        records = []
        for step in range(20):
            base = wrapper.action(None, None, None)
            requested = base.astype(float).copy()
            requested[0] += .1 * step  # Residual drift must not move the held arm.
            executed, meta = wrapper.constrain_action(requested)
            fingers = {'jaw1_hinge': executed[5], 'jaw2_hinge': -executed[5]}
            wrapper.observe_execution(requested, executed, executed, fingers)
            records.append((executed, meta, wrapper.source.cursor))
            if wrapper.native_index == 7:
                break
        paused = [r for r in records if r[1]['phase'] in ('arm_settle', 'close_jaw', 'closed_settle')]
        self.assertGreater(len(paused), 5)
        self.assertTrue(all(r[2] == 4 for r in paused))
        self.assertTrue(all(r[1]['native_chunk_phase'] == 3 / 8 for r in paused))
        for command, _, _ in paused:
            np.testing.assert_allclose(command[:5], paused[0][0][:5])
        resumed = [r for r in records if r[1]['phase'] == 'resume_closed']
        self.assertTrue(resumed)
        self.assertTrue(all(r[0][5] == 0 for r in resumed))
        self.assertIsNone(wrapper.failure)
        self.assertEqual(wrapper.interventions, 1)
        wrapper.reset()
        self.assertEqual(wrapper.interventions, 0)
        self.assertIsNone(wrapper.anchor)
        self.assertEqual(wrapper.source.cursor, 0)

    def test_arm_timeout_does_not_allow_open_jaw_lift(self):
        wrapper = self.wrapper(maximum_arm_settle_steps=3)
        for _ in range(12):
            raw = wrapper.action(None, None, None)
            executed, meta = wrapper.constrain_action(raw)
            measured = executed.copy()
            if meta['phase'] == 'arm_settle':
                measured[0] += 10
            fingers = {'jaw1_hinge': executed[5], 'jaw2_hinge': -executed[5]}
            wrapper.observe_execution(raw, executed, measured, fingers)
            if wrapper.failure:
                break
        self.assertEqual(wrapper.failure, 'timing_arm_settle_timeout')
        self.assertEqual(wrapper.source.cursor, 4)
        self.assertEqual(wrapper.phase, 'failed')

    def test_finger_timeout_accepts_no_further_arm_actions(self):
        wrapper = self.wrapper(maximum_closed_settle_steps=3)
        for _ in range(20):
            raw = wrapper.action(None, None, None)
            executed, meta = wrapper.constrain_action(raw)
            angle = 10 if meta['phase'] == 'closed_settle' else executed[5]
            fingers = {'jaw1_hinge': angle, 'jaw2_hinge': -angle}
            wrapper.observe_execution(raw, executed, executed, fingers)
            if wrapper.failure:
                break
        self.assertEqual(wrapper.failure, 'timing_finger_settle_timeout')
        self.assertEqual(wrapper.source.cursor, 4)

    def test_adaptation_can_correct_arm_pose_before_freezing_for_closure(self):
        wrapper = self.wrapper(minimum_arm_adaptation_steps=4, maximum_arm_settle_steps=12,
                               alignment_open_jaw_deg=10.)
        alignment_targets, closed_targets = [], []
        for step in range(30):
            base = wrapper.action(None, None, None)
            requested = base.astype(float).copy()
            requested[0] += min(step, 6) * .1
            command, metadata = wrapper.constrain_action(requested)
            fingers = {'jaw1_hinge': command[5], 'jaw2_hinge': -command[5]}
            if metadata['phase'] == 'arm_settle':
                alignment_targets.append(command.copy())
                self.assertEqual(command[5], 10.)
            elif metadata['phase'] in ('close_jaw', 'closed_settle'):
                closed_targets.append(command.copy())
            wrapper.observe_execution(requested, command, command, fingers)
            if wrapper.native_index == 7:
                break
        self.assertGreaterEqual(len(alignment_targets), 4)
        self.assertGreater(alignment_targets[-1][0], alignment_targets[0][0])
        self.assertTrue(closed_targets)
        for command in closed_targets:
            np.testing.assert_allclose(command[:5], alignment_targets[-1][:5])
        self.assertIsNone(wrapper.failure)

    def test_latched_adaptation_settles_despite_continuing_exploration(self):
        from verify_live_timing import TimingAudit
        wrapper = self.wrapper(minimum_arm_adaptation_steps=4, maximum_arm_settle_steps=12,
                               alignment_open_jaw_deg=10., latch_after_adaptation=True)
        audit = TimingAudit(wrapper.mapping, np.ones(6), wrapper.timing_config.to_dict(), window=8)
        held = []
        for step in range(30):
            base = wrapper.action(None, None, None)
            requested = base.astype(float).copy()
            requested[0] += step  # Never stops changing, even after adaptation.
            command, metadata = wrapper.constrain_action(requested)
            audit.check({'act_timing': metadata,
                         'learning': {'applied_residual': (requested-base).tolist()},
                         'residual': {'requested_legacy': requested.tolist(), 'timed_legacy': command.tolist()},
                         'score': {'reason': 'running'}})
            fingers = {'jaw1_hinge': command[5], 'jaw2_hinge': -command[5]}
            if metadata['phase'] == 'arm_settle' and metadata['phase_step'] <= 4:
                self.assertEqual(command[0], requested[0])
            if metadata['phase'] in ('close_jaw', 'closed_settle') or (
                    metadata['phase'] == 'arm_settle' and metadata['phase_step'] >= 4):
                held.append(command[:5].copy())
            wrapper.observe_execution(requested, command, command, fingers)
            if wrapper.native_index == 7:
                break
        self.assertGreater(len(held), 5)
        for command in held:
            np.testing.assert_equal(command, held[0])
        self.assertIsNone(wrapper.failure)
        self.assertEqual(wrapper.interventions, 1)

    def test_disabled_latch_preserves_old_manifest_serialization(self):
        self.assertNotIn('latch_after_adaptation', TimingConfig().to_dict())
        self.assertTrue(TimingConfig(latch_after_adaptation=True).to_dict()['latch_after_adaptation'])
        self.assertNotIn('board_pickup_bounds_xy_m', TimingConfig().to_dict())
        self.assertNotIn('suppress_board_release_closures', TimingConfig().to_dict())

    def test_release_history_distinguishes_transport_release_from_next_pickup(self):
        class Kinematics(ToyKinematics):
            def point(self, q):
                return np.array([q[0], 0., q[2]])
        config = TimingConfig(suppress_board_release_closures=True,
                              board_pickup_bounds_xy_m=(-.09, -.09, .09, .09))
        history = GraspHistory(Kinematics(), config)
        q = np.array([0., 100., .03, 0., 0., 10.])
        history.observe(q, {'jaw1_hinge': 10., 'jaw2_hinge': -10.}, q, 'arm_settle', 1)
        closed = q.copy(); closed[5] = -2.5
        # Unobstructed closure and asymmetric finger contact are not a grasp.
        for fingers in ({'jaw1_hinge': -2.5, 'jaw2_hinge': 2.5},
                        {'jaw1_hinge': 2.4, 'jaw2_hinge': 2.5}):
            for _ in range(4):
                history.observe(q, fingers, closed, 'closed_settle', 1)
            self.assertFalse(history.holding)
        contact = {'jaw1_hinge': 2.4, 'jaw2_hinge': -2.4}
        for _ in range(3):
            history.observe(q, contact, closed, 'closed_settle', 1)
        self.assertTrue(history.holding)
        q[2] = .05
        history.observe(q, contact, closed, 'normal', 1)
        self.assertTrue(history.lifted)
        # Forecast opens on the board before closing empty: do not freeze it.
        self.assertFalse(history.permits(self.commands()))
        # A tabletop release followed by return to the board can be a pickup.
        forecast = self.commands(); forecast[:6, 0] = .13
        self.assertTrue(history.permits(forecast))
        q[0] = .13
        history.observe(q, {'jaw1_hinge': 8., 'jaw2_hinge': -8.}, q, 'normal', 1)
        self.assertFalse(history.holding)
        self.assertFalse(history.board_release)
        self.assertEqual(history.releases, 1)
        # The attacker's second, on-board release completes the guarded sequence.
        q[0], q[2] = 0., .03
        history.observe(q, {'jaw1_hinge': 10., 'jaw2_hinge': -10.}, q, 'arm_settle', 2)
        for _ in range(3):
            history.observe(q, contact, closed, 'closed_settle', 2)
        q[2] = .05
        history.observe(q, contact, closed, 'normal', 2)
        history.observe(q, {'jaw1_hinge': 8., 'jaw2_hinge': -8.}, q, 'normal', 2)
        self.assertTrue(history.board_release)
        self.assertFalse(history.permits(forecast))
        self.assertEqual(history.releases, 2)

    def test_tabletop_release_is_not_regripped_by_board_pickup_guard(self):
        class TabletopKinematics(ToyKinematics):
            def point(self, q):
                return np.array([.13, 0., q[2]])
        config = TimingConfig(guard_all_closures=True,
                              board_pickup_bounds_xy_m=(-.09, -.09, .09, .09))
        self.assertIsNotNone(early_lift_plan(self.commands(), TabletopKinematics()))
        self.assertIsNone(early_lift_plan(self.commands(), TabletopKinematics(), config))
        self.assertEqual(early_lift_plan(self.commands(), ToyKinematics(), config)['hold_index'], 3)
        wrapper = CloseBeforeLiftACT(CachedPolicy(self.commands()),
                                      {'scale': [1]*6, 'legacy_home': [0]*6, 'sim_home_deg': [0]*6},
                                      TabletopKinematics(), config)
        for _ in range(8):
            raw = wrapper.action(None, None, None)
            command, metadata = wrapper.constrain_action(raw)
            np.testing.assert_equal(command, raw)
            self.assertEqual(metadata['phase'], 'normal')
            wrapper.observe_execution(raw, command, command,
                                      {'jaw1_hinge': command[5], 'jaw2_hinge': -command[5]})
        self.assertEqual(wrapper.interventions, 0)
        self.assertEqual(wrapper.source.cursor, 8)

    def test_premature_on_board_victim_release_does_not_disable_capture_retries(self):
        config = TimingConfig(suppress_board_release_closures=True, required_tabletop_releases=1,
                              board_pickup_bounds_xy_m=(-.09, -.09, .09, .09))
        history = GraspHistory(ToyKinematics(), config)
        q = np.array([0., 100., .03, 0., 0., 10.])
        history.observe(q, {'jaw1_hinge': 10., 'jaw2_hinge': -10.}, q, 'arm_settle', 1)
        closed = q.copy(); closed[5] = -2.5
        for _ in range(3):
            history.observe(q, {'jaw1_hinge': 2.4, 'jaw2_hinge': -2.4}, closed, 'closed_settle', 1)
        q[2] = .05
        history.observe(q, {'jaw1_hinge': 2.4, 'jaw2_hinge': -2.4}, closed, 'normal', 1)
        history.observe(q, {'jaw1_hinge': 8., 'jaw2_hinge': -8.}, q, 'normal', 1)
        self.assertFalse(history.board_release)
        self.assertEqual(history.tabletop_releases, 0)
        self.assertTrue(history.permits(self.commands()))

    def test_moving_fingers_are_not_contact_and_lost_grasp_clears_history(self):
        config = TimingConfig(suppress_board_release_closures=True, confirm_settled_contact=True,
                              board_pickup_bounds_xy_m=(-.09, -.09, .09, .09))
        history = GraspHistory(ToyKinematics(), config)
        q = np.array([0., 100., .03, 0., 0., 10.])
        history.observe(q, {'jaw1_hinge': 10., 'jaw2_hinge': -10.}, q, 'arm_settle', 1)
        closed = q.copy(); closed[5] = -2.5
        for angle in [8., 6., 4., 2.4]:
            history.observe(q, {'jaw1_hinge': angle, 'jaw2_hinge': -angle}, closed, 'closed_settle', 1)
            self.assertFalse(history.holding)
        for _ in range(3):
            history.observe(q, {'jaw1_hinge': 2.4, 'jaw2_hinge': -2.4}, closed, 'closed_settle', 1)
        self.assertTrue(history.holding)
        q[2] = .05
        for _ in range(3):
            history.observe(q, {'jaw1_hinge': -2.5, 'jaw2_hinge': 2.5}, closed, 'normal', 1)
        self.assertFalse(history.holding)
        self.assertFalse(history.lifted)

    def test_board_crossing_release_cannot_use_an_earlier_on_board_hold(self):
        class CrossingKinematics(ToyKinematics):
            def point(self, q):
                return np.array([q[0]*.025, 0., q[2]])
        config = TimingConfig(guard_all_closures=True,
                              board_pickup_bounds_xy_m=(-.09, -.09, .09, .09))
        self.assertIsNone(early_lift_plan(self.commands(), CrossingKinematics(), config))

    def test_pickup_area_comes_from_static_square_centres(self):
        squares = [[x*.0225, y*.0225, .014] for x in range(8) for y in range(8)]
        np.testing.assert_allclose(board_pickup_bounds(squares), [-.01125, -.01125, .16875, .16875])
        with self.assertRaises(ValueError):
            board_pickup_bounds([[0, 0, 0]] * 64)


if __name__ == '__main__':
    unittest.main()
