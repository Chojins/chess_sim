import copy
import unittest

import chess
import numpy as np

from board_tasks import BoardLayout
from chess_task import LegalMoveTask, TaskConfig


def layout():
    squares = np.array([[f * .0225, r * .0225, .014] for r in range(8) for f in range(8)])
    paths, centers, counts = [], [], {}
    for square, piece in sorted(chess.Board().piece_map().items()):
        color, name = ('blue' if piece.color else 'red'), chess.piece_name(piece.piece_type)
        key = (color, name)
        n = counts.get(key, 0)
        counts[key] = n + 1
        paths.append(f'/World/chess_pieces_{color}/{name}' + (f'_{n:02}' if n else ''))
        centers.append([*squares[square, :2], .026])
    return BoardLayout(paths, centers, squares, .014)


class BoardTaskTests(unittest.TestCase):
    def test_history_capture_identity_and_rhs(self):
        b = layout()
        centers, g = b.plan(['e2e4', 'd7d5'], 'e4d5')
        self.assertEqual(g['source_index'], b.start[chess.E2])
        self.assertEqual(g['capture_index'], b.start[chess.D7])
        np.testing.assert_allclose(centers[g['capture_index'], :2], b.squares[chess.D5, :2])
        self.assertGreater(g['capture_zones']['blue']['min_xy_m'][0], b.squares[chess.H4, 0])
        self.assertLess(g['capture_zones']['red']['max_xy_m'][0], b.squares[chess.A4, 0])
        parked, _ = b.plan(['e2e4', 'd7d5', 'e4d5'], 'd8d5')
        self.assertAlmostEqual(parked[g['capture_index'], 2], .012)

    def test_en_passant_and_invalid_moves(self):
        b = layout()
        _, g = b.plan(['e2e4', 'a7a6', 'e4e5', 'd7d5'], 'e5d6')
        self.assertTrue(g['en_passant'])
        self.assertEqual(g['capture_index'], b.start[chess.D7])
        for history, move in [([], 'e2e5'), (['e7e5'], 'e2e4'), ([], 'e2e2')]:
            with self.assertRaises(ValueError):
                b.plan(history, move)

    def observation(self):
        centers, geometry = layout().plan(['e2e4', 'd7d5'], 'e4d5')
        return {'task_geometry': geometry, 'measured_joint_deg': [0.] * 6,
                'gripper_tip_proxy_m': [0., 0., .2],
                'pieces': {'piece_centers_m': centers.tolist(), 'piece_upright_cosines': [1.] * 32}}

    def finish_attacker(self, task, obs):
        i = obs['task_geometry']['source_index']
        target = np.array(obs['task_geometry']['target_center_m'])
        obs['measured_joint_deg'][0] = 20
        obs['pieces']['piece_centers_m'][i] = (target + [0, 0, .01]).tolist()
        for _ in range(3): task.update(obs)
        obs['pieces']['piece_centers_m'][i] = target.tolist()
        task.update(obs)
        obs['measured_joint_deg'][0] = 0
        for _ in range(10): result = task.update(obs)
        return result

    def test_capture_requires_correct_piece_released_on_table(self):
        obs = self.observation()
        task = LegalMoveTask(copy.deepcopy(obs))
        i = obs['task_geometry']['capture_index']
        zone = obs['task_geometry']['capture_zone']
        obs['measured_joint_deg'][0] = 20
        obs['pieces']['piece_centers_m'][i][2] += .01
        for _ in range(3): task.update(obs)
        xy = (np.array(zone['min_xy_m']) + zone['max_xy_m']) / 2
        obs['pieces']['piece_centers_m'][i] = [*xy, .012]
        for _ in range(11): score = task.update(obs)[3]
        self.assertTrue(score['capture']['released_stable'])
        reward, terminated, _, score = self.finish_attacker(task, obs)
        self.assertTrue(terminated and score['success'])
        self.assertGreater(reward, 9)

    def test_intermediate_capture_home_can_continue_to_complete_attacker(self):
        obs = self.observation()
        task = LegalMoveTask(copy.deepcopy(obs), TaskConfig(capture_home_retry=True))
        i = task.capture_index
        zone = obs['task_geometry']['capture_zone']
        obs['measured_joint_deg'][0] = 20
        obs['pieces']['piece_centers_m'][i][2] += .01
        for _ in range(3): task.update(obs)
        xy = (np.array(zone['min_xy_m']) + zone['max_xy_m']) / 2
        obs['pieces']['piece_centers_m'][i] = [*xy, .012]
        for _ in range(11): task.update(obs)
        obs['measured_joint_deg'][0] = 0
        for _ in range(15):
            _, terminated, truncated, score = task.update(obs)
            self.assertFalse(terminated or truncated or score['success'])
        self.assertTrue(score['home_failure_deferred'])
        _, terminated, _, score = self.finish_attacker(task, obs)
        self.assertTrue(terminated and score['success'])

    def test_capture_retry_keeps_timeout_and_order_failure(self):
        obs = self.observation()
        task = LegalMoveTask(copy.deepcopy(obs), TaskConfig(max_steps=20, capture_home_retry=True))
        obs['measured_joint_deg'][0] = 20
        task.update(obs)
        obs['measured_joint_deg'][0] = 0
        for _ in range(19): result = task.update(obs)
        self.assertFalse(result[1] or result[3]['success'])
        self.assertTrue(result[2])
        self.assertEqual(result[3]['reason'], 'timeout')
        task = LegalMoveTask(copy.deepcopy(obs), TaskConfig(capture_home_retry=True))
        obs['pieces']['piece_centers_m'][task.source_index][0] += .012
        _, terminated, _, score = task.update(obs)
        self.assertTrue(terminated)
        self.assertEqual(score['reason'], 'capture_order_violation')

    def test_capture_home_retry_is_opt_in(self):
        obs = self.observation(); task = LegalMoveTask(copy.deepcopy(obs))
        obs['measured_joint_deg'][0] = 20
        task.update(obs)
        obs['measured_joint_deg'][0] = 0
        for _ in range(10): result = task.update(obs)
        self.assertTrue(result[1])
        self.assertEqual(result[3]['reason'], 'failed_grasp')
        with self.assertRaises(ValueError):
            TaskConfig(capture_home_retry=1)

    def test_toppled_capture_completes_but_upright_earns_more(self):
        outcomes = []
        for upright, z in [(1., .012), (-.2, .0055)]:
            obs = self.observation()
            task = LegalMoveTask(copy.deepcopy(obs))
            i = task.capture_index
            obs['measured_joint_deg'][0] = 20
            obs['pieces']['piece_centers_m'][i][2] += .01
            for _ in range(3): task.update(obs)
            zone = obs['task_geometry']['capture_zone']
            xy = (np.array(zone['min_xy_m']) + zone['max_xy_m']) / 2
            obs['pieces']['piece_centers_m'][i] = [*xy, z]
            obs['pieces']['piece_upright_cosines'][i] = upright
            releases = []
            for _ in range(11):
                score = task.update(obs)[3]
                releases.append(score['reward_terms'].get('capture_release', 0))
            self.assertTrue(score['capture']['released_stable'])
            self.assertAlmostEqual(sum(releases), .25)
            reward, terminated, _, score = self.finish_attacker(task, obs)
            self.assertTrue(terminated and score['success'])
            self.assertEqual(score['capture']['upright_stable'], upright == 1.)
            outcomes.append(reward)
        self.assertAlmostEqual(outcomes[0] - outcomes[1], .75)

    def test_invalid_capture_rest_still_incomplete(self):
        for condition in ['outside', 'high', 'below_table', 'held', 'moving']:
            with self.subTest(condition=condition):
                obs = self.observation(); task = LegalMoveTask(copy.deepcopy(obs)); i = task.capture_index
                obs['measured_joint_deg'][0] = 20
                obs['pieces']['piece_centers_m'][i][2] += .01
                for _ in range(3): task.update(obs)
                zone = obs['task_geometry']['capture_zone']
                xy = (np.array(zone['min_xy_m']) + zone['max_xy_m']) / 2
                obs['pieces']['piece_centers_m'][i] = [*xy, .0055]
                obs['pieces']['piece_upright_cosines'][i] = -.2
                if condition == 'outside': obs['pieces']['piece_centers_m'][i][0] = zone['max_xy_m'][0] + .02
                if condition == 'high': obs['pieces']['piece_centers_m'][i][2] = .05
                if condition == 'below_table': obs['pieces']['piece_centers_m'][i][2] = -.01
                if condition == 'held': obs['gripper_tip_proxy_m'] = obs['pieces']['piece_centers_m'][i].copy()
                for step in range(12):
                    if condition == 'moving': obs['pieces']['piece_centers_m'][i][1] += .001
                    score = task.update(obs)[3]
                self.assertFalse(score['capture']['released_stable'])
                self.assertFalse(task.capture_bonus)

    def test_capture_lift_reward_requires_sustained_lift_and_is_once_only(self):
        obs = self.observation(); task = LegalMoveTask(copy.deepcopy(obs)); i = task.capture_index
        start = obs['pieces']['piece_centers_m'][i][2]
        paid = []
        # Brief bumps reset the streak; only three consecutive lifted frames count.
        for z in [start+.006, start+.006, start, start+.006, start+.006, start+.006,
                  start+.006, start, start+.006, start+.006, start+.006]:
            obs['pieces']['piece_centers_m'][i][2] = z
            reward, _, _, score = task.update(obs)
            self.assertAlmostEqual(reward, sum(score['reward_terms'].values()))
            paid.append(score['reward_terms'].get('capture_lift', 0.))
        self.assertEqual(paid, [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0])
        self.assertFalse(task.capture_bonus)  # lifting alone is not removal

    def test_attacker_only_is_not_capture_success(self):
        obs = self.observation()
        task = LegalMoveTask(copy.deepcopy(obs))
        obs['pieces']['piece_centers_m'][task.source_index][0] += .012
        reward, terminated, _, score = task.update(obs)
        self.assertTrue(terminated)
        self.assertFalse(score['success'])
        self.assertEqual(score['reason'], 'capture_order_violation')
        self.assertLess(reward, 1)

    def test_attacker_nudges_allowed_until_square_boundary(self):
        for offset, fails in [([.008, 0, 0], False), ([.008, .008, 0], False),
                              ([0, 0, .002], False), ([.012, 0, 0], True),
                              ([0, -.012, 0], True)]:
            with self.subTest(offset=offset):
                obs = self.observation()
                task = LegalMoveTask(copy.deepcopy(obs))
                i = task.source_index
                obs['pieces']['piece_centers_m'][i] = (task.initial_centers[i] + offset).tolist()
                _, terminated, _, score = task.update(obs)
                self.assertEqual(terminated, fails)
                self.assertEqual(score['capture']['attacker_inside_original_square'], not fails)
                self.assertEqual(score['reason'], 'capture_order_violation' if fails else None)

    def test_boundary_is_original_square_not_offset_reset_position(self):
        obs = self.observation()
        i = obs['task_geometry']['source_index']
        obs['pieces']['piece_centers_m'][i][0] += .002
        task = LegalMoveTask(copy.deepcopy(obs))
        obs['pieces']['piece_centers_m'][i][0] += .010
        self.assertEqual(task.update(obs)[3]['reason'], 'capture_order_violation')

    def test_wrong_victim_is_disturbance(self):
        obs = self.observation()
        task = LegalMoveTask(copy.deepcopy(obs))
        protected = next(i for i in range(32) if i not in (task.source_index, task.capture_index))
        obs['pieces']['piece_centers_m'][protected][0] += .013
        _, term, _, score = task.update(obs)
        self.assertTrue(term)
        self.assertEqual(score['reason'], 'disturbed_other_piece')

    def test_nudges_inside_square_allowed_but_tip_or_exit_fails(self):
        for shift, upright, fails in [(0.008, 1., False), (.012, 1., True), (-.012, 1., True), (0., .5, True)]:
            with self.subTest(shift=shift, upright=upright):
                obs = self.observation()
                task = LegalMoveTask(copy.deepcopy(obs))
                protected = next(i for i in range(32) if i not in (task.source_index, task.capture_index))
                obs['pieces']['piece_centers_m'][protected][0] += shift
                obs['pieces']['piece_upright_cosines'][protected] = upright
                _, term, _, score = task.update(obs)
                self.assertEqual(term, fails)
                self.assertEqual(score['other_pieces_disturbed'], int(fails))


if __name__ == '__main__':
    unittest.main()
