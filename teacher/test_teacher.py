"""State-teacher contracts without starting Isaac or using stored recordings."""
import copy
import json
from pathlib import Path
from unittest import TestCase, main
from unittest.mock import Mock

import numpy as np
import requests
from grasp_geometry import GRASP_REFERENCE

from teacher.environment import ASSETS, StateTeacherEnv, TeacherConfig, TeacherFeatures
from teacher.transport import PROTOCOL, StateClient
import test_board_tasks


class FakeClient:
    def __init__(self):
        self.failure = False
        self.sequence = 0

    def reset(self, options):
        self.sequence += 1
        self.actual = test_board_tasks.BoardTaskTests().observation()
        mapping = json.loads(Path('calibration/legacy_blue_joint_map_physics.json').read_text())
        spec = json.loads((ASSETS/'kinematics.json').read_text())
        self.actual.update(measured_joint_deg=mapping['sim_home_deg'],
                           target_legacy_action=mapping['legacy_home'], joint_map=mapping,
                           measured_finger_joint_deg={'jaw1_hinge':0., 'jaw2_hinge':0.},
                           joint_limits_deg=spec['limits'], physics_time=0., physics_settings={},
                           grasp_reference=GRASP_REFERENCE)
        self.actual['pieces']['piece_paths'] = [str(i) for i in range(32)]
        self.actual['pieces']['piece_grasp_targets_m'] = copy.deepcopy(self.actual['pieces']['piece_centers_m'])
        return {'sequence':self.sequence, 'simulation_time':0., 'observation':copy.deepcopy(self.actual)}

    def step(self, action):
        if self.failure:
            raise OSError('Lost response after execution')
        self.sequence += 1
        mapping = self.actual['joint_map']
        physical = np.array(mapping['sim_home_deg'])+np.array(mapping['scale'])*(action-mapping['legacy_home'])
        self.actual['measured_joint_deg'] = physical.tolist()
        self.actual['target_legacy_action'] = np.asarray(action).tolist()
        self.actual['physics_time'] += 1/30
        return {'sequence':self.sequence, 'simulation_time':self.actual['physics_time'],
                'observation':copy.deepcopy(self.actual)}

    def close(self):
        pass


class TeacherTests(TestCase):
    def make_env(self, **config):
        return StateTeacherEnv(client=FakeClient(), config=TeacherConfig(**config))

    def test_action_mapping_speed_limit_and_goal_observation(self):
        env = self.make_env()
        before = env.command.copy()
        obs, _, _, _, info = env.step(np.ones(6, np.float32))
        self.assertEqual(obs.shape, (424,))
        np.testing.assert_allclose(env.command-before, 2.)
        self.assertTrue(env.observation_space.contains(obs))
        mapping = env.mapping
        physical = np.asarray(mapping['sim_home_deg'])+np.asarray(mapping['scale'])*(np.array(info['teacher']['executed_legacy'])-mapping['legacy_home'])
        np.testing.assert_allclose(physical, env.command)
        self.assertFalse(info['teacher']['act_or_capture_supervisor'])
        roles = env.encoder.roles
        self.assertEqual(roles[:,0].sum(),1)
        self.assertEqual(roles[:,1].sum(),1)

    def test_joint_limits_and_abrupt_reversal_penalty(self):
        env = self.make_env()
        positive = env.step(np.ones(6, np.float32))[4]
        negative = env.step(-np.ones(6, np.float32))[4]
        self.assertGreater(negative['teacher']['penalties']['command_acceleration'],
                           positive['teacher']['penalties']['command_acceleration'])
        for _ in range(60):
            _, _, terminated, truncated, _ = env.step(-np.ones(6, np.float32))
            self.assertTrue(np.all(env.command>=env.low))
            self.assertTrue(np.all(env.command<=env.high))
            if terminated or truncated:
                break

    def test_uncertain_step_requires_reset(self):
        env = self.make_env()
        env.client.failure = True
        with self.assertRaises(OSError):
            env.step(np.zeros(6,np.float32))
        with self.assertRaises(RuntimeError):
            env.step(np.zeros(6,np.float32))
        self.assertEqual(env.task.step_count,0)

    def test_timeout_keeps_potential_for_value_bootstrap(self):
        env = self.make_env(max_steps=1)
        previous = env.previous_potential
        _, _, terminated, truncated, info = env.step(np.zeros(6,np.float32))
        self.assertFalse(terminated)
        self.assertTrue(truncated)
        np.testing.assert_allclose(info['teacher']['shaping'],
            env.config.shaping_weight*(env.config.gamma*info['teacher']['potential']-previous))

    def test_piece_order_change_and_invalid_actions_fail(self):
        env = self.make_env()
        for action in [np.ones(5), np.ones(6)*1.1, np.ones(6)*np.nan]:
            with self.assertRaises(ValueError):
                env.step(action)
        env.actual['pieces']['piece_paths'].reverse()
        with self.assertRaises(ValueError):
            env.encoder.encode(env.actual,env.task,env.command,env.velocity,env.previous_q)

    def test_configuration_rejects_invalid_values(self):
        for options in [{'max_steps':0}, {'gamma':float('nan')}, {'jitter_mm':3.},
                        {'max_velocity_deg_s':(1.,)*5}, {'acceleration_weight':-1.}]:
            with self.assertRaises(ValueError):
                TeacherConfig(**options)

    def test_delta_actions_hold_zero_and_invert_physical_targets(self):
        env=self.make_env(action_mode='delta',acceleration_cost_cap=.003)
        command=env.command.copy()
        env.step(np.zeros(6,np.float32))
        np.testing.assert_array_equal(env.command,command)
        target=command+np.array([.4,-.4,.3,-.3,.2,.1])
        action=env.normalized_target(target)
        np.testing.assert_allclose(action,(target-command)/2,atol=1e-7)
        _,_,_,_,info=env.step(action)
        np.testing.assert_allclose(env.command,target,atol=1e-6)
        self.assertLessEqual(info['teacher']['penalties']['command_acceleration'],.003)

    def test_arm_delta_and_gripper_position_share_the_target_speed_limit(self):
        env=self.make_env(action_mode='arm_delta_gripper_position')
        before=env.command.copy()
        target=before+np.array([.1,.2,.3,.4,.5,1.])
        action=env.normalized_target(target)
        env.step(action)
        np.testing.assert_allclose(env.command,target,atol=2e-6)
        action=np.zeros(6,np.float32);action[5]=-1
        before=env.command.copy()
        env.step(action)
        np.testing.assert_allclose(env.command[:5],before[:5])
        self.assertLessEqual(abs(env.command[5]-before[5]),2.000001)

    def test_actual_sac_updates_with_monitor_and_excludes_validation_prior(self):
        import torch
        from stable_baselines3.common.monitor import Monitor
        from teacher.train import PriorSAC
        torch.set_num_threads(1)
        env = self.make_env(max_steps=10)
        monitored = Monitor(env)
        model = PriorSAC('MlpPolicy', monitored, learning_starts=1, batch_size=2,
                         buffer_size=20, policy_kwargs={'net_arch':[16,16]}, device='cpu', seed=9)
        obs, _ = env.reset(seed=9)
        data = {'observations':np.stack([obs,obs+100]), 'actions':np.zeros((2,6),np.float32),
                'validation':np.array([False,True])}
        model.configure_prior(data,.1,9)
        self.assertEqual(len(model.prior_observations),1)
        model.learn(3)
        self.assertEqual(model.replay_buffer.size(),3)
        self.assertEqual(model._n_updates,2)


class TransportTests(TestCase):
    def test_mismatched_revision_and_lost_reply_do_not_retry(self):
        client = StateClient()
        before = {'protocol':PROTOCOL,'sequence':4,'simulation_time':0.}
        client.latest, client.uncertain = before, False
        client.session = Mock()
        client.session.post.return_value.json.return_value = {**before,'sequence':6, 'simulation_time':1/30}
        with self.assertRaises(RuntimeError):
            client.step([0.]*6)
        self.assertTrue(client.uncertain)
        self.assertEqual(client.session.post.call_count,1)
        client.uncertain = False
        client.session.post.side_effect = requests.ConnectionError('lost')
        with self.assertRaises(requests.ConnectionError):
            client.step([0.]*6)
        with self.assertRaises(RuntimeError):
            client.step([0.]*6)
        self.assertEqual(client.session.post.call_count,2)


if __name__ == '__main__':
    main()
