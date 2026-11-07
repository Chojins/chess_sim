"""Whole-scene checkpoint integrity and strict grasp/lift diagnostic tests."""
import copy
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import numpy as np

from teacher.reset_bank import ResetBank
from teacher.near_grasp import GraspLiftTask,LiftConfig
from test_board_tasks import layout


class Robot:
    dof_names=['a','b','c','d','e','f','g']
    def __init__(self):
        self.q=np.arange(7,dtype=np.float32)[None]*.1
        self.qd=self.q*.01
        self.targets=self.q+.01
        self.vtargets=np.zeros_like(self.q)
    def get_joint_positions(self): return self.q.copy()
    def get_joint_velocities(self): return self.qd.copy()
    def get_applied_actions(self): return SimpleNamespace(joint_positions=self.targets.copy(),joint_velocities=self.vtargets.copy())
    def set_joint_positions(self,v): self.q=v.copy()
    def set_joint_velocities(self,v): self.qd=v.copy()
    def set_joint_position_targets(self,v): self.targets=v.copy()
    def set_joint_velocity_targets(self,v): self.vtargets=v.copy()


class Pieces:
    def __init__(self):
        self.positions=np.arange(96,dtype=np.float32).reshape(32,3)*.001
        self.rotations=np.tile([1.,0,0,0],(32,1)).astype(np.float32)
        self.velocities=np.ones((32,6),np.float32)*.01
    def get_world_poses(self): return self.positions.copy(),self.rotations.copy()
    def get_velocities(self): return self.velocities.copy()
    def set_world_poses(self,p,q): self.positions=p.copy();self.rotations=q.copy()
    def set_velocities(self,v): self.velocities=v.copy()


def control():
    view=Pieces()
    pieces=SimpleNamespace(view=view,paths=[f'piece_{i}' for i in range(32)],
        initial_poses=view.get_world_poses(),initial_centers=view.positions.copy(),maximum_lift=np.zeros(32),active_index=8)
    world=SimpleNamespace(current_time=20.,pause=lambda:None)
    driver=SimpleNamespace(pieces=pieces,robots={'blue':Robot(),'red':Robot()},world=world,tracking=[])
    c=SimpleNamespace(driver=driver,mapping=SimpleNamespace(config={'test':'legacy'}),last_target=np.arange(6,dtype=float),
        task_geometry={'move':'e2e4'},reset_options={'history':[],'move':'e2e4'},step_id=130,episode_id=1,start_time=0.)
    c.observe=lambda:{'physics_settings':{'physics_dt':1/240},'task_geometry':copy.deepcopy(c.task_geometry),
        'physics_time':c.driver.world.current_time-c.start_time,'control_step':c.step_id}
    return c


class ResetTests(unittest.TestCase):
    def test_restore_all_pieces_both_robots_targets_and_original_baseline(self):
        c=control()
        with TemporaryDirectory() as tmp:
            bank=ResetBank(c,tmp,[])
            baseline=c.observe();bank.set_baseline(baseline)
            bank.save('near',{'split':'train'})
            before={color:(r.q.copy(),r.qd.copy(),r.targets.copy()) for color,r in c.driver.robots.items()}
            positions=c.driver.pieces.view.positions.copy()
            for r in c.driver.robots.values(): r.q+=1;r.qd+=2;r.targets+=3
            c.driver.pieces.view.positions+=10
            c.driver.pieces.view.velocities*=5
            c.driver.pieces.initial_centers+=20
            _,context=bank.restore('near')
            np.testing.assert_array_equal(c.driver.pieces.view.positions,positions)
            np.testing.assert_array_equal(c.driver.pieces.initial_centers,positions)
            for color,r in c.driver.robots.items():
                for value,expected in zip((r.q,r.qd,r.targets),before[color]): np.testing.assert_array_equal(value,expected)
            self.assertEqual(context['baseline_observation'],baseline)
            self.assertEqual(max(context['parity'].values()),0)
            self.assertEqual(c.observe()['physics_time'],0)
            self.assertEqual(c.step_id,0)

    def test_bad_name_identity_and_incomplete_arrays_fail_before_mutation(self):
        c=control()
        with TemporaryDirectory() as tmp:
            bank=ResetBank(c,tmp,[]);bank.set_baseline(c.observe())
            with self.assertRaises(ValueError): bank.save('../escape',{})
            bank.save('near',{})
            path=Path(tmp,'near.npz')
            with np.load(path,allow_pickle=False) as archive: arrays={k:archive[k].copy() for k in archive.files}
            del arrays['red_qd']
            np.savez_compressed(path,**arrays)
            before=c.driver.robots['blue'].q.copy()
            with self.assertRaises(ValueError): bank.restore('near')
            np.testing.assert_array_equal(c.driver.robots['blue'].q,before)
            c.mapping.config={'test':'different'}
            with self.assertRaises(ValueError): ResetBank(c,tmp,[])


def observation():
    board=layout();centers,geometry=board.plan([],'e2e4')
    return {'task_geometry':geometry,'measured_joint_deg':[0.]*6,'gripper_tip_proxy_m':[0.,0.,.2],
        'measured_finger_joint_deg':{'jaw1_hinge':5.,'jaw2_hinge':-5.},
        'pieces':{'piece_centers_m':centers.tolist(),'piece_upright_cosines':[1.]*32,
                  'piece_grasp_targets_m':(centers+[0,0,.005]).tolist()}}


class LiftTests(unittest.TestCase):
    def test_requires_real_piece_lift_contact_and_stable_hold(self):
        initial=observation();actual=copy.deepcopy(initial)
        task=GraspLiftTask(initial,actual,LiftConfig(hold_steps=3))
        index=task.index
        point=np.asarray(actual['pieces']['piece_grasp_targets_m'][index])
        command=np.zeros(6);command[5]=-2.5
        self.assertFalse(task.update(actual,point,command)[1])
        actual['pieces']['piece_centers_m'][index][2]+=.012
        actual['pieces']['piece_grasp_targets_m'][index][2]+=.012
        point=np.asarray(actual['pieces']['piece_grasp_targets_m'][index])
        actual['measured_finger_joint_deg']={'jaw1_hinge':2.,'jaw2_hinge':-2.}
        for _ in range(3): self.assertFalse(task.update(actual,point,command)[1])
        _,terminated,truncated,score=task.update(actual,point,command)
        self.assertTrue(terminated and score['success']);self.assertFalse(truncated)
        self.assertFalse(score['complete_chess_move'])

    def test_reset_cannot_inherit_a_pre_lifted_pawn(self):
        initial=observation();actual=copy.deepcopy(initial)
        actual['pieces']['piece_centers_m'][initial['task_geometry']['source_index']][2]+=.012
        with self.assertRaises(ValueError): GraspLiftTask(initial,actual)

    def test_closed_contact_start_does_not_count_as_a_lift(self):
        initial=observation();actual=copy.deepcopy(initial)
        actual['measured_finger_joint_deg']={'jaw1_hinge':2.,'jaw2_hinge':-2.}
        task=GraspLiftTask(initial,actual,LiftConfig(horizon=32,hold_steps=3))
        point=np.asarray(actual['pieces']['piece_grasp_targets_m'][task.index])
        command=np.zeros(6);command[5]=-2.5
        for _ in range(5):
            result=task.update(actual,point,command)
            self.assertFalse(result[3]['success'])
            self.assertEqual(result[3]['hold_streak'],0)
        actual['pieces']['piece_centers_m'][task.index][2]+=.012
        actual['pieces']['piece_grasp_targets_m'][task.index][2]+=.012
        point=np.asarray(actual['pieces']['piece_grasp_targets_m'][task.index])
        for _ in range(3): result=task.update(actual,point,command)
        self.assertTrue(result[3]['success'])

    def test_tool_lift_without_pawn_and_open_jaws_do_not_succeed(self):
        initial=observation();actual=copy.deepcopy(initial)
        task=GraspLiftTask(initial,actual,LiftConfig(horizon=3))
        point=np.asarray(actual['pieces']['piece_grasp_targets_m'][task.index])+[0,0,.02]
        command=np.zeros(6)
        for _ in range(3): result=task.update(actual,point,command)
        self.assertFalse(result[1]);self.assertTrue(result[2]);self.assertFalse(result[3]['success'])


if __name__=='__main__':
    unittest.main()
