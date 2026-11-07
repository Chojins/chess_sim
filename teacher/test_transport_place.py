"""Placement needs a real held start, on-square opening and an upright release."""
import copy
import unittest
import numpy as np

from teacher.test_resets import observation
from teacher.transport_place import TransportPlaceTask,PlaceConfig


class PlaceTests(unittest.TestCase):
    def setUp(self):
        self.baseline=observation();self.actual=copy.deepcopy(self.baseline)
        self.index=self.baseline['task_geometry']['source_index']
        self.command=np.array([0.,0,0,0,0,-3])
        self.history={'prior_lift_verified':True,'prior_max_lift_m':.015}
        self.actual['measured_finger_joint_deg']={'jaw1_hinge':2.,'jaw2_hinge':-2.}
        self.move(np.asarray(self.baseline['pieces']['piece_centers_m'][self.index])+[0,0,.015])
    def move(self,center):
        self.actual['pieces']['piece_centers_m'][self.index]=np.asarray(center).tolist()
        self.actual['pieces']['piece_grasp_targets_m'][self.index]=(np.asarray(center)+[0,0,.005]).tolist()
    def point(self):
        return np.asarray(self.actual['pieces']['piece_grasp_targets_m'][self.index])
    def task(self):
        return TransportPlaceTask(self.baseline,self.actual,self.point(),self.command,self.history,PlaceConfig(stable_steps=3))
    def open(self,task,point=None):
        self.command[5]=6
        self.actual['measured_finger_joint_deg']={'jaw1_hinge':5.,'jaw2_hinge':-5.}
        return task.update(self.actual,self.point() if point is None else point,self.command)
    def test_held_start_and_verified_history_are_required(self):
        self.history['prior_lift_verified']=False
        with self.assertRaises(ValueError): self.task()
        self.history['prior_lift_verified']=True
        self.actual['measured_finger_joint_deg']={'jaw1_hinge':-3.,'jaw2_hinge':3.}
        with self.assertRaises(ValueError): self.task()
    def test_release_at_wrong_square_fails_before_magnetic_centering(self):
        task=self.task();target=np.asarray(task.target)
        self.move(target+[0,-.03,.002])
        task.update(self.actual,self.point(),self.command)
        self.command[5]=6
        first=task.update(self.actual,self.point(),self.command)
        self.assertFalse(first[1])
        self.move(target)
        result=self.open(task)
        self.assertTrue(result[1]);self.assertFalse(result[3]['success'])
        self.assertEqual(result[3]['reason'],'release_outside_target_or_too_high')
    def test_high_release_fails(self):
        task=self.task();self.move(task.target+[0,0,.015])
        task.update(self.actual,self.point(),self.command)
        result=self.open(task)
        self.assertTrue(result[1]);self.assertFalse(result[3]['success'])
    def test_upright_on_target_release_and_clearance_pass(self):
        task=self.task();self.move(task.target)
        task.update(self.actual,self.point(),self.command)
        self.assertFalse(self.open(task)[1])
        for _ in range(3):
            result=task.update(self.actual,self.point()+[0,0,.04],self.command)
        self.assertTrue(result[3]['success']);self.assertFalse(result[3]['complete_chess_move'])
    def test_closed_gripper_on_target_does_not_pass(self):
        task=self.task();self.move(task.target)
        for _ in range(5):
            result=task.update(self.actual,self.point(),self.command)
            self.assertFalse(result[3]['success']);self.assertFalse(result[3]['released_on_target'])
    def test_aborted_opening_can_be_retried(self):
        task=self.task();self.move(task.target);task.update(self.actual,self.point(),self.command)
        self.command[5]=3
        task.update(self.actual,self.point(),self.command)
        self.command[5]=-3;task.update(self.actual,self.point(),self.command)
        self.assertIsNone(task.pending_release)
        self.assertFalse(self.open(task)[1])
    def test_movement_during_first_opening_step_cannot_hide_wrong_square(self):
        task=self.task();before=self.actual['pieces']['piece_centers_m'][self.index].copy()
        self.move(task.target)
        result=self.open(task)
        self.assertEqual(result[3]['release_intent']['center_m'],before)
        self.assertEqual(result[3]['reason'],'release_outside_target_or_too_high')
    def test_neighbours_may_shift_inside_their_square(self):
        task=self.task();other=(self.index+1)%32
        self.actual['pieces']['piece_centers_m'][other][0]+=.004
        result=task.update(self.actual,self.point(),self.command)
        self.assertFalse(result[1])
        self.actual['pieces']['piece_centers_m'][other][0]+=.03
        self.assertEqual(task.update(self.actual,self.point(),self.command)[3]['reason'],'disturbed_other_piece')


if __name__=='__main__': unittest.main()
