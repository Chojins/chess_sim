import unittest
import numpy as np
from teacher.student_data import chunk, episode_split, canonical_joint_names,real_episode_split
from teacher.act_student import limited_joint_target


class StudentDataTests(unittest.TestCase):
    def test_real_replay_spreads_over_recordings_without_held_episode_leakage(self):
        numbers=list(range(577));split=real_episode_split(numbers,32,4,'spread')
        self.assertEqual(len(split['train']),32);self.assertEqual(len(split['validation']),4)
        self.assertFalse(set(split['train'])&set(split['validation']))
        self.assertLess(split['train'][0],10);self.assertGreater(split['train'][-1],560)
        self.assertEqual(real_episode_split(numbers,32,4),{'train':numbers[:32],'validation':numbers[-4:]})
        with self.assertRaises(ValueError):real_episode_split([1,1,2],1,1,'spread')

    def test_sim_selection_rejects_improvement_that_exceeds_real_retention_gate(self):
        from teacher.train_student import real_replay_retained
        initial={'real_l1':.25}
        self.assertTrue(real_replay_retained({'real_l1':.27},initial,.1,.01))
        self.assertFalse(real_replay_retained({'real_l1':.32},initial,.1,.01))
        self.assertTrue(real_replay_retained({'real_l1':.32},initial,None,.01))
    def test_only_known_joint_namespace_is_aliased_without_reordering(self):
        self.assertEqual(canonical_joint_names(['main_wrist_roll','main_gripper']),['wrist_roll','gripper'])
        self.assertEqual(canonical_joint_names(['other_gripper','main_elbow_flex']),['other_gripper','elbow_flex'])

    def test_chunks_pad_without_crossing_episode(self):
        actions = np.arange(24, dtype=np.float32).reshape(4, 6)
        targets, mask = chunk(actions, 2, 5)
        np.testing.assert_array_equal(targets, actions[[2,3,3,3,3]])
        np.testing.assert_array_equal(mask, [False,False,True,True,True])
        with self.assertRaises(ValueError): chunk(actions, 4, 5)

    def test_each_move_has_disjoint_whole_episode_validation(self):
        episodes = [{'move':m, 'episode_path':m+str(i)} for m in ('e2e4','d2d4') for i in range(3)]
        split = episode_split(episodes)
        self.assertEqual(len(split['train']),4); self.assertEqual(len(split['validation']),2)
        self.assertFalse({e['episode_path'] for e in split['train']} & {e['episode_path'] for e in split['validation']})
        self.assertEqual({e['move'] for e in split['train']}, {e['move'] for e in split['validation']})
        with self.assertRaises(ValueError): episode_split(episodes[:1])

    def test_slew_advances_arm_and_jaw_by_same_fraction(self):
        mapping = {'sim_home_deg':[0]*6,'scale':[1]*6,'legacy_home':[0]*6}
        target = limited_joint_target([10,0,0,0,0,-20],np.zeros(6),mapping,[-100]*6,[100]*6,[15]*5+[45])
        np.testing.assert_allclose(target,[.5,0,0,0,0,-1])
        # Moving jaw alone still uses its authored velocity, without a grasp gate.
        target = limited_joint_target([0,0,0,0,0,-20],np.zeros(6),mapping,[-100]*6,[100]*6,[15]*5+[45])
        self.assertEqual(target[-1],-1.5)


if __name__ == '__main__': unittest.main()
