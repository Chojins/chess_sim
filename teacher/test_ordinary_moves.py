import copy
import unittest
import numpy as np
import torch
from teacher.fit_oracle import fresh_warm_actor,relational_warm_actor,geometry_warm_actor
from teacher.geometry_features import GeometryFeatures,relational_values,PostureGeometryFeatures,posture_values
from teacher.bootstrap_move import OfflineSpace
from teacher.grasp_checkpoints import ConservativeSAC
from teacher.grasp_checkpoints import GraspSelection
from pathlib import Path
from tempfile import TemporaryDirectory
from teacher.move_teacher import MoveTask,MoveConfig,task_type_for_objective
from teacher.ordinary_moves import ordinary_case,validate_case_geometry,OrdinaryMoveTask
from teacher.test_resets import observation
from test_board_tasks import layout


class OrdinaryGoalTests(unittest.TestCase):
    def test_posture_warmstart_preserves_actor_and_exposes_full_arm_configuration(self):
        torch.set_num_threads(1)
        parent=ConservativeSAC('MlpPolicy',OfflineSpace(76),seed=5,device='cpu',buffer_size=32,
            policy_kwargs={'net_arch':[16,16],'features_extractor_class':GeometryFeatures,
                'features_extractor_kwargs':{'mean':[0.]*39,'scale':[1.]*39}},verbose=0)
        x=np.random.default_rng(9).normal(0,.1,(8,76)).astype(np.float32)
        result=geometry_warm_actor(parent,torch.tensor(x),6,.997,PostureGeometryFeatures,posture_values)
        self.assertEqual(result.actor.features_extractor.features_dim,57)
        np.testing.assert_allclose(result.predict(x,deterministic=True)[0],parent.predict(x,deterministic=True)[0],atol=1e-6,rtol=0)
        self.assertEqual(result.replay_buffer.size(),0);self.assertEqual(result._n_updates,0)
        self.assertTrue(torch.equal(result.actor.latent_pi[0].weight[:,39:],torch.zeros((16,18))))

    def test_relational_warmstart_adds_distances_without_changing_existing_mean(self):
        torch.set_num_threads(1)
        parent=ConservativeSAC('MlpPolicy',OfflineSpace(76),seed=5,device='cpu',buffer_size=32,
            policy_kwargs={'net_arch':[16,16],'features_extractor_class':GeometryFeatures,
                'features_extractor_kwargs':{'mean':[0.]*39,'scale':[1.]*39}},verbose=0)
        x=np.random.default_rng(9).normal(0,.1,(8,76)).astype(np.float32)
        result=relational_warm_actor(parent,torch.tensor(x),6,.997)
        np.testing.assert_allclose(result.predict(x,deterministic=True)[0],parent.predict(x,deterministic=True)[0],atol=1e-6,rtol=0)
        self.assertEqual(result.actor.features_extractor.features_dim,42)
        self.assertEqual(result.replay_buffer.size(),0);self.assertEqual(result._n_updates,0)
        self.assertTrue(torch.equal(result.actor.latent_pi[0].weight[:,39:],torch.zeros((16,3))))
    def test_relational_distances_have_physical_units_and_no_clock_input(self):
        x=torch.zeros((2,76));x[:,12:15]=torch.tensor([.1,.2,-.1]);x[:,18:20]=torch.tensor([0.,.3])
        x[:,17]=.2;x[:,70]=1;x[1,67]=.9
        y=relational_values(x);torch.testing.assert_close(y[0],y[1])
        np.testing.assert_allclose(y[0,-3:].numpy(),[np.hypot(.003,.006)/.01,.6,.14],atol=1e-6)
    def test_selection_counts_each_goal_and_checks_case_identity(self):
        cases=[ordinary_case([],'e2e4'),ordinary_case([],'d2d4')]
        entries=[{'name':str(i),'family':'home','case':c} for i,c in enumerate(cases)]
        def rows(successes):
            return [{'reset_name':e['name'],'reset_family':'home','case':e['case'],
                'score':{'success':s,'complete_chess_move':s,'objective':'ordinary_move'}} for e,s in zip(entries,successes)]
        with TemporaryDirectory() as tmp:
            file=Path(tmp)/'actor.zip';file.write_bytes(b'actor')
            selection=GraspSelection(tmp,entries,complete_moves=True,objective='ordinary_move',group_by_case=True)
            selection.consider(file,'one_goal',0,rows([True,False]))
            self.assertEqual(selection.best['rank'],[0.,1])
            self.assertTrue(selection.consider(file,'both_goals',1,rows([True,True])))
            bad=rows([True,True]);bad[0]['case']=cases[1]
            with self.assertRaises(ValueError): selection.consider(file,'wrong_goal',2,bad)
    def test_camera_margin_or_robot_contact_disqualifies_success_without_rewriting_task_score(self):
        case=ordinary_case([],'e2e4');entry={'name':'home','family':'home','case':case}
        row={'reset_name':'home','reset_family':'home','case':case,
            'score':{'success':True,'complete_chess_move':True,'objective':'ordinary_move'}}
        with TemporaryDirectory() as tmp:
            checkpoint=Path(tmp)/'actor.zip';checkpoint.write_bytes(b'actor')
            selection=GraspSelection(tmp,[entry],complete_moves=True,objective='ordinary_move',group_by_case=True)
            row['camera_clearance_audit']={'passed':False}
            selection.consider(checkpoint,'too_close',0,[row]);self.assertEqual(selection.best['successes'],0)
            self.assertTrue(row['score']['success'])
            row['camera_clearance_audit']={'passed':True};row['robot_collision_audit']={'contact_free':False}
            self.assertFalse(selection.consider(checkpoint,'contact',1,[row]))
            row['robot_collision_audit']={'contact_free':True}
            self.assertTrue(selection.consider(checkpoint,'clear',2,[row]));self.assertEqual(selection.best['successes'],1)
    def test_RL_actor_warmstart_preserves_mean_with_fresh_critics_and_replay(self):
        torch.set_num_threads(1)
        parent=ConservativeSAC('MlpPolicy',OfflineSpace(76),seed=3,device='cpu',
            buffer_size=32,policy_kwargs={'net_arch':[16,16]},verbose=0)
        parent._n_updates=100
        with torch.no_grad():
            for value in parent.critic.parameters(): value.fill_(5)
        result=fresh_warm_actor(parent,4,.997)
        x=np.random.default_rng(5).normal(size=(4,76)).astype(np.float32)
        np.testing.assert_array_equal(result.predict(x,deterministic=True)[0],parent.predict(x,deterministic=True)[0])
        self.assertEqual(result._n_updates,0);self.assertEqual(result.replay_buffer.size(),0)
        self.assertFalse(result.actor.optimizer.state)
        self.assertTrue(any(not torch.equal(v,parent.critic.state_dict()[k]) for k,v in result.critic.state_dict().items()))
    def test_requested_pawn_and_knight_moves_and_legal_history(self):
        for text in ('e2e4','e2e3','d2d4','c2c4','g1f3'):
            case=ordinary_case([],text);self.assertEqual(case['move'],text)
        self.assertEqual(ordinary_case(['e2e4','e7e5'],'g1f3')['history'],['e2e4','e7e5'])
    def test_illegal_capture_red_and_unsupported_piece_goals_rejected(self):
        for history,text in [([], 'e2e5'),(['e7e5'],'e2e4'),(['e2e4'],'d7d5'),
                (['e2e4','d7d5'],'e4d5'),(['e2e4','e7e5'],'f1c4')]:
            with self.subTest(history=history,move=text),self.assertRaises(ValueError): ordinary_case(history,text)
    def baseline(self,move):
        baseline=observation();centers,geometry=layout().plan([],move)
        baseline['task_geometry']=geometry;baseline['pieces']['piece_centers_m']=centers.tolist()
        baseline['pieces']['piece_grasp_targets_m']=(centers+[0,0,.005]).tolist()
        return baseline
    def test_each_goal_selects_its_piece_and_destination_and_requires_real_lift(self):
        for text in ('e2e3','d2d4','c2c4'):
            baseline=self.baseline(text);i=baseline['task_geometry']['source_index'];actual=copy.deepcopy(baseline)
            point=np.asarray(actual['pieces']['piece_grasp_targets_m'][i]);command=np.zeros(6)
            task=OrdinaryMoveTask(baseline,actual,point,command,{},MoveConfig(horizon=3))
            self.assertEqual(task.index,i);np.testing.assert_array_equal(task.target,baseline['task_geometry']['target_center_m'])
            for _ in range(3): _,done,truncated,score=task.update(actual,point,command)
            self.assertFalse(done);self.assertTrue(truncated);self.assertFalse(score['success']);self.assertEqual(score['objective'],'ordinary_move')
            with self.assertRaises(ValueError): MoveTask(baseline,actual,point,command,{})
    def test_checkpoint_goal_mismatch_is_rejected(self):
        _,geometry=layout().plan([],'d2d4')
        validate_case_geometry(ordinary_case([],'d2d4'),geometry)
        with self.assertRaises(ValueError): validate_case_geometry(ordinary_case([],'e2e4'),geometry)
    def test_e2e4_physical_rules_unchanged_by_objective_version(self):
        baseline=self.baseline('e2e4');actual=copy.deepcopy(baseline);i=baseline['task_geometry']['source_index']
        point=np.asarray(actual['pieces']['piece_grasp_targets_m'][i]);command=np.zeros(6)
        old=MoveTask(baseline,actual,point,command,{});new=OrdinaryMoveTask(baseline,actual,point,command,{})
        a=old.update(actual,point,command);b=new.update(actual,point,command)
        self.assertEqual(a[:3],b[:3]);sa=a[3].copy();sb=b[3].copy()
        for key in ('objective','task_version'):sa.pop(key);sb.pop(key)
        self.assertEqual(sa,sb)
        self.assertIs(task_type_for_objective('unified_e2e4'),MoveTask)
        self.assertIs(task_type_for_objective('ordinary_move'),OrdinaryMoveTask)


if __name__=='__main__': unittest.main()
