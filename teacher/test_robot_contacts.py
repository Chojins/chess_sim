import unittest
from types import SimpleNamespace
from teacher.robot_contacts import RobotContacts,robot_pair,audit_collision_trace,contact_costs,collision_cost,contact_terminal


class RobotContactTests(unittest.TestCase):
    def test_contact_terminal_ends_unsafe_episode_without_changing_legacy_default(self):
        o={'physics_settings':{'contacts':{'robot_self_collisions':True}},'robot_contacts':{
            'reporting_enabled':True,'self_contact':True,'inter_robot_contact':False}}
        score={'success':True,'complete_chess_move':True,'reason':'placed_released_home'}
        self.assertEqual(contact_terminal(o,False,10.,True,False,score),(10.,True,False,score))
        value,done,truncated,checked=contact_terminal(o,True,10.,True,False,score)
        self.assertEqual((value,done,truncated),(-2.,True,False));self.assertEqual(checked['reason'],'robot_self_contact')
        self.assertFalse(checked['success']);self.assertTrue(score['success'])

    def test_collision_cost_uses_measured_contact_and_requires_enabled_physics(self):
        costs=contact_costs({'self':.05,'inter_robot':.1})
        self.assertEqual(collision_cost({},contact_costs()),0.)
        observation={'physics_settings':{'contacts':{'robot_self_collisions':True}},
            'robot_contacts':{'reporting_enabled':True,'self_contact':True,'inter_robot_contact':False}}
        self.assertEqual(collision_cost(observation,costs),.05)
        observation['robot_contacts']['inter_robot_contact']=True
        self.assertAlmostEqual(collision_cost(observation,costs),.15)
        observation['physics_settings']['contacts']['robot_self_collisions']=False
        with self.assertRaises(ValueError):collision_cost(observation,costs)
        for value in [{'self':-.1},{'inter_robot':float('nan')},{'unknown':1}]:
            with self.assertRaises(ValueError):contact_costs(value)

    def test_clearance_audit_requires_reports_and_rejects_contact(self):
        initial={'physics_settings':{'contacts':{'robot_self_collisions':False}}}
        self.assertIsNone(audit_collision_trace(initial,[]))
        initial['physics_settings']['contacts']['robot_self_collisions']=True
        with self.assertRaises(ValueError):audit_collision_trace(initial,[{'observation':{}}])
        observation={'robot_contacts':{'reporting_enabled':True,'pairs':[],'self_contact':False,'inter_robot_contact':False}}
        self.assertTrue(audit_collision_trace(initial,[{'observation':observation}])['contact_free'])
        observation['robot_contacts'].update(self_contact=True,pairs=[{'body0':'/World/so100_blue/gripper',
            'body1':'/World/so100_blue/shoulder','contact_samples':1}])
        self.assertFalse(audit_collision_trace(initial,[{'observation':observation}])['contact_free'])

    def test_legitimate_piece_contacts_are_not_robot_collision_flags(self):
        a='/World/so100_blue/gripper'
        self.assertIsNone(robot_pair(a,'/World/chess_pieces_red/pawn_03'))
        self.assertEqual(robot_pair(a,'/World/so100_blue/shoulder'),'self')
        self.assertEqual(robot_pair(a,'/World/so100_red/shoulder'),'inter_robot')
        self.assertIsNone(robot_pair(a,'/World/so100_blue_extra/shoulder'))

    def test_substep_contacts_accumulate_and_control_window_clears(self):
        report=RobotContacts.__new__(RobotContacts);report.decode=lambda p:p;report.clear()
        def header(a,b,n):
            return SimpleNamespace(actor0=a,actor1=b,num_contact_data=n,contact_data_offset=0)
        contact=SimpleNamespace(impulse=[.001,0,0],separation=-.0002)
        robot=header('/World/so100_blue/gripper','/World/so100_blue/shoulder',1)
        piece=header('/World/so100_blue/gripper','/World/chess_pieces_red/pawn_03',1)
        lost=header('/World/so100_blue/gripper','/World/so100_red/shoulder',0)
        report.record([robot,piece,lost],[contact]);report.record([robot],[contact])
        data=report.snapshot();self.assertTrue(data['self_contact']);self.assertFalse(data['inter_robot_contact'])
        self.assertEqual(len(data['pairs']),1);self.assertEqual(data['pairs'][0]['contact_samples'],2)
        self.assertAlmostEqual(data['pairs'][0]['sum_impulse_Ns'],.002)
        self.assertAlmostEqual(data['pairs'][0]['minimum_separation_m'],-.0002)
        report.clear();self.assertFalse(report.snapshot()['self_contact']);self.assertEqual(report.snapshot()['pairs'],[])


if __name__=='__main__':unittest.main()
