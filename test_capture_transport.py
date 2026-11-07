"""Check physical limits, upright waypoints and calibration-only construction."""
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
import numpy as np
from capture_demonstrations import JointKinematics
from capture_transport import TransportConfig, CaptureTransportACT, build_route, rotation, settings_from_command, pickup_plan_allowed
from dataclasses import replace


class TransportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path=Path('outputs/capture-supervised-kinematics-v10.json')
        if not path.exists():raise unittest.SkipTest('Exported robot kinematics not installed')
        cls.kin=JointKinematics(json.loads(path.read_text()))
        cls.q=np.array([2.7635,120.0175,-59.3400,3.3677,75.1395,-2.5])
        cls.config=TransportConfig((.0995,-.263),(.2195,-.113),(-.00965,-.17679),.01421)

    def test_route_keeps_payload_upright_and_releases_only_over_table(self):
        height=float(self.kin.point(self.q)[2]-self.config.board_top_m)
        route,up,xy=build_route(self.kin,self.q,height,self.config)
        previous=self.q
        for name,q in route:
            self.assertTrue(np.all(q>=self.kin.limits[:,0]) and np.all(q<=self.kin.limits[:,1]))
            self.assertLessEqual(np.max(np.abs(q[:5]-previous[:5])),self.config.joint_step_deg+1e-7)
            self.assertGreater(float((up@rotation(self.kin.specification,q))[2]),np.cos(np.deg2rad(1.)))
            if name=='release':
                np.testing.assert_allclose(self.kin.point(q)[:2],xy,atol=.0005)
                self.assertAlmostEqual(self.kin.point(q)[2],height+.001,delta=.0005)
            previous=q
        self.assertEqual(sum(name=='release' for name,q in route),1)

    def test_settings_need_no_piece_or_task_truth(self):
        squares=[[x*.0225,y*.0225,.014] for y in range(8) for x in range(8)]
        contact={'square_centers_m':squares,'board_top_m':.014,'board_frame_bounds_m':[[-.01,-.01,0],[.17,.17,.014]]}
        base=SimpleNamespace(latest={'observation':{'physics_settings':{'contacts':contact}}})
        config=settings_from_command(base,{'history':['e2e4','d7d5'],'move':'e4d5'})
        np.testing.assert_allclose(config.pickup_xy_m,[3*.0225,4*.0225])
        self.assertGreater(config.zone_min_xy_m[0],max(p[0] for p in squares))

    def test_invalid_grasp_height_is_rejected_before_motion(self):
        with self.assertRaises(ValueError):build_route(self.kin,self.q,.10,self.config)

    def test_destination_release_is_not_guarded_as_attacker_pickup(self):
        source=np.array([.01285,-.19929])
        actual_pickup={'approach_xy_m':[.016,-.193],'closing_xy_m':[.015,-.196]}
        recorded_false_guard={'approach_xy_m':[-.01999,-.17365],'closing_xy_m':[-.025,-.16]}
        self.assertTrue(pickup_plan_allowed(actual_pickup,source))
        self.assertFalse(pickup_plan_allowed(recorded_false_guard,source))

    def test_rejected_immediate_guard_preserves_cached_action_and_guard_budget(self):
        wrong={'approach_xy_m':[-.020,-.174],'closing_xy_m':[-.025,-.16]}
        base=np.array([1.,100.,-60.,3.,75.,8.])
        class Timed:
            kinematics=self.kin
            grasp_history=object()
            def reset(self):self.interventions=1;self.phase='normal';self.plan=None
            def _physical(self,q):return np.asarray(q)
            def action(self,*args):
                self.plan=wrong;self.phase='arm_settle';self.interventions+=1
                return base.copy()
            def constrain_action(self,requested):return requested,{'phase':self.phase,'plan':self.plan}
        timed=Timed();config=replace(self.config,attacker_xy_m=(.01285,-.19929))
        controller=CaptureTransportACT(timed,config);controller.used=True
        np.testing.assert_equal(controller.action(base,None,None),base)
        self.assertEqual(timed.interventions,1)
        self.assertEqual(timed.phase,'normal')
        self.assertIsNone(timed.plan)
        _,meta=controller.constrain_action(base)
        self.assertEqual(meta['rejected_destination_pickup_plan'],wrong)


if __name__=='__main__':unittest.main()
