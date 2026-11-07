"""Camera/state alignment and the no-physics-advance rendering guard."""
import io
import json
import unittest
from unittest.mock import Mock
import numpy as np
from teacher.rendering import CameraStore,CameraCapture,pack_pair
from teacher.review_policy import matched_images,motion_metrics


class RenderingTests(unittest.TestCase):
    def test_pair_and_state_share_one_revision_and_physics_sample(self):
        obs={'physics_time':1.,'control_step':30,'mode':'policy_control'}
        images={n:np.full((4,4,3),i,dtype=np.uint8) for i,n in enumerate(('red','blue'))}
        capture=Mock();capture.capture.return_value=(pack_pair(images,obs,31),{'red':b'r','blue':b'b'},{'physics_drift_max':0.})
        store=CameraStore(capture);snapshot=store.publish(obs,31,reset_context={'name':'test'})
        with np.load(io.BytesIO(store.pair),allow_pickle=False) as pair:
            self.assertEqual(int(pair['frame_id']),snapshot['sequence'])
            self.assertEqual(json.loads(str(pair['observation_json'])),snapshot['observation'])
        camera=Mock();camera.read.return_value=({'frame_id':31,'simulation_time':1.,'observation':obs},images)
        self.assertEqual(set(matched_images(camera,snapshot)),{'red','blue'})
        camera.read.return_value[0]['frame_id']=30
        with self.assertRaises(RuntimeError): matched_images(camera,snapshot)
    def test_rendering_cannot_advance_physics_or_move_a_body(self):
        capture=CameraCapture.__new__(CameraCapture);capture.physical_state=Mock(return_value=[np.zeros(3),np.zeros(2)])
        self.assertEqual(capture.check_unchanged([np.zeros(3),np.zeros(2)]),0)
        with self.assertRaises(RuntimeError): capture.check_unchanged([np.ones(3)*.001,np.zeros(2)])
    def test_pair_shape_validation_and_motion_units(self):
        with self.assertRaises(ValueError): pack_pair({'red':np.zeros((4,4,3),np.uint8)}, {'physics_time':0.},0)
        rows=[{'observation':{'measured_joint_deg':[i*.1]*6},'teacher':{'command_joint_deg':[i*.2]*6}} for i in range(4)]
        stats=motion_metrics(rows)
        self.assertAlmostEqual(stats['max_measured_arm_speed_deg_s'],3)
        self.assertAlmostEqual(stats['max_arm_target_speed_deg_s'],6)


if __name__=='__main__': unittest.main()
