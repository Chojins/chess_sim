"""Contracts for ordinary-move initialization and training-only normalization."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import gymnasium as gym
import numpy as np
import torch

from teacher.imitation import StandardizedFeatures,load_demonstrations,fit_actor
from teacher.reference import audit_reference
from teacher.environment import ASSETS,TeacherConfig


class ImitationTests(unittest.TestCase):
    def test_batched_midpoint_matches_approved_numpy_geometry(self):
        from review_grasp_geometry import opening_center
        from teacher.kinematics import OpeningMidpoint
        spec=json.loads((ASSETS/'kinematics.json').read_text())
        anchors=json.loads((ASSETS/'finger_reference.json').read_text())['anchors_jaw_local_m']
        rng=np.random.default_rng(55)
        limits=np.array(spec['limits'])
        q=rng.uniform(limits[:,0],limits[:,1],(12,6)).astype(np.float32)
        jaw2=-q[:,5]
        actual=OpeningMidpoint(spec,anchors)(torch.from_numpy(q),torch.from_numpy(jaw2)).numpy()
        expected=np.array([opening_center(spec,anchors,values,other) for values,other in zip(q,jaw2)])
        np.testing.assert_allclose(actual,expected,atol=2e-7)

    def test_standardization_is_saved_and_rejects_invalid_scale(self):
        space=gym.spaces.Box(-np.inf,np.inf,(3,),np.float32)
        extractor=StandardizedFeatures(space,[1,2,3],[2,2,2])
        np.testing.assert_allclose(extractor(torch.tensor([[3.,4.,5.]])).numpy(),[[1,1,1]])
        self.assertEqual(set(extractor.state_dict()),{'mean','scale'})
        with self.assertRaises(ValueError):
            StandardizedFeatures(space,[0]*3,[1,0,1])

    def test_validation_split_requires_whole_separate_episodes(self):
        for indices in [[],[0,1],[5]]:
            with self.assertRaises(ValueError):
                load_demonstrations(['unused_a','unused_b'],indices,{},[],TeacherConfig())

    def test_time_clipping_does_not_change_other_state_inputs(self):
        space=gym.spaces.Box(-np.inf,np.inf,(3,),np.float32)
        extractor=StandardizedFeatures(space,[0,0,0],[1,1,1],time_max=.5)
        value=extractor(torch.tensor([[.3,.4,1.2]])).numpy()
        np.testing.assert_allclose(value,[[.3,.4,.5]])

    def test_capture_reference_is_not_accepted_as_ordinary(self):
        with TemporaryDirectory() as tmp:
            initial={'telemetry':{'observation':{'task_geometry':{
                'move':'e4d5','capture_index':24,'history':['e2e4','d7d5']}}}}
            Path(tmp,'initial.json').write_text(json.dumps(initial))
            with self.assertRaises(ValueError):
                audit_reference(tmp)


if __name__=='__main__':
    unittest.main()
