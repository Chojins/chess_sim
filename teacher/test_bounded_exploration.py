from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import numpy as np
import torch
from teacher.bootstrap_move import OfflineSpace
from teacher.grasp_checkpoints import ConservativeSAC
from teacher.bounded_exploration import BoundedSACPolicy


class ExplorationTests(unittest.TestCase):
    def test_warm_switch_and_reload_keep_neural_mean_exact_while_limiting_noise(self):
        torch.set_num_threads(1)
        model=ConservativeSAC('MlpPolicy',OfflineSpace(8),device='cpu',seed=12,
            buffer_size=32,policy_kwargs={'net_arch':[16,16]},verbose=0)
        with torch.no_grad(): model.actor.log_std.weight.fill_(3);model.actor.log_std.bias.fill_(8)
        obs=np.random.default_rng(12).normal(size=(20,8)).astype(np.float32)
        expected=model.predict(obs,deterministic=True)[0]
        with TemporaryDirectory() as tmp:
            source=Path(tmp)/'source.zip';target=Path(tmp)/'bounded.zip';model.save(source)
            bounded=ConservativeSAC.load(source,device='cpu',policy_class=BoundedSACPolicy)
            np.testing.assert_array_equal(bounded.predict(obs,deterministic=True)[0],expected)
            with torch.no_grad(): _,log_std,_=bounded.actor.get_action_dist_params(torch.tensor(obs))
            self.assertLessEqual(float(log_std.max()),-2.5);self.assertGreaterEqual(float(log_std.min()),-5.)
            bounded.save(target);restored=ConservativeSAC.load(target,device='cpu')
            self.assertIsInstance(restored.policy,BoundedSACPolicy)
            np.testing.assert_array_equal(restored.predict(obs,deterministic=True)[0],expected)
            with torch.no_grad(): _,loaded_std,_=restored.actor.get_action_dist_params(torch.tensor(obs))
            torch.testing.assert_close(loaded_std,log_std)


if __name__=='__main__': unittest.main()
