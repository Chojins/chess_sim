import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import numpy as np
from teacher.fit_oracle import load_training
from teacher.grasp_checkpoints import file_hash


class OracleDataTests(unittest.TestCase):
    def archive(self,root,name,train_seed,reserved_seed,split='train'):
        bank=root/f'{name}_bank.json';entry={'name':name,'seed':train_seed,'split':split}
        bank.write_text(json.dumps({'entries':[entry,{'name':name+'_reserved','seed':reserved_seed,'split':'test'}]}))
        manifest=root/f'{name}_manifest.json'
        trial=root/name;trial.mkdir()
        (trial/'initial.json').write_text(json.dumps({'telemetry':{'observation':{'physics_settings':{'contacts':{'robot_self_collisions':False}}}}}))
        manifest.write_text(json.dumps({'bank':str(bank),'bank_sha256':file_hash(bank),
            'policy_action_space':'opening_xyz_pitch_roll_binary_jaw_v1','case':{'move':'e2e4'},
            'action_config':{},'move_config':{},'observation_schema':{},'joint_map':{},'cartesian_config':{}}))
        meta={'training_only':True,'sources':[{'split':split,'seed':train_seed,'reset_name':name,'path':str(trial)}],
            'manifest':str(manifest),'manifest_sha256':file_hash(manifest)}
        path=root/f'{name}.npz'
        np.savez(path,metadata_json=np.array(json.dumps(meta)),observation=np.zeros((1,76),np.float32),
            action=np.zeros((1,6),np.float32),free_arm_before=[True],released_before=[False],
            all_centers_m=np.zeros((1,32,3)),all_upright=np.ones((1,32)),source_index=[0],home_joint_deg=np.zeros((1,6)))
        return path,bank
    def test_training_states_can_aggregate_across_disjoint_banks(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);a,_=self.archive(root,'a',1,2);b,_=self.archive(root,'b',3,4)
            data,sources,manifest=load_training([a,b]);self.assertEqual(data['observation'].shape,(2,76));self.assertEqual(len(sources),2)
    def test_cross_bank_reuse_of_reserved_seed_is_rejected(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);a,_=self.archive(root,'a',1,2);b,_=self.archive(root,'b',2,3)
            with self.assertRaisesRegex(ValueError,'seed leakage'): load_training([a,b])
    def test_heldout_episode_or_changed_bank_is_rejected(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);a,_=self.archive(root,'a',1,2,'validation')
            with self.assertRaises(ValueError): load_training([a])
            b,bank=self.archive(root,'b',3,4);bank.write_text('{}')
            with self.assertRaisesRegex(ValueError,'bank changed'): load_training([b])


if __name__=='__main__': unittest.main()
