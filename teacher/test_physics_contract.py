import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from teacher.physics_contract import physics_signature,source_physics


class PhysicsContractTests(unittest.TestCase):
    def test_only_initial_settling_diagnostics_are_removed(self):
        a={'contacts':{'robot_self_collisions':False},'magnets':{'energy_j':.001,'initial_settling':{'steps':2}}}
        b={'contacts':{'robot_self_collisions':False},'magnets':{'energy_j':.001,'initial_settling':{'steps':7}}}
        self.assertEqual(physics_signature(a),physics_signature(b))
        b['contacts']['robot_self_collisions']=True
        self.assertNotEqual(physics_signature(a),physics_signature(b))
        self.assertIn('initial_settling',a['magnets'])

    def test_self_disabled_sources_cannot_mix_with_new_collision_world(self):
        with TemporaryDirectory() as tmp:
            path=Path(tmp);settings={'contacts':{'robot_self_collisions':False}}
            (path/'initial.json').write_text(json.dumps({'telemetry':{'observation':{'physics_settings':settings}}}))
            self.assertEqual(source_physics({},[{'path':tmp}]),settings)
            with self.assertRaises(ValueError):source_physics({'physics_settings':{'contacts':{'robot_self_collisions':True}}},[{'path':tmp}])


if __name__=='__main__':unittest.main()
