"""Reconstruct Cartesian control, then independently audit the physical joint trace."""
from dataclasses import asdict
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
from teacher.environment import ASSETS
from teacher.task_space import CartesianControl,CartesianConfig
from teacher.verify_move import audit_trial as audit_joints


def audit_trial(path,manifest):
    path=Path(path);initial=json.loads((path/'initial.json').read_text());actual=initial['telemetry']['observation']
    mapping=actual['joint_map'];limits=np.array(actual['joint_limits_deg']);low,high=limits[:,0]+1e-4,limits[:,1]-1e-4
    current=np.clip(np.array(mapping['sim_home_deg'])+np.array(mapping['scale'])*(np.array(actual['target_legacy_action'])-mapping['legacy_home']),low,high)
    control=CartesianControl(json.loads((ASSETS/'kinematics.json').read_text()),
        json.loads((ASSETS/'finger_reference.json').read_text())['anchors_jaw_local_m'],low,high,
        manifest['action_config']['max_velocity_deg_s'],CartesianConfig(**manifest['cartesian_config']))
    rows=[]
    for line in (path/'steps.jsonl').read_text().splitlines():
        row=json.loads(line);target=control.command(current,actual,row['action'])
        np.testing.assert_allclose(target,row['teacher']['command_joint_deg'],atol=1e-5)
        joint=np.clip((target-current)/control.max_step,-1,1).astype(np.float32)
        np.testing.assert_allclose(joint,row['teacher']['executed_joint_action'],atol=1e-5)
        # This temporary derived view exposes the final joint commands to the
        # existing independent goal/reward auditor. Original data is untouched.
        row['action']=row['teacher']['executed_joint_action'];rows.append(row)
        current=np.asarray(row['teacher']['command_joint_deg']);actual=row['observation']
    with TemporaryDirectory() as temporary:
        derived=Path(temporary)
        for name in ('initial.json','result.json'): (derived/name).write_bytes((path/name).read_bytes())
        (derived/'steps.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
        report=audit_joints(derived,manifest)
    report.update(path=str(path.resolve()),cartesian_control_verified=True)
    return report
