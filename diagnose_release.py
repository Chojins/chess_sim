"""Offline heights-v4 e2e4 release-position audit; does not advance physics.
Run with ./run_usd_python.sh diagnose_release.py.
The bounded fit checks point position only, not collisions or grasp stability.
"""
import json,numpy as np
from pathlib import Path
from pxr import Usd,Gf
from scipy.optimize import least_squares
from episode_replay import EpisodeReplay
from grasp_geometry import GRASP_LOCAL_M
root=Path('outputs/opening-rl-heights-v4/initial_e2e4/learned_801')
initial=json.loads((root/'initial.json').read_text())['telemetry']['observation'];g=initial['task_geometry'];i=g['source_index']
target=np.array(g['target_center_m'])+np.array(initial['pieces']['piece_grasp_targets_m'][i])-initial['pieces']['piece_centers_m'][i]
s=Usd.Stage.Open('Sim_Stage.usd');s.SetEditTarget(s.GetSessionLayer());fk=EpisodeReplay(s,'outputs/replay/episode_000000.npz','calibration/legacy_blue_joint_map_physics.json','outputs/placement-audit',kinematic=False)
rows=[json.loads(x) for x in (root/'steps.jsonl').read_text().splitlines()];report=[]
for step in [160,190,191,193]:
 r=rows[step-1];applied=np.array(r['learning']['applied_residual']);executed=fk.mapping.to_sim_deg(np.array(r['residual']['executed_legacy']));base=executed-applied*np.array([2,2,2,2,2,3])
 def point(q):
  angles=dict(zip(fk.mapping.sim_names,q));angles['jaw2_hinge']=-q[5]
  return np.array(fk.forward(angles)['/World/so100_blue/gripper'].Transform(Gf.Vec3d(*GRASP_LOCAL_M)))
 def error(delta):return (point(base+np.r_[delta,0])-target)*1000
 fit=least_squares(error,np.zeros(5),bounds=(-np.ones(5)*2,np.ones(5)*2))
 report.append({'step':step,'shoulder_deg':float(executed[1]),'commanded_grasp_point_m':point(executed).tolist(),'target_grasp_point_m':target.tolist(),'commanded_error_mm':float(np.linalg.norm(point(executed)-target)*1000),'best_bounded_position_error_mm':float(np.linalg.norm(fit.fun)),'correction_deg':fit.x.tolist()})
Path('outputs/placement-audit/release.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
