"""Exact CAD triangle-mesh gaps for the camera housing near joint 2.

Run through run_dataset_python.sh. These queries audit geometric clearance;
PhysX contact reports remain the authority for the executed collision shapes.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from review_grasp_geometry import joint_transforms
from teacher.episode_data import digest


def clearance_reward_contract(meshes,buffer_m=.008,minimum_m=.005,weight=.04):
    """Version a training cost separately from the unchanged acceptance gate."""
    if not np.isfinite([buffer_m,minimum_m,weight]).all() or not 0<=minimum_m<buffer_m or weight<=0:
        raise ValueError('Invalid camera clearance reward buffer or weight')
    meshes=Path(meshes).resolve()
    return {'version':'quadratic_camera_buffer_v1','meshes':str(meshes),'sha256':digest(meshes),
        'buffer_m':float(buffer_m),'minimum_m':float(minimum_m),'weight':float(weight)}


def clearance_buffer_cost(gap_m,contract):
    if not np.isfinite(gap_m) or gap_m<0:raise ValueError('Invalid measured camera gap')
    return contract['weight']*max(0.,(contract['buffer_m']-gap_m)/(contract['buffer_m']-contract['minimum_m']))**2


class ClearanceReward:
    def __init__(self,contract):
        expected=clearance_reward_contract(contract['meshes'],contract['buffer_m'],contract['minimum_m'],contract['weight'])
        if contract!=expected:raise ValueError('Camera reward geometry or version changed')
        self.contract=expected;self.model=CameraClearance(contract['meshes'])

    def evaluate(self,observation):
        gaps=self.model.distances(observation['measured_joint_deg'],observation['measured_finger_joint_deg']['jaw2_hinge'])
        gap=float(gaps.min())
        return {'minimum_gap_m':gap,'cost':clearance_buffer_cost(gap,self.contract),
            'pair_gaps_m':dict(zip(self.model.obstacle_names,gaps.tolist()))}


class CameraClearance:
    obstacle_names=('shoulder__visuals__Rotation_Pitch_Motor','shoulder__visuals__Rotation_Pitch',
        'upper_arm__visuals__Upper_Arm','upper_arm__visuals__Upper_Arm_Motor')
    camera_name='gripper__visuals__Fixed_Jaw_mod_m'

    def __init__(self,meshes,kinematics='teacher/assets/kinematics.json'):
        import fcl
        self.fcl=fcl;self.meshes=Path(meshes);self.spec=json.loads(Path(kinematics).read_text())
        self.geometry={};self.objects={}
        with np.load(meshes,allow_pickle=False) as archive:
            for name in (self.camera_name,*self.obstacle_names):
                vertices=np.ascontiguousarray(archive[name+'_vertices'],dtype=np.float64)
                faces=np.ascontiguousarray(archive[name+'_faces'],dtype=np.int32)
                model=fcl.BVHModel();model.beginModel(len(faces),len(vertices));model.addSubModel(vertices,faces);model.endModel()
                self.geometry[name]=model;self.objects[name]=fcl.CollisionObject(model)
        self.request=fcl.DistanceRequest(enable_nearest_points=True)

    def distances(self,q,jaw2=None):
        q=np.asarray(q,float)
        if q.shape!=(6,) or not np.isfinite(q).all(): raise ValueError('Expected six finite physical joint angles')
        frames=joint_transforms(self.spec,q,jaw2)
        for name,object in self.objects.items():
            matrix=frames['/World/so100_blue/'+name.split('__')[0]]
            object.setTransform(self.fcl.Transform(np.ascontiguousarray(matrix[:3,:3].T),np.ascontiguousarray(matrix[3,:3])))
        values=[]
        for name in self.obstacle_names:
            result=self.fcl.DistanceResult()
            distance=self.fcl.distance(self.objects[self.camera_name],self.objects[name],self.request,result)
            if not np.isfinite(distance): raise ValueError('Invalid triangle-mesh distance')
            values.append(max(0.,float(distance)))
        return np.asarray(values)

    def linearization(self,q,jaw2=None,epsilon_deg=.01):
        baseline=self.distances(q,jaw2)
        jacobian=np.stack([(self.distances(np.asarray(q)+np.eye(6)[j]*epsilon_deg,jaw2)-baseline)/epsilon_deg
            for j in range(5)],axis=1)
        return baseline,jacobian


def audit_camera_trial(trial,meshes,minimum_m=.005,expected_sha256=None,write=True):
    trial=Path(trial);meshes=Path(meshes)
    if expected_sha256 is not None and digest(meshes)!=expected_sha256: raise ValueError('Camera clearance geometry changed')
    if not np.isfinite(minimum_m) or minimum_m<0: raise ValueError('Invalid camera clearance margin')
    model=CameraClearance(meshes);initial=trial/'initial.json';steps=trial/'steps.jsonl'
    observations=[json.loads(initial.read_text())['telemetry']['observation']]
    observations.extend(json.loads(s)['observation'] for s in steps.read_text().splitlines())
    minimum=np.full(4,np.inf);where=np.zeros(4,int)
    for i,o in enumerate(observations):
        value=model.distances(o['measured_joint_deg'],o['measured_finger_joint_deg']['jaw2_hinge'])
        changed=value<minimum;minimum=np.minimum(value,minimum);where[changed]=i
    report={'passed':bool(np.min(minimum)>=minimum_m),'minimum_required_m':minimum_m,'minimum_gap_m':float(min(minimum)),
        'minima':{name:{'gap_m':float(value),'step':int(step)} for name,value,step in zip(model.obstacle_names,minimum,where)},
        'sampled_frames':len(observations),'mesh_sha256':digest(meshes),'steps_sha256':digest(steps),
        'scope':'Four named camera-versus-shoulder/upper-arm CAD surfaces at control frames; no between-frame or other-robot clearance certificate'}
    if write:(trial/'camera_clearance_audit.json').write_text(json.dumps(report,indent=2))
    return report


def append_clearance_audit(path,manifest,report):
    if 'camera_clearance_contract' in manifest:
        contract=manifest['camera_clearance_contract']
        report['camera_clearance_audit']=audit_camera_trial(path,contract['meshes'],contract['minimum_m'],contract['sha256'])
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--meshes',type=Path,required=True);p.add_argument('--trial',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--every',type=int,default=1)
    args=p.parse_args()
    if args.output.exists() or args.every<1: p.error('Need a new output and positive stride')
    model=CameraClearance(args.meshes);initial_path=args.trial/'initial.json';steps=args.trial/'steps.jsonl'
    initial=json.loads(initial_path.read_text())['telemetry']['observation'];rows=[json.loads(s) for s in steps.read_text().splitlines()]
    samples=[]
    for step,o in [(0,initial)]+[(r['step'],r['observation']) for r in rows if r['step']%args.every==0]:
        gaps=model.distances(o['measured_joint_deg'],o['measured_finger_joint_deg']['jaw2_hinge'])
        samples.append({'step':step,'distance_mm':dict(zip(model.obstacle_names,(gaps*1000).tolist()))})
    minima={name:min(({'step':r['step'],'gap_mm':r['distance_mm'][name]} for r in samples),key=lambda x:x['gap_mm']) for name in model.obstacle_names}
    report={'format':'camera_servo_triangle_clearance_v1','minima':minima,'samples':samples,
        'source_hashes':{str(x.resolve()):digest(x) for x in (args.meshes,initial_path,steps)},
        'scope':'Camera/gripper body against the four named shoulder/upper-arm surfaces only; other robot and other links are checked by PhysX contacts, not this metric.',
        'limitations':'Unsigned triangle surface gaps at sampled frames. Does not certify between-frame separation or absence of complete solid containment.'}
    args.output.write_text(json.dumps(report,indent=2));print(json.dumps(minima,indent=2))


if __name__=='__main__':main()
