"""Compare reported collision points with actual CAD triangle surfaces offline.

The contact point belongs to a physics substep; logged joints are end-of-frame.
Small distances therefore do not prove penetration or contact-mesh fidelity.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from review_grasp_geometry import joint_transforms,transform
from teacher.episode_data import digest
from teacher.mesh_distance import point_mesh_distance


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--meshes',type=Path,required=True);p.add_argument('--mesh-manifest',type=Path,required=True)
    p.add_argument('--trial',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--kinematics',type=Path,default=Path('teacher/assets/kinematics.json'))
    p.add_argument('--maximum-samples',type=int,default=80)
    args=p.parse_args()
    if args.output.exists() or args.maximum_samples<1: p.error('Need a new output and positive sample budget')
    metadata=json.loads(args.mesh_manifest.read_text());spec=json.loads(args.kinematics.read_text())
    with np.load(args.meshes,allow_pickle=False) as archive:
        surfaces={m['path']:archive[m['name']+'_vertices'][archive[m['name']+'_faces']] for m in metadata}
    source=args.trial/'steps.jsonl';rows=[json.loads(s) for s in source.read_text().splitlines()]
    selected=[r for r in rows if r['observation'].get('robot_contacts',{}).get('pairs')]
    ids=np.unique(np.linspace(0,len(selected)-1,min(len(selected),args.maximum_samples),dtype=int)) if selected else []
    records=[]
    for i in ids:
        row=selected[i];o=row['observation'];frames=joint_transforms(spec,o['measured_joint_deg'],o['measured_finger_joint_deg']['jaw2_hinge'])
        for pair in o['robot_contacts']['pairs']:
            point=pair.get('deepest_contact_position_m')
            if point is None: continue
            distances={}
            for path in sorted({c for cs in pair['collider_pairs'] for c in cs}):
                if path not in surfaces: continue
                body='/'.join(path.split('/')[:4]);local=transform(np.asarray(point)[None],np.linalg.inv(frames[body]))
                distances[path]={'surface_distance_mm':float(point_mesh_distance(local,surfaces[path])[0]*1000),
                    'body_local_point_mm':(local[0]*1000).tolist()}
            records.append({'step':row['step'],'bodies':[pair['body0'],pair['body1']],
                'solver_separation_mm':pair['minimum_separation_m']*1000,'candidate_collider_distances':distances,
                'multiple_collider_pairs':len(pair['collider_pairs'])>1})
    summary={}
    for path in surfaces:
        values=[r['candidate_collider_distances'][path]['surface_distance_mm'] for r in records if path in r['candidate_collider_distances']]
        if values: summary[path]={'samples':len(values),'distance_mm_percentiles':dict(zip(['min','median','p95','max'],map(float,np.percentile(values,[0,50,95,100]))))}
    report={'format':'robot_contact_triangle_surface_audit_v1','samples':records,'summary':summary,
        'source_hashes':{str(path.resolve()):digest(path) for path in (args.meshes,args.mesh_manifest,source,args.kinematics)},
        'limitations':'Unsigned distances. Substep contact points with end-of-control-frame FK; candidate collider list may include multiple substep contacts. Not an exact mesh/mesh penetration or clearance certificate.'}
    args.output.write_text(json.dumps(report,indent=2));print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
