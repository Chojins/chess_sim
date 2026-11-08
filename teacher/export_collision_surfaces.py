"""Export authored robot triangles in rigid-link coordinates for CAD gap audits.

Run with run_usd_python.sh. Reads USD only; never changes saved assets or physics.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def main():
    from pxr import Usd,UsdGeom
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--usd',type=Path,default=Path('Sim_Stage.usd'));p.add_argument('--output',type=Path,required=True)
    p.add_argument('--links',nargs='+',default=['gripper','shoulder','upper_arm'])
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    stage=Usd.Stage.Open(str(args.usd));cache=UsdGeom.XformCache();data={};manifest=[]
    for link in args.links:
        body=stage.GetPrimAtPath('/World/so100_blue/'+link)
        if not body:raise ValueError('Missing robot link: '+link)
        for prim in Usd.PrimRange(body,Usd.TraverseInstanceProxies()):
            if not prim.IsA(UsdGeom.Mesh):continue
            mesh=UsdGeom.Mesh(prim);counts=np.asarray(mesh.GetFaceVertexCountsAttr().Get())
            if not np.all(counts==3):raise ValueError('Expected triangulated robot surface')
            vertices=np.asarray(mesh.GetPointsAttr().Get());indices=np.asarray(mesh.GetFaceVertexIndicesAttr().Get())
            matrix=cache.GetLocalToWorldTransform(prim)*cache.GetLocalToWorldTransform(body).GetInverse()
            points=(np.c_[vertices,np.ones(len(vertices))]@np.asarray(matrix))[:,:3]
            name=link+'__'+('visuals' if '/visuals/' in str(prim.GetPath()) else 'collisions')+'__'+str(prim.GetPath()).split('/')[-2]
            if name+'_vertices' in data:raise ValueError('Ambiguous mesh name')
            data[name+'_vertices']=points;data[name+'_faces']=indices.reshape(-1,3)
            manifest.append({'name':name,'path':str(prim.GetPath()),'triangles':len(indices)//3,
                'collision_enabled':prim.GetAttribute('physics:collisionEnabled').Get(),
                'approximation':prim.GetAttribute('physics:approximation').Get()})
    np.savez_compressed(args.output/'meshes.npz',**data)
    (args.output/'meshes.json').write_text(json.dumps(manifest,indent=2))
    layers={str(Path(layer.realPath).resolve()):hashlib.sha256(Path(layer.realPath).read_bytes()).hexdigest()
        for layer in stage.GetUsedLayers() if layer.realPath and Path(layer.realPath).is_file()}
    (args.output/'source_files.json').write_text(json.dumps(layers,indent=2))
    print(json.dumps({'meshes':len(manifest),'source_layers':len(layers),'saved_USD_modified':False}))


if __name__=='__main__':main()
