"""Exact unsigned point/triangle-surface distances for offline contact audits."""
import numpy as np


def point_mesh_distance(points,triangles):
    points=np.asarray(points,dtype=float);triangles=np.asarray(triangles,dtype=float)
    if (points.ndim!=2 or points.shape[1]!=3 or triangles.ndim!=3 or triangles.shape[1:]!=(3,3)
            or not len(triangles) or not np.isfinite(points).all() or not np.isfinite(triangles).all()):
        raise ValueError('Expected finite points and a nonempty triangle surface')
    a,b,c=triangles.transpose(1,0,2);ab=b-a;ac=c-a
    normal=np.cross(ab,ac);norm2=np.einsum('ij,ij->i',normal,normal)
    aa=np.einsum('ij,ij->i',ab,ab);bb=np.einsum('ij,ij->i',ac,ac);cc=np.einsum('ij,ij->i',ab,ac)
    denominator=aa*bb-cc*cc;valid=denominator>1e-24;safe=np.where(valid,denominator,1)
    result=[]
    for start in range(0,len(points),16):
        p=points[start:start+16,None,:];ap=p-a
        u=np.einsum('mij,ij->mi',ap,ab);v=np.einsum('mij,ij->mi',ap,ac)
        s=(bb*u-cc*v)/safe;t=(aa*v-cc*u)/safe
        inside=valid & (s>=0) & (t>=0) & (s+t<=1)
        plane=np.einsum('mij,ij->mi',ap,normal)**2/np.where(norm2>0,norm2,1)
        distance=np.where(inside,plane,np.inf)
        for x,y in ((a,b),(b,c),(c,a)):
            edge=y-x;length=np.einsum('ij,ij->i',edge,edge)
            alpha=np.clip(np.einsum('mij,ij->mi',p-x,edge)/np.where(length>0,length,1),0,1)
            delta=p-(x+alpha[:,:,None]*edge)
            distance=np.minimum(distance,np.einsum('mij,mij->mi',delta,delta))
        result.extend(np.sqrt(distance.min(axis=1)))
    return np.asarray(result)
