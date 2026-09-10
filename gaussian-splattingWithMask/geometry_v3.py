"""Fixed LiDAR references and independently switchable regularizers. CPU-testable."""
import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

def normals_to_quaternions(normals):
    n=np.asarray(normals, dtype=np.float64)
    valid=np.isfinite(n).all(1)&(np.linalg.norm(n,axis=1)>1e-8)
    n=n.copy(); n[~valid]=[0,0,1]; n/=np.linalg.norm(n,axis=1,keepdims=True)
    ref=np.zeros_like(n); ref[:,2]=1; ref[np.abs(n[:,2])>.9]=[1,0,0]
    t=np.cross(ref,n); t/=np.linalg.norm(t,axis=1,keepdims=True)
    matrix=np.stack([t,np.cross(n,t),n],axis=-1)
    # SciPy covers all rotation branches; baseline's uninitialized branch is bypassed.
    xyzw=Rotation.from_matrix(matrix).as_quat()
    return xyzw[:,[3,0,1,2]].astype(np.float32)

def normal_axis(q):
    w,x,y,z=F.normalize(q,dim=-1).unbind(-1)
    return torch.stack([2*(x*z+w*y),2*(y*z-w*x),1-2*(x*x+y*y)],-1)

def build_reference(xyz, camera_centers, a):
    xyz=np.asarray(xyz,dtype=np.float64)
    if len(xyz)<=a.knn or not np.isfinite(xyz).all(): raise ValueError('Invalid/too small cloud')
    tree=cKDTree(xyz); distance,ids=tree.query(xyz,k=a.knn+1,workers=-1)
    raw_h=np.median(distance[:,1:5],axis=1)
    positive=raw_h[raw_h>1e-8]
    if not len(positive): raise ValueError('Point cloud is degenerate')
    median=float(np.median(positive)); h=np.clip(raw_h,median*.5,median*2)
    normal=np.empty_like(xyz); plane=np.empty_like(xyz); confidence=np.zeros(len(xyz))
    for start in range(0,len(xyz),16384):
        end=min(start+16384,len(xyz)); nei=xyz[ids[start:end,1:]]
        mask=distance[start:end,1:]<=a.neighbor_radius_ratio*h[start:end,None]
        count=mask.sum(1); mean=(nei*mask[:,:,None]).sum(1)/np.maximum(count[:,None],1)
        delta=(nei-mean[:,None,:])*mask[:,:,None]
        cov=np.einsum('nki,nkj->nij',delta,delta)/np.maximum(count[:,None,None],1)
        eig,vec=np.linalg.eigh(cov); n=vec[:,:,0]
        conf=(eig[:,1]-eig[:,0])/np.maximum(eig[:,2],1e-12)
        conf[(count<6)|(conf<a.planarity_min)|(raw_h[start:end]>median*3)]=0
        normal[start:end]=n
        # Nearest point on fitted plane; anchor remains original input point.
        plane[start:end]=xyz[start:end]-np.sum((xyz[start:end]-mean)*n,axis=1)[:,None]*n
        confidence[start:end]=conf
    if a.orient_cameras:
        centers=np.asarray(camera_centers)
        if not len(centers): raise ValueError('Camera orientation requested without train cameras')
        # Local camera vote; no visibility claim. Render normals are also view-oriented.
        k=min(8,len(centers)); dist,idx=cKDTree(centers).query(xyz,k=k)
        dist=np.asarray(dist).reshape(len(xyz),k); idx=np.asarray(idx).reshape(len(xyz),k)
        direction=centers[idx]-xyz[:,None,:]
        direction/=np.maximum(np.linalg.norm(direction,axis=-1,keepdims=True),1e-8)
        score=(np.sum(direction*normal[:,None,:],axis=-1)/np.maximum(dist,.1)).sum(1)
        normal[score<0]*=-1
        print('[NORMAL] camera-local vote; orientation is heuristic, not occlusion-tested',flush=True)
    return dict(anchor=xyz.astype('f4'),plane=plane.astype('f4'),normal=normal.astype('f4'),
                spacing=h.astype('f4'),confidence=confidence.astype('f4'),
                source_id=np.arange(len(xyz),dtype=np.int64))

def tensor_reference(reference, device):
    return {k:torch.as_tensor(v,device=device) for k,v in reference.items()}

def measures(xyz,scales,rotation,r):
    delta=xyz-r['anchor']; n=r['normal']; h=r['spacing']
    perpendicular=((xyz-r['plane'])*n).sum(-1)
    tangent=torch.linalg.vector_norm(delta-(delta*n).sum(-1,keepdim=True)*n,dim=-1)
    alignment=(normal_axis(rotation)*n).sum(-1).clamp(-1,1)
    return perpendicular,tangent,alignment,h

def geometry_losses(xyz,scales,rotation,r,a,iteration):
    d,t,c,h=measures(xyz,scales,rotation,r)
    conf=r['confidence']; terms={}; total=xyz.new_zeros(())
    for name in ('surface','tangent','normal','flatten','size'):
        enabled=getattr(a,name+'_loss'); start=getattr(a,name+'_start')
        ramp=min(1.,max(0.,(iteration-start)/max(1,getattr(a,name+'_warmup'))))
        weight=getattr(a,'lambda_'+name)*ramp if enabled and iteration>=start else 0.
        raw=xyz.new_zeros(())
        if weight:
            if name=='surface':
                v=F.smooth_l1_loss(d/(a.surface_tolerance_ratio*h),torch.zeros_like(d),reduction='none')
            elif name=='tangent': v=F.relu(t/(a.tangent_radius_ratio*h)-1).square()
            elif name=='normal': v=1-c.square()
            elif name=='flatten': v=F.relu(scales[:,2]/(a.thickness_ratio*h)-1).square()
            else: v=F.relu(scales[:,:2]/(a.size_ratio*h[:,None])-1).square().mean(-1)
            # Shape limits also apply where LiDAR normals are unreliable.
            w=torch.ones_like(conf) if name in ('flatten','size','tangent') else conf
            raw=(v*w).sum()/w.sum().clamp_min(1e-8)
            total=total+weight*raw
        terms[name]={'raw':float(raw.detach()),'weight':weight,'weighted':float(raw.detach())*weight,
                     'state':'OFF' if not enabled else ('ACTIVE' if weight else 'WAITING_OR_ZERO_WEIGHT')}
    return total,terms

@torch.no_grad()
def diagnostics(xyz,scales,rotation,r,a):
    d,t,c,h=measures(xyz,scales,rotation,r)
    def quantiles(v):
        if not v.numel(): return None
        return torch.quantile(v.float(),v.new_tensor([.5,.95,.99,1.],dtype=torch.float32)).cpu().tolist()
    valid=r['confidence']>0
    return {'count':len(xyz),'reference_valid_fraction':float(valid.float().mean()),
            'abs_plane_distance_all':quantiles(d.abs()),'abs_plane_distance_valid':quantiles(d[valid].abs()),
            'tangent_distance':quantiles(t),'normal_angle_deg_valid':quantiles(torch.rad2deg(torch.acos(c[valid].abs()))),
            'thickness':quantiles(scales[:,2]),'max_tangent_scale':quantiles(scales[:,:2].max(1).values),
            'thickness_exceed_fraction':float((scales[:,2]>a.thickness_ratio*h).float().mean()),
            'size_exceed_fraction':float((scales[:,:2].max(1).values>a.size_ratio*h).float().mean()),
            'spacing':quantiles(h)}
