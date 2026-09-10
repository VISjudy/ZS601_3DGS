"""Opaque 1-sigma diagnostic, true thickness. Independent of training rasterizer.

CuPy FP64 ray/ellipsoid intersections; nearest positive FP32 depth wins.
Color is SH DC with directional lighting, not the view-dependent training RGB.
"""
import math,time
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

KERNEL=r'''
__device__ double hit(const double* g,double dx,double dy,double* normal){
 double o[3],d[3],a=0,b=0;
 for(int j=0;j<3;j++){
  o[j]=-(g[3+j]*g[0]+g[6+j]*g[1]+g[9+j]*g[2])/g[12+j];
  d[j]=(g[3+j]*dx+g[6+j]*dy+g[9+j])/g[12+j];
  a+=d[j]*d[j];b+=o[j]*d[j];
 }
 double h=-b/a,vv=0;
 for(int j=0;j<3;j++){double v=o[j]+h*d[j];vv+=v*v;}
 if(vv>1)return -1;
 double dt=sqrt(fmax(0.,(1-vv)/a)),t=h-dt;
 if(t<=0.01)t=h+dt;
 if(t<=0.01)return -1;
 if(normal)for(int k=0;k<3;k++){
  normal[k]=0;
  for(int j=0;j<3;j++)normal[k]+=g[3+3*k+j]*(o[j]+t*d[j])/g[12+j];
 }
 return t;
}
extern "C" __global__ void raster(const double* gs,const int* boxes,int n,int w,int h,double fx,double fy,double cx,double cy,unsigned long long* keys){
 int i=blockDim.x*blockIdx.x+threadIdx.x;if(i>=n)return;
 const double* g=gs+15*i;const int* bb=boxes+4*i;
 for(int y=bb[1];y<=bb[3];y++)for(int x=bb[0];x<=bb[2];x++){
  double t=hit(g,(x+0.5-cx)/fx,(y+0.5-cy)/fy,0);
  if(t>0){unsigned long long key=((unsigned long long)__float_as_uint((float)t)<<32)|(unsigned int)i;atomicMin(keys+y*w+x,key);}
 }
}
extern "C" __global__ void shade(const double* gs,const float* colors,const unsigned long long* keys,int w,int h,double fx,double fy,double cx,double cy,unsigned char* rgb){
 int p=blockDim.x*blockIdx.x+threadIdx.x;if(p>=w*h)return;
 if(keys[p]==~0ULL){for(int k=0;k<3;k++)rgb[p*3+k]=20;return;}
 int i=(unsigned int)keys[p];double nn[3],dx=(p%w+0.5-cx)/fx,dy=(p/w+0.5-cy)/fy;
 hit(gs+15*i,dx,dy,nn);
 double norm=sqrt(nn[0]*nn[0]+nn[1]*nn[1]+nn[2]*nn[2]);
 for(int k=0;k<3;k++)nn[k]/=norm;
 if(nn[0]*dx+nn[1]*dy+nn[2]>0)for(int k=0;k<3;k++)nn[k]=-nn[k];
 double light=0.25+0.75*fmax(0.,-0.35*nn[0]-0.45*nn[1]-0.8215838*nn[2]);
 for(int k=0;k<3;k++)rgb[p*3+k]=(unsigned char)fmin(255.,255*colors[i*3+k]*light);
}
'''

class EllipsoidRenderer:
    def __init__(self,g):
        import cupy as cp
        self.cp=cp
        # CPU copies synchronize PyTorch; no shared stream ownership or model mutation.
        self.xyz=g.get_xyz.detach().cpu().numpy().astype(np.float64)
        self.scales=g.get_scaling.detach().cpu().numpy().astype(np.float64)
        q=g.get_rotation.detach().cpu().numpy()
        self.rot=Rotation.from_quat(q[:,[1,2,3,0]]).as_matrix()
        self.opacity=g.get_opacity.detach().cpu().numpy().reshape(-1)
        dc=g.get_features.detach()[:,0,:].cpu().numpy()
        self.colors=np.clip(.5+.28209479177387814*dc,0,1).astype(np.float32)
        if not np.isfinite(self.scales).all() or not (self.scales>0).all():
            raise ValueError('Invalid ellipsoid scales')
        self.module=cp.RawModule(code=KERNEL)
        self.raster=self.module.get_function('raster');self.shade=self.module.get_function('shade')

    def render(self,cam):
        cp=self.cp;start=time.perf_counter()
        w,h=cam.image_width,cam.image_height
        fx=w/(2*math.tan(cam.FoVx/2));fy=h/(2*math.tan(cam.FoVy/2));cx=w/2;cy=h/2
        rr=np.asarray(cam.R);tt=np.asarray(cam.T)
        c=self.xyz@rr+tt;axes=np.einsum('ij,njk->nik',rr.T,self.rot);ss=self.scales
        ext=np.sqrt(np.sum((axes*ss[:,None,:])**2,axis=2))
        keep=(self.opacity>=.05)&(c[:,2]+ext[:,2]>.01)
        ids=np.flatnonzero(keep);c=c[keep];axes=axes[keep];ss=ss[keep];ext=ext[keep]
        zlo=np.maximum(.01,c[:,2]-ext[:,2]);zhi=np.maximum(.01,c[:,2]+ext[:,2])
        xx=np.stack([(c[:,0]+sx*ext[:,0])/z for sx in [-1,1] for z in [zlo,zhi]],axis=1)*fx+cx
        yy=np.stack([(c[:,1]+sy*ext[:,1])/z for sy in [-1,1] for z in [zlo,zhi]],axis=1)*fy+cy
        boxes=np.column_stack([np.floor(xx.min(1))-1,np.floor(yy.min(1))-1,
                               np.ceil(xx.max(1))+1,np.ceil(yy.max(1))+1])
        # Clip before integer conversion; extremely large footprints must not overflow.
        boxes=np.clip(boxes,[-1,-1,-1,-1],[w,h,w,h]).astype(np.int32)
        boxes[:,0]=np.maximum(0,boxes[:,0]);boxes[:,1]=np.maximum(0,boxes[:,1])
        boxes[:,2]=np.minimum(w-1,boxes[:,2]);boxes[:,3]=np.minimum(h-1,boxes[:,3])
        visible=(boxes[:,2]>=boxes[:,0])&(boxes[:,3]>=boxes[:,1]);ids=ids[visible];n=len(ids)
        if not n:return Image.new('RGB',(w,h),(20,20,20)),{'sigma':1,'candidate_count':0,'coverage':0.}
        gs=cp.asarray(np.column_stack([c,axes.reshape(-1,9),ss])[visible])
        bb=cp.asarray(boxes[visible]);colors=cp.asarray(self.colors[ids])
        sentinel=np.uint64(0xffffffffffffffff)
        keys=cp.full(w*h,sentinel,dtype=cp.uint64);rgb=cp.empty((h,w,3),dtype=cp.uint8)
        intr=(np.int32(w),np.int32(h),np.float64(fx),np.float64(fy),np.float64(cx),np.float64(cy))
        self.raster(((n+127)//128,),(128,),(gs,bb,np.int32(n),*intr,keys))
        self.shade(((w*h+255)//256,),(256,),(gs,colors,keys,*intr,rgb))
        array=rgb.get()
        stats={'sigma':1,'opacity_threshold':.05,'candidate_count':n,
               'coverage':float(cp.mean(keys!=sentinel).get()),'seconds':time.perf_counter()-start,
               'color':'SH DC + directional light','true_thickness':True}
        return Image.fromarray(array),stats
