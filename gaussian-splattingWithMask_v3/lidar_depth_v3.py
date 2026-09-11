"""E-only LiDAR z-buffer depth targets and differentiable expected-depth loss."""
from collections import OrderedDict
from pathlib import Path
import hashlib
import json
import numpy as np
import torch
import torch.nn.functional as F


class LidarDepthProvider:
    def __init__(self,points,args,input_identity):
        self.args=args
        self.root=Path(args.lidar_depth_cache).expanduser().resolve()
        if '/drive/' in self.root.as_posix().lower():
            raise ValueError('LiDAR depth cache must stay on Colab local disk, not Google Drive')
        self.root.mkdir(parents=True,exist_ok=False)
        self.points=torch.as_tensor(np.asarray(points),dtype=torch.float32,device='cuda')
        self.memory=OrderedDict(); self.generated=0
        self.identity={
            'point_cloud':input_identity['point_cloud'],
            'cameras_file':input_identity['cameras_file'],
            'min':args.lidar_depth_min,'max':args.lidar_depth_max,
            'splat_radius':args.lidar_depth_splat_radius,
            'edge_relative':args.lidar_depth_edge_relative,
            'edge_absolute':args.lidar_depth_edge_absolute,
        }
        (self.root/'cache_manifest.json').write_text(
            json.dumps({'identity':self.identity,'storage':'temporary local float16 depth + packed valid mask',
                        'depth':'camera z from raw LiDAR z-buffer; edge-discontinuous splats rejected'},indent=2),
            encoding='utf-8')
        print('[LIDAR DEPTH CACHE]',json.dumps({'path':str(self.root),'points':len(self.points),
            'persistent_to_drive':False,**self.identity},indent=2),flush=True)

    def _key(self,cam):
        camera={'name':cam.image_name,'width':int(cam.image_width),'height':int(cam.image_height),
                'fovx':float(cam.FoVx),'fovy':float(cam.FoVy),
                'R':np.asarray(cam.R).round(12).tolist(),'T':np.asarray(cam.T).round(12).tolist()}
        payload=json.dumps({'input':self.identity,'camera':camera},sort_keys=True,separators=(',',':'))
        return hashlib.sha256(payload.encode()).hexdigest()

    @torch.no_grad()
    def _generate(self,cam):
        height,width=int(cam.image_height),int(cam.image_width)
        fx=width/(2*np.tan(float(cam.FoVx)/2)); fy=height/(2*np.tan(float(cam.FoVy)/2))
        cx,cy=width/2.,height/2.
        rotation=torch.as_tensor(np.asarray(cam.R),device='cuda',dtype=torch.float32)
        translation=torch.as_tensor(np.asarray(cam.T),device='cuda',dtype=torch.float32)
        zbuffer=torch.full((height*width,),float('inf'),device='cuda')
        chunk=self.args.lidar_depth_chunk
        for start in range(0,len(self.points),chunk):
            point=self.points[start:start+chunk]
            camera=point@rotation+translation
            z=camera[:,2]
            valid=(torch.isfinite(camera).all(1)&(z>=self.args.lidar_depth_min)&
                   (z<=self.args.lidar_depth_max))
            u=torch.round(fx*camera[:,0]/z.clamp_min(1e-8)+cx).to(torch.long)
            v=torch.round(fy*camera[:,1]/z.clamp_min(1e-8)+cy).to(torch.long)
            valid&=(u>=0)&(u<width)&(v>=0)&(v<height)
            if valid.any():
                index=v[valid]*width+u[valid]
                zbuffer.scatter_reduce_(0,index,z[valid],reduce='amin',include_self=True)
        depth=zbuffer.view(1,1,height,width)
        hit=torch.isfinite(depth)
        radius=self.args.lidar_depth_splat_radius
        if radius:
            kernel=2*radius+1
            local_min=-F.max_pool2d(torch.where(hit,-depth,depth.new_full((),-torch.inf)),
                                    kernel,stride=1,padding=radius)
            local_max=F.max_pool2d(torch.where(hit,depth,depth.new_full((),-torch.inf)),
                                  kernel,stride=1,padding=radius)
            valid=torch.isfinite(local_min)&torch.isfinite(local_max)
            spread=local_max-local_min
            valid&=spread<=(self.args.lidar_depth_edge_absolute+
                            self.args.lidar_depth_edge_relative*local_min)
            depth=local_min
        else:
            valid=hit
        depth=torch.where(valid,depth,torch.zeros_like(depth))[0,0]
        return depth.cpu(),valid[0,0].cpu()

    def get(self,cam):
        key=self._key(cam)
        if key in self.memory:
            depth,valid=self.memory.pop(key); self.memory[key]=(depth,valid)
        else:
            path=self.root/(key+'.npz')
            if path.is_file():
                with np.load(path) as data:
                    depth=torch.from_numpy(data['depth'].astype(np.float32))
                    shape=tuple(int(x) for x in data['shape'])
                    valid=torch.from_numpy(np.unpackbits(data['valid'])[:np.prod(shape)].reshape(shape).astype(bool))
            else:
                depth,valid=self._generate(cam)
                packed=np.packbits(valid.numpy().reshape(-1))
                temporary=self.root/(key+'.tmp.npz')
                np.savez_compressed(temporary,depth=depth.numpy().astype(np.float16),
                                    valid=packed,shape=np.asarray(depth.shape,dtype=np.int32))
                temporary.replace(path)
                self.generated+=1
                if self.generated<=5 or self.generated%100==0:
                    values=depth[valid]
                    print('[LIDAR DEPTH]',json.dumps({'generated':self.generated,
                        'camera':cam.image_name,'valid_pixels':int(valid.sum()),
                        'coverage':float(valid.float().mean()),
                        'min':float(values.min()) if len(values) else None,
                        'median':float(values.median()) if len(values) else None,
                        'max':float(values.max()) if len(values) else None}),flush=True)
            self.memory[key]=(depth,valid)
            while len(self.memory)>self.args.lidar_depth_cache_memory:
                self.memory.popitem(last=False)
        return depth.cuda(non_blocking=True),valid.cuda(non_blocking=True)


def render_expected_center_depth(cam,g,pipe):
    """Differentiable opacity-normalized expected camera-z of Gaussian centers."""
    from render_v3 import attributes
    transform=cam.world_view_transform.cuda()
    z=(g.get_xyz@transform[:3,:3]+transform[3,:3])[:,2]
    accum=attributes(cam,g,pipe,torch.stack((z,torch.ones_like(z),torch.zeros_like(z)),dim=-1))
    alpha=accum[1]
    depth=accum[0]/alpha.clamp_min(1e-8)
    return depth,alpha


def lidar_depth_term(cam,g,pipe,provider,args,iteration):
    configured=args.lambda_lidar_depth
    if not args.lidar_depth_loss or iteration<args.lidar_depth_start or configured==0:
        return g.get_xyz.new_zeros(()),{'raw':0.,'weight':0.,'weighted':0.,
            'valid_pixels':0,'lidar_pixels':0,'rendered_fraction':0.,
            'state':'OFF' if not args.lidar_depth_loss else 'WAITING_OR_ZERO_WEIGHT'}
    ramp=min(1.,max(0.,(iteration-args.lidar_depth_start)/max(1,args.lidar_depth_warmup)))
    weight=configured*ramp
    if weight==0:
        return g.get_xyz.new_zeros(()),{'raw':0.,'weight':0.,'weighted':0.,
            'valid_pixels':0,'lidar_pixels':0,'rendered_fraction':0.,'state':'WARMUP'}
    target,target_valid=provider.get(cam)
    predicted,alpha=render_expected_center_depth(cam,g,pipe)
    valid=target_valid&torch.isfinite(predicted)&(predicted>=args.lidar_depth_min)&(
          predicted<=args.lidar_depth_max)&(alpha>=args.lidar_depth_alpha_min)
    if cam.alpha_mask is not None:
        valid&=cam.alpha_mask.cuda()[0]>.5
    count=int(valid.sum()); lidar_count=int(target_valid.sum())
    rendered_fraction=count/max(lidar_count,1)
    if count<args.lidar_depth_min_pixels:
        return g.get_xyz.new_zeros(()),{'raw':0.,'weight':0.,'weighted':0.,
            'valid_pixels':count,'lidar_pixels':lidar_count,
            'rendered_fraction':rendered_fraction,'state':'INSUFFICIENT_VALID_PIXELS'}
    residual=torch.log(predicted[valid].clamp_min(1e-6))-torch.log(target[valid].clamp_min(1e-6))
    raw=F.smooth_l1_loss(residual,torch.zeros_like(residual),
                         beta=args.lidar_depth_huber_beta,reduction='mean')
    return weight*raw,{'raw':float(raw.detach()),'weight':weight,
        'weighted':float(raw.detach())*weight,'valid_pixels':count,
        'lidar_pixels':lidar_count,'rendered_fraction':rendered_fraction,'state':'ACTIVE'}
