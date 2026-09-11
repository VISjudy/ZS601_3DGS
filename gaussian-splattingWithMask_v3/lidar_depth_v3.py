"""E-only occlusion-aware LiDAR pseudo-depth and distance-weighted loss."""
from collections import OrderedDict
from pathlib import Path
import csv
import hashlib
import json
import re
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


class LidarDepthProvider:
    def __init__(self,points,args,input_identity):
        self.args=args
        self.root=Path(args.lidar_depth_cache).expanduser().resolve()
        if '/drive/' in self.root.as_posix().lower():
            raise ValueError('LiDAR cache must stay on Colab local disk, not Google Drive')
        self.root.mkdir(parents=True,exist_ok=False)
        self.points_cpu=np.asarray(points,dtype=np.float64)
        self.points=torch.as_tensor(self.points_cpu,dtype=torch.float32,device='cuda')
        self.memory=OrderedDict(); self.generated=0
        self.identity={
            'version':'lidar_camera_z_v1','point_cloud':input_identity['point_cloud'],
            'cameras_file':input_identity['cameras_file'],
            'min':args.lidar_depth_min,'max':args.lidar_depth_max,
            'zbuffer':'nearest positive camera-z per rounded pixel',
            'splat_radius':args.lidar_depth_splat_radius,
            'min_neighbors':args.lidar_depth_min_neighbors,
            'edge_relative':args.lidar_depth_edge_relative,
            'edge_absolute':args.lidar_depth_edge_absolute,
        }
        (self.root/'cache_manifest.json').write_text(json.dumps({
            'identity':self.identity,'storage':'temporary local float16 depth plus packed mask',
            'occlusion':'nearest-depth z-buffer before conservative hole filling',
            'invalid':'nonfinite, behind/too-near/too-far/outside-image and discontinuous neighborhoods'},indent=2),
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
        for start in range(0,len(self.points),self.args.lidar_depth_chunk):
            point=self.points[start:start+self.args.lidar_depth_chunk]
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
            neighbors=F.avg_pool2d(hit.float(),kernel,stride=1,padding=radius)*kernel*kernel
            valid=(torch.isfinite(local_min)&torch.isfinite(local_max)&
                   (neighbors>=self.args.lidar_depth_min_neighbors))
            spread=local_max-local_min
            valid&=spread<=(self.args.lidar_depth_edge_absolute+
                            self.args.lidar_depth_edge_relative*local_min)
            depth=local_min
        else:
            valid=hit
        valid&=(depth>=self.args.lidar_depth_min)&(depth<=self.args.lidar_depth_max)
        depth=torch.where(valid,depth,torch.zeros_like(depth))[0,0]
        return depth.cpu(),valid[0,0].cpu()

    def _get_cpu(self,cam):
        key=self._key(cam)
        if key in self.memory:
            depth,valid=self.memory.pop(key); self.memory[key]=(depth,valid)
            return depth,valid
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
        return depth,valid

    def get(self,cam):
        depth,valid=self._get_cpu(cam)
        return depth.cuda(non_blocking=True),valid.cuda(non_blocking=True)

    def _backproject_validation(self,cam,depth,valid,tree):
        pixel=torch.nonzero(valid,as_tuple=False)
        if not len(pixel): raise ValueError(f'No valid LiDAR depth for {cam.image_name}')
        if len(pixel)>self.args.lidar_depth_backproject_samples:
            choose=torch.linspace(0,len(pixel)-1,self.args.lidar_depth_backproject_samples).round().long()
            pixel=pixel[choose]
        v=pixel[:,0].numpy().astype(np.float64); u=pixel[:,1].numpy().astype(np.float64)
        z=depth[pixel[:,0],pixel[:,1]].numpy().astype(np.float64)
        width,height=int(cam.image_width),int(cam.image_height)
        fx=width/(2*np.tan(float(cam.FoVx)/2)); fy=height/(2*np.tan(float(cam.FoVy)/2))
        camera=np.stack(((u-width/2.)*z/fx,(v-height/2.)*z/fy,z),axis=1)
        world=(camera-np.asarray(cam.T,dtype=np.float64))@np.asarray(cam.R,dtype=np.float64).T
        distance,_=tree.query(world,k=1,workers=-1)
        tolerance=self.args.lidar_depth_backproject_tolerance
        return {'samples':len(distance),'mean':float(np.mean(distance)),
                'median':float(np.median(distance)),'p95':float(np.quantile(distance,.95)),
                'max':float(np.max(distance)),'pass_fraction':float(np.mean(distance<=tolerance)),
                'tolerance':tolerance}

    def _save_depth(self,path,depth,valid):
        span=self.args.lidar_depth_max-self.args.lidar_depth_min
        normalized=((depth.numpy()-self.args.lidar_depth_min)/span).clip(0,1)
        encoded=np.zeros(depth.shape,dtype=np.uint16)
        encoded[valid.numpy()]=np.round(normalized[valid.numpy()]*65534+1).astype(np.uint16)
        Image.fromarray(encoded,mode='I;16').save(path)
        return normalized

    def _save_color(self,path,normalized,valid):
        x=normalized
        rgb=np.stack((np.clip(1.5-np.abs(4*x-3),0,1),
                      np.clip(1.5-np.abs(4*x-2),0,1),
                      np.clip(1.5-np.abs(4*x-1),0,1)),axis=-1)
        rgb[~valid.numpy()]=0
        Image.fromarray(np.round(rgb*255).astype(np.uint8)).save(path)

    def prepare_and_export(self,camera_groups,export_path):
        from scipy.spatial import cKDTree
        export=Path(export_path).expanduser().resolve()
        export.mkdir(parents=True,exist_ok=False)
        tree=cKDTree(self.points_cpu)
        fields=['role','camera_index','image_name','depth_u16','color_preview',
                'valid_pixels','coverage','depth_min','depth_median','depth_max',
                'backproject_samples','backproject_mean','backproject_median',
                'backproject_p95','backproject_max','backproject_pass_fraction']
        manifest=export/'depth_manifest.csv'
        records=[]; total=sum(len(v) for v in camera_groups.values()); done=0
        with manifest.open('x',newline='',encoding='utf-8') as handle:
            writer=csv.DictWriter(handle,fieldnames=fields); writer.writeheader()
            for role,cameras in camera_groups.items():
                raw_dir=export/role; raw_dir.mkdir()
                preview_dir=export/(role+'_preview') if role in ('val','test') else None
                if preview_dir: preview_dir.mkdir()
                for index,cam in enumerate(cameras):
                    depth,valid=self._get_cpu(cam)
                    check=self._backproject_validation(cam,depth,valid,tree)
                    if check['p95']>self.args.lidar_depth_backproject_tolerance:
                        raise ValueError(f'Backprojection validation failed for {cam.image_name}: {check}')
                    safe=re.sub(r'[^A-Za-z0-9_.-]+','_',Path(cam.image_name).stem)
                    stem=f'{index:05d}_{safe}_{self._key(cam)[:10]}'
                    depth_rel=Path(role)/(stem+'_depth_u16.png')
                    normalized=self._save_depth(export/depth_rel,depth,valid)
                    preview_rel=''
                    if preview_dir:
                        preview_rel=Path(role+'_preview')/(stem+'_depth_color.png')
                        self._save_color(export/preview_rel,normalized,valid)
                    values=depth[valid]
                    row={'role':role,'camera_index':index,'image_name':cam.image_name,
                         'depth_u16':str(depth_rel),'color_preview':str(preview_rel),
                         'valid_pixels':int(valid.sum()),'coverage':float(valid.float().mean()),
                         'depth_min':float(values.min()),'depth_median':float(values.median()),
                         'depth_max':float(values.max()),
                         **{'backproject_'+k:v for k,v in check.items()
                            if k not in ('tolerance',)}}
                    writer.writerow(row); handle.flush(); records.append(row); done+=1
                    if done<=5 or done%100==0 or done==total:
                        print(f'[LIDAR DEPTH EXPORT] {done}/{total} {role} {cam.image_name} '
                              f'coverage={row["coverage"]:.4f} backproject_p95={row["backproject_p95"]:.5f}',flush=True)
        summary={'identity':self.identity,'camera_count':len(records),
            'depth_encoding':{'file':'16-bit PNG','invalid':0,'valid_code_range':[1,65535],
                'decode':f'depth={self.args.lidar_depth_min}+(code-1)/65534*'
                         f'{self.args.lidar_depth_max-self.args.lidar_depth_min}'},
            'occlusion':'nearest camera-z z-buffer; discontinuous splat neighborhoods rejected',
            'backprojection':{'world_formula':'(camera_xyz-T) @ R.T',
                'nearest_lidar_p95_max':max(r['backproject_p95'] for r in records),
                'required_p95_max':self.args.lidar_depth_backproject_tolerance},
            'coverage_mean':float(np.mean([r['coverage'] for r in records])),
            'manifest':'depth_manifest.csv'}
        (export/'dataset_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
        print('[LIDAR DEPTH DATASET READY]',json.dumps(summary,indent=2),flush=True)
        return summary


def render_expected_center_depth(cam,g,pipe):
    """Differentiable opacity-normalized expected camera-z of Gaussian centers."""
    from render_v3 import attributes
    transform=cam.world_view_transform.cuda()
    z=(g.get_xyz@transform[:3,:3]+transform[3,:3])[:,2]
    accum=attributes(cam,g,pipe,torch.stack((z,torch.ones_like(z),torch.zeros_like(z)),dim=-1))
    alpha=accum[1]
    return accum[0]/alpha.clamp_min(1e-8),alpha


def lidar_depth_term(cam,g,pipe,provider,args,iteration):
    configured=args.lambda_lidar_depth
    empty={'raw':0.,'weight':0.,'weighted':0.,'valid_pixels':0,'lidar_pixels':0,
           'rendered_fraction':0.,'mean_target_depth':0.,'distance_weight_mean':0.}
    if not args.lidar_depth_loss or iteration<args.lidar_depth_start or configured==0:
        return g.get_xyz.new_zeros(()),{**empty,
            'state':'OFF' if not args.lidar_depth_loss else 'WAITING_OR_ZERO_WEIGHT'}
    ramp=min(1.,max(0.,(iteration-args.lidar_depth_start)/max(1,args.lidar_depth_warmup)))
    weight=configured*ramp
    if weight==0: return g.get_xyz.new_zeros(()),{**empty,'state':'WARMUP'}
    target,target_valid=provider.get(cam)
    predicted,alpha=render_expected_center_depth(cam,g,pipe)
    valid=(target_valid&torch.isfinite(predicted)&(predicted>=args.lidar_depth_min)&
           (predicted<=args.lidar_depth_max)&(alpha>=args.lidar_depth_alpha_min))
    if cam.alpha_mask is not None: valid&=cam.alpha_mask.cuda()[0]>.5
    count=int(valid.sum()); lidar_count=int(target_valid.sum())
    fraction=count/max(lidar_count,1)
    if count<args.lidar_depth_min_pixels:
        return g.get_xyz.new_zeros(()),{**empty,'valid_pixels':count,
            'lidar_pixels':lidar_count,'rendered_fraction':fraction,
            'state':'INSUFFICIENT_VALID_PIXELS'}
    target_values=target[valid]; predicted_values=predicted[valid]
    relative=(predicted_values-target_values)/target_values.clamp_min(1e-6)
    per_pixel=F.smooth_l1_loss(relative,torch.zeros_like(relative),
                               beta=args.lidar_depth_huber_beta,reduction='none')
    reference_distance=target_values.detach().median()
    distance_weight=(reference_distance/target_values.detach().clamp_min(1e-6)).pow(
        args.lidar_depth_distance_power)
    distance_weight=distance_weight.clamp(args.lidar_depth_weight_min,args.lidar_depth_weight_max)
    raw=(per_pixel*distance_weight).sum()/distance_weight.sum().clamp_min(1e-8)
    return weight*raw,{'raw':float(raw.detach()),'weight':weight,
        'weighted':float(raw.detach())*weight,'valid_pixels':count,
        'lidar_pixels':lidar_count,'rendered_fraction':fraction,
        'mean_target_depth':float(target_values.mean()),
        'distance_weight_mean':float(distance_weight.mean()),'state':'ACTIVE'}
