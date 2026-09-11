"""Diagnostic attribute passes using unchanged baseline CUDA. Not depth supervision."""
from pathlib import Path
import json, math
import numpy as np
import torch
from PIL import Image
from geometry_v3 import normal_axis
from runtime_v3 import append_csv

def attributes(cam,g,pipe,colors):
    from diff_gaussian_rasterization import GaussianRasterizationSettings,GaussianRasterizer
    settings=GaussianRasterizationSettings(image_height=cam.image_height,image_width=cam.image_width,
        tanfovx=math.tan(cam.FoVx/2),tanfovy=math.tan(cam.FoVy/2),
        bg=torch.zeros(3,device='cuda'),scale_modifier=1.,viewmatrix=cam.world_view_transform.cuda(),
        projmatrix=cam.full_proj_transform.cuda(),sh_degree=0,campos=cam.camera_center.cuda(),
        prefiltered=False,debug=pipe.debug,antialiasing=pipe.antialiasing)
    image,_,_=GaussianRasterizer(settings)(means3D=g.get_xyz,means2D=torch.zeros_like(g.get_xyz),
        shs=None,colors_precomp=colors.contiguous(),opacities=g.get_opacity,
        scales=g.get_scaling,rotations=g.get_rotation,cov3D_precomp=None)
    return image

@torch.no_grad()
def render_geometry(cam,g,pipe):
    transform=cam.world_view_transform.cuda()
    z=(g.get_xyz@transform[:3,:3]+transform[3,:3])[:,2]
    # No Python [0,1] clamp: raw z attributes can exceed one scene unit.
    accum=attributes(cam,g,pipe,torch.stack([z,torch.ones_like(z),torch.zeros_like(z)],-1))
    alpha=accum[1]; valid=alpha>.05
    depth=accum[0]/alpha.clamp_min(1e-8)
    n=normal_axis(g.get_rotation)
    toward=cam.camera_center.cuda()-g.get_xyz
    n=torch.where(((n*toward).sum(-1)<0)[:,None],-n,n)
    n=n@transform[:3,:3]  # row-vector world -> camera
    encoded=attributes(cam,g,pipe,(n+1)/2)
    normals=2*encoded/alpha.clamp_min(1e-8)[None]-1
    normal_valid=valid&(torch.linalg.vector_norm(normals,dim=0)>1e-6)
    normals=torch.nn.functional.normalize(normals,dim=0)
    normals[:,~normal_valid]=0
    valid=valid&torch.isfinite(depth)&(depth>0)
    depth[~valid]=float('nan')
    return depth,normals,alpha,valid,normal_valid

def save_rgb(path,tensor):
    arr=(tensor.detach().clamp(0,1).permute(1,2,0).cpu().numpy()*255).round().astype('uint8')
    Image.fromarray(arr).save(path)

@torch.no_grad()
def export_val(a,iteration,cameras,g,pipe):
    from gaussian_renderer import render
    out=Path(a.model_path)/'val_v3'/f'iteration_{iteration:06d}'
    out.mkdir(parents=True,exist_ok=False)
    manifest=[]; black=torch.zeros(3,device='cuda')
    ellipsoids=None
    if a.val_ellipsoids=='on':
        from ellipsoid_v3 import EllipsoidRenderer
        ellipsoids=EllipsoidRenderer(g)
    for i,cam in enumerate(cameras):
        stem=f'val{i:02d}'
        rgb=render(cam,g,pipe,black)['render']
        depth,normal,alpha,valid,nvalid=render_geometry(cam,g,pipe)
        save_rgb(out/(stem+'_rgb.png'),rgb)
        ell_stat=None
        if ellipsoids is not None:
            ell_image,ell_stat=ellipsoids.render(cam)
            ell_image.save(out/(stem+'_ellipsoid.png'))
        save_rgb(out/(stem+'_normal.png'),torch.where(nvalid[None],(normal+1)/2,0.))
        # Fixed color scale across all cameras/iterations; zero is black, far is white.
        gray=torch.nan_to_num(depth,nan=0.).clamp(0,a.depth_visual_max)/a.depth_visual_max
        save_rgb(out/(stem+'_depth.png'),gray[None].repeat(3,1,1))
        if a.val_npz=='on':
            np.savez_compressed(out/(stem+'_geometry.npz'),depth_z=depth.cpu().numpy(),
                normal_camera=normal.cpu().numpy(),alpha=alpha.cpu().numpy(),
                depth_valid=valid.cpu().numpy(),normal_valid=nvalid.cpu().numpy())
        if iteration==0: save_rgb(out/(stem+'_gt_masked.png'),cam.original_image.cuda())
        mask=cam.alpha_mask.cuda() if cam.alpha_mask is not None else torch.ones_like(rgb[:1])
        err=(rgb-cam.original_image.cuda()).square()*mask
        # GT is already masked; multiply residual by mask again for excluded pixels.
        mse=float(err.sum()/(3*mask.sum()).clamp_min(1))
        from utils.loss_utils import ssim
        mae=float(((rgb-cam.original_image.cuda()).abs()*mask).sum()/(3*mask.sum()).clamp_min(1))
        zero_mask_ssim=float(ssim(rgb*mask,cam.original_image.cuda()))
        metrics={'iteration':iteration,'camera_index':i,'image_name':cam.image_name,
                 'masked_psnr':float(-10*np.log10(max(mse,1e-12))),
                 'masked_mae':mae,'ssim_zero_mask_full_image':zero_mask_ssim,'count':len(g.get_xyz)}
        append_csv(Path(a.model_path)/'val_metrics.csv',metrics)
        manifest.append({'index':i,'image_name':cam.image_name,'R':np.asarray(cam.R).tolist(),
            'T':np.asarray(cam.T).tolist(),'valid_depth_fraction':float(valid.float().mean()),
            'masked_psnr':metrics['masked_psnr'],'masked_mae':mae,
            'ssim_zero_mask_full_image':zero_mask_ssim,'ellipsoid':ell_stat})
    append_csv(Path(a.model_path)/'val_metrics.csv',{'iteration':iteration,'camera_index':-1,
        'image_name':'MEAN','masked_psnr':float(np.mean([m['masked_psnr'] for m in manifest])),
        'masked_mae':float(np.mean([m['masked_mae'] for m in manifest])),
        'ssim_zero_mask_full_image':float(np.mean([m['ssim_zero_mask_full_image'] for m in manifest])),
        'count':len(g.get_xyz)})
    (out/'manifest.json').write_text(json.dumps({'iteration':iteration,'units':a.units,
        'depth':'opacity-normalized expected Gaussian-center camera z; not unbiased surface depth',
        'normal':'camera coordinates, oriented toward this camera, RGB=(normal+1)/2',
        'depth_display_range':[0,a.depth_visual_max],'cameras':manifest},indent=2),encoding='utf-8')
    print(f'[VAL] iter={iteration} RGB/normal/depth/ellipsoid={a.val_ellipsoids} raw_npz={a.val_npz}: {out}',flush=True)


def _masked_metrics(cam,rgb):
    from utils.loss_utils import ssim
    gt=cam.original_image.cuda()
    mask=cam.alpha_mask.cuda() if cam.alpha_mask is not None else torch.ones_like(rgb[:1])
    denom=(3*mask.sum()).clamp_min(1)
    residual=rgb-gt
    mse=float((residual.square()*mask).sum()/denom)
    mae=float((residual.abs()*mask).sum()/denom)
    return {'masked_psnr':float(-10*np.log10(max(mse,1e-12))),
            'masked_mae':mae,
            'ssim_zero_mask_full_image':float(ssim(rgb*mask,gt)),
            'valid_pixel_fraction':float(mask.mean())}

def rank_worst_test(records,count=10):
    """Stable ranking contract: low masked PSNR first, then image name."""
    return sorted(records,key=lambda r:(r['masked_psnr'],r['image_name']))[:min(count,len(records))]

@torch.no_grad()
def export_final_test(a,iteration,cameras,g,pipe):
    """Evaluate all test cameras, then save heavy diagnostics only for the worst ten."""
    from gaussian_renderer import render
    if iteration!=150000:
        raise ValueError('Final test export is reserved for the completed 150000 iteration model')
    if not cameras:
        raise ValueError('Final test requires a nonempty explicit test camera list')
    out=Path(a.model_path)/'test_final'/f'iteration_{iteration:06d}'
    out.mkdir(parents=True,exist_ok=False)
    metric_path=out/'test_metrics.csv'
    black=torch.zeros(3,device='cuda')
    records=[]
    for i,cam in enumerate(cameras):
        rgb=render(cam,g,pipe,black)['render']
        row={'iteration':iteration,'camera_index':i,'image_name':cam.image_name,
             **_masked_metrics(cam,rgb),'count':len(g.get_xyz)}
        append_csv(metric_path,row); records.append(row)
        if (i+1)%25==0 or i+1==len(cameras):
            print(f'[FINAL TEST] metrics {i+1}/{len(cameras)}',flush=True)
    numeric=('masked_psnr','masked_mae','ssim_zero_mask_full_image','valid_pixel_fraction')
    aggregates={}
    for kind,fn in (('MEAN',np.mean),('MEDIAN',np.median),('MIN',np.min),('MAX',np.max)):
        row={'iteration':iteration,'camera_index':-1,'image_name':kind,'count':len(g.get_xyz)}
        row.update({k:float(fn([r[k] for r in records])) for k in numeric})
        append_csv(metric_path,row); aggregates[kind.lower()]=row
    worst=rank_worst_test(records,10)
    ellipsoids=None
    from ellipsoid_v3 import EllipsoidRenderer
    ellipsoids=EllipsoidRenderer(g)
    exported=[]
    for rank,row in enumerate(worst,1):
        cam=cameras[row['camera_index']]
        stem=f'worst_{rank:02d}_cam_{row["camera_index"]:04d}'
        rgb=render(cam,g,pipe,black)['render']
        depth,normal,alpha,valid,nvalid=render_geometry(cam,g,pipe)
        save_rgb(out/(stem+'_rgb.png'),rgb)
        ell_image,ell_stat=ellipsoids.render(cam)
        ell_image.save(out/(stem+'_ellipsoid.png'))
        save_rgb(out/(stem+'_normal.png'),torch.where(nvalid[None],(normal+1)/2,0.))
        gray=torch.nan_to_num(depth,nan=0.).clamp(0,a.depth_visual_max)/a.depth_visual_max
        save_rgb(out/(stem+'_depth.png'),gray[None].repeat(3,1,1))
        exported.append({'rank':rank,**row,'files':{
            'rgb':stem+'_rgb.png','ellipsoid_1sigma':stem+'_ellipsoid.png',
            'depth':stem+'_depth.png','normal':stem+'_normal.png'},
            'valid_depth_fraction':float(valid.float().mean()),'ellipsoid':ell_stat})
        print(f'[FINAL TEST WORST] rank={rank} camera={row["image_name"]} psnr={row["masked_psnr"]:.4f}',flush=True)
    summary={'iteration':iteration,'camera_count':len(records),'ranking':'masked_psnr ascending, tie=image_name',
        'metrics_contract':{
            'masked_psnr':'PSNR from MSE normalized by valid mask pixels and 3 RGB channels',
            'masked_mae':'MAE normalized by valid mask pixels and 3 RGB channels',
            'ssim_zero_mask_full_image':'SSIM on prediction and GT after excluded pixels are zeroed; full-image SSIM',
            'ellipsoid':'opaque 1-sigma Gaussian ellipsoids with SH DC color and diagnostic lighting',
            'depth':'opacity-normalized expected Gaussian-center camera z; diagnostic only'},
        'aggregates':aggregates,'worst10':exported,'raw_geometry_npz_saved':False}
    (out/'test_summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
    print(f'[FINAL TEST DONE] cameras={len(records)} worst={len(worst)} metrics={metric_path}',flush=True)
    return summary,out
