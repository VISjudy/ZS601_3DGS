"""Diagnostic attribute passes using unchanged baseline CUDA. Not depth supervision."""
from pathlib import Path
import json, math
import numpy as np
import torch
from PIL import Image
from geometry_v3 import normal_axis

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
    for i,cam in enumerate(cameras):
        stem=f'val{i:02d}'
        rgb=render(cam,g,pipe,black)['render']
        depth,normal,alpha,valid,nvalid=render_geometry(cam,g,pipe)
        save_rgb(out/(stem+'_rgb.png'),rgb)
        save_rgb(out/(stem+'_normal.png'),torch.where(nvalid[None],(normal+1)/2,0.))
        # Fixed color scale across all cameras/iterations; zero is black, far is white.
        gray=torch.nan_to_num(depth,nan=0.).clamp(0,a.depth_visual_max)/a.depth_visual_max
        save_rgb(out/(stem+'_depth.png'),gray[None].repeat(3,1,1))
        np.savez_compressed(out/(stem+'_geometry.npz'),depth_z=depth.cpu().numpy(),
            normal_camera=normal.cpu().numpy(),alpha=alpha.cpu().numpy(),
            depth_valid=valid.cpu().numpy(),normal_valid=nvalid.cpu().numpy())
        if iteration==0: save_rgb(out/(stem+'_gt_masked.png'),cam.original_image.cuda())
        mask=cam.alpha_mask.cuda() if cam.alpha_mask is not None else torch.ones_like(rgb[:1])
        err=(rgb-cam.original_image.cuda()).square()*mask
        # GT is already masked; multiply residual by mask again for excluded pixels.
        mse=float(err.sum()/(3*mask.sum()).clamp_min(1))
        manifest.append({'index':i,'image_name':cam.image_name,'R':np.asarray(cam.R).tolist(),
            'T':np.asarray(cam.T).tolist(),'valid_depth_fraction':float(valid.float().mean()),
            'masked_psnr':float(-10*np.log10(max(mse,1e-12)))})
    (out/'manifest.json').write_text(json.dumps({'iteration':iteration,'units':a.units,
        'depth':'opacity-normalized expected Gaussian-center camera z; not unbiased surface depth',
        'normal':'camera coordinates, oriented toward this camera, RGB=(normal+1)/2',
        'depth_display_range':[0,a.depth_visual_max],'cameras':manifest},indent=2),encoding='utf-8')
    print(f'[VAL] iter={iteration} RGB/normal/depth + raw NPZ: {out}',flush=True)
