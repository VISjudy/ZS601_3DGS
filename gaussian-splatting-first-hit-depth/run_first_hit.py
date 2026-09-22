"""Matched first-contributor depth experiments. Inference only, no optimisation."""
import argparse
import json
import math
from pathlib import Path
import sys
import time

def raster(cam, pc, bg, mode):
    import torch
    from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
    s=GaussianRasterizationSettings(image_height=cam.image_height,image_width=cam.image_width,
        tanfovx=cam.tanfovx,tanfovy=cam.tanfovy,bg=torch.full((3,),float(bg),device='cuda'),
        scale_modifier=1.,viewmatrix=cam.world_view_transform,projmatrix=cam.full_proj_transform,
        sh_degree=0,campos=cam.camera_center,prefiltered=False,debug=False,antialiasing=False,depth_mode=mode)
    return GaussianRasterizer(s)(means3D=pc.get_xyz,means2D=torch.zeros_like(pc.get_xyz),
        shs=pc.get_features,opacities=pc.get_opacity,scales=pc.get_scaling,rotations=pc.get_rotation)

def analytic_tests():
    import numpy as np
    import torch
    from torch import nn
    from render_from_sparse_v4 import GaussianModel, camera, check_analytic_cuda
    from colmap_io import View
    checks={'legacy_isolated':check_analytic_cuda()}
    # Rotated/translated camera and anisotropic, rotated foreground Gaussian.
    q=[math.cos(.19),0,math.sin(.19),0]
    v=View(1,1,'test.png',q,[.1,-.2,.3],64,64,60,60,32,32)
    R=v.matrices()[0][:3,:3]; t=np.asarray(v.t)
    cam=camera(v)
    mus=np.array([[.025,.01,2.0],[.025,.01,4.]])
    xyz=(mus-t)@R
    scales=np.array([[.06,.11,.18],[.4,.4,.4]])
    rot=np.array([[math.cos(.3),0,math.sin(.3),0],[1,0,0,0]])
    pc=GaussianModel(0)
    vals={'_xyz':xyz,'_scaling':np.log(scales),'_rotation':rot,
          '_opacity':np.full((2,1),20.),'_features_dc':np.zeros((2,1,3)),
          '_features_rest':np.zeros((2,0,3))}
    for k,a in vals.items():setattr(pc,k,nn.Parameter(torch.tensor(a,dtype=torch.float32,device='cuda'),requires_grad=False))
    b0,_,raw0=raster(cam,pc,0,0); w0,_,_=raster(cam,pc,1,0)
    b1,_,raw1=raster(cam,pc,0,1); b2,_,raw2=raster(cam,pc,0,2)
    assert torch.equal(b0,b1) and torch.equal(b1,b2)
    assert torch.equal(raw1[1:],raw2[1:])
    aux=raw2.cpu().numpy();ids=aux[1].astype('int32')-1
    valid=ids>=0; yy,xx=np.where(valid)
    d=np.stack([(xx+.5-v.cx)/v.fx,(yy+.5-v.cy)/v.fy,np.ones(len(xx))],1)@R
    # Use actual float32 inputs but independent float64 CPU matrix operations.
    eye=cam.camera_center.cpu().numpy().astype('float64')
    xyz32=pc.get_xyz.cpu().numpy().astype('float64')
    cov=pc.get_covariance().cpu().numpy().astype('float64')
    errs=[];miderrs=[];qerrs=[]
    for j in range(2):
        m=ids[yy,xx]==j
        s=cov[j];S=np.array([[s[0],s[1],s[2]],[s[1],s[3],s[4]],[s[2],s[4],s[5]]]);Q=np.linalg.inv(S)
        ds=d[m];mm=eye-xyz32[j]
        peak=-(ds@Q@mm)/np.einsum('ij,jk,ik->i',ds,Q,ds)
        residual=mm+ds*peak[:,None]
        qmin=np.einsum('ij,jk,ik->i',residual,Q,residual)
        errs.extend(abs(peak-aux[3,yy[m],xx[m]]));qerrs.extend(abs(qmin-aux[7,yy[m],xx[m]]))
        hit=aux[5,yy[m],xx[m]]>0
        miderrs.extend(abs((aux[4,yy[m],xx[m]][hit]+aux[5,yy[m],xx[m]][hit])/2-aux[3,yy[m],xx[m]][hit]))
    assert max(errs)<2e-5 and max(qerrs)<2e-3 and max(miderrs)<1e-5
    first0=(raw1[1]==1); assert first0.sum()>10
    centererr=float((1/raw1[0][first0]-raw1[2][first0]).abs().max())
    assert centererr<1e-5
    alpha=(1-(w0-b0).mean(0)).clamp(0,1)
    harmonic=alpha/raw0[0].clamp_min(1e-8)
    mix=float((harmonic[first0]-raw1[2][first0]).max())
    assert mix>.1 # Distant background changes harmonic Z, not the first ID's Z.
    assert torch.all(raw2[:,~torch.as_tensor(valid,device='cuda')]==0)
    checks.update(dict(status='PASS',rgb_bit_exact_all_modes=True,same_first_ids=True,
        cpu_peak_max_error_m=float(max(errs)),cpu_qmin_max_error=float(max(qerrs)),
        real_intersections_midpoint_max_error_m=float(max(miderrs)),center_max_error_m=centererr,
        harmonic_background_mixing_max_m=mix,nonintersection_pixels=int(((aux[7]>9)&valid).sum())))
    # Explicitly refuse unsupported gradients in first-hit modes.
    pc._xyz.requires_grad_(True)
    try:
        with torch.enable_grad():
            color,_,_=raster(cam,pc,0,1);color.sum().backward()
    except RuntimeError as e:
        assert 'inference-only' in str(e);checks['backward_guard']=True
    else:raise AssertionError('First-hit backward must be rejected')
    return checks

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--package',required=True);p.add_argument('--root',required=True)
    a=p.parse_args();root=Path(a.root);sys.path.insert(0,a.package)
    import numpy as np
    from PIL import Image
    import torch
    from torch import nn
    from render_from_sparse_v4 import GaussianModel,camera
    from colmap_io import View,write_json,sha256,encode_depth
    spec=json.loads((root/'run_spec.json').read_text())
    views=[View(**v) for v in json.loads((root/'selected_views.json').read_text())]
    for v in views:assert v.cx==v.width/2 and v.cy==v.height/2
    gauss=root/'input/gaussian_opacity1_scale05.ply'
    assert sha256(gauss)==spec['source_gaussian_sha256']
    out=root/'results';out.mkdir(exist_ok=False)
    with torch.no_grad():
        write_json(out/'analytic_tests.json',analytic_tests())
        pc=GaussianModel(0);pc.load_ply(str(gauss))
        assert len(pc.get_xyz)==spec['point_count'] and torch.all(pc.get_opacity==1).item()
        assert torch.equal(pc.get_scaling[:,0],pc.get_scaling[:,1]) and torch.equal(pc.get_scaling[:,1],pc.get_scaling[:,2])
        scales0=pc._scaling.detach().clone(); xyz0=pc.get_xyz.detach().clone()
        baseline=[]; identity=[];diagnostics=[]
        configs=[('method_c_first_center_s05',1,.5),('method_d_first_peak_s05',2,.5),('method_e_first_peak_s01',2,.1)]
        for method,mode,scale in configs:
            dest=out/method;dest.mkdir()
            pc._scaling=nn.Parameter(scales0 if scale==.5 else scales0+math.log(scale/.5),requires_grad=False)
            assert torch.equal(pc.get_xyz,xyz0)
            for n in ['images','alpha','masks','depth','depth_mask','depth_mask_alpha50','depth_float','aux']:(dest/n).mkdir()
            rows=[]
            for v in views:
                started=time.time();cam=camera(v)
                b,r,raw=raster(cam,pc,0,mode)
                w,_,rw=raster(cam,pc,1,0) # Weighted mode avoids recomputing unused ray diagnostics.
                alpha=(1-(w-b).mean(0)).clamp(0,1)
                rgb=np.rint(b.clamp(0,1).permute(1,2,0).cpu().numpy()*255).astype('uint8')
                al=alpha.cpu().numpy();aux=raw.cpu().numpy()
                valid=(aux[1]>0)&(aux[0]>0)&np.isfinite(aux[0])
                depth=np.zeros_like(aux[0]);depth[valid]=1/aux[0][valid]
                assert not np.any((aux[1]>0)&~valid) and np.isfinite(aux).all()
                d16=encode_depth(depth,valid)
                arrays={'images':rgb,'alpha':np.rint(al*65535).astype('uint16'),
                    'masks':(al>=.95).astype('uint8')*255,'depth':d16,
                    'depth_mask':valid.astype('uint8')*255,
                    'depth_mask_alpha50':(valid&(al>=.5)).astype('uint8')*255}
                for n,img in arrays.items():
                    Image.fromarray(img).save(dest/n/v.name)
                    assert np.array_equal(np.asarray(Image.open(dest/n/v.name)),img)
                stem=Path(v.name).stem
                np.save(dest/'depth_float'/f'{stem}.npy',depth)
                np.savez_compressed(dest/'aux'/f'{stem}.npz',first_id=aux[1].astype('int32')-1,
                    center_z=aux[2],peak_z=aux[3],entry_z=aux[4],exit_z=aux[5],first_alpha=aux[6],qmin=aux[7])
                if method=='method_c_first_center_s05':
                    b0,_,raw0=raster(cam,pc,0,0)
                    assert torch.equal(b0,b)
                    old=root/'baseline'/v.name
                    assert np.array_equal(rgb,np.asarray(Image.open(old)))
                    harmonic=np.where(al>=.5,al/np.maximum(raw0[0].cpu().numpy(),1e-8),0)
                    dh=encode_depth(harmonic,harmonic>0)
                    oldd=np.asarray(Image.open(root/'baseline_depth'/v.name))
                    assert np.array_equal(dh,oldd),(v.name,int(np.max(abs(dh.astype('int32')-oldd))))
                    baseline.append(dict(name=v.name,rgb_bit_exact=True,legacy_depth_png_bit_exact=True))
                if method=='method_d_first_peak_s05':
                    c=out/'method_c_first_center_s05'
                    prior=np.load(c/'aux'/f'{stem}.npz')
                    assert np.array_equal(aux[1].astype('int32')-1,prior['first_id'])
                    assert np.array_equal(rgb,np.asarray(Image.open(c/'images'/v.name)))
                    identity.append(dict(name=v.name,first_ids_equal=True,rgb_equal=True,
                        center_vs_peak_mean_abs_mm=float(np.abs(aux[2][valid]-aux[3][valid]).mean()*1000)))
                row=dict(name=v.name,mode=mode,scale=scale,coverage=float(valid.mean()),
                    alpha50_coverage=float((valid&(al>=.5)).mean()),rgb_mask_gap=float((al<.95).mean()),
                    strictly_empty_alpha16=float((arrays['alpha']==0).mean()),
                    first_hit_without_real_3sigma_intersection_fraction=float((aux[7][valid]>9).mean()),
                    first_alpha_median=float(np.median(aux[6][valid])),
                    quantization_max_mm=float(np.max(abs(d16[valid]/1000-depth[valid]))*1000),seconds=time.time()-started)
                rows.append(row);print(json.dumps(dict(method=method,**row)),flush=True)
            write_json(dest/'render_summary.json',dict(status='PASS',rows=rows,mode=mode,scale=scale,
                primary_mask='positive finite first-contributor Z; no alpha50 cutoff',
                depth='1/raw_invZ, never alpha/raw_invZ',ordering='center-Z sorted tile contributors',
                no_finite_ellipsoid_gate=True))
            if scale==.1:
                target=dest/'point_cloud/iteration_0/point_cloud.ply';pc.save_ply(str(target))
                write_json(dest/'initialization.json',dict(ply_sha256=sha256(target),source=spec,
                    changed_fields='log_scale only: add log(0.1/0.5)',optimization_steps=0))
        write_json(out/'regression_tests.json',dict(status='PASS',legacy=baseline,matched_center_peak=identity))
        write_json(out/'RENDER_COMPLETE.json',dict(status='30_FIRST_HIT_DEPTH_VIEWS_RENDERED_NOT_TRAINED',views=30,
            source_gaussian_sha256=spec['source_gaussian_sha256'],optimization_steps=0))

if __name__=='__main__':main()
