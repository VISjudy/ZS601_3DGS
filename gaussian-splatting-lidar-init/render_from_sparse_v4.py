#!/usr/bin/env python3
"""Audited camera-only adaptation of the user's Drive render_from_sparse_v4.py.

See provenance/ for the exact original and hashes. Retains GaussianModel,
SH0 initialization, PLY reload and black/white alpha recovery; no optimization.
The CUDA rasterizer is the repository's pinned, unchanged Graphdeco version.
"""
import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import shutil
import time
from types import SimpleNamespace
import numpy as np
from PIL import Image
from plyfile import PlyData
import torch
from torch import nn
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.graphics_utils import BasicPointCloud
from colmap_io import read_views, write_sparse, write_json, sha256, encode_depth, View


def load_cloud(path):
    p = PlyData.read(str(path))['vertex']
    required = ['x', 'y', 'z', 'red', 'green', 'blue']
    if any(n not in p.data.dtype.names for n in required):
        raise ValueError('Input must be a colored point PLY, not Gaussian parameter PLY')
    xyz = np.column_stack([p[n] for n in required[:3]]).astype(np.float32)
    rgb = np.column_stack([p[n] for n in required[3:]])
    if len(xyz) < 4 or not np.isfinite(xyz).all() or np.any(rgb < 0) or np.any(rgb > 255):
        raise ValueError('Invalid point cloud')
    return xyz, rgb.astype(np.uint8)


def camera(v):
    view, proj = v.matrices()
    # CUDA expects the same transposed matrices as the original Graphdeco Camera.
    vt = torch.tensor(view.T, dtype=torch.float32, device='cuda')
    pt = torch.tensor(proj.T, dtype=torch.float32, device='cuda')
    return SimpleNamespace(
        image_width=v.width, image_height=v.height,
        tanfovx=v.width/(2*v.fx), tanfovy=v.height/(2*v.fy),
        world_view_transform=vt.contiguous(), full_proj_transform=(vt @ pt).contiguous(),
        camera_center=torch.tensor(np.linalg.inv(view)[:3, 3], dtype=torch.float32, device='cuda'))


def rasterize(cam, pc, background):
    settings = GaussianRasterizationSettings(
        image_height=cam.image_height, image_width=cam.image_width,
        tanfovx=cam.tanfovx, tanfovy=cam.tanfovy,
        bg=torch.full((3,), float(background), device='cuda'), scale_modifier=1.0,
        viewmatrix=cam.world_view_transform, projmatrix=cam.full_proj_transform,
        sh_degree=0, campos=cam.camera_center, prefiltered=False,
        debug=False, antialiasing=False)
    result = GaussianRasterizer(raster_settings=settings)(
        means3D=pc.get_xyz, means2D=torch.zeros_like(pc.get_xyz),
        shs=pc.get_features, colors_precomp=None, opacities=pc.get_opacity,
        scales=pc.get_scaling, rotations=pc.get_rotation, cov3D_precomp=None)
    if len(result) != 3:
        raise RuntimeError('Expected color, radii, weighted inverse-Z; incompatible rasterizer')
    rgb, radii, invz_sum = result
    if not torch.isfinite(rgb).all() or not torch.isfinite(invz_sum).all():
        raise RuntimeError('Non-finite rasterizer output')
    return rgb, radii, invz_sum


def render(cam, pc):
    black, radii, invz_sum = rasterize(cam, pc, 0)
    white, _, invz_white = rasterize(cam, pc, 1)
    alpha = (1 - (white-black).mean(dim=0)).clamp(0, 1)
    # Kernel returns sum(T_i * alpha_i / Z_i), not normalized inverse depth.
    invz_sum = invz_sum.squeeze(0)
    valid = (alpha > 1e-6) & (invz_sum > 1e-8)
    depth = torch.where(valid, alpha / invz_sum.clamp_min(1e-8), 0)
    straight = torch.where(valid.unsqueeze(0), black / alpha.clamp_min(1e-6), 0).clamp(0, 1)
    if float((invz_white.squeeze(0)-invz_sum).abs().max()) > 1e-5:
        raise RuntimeError('Inverse depth unexpectedly depends on background')
    return (black.clamp(0, 1).permute(1, 2, 0).cpu().numpy(),
            straight.permute(1, 2, 0).cpu().numpy(), alpha.cpu().numpy(),
            depth.cpu().numpy(), int((radii > 0).sum().item()))


def stats(t):
    a = t.detach().cpu().numpy()
    return dict(min=float(a.min()), median=float(np.median(a)), max=float(a.max()))


def initialize(xyz, rgb, opacity, scale_factor, uniform):
    pc = GaussianModel(0)
    pc.create_from_pcd(BasicPointCloud(xyz, rgb.astype(np.float32)/255, np.zeros_like(xyz)), 1.0)
    target = torch.full((len(xyz), 1), opacity, dtype=torch.float32, device='cuda')
    pc._opacity = nn.Parameter(torch.logit(target), requires_grad=False)
    scale = pc.get_scaling * scale_factor if uniform is None else torch.full_like(pc.get_scaling, uniform)
    pc._scaling = nn.Parameter(torch.log(scale), requires_grad=False)
    pc.active_sh_degree = 0
    return pc


def check_analytic_cuda():
    # An isolated Gaussian at Z=2 must have depth=2 even where alpha is low.
    pc = GaussianModel(0)
    values = {
        '_xyz': [[[0, 0, 2]], (1, 3)],
        '_features_dc': [[(0.2-0.5)/0.28209479177387814,
                          (0.6-0.5)/0.28209479177387814,
                          (0.9-0.5)/0.28209479177387814], (1, 1, 3)],
        '_scaling': [[math.log(0.08)]*3, (1, 3)],
        '_rotation': [[1, 0, 0, 0], (1, 4)],
        '_opacity': [[math.log(0.7/0.3)], (1, 1)]}
    for name, (value, shape) in values.items():
        setattr(pc, name, nn.Parameter(torch.tensor(value, dtype=torch.float32, device='cuda').reshape(shape), requires_grad=False))
    pc._features_rest = nn.Parameter(torch.empty((1, 0, 3), device='cuda'), requires_grad=False)
    v = View(1, 1, 'analytic.png', [1, 0, 0, 0], [0, 0, 0], 64, 64, 60, 60, 32, 32)
    with torch.no_grad():
        black, straight, alpha, depth, _ = render(camera(v), pc)
    m = alpha > 0.01
    assert m.sum() > 10 and alpha.max() < 0.71
    depth_error = float(np.abs(depth[m]-2).max())
    color_error = float(np.abs(straight[m]-[0.2, 0.6, 0.9]).max())
    assert depth_error < 1e-4 and color_error < 1e-4, (depth_error, color_error)
    return dict(valid_pixels=int(m.sum()), max_depth_error_m=depth_error, max_straight_rgb_error=color_error)


def run(args):
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required; use the Colab notebook')
    if not 0 < args.opacity < 1 or args.init_scale_factor <= 0 or (args.init_scale_uniform is not None and args.init_scale_uniform <= 0):
        raise ValueError('Finite opacity in (0,1) and positive scales required')
    if not 0 < args.depth_alpha_min <= args.mask_alpha_min <= 1:
        raise ValueError('Invalid alpha thresholds')
    out = Path(args.output)
    if out.exists():
        raise FileExistsError('Refusing to replace an existing output: ' + str(out))
    views = read_views(args.sparse)
    if args.view_ids:
        selected = set(map(int, args.view_ids.split(',')))
        views = [v for v in views if v.image_id in selected]
        if len(views) != len(selected):
            raise ValueError('Requested view IDs missing')
    if args.max_images > 0:
        views = views[:args.max_images]
    xyz, rgb = load_cloud(args.point_cloud)
    out.mkdir(parents=True, exist_ok=False)
    args_record = vars(args).copy()
    write_json(out/'config.json', dict(arguments=args_record, input_ply_sha256=sha256(args.point_cloud),
        input_cameras_sha256=sha256(Path(args.sparse)/'cameras.txt'), input_images_sha256=sha256(Path(args.sparse)/'images.txt'),
        optimization_steps=0, sh_degree=0, world_units='metres', point_count=len(xyz), view_count=len(views)))
    write_json(out/'analytic_cuda_check.json', check_analytic_cuda())
    pc = initialize(xyz, rgb, args.opacity, args.init_scale_factor, args.init_scale_uniform)
    gauss_path = out/'point_cloud/iteration_0/point_cloud.ply'
    pc.save_ply(str(gauss_path))
    reloaded = GaussianModel(0)
    reloaded.load_ply(str(gauss_path))
    attrs = ['_xyz', '_features_dc', '_features_rest', '_opacity', '_scaling', '_rotation']
    for attr in attrs:
        if not torch.equal(getattr(pc, attr), getattr(reloaded, attr)):
            raise RuntimeError('Gaussian PLY reload changed ' + attr)
    write_json(out/'initialization.json', dict(point_count=len(xyz), optimization_steps=0,
        sh_degree=0, requested_opacity=args.opacity, actual_opacity=stats(reloaded.get_opacity),
        opacity_logits=stats(reloaded._opacity), scales_metres=stats(reloaded.get_scaling),
        gaussian_ply_sha256=sha256(gauss_path), ply_reload_exact=True,
        xyz_equal_input=bool(np.array_equal(reloaded.get_xyz.detach().cpu().numpy(), xyz)),
        kernel_per_fragment_alpha_cap=0.99, kernel_near_cutoff_m=0.2))
    del pc
    write_sparse(out/'sparse/0', views, xyz, rgb)
    shutil.copyfile(args.point_cloud, out/'sparse/0/points3D.ply')
    write_json(out/'views.json', [asdict(v) for v in views])
    for name in ['images', 'color', 'rgba', 'alpha', 'masks', 'depth', 'depth_mask']:
        (out/name).mkdir()
    started = time.time()
    rows = []
    with torch.no_grad(), (out/'render_progress.jsonl').open('x') as progress:
        for i, v in enumerate(views):
            t = time.time()
            native, straight, alpha, depth, visible = render(camera(v), reloaded)
            rgb8 = np.rint(native*255).astype(np.uint8)
            straight8 = np.rint(straight*255).astype(np.uint8)
            a8 = np.rint(alpha*255).astype(np.uint8)
            a16 = np.rint(alpha*65535).astype(np.uint16)
            dvalid = (alpha >= args.depth_alpha_min) & (depth > 0)
            outputs = dict(images=rgb8, color=straight8,
                rgba=np.dstack([straight8, a8]), alpha=a16,
                masks=(alpha >= args.mask_alpha_min).astype(np.uint8)*255,
                depth=encode_depth(depth, dvalid), depth_mask=dvalid.astype(np.uint8)*255)
            for folder, value in outputs.items():
                Image.fromarray(value).save(out/folder/v.name)
            row = dict(index=i+1, image_id=v.image_id, name=v.name, visible_gaussians=visible,
                coverage_alpha95=float((alpha>=args.mask_alpha_min).mean()),
                coverage_alpha50=float(dvalid.mean()), seconds=time.time()-t)
            rows.append(row)
            progress.write(json.dumps(row, allow_nan=False)+'\n')
            progress.flush()
            if i % 10 == 0 or i == len(views)-1:
                print(json.dumps(row), flush=True)
    write_json(out/'render_summary.json', dict(views=len(views), seconds=time.time()-started,
        mean_coverage_alpha95=float(np.mean([r['coverage_alpha95'] for r in rows])),
        mean_coverage_alpha50=float(np.mean([r['coverage_alpha50'] for r in rows])), rows=rows))
    write_json(out/'RENDER_COMPLETE.json', dict(status='RENDERED_INITIALIZATION_NOT_TRAINED', views=len(views),
        point_count=len(xyz), optimization_steps=0, gaussian_ply_sha256=sha256(gauss_path)))
    print('RENDER_COMPLETE', flush=True)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--point-cloud', required=True)
    p.add_argument('--sparse', required=True, help='Directory containing COLMAP cameras.txt and images.txt')
    p.add_argument('--output', required=True, help='Must not exist')
    p.add_argument('--opacity', type=float, default=0.999999)
    p.add_argument('--init-scale-factor', type=float, default=0.5)
    p.add_argument('--init-scale-uniform', type=float, default=None, help='Optional metre-valued override; default uses 3-NN')
    p.add_argument('--mask-alpha-min', type=float, default=0.95)
    p.add_argument('--depth-alpha-min', type=float, default=0.5)
    p.add_argument('--max-images', type=int, default=0, help='0 means all, no implicit train/test split')
    p.add_argument('--view-ids', default='', help='Optional comma-separated smoke camera IDs')
    return p


if __name__ == '__main__':
    with torch.no_grad():
        run(parser().parse_args())
