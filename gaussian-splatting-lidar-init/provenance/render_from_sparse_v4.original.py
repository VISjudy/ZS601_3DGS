#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import torch
import numpy as np
#import cv2
from tqdm import tqdm
from PIL import Image
from utils.graphics_utils import getProjectionMatrix, geom_transform_points

from arguments import ModelParams, PipelineParams
from scene import Scene
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from utils.general_utils import safe_state


def my_render(viewpoint_camera, pc, pipe, bg_color, scaling_modifier=1.0, separate_sh=False, override_color=None, use_trained_exp=False):
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except Exception:
        pass

    import math
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    try:
        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform,
            sh_degree=pc.active_sh_degree,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            debug=True,
            antialiasing=False,
        )
    except TypeError:
        raster_settings = GaussianRasterizationSettings(
            int(viewpoint_camera.image_height),
            int(viewpoint_camera.image_width),
            tanfovx,
            tanfovy,
            bg_color,
            scaling_modifier,
            viewpoint_camera.world_view_transform,
            viewpoint_camera.full_proj_transform,
            pc.active_sh_degree,
            viewpoint_camera.camera_center,
            False,
            True,
            False,
        )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    scales = None
    rotations = None
    cov3D_precomp = None
    if getattr(pipe, "compute_cov3D_python", False):
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    shs = None
    colors_precomp = None
    dc = None

    if override_color is None:
        if getattr(pipe, "convert_SHs_python", False):
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree + 1) ** 2)
            dir_pp = pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1)
            dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            if separate_sh:
                dc, shs = pc.get_features_dc, pc.get_features_rest
            else:
                shs = pc.get_features
    else:
        colors_precomp = override_color

    try:
        if separate_sh:
            rendered_image, radii, invdepth_image = rasterizer(
                means3D=means3D,
                means2D=means2D,
                dc=dc,
                shs=shs,
                colors_precomp=colors_precomp,
                opacities=opacity,
                scales=scales,
                rotations=rotations,
                cov3D_precomp=cov3D_precomp,
            )
        else:
            rendered_image, radii, invdepth_image = rasterizer(
                means3D=means3D,
                means2D=means2D,
                shs=shs,
                colors_precomp=colors_precomp,
                opacities=opacity,
                scales=scales,
                rotations=rotations,
                cov3D_precomp=cov3D_precomp,
            )

        if invdepth_image is not None:
            valid_mask = invdepth_image > 0
            depth_image = torch.zeros_like(invdepth_image)

            min_invdepth = 0.001
            clamped_invdepth = torch.clamp(invdepth_image, min=min_invdepth)
            depth_image[valid_mask] = 1.0 / clamped_invdepth[valid_mask]

            if valid_mask.any():
                print(
                    f"[DEBUG] Depth range: "
                    f"[{float(depth_image[valid_mask].min()):.2f}, "
                    f"{float(depth_image[valid_mask].max()):.2f}]"
                )
        else:
            depth_image = None

    except Exception as e:
        print(f"[WARN] Rasterizer call failed with depth, trying without: {e}")
        if separate_sh:
            rendered_image, radii = rasterizer(
                means3D=means3D,
                means2D=means2D,
                dc=dc,
                shs=shs,
                colors_precomp=colors_precomp,
                opacities=opacity,
                scales=scales,
                rotations=rotations,
                cov3D_precomp=cov3D_precomp,
            )
        else:
            rendered_image, radii = rasterizer(
                means3D=means3D,
                means2D=means2D,
                shs=shs,
                colors_precomp=colors_precomp,
                opacities=opacity,
                scales=scales,
                rotations=rotations,
                cov3D_precomp=cov3D_precomp,
            )
        depth_image = None

    if use_trained_exp:
        exposure = pc.get_exposure_from_name(viewpoint_camera.image_name)
        rendered_image = (
            torch.matmul(rendered_image.permute(1, 2, 0), exposure[:3, :3])
            .permute(2, 0, 1)
            + exposure[:3, 3, None, None]
        )

    rendered_image = rendered_image.clamp(0, 1)

    return {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter": (radii > 0).nonzero(),
        "radii": radii,
        "depth": depth_image,
    }


def save_depth_formats(depth, valid_mask, output_base_path, img_name, global_depth_range=None):
    H, W = depth.shape[1], depth.shape[2]

    metric_depth = depth.squeeze(0).cpu().numpy()

    depth_mm = (metric_depth * 1000).astype(np.uint32)
    Image.fromarray(depth_mm).save(os.path.join(output_base_path, "depth_metric", f"{img_name}.png"))

    if global_depth_range is not None:
        depth_min, depth_max = global_depth_range
        print(f"[DEBUG] Using global depth range: [{depth_min:.2f}, {depth_max:.2f}] for {img_name}")
    else:
        valid_depth = metric_depth[valid_mask.squeeze(0).cpu().numpy() > 0]
        if len(valid_depth) > 0:
            depth_min = np.percentile(valid_depth, 1)
            depth_max = np.percentile(valid_depth, 99)
        else:
            depth_min, depth_max = 0, 1

    if abs(depth_max - depth_min) < 1e-6:
        depth_max = depth_min + 1e-6

    depth_vis = metric_depth.copy()
    depth_vis = np.clip(depth_vis, depth_min, depth_max)
    depth_vis = 255 * ((depth_vis - depth_min) / (depth_max - depth_min + 1e-6))
    depth_vis[metric_depth == 0] = 0
    depth_vis = depth_vis.astype(np.uint8)

    Image.fromarray(depth_vis).save(os.path.join(output_base_path, "depth_visual", f"{img_name}.png"))

    depth_rgba = np.zeros((H, W, 4), dtype=np.uint8)
    depth_rgba[:, :, 0] = depth_vis
    depth_rgba[:, :, 1] = depth_vis
    depth_rgba[:, :, 2] = depth_vis
    depth_rgba[:, :, 3] = (valid_mask.squeeze(0).cpu().numpy() * 255).astype(np.uint8)

    Image.fromarray(depth_rgba).save(os.path.join(output_base_path, "depth_rgba", f"{img_name}.png"))

    mask = (valid_mask.squeeze(0).cpu().numpy() * 255).astype(np.uint8)
    Image.fromarray(mask).save(os.path.join(output_base_path, "depth_mask", f"{img_name}.png"))


def compute_depth_range_from_colmap(sparse_path, percentile_min=5, percentile_max=95, max_images=None):
    import struct

    print("[INFO] Computing depth range from COLMAP sparse reconstruction...")

    points3d_file = os.path.join(sparse_path, "points3D.txt")
    points = []

    if os.path.exists(points3d_file):
        with open(points3d_file, "r") as f:
            for line in f:
                if line.startswith("#"):
                    continue
                parts = line.strip().split()
                if len(parts) >= 4:
                    x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                    points.append([x, y, z])

    if len(points) == 0:
        points3d_bin = os.path.join(sparse_path, "points3D.bin")
        if os.path.exists(points3d_bin):
            with open(points3d_bin, "rb") as f:
                num_points = struct.unpack("Q", f.read(8))[0]
                for _ in range(num_points):
                    _point_id = struct.unpack("Q", f.read(8))[0]
                    xyz = struct.unpack("ddd", f.read(24))
                    _rgb = struct.unpack("BBB", f.read(3))
                    _error = struct.unpack("d", f.read(8))[0]
                    track_length = struct.unpack("Q", f.read(8))[0]
                    for _ in range(track_length):
                        f.read(8)
                        f.read(8)
                    points.append(list(xyz))

    if len(points) == 0:
        print("[WARN] No 3D points found in COLMAP reconstruction!")
        return 0.1, 100.0

    points = np.array(points)
    print(f"[INFO] Loaded {len(points)} 3D points")

    images_file = os.path.join(sparse_path, "images.txt")
    camera_centers = []

    if os.path.exists(images_file):
        with open(images_file, "r") as f:
            lines = f.readlines()

            i = 0
            while i < len(lines):
                if max_images is not None and max_images > 0 and len(camera_centers) >= max_images:
                    break

                line = lines[i].strip()

                if (not line) or line.startswith("#"):
                    i += 1
                    continue

                parts = line.split()
                if len(parts) >= 10:
                    qw, qx, qy, qz = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                    tx, ty, tz = float(parts[5]), float(parts[6]), float(parts[7])

                    R = np.array([
                        [1 - 2 * qy ** 2 - 2 * qz ** 2, 2 * qx * qy - 2 * qz * qw, 2 * qx * qz + 2 * qy * qw],
                        [2 * qx * qy + 2 * qz * qw, 1 - 2 * qx ** 2 - 2 * qz ** 2, 2 * qy * qz - 2 * qx * qw],
                        [2 * qx * qz - 2 * qy * qw, 2 * qy * qz + 2 * qx * qw, 1 - 2 * qx ** 2 - 2 * qy ** 2],
                    ])

                    t = np.array([tx, ty, tz])
                    C = -R.T @ t
                    camera_centers.append(C)

                    i += 2
                else:
                    i += 1

    if len(camera_centers) == 0:
        images_bin = os.path.join(sparse_path, "images.bin")
        if os.path.exists(images_bin):
            with open(images_bin, "rb") as f:
                num_images = struct.unpack("Q", f.read(8))[0]
                for _ in range(num_images):
                    if max_images is not None and max_images > 0 and len(camera_centers) >= max_images:
                        break

                    _image_id = struct.unpack("I", f.read(4))[0]
                    qw, qx, qy, qz = struct.unpack("dddd", f.read(32))
                    tx, ty, tz = struct.unpack("ddd", f.read(24))
                    _camera_id = struct.unpack("I", f.read(4))[0]

                    while True:
                        char = f.read(1)
                        if char == b"\x00":
                            break

                    num_points2D = struct.unpack("Q", f.read(8))[0]
                    for _ in range(num_points2D):
                        f.read(24)

                    R = np.array([
                        [1 - 2 * qy ** 2 - 2 * qz ** 2, 2 * qx * qy - 2 * qz * qw, 2 * qx * qz + 2 * qy * qw],
                        [2 * qx * qy + 2 * qz * qw, 1 - 2 * qx ** 2 - 2 * qz ** 2, 2 * qy * qz - 2 * qx * qw],
                        [2 * qx * qz - 2 * qy * qw, 2 * qy * qz + 2 * qx * qw, 1 - 2 * qx ** 2 - 2 * qy ** 2],
                    ])

                    t = np.array([tx, ty, tz])
                    C = -R.T @ t
                    camera_centers.append(C)

    if len(camera_centers) == 0:
        print("[WARN] No camera positions found!")
        min_coords = np.min(points, axis=0)
        max_coords = np.max(points, axis=0)
        scene_scale = np.linalg.norm(max_coords - min_coords)
        return scene_scale * 0.01, scene_scale * 0.5

    camera_centers = np.array(camera_centers)
    print(f"[INFO] Loaded {len(camera_centers)} camera positions")

    all_distances = []
    for cam_center in camera_centers:
        distances = np.linalg.norm(points - cam_center, axis=1)
        all_distances.extend(distances)

    all_distances = np.array(all_distances)

    depth_min = np.percentile(all_distances, percentile_min)
    depth_max = np.percentile(all_distances, percentile_max)

    if depth_max - depth_min < 0.1:
        depth_max = depth_min + 1.0

    depth_max = min(depth_max, depth_min * 20)

    print(f"[INFO] Computed depth range from COLMAP: [{depth_min:.2f}, {depth_max:.2f}] meters")
    print(
        f"[INFO] Distance statistics: "
        f"min={all_distances.min():.2f}, "
        f"max={all_distances.max():.2f}, "
        f"median={np.median(all_distances):.2f}"
    )

    return depth_min, depth_max


def main():
    parser = argparse.ArgumentParser(description="Render RGBA and depth from COLMAP sparse/0 using 3DGS")

    model = ModelParams(parser)
    pipeline = PipelineParams(parser)

    parser.add_argument("--output_path", type=str, required=True, help="输出结果保存目录")
    parser.add_argument("--iteration", default=None, type=int, help="若提供则加载已训练迭代，否则用点云初始化")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--zfar", type=float, default=100.0, help="渲染投影的远裁剪面，增大以覆盖大尺度场景")
    parser.add_argument("--init_scale_factor", type=float, default=0.5, help="初始化球半径为KNN距离的倍数，默认0.5")
    parser.add_argument("--init_scale_uniform", type=float, default=0.001, help="如果指定，使用统一的球半径值而不是KNN计算")
    parser.add_argument("--max_images", type=int, default=100, help="最多渲染多少张图片。默认100；设为-1或0表示处理全部图片。")

    args = parser.parse_args()

    safe_state(args.quiet)

    dataset = model.extract(args)
    pipe = pipeline.extract(args)

    dataset.max_images = args.max_images
    dataset.data_device = "cpu"

    pipe.debug = True
    pipe.antialiasing = False

    if getattr(dataset, "model_path", None) in (None, ""):
        dataset.model_path = os.path.abspath(args.output_path)

    os.makedirs(dataset.model_path, exist_ok=True)

    sp = os.path.abspath(dataset.source_path)

    if os.path.isdir(os.path.join(sp, "sparse")):
        pass
    elif os.path.basename(sp).lower() == "sparse":
        dataset.source_path = os.path.dirname(sp)
    elif os.path.basename(sp) == "0" and os.path.basename(os.path.dirname(sp)).lower() == "sparse":
        dataset.source_path = os.path.dirname(os.path.dirname(sp))
    else:
        raise ValueError(f"source_path 不包含 sparse/0: {dataset.source_path}")

    dataset.sh_degree = 0

    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)

    if args.iteration is None:
        num_pts = gaussians.get_xyz.shape[0]

        target_alpha = torch.full((num_pts, 1), 0.99, dtype=torch.float32, device="cuda")
        gaussians._opacity.data = gaussians.inverse_opacity_activation(target_alpha)

        if args.init_scale_uniform is not None:
            print(f"[INFO] Setting uniform initial scale: {args.init_scale_uniform}")
            uniform_scale = torch.full((num_pts, 3), args.init_scale_uniform, dtype=torch.float32, device="cuda")
            gaussians._scaling.data = gaussians.scaling_inverse_activation(uniform_scale)
        else:
            print(f"[INFO] Scaling initial scales by factor: {args.init_scale_factor}")
            current_scales = gaussians.get_scaling
            new_scales = current_scales * args.init_scale_factor
            gaussians._scaling.data = gaussians.scaling_inverse_activation(new_scales)

        gaussians.active_sh_degree = 0
        scene.save(0)

    ply_path_candidates = [
        os.path.join(dataset.model_path, "point_cloud", "iteration_0", "point_cloud.ply"),
        os.path.join(dataset.model_path, "point_cloud", "iteration_0", "level_0.ply"),
    ]

    ply_path = None
    for p in ply_path_candidates:
        if os.path.exists(p):
            ply_path = p
            break

    if ply_path is None:
        raise FileNotFoundError(
            "Cannot find saved ply. Tried:\n" + "\n".join(ply_path_candidates)
        )

    gaussians_loaded = GaussianModel(dataset.sh_degree)
    gaussians_loaded.load_ply(ply_path)
    gaussians_loaded.active_sh_degree = 0

    del gaussians
    torch.cuda.empty_cache()

    os.makedirs(args.output_path, exist_ok=True)

    color_dir = os.path.join(args.output_path, "color")
    rgba_dir = os.path.join(args.output_path, "rgba")
    depth_metric_dir = os.path.join(args.output_path, "depth_metric")
    depth_visual_dir = os.path.join(args.output_path, "depth_visual")
    depth_rgba_dir = os.path.join(args.output_path, "depth_rgba")
    depth_mask_dir = os.path.join(args.output_path, "depth_mask")

    for d in [color_dir, rgba_dir, depth_metric_dir, depth_visual_dir, depth_rgba_dir, depth_mask_dir]:
        os.makedirs(d, exist_ok=True)

    bg_black = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device="cuda")
    bg_white = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32, device="cuda")

    train_cams = scene.getTrainCameras()
    test_cams = scene.getTestCameras()

    for cam in train_cams + test_cams:
        cam.zfar = float(args.zfar)
        cam.projection_matrix = getProjectionMatrix(
            znear=cam.znear,
            zfar=cam.zfar,
            fovX=cam.FoVx,
            fovY=cam.FoVy
        ).transpose(0, 1).cuda()

        cam.full_proj_transform = (
            cam.world_view_transform.unsqueeze(0)
            .bmm(cam.projection_matrix.unsqueeze(0))
        ).squeeze(0)

    cams = train_cams + test_cams
    total_cams = len(cams)

    if args.max_images is not None and args.max_images > 0:
        cams = cams[:args.max_images]
        print(f"[INFO] max_images={args.max_images}, rendering first {len(cams)} / {total_cams} cameras.")
    else:
        print(f"[INFO] max_images={args.max_images}, rendering all {total_cams} cameras.")

    if len(cams) == 0:
        print("[WARN] No cameras to render. Please check dataset or max_images.")
        return

    sparse_path = os.path.join(dataset.source_path, "sparse", "0")

    if os.path.exists(sparse_path):
        global_depth_range = compute_depth_range_from_colmap(
            sparse_path,
            percentile_min=2,
            percentile_max=98,
            max_images=args.max_images
        )
    else:
        print("[WARN] COLMAP sparse folder not found, using default depth range")
        global_depth_range = 0.1, 100.0

    with torch.no_grad():
        for idx, cam in enumerate(tqdm(cams, desc="Rendering cameras")):
            out_black = my_render(cam, gaussians_loaded, pipe, bg_black)
            out_white = my_render(cam, gaussians_loaded, pipe, bg_white)

            if idx == 0:
                print(f"[DEBUG] Camera {cam.image_name}: render output keys: {list(out_black.keys())}")
                if out_black["depth"] is not None:
                    d = out_black["depth"]
                    print(f"[DEBUG] Depth shape: {d.shape}, range: [{float(d.min()):.3f}, {float(d.max()):.3f}]")

            rgb_black = out_black["render"]
            rgb_white = out_white["render"]

            depth = out_black["depth"]

            if depth is None:
                print(f"[WARN] No depth returned for {cam.image_name}, using zero depth map.")
                H, W = rgb_black.shape[1], rgb_black.shape[2]
                depth = torch.zeros(1, H, W, dtype=torch.float32, device="cuda")

            alpha = 1.0 - (rgb_white - rgb_black).mean(dim=0, keepdim=True).clamp(0.0, 1.0)

            eps = 1e-6
            color = (rgb_black / (alpha + eps)).clamp(0.0, 1.0)

            valid_mask = (depth > 0).float()

            img_name = os.path.splitext(cam.image_name)[0]

            Image.fromarray(
                (color.permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
            ).save(os.path.join(color_dir, f"{img_name}.png"))

            rgba_img = torch.cat([color, alpha], dim=0)
            rgba_np = (rgba_img.permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
            Image.fromarray(rgba_np).save(os.path.join(rgba_dir, f"{img_name}.png"))

            save_depth_formats(
                depth,
                valid_mask,
                args.output_path,
                img_name,
                global_depth_range=global_depth_range
            )

            del out_black, out_white, rgb_black, rgb_white, depth, alpha, color, valid_mask
            torch.cuda.empty_cache()

    print(f"[INFO] Rendering complete. Results saved to {args.output_path}")


if __name__ == "__main__":
    main()