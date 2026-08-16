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

    # 兼容不同签名构造 GaussianRasterizationSettings
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
            debug=True,  # 确保返回深度信息
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
            True,  # debug=True
            False,  # antialiasing=False
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
    if override_color is None:
        if getattr(pipe, "convert_SHs_python", False):
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree + 1) ** 2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
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

    # 调用rasterizer
    try:
        if separate_sh:
            rendered_image, radii, invdepth_image, _normal_image = rasterizer(
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
            rendered_image, radii, invdepth_image, _normal_image = rasterizer(
                means3D=means3D,
                means2D=means2D,
                shs=shs,
                colors_precomp=colors_precomp,
                opacities=opacity,
                scales=scales,
                rotations=rotations,
                cov3D_precomp=cov3D_precomp,
            )
            
        # 将逆深度转换为深度
        if invdepth_image is not None:
            # 逆深度为0的地方表示无效像素
            valid_mask = invdepth_image > 0
            depth_image = torch.zeros_like(invdepth_image)
            
            # 设置一个合理的逆深度最小值，避免产生极大深度
            min_invdepth = 0.001  # 对应最大深度1000
            clamped_invdepth = torch.clamp(invdepth_image, min=min_invdepth)
            
            depth_image[valid_mask] = 1.0 / clamped_invdepth[valid_mask]
            
            print(f"[DEBUG] Depth range: [{float(depth_image[valid_mask].min()):.2f}, {float(depth_image[valid_mask].max()):.2f}]")
        else:
            depth_image = None

    except Exception as e:
        print(f"[WARN] Rasterizer call failed with depth, trying without: {e}")
        if separate_sh:
            rendered_image, radii, _invdepth2, _normal2 = rasterizer(
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
            rendered_image, radii, _invdepth2, _normal2 = rasterizer(
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
        rendered_image = torch.matmul(rendered_image.permute(1, 2, 0), exposure[:3, :3]).permute(2, 0, 1) + exposure[:3, 3, None, None]

    rendered_image = rendered_image.clamp(0, 1)
    return {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter": (radii > 0).nonzero(),
        "radii": radii,
        "depth": depth_image,
    }

def save_depth_formats(depth, valid_mask, output_base_path, img_name, global_depth_range=None):
    """
    保存4种深度格式
    depth: 1xHxW tensor，metric尺度的深度图
    valid_mask: 1xHxW tensor，有效像素掩码
    global_depth_range: (min, max) tuple，全局深度范围
    """
    H, W = depth.shape[1], depth.shape[2]
    
    # 1. Metric尺度的单通道深度图
    metric_depth = depth.squeeze(0).cpu().numpy()  # HxW

    # 保存为numpy数组（.npy格式），保留完整精度
    #np.save(os.path.join(output_base_path, "depth_metric", f"{img_name}.npy"), metric_depth)

    # 也保存为32位PNG（毫米为单位），使用OpenCV
    # uint32 支持范围 0-4294967295，足够存储毫米级深度（最大 4294 km）
    depth_mm = (metric_depth * 1000).astype(np.uint32)  # 转换为毫米，使用 uint32

    Image.fromarray(depth_mm).save(os.path.join(output_base_path, "depth_metric", f"{img_name}.png"))

    # 2. 归一化到0-255的可视化深度图
    if global_depth_range is not None:
        depth_min, depth_max = global_depth_range
        print(f"[DEBUG] Using global depth range: [{depth_min:.2f}, {depth_max:.2f}] for {img_name}")
    else:
        # 回退到局部范围
        valid_depth = metric_depth[valid_mask.squeeze(0).cpu().numpy() > 0]
        if len(valid_depth) > 0:
            depth_min = np.percentile(valid_depth, 1)
            depth_max = np.percentile(valid_depth, 99)
        else:
            depth_min, depth_max = 0, 1
    
    # 归一化（近处暗，远处亮）
    depth_vis = metric_depth.copy()
    depth_vis = np.clip(depth_vis, depth_min, depth_max)
    depth_vis = 255 * ((depth_vis - depth_min) / (depth_max - depth_min + 1e-6))
    depth_vis[metric_depth == 0] = 0  # 无效区域设为0
    depth_vis = depth_vis.astype(np.uint8)
    
    Image.fromarray(depth_vis).save(os.path.join(output_base_path, "depth_visual", f"{img_name}.png"))
    
    # 3. RGBA格式的4通道深度图
    # RGB通道存储归一化深度值，A通道存储有效掩码
    depth_rgba = np.zeros((H, W, 4), dtype=np.uint8)
    depth_rgba[:, :, 0] = depth_vis  # R
    depth_rgba[:, :, 1] = depth_vis  # G
    depth_rgba[:, :, 2] = depth_vis  # B
    depth_rgba[:, :, 3] = (valid_mask.squeeze(0).cpu().numpy() * 255).astype(np.uint8)  # A
    
    Image.fromarray(depth_rgba).save(os.path.join(output_base_path, "depth_rgba", f"{img_name}.png"))
    
    # 4. 有效掩码（二值图）
    mask = (valid_mask.squeeze(0).cpu().numpy() * 255).astype(np.uint8)
    Image.fromarray(mask).save(os.path.join(output_base_path, "depth_mask", f"{img_name}.png"))


def compute_depth_range_from_colmap(sparse_path, percentile_min=5, percentile_max=95):
    """
    从COLMAP的sparse重建结果中计算深度范围
    """
    import struct
    
    print("[INFO] Computing depth range from COLMAP sparse reconstruction...")
    
    # 读取3D点
    points3d_file = os.path.join(sparse_path, "points3D.txt")
    points = []
    
    if os.path.exists(points3d_file):
        with open(points3d_file, 'r') as f:
            for line in f:
                if line.startswith('#'):
                    continue
                parts = line.strip().split()
                if len(parts) >= 4:
                    # POINT3D_ID X Y Z R G B ERROR TRACK[]
                    x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                    points.append([x, y, z])
    
    # 如果txt文件不存在，尝试读取二进制文件
    if len(points) == 0:
        points3d_bin = os.path.join(sparse_path, "points3D.bin")
        if os.path.exists(points3d_bin):
            with open(points3d_bin, "rb") as f:
                num_points = struct.unpack('Q', f.read(8))[0]
                for _ in range(num_points):
                    point_id = struct.unpack('Q', f.read(8))[0]
                    xyz = struct.unpack('ddd', f.read(24))
                    rgb = struct.unpack('BBB', f.read(3))
                    error = struct.unpack('d', f.read(8))[0]
                    track_length = struct.unpack('Q', f.read(8))[0]
                    for _ in range(track_length):
                        f.read(8)  # image_id
                        f.read(8)  # point2D_idx
                    points.append(list(xyz))
    
    if len(points) == 0:
        print("[WARN] No 3D points found in COLMAP reconstruction!")
        return (0.1, 100.0)
    
    points = np.array(points)
    print(f"[INFO] Loaded {len(points)} 3D points")
    
    # 读取相机位置
    images_file = os.path.join(sparse_path, "images.txt")
    camera_centers = []
    
    if os.path.exists(images_file):
        with open(images_file, 'r') as f:
            lines = f.readlines()
            for i in range(0, len(lines), 2):
                if lines[i].startswith('#'):
                    continue
                # 第一行包含：IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME
                parts = lines[i].strip().split()
                if len(parts) >= 8:
                    qw, qx, qy, qz = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                    tx, ty, tz = float(parts[5]), float(parts[6]), float(parts[7])
                    
                    # 从四元数和平移计算旋转矩阵
                    R = np.array([
                        [1 - 2*qy**2 - 2*qz**2, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
                        [2*qx*qy + 2*qz*qw, 1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qx*qw],
                        [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx**2 - 2*qy**2]
                    ])
                    t = np.array([tx, ty, tz])
                    # 相机中心 C = -R^T * t
                    C = -R.T @ t
                    camera_centers.append(C)
    
    if len(camera_centers) == 0:
        # 如果txt不存在，尝试读取二进制文件
        images_bin = os.path.join(sparse_path, "images.bin")
        if os.path.exists(images_bin):
            with open(images_bin, "rb") as f:
                num_images = struct.unpack('Q', f.read(8))[0]
                for _ in range(num_images):
                    image_id = struct.unpack('I', f.read(4))[0]
                    qw, qx, qy, qz = struct.unpack('dddd', f.read(32))
                    tx, ty, tz = struct.unpack('ddd', f.read(24))
                    camera_id = struct.unpack('I', f.read(4))[0]
                    
                    # 读取图像名称
                    name_length = 0
                    while True:
                        char = f.read(1)
                        if char == b'\x00':
                            break
                        name_length += 1
                    
                    # 读取2D点
                    num_points2D = struct.unpack('Q', f.read(8))[0]
                    for _ in range(num_points2D):
                        f.read(24)  # x, y, point3D_id
                    
                    # 计算相机中心
                    R = np.array([
                        [1 - 2*qy**2 - 2*qz**2, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
                        [2*qx*qy + 2*qz*qw, 1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qx*qw],
                        [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx**2 - 2*qy**2]
                    ])
                    t = np.array([tx, ty, tz])
                    C = -R.T @ t
                    camera_centers.append(C)
    
    if len(camera_centers) == 0:
        print("[WARN] No camera positions found!")
        # 使用点云边界估计
        min_coords = np.min(points, axis=0)
        max_coords = np.max(points, axis=0)
        scene_scale = np.linalg.norm(max_coords - min_coords)
        return (scene_scale * 0.01, scene_scale * 0.5)
    
    camera_centers = np.array(camera_centers)
    print(f"[INFO] Loaded {len(camera_centers)} camera positions")
    
    # 计算所有相机到所有点的距离
    all_distances = []
    for cam_center in camera_centers:
        distances = np.linalg.norm(points - cam_center, axis=1)
        all_distances.extend(distances)
    
    all_distances = np.array(all_distances)
    
    # 使用百分位数确定深度范围
    depth_min = np.percentile(all_distances, percentile_min)
    depth_max = np.percentile(all_distances, percentile_max)
    
    # 确保有合理的范围
    if depth_max - depth_min < 0.1:
        depth_max = depth_min + 1.0
    
    # 限制最大深度，避免过大的值
    depth_max = min(depth_max, depth_min * 20)  # 最大深度不超过最小深度的20倍
    
    print(f"[INFO] Computed depth range from COLMAP: [{depth_min:.2f}, {depth_max:.2f}] meters")
    print(f"[INFO] Distance statistics: min={all_distances.min():.2f}, max={all_distances.max():.2f}, "
          f"median={np.median(all_distances):.2f}")
    
    return (depth_min, depth_max)

def main():
    parser = argparse.ArgumentParser(description="Render RGBA and depth from COLMAP sparse/0 using 3DGS")
    model = ModelParams(parser)
    pipeline = PipelineParams(parser)
    parser.add_argument("--output_path", type=str, required=True, help="输出结果保存目录")
    parser.add_argument("--iteration", default=None, type=int, help="若提供则加载已训练迭代，否则用点云初始化")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--zfar", type=float, default=100.0, help="渲染投影的远裁剪面，增大以覆盖大尺度场景")
    parser.add_argument("--init_scale_factor", type=float, default=0.5, 
                       help="初始化球半径为KNN距离的倍数，默认0.5")
    parser.add_argument("--init_scale_uniform", type=float, default=0.001, 
                       help="如果指定，使用统一的球半径值而不是KNN计算")
    
    args = parser.parse_args()

    safe_state(args.quiet)

    dataset = model.extract(args)
    pipe = pipeline.extract(args)
    # 强制配置：关闭抗锯齿并打开debug，确保返回深度
    pipe.debug = True
    pipe.antialiasing = False

    # 若未提供 model_path，则使用 output_path 作为模型工作目录
    if getattr(dataset, "model_path", None) in (None, ""):
        dataset.model_path = os.path.abspath(args.output_path)
    os.makedirs(dataset.model_path, exist_ok=True)

    # 兼容三种传参：数据根目录 / .../sparse / .../sparse/0
    sp = os.path.abspath(dataset.source_path)
    if os.path.isdir(os.path.join(sp, "sparse")):
        pass
    elif os.path.basename(sp).lower() == "sparse":
        dataset.source_path = os.path.dirname(sp)
    elif os.path.basename(sp) == "0" and os.path.basename(os.path.dirname(sp)).lower() == "sparse":
        dataset.source_path = os.path.dirname(os.path.dirname(sp))
    else:
        raise ValueError(f"source_path 不包含 sparse/0: {dataset.source_path}")

    # 强制使用 sh_degree=0
    dataset.sh_degree = 0

    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)

    # 修改初始化参数（仅在从点云初始化时生效）
    if args.iteration is None:
        num_pts = gaussians.get_xyz.shape[0]
        
        # 修改不透明度
        target_alpha = torch.full((num_pts, 1), 0.99, dtype=torch.float32, device="cuda")
        gaussians._opacity.data = gaussians.inverse_opacity_activation(target_alpha)
        
        # 修改球半径（scale）
        if args.init_scale_uniform is not None:
            # 使用统一的球半径
            print(f"[INFO] Setting uniform initial scale: {args.init_scale_uniform}")
            uniform_scale = torch.full((num_pts, 3), args.init_scale_uniform, dtype=torch.float32, device="cuda")
            gaussians._scaling.data = gaussians.scaling_inverse_activation(uniform_scale)
        else:
            # 使用KNN距离的倍数
            print(f"[INFO] Scaling initial scales by factor: {args.init_scale_factor}")
            current_scales = gaussians.get_scaling  # 获取当前的scale（已经通过KNN计算）
            new_scales = current_scales * args.init_scale_factor
            gaussians._scaling.data = gaussians.scaling_inverse_activation(new_scales)
        
        gaussians.active_sh_degree = 0
        
        # 保存初始化后的3DGS为PLY
        scene.save(0)

    # 二次加载：从保存的PLY读取，再进行渲染
    ply_path = os.path.join(dataset.model_path, "point_cloud", "iteration_0", "point_cloud.ply")
    gaussians_loaded = GaussianModel(dataset.sh_degree)
    gaussians_loaded.load_ply(ply_path)
    gaussians_loaded.active_sh_degree = 0

    # 创建输出目录
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

    # 更新所有相机的zfar并重建投影矩阵
    train_cams = scene.getTrainCameras()
    test_cams = scene.getTestCameras()
    for cam in train_cams + test_cams:
        cam.zfar = float(args.zfar)
        cam.projection_matrix = getProjectionMatrix(znear=cam.znear, zfar=cam.zfar, fovX=cam.FoVx, fovY=cam.FoVy).transpose(0,1).cuda()
        cam.full_proj_transform = (cam.world_view_transform.unsqueeze(0).bmm(cam.projection_matrix.unsqueeze(0))).squeeze(0)

    cams = train_cams + test_cams

    # 从COLMAP sparse文件计算全局深度范围
    sparse_path = os.path.join(args.model_path, "sparse", "0")
    if os.path.exists(sparse_path):
        global_depth_range = compute_depth_range_from_colmap(sparse_path,percentile_min=2,percentile_max=98)
    else:
        print("[WARN] COLMAP sparse folder not found, using default depth range")
        global_depth_range = (0.1, 100.0)

    with torch.no_grad():
        for idx, cam in enumerate(tqdm(cams, desc="Rendering cameras")):
            out_black = my_render(cam, gaussians_loaded, pipe, bg_black)
            out_white = my_render(cam, gaussians_loaded, pipe, bg_white)

            if idx == 0:
                print(f"[DEBUG] Camera {cam.image_name}: render output keys: {list(out_black.keys())}")
                if out_black["depth"] is not None:
                    d = out_black["depth"]
                    print(f"[DEBUG] Depth shape: {d.shape}, range: [{float(d.min()):.3f}, {float(d.max()):.3f}]")

            rgb_black = out_black["render"]  # 3xHxW
            rgb_white = out_white["render"]  # 3xHxW

            # 从rasterizer获取的深度
            depth_from_raster = out_black["depth"]
            depth = depth_from_raster
            # # 如果rasterizer没有返回深度，使用Z-buffer方法计算
            # if depth_from_raster is None:
            #     print(f"[WARN] No depth from rasterizer for {cam.image_name}, using Z-buffer fallback")
            #     H, W = rgb_black.shape[1], rgb_black.shape[2]
            #     # Z-buffer depth calculation
            #     pts_cam = geom_transform_points(gaussians_loaded.get_xyz, cam.world_view_transform)
            #     z_cam = pts_cam[:, 2]
            #     z_cam = -z_cam  # Convert to positive depth
                
            #     pts_ndc = geom_transform_points(gaussians_loaded.get_xyz, cam.full_proj_transform)
            #     u = (pts_ndc[:, 0] * 0.5 + 0.5) * (W - 1)
            #     v = (1.0 - (pts_ndc[:, 1] * 0.5 + 0.5)) * (H - 1)
                
            #     # Only consider points in front of camera and within NDC bounds
            #     valid = (pts_ndc[:, 0].abs() <= 1) & (pts_ndc[:, 1].abs() <= 1) & (z_cam > 0)
                
            #     if valid.any():
            #         u_i = u[valid].round().clamp(0, W - 1).long()
            #         v_i = v[valid].round().clamp(0, H - 1).long()
            #         z_vals = z_cam[valid]
                    
            #         # Create depth buffer
            #         depth_img = torch.full((H, W), float('inf'), dtype=torch.float32, device='cuda')
                    
            #         # Scatter minimum depth values
            #         for i in range(len(u_i)):
            #             ui, vi, zi = int(u_i[i]), int(v_i[i]), float(z_vals[i])
            #             if zi < depth_img[vi, ui]:
            #                 depth_img[vi, ui] = zi
                    
            #         # Set invalid pixels to 0
            #         depth_img[depth_img == float('inf')] = 0.0
            #         depth = depth_img.unsqueeze(0)
            #     else:
            #         print(f"[WARN] No valid points for Z-buffer depth for {cam.image_name}")
            #         depth = torch.zeros(1, H, W, dtype=torch.float32, device='cuda')
            # else:
            #     depth = depth_from_raster

            # 由双背景恢复 alpha
            alpha = 1.0 - (rgb_white - rgb_black).mean(dim=0, keepdim=True).clamp(0.0, 1.0)

            # 非预乘颜色
            eps = 1e-6
            color = (rgb_black / (alpha + eps)).clamp(0.0, 1.0)

            # 有效掩码：深度大于0的像素
            valid_mask = (depth > 0).float()

            # 保存结果
            img_name = os.path.splitext(cam.image_name)[0]
            
            # 保存彩色图像
            Image.fromarray((color.permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)).save(
                os.path.join(color_dir, f"{img_name}.png"))
            
            # 保存RGBA图像
            rgba_img = torch.cat([color, alpha], dim=0)  # 4xHxW
            rgba_np = (rgba_img.permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
            Image.fromarray(rgba_np).save(os.path.join(rgba_dir, f"{img_name}.png"))

            # 保存所有深度格式（传入全局深度范围）
            save_depth_formats(depth, valid_mask, args.output_path, img_name, 
                             global_depth_range=global_depth_range)

    print(f"[INFO] Rendering complete. Results saved to {args.output_path}")


if __name__ == "__main__":
    main()