#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
from utils.graphics_utils import getProjectionMatrix, geom_transform_points

from arguments import ModelParams, PipelineParams
from scene import Scene
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from utils.general_utils import safe_state

def save_depth_rgba(path, depth_tensor, valid_mask):
    """保存深度图为RGBA格式，无效区域为透明"""
    depth = depth_tensor.squeeze(0).cpu().numpy()
    mask = valid_mask.squeeze(0).cpu().numpy().astype(bool)
    
    # 计算深度可视化
    vis = np.zeros_like(depth, dtype=np.float32)
    if mask.any():
        valid_depths = depth[mask]
        
        # 使用IQR方法确定合理范围
        median_depth = np.median(valid_depths)
        q1 = np.percentile(valid_depths, 25)
        q3 = np.percentile(valid_depths, 75)
        iqr = q3 - q1
        
        dmin = max(valid_depths.min(), q1 - 1.5 * iqr)
        dmax = min(valid_depths.max(), q3 + 3.0 * iqr)
        
        # 如果范围太大，限制在合理范围内
        if dmax > 5 * median_depth:
            dmax = np.percentile(valid_depths, 80)
        
        # 线性映射到0-1
        if dmax > dmin:
            vis = np.clip((depth - dmin) / (dmax - dmin + 1e-8), 0, 1)
            vis[~mask] = 0
        else:
            vis[mask] = 0.5
        
        # 反转深度图（近处亮，远处暗）
        vis = 1.0 - vis
        vis[~mask] = 0
    
    # 创建RGBA图像
    # 深度可视化作为灰度图复制到RGB三个通道
    rgb = np.stack([vis, vis, vis], axis=-1)  # HxWx3
    
    # Alpha通道：有效区域为1，无效区域为0
    alpha = mask.astype(np.float32)
    
    # 合并为RGBA
    rgba = np.concatenate([rgb, alpha[..., np.newaxis]], axis=-1)  # HxWx4
    
    # 转换为uint8并保存
    rgba_u8 = (rgba * 255.0).astype(np.uint8)
    Image.fromarray(rgba_u8, mode="RGBA").save(path)

def save_depth_visual(path_png_vis, path_png_raw, depth_tensor, valid_mask):
    # depth_tensor: 1xHxW float32 (depth in camera space), valid_mask: 1xHxW in {0,1}
    depth = depth_tensor.squeeze(0).cpu().numpy()
    mask = valid_mask.squeeze(0).cpu().numpy().astype(bool)

    # Visualization 0-255 within valid pixels range
    vis = np.zeros_like(depth, dtype=np.float32)
    if mask.any():
        valid_depths = depth[mask]
        
        # 使用中位数的倍数来确定合理范围
        median_depth = np.median(valid_depths)
        
        # 方法1：使用IQR（四分位距）来确定合理范围
        q1 = np.percentile(valid_depths, 25)
        q3 = np.percentile(valid_depths, 75)
        iqr = q3 - q1
        
        # 定义合理范围：中位数附近的一定倍数内
        # 或者使用 Q1-1.5*IQR 到 Q3+1.5*IQR 的标准异常值检测范围
        dmin = max(valid_depths.min(), q1 - 1.5 * iqr)
        dmax = min(valid_depths.max(), q3 + 3.0 * iqr)  # 使用3倍IQR以包含更多数据
        
        # 如果范围仍然太大，使用80%百分位数
        if dmax > 5 * median_depth:
            dmax = np.percentile(valid_depths, 80)
        
        print(f"[DEBUG] Depth vis range: [{dmin:.2f}, {dmax:.2f}] (median: {median_depth:.2f})")
        
        # 线性映射到0-1
        if dmax > dmin:
            # 对所有像素应用相同的映射
            vis = np.clip((depth - dmin) / (dmax - dmin + 1e-8), 0, 1)
            # 将无效像素设为0
            vis[~mask] = 0
        else:
            vis[mask] = 0.5
        
        # 反转深度图（近处亮，远处暗）
        vis = 1.0 - vis
        # 确保无效区域是黑色
        vis[~mask] = 0
        
    vis_u8 = (vis * 255.0).astype(np.uint8)
    Image.fromarray(vis_u8, mode="L").save(path_png_vis)

    # Save raw depth as numpy array
    raw_dir = os.path.dirname(path_png_raw)
    os.makedirs(raw_dir, exist_ok=True)
    raw_base = os.path.splitext(os.path.basename(path_png_raw))[0]
    np.save(os.path.join(raw_dir, raw_base + ".npy"), depth)

    # Also save as 16-bit PNG with scaling
    if mask.any():
        valid_depths = depth[mask]
        # 使用相同的范围
        q1 = np.percentile(valid_depths, 25)
        q3 = np.percentile(valid_depths, 75)
        iqr = q3 - q1
        lo = max(valid_depths.min(), q1 - 1.5 * iqr)
        hi = min(valid_depths.max(), q3 + 3.0 * iqr)
        
        if hi > lo:
            depth_normalized = np.clip((depth - lo) / (hi - lo), 0.0, 1.0)
        else:
            depth_normalized = np.ones_like(depth) * 0.5
    else:
        depth_normalized = np.zeros_like(depth)
    
    depth_16u = (depth_normalized * 65535.0 + 0.5).astype(np.uint16)
    Image.fromarray(depth_16u, mode="I;16").save(path_png_raw)

def save_png_rgba(path, rgb_tensor, alpha_tensor):
    # rgb_tensor, alpha_tensor: torch tensors in [0,1], shapes CxHxW (C=3) and 1xHxW
    rgb = (rgb_tensor.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
    a = (alpha_tensor.clamp(0, 1).squeeze(0).cpu().numpy() * 255.0).astype(np.uint8)
    rgba = np.dstack([rgb, a])
    Image.fromarray(rgba, mode="RGBA").save(path)


def save_depth_visual(path_png_vis, path_png_raw, depth_tensor, valid_mask):
    # depth_tensor: 1xHxW float32 (depth in camera space), valid_mask: 1xHxW in {0,1}
    depth = depth_tensor.squeeze(0).cpu().numpy()
    mask = valid_mask.squeeze(0).cpu().numpy().astype(bool)

    # Visualization 0-255 within valid pixels range
    vis = np.zeros_like(depth, dtype=np.float32)
    if mask.any():
        valid_depths = depth[mask]
        # Use percentiles to handle outliers
        dmin = float(np.percentile(valid_depths, 1))
        dmax = float(np.percentile(valid_depths, 99))
        if dmax > dmin:
            vis[mask] = np.clip((depth[mask] - dmin) / (dmax - dmin), 0, 1)
        else:
            vis[mask] = 0.0
    vis_u8 = (vis * 255.0).astype(np.uint8)
    Image.fromarray(vis_u8, mode="L").save(path_png_vis)

    # Save raw depth as numpy array
    raw_dir = os.path.dirname(path_png_raw)
    os.makedirs(raw_dir, exist_ok=True)
    raw_base = os.path.splitext(os.path.basename(path_png_raw))[0]
    np.save(os.path.join(raw_dir, raw_base + ".npy"), depth)

    # Also save as 16-bit PNG with scaling
    if mask.any():
        valid_depths = depth[mask]
        lo = float(np.percentile(valid_depths, 1))
        hi = float(np.percentile(valid_depths, 99))
        if hi > lo:
            depth_normalized = np.clip((depth - lo) / (hi - lo), 0.0, 1.0)
        else:
            depth_normalized = np.zeros_like(depth)
    else:
        depth_normalized = np.zeros_like(depth)
    
    depth_16u = (depth_normalized * 65535.0 + 0.5).astype(np.uint16)
    Image.fromarray(depth_16u, mode="I;16").save(path_png_raw)


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
        # 在 my_render 函数中，转换逆深度到深度后添加调试信息
        if invdepth_image is not None:
            # 先看看逆深度的范围
            print(f"[DEBUG] Invdepth range: [{float(invdepth_image.min()):.6f}, {float(invdepth_image.max()):.6f}]")
            
            # 逆深度为0的地方表示无效像素
            valid_mask = invdepth_image > 0
            depth_image = torch.zeros_like(invdepth_image)
            
            # 检查极小的逆深度值（会产生极大的深度）
            very_small_invdepth = (invdepth_image > 0) & (invdepth_image < 0.0001)
            #if very_small_invdepth.any():
                #print(f"[WARNING] Found {very_small_invdepth.sum().item()} pixels with very small invdepth < 0.0001")
            
            # 设置一个合理的逆深度最小值，避免产生极大深度
            min_invdepth = 0.01  # 对应最大深度100
            clamped_invdepth = torch.clamp(invdepth_image, min=min_invdepth)
            
            depth_image[valid_mask] = 1.0 / clamped_invdepth[valid_mask]
            
            #print(f"[DEBUG] Depth range after conversion: [{float(depth_image[valid_mask].min()):.2f}, {float(depth_image[valid_mask].max()):.2f}]")
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


def main():
    parser = argparse.ArgumentParser(description="Render RGBA and depth from COLMAP sparse/0 using 3DGS")
    model = ModelParams(parser)
    pipeline = PipelineParams(parser)
    parser.add_argument("--output_path", type=str, required=True, help="输出结果保存目录")
    parser.add_argument("--iteration", default=None, type=int, help="若提供则加载已训练迭代，否则用点云初始化")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--zfar", type=float, default=1000.0, help="渲染投影的远裁剪面，增大以覆盖大尺度场景")
        # 添加球半径控制参数
    parser.add_argument("--init_scale_factor", type=float, default=0.5, 
                       help="初始化球半径为KNN距离的倍数，默认0.5")
    parser.add_argument("--init_scale_uniform", type=float, default=None, 
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

    # 将初始不透明度改为 0.99（仅在从点云初始化时生效）
    # if args.iteration is None:
    #     num_pts = gaussians.get_xyz.shape[0]
    #     target_alpha = torch.full((num_pts, 1), 0.99, dtype=torch.float32, device="cuda")
    #     gaussians._opacity.data = gaussians.inverse_opacity_activation(target_alpha)
    #     gaussians.active_sh_degree = 0
    #     # 保存初始化后的3DGS为PLY
    #     scene.save(0)
    
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
            print('uniform_scale:',uniform_scale)
            gaussians._scaling.data = gaussians.scaling_inverse_activation(uniform_scale)
        else:
            # 使用KNN距离的倍数
            print(f"[INFO] Scaling initial scales by factor: {args.init_scale_factor}")
            current_scales = gaussians.get_scaling  # 获取当前的scale（已经通过KNN计算）
            new_scales = current_scales * args.init_scale_factor
            print('new_scales:',new_scales)
            gaussians._scaling.data = gaussians.scaling_inverse_activation(new_scales)
        
        gaussians.active_sh_degree = 0
        
        # 保存初始化后的3DGS为PLY
        scene.save(0)

    # 二次加载：从保存的PLY读取，再进行渲染
    #ply_path = os.path.join(dataset.model_path, "point_cloud", "iteration_0", "point_cloud.ply")
    ply_path = r"/content/drive/MyDrive/3views_exp/MA_llff_data_3views_virtualimg/outputAlignPc2imgScale05TranRot/point_cloud/iteration_10000/point_cloud.ply"
    gaussians_loaded = GaussianModel(dataset.sh_degree)
    gaussians_loaded.load_ply(ply_path)
    gaussians_loaded.active_sh_degree = 0

    os.makedirs(args.output_path, exist_ok=True)
    color_dir = os.path.join(args.output_path, "color")
    rgba_dir = os.path.join(args.output_path, "rgba")
    depth_vis_dir = os.path.join(args.output_path, "depth_vis")
    depth_rgba_dir = os.path.join(args.output_path, "depth_rgba")  # 新增RGBA深度图目录
    depth_raw_dir = os.path.join(args.output_path, "depth_raw")
    mask_dir = os.path.join(args.output_path, "mask")
    
    for d in [color_dir, rgba_dir, depth_vis_dir, depth_rgba_dir, depth_raw_dir, mask_dir]:
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

    with torch.no_grad():
        for idx, cam in enumerate(tqdm(cams, desc="Rendering cameras")):
            out_black = my_render(cam, gaussians_loaded, pipe, bg_black)
            out_white = my_render(cam, gaussians_loaded, pipe, bg_white)

            if idx == 0:
                print(f"[DEBUG] Camera {cam.image_name}: render output keys: {list(out_black.keys())}")
                if out_black["depth"] is not None:
                    d = out_black["depth"]
                    #print(f"[DEBUG] Depth shape: {d.shape}, range: [{float(d.min()):.3f}, {float(d.max()):.3f}]")

            rgb_black = out_black["render"]  # 3xHxW
            rgb_white = out_white["render"]  # 3xHxW

            # 从rasterizer获取的深度
            depth_from_raster = out_black["depth"]
            
            # 如果rasterizer没有返回深度，使用Z-buffer方法计算
            if depth_from_raster is None:
                print(f"[WARN] No depth from rasterizer for {cam.image_name}, using Z-buffer fallback")
                H, W = rgb_black.shape[1], rgb_black.shape[2]
                # Z-buffer depth calculation
                pts_cam = geom_transform_points(gaussians_loaded.get_xyz, cam.world_view_transform)
                z_cam = pts_cam[:, 2]  # Note: in view space, z is typically negative, but we want positive depth
                z_cam = -z_cam  # Convert to positive depth
                
                pts_ndc = geom_transform_points(gaussians_loaded.get_xyz, cam.full_proj_transform)
                u = (pts_ndc[:, 0] * 0.5 + 0.5) * (W - 1)
                v = (1.0 - (pts_ndc[:, 1] * 0.5 + 0.5)) * (H - 1)
                
                # Only consider points in front of camera and within NDC bounds
                valid = (pts_ndc[:, 0].abs() <= 1) & (pts_ndc[:, 1].abs() <= 1) & (z_cam > 0)
                
                if valid.any():
                    u_i = u[valid].round().clamp(0, W - 1).long()
                    v_i = v[valid].round().clamp(0, H - 1).long()
                    z_vals = z_cam[valid]
                    
                    # Create depth buffer
                    depth_img = torch.full((H, W), float('inf'), dtype=torch.float32, device='cuda')
                    
                    # Scatter minimum depth values
                    for i in range(len(u_i)):
                        ui, vi, zi = int(u_i[i]), int(v_i[i]), float(z_vals[i])
                        if zi < depth_img[vi, ui]:
                            depth_img[vi, ui] = zi
                    
                    # Set invalid pixels to 0
                    depth_img[depth_img == float('inf')] = 0.0
                    depth = depth_img.unsqueeze(0)
                else:
                    print(f"[WARN] No valid points for Z-buffer depth for {cam.image_name}")
                    depth = torch.zeros(1, H, W, dtype=torch.float32, device='cuda')
            else:
                depth = depth_from_raster

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
            
            # 保存RGBA
            save_png_rgba(os.path.join(rgba_dir, f"{img_name}.png"), color, alpha)

            # 保存掩码
            mask_u8 = (valid_mask.squeeze(0).cpu().numpy() * 255.0).astype(np.uint8)
            Image.fromarray(mask_u8, mode="L").save(os.path.join(mask_dir, f"{img_name}.png"))

            # 保存深度前添加统计信息
            # if idx == 0 or idx % 10 == 0:
            #     depth_np = depth.squeeze(0).cpu().numpy()
            #     valid_depth = depth_np[depth_np > 0]
            #     if len(valid_depth) > 0:
            #         print(f"\n[INFO] {cam.image_name} depth statistics:")
            #         print(f"  - Valid pixels: {len(valid_depth)} / {depth_np.size} ({100*len(valid_depth)/depth_np.size:.1f}%)")
            #         print(f"  - Range: [{valid_depth.min():.2f}, {valid_depth.max():.2f}]")
            #         print(f"  - Mean: {valid_depth.mean():.2f}, Median: {np.median(valid_depth):.2f}")
            #         print(f"  - Percentiles [5%, 25%, 75%, 95%]: "
            #               f"[{np.percentile(valid_depth, 5):.2f}, "
            #               f"{np.percentile(valid_depth, 25):.2f}, "
            #               f"{np.percentile(valid_depth, 75):.2f}, "
            #               f"{np.percentile(valid_depth, 95):.2f}]")

            # 保存深度
            save_depth_visual(
                os.path.join(depth_vis_dir, f"{img_name}.png"),
                os.path.join(depth_raw_dir, f"{img_name}.png"),
                depth,
                valid_mask,
            )

            # 保存RGBA格式的深度图
            save_depth_rgba(
                os.path.join(depth_rgba_dir, f"{img_name}.png"),
                depth,
                valid_mask
            )


if __name__ == "__main__":
    main()