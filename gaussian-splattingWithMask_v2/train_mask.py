#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import time
import math
import torch
import numpy as np
import random
from random import randint
from PIL import Image
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text
from scene.dataset_readers import readColmapCameras
from utils.camera_utils import cameraList_from_camInfos
from utils.general_utils import safe_state, get_expon_lr_func
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
except:
    FUSED_SSIM_AVAILABLE = False

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, val_file=""):

    # 新功能 2：懒加载模式——图片按需从磁盘读取，只缓存最近使用的 lazy_cache 张，防止爆 RAM
    if getattr(dataset, "lazy_load", False):
        from scene.cameras import set_lazy_cache_size
        set_lazy_cache_size(dataset.lazy_cache)
        print("[lazy_load] 已启用图片懒加载，缓存容量 = {} 张".format(dataset.lazy_cache))

    if not SPARSE_ADAM_AVAILABLE and opt.optimizer_type == "sparse_adam":
        sys.exit(f"Trying to use sparse adam but it is not installed, please install the correct rasterizer using pip install [3dgs_accel].")

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    # 参数合法性：形状约束必须配合 2D 椭球初始化使用（否则初始 z 尺度无意义）
    if dataset.freeze_2d_z and not dataset.init_2d:
        sys.exit("[参数错误] --freeze_2d_z 需与 --init_2d 同时开启")
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type, dataset.init_2d, dataset.init_2d_z, dataset.freeze_2d_z)
    print("[实验配置] init_2d(2D椭球初始化)={} / freeze_2d_z(训练中形状约束)={}".format(
        dataset.init_2d, dataset.freeze_2d_z))
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    # 新功能：val10 渲染（--val_file 指向 images-val10.txt，默认空 = 关闭）
    val_cameras = []
    if val_file:
        val_extr = read_extrinsics_text(val_file)
        val_intr = read_intrinsics_text(os.path.join(dataset.source_path, "sparse/0/cameras.txt"))
        # 修复：val 相机同样加载 mask，否则与训练目标（mask 区域渲染为黑）不一致，
        # 指标会被 mask 区域的原图内容系统性拉低
        val_masks_dir = os.path.join(dataset.source_path, dataset.alpha_masks) if dataset.alpha_masks else ""
        val_cam_infos = readColmapCameras(cam_extrinsics=val_extr, cam_intrinsics=val_intr, depths_params=None,
                                          images_folder=os.path.join(dataset.source_path, "images"),
                                          masks_folder=val_masks_dir, depths_folder="", test_cam_names_list=[])
        val_cameras = cameraList_from_camInfos(val_cam_infos, 1.0, dataset, False, True)
        print("[val_file] 已加载 {} 个 val 相机：{}".format(len(val_cameras), val_file))
    # D2：loss 记录到 loss_log.csv；阶段2新增 normal_loss/scale_reg/size_reg 三列
    loss_log_path = os.path.join(scene.model_path, "loss_log.csv")
    if not os.path.exists(loss_log_path):
        with open(loss_log_path, "w") as f:
            f.write("iteration,L1,SSIM_loss,depth_loss,normal_loss,scale_reg,size_reg,total\n")

    # 阶段2新功能：打印法向约束 loss 配置，确认实验变量生效
    print("[法向约束配置] lambda_scale={} / lambda_normal={} / lambda_size={}（0 = 对应项关闭）".format(
        dataset.lambda_scale, dataset.lambda_normal, dataset.lambda_size))

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    use_sparse_adam = opt.optimizer_type == "sparse_adam" and SPARSE_ADAM_AVAILABLE 
    depth_l1_weight = get_expon_lr_func(opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=opt.iterations)

    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))
    ema_loss_for_log = 0.0
    ema_Ll1depth_for_log = 0.0
    dead_view_skips = 0  # 新功能：无有效像素（mask 取反后全 0）视图跳过计数
    train_start_time = time.time()  # 新功能：记录训练总耗时（val 指标 csv 使用）

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifier=scaling_modifer, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_indices = list(range(len(viewpoint_stack)))
        rand_idx = randint(0, len(viewpoint_indices) - 1)
        viewpoint_cam = viewpoint_stack.pop(rand_idx)
        vind = viewpoint_indices.pop(rand_idx)
        # 新功能：跳过无任何有效像素的视图（mask 已在加载时取反，sum=0 即原图 mask 全白、
        # 整帧都是动态物体）；放回重抽，最多重试 100 次避免极端情况死循环
        for _retry in range(100):
            if viewpoint_cam.alpha_mask is None or float(viewpoint_cam.alpha_mask.sum()) > 0.0:
                break
            dead_view_skips += 1
            if dead_view_skips <= 5 or dead_view_skips % 1000 == 0:
                print("[warning] 跳过无有效像素的视图：{}（累计跳过 {}）".format(viewpoint_cam.image_name, dead_view_skips))
            viewpoint_stack.append(viewpoint_cam)
            viewpoint_indices.append(vind)
            if not viewpoint_stack:
                viewpoint_stack = scene.getTrainCameras().copy()
                viewpoint_indices = list(range(len(viewpoint_stack)))
            rand_idx = randint(0, len(viewpoint_indices) - 1)
            viewpoint_cam = viewpoint_stack.pop(rand_idx)
            vind = viewpoint_indices.pop(rand_idx)

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        if viewpoint_cam.alpha_mask is not None:
            alpha_mask = viewpoint_cam.alpha_mask.cuda()
            image *= alpha_mask

        # Loss



        gt_image = viewpoint_cam.original_image.cuda()


        Ll1 = l1_loss(image, gt_image) 
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        if FUSED_SSIM_AVAILABLE:
            ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
        else:
            ssim_value = ssim(image, gt_image)
        #Ll1 = l1_loss(image, gt_image)
        #loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)

        # Depth regularization
        Ll1depth_pure = 0.0
        if depth_l1_weight(iteration) > 0 and viewpoint_cam.depth_reliable:
            invDepth = render_pkg["depth"]
            mono_invdepth = viewpoint_cam.invdepthmap.cuda()
            depth_mask = viewpoint_cam.depth_mask.cuda()

            Ll1depth_pure = torch.abs((invDepth  - mono_invdepth) * depth_mask).mean()
            Ll1depth = depth_l1_weight(iteration) * Ll1depth_pure 
            loss += Ll1depth
            Ll1depth = Ll1depth.item()
        else:
            Ll1depth = 0

        # 阶段2新功能：法向一致性 loss（渲染法向 vs 深度导出法向，不依赖外部 GT）
        # 仅在 --lambda_normal > 0 时计算；mask 区域外（alpha_mask == 0）像素不参与
        loss_normal_v = 0.0
        if dataset.lambda_normal > 0.0:
            n_rendered = render_pkg["normal"][:, 1:-1, 1:-1]              # 3, H-2, W-2
            n_depth = normal_from_invdepth(render_pkg["depth"], viewpoint_cam)  # 3, H-2, W-2
            cos = torch.sum(n_rendered * n_depth, dim=0) / (
                torch.norm(n_rendered, dim=0) * torch.norm(n_depth, dim=0) + 1e-6)
            valid_n = render_pkg["depth"][0, 1:-1, 1:-1] > 1e-4
            if viewpoint_cam.alpha_mask is not None:
                valid_n = valid_n & (alpha_mask[0, 1:-1, 1:-1] > 0.5)
            if int(valid_n.sum()) > 0:
                loss_normal = (1.0 - torch.abs(cos[valid_n])).mean()
                loss += dataset.lambda_normal * loss_normal
                loss_normal_v = loss_normal.item()

        # 阶段2新功能：scale 正则——惩罚 z 轴厚度 |exp(scaling)[:,2]| 均值（参考 2DGS），
        # 仅在 --lambda_scale > 0 时计算；梯度直接经 get_scaling 回传到 _scaling
        loss_scale_v = 0.0
        if dataset.lambda_scale > 0.0:
            loss_scale = torch.abs(gaussians.get_scaling[:, 2:3]).mean()
            loss += dataset.lambda_scale * loss_scale
            loss_scale_v = loss_scale.item()

        # 阶段2新功能：形状约束（防大椭球）——逐高斯取最长尺度轴（exp 后真实尺度）的均值，
        # 与 lambda_scale 互补：后者只压法向轴厚度，此项限制切向铺展；
        # 仅在 --lambda_size > 0 时计算（默认 0，不显式开启时既有实验口径不变）
        loss_size_v = 0.0
        if dataset.lambda_size > 0.0:
            loss_size = gaussians.get_scaling.max(dim=1, keepdim=True)[0].mean()
            loss += dataset.lambda_size * loss_size
            loss_size_v = loss_size.item()

        # D2：每 10 轮追加记录 {iteration, L1, SSIM loss, depth loss, normal loss, scale reg, size reg, total}
        if iteration % 10 == 0:
            with open(loss_log_path, "a") as f:
                f.write("{},{:.6f},{:.6f},{:.6f},{:.6f},{:.6f},{:.6f},{:.6f}\n".format(
                    iteration, Ll1.item(), 1.0 - ssim_value.item(), float(Ll1depth),
                    float(loss_normal_v), float(loss_scale_v), float(loss_size_v), loss.item()))
            # 阶段2新功能：每 10 轮 print 新增的 Normal Loss / Scale Reg / Size Reg 项（仅在开启时打印）
            if dataset.lambda_normal > 0.0 or dataset.lambda_scale > 0.0 or dataset.lambda_size > 0.0:
                print("[loss] iter {}: Normal Loss {:.6f} / Scale Reg {:.6f} / Size Reg {:.6f} / total {:.6f}".format(
                    iteration, loss_normal_v, loss_scale_v, loss_size_v, loss.item()))

        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_Ll1depth_for_log = 0.4 * Ll1depth + 0.6 * ema_Ll1depth_for_log

            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}", "Depth Loss": f"{ema_Ll1depth_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background, 1., SPARSE_ADAM_AVAILABLE, None, dataset.train_test_exp), dataset.train_test_exp)
            if iteration in testing_iterations:
                plot_loss_curves(loss_log_path, scene.model_path, iteration)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # 阶段2新功能：val10 渲染升级为每 100 轮（含第 1 轮），输出 RGB + 光栅化器法向图
            if val_cameras and iteration % 100 == 1:
                render_val_cameras(val_cameras, gaussians, pipe, background, iteration, scene.model_path,
                                   SPARSE_ADAM_AVAILABLE, dataset.train_test_exp, elapsed_sec=time.time() - train_start_time)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold, radii)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.exposure_optimizer.step()
                gaussians.exposure_optimizer.zero_grad(set_to_none = True)
                if use_sparse_adam:
                    visible = radii > 0
                    gaussians.optimizer.step(visible, radii.shape[0])
                    gaussians.optimizer.zero_grad(set_to_none = True)
                else:
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none = True)

            # 新功能 1：2D 椭球形状约束（--freeze_2d_z）——每步优化后强制复位 z 轴缩放（z 通道不参与优化）；
            # 与初始化开关解耦：C 组（仅 --init_2d）不执行，允许椭球厚度自由优化
            if gaussians.freeze_2d_z:
                gaussians.freeze_z_scaling()

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

def normal_from_invdepth(invdepth, viewpoint_camera):
    """阶段2新功能：由渲染的逆深度图导出法向量图（参考 2DGS，不依赖外部 GT）。
    逆深度 → 深度 → 相机系反投影 3D 点 → 中心差分叉积得面法向（内部像素 H-2 x W-2），
    与光栅化器输出的法向一样朝向相机（叉积方向取 dPdy x dPdx），归一化后返回 3 x (H-2) x (W-2)"""
    W = int(viewpoint_camera.image_width)
    H = int(viewpoint_camera.image_height)
    fx = W / (2.0 * math.tan(viewpoint_camera.FoVx * 0.5))
    fy = H / (2.0 * math.tan(viewpoint_camera.FoVy * 0.5))
    cx, cy = (W - 1) / 2.0, (H - 1) / 2.0
    depth = 1.0 / invdepth[0].clamp(min=1e-6)  # H, W
    dev = depth.device
    y, x = torch.meshgrid(torch.arange(H, device=dev, dtype=torch.float32),
                          torch.arange(W, device=dev, dtype=torch.float32), indexing="ij")
    X = (x - cx) / fx * depth
    Y = (y - cy) / fy * depth
    P = torch.stack([X, Y, depth], dim=-1)          # H, W, 3
    dPdx = (P[:, 2:, :] - P[:, :-2, :]) * 0.5       # H, W-2, 3
    dPdy = (P[2:, :, :] - P[:-2, :, :]) * 0.5       # H-2, W, 3
    dPdx = dPdx[1:-1, :, :]                          # H-2, W-2, 3
    dPdy = dPdy[:, 1:-1, :]                          # H-2, W-2, 3
    n = torch.cross(dPdy, dPdx, dim=-1)
    n = torch.nn.functional.normalize(n, dim=-1)
    return n.permute(2, 0, 1).contiguous()           # 3, H-2, W-2

def plot_loss_curves(loss_log_path, model_path, iteration):
    """D2：根据 loss_log.csv 输出损失曲线图（总曲线 + 前 20000 轮放大）；阶段2新增 normal/scale 列"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[提示] 未安装 matplotlib，无法输出损失曲线图，请执行：pip install matplotlib")
        return
    data = np.loadtxt(loss_log_path, delimiter=",", skiprows=1)
    if data.ndim == 1:
        data = data[None]
    it = data[:, 0]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    # 兼容旧版 5 列、阶段2 7 列与含 size_reg 的 8 列 csv
    if data.shape[1] >= 8:
        cols = [(1, "L1"), (2, "SSIM loss"), (3, "depth loss"), (4, "normal loss"),
                (5, "scale reg"), (6, "size reg"), (7, "total")]
    elif data.shape[1] >= 7:
        cols = [(1, "L1"), (2, "SSIM loss"), (3, "depth loss"), (4, "normal loss"), (5, "scale reg"), (6, "total")]
    else:
        cols = [(1, "L1"), (2, "SSIM loss"), (3, "depth loss"), (4, "total")]
    for col, name in cols:
        axes[0].plot(it, data[:, col], label=name, linewidth=0.8)
        mask = it <= 20000
        axes[1].plot(it[mask], data[mask, col], label=name, linewidth=0.8)
    axes[0].set_title("Loss curves (all iterations)")
    axes[0].set_xlabel("iteration"); axes[0].set_ylabel("loss"); axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].set_title("Loss curves (first 20000 iterations)")
    axes[1].set_xlabel("iteration"); axes[1].set_ylabel("loss"); axes[1].legend(); axes[1].grid(alpha=0.3)
    out_path = os.path.join(model_path, "loss_curves_{}.png".format(iteration))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print("\n[ITER {}] 损失曲线图已保存：{}".format(iteration, out_path))

def render_val_cameras(val_cameras, gaussians, pipe, background, iteration, model_path, separate_sh, train_test_exp, elapsed_sec=0.0):
    """新功能：渲染全部 val 相机，输出当前轮次的 RGB 渲染图与法向量方向颜色贴图（阶段2升级为
    直接使用光栅化器 normal 通道，与训练 loss 同口径）；GT 图只在第一轮保存；
    同时计算 val 集 L1/PSNR/SSIM 并追加到 val_metrics.csv"""
    out_dir = os.path.join(model_path, "val_render", "iter_{}".format(iteration))
    os.makedirs(out_dir, exist_ok=True)
    # GT 只在第一轮保存：以 iter_1 目录下是否已有 GT 图为准（断点续训且 iter_1 缺失时也会补存）
    save_gt = not os.path.exists(os.path.join(model_path, "val_render", "iter_1", "val0_gt.png"))

    l1_sum = 0.0
    psnr_sum = 0.0
    ssim_sum = 0.0
    for idx, cam in enumerate(val_cameras):
        render_pkg = render(cam, gaussians, pipe, background, 1.0, separate_sh, None, train_test_exp)
        img = torch.clamp(render_pkg["render"], 0.0, 1.0)

        # 指标计算：与 evaluate.py 一致，渲染图先乘 alpha_mask（original_image 加载时已乘过 mask）
        gt = torch.clamp(cam.original_image.cuda(), 0.0, 1.0)
        img_eval = img * cam.alpha_mask.cuda() if cam.alpha_mask is not None else img
        l1_sum += l1_loss(img_eval, gt).mean().item()
        psnr_sum += psnr(img_eval, gt).mean().item()
        ssim_sum += ssim(img_eval, gt).item()

        arr = (img * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy()
        Image.fromarray(arr).save(os.path.join(out_dir, "val{}_render.png".format(idx)))
        if save_gt:
            gt_arr = (gt * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy()
            Image.fromarray(gt_arr).save(os.path.join(out_dir, "val{}_gt.png".format(idx)))

        # 阶段2新功能：直接用光栅化器 normal 通道（已朝向相机）生成方向颜色贴图 RGB=(n+1)/2，
        # 无需额外渲染一次，与 normal loss 输入同口径，可与阶段0点云法向投影图对照
        nmap = torch.clamp(render_pkg["normal"], -1.0, 1.0)
        nimg = (nmap + 1.0) * 0.5
        narr = (nimg * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy()
        Image.fromarray(narr).save(os.path.join(out_dir, "val{}_normal.png".format(idx)))

    # val 指标记录到 val_metrics.csv，便于追踪训练过程中的变化；
    # 新增：当前高斯点云数量与训练耗时（秒），方便观察致密化与速度
    n = len(val_cameras)
    n_gauss = int(gaussians.get_xyz.shape[0])
    metrics_path = os.path.join(model_path, "val_metrics.csv")
    if not os.path.exists(metrics_path):
        with open(metrics_path, "w") as f:
            f.write("iteration,L1,PSNR,SSIM,num_gaussians,elapsed_sec\n")
    with open(metrics_path, "a") as f:
        f.write("{},{:.6f},{:.4f},{:.4f},{},{:.1f}\n".format(
            iteration, l1_sum / n, psnr_sum / n, ssim_sum / n, n_gauss, elapsed_sec))

    print("\n[val_metrics] iter {}: L1 {:.6f} / PSNR {:.3f} / SSIM {:.4f} / 高斯数 {} / 耗时 {:.0f}s（已记录到 {}）".format(
        iteration, l1_sum / n, psnr_sum / n, ssim_sum / n, n_gauss, elapsed_sec, metrics_path))
    print("[val_render] iter {} 已保存渲染图+法向图{}到 {}".format(
        iteration, "（含 GT，仅首轮保存）" if save_gt else "", out_dir))
    torch.cuda.empty_cache()

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, train_test_exp):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if train_test_exp:
                        image = image[..., image.shape[-1] // 2:]
                        gt_image = gt_image[..., gt_image.shape[-1] // 2:]
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[10_000,30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[10_000,30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--disable_viewer', action='store_true', default=False)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    # 新功能参数
    parser.add_argument("--seed", type=int, default=42, help="随机种子，保证切分/训练可复现")
    parser.add_argument("--val_file", type=str, default="", help="val 相机外参文件（如 images-val10.txt），默认空=关闭 val 渲染")
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)
    # 显式设定随机种子，保证可复现
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Start GUI server, configure and run training
    if not args.disable_viewer:
        network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args.val_file)

    # All done
    print("\nTraining complete.")
