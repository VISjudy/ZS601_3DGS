#
# 阶段 1-B 评价指标工具
#
# 功能：
#   1) 图像指标：test 集（images_test.txt）上的 PSNR / L1 / SSIM（与 metrics.py 同口径）
#   2) 几何指标（cloud2cloud）：训练后高斯中心点与激光点云 points3D.txt XYZ 的最近邻距离统计
#   3) 输出到 <model_path>/metrics.json 与终端表格
#
# 用法示例：
#   python evaluate.py -m output/zs601_init2d -s D:\LCC\ZS601_meetingroom_dev\dataset
#

import os
import sys
import json
import numpy as np
import torch
from argparse import ArgumentParser
from tqdm import tqdm
from scipy.spatial import cKDTree

from scene import Scene, GaussianModel
from gaussian_renderer import render
from utils.loss_utils import l1_loss, ssim
from utils.image_utils import psnr
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, read_points3D_text
from scene.dataset_readers import readColmapCameras
from utils.camera_utils import cameraList_from_camInfos
from arguments import ModelParams, PipelineParams, get_combined_args


def load_test_cameras(args, test_file):
    """从 test 外参文件构建相机列表（与训练时 val 相机加载方式一致）"""
    test_extr = read_extrinsics_text(test_file)
    test_intr = read_intrinsics_text(os.path.join(args.source_path, "sparse/0/cameras.txt"))
    # 修复：test 相机同样加载 mask，与训练目标一致（mask 区域渲染为黑），
    # 否则指标被 mask 区域的原图内容系统性拉低
    masks_dir = os.path.join(args.source_path, args.alpha_masks) if args.alpha_masks else ""
    test_cam_infos = readColmapCameras(cam_extrinsics=test_extr, cam_intrinsics=test_intr, depths_params=None,
                                       images_folder=os.path.join(args.source_path, "images"),
                                       masks_folder=masks_dir, depths_folder="", test_cam_names_list=[])
    return cameraList_from_camInfos(test_cam_infos, 1.0, args, False, True)


def evaluate_images(test_cameras, gaussians, pipe, background):
    """渲染 test 集并计算 PSNR / L1 / SSIM"""
    psnrs, l1s, ssims = [], [], []
    per_view = {}
    for cam in tqdm(test_cameras, desc="渲染并计算图像指标"):
        with torch.no_grad():
            image = torch.clamp(render(cam, gaussians, pipe, background)["render"], 0.0, 1.0)
        gt_image = torch.clamp(cam.original_image.to("cuda"), 0.0, 1.0)
        if cam.alpha_mask is not None:
            # 与训练一致：mask 区域外不参与比较（gt 已在加载时乘过 mask）；
            # alpha_mask 可能位于 data_device（如 --data_device cpu），需移到 GPU
            image = image * cam.alpha_mask.to("cuda")
        p = psnr(image, gt_image).mean().item()
        l = l1_loss(image, gt_image).mean().item()
        s = ssim(image, gt_image).item()
        psnrs.append(p); l1s.append(l); ssims.append(s)
        per_view[cam.image_name] = {"PSNR": p, "L1": l, "SSIM": s}
    return {
        "PSNR": float(np.mean(psnrs)),
        "L1": float(np.mean(l1s)),
        "SSIM": float(np.mean(ssims)),
        "num_views": len(test_cameras),
    }, per_view


def evaluate_geometry(gaussians, points3d_path, chunk=200000):
    """cloud2cloud：高斯中心点 -> 激光点云最近邻距离统计（cKDTree，分块向量化）"""
    print("读取激光点云（几何参考）：{}".format(points3d_path))
    xyz_ref, _, _ = read_points3D_text(points3d_path)
    print("参考点云点数：{}".format(xyz_ref.shape[0]))

    centers = gaussians.get_xyz.detach().cpu().numpy()
    print("高斯中心点数：{}".format(centers.shape[0]))
    # 防御：过滤非有限坐标（NaN/Inf，可能来自历史模型），否则 cKDTree.query 报错
    finite_mask = np.isfinite(centers).all(axis=1)
    n_bad = int((~finite_mask).sum())
    if n_bad > 0:
        print("警告：发现 {} 个非有限坐标（NaN/Inf）高斯，已从几何统计中剔除".format(n_bad))
        centers = centers[finite_mask]

    tree = cKDTree(xyz_ref)
    dists = np.empty(centers.shape[0], dtype=np.float64)
    for s in range(0, centers.shape[0], chunk):
        e = min(s + chunk, centers.shape[0])
        d, _ = tree.query(centers[s:e], workers=-1)
        dists[s:e] = d

    stats = {
        "direction": "gaussian_center -> laser_point(nearest)",
        "num_gaussians": int(centers.shape[0]),
        "num_nonfinite_skipped": n_bad,
        "num_ref_points": int(xyz_ref.shape[0]),
        "mean": float(dists.mean()),
        "median": float(np.median(dists)),
        "rmse": float(np.sqrt((dists ** 2).mean())),
        "p90": float(np.percentile(dists, 90)),
    }
    return stats


if __name__ == "__main__":
    parser = ArgumentParser(description="Evaluation script parameters")
    lp = ModelParams(parser, sentinel=True)
    pp = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--test_file", default="", type=str, help="test 外参文件，默认 <source>/sparse/0/images_test.txt")
    parser.add_argument("--points3d", default="", type=str, help="几何参考点云，默认 <source>/sparse/0/points3D.txt")
    parser.add_argument("--skip_images", action="store_true", help="跳过图像指标（只算几何）")
    parser.add_argument("--skip_geometry", action="store_true", help="跳过几何指标（只算图像）")
    args = get_combined_args(parser)
    print("Evaluating " + args.model_path)

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    gaussians = GaussianModel(args.sh_degree)
    scene = Scene(args, gaussians, load_iteration=args.iteration, shuffle=False)

    background = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")

    metrics = {
        "model_path": args.model_path,
        "iteration": scene.loaded_iter,
        "source_path": args.source_path,
    }

    # 1) 图像指标
    if not args.skip_images:
        test_file = args.test_file if args.test_file else os.path.join(args.source_path, "sparse/0/images_test.txt")
        assert os.path.exists(test_file), "找不到 test 外参文件：{}（请先运行 preprocess.py 生成）".format(test_file)
        test_cameras = load_test_cameras(args, test_file)
        print("test 相机数：{}（{}）".format(len(test_cameras), test_file))
        image_metrics, per_view = evaluate_images(test_cameras, scene.gaussians, pp.extract(args), background)
        metrics["image_metrics"] = image_metrics
        metrics["per_view"] = per_view
        torch.cuda.empty_cache()

    # 2) 几何指标
    if not args.skip_geometry:
        points3d_path = args.points3d if args.points3d else os.path.join(args.source_path, "sparse/0/points3D.txt")
        assert os.path.exists(points3d_path), "找不到参考点云：{}".format(points3d_path)
        metrics["geometry"] = evaluate_geometry(scene.gaussians, points3d_path)

    # 3) 输出
    with open(os.path.join(args.model_path, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print("评价结果（iteration={}）".format(scene.loaded_iter))
    print("=" * 60)
    if "image_metrics" in metrics:
        m = metrics["image_metrics"]
        print("图像指标（{} 张 test 图）：".format(m["num_views"]))
        print("  PSNR : {:.4f}".format(m["PSNR"]))
        print("  L1   : {:.6f}".format(m["L1"]))
        print("  SSIM : {:.4f}".format(m["SSIM"]))
    if "geometry" in metrics:
        g = metrics["geometry"]
        print("几何指标（{}）：".format(g["direction"]))
        print("  高斯数：{}  参考点数：{}".format(g["num_gaussians"], g["num_ref_points"]))
        print("  mean   : {:.5f} m".format(g["mean"]))
        print("  median : {:.5f} m".format(g["median"]))
        print("  RMSE   : {:.5f} m".format(g["rmse"]))
        print("  p90    : {:.5f} m".format(g["p90"]))
    print("结果已写入：{}".format(os.path.join(args.model_path, "metrics.json")))
