#
# 阶段 0：数据预处理独立脚本（一次性前置工作）
#
# 功能：
#   0-1 激光点云转换：读取 .las，提取 XYZ+RGB，转 COLMAP points3D.txt 格式（替换 sparse/0/points3D.txt，
#       原文件备份为 points3D.txt.bak），并生成带颜色的 sparse/0/points3D.ply
#   0-2 法向量计算：KNN k=16 邻域 PCA 求逐点法向量，统一朝向后写入 points3D.ply 的 nx,ny,nz
#   0-3 训练/测试切分：解析 sparse/0/images.txt，5% 随机测试集；训练集回写 images.txt，
#       测试集存 images_test.txt，测试集中再随机选 val_num 张存 images-val10.txt
#   0-4 法向量方向图验证渲染：用 val 相机把点云法向量投影渲染成方向图，配同视角真实图像副本
#
# 用法示例：
#   python preprocess.py --data_path D:\LCC\ZS601_meetingroom_dev\dataset --seed 42 --val_num 10 --test_ratio 0.05
#

import os
import sys
import time
import json
import shutil
import argparse
import numpy as np
from PIL import Image
from plyfile import PlyData, PlyElement
from scipy.spatial import cKDTree


# ----------------------------------------------------------------------------
# COLMAP 文本格式读取（内联实现，与 scene/colmap_loader.py 一致；
# 内联是为了让预处理脚本不依赖 torch，可在任意环境独立运行）
# ----------------------------------------------------------------------------
def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])


def read_intrinsics_text(path):
    cameras = {}
    with open(path, "r") as fid:
        for line in fid:
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                camera_id = int(elems[0])
                model = elems[1]
                width = int(elems[2])
                height = int(elems[3])
                params = np.array(tuple(map(float, elems[4:])))
                cameras[camera_id] = {"id": camera_id, "model": model,
                                      "width": width, "height": height, "params": params}
    return cameras


def read_extrinsics_text(path):
    """解析 images.txt，仅保留外参信息（qvec/tvec/camera_id/name），跳过 2D 点行"""
    images = {}
    with open(path, "r") as fid:
        lines = fid.read().splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if len(line) == 0 or line[0] == "#":
            i += 1
            continue
        elems = line.split()
        image_id = int(elems[0])
        qvec = np.array(tuple(map(float, elems[1:5])))
        tvec = np.array(tuple(map(float, elems[5:8])))
        camera_id = int(elems[8])
        image_name = elems[9]
        images[image_id] = {"id": image_id, "qvec": qvec, "tvec": tvec,
                            "camera_id": camera_id, "name": image_name}
        i += 2  # 跳过下一行 POINTS2D
    return images


def print_header(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# ----------------------------------------------------------------------------
# 0-1 激光点云转换
# ----------------------------------------------------------------------------
def convert_las_to_colmap(las_path, sparse_dir, chunk_size):
    print_header("0-1 激光点云转换（las -> COLMAP points3D.txt / points3D.ply）")
    try:
        import laspy
    except ImportError:
        print("缺少 laspy，请先执行：pip install laspy")
        sys.exit(1)

    t0 = time.time()
    las = laspy.read(las_path)
    xyz = np.asarray(las.xyz, dtype=np.float64)  # (N,3)，laspy 已按 scale/offset 还原真实坐标
    n_pts = xyz.shape[0]
    print("读取 las 完成，点数：{}，耗时 {:.2f}s".format(n_pts, time.time() - t0))

    # RGB：部分 las 为 16bit，需要压缩到 8bit
    rgb = np.stack([np.asarray(las.red), np.asarray(las.green), np.asarray(las.blue)], axis=1)
    if rgb.max() > 255:
        print("RGB 为 16bit（max={}），右移 8bit 压缩到 8bit".format(rgb.max()))
        rgb = (rgb >> 8)
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    rgb_valid = (rgb.sum(axis=1) > 0).mean()
    print("RGB 有效性：非全黑点比例 {:.2%}（RGB max={} min={}）".format(rgb_valid, rgb.max(), rgb.min()))

    print("las 坐标范围（即 COLMAP 世界坐标，未做变换）：")
    for i, name in enumerate("XYZ"):
        print("  {}: [{:.3f}, {:.3f}]".format(name, xyz[:, i].min(), xyz[:, i].max()))

    # ---- 备份并替换 points3D.txt ----
    txt_path = os.path.join(sparse_dir, "points3D.txt")
    bak_path = txt_path + ".bak"
    if os.path.exists(txt_path) and not os.path.exists(bak_path):
        shutil.copy2(txt_path, bak_path)
        print("已备份原文件 -> {}".format(bak_path))

    # 分块写（向量化，禁止逐点 Python 循环）：POINT3D_ID X Y Z R G B ERROR TRACK[]
    ids = np.arange(1, n_pts + 1, dtype=np.int64)
    err = np.full((n_pts, 1), -1, dtype=np.int64)
    data = np.concatenate([ids[:, None].astype(np.float64), xyz,
                           rgb.astype(np.float64), err.astype(np.float64)], axis=1)
    fmt = ["%d", "%.10f", "%.10f", "%.10f", "%d", "%d", "%d", "%d"]
    t0 = time.time()
    with open(txt_path, "w") as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        f.write("# Number of points: {} (generated by preprocess.py from {})\n".format(n_pts, os.path.basename(las_path)))
        for s in range(0, n_pts, chunk_size):
            import io
            buf = io.StringIO()
            np.savetxt(buf, data[s:s + chunk_size], fmt=fmt)
            f.write(buf.getvalue())
    print("写出 points3D.txt 完成：{} 点，耗时 {:.2f}s".format(n_pts, time.time() - t0))

    # ---- 生成 points3D.ply（若已存在先删除，否则 dataset_readers.py 不会重建）----
    ply_path = os.path.join(sparse_dir, "points3D.ply")
    if os.path.exists(ply_path):
        os.remove(ply_path)
        print("已删除旧 ply：{}".format(ply_path))
    normals_zero = np.zeros_like(xyz, dtype=np.float32)  # 法向量由 0-2 写入
    write_ply(ply_path, xyz, normals_zero, rgb)
    print("写出 points3D.ply（颜色，法向量暂为 0，由 0-2 填充）：{}".format(ply_path))

    return xyz, rgb


def write_ply(path, xyz, normals, rgb):
    """写出带 nx,ny,nz 与 RGB 的 ply（与 dataset_readers.storePly 的字段结构一致）"""
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
             ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
             ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    elements = np.empty(xyz.shape[0], dtype=dtype)
    elements['x'], elements['y'], elements['z'] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    elements['nx'], elements['ny'], elements['nz'] = normals[:, 0], normals[:, 1], normals[:, 2]
    elements['red'], elements['green'], elements['blue'] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    vertex_element = PlyElement.describe(elements, 'vertex')
    PlyData([vertex_element]).write(path)


# ----------------------------------------------------------------------------
# 0-2 法向量计算（KNN k=16 邻域 PCA）
# ----------------------------------------------------------------------------
def compute_normals(xyz, ply_path, rgb, knn_k, chunk_size, workers):
    print_header("0-2 法向量计算（KNN k={} 邻域 PCA，分块向量化）".format(knn_k))
    n_pts = xyz.shape[0]
    t0 = time.time()
    tree = cKDTree(xyz)
    print("cKDTree 建索引耗时 {:.2f}s".format(time.time() - t0))

    centroid = xyz.mean(axis=0)
    normals = np.empty((n_pts, 3), dtype=np.float32)
    planarity = np.empty(n_pts, dtype=np.float32)  # 中/最小特征值比，越大越平面
    t0 = time.time()
    for s in range(0, n_pts, chunk_size):
        e = min(s + chunk_size, n_pts)
        _, idx = tree.query(xyz[s:e], k=knn_k + 1, workers=workers)  # 含自身
        nbrs = xyz[idx[:, 1:]]                                       # (C,k,3)，去掉自身
        centered = nbrs - xyz[s:e, None, :]
        cov = np.einsum('cki,ckj->cij', centered, centered) / knn_k  # (C,3,3)
        eigvals, eigvecs = np.linalg.eigh(cov)                       # 升序
        normals[s:e] = eigvecs[:, :, 0]                              # 最小特征值对应特征向量
        planarity[s:e] = eigvals[:, 1] / np.maximum(eigvals[:, 0], 1e-12)
        if (s // chunk_size) % 5 == 0:
            print("  法向量计算进度 {}/{} ...".format(e, n_pts))
    print("法向量 PCA 计算耗时 {:.2f}s".format(time.time() - t0))

    # 统一朝向：法向量应与“质心->该点”方向同向（点积为负则翻转），使法向大致朝外
    out_dir = xyz - centroid
    out_dir /= np.maximum(np.linalg.norm(out_dir, axis=1, keepdims=True), 1e-12)
    flip_mask = np.sum(normals * out_dir, axis=1) < 0
    normals[flip_mask] *= -1
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    print("朝向统一：翻转比例 {:.2%}".format(flip_mask.mean()))

    print("法向量统计：nz 均值 {:.3f}；平面性(λ1/λ0)中位数 {:.1f}，>10 的点占 {:.2%}".format(
        normals[:, 2].mean(), np.median(planarity), (planarity > 10).mean()))

    # 重写 ply，写入 nx,ny,nz
    write_ply(ply_path, xyz, normals, rgb)
    print("已将法向量写入 {}".format(ply_path))
    return normals


# ----------------------------------------------------------------------------
# 0-3 训练/测试切分（按完整两行记录切分）
# ----------------------------------------------------------------------------
def split_images(sparse_dir, intermediate_dir, seed, test_ratio, val_num):
    print_header("0-3 训练/测试切分（seed={}, test_ratio={}, val_num={}）".format(seed, test_ratio, val_num))
    images_path = os.path.join(sparse_dir, "images.txt")
    bak_path = images_path + ".bak"

    # 若已有备份，则以备份（原始完整文件）为切分来源，保证重复运行结果一致
    source_path = bak_path if os.path.exists(bak_path) else images_path
    if source_path == images_path:
        shutil.copy2(images_path, bak_path)
        print("已备份原 images.txt -> {}".format(bak_path))
    else:
        print("检测到已有备份，从原始文件切分：{}".format(bak_path))

    # 解析完整记录（每图两行：外参行 + 2D 点行），保留原始文本以便原样写回
    records = []  # (line1, line2)
    with open(source_path, "r") as f:
        lines = f.read().splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line == "" or line.startswith("#"):
            i += 1
            continue
        line2 = lines[i + 1]  # 2D 点行（可能为空行）
        records.append((lines[i], line2))
        i += 2
    n_total = len(records)
    print("总图像数：{}".format(n_total))

    n_test = max(1, int(round(n_total * test_ratio)))
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n_total)
    test_indices = set(perm[:n_test].tolist())
    test_records = [records[i] for i in sorted(test_indices)]
    train_records = [records[i] for i in range(n_total) if i not in test_indices]

    # 测试集中再随机选 val_num 张（在 test_records 内部下标中抽）
    val_pick = rng.choice(len(test_records), size=min(val_num, len(test_records)), replace=False)
    val_records = [test_records[i] for i in sorted(val_pick.tolist())]

    def write_colmap_images(path, recs, header_note):
        with open(path, "w") as f:
            f.write("# Image list with two lines of data per image:\n")
            f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
            f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
            f.write("# Number of images: {} ({})\n".format(len(recs), header_note))
            for l1, l2 in recs:
                f.write(l1 + "\n")
                f.write(l2 + "\n")

    write_colmap_images(images_path, train_records, "train, written by preprocess.py")
    write_colmap_images(os.path.join(sparse_dir, "images_test.txt"), test_records, "test split")
    val_path = os.path.join(sparse_dir, "images-val10.txt")
    write_colmap_images(val_path, val_records, "val subset of test")

    val_names = [r[0].split()[9] for r in val_records]
    print("切分完成：总数 {} / 训练 {} / 测试 {} / val {}".format(n_total, len(train_records), len(test_records), len(val_records)))
    print("val 图像名列表：")
    for name in val_names:
        print("  " + name)
    with open(os.path.join(intermediate_dir, "val_image_names.txt"), "w") as f:
        f.write("\n".join(val_names) + "\n")
    return val_names


# ----------------------------------------------------------------------------
# 0-4 法向量方向图验证渲染（val 相机，numpy 向量化投影 + z-buffer）
# ----------------------------------------------------------------------------
def render_normal_maps(data_path, sparse_dir, intermediate_dir, xyz, normals):
    print_header("0-4 法向量方向图验证渲染（val 相机投影点云法向量）")
    val_path = os.path.join(sparse_dir, "images-val10.txt")
    intrinsics = read_intrinsics_text(os.path.join(sparse_dir, "cameras.txt"))
    extrinsics = read_extrinsics_text(val_path)
    out_dir = os.path.join(intermediate_dir, "normal_visual")
    os.makedirs(out_dir, exist_ok=True)

    valid_ratios = {}
    for idx, key in enumerate(sorted(extrinsics)):
        extr = extrinsics[key]
        intr = intrinsics[extr["camera_id"]]
        assert intr["model"] == "PINHOLE", "仅支持 PINHOLE 相机模型"
        fx, fy, cx, cy = intr["params"][0], intr["params"][1], intr["params"][2], intr["params"][3]
        W, H = intr["width"], intr["height"]

        R = qvec2rotmat(extr["qvec"])
        t = np.asarray(extr["tvec"])
        # 世界坐标 -> 相机坐标（numpy 向量化，百万点亦可分块）
        x_cam = xyz @ R.T + t
        z = x_cam[:, 2]
        valid = z > 1e-3
        u = np.full(z.shape[0], -1.0)
        v = np.full(z.shape[0], -1.0)
        u[valid] = fx * x_cam[valid, 0] / z[valid] + cx
        v[valid] = fy * x_cam[valid, 1] / z[valid] + cy
        ui = np.floor(u).astype(np.int64)
        vi = np.floor(v).astype(np.int64)
        valid &= (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)

        # z-buffer：按深度从远到近排序后依次写入，最近点最后覆盖
        order = np.argsort(-z[valid])            # 远的在前
        flat_idx = (vi[valid] * W + ui[valid])[order]
        point_idx = np.nonzero(valid)[0][order]
        idx_buf = np.full(H * W, -1, dtype=np.int64)
        idx_buf[flat_idx] = point_idx            # 后写（更近）覆盖先写

        img = np.zeros((H * W, 3), dtype=np.uint8)
        pix_valid = idx_buf >= 0
        color = (normals[idx_buf[pix_valid]] + 1.0) / 2.0   # (n+1)/2
        img[pix_valid] = np.clip(color * 255.0, 0, 255).astype(np.uint8)
        img = img.reshape(H, W, 3)

        ratio = pix_valid.mean()
        valid_ratios[extr["name"]] = float(ratio)
        Image.fromarray(img).save(os.path.join(out_dir, "val{}_normal.png".format(idx)))
        # 同视角真实图像副本
        gt_src = os.path.join(data_path, "images", extr["name"])
        if os.path.exists(gt_src):
            Image.open(gt_src).save(os.path.join(out_dir, "val{}_gt.png".format(idx)))
        else:
            print("  [警告] 找不到真实图像：{}".format(gt_src))
        print("  val{} {} 有效像素比例：{:.2%}".format(idx, extr["name"], ratio))

    return valid_ratios


# ----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="阶段 0 数据预处理脚本")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_num", type=int, default=10)
    parser.add_argument("--test_ratio", type=float, default=0.05)
    parser.add_argument("--intermediate", type=str, default="")
    parser.add_argument("--las_name", type=str, default="ZS601_3cm_sample.las")
    parser.add_argument("--knn_k", type=int, default=16)
    parser.add_argument("--chunk", type=int, default=200000, help="法向量/写文件分块大小")
    parser.add_argument("--workers", type=int, default=-1, help="cKDTree 查询并行线程数，-1 为全部")
    args = parser.parse_args()

    data_path = args.data_path
    sparse_dir = os.path.join(data_path, "sparse", "0")
    intermediate_dir = args.intermediate if args.intermediate else os.path.join(data_path, "intermediate")
    os.makedirs(intermediate_dir, exist_ok=True)
    las_path = os.path.join(data_path, args.las_name)
    assert os.path.exists(las_path), "找不到 las 文件：{}".format(las_path)

    t_total = time.time()
    # 0-1
    xyz, rgb = convert_las_to_colmap(las_path, sparse_dir, args.chunk)
    ply_path = os.path.join(sparse_dir, "points3D.ply")
    # 0-2
    normals = compute_normals(xyz, ply_path, rgb, args.knn_k, args.chunk, args.workers)
    # 0-3
    val_names = split_images(sparse_dir, intermediate_dir, args.seed, args.test_ratio, args.val_num)
    # 相机质心与点云质心距离对照（用切分后的训练集 images.txt 外参）
    extrinsics = read_extrinsics_text(os.path.join(sparse_dir, "images.txt"))
    cam_centers = np.stack([
        -qvec2rotmat(extrinsics[k]["qvec"]).T @ np.asarray(extrinsics[k]["tvec"]) for k in extrinsics
    ])
    pcd_centroid = xyz.mean(axis=0)
    cam_centroid = cam_centers.mean(axis=0)
    dist_center = np.linalg.norm(pcd_centroid - cam_centroid)
    print("\n点云质心：{}, 相机质心：{}, 二者距离：{:.3f} m".format(
        np.round(pcd_centroid, 3).tolist(), np.round(cam_centroid, 3).tolist(), dist_center))
    if dist_center > 5.0:
        print("[警告] 点云质心与相机质心距离过大，请检查单位/轴向是否一致！")
    # 0-4
    valid_ratios = render_normal_maps(data_path, sparse_dir, intermediate_dir, xyz, normals)

    stats = {
        "num_points": int(xyz.shape[0]),
        "xyz_min": xyz.min(axis=0).tolist(),
        "xyz_max": xyz.max(axis=0).tolist(),
        "pcd_centroid": pcd_centroid.tolist(),
        "cam_centroid": cam_centroid.tolist(),
        "pcd_cam_centroid_dist": float(dist_center),
        "num_train": int(len(extrinsics)),
        "val_names": val_names,
        "val_valid_pixel_ratio": valid_ratios,
    }
    with open(os.path.join(intermediate_dir, "preprocess_stats.json"), "w") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print_header("阶段 0 预处理全部完成，总耗时 {:.1f}s".format(time.time() - t_total))
    print("产物清单：")
    print("  " + os.path.join(sparse_dir, "points3D.txt（含 .bak 备份）"))
    print("  " + ply_path + "（颜色 + 法向量）")
    print("  " + os.path.join(sparse_dir, "images.txt / images_test.txt / images-val10.txt"))
    print("  " + os.path.join(intermediate_dir, "normal_visual", "val*_normal.png / val*_gt.png"))
    print("  " + os.path.join(intermediate_dir, "preprocess_stats.json"))


if __name__ == "__main__":
    main()
