# 1 cm mesh 点云 + scale 0.5：完整 200 虚拟视角

已完成 200 个唯一相机，7180820 个初始化高斯，k=3、opacity=0.999999、SH0，优化步数为 0。sigma 中位数 4.397661 mm。原四个冒烟相机的 28 张 PNG 像素逐值完全一致；所有 Gaussian PLY 字段与 v004 完全相同。相机集合、内参和外参保持原 virtual_near 数据。

全 200 张 Black RGB 对 Blender GT 的 PSNR 均值 19.40260 dB，SSIM 0.756344；深度 MAE 16.2987 mm。掩膜间隙比例平均 0.234295%，最大 2.838871%（003517.png）。严格 alpha=0 总像素数 0。全量均值不能与四视角均值混作同一组比较。

## 目录与 PNG 格式

数据根为 `full/`，PNG 尺寸均为 640 × 1108，文件名对应 COLMAP 图像名。

| 目录 | 格式 | 含义 |
|---|---|---|
| images | uint8 RGB，3 通道 | 标准黑底合成 Black RGB，作为本轮虚拟训练图 |
| color | uint8 RGB，3 通道 | 除以累计 alpha 的 Straight RGB |
| rgba | uint8 RGBA，4 通道 | Straight RGB 和累计 alpha |
| alpha | uint16 灰度，1 通道 | alpha = 像素值 / 65535 |
| masks | uint8 灰度，0/255 | 白色有效 alpha≥0.95，黑色无效；训练时黑色像素不参与损失 |
| low_coverage_masks | uint8 灰度，0/255 | 上一项反相，白色为间隙/覆盖不足 alpha<0.95 |
| no_coverage_masks | uint8 灰度，0/255 | 白色仅表示严格 alpha16=0，完全无高斯贡献 |
| training_masks_nonempty | uint8 灰度，0/255 | 只排除严格 alpha16=0 的替代有效掩膜；不等同于 masks/ |
| depth | uint16 灰度，1 通道 | 毫米 camera-Z，Z米=像素值/1000；0无效 |
| depth_mask | uint8 灰度，0/255 | 白色为有效正深度且 alpha≥0.5 |

按用户修正，“颗粒感/间隙”只用 `count(masks==0)/总像素数` 表示。阈值是 alpha=0.95，包含覆盖不足位置；严格 alpha=0 比例另列。对于训练，使用 `images/` + `masks/`，并在读取和损失里明确应用 mask。仅把文件放在目录中不会使所有训练器自动使用它。

深度是 alpha 归一化的调和 camera-Z：`Z = alpha / sum(T_i * alpha_i / Z_i)`，不是欧氏射线长度或精确 mesh 表面交点。各向同性初始化高斯没有确定的表面法向，未据此生成 normal；对应 Blender normal GT 保留在 `D:/codex/blenderProject/scenes/zs601-meetingroom/synthetic-training-v001/virtual_near/normal/`，按同相机配对。

`full/point_cloud/iteration_0/point_cloud.ply` 是本轮 Gaussian PLY：log 尺度、logit 不透明度、SH0 系数、单位旋转。`full/sparse/0/points3D.ply` 是普通彩色输入点云；同目录保留 `cameras.txt`、`images.txt`、`points3D.txt`，单位米。不要混淆两个 PLY。

## 验证与复现

2000 张 PNG 全部回读，掩膜极性、值域、深度无效值、RGB/alpha 重建、全部相机和 PLY 参数已检查。逐视角视觉、深度及间隙数据见 `metrics/per_view_metrics.csv`，完整检查见 `verification.json`。实际执行的 5 个代码单元 notebook、运行环境、输入/源代码哈希、日志和参数见 `reproducibility/`。

1 cm 点云只用于虚拟视角渲染，正式训练保留原 3 cm 点云，SHA256 未改变。此结果是初始化投影渲染，不是训练收敛结果。虚拟图应只加入训练集，不加入测试或验证集。
