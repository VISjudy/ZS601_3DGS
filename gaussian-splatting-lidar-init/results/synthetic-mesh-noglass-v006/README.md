# No-glass mesh initialization v006

The following report describes verified local artifacts. Large PLY and image
files are retained in `3dgsResult/zs601-mesh-noglass-v006`, not committed here.
Workspace-specific audit/export scripts are an exact reproducibility record;
they expect the Blender project layout and preserved v003 inputs.

Actual execution: [Colab notebook](../../notebooks/ZS601_NoGlass_1cm_Scale05_Full200.executed.ipynb).
The CLI uploads the hashed point-cloud/camera/source bundles expected by this
notebook. Renderer is unchanged at commit4c7186e363f14050c4977bb192f12ed237112764.

# 去玻璃点云与 200 张虚拟视图 v006

已核验：33 个透明玻璃部件的 176,124 个表面点全部排除；1 cm 点云 7,004,696 点，3 cm 点云 797,520 点。两个文件均为普通彩色 PLY，单位米，含 XYZ、单位法向 nx/ny/nz（float32）及 sRGB red/green/blue（uint8）。

| 用途 | 文件 |
|---|---|
| 生成虚拟图像 | 1 cm 点云（本地文件 `cloud_1cm/points_mesh_1cm_noglass.ply`） |
| 后续正式训练初始化 | 3 cm 点云（本地文件 `cloud_3cm/points_mesh_3cm_noglass.ply`） |
| 本轮初始化高斯参数 | Gaussian PLY（本地文件 `full/point_cloud/iteration_0/point_cloud.ply`） |

新 3 cm 文件是完整非玻璃 mesh 表面点的体素子集，与旧的训练 RGB-D 融合 3 cm 云来源和点数不同。旧文件保留。本轮没有执行正式训练，也没有自动改写旧数据集的 sparse 初始化文件；训练时须显式选择这里的新 3 cm 文件。

## 玻璃识别与采样

冻结源场景中的 `ZS601_Semantic_CabinetGlass` 具有真实自定义属性 `semantic_class=clear_glass`，Principled Transmission Weight=1、IOR≈1.45。根据实际材质槽逐三角面映射，排除 16 个柜门玻璃、16 个玻璃搁板和 1 个门窗玻璃部件。半透塑料、电视屏幕、控制面板和发光涂层仍保留，未仅凭名字含 Glass 排除。

1 cm 使用上一轮确定性 mesh 采样中的非玻璃点，全部保留点的坐标、颜色、法向逐值一致；3 cm 在同一非玻璃集合中按每部件世界坐标体素选择最接近体素中心的真实表面代表点。3 cm 是 1 cm 的子集。体素尺寸不是严格点间最小距离。此为面采样近似，不模拟 LiDAR 扫描遮挡、回波和材质响应。

审计见 [材质判定](audit/semantic_decision.json)、[实际 Blender 材质](audit/material_audit.json) 和 [全点独立检查](cloud_verification.json)。`audit/glass_face_mask.npy` 是初步宽泛透射候选；最终排除依据为 `excluded_clear_glass_faces.npy`。这些是三维导出审计数组，没有新增图像语义 mask。

## 200 张虚拟视图

沿用原 virtual_near 的 200 个相机、640×1108 分辨率、k=3、scale=0.5、opacity=0.999999、SH0；优化步数为 0。移除玻璃后重新计算剩余点的 3NN 尺度，sigma 中位数 4.407881 mm。20,000 个独立精确 3NN 参考检查通过。

| 全 200 张指标 | 原 v005 | 去玻璃 v006 |
|---|---:|---:|
| Black RGB PSNR / dB | 19.402596 | 22.025271 |
| SSIM | 0.756344 | 0.769143 |
| 原有效 mask 的无效像素比例 / % | 0.234295 | 0.250455 |
| depth MAE / mm | 16.298702 | 16.126450 |

003193 玻璃柜案例 PSNR：4.480875 → 16.352816 dB。完整逐视角数据见 [CSV](metrics/per_view_metrics.csv) 与 [配对对比](metrics/versus_glass_v005.csv)。PSNR 使用同一 GT 有效区域，SSIM 使用同一有效窗口；深度排除原 GT 无效/透明几何，仅为被评估不透明区域精度。

!玻璃柜对照（本地文件 `comparisons/003193_three_way.png`）

## 图像格式及 mask

所有图像文件名与 COLMAP `full/sparse/0/images.txt` 对应。相机 txt 与原数据一致；`full/sparse/0/points3D.ply`、`points3D.txt` 记录本轮 1 cm 投影输入，正式训练初始化使用上方独立 3 cm 文件。

| full/ 子目录 | 格式 | 说明 |
|---|---|---|
| images | uint8 RGB | 黑底合成 RGB，虚拟训练图 |
| color | uint8 RGB | 除以累计 alpha 的 Straight RGB |
| rgba | uint8 RGBA | Straight RGB 与 alpha |
| alpha | uint16 单通道 | 累计 alpha=值/65535 |
| masks | uint8 单通道 0/255 | 白色有效，原浮点 alpha≥0.95 |
| low_coverage_masks | uint8 单通道 0/255 | masks 反相，白色为覆盖不足 |
| no_coverage_masks | uint8 单通道 0/255 | 白色仅表示 alpha16=0 |
| training_masks_nonempty | uint8 单通道 0/255 | 只忽略 alpha16=0 的替代有效 mask |
| depth | uint16 单通道 | camera-Z 毫米，0 无效；alpha 归一化的调和深度 |
| depth_mask | uint8 单通道 0/255 | 白色表示正深度且 alpha≥0.5 |

按用户定义，颗粒感/间隙比例为 `count(masks==0)/总像素数`；严格 alpha16=0 比例另列。原 mask 逻辑保留，没有玻璃语义 mask。PLY 附带 mesh 法向，但各向同性初始化高斯不提供确定表面法向图；同相机 Blender normal 真值仍在原 `synthetic-training-v001/virtual_near/normal/`。

## 限制与复现

本轮只改变采样是否包含玻璃，没有重新拟合颜色。非玻璃点沿用原训练视图取色及未观测表面的材质底色回退；柜内遮挡区域仍可能缺少纹理、光照和反射。移除玻璃不会重现玻璃的物理反射与折射。以上是初始化投影与数据检查，不是 3DGS 训练收敛结果。

两种点云回读、全部表面点距离/体素唯一性/来源、200 个相机、2000 张 PNG、Gaussian PLY、14 张预检重复图、notebook 的全部 6 个代码单元均已检查。源 .blend、旧点云与旧结果保留。实际执行 notebook、代码与日志见 `reproducibility/`。