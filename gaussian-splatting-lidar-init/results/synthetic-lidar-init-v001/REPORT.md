# ZS601 彩色仿真点云初始化 3DGS

本轮已在 Colab L4 通过官方 CLI 实际执行并取回，优化步数为 **0**。
输入为 `synthetic-training-v001` 仅训练视图融合的 3cm 体素彩色点云，
共 **309,240 个高斯**。使用原有 `virtual_near` 的 **200 个相机**，
分辨率 **640 × 1108**，相机内外参和文件名保留。

- 初始化高斯：`point_cloud/iteration_0/point_cloud.ply`。
- 原始彩色点云：`sparse/0/points3D.ply`；对应点数据也写入 `points3D.txt`。
- `images/`：200 张标准黑底 alpha 合成 RGB。
- `color/`：200 张除以累计 alpha 后的 RGB，符合原 v4 输出意图。
- `rgba/`、`alpha/`、`masks/`、`depth/`、`depth_mask/`：各 200 张 PNG。
- `sparse/0/`：对应 COLMAP TXT 内外参和点数据。
- 格式、位深、通道、坐标和解码方法见 [FORMAT.md](FORMAT.md)。
- 实际运行 notebook、固定源码归档、环境和编译/测试日志在 `reproducibility/`。
- `per_view_metrics.csv` 包含全部 200 张图的评价；`comparison_contact_sheet.jpg` 为抽样对照。

模型 opacity 目标 **0.999999**，存储为有限 logit；实际解码约
**0.999998986721**。标准光栅器单片段 alpha 上限仍为 **0.99**。
尺度使用 **0.5 × 三近邻均方距离的平方根**，中位数
**12.9245 mm**，SH 阶数为 0。

## 本轮测量结果

| 指标（200 视角均值） | 数值 |
|---|---:|
| 黑底 RGB PSNR，GT 有效区域 | 21.0055 dB |
| 去 alpha RGB PSNR，GT 有效区域 | 23.2903 dB |
| 去 alpha RGB PSNR，alpha≥0.95 有效交集 | 22.5894 dB |
| 黑底 RGB SSIM，完整有效窗口 | 0.745425 |
| 去 alpha RGB SSIM，高覆盖完整窗口 | 0.798365 |
| alpha≥0.95 像素比例 | 69.665% |
| depth 有效像素比例 | 99.263% |
| depth MAE，与 Blender 几何 GT 有效交集 | 34.1743 mm |

RGB/深度评价对象是已冻结的 Blender 仿真 GT。深度采用 alpha 归一化的
调和 camera-Z，PNG 为 uint16 单通道毫米，0 无效；它是高斯合成深度，
不是精确表面求交深度。单高斯已知 Z 的 CUDA 数值测试已通过。

这是**初始化渲染结果，未做训练**。近距离表面仍可见颗粒、覆盖缝隙和模糊，
不宣称达到 Blender GT 或实拍逼真度。`color/` 若加入训练，应配合有效掩膜，
并维持这些虚拟图像仅用于训练增强的既定划分。各向同性高斯没有唯一表面法向，
本轮未伪造 normal 图；原数据集保留 Blender normal GT。

已验证：200 个相机精确对应、1400 张 PNG 格式/解码、点坐标完全保留、
高斯 PLY 精确重载、全部参数有限、实际执行 notebook 五个代码单元无错误，
全部清单文件 SHA256 复核通过。首次缺少 cstdint 头文件的编译失败记录保存在实验目录，
正式使用提交 `4c7186e363f14050c4977bb192f12ed237112764`；数学渲染内核未改动。

GitHub 分支：[synthetic-lidar-init-v4](https://github.com/VISjudy/ZS601_3DGS/tree/synthetic-lidar-init-v4/gaussian-splatting-lidar-init)。
