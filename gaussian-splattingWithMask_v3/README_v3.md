# ZS601 LiDAR 3DGS v3 — 实验 E

实验 E 以 C 为基线：保留法向/扁平初始化、相机定向、五项几何损失、硬尺度上限和低 opacity 剪枝；关闭增密与 opacity reset；新增 LiDAR 伪深度监督。所有功能仍可用同名 `on/off` 参数独立覆盖，组合入口为 `--experiment E`。

[打开 E 组 Colab](https://colab.research.google.com/github/VISjudy/ZS601_3DGS/blob/v3-e/gaussian-splattingWithMask_v3/colab/ZS601_E_v3_150k.ipynb)

## 深度数据契约

- 深度定义为相机坐标系 z，不是欧氏射线距离。
- 世界点按 `camera = world @ R + T` 变换，使用当前居中 FoV 内参；投影以 `(u+0.5,v+0.5)` 为像素中心。
- 同一像素只保留最小正 z，形成遮挡感知 z-buffer。
- 非有限值、相机后方、图像外、过近、过远的点均无效。
- 半径 1 的补洞要求至少 2 个原始 LiDAR 命中；局部深度跨度必须满足 `spread <= 0.02 + 0.02 * nearest_depth`，遮挡边缘不会跨层补值。
- 伪 GT 保存为 16-bit PNG：0 是无效值，1–65535 线性映射 `lidar_depth_min..lidar_depth_max`。
- 正式训练前会重新读取已保存 PNG，反投影为 3D，再计算到原始 LiDAR 的最近邻距离及源点重投影误差。默认要求每张至少 64 个像素且覆盖率至少 0.1%；逐相机 Q99 最近邻距离不超过 0.06 场景单位、阈值内比例至少 99%、Q99 重投影误差不超过 2 px，否则训练不会开始。
- `depth_manifest.csv` 保存逐相机覆盖率、深度范围、PNG 量化误差和反投影误差；`dataset_summary.json` 与 `verification.json` 保存总体验收。
- train/val/test 的原始深度 PNG 全部保留。为控制 Drive 空间，彩色预览只保存 val/test；训练缓存使用同一 uint16 编码的 Colab 本地 PNG，不写入 Drive。
- 恢复训练可以传入既有的已验证深度目录。程序只读复用，不覆盖内容；身份或验证不匹配会停止。

## 深度损失

预测深度使用现有可微高斯属性光栅化，计算 opacity 归一化的高斯中心 camera-z。监督像素必须同时满足：

1. LiDAR 深度有效；
2. 预测深度有限并位于配置范围；
3. 渲染 alpha 不低于阈值；
4. 位于图像有效 mask 内。

每像素先计算相对深度误差 `(pred-gt)/gt` 的 Smooth L1，再乘距离权重：

```
weight = clamp((median_valid_depth / gt_depth) ** distance_power,
               weight_min, weight_max)
```

默认 `distance_power=1`、范围 `[0.25, 4]`，因此近处几何得到更高权重，同时限制极端放大。最终深度项从第 1000 步开始，在 4000 步内升至 `lambda_lidar_depth=0.05`。这些参数均可独立修改。

`loss_log.csv` 每步记录深度 loss 的 raw/weight/weighted、有效像素数、LiDAR 像素数、渲染覆盖率、平均目标深度、平均距离权重和状态。运行开头的 `[RUN]` 会打印 E preset 与所有覆盖参数；预生成阶段打印覆盖率和反投影 P95；manifest 另保存 Q99、通过率、重投影误差、文件尺寸和 SHA-256。

## 正式运行规则

- 训练 150000 步。
- 固定 val 每 5000 步输出 RGB、法向、深度和 1σ 彩色椭球 PNG；不保存 geometry.npz。
- checkpoint 和 PLY 只在 50000、100000、150000 步保存。
- 第 150000 步对完整 test 集计算指标，保存 `test_metrics.csv`，并为 masked PSNR 最差的 10 个相机保存 RGB、1σ 椭球、深度和法向图。
- 运行结束生成 `experiment_summary.md`，列出相对 baseline A 的功能差异、关键超参数、LiDAR 深度验证、最终 test 指标和结果分析。
- 每次模型输出使用新目录。数据集、旧结果和旧 checkpoint 不覆盖。

## 验证

CPU 测试：

```bash
python -m unittest -v test_geometry_v3 test_scale_bounds_v3 test_lidar_depth_v3
```

Colab notebook 还执行 L4 检查、CUDA 扩展构建、200 步 E 冒烟、深度 loss 生效检查及正式产物验收。CPU 测试不代表 CUDA 或正式训练已经完成。
