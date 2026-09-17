# 统一实验输出规范

## 目录

```text
<run_id>/
├── run_config.json
├── run_manifest.json
├── environment.json
├── evaluation_status.json
├── loss_log.csv
├── training_progress.csv
├── val_metrics.csv
├── geometry_metrics.csv
├── val/
│   └── iteration_000000/
│       ├── manifest.json
│       └── <camera>/
│           ├── rgb_render.png
│           ├── rgb_gt.png
│           ├── ellipsoid_1sigma.png
│           ├── depth_render.png
│           ├── normal_render.png
│           ├── depth_gt.png
│           ├── depth_valid.png
│           ├── normal_gt.png
│           └── normal_valid.png
├── checkpoints/
├── point_cloud/
├── test_final/
│   └── iteration_<final>/
│       ├── test_metrics_per_camera.csv
│       ├── test_summary.json
│       └── worst10/
├── geometry_test/
│   ├── geometry_metrics.json
│   ├── geometry_metrics.csv
│   └── manifest.json
├── results_table.csv
├── results_table.md
├── results_table.tex
├── experiment_summary.md
├── verification.json
└── completed.json
```

项目已有命名可以通过验收脚本参数适配；新项目优先使用上述结构。

## CSV 最低字段

- `loss_log.csv`：`iteration,total_loss`，以及每项 loss 的 `raw/weight/weighted/state`。
- `training_progress.csv`：`iteration,elapsed_seconds,gaussian_count`。
- `val_metrics.csv`：`iteration,image_name,psnr,ssim,mae`，每轮包含 `MEAN` 汇总行。
- `geometry_metrics.csv`：`iteration,gaussian_count`，以及项目定义的贴面距离、尺度和法向统计。
- `test_metrics_per_camera.csv`：`image_name,psnr,ssim,mae`。
- `geometry_test/geometry_metrics.csv`：最终模型的 accuracy、completeness、Chamfer-L1/L2 及各距离阈值 F-score。

mask 指标必须写明有效区域、归一化分母和 SSIM 语义。

## 固定频率

- Val：iteration 0、每 5000 步、最终步。
- Checkpoint/模型：每 50000 步、最终步。
- 默认不保存大体积逐轮 NPZ；如需保存必须显式开启。

## Val 诊断语义

- `ellipsoid_1sigma.png` 使用当前高斯的三个真实尺度轴和旋转；颜色来自项目定义的可解释颜色源。
- 深度图注明 camera-z 或 ray distance、近远范围及无效值。
- 法向图注明坐标系和 RGB 编码；无效像素由独立 valid mask 表示。
- 伪 GT 与渲染图必须使用同一相机、分辨率和遮挡规则。

## 正式训练后的强制评估

完整轮次训练结束后必须执行两类独立评估：

1. **测试集视觉指标**：对全部 test 相机逐张计算 PSNR、SSIM、MAE，保存逐相机 CSV 与汇总 JSON；按主指标排序保存最差 10 张的 RGB、1σ 椭球、深度和法向诊断。
2. **最终点云几何指标**：将最终高斯中心或导出的表面点与已登记参考点云比较，保存 prediction→reference accuracy、reference→prediction completeness、Chamfer-L1/L2、明确阈值下的 Precision/Recall/F-score；当两侧都有法向时增加 normal consistency，有 Gaussian scale 字段时增加尺度与长宽比统计。

几何报告必须记录单位、坐标对齐说明、采样数量、阈值、输入路径和 SHA256。`reference_role=initialization_lidar` 只证明最终模型贴合初始化 LiDAR；要声称独立几何精度，必须使用 `heldout_lidar` 或 `mesh_gt`。


## 缺失指标

不同项目无法提供某一诊断或几何真值时，不生成伪数据。将该项在 `evaluation_status.json` 标为 `skipped` 并写明原因、尝试过的 adapter 或命令。`summarize_experiment.py` 会把对应字段写为 `null`，并在“未运行或缺失的指标”中列出原因。没有状态文件和原因的缺失会导致正式验收失败。
