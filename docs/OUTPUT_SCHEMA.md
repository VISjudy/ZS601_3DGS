# 统一实验输出规范

## 目录

```text
<run_id>/
├── run_config.json
├── run_manifest.json
├── environment.json
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
