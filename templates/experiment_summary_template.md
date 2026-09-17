# 实验 <group> 总结

- 运行 ID：`<run_id>`
- Git commit：`<commit>`
- 数据 manifest：`<path and sha256>`
- GPU/环境：`<environment.json>`
- 状态：`planned/running/verified/failed`

## 相对 baseline 的修改

- `<feature>`：`off → on`

## 关键非默认超参数

| 参数 | 本实验 | baseline/default | 原因 |
|---|---:|---:|---|
| `<name>` | `<value>` | `<value>` | `<reason>` |

## 训练过程

- 训练时间：`<seconds>`
- 最终高斯数量：`<count>`
- Loss 状态：`<finite/anomaly>`
- Val 阶段观察：`<evidence>`

## 最终 Test

| 指标 | 本实验 | baseline | 差值 |
|---|---:|---:|---:|
| PSNR | `<value>` | `<value>` | `<delta>` |
| SSIM | `<value>` | `<value>` | `<delta>` |
| MAE | `<value>` | `<value>` | `<delta>` |

## 最终几何测试

- 参考点云角色：`<initialization_lidar/heldout_lidar/mesh_gt>`
- 单位与对齐：`<unit and alignment>`

| Accuracy P95 | Completeness P95 | Chamfer-L1 | Chamfer-L2 | F-score@阈值 |
|---:|---:|---:|---:|---:|
| `<value>` | `<value>` | `<value>` | `<value>` | `<value>` |

## 几何与可视化分析

- `<observation linked to files/metrics>`

## 结论与下一步

- `<conclusion>`
- `<next controlled experiment>`
