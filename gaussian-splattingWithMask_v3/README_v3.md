# ZS601 LiDAR 3DGS v3 标准实验框架

本目录包含 ZS601 L2 PRO 数据的 mask-aware LiDAR 3DGS 标准化实现。代码同时支持项目 baseline 和 A–E 几何实验，通过统一实验组与独立 `auto|on|off` 开关组合，不需要复制多份训练代码。

## 快速入口

- 环境和完整操作习惯：[`docs/ENVIRONMENT_AND_WORKFLOW.md`](docs/ENVIRONMENT_AND_WORKFLOW.md)
- 输入数据规范：[`docs/DATASET_INPUT.md`](docs/DATASET_INPUT.md)
- SfM、LiDAR、法向与深度预处理：[`docs/DATA_PREPROCESSING.md`](docs/DATA_PREPROCESSING.md)
- Val 椭球、深度和法向诊断：[`docs/VAL_DIAGNOSTICS.md`](docs/VAL_DIAGNOSTICS.md)
- 正式实验规则：[`FORMAL_EXPERIMENT_RULES.md`](FORMAL_EXPERIMENT_RULES.md)
- Agent 协作规范：[`AGENTS.md`](AGENTS.md)
- 恢复记忆：[`MEMORY.md`](MEMORY.md)
- 机器可读实验登记：[`experiments/experiment_registry.json`](experiments/experiment_registry.json)
- 标准 Colab：[`colab/ZS601_STANDARD_v3.ipynb`](colab/ZS601_STANDARD_v3.ipynb)

## 主要命令

```bash
# 分阶段预处理
python scripts/preprocess_dataset_v3.py --experiment-group A --stage all ...
python scripts/validate_dataset_v3.py --output <scene>/processed_v3

# 训练。未传 --experiment 时当前默认是 A；baseline 必须显式指定 original。
python train_mask_v3.py --experiment original ...
python train_mask_v3.py --experiment B ...
python train_mask_v3.py --experiment custom --init_flatten on --surface_loss on ...

# 补渲染固定相机诊断到新目录
python scripts/render_val_diagnostics_v3.py ...

# 验收实验目录
python scripts/verify_run_outputs_v3.py --run <output> --iterations 150000 --val-interval 5000
```

## 实验组

| 组 | 新增功能 |
|---|---|
| `original` | 新训练约束全部关闭；项目 baseline |
| `A` | 法向/扁平初始化、相机定向、剪枝、诊断 |
| `B` | A + 五项几何 loss |
| `C` | B + 硬尺度边界 |
| `D` | C + 贴面受控增密 |
| `E` | C + LiDAR 深度监督 |
| `custom` | 逐项组合 |

这里的扁平初始化仍使用 3DGS 光栅器，只把 3D 高斯的局部第三轴压薄并按点云法向初始化，不等同于完整 2DGS 圆盘光栅器。

## 当前验证边界

提交 `09cf061` 已于 2026-09-17 在全新 NVIDIA L4 会话完成两个 CUDA 扩展编译并通过 54/54 项测试。该验收覆盖代码、CUDA 和配置逻辑；当前标准化提交尚未用真实 ZS601 数据完成新的 200 步冒烟或 150k 正式训练。
