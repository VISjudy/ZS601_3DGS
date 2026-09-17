# 环境配置与标准实验工作流

本文记录 ZS601 LiDAR 3DGS v3 在新 Colab 会话中可重复执行的环境配置、数据搬运、预处理、冒烟、正式训练、验证和恢复习惯。可执行入口是 `colab/ZS601_STANDARD_v3.ipynb`。

## 1. 已验证环境

2026-09-17 对提交 `09cf061` 的 L4 验收环境：

- GPU：NVIDIA L4，23034 MiB。
- Driver：580.82.07。
- Python：3.13.15。
- PyTorch：2.11.0+cu128。
- PyTorch CUDA runtime：12.8。
- CUDA 扩展：`diff-gaussian-rasterization`、`simple-knn` 均从源码编译并成功导入。
- 自动测试：54 passed，0 failed。

以上版本是已验证组合，不表示只能使用这些精确版本。环境变化后必须重新编译 CUDA 扩展并执行测试，不能复用旧会话的编译结论。

## 2. 文件职责

| 文件 | 用途 |
|---|---|
| `AGENTS.md` | 给新 Codex 任务加载的稳定规则、入口和强制流程 |
| `MEMORY.md` | 数据语义、实验习惯和恢复指针；不存放未经验证的实验结论 |
| `experiment_presets_v3.py` | `original/A/B/C/D/E/custom` 功能矩阵 |
| `experiments/experiment_registry.json` | 机器可读的实验计划、状态和真实证据登记 |
| `colab/ZS601_STANDARD_v3.ipynb` | 显式逐单元的完整 Colab 流程 |
| `docs/DATASET_INPUT.md` | 原始 RGB、mask、相机和 LiDAR 输入规范 |
| `docs/DATA_PREPROCESSING.md` | SfM、点云法向、相机定向、深度/法向监督图和回投验证 |
| `docs/VAL_DIAGNOSTICS.md` | RGB、1σ 椭球、深度、法向和最差相机输出协议 |
| `FORMAL_EXPERIMENT_RULES.md` | 200 步冒烟与 150k 正式实验验收要求 |

`AGENTS.md` 使用 Codex 约定的复数大写文件名，不另建含义重复的 `agent.md`。

## 3. 环境建立

在新的 Colab L4 会话中先打印 Python、GPU、显存、驱动、PyTorch 和 CUDA runtime。若正式实验要求 L4而实际 GPU 不匹配，立即停止。

```bash
python -m pip install -q plyfile laspy scipy pillow
python -m pip install -v --no-build-isolation submodules/diff-gaussian-rasterization
python -m pip install -v --no-build-isolation submodules/simple-knn
python -m unittest -v \
  test_experiment_presets_v3 test_preprocess_v3 test_geometry_v3 \
  test_scale_bounds_v3 test_surface_densify_v3 test_lidar_depth_v3
```

编译失败时保留完整 `pip -v` 和 nvcc 输出，根据第一条真实编译错误定位；不要只根据安静模式的 pip 末尾报错猜测。

## 4. 数据搬运原则

1. Drive 输入位置：`MyDrive/LCCDataset/zs601_output/<scene_name>/`。
2. 原始 RGB、mask、相机标定、LAS/PLY 和历史结果只读。
3. 先从 Drive 复制到 `/content` 本地磁盘，再做预处理和训练。
4. 派生数据写入新的 `processed_v3/`；现有目录存在时停止并人工核对。
5. 训练先写 Colab 本地新目录，完成后复制到新的 Drive 运行目录。
6. Drive 回读验证成功后才能把实验登记为完成并释放 GPU。

## 5. 标准执行顺序

1. 选择唯一 `RUN_ID`、实验组和 Git commit。
2. 检查 L4、Python、PyTorch、CUDA，并编译扩展。
3. 复制场景数据到 Colab 本地。
4. 依次运行 `sfm`、`lidar`、`supervision`、`validate` 预处理阶段。
5. 检查 val10 恰好十个相机、划分互斥、投影叠加、法向覆盖和深度回投误差。
6. 用正式 150k 的学习率 horizon 运行 200 步冒烟。
7. 冒烟必须包含 iteration 0 和最终轮诊断、有限 loss、checkpoint 与验证报告。
8. 冒烟通过后使用新输出目录启动 150000 步正式训练。
9. 正式训练结束后运行完整 test、最差 10 相机诊断和结果表生成。
10. 将结果复制到 Drive，回读关键文件并执行 `verify_run_outputs_v3.py`。
11. 更新实验登记和总记录，最后释放 Colab。

## 6. 脚本入口

| 脚本 | 功能 |
|---|---|
| `scripts/preprocess_dataset_v3.py` | `sfm/lidar/supervision/validate/all` 分阶段预处理 |
| `scripts/validate_dataset_v3.py` | 独立验证 `processed_v3` 完整性和阻塞错误 |
| `scripts/render_val_diagnostics_v3.py` | 从模型 PLY 和固定相机补渲染 RGB、1σ 彩色椭球、深度和法向到新目录 |
| `scripts/verify_run_outputs_v3.py` | 检查 val 时间点、checkpoint、最终 test、最差 10 和结果表 |
| `scripts/build_experiment_record_docx.py` | 从真实实验登记生成 DOCX 总记录 |

`render_val_diagnostics_v3.py` 不修改训练模型，输出到新的目录；1σ 椭球显示当前高斯真实三个尺度轴的可视化近似。

## 7. 实验组和独立开关

- `original`：项目 baseline，关闭新增训练约束，保留原始 3DGS 增密/剪枝路径和诊断输出。
- `A`：法向/扁平初始化、相机定向、剪枝和验证诊断。
- `B`：A + 中心贴面、切向、法向、扁平化和尺寸五项几何 loss。
- `C`：B + 硬尺度边界。
- `D`：C + 贴面受控增密。
- `E`：C + LiDAR 深度监督。
- `custom`：所有新增功能默认关闭，逐项开启。

每项功能使用 `auto|on|off`。`auto` 继承实验组，`on/off` 显式覆盖。启动时必须检查控制台功能矩阵和 `run_config.json`，不能只根据输出目录名推断开启状态。

## 8. 固定保存协议

- 新训练从 iteration 0 保存验证诊断，之后每 5000 步保存一次。
- 150k 共计划 31 个验证时间点。
- checkpoint 与 PLY 每 50000 步保存一次。
- 默认不保存 `geometry.npz`。
- 每个 val 相机保存 RGB render/GT、1σ 彩色椭球、深度、法向、LiDAR 伪 GT 及 valid mask。
- 正式结束保存每相机测试指标、最差 10 相机可视化、loss/训练进度/几何变化 CSV、Markdown/LaTeX 表格和 `experiment_summary.md`。

## 9. 恢复与失败处理

- 不覆盖旧运行目录；恢复训练也写入新的输出目录并记录父 checkpoint。
- 会话中断前先检查进程、日志、checkpoint 和 Drive 状态，避免启动重复正式实验。
- `completed.json`、最终测试、三个正式 checkpoint 和 Drive 回读证据缺一项时，不登记为“正式完成”。
- 认证、Drive 授权和密码输入由用户本人完成；日志和文档不得保存 token、授权码或账户信息。
