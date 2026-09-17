# ZS601 3DGS v3 协作说明

本目录是 v3 标准化实验实现。上级 `AGENTS.md` 的安全、Git、实验和验证规则继续生效。

## 稳定规则

- 原始 RGB、mask、相机标定、LAS/PLY 和已有实验产物只读；派生数据写入场景的 `processed_v3/`。
- 不覆盖既有输出。训练、恢复、预处理和评估都使用新的运行目录。
- 训练相机可参与法向可见侧定向；val/test 相机不得参与。
- 深度统一为相机坐标 z-depth。无效深度和法向使用独立 valid mask。
- 实验状态和指标必须有日志、manifest、实验 summary 或 Drive 回读证据；缺少证据时写“待核验”或 `null`。
- `original` 关闭所有新增功能时应保持原始路径，不新增 loss、优化器项或随机数调用。

## 入口与文档

- 训练：`train_mask_v3.py`
- 参数：`arguments_v3.py`
- 数据与几何：`data_v3.py`、`geometry_v3.py`
- 渲染与运行状态：`render_v3.py`、`runtime_v3.py`
- 环境与实验流程：`docs/ENVIRONMENT_AND_WORKFLOW.md`
- 输入规范：`docs/DATASET_INPUT.md`
- 预处理：`docs/DATA_PREPROCESSING.md`
- 验证诊断：`docs/VAL_DIAGNOSTICS.md`
- 实验登记：`experiments/experiment_registry.json`
- Val 诊断补渲染：`scripts/render_val_diagnostics_v3.py`
- 运行产物验收：`scripts/verify_run_outputs_v3.py`
- 总记录生成：`scripts/build_experiment_record_docx.py`

## 标准流程

1. 校验数据身份、路径、坐标、单位和划分。
2. 运行 SfM、LiDAR、监督图和 validate 阶段，保留各阶段 manifest。
3. 通过本地静态检查和目标测试。
4. 优先用 L4 完成 200 步冒烟，回读 Drive `verification.json`。
5. L4 正式运行，按 5k 验证、50k checkpoint 协议保存。
6. 从 Drive 回读并核验产物后更新实验登记和总记录。
7. 用 Documents 工作流渲染 DOCX，逐页检查中文字体、分页和表格截断。

## 实验产物

正式实验使用新目录，并保留运行配置、Git commit、数据 manifest 哈希、GPU、随机种子、日志、指标、固定相机渲染、checkpoint 和 Drive 回读证据。实验登记中的 `verified` 信息必须能反查到这些文件。
