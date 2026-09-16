# v3 项目恢复说明

此文件只保存稳定语义和恢复指针，不保存未经核验的实验结果。当前状态以仓库根目录 `prompt/registers/experiment-state.md`、`experiments/INDEX.md`、运行目录和 Drive 回读证据为准。

## 数据语义

- Drive 逻辑入口：`MyDrive/LCCDataset/zs601_output/<scene_name>/`。
- LiDAR 输入是上游已去除动态目标的彩色点云；本流程不重复删除动态目标。
- 原始输入只读，所有派生数据写入 `processed_v3/`。
- COLMAP 外参为 world-to-camera，`C = -R^T t`。
- 深度定义为相机坐标 z-depth；长度单位必须按实际数据校验。
- mask 的黑白含义必须用样本确认并写入 manifest。
- PCA 法向符号由训练相机投票确定，表示可见侧，不是严格语义内外。
- 监督图的无效像素由独立 valid mask 表达，不能根据黑色判断。

## 实验协议

- 固定 val10。
- 新训练在 iteration 0、每 5000 步和最终轮输出诊断。
- 150000 步计划对应 31 个验证时间点。
- 每 50000 步保存 checkpoint。
- 正式 150k 使用 L4；T4/L4 均可用于编译与 200 步冒烟。
- 只有 Drive 回读证据齐全后，运行状态才能登记为完成。

## 恢复指针

- 项目当前状态：仓库根目录 `prompt/registers/experiment-state.md`
- 实验历史：仓库根目录 `experiments/INDEX.md`
- 数据规范：`docs/DATASET_INPUT.md`
- 预处理规范：`docs/DATA_PREPROCESSING.md`
- 验证规范：`docs/VAL_DIAGNOSTICS.md`
- 机器可读实验登记：`experiments/experiment_registry.json`
- DOCX 总记录：`reports/ZS601_3DGS_实验总记录_v4.docx`

更新实验状态时先刷新当前项目文件和 Drive 目录。一个实验组完成不等于整个对比完成；恢复训练也不得覆盖历史目录。
