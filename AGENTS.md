# 3DGS 统一实验协作规则

本文件适用于采用本模板的 3DGS 项目。项目自己的 AGENTS.override.md 可以补充算法入口和数据路径，但不得弱化原始数据保护、实验证据和不可覆盖规则。

## 1. 数据与安全

- 原始 RGB、mask、相机标定、点云、历史 checkpoint 和实验产物只读。
- 所有预处理、训练、恢复和评估写入新的唯一目录。
- 删除本地、云端或实验文件前必须获得用户对明确对象的确认。
- 不在日志、文档或仓库中保存 token、授权码、密码或账户标识。

## 2. 功能与配置

- 每项新增功能必须有独立开关；推荐 `auto|on|off`。
- 可以定义实验组 preset，但 preset 必须集中维护并能被独立开关覆盖。
- 关闭新增功能时必须保持项目 baseline 的优化路径和随机行为。
- 启动时打印实验组、覆盖项、最终功能矩阵和关键非默认参数。
- 同样信息写入 `run_config.json`，同时记录 Git commit、数据 manifest 哈希、GPU、随机种子和输出目录。

## 3. 标准流程

1. 验证环境、GPU 和 CUDA 扩展。
2. 将云端数据复制到计算节点本地盘，训练期间不直接读取网络盘大文件。
3. 运行数据预处理和独立验证，保存 manifest。
4. 运行 200 步冒烟，并使用正式实验的学习率 horizon。
5. 冒烟通过后在新目录运行正式实验。
6. 固定验证相机在 iteration 0、每 5000 步和最终步输出诊断。
7. 每 50000 步保存 checkpoint；除非项目另有明确登记，不高频占用云存储。
8. 正式结束后计算完整 test 指标、最差 10 相机诊断和论文格式结果表。
9. 复制到持久存储后回读验证，随后释放 GPU。

## 4. 统一产物

正式实验至少保存：

- `run_config.json`、`run_manifest.json`、`environment.json`
- `loss_log.csv`、`training_progress.csv`、`val_metrics.csv`、`geometry_metrics.csv`
- 固定验证相机的 RGB、1σ 椭球、深度、法向和相应 valid mask
- `checkpoints/iteration_*.pth` 和模型几何文件
- `test_final/test_metrics_per_camera.csv`、`test_summary.json`
- 最差 10 个测试相机的 RGB、椭球、深度和法向
- `results_table.csv`、`results_table.md`、`results_table.tex`
- `experiment_summary.md`、`verification.json`、`completed.json`

## 5. 完成判定

- 页面显示、进程存在、单条日志或一个 checkpoint 都不代表实验完成。
- “完成”必须有有限 loss、预期 checkpoint、最终 test、总结、completed 标记和持久存储回读证据。
- 缺少证据时使用“待核验”或 `null`，不得补造结果。
- 一个实验组完成不等于整组对比完成。

## 6. 任务索引

| 关键词 | 文档 |
|---|---|
| 环境、GPU、CUDA、Colab | `docs/ENVIRONMENT.md`、`docs/COLAB_WORKFLOW.md` |
| 数据、相机、点云、mask、预处理 | `docs/DATASET_CONTRACT.md` |
| 冒烟、正式训练、恢复、完成 | `docs/EXPERIMENT_PROTOCOL.md` |
| CSV、Val、checkpoint、最差10、表格 | `docs/OUTPUT_SCHEMA.md` |
| 椭球、深度、法向、adapter | `docs/ADAPTER_GUIDE.md` |
