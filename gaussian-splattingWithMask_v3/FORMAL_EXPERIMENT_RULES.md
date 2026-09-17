# v3 正式实验与产物规则

本规则适用于 `original/A/B/C/D/E/custom`。每组的功能矩阵以 `experiment_presets_v3.py` 和运行时 `run_config.json` 为准。

1. 每次运行记录实验组、独立覆盖项、Git commit、GPU、随机种子、数据 manifest 与输出目录。
2. 使用显式且互斥的 train、固定 10 张 val 和完整 test 相机列表。
3. 原始数据只读；Drive 输入先复制到 Colab 本地，预处理和训练使用新目录。
4. 预处理必须通过相机/RGB/mask 对应、`.bin/.txt` 一致性、点云尺度、投影、法向定向和深度回投验证。
5. 正式训练前运行 200 步冒烟，并保持正式实验的 150000 步学习率 horizon。
6. 新训练在 iteration 0、之后每 5000 步和最终步保存固定 val10：RGB render/GT、1σ 彩色椭球、渲染深度、渲染法向、LiDAR 伪 GT 与 valid mask。默认不保存 `geometry.npz`。
7. 150k 正式实验在 50000、100000、150000 步保存 checkpoint 与 PLY。
8. 第 150000 步对完整 test 集计算逐相机指标，保存汇总，并为指标最差 10 个相机保存 RGB、1σ 椭球、深度和法向。
9. 每步记录 `loss_log.csv`；同时保存 `training_progress.csv`、`val_metrics.csv`、`geometry_metrics.csv` 和运行时间、高斯数量变化。
10. 结束时生成 CSV、Markdown、LaTeX 论文格式结果表和 `experiment_summary.md`，明确相对 baseline 改动、非默认超参数、结果与限制。
11. E 或显式开启 LiDAR 深度监督时，伪 GT 使用 camera-z、z-buffer、近远裁剪、独立 valid mask和距离加权；正式训练前必须通过保存文件解码、三维反投影及重投影验收。
12. 任何训练或恢复使用新输出目录。已有深度缓存只有身份和验证一致时才能只读复用。
13. Drive 回读必须确认 `completed.json`、CSV、三个 checkpoint、最终 test、最差 10 和总结文档存在。完成回读后才能把状态改为“已完成”并释放 GPU。
14. 缺少真实日志或文件时在实验登记中写“待核验”或 `null`，不得补造结果。
