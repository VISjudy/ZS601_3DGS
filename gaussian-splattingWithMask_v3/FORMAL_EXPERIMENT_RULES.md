# 正式实验输出规则

适用于 v3-e 的 150000 步正式实验。

1. 使用显式且互斥的 train、固定 10 张 val、完整 test 相机列表。
2. 输入数据先从 Google Drive 复制到 Colab 本地；模型输出和持久化伪深度写入新的 Drive 目录。
3. 正式训练开始前，必须完成所有 train/val/test 相机的 LiDAR 伪深度导出：
   - 最近 camera-z z-buffer 处理遮挡；
   - 过滤非有限、图像外、相机后方、过近和过远值；
   - 只在局部深度一致时补空洞；
   - 保存 16-bit PNG，0 为无效；
   - 从保存后的 PNG 解码、反投影到 3D，并对原始 LiDAR 做最近邻检查；
   - 任一相机反投影 P95 超过阈值时，训练不得开始。
4. 固定 val 在第 0 步、每 5000 步和最终步输出。保存 RGB、法向、深度、1σ 彩色椭球 PNG，不保存 geometry.npz。
5. checkpoint 与 PLY 在第 50000、100000、150000 步保存。
6. 最终第 150000 步必须：
   - 对完整 test 集计算 masked PSNR、masked MAE 和明确标注语义的 SSIM；
   - 保存 test_metrics.csv 和 test_summary.json；
   - 对 masked PSNR 最差的 10 张保存 RGB、1σ 椭球、深度、法向图；
   - 生成 experiment_summary.md，记录相对 baseline A 的功能差异、关键超参数、深度数据验证、训练统计和结果分析。
7. loss_log.csv 每一步记录所有 loss，包括 LiDAR 深度 raw、调度权重、加权值、有效像素数、覆盖率与距离权重。
8. 每次训练或恢复使用新的模型输出目录。完整且身份一致的 LiDAR 深度数据集可只读复用；不得覆盖不完整或不匹配的目录。
9. 释放 Colab 前必须确认 completed.json、CSV、checkpoint、最终 test、总结文档以及伪深度 verification.json 已写入 Drive。
