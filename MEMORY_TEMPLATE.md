# 3DGS 项目记忆模板

此文件只保存稳定事实、当前状态和恢复指针。实验详细日志留在 `experiments/` 或输出目录中，不把计划写成已完成结果。

## 项目身份

- 项目名称：`<project>`
- 仓库：`<repository>`
- 当前 baseline：`<commit/tag>`
- 训练入口：`<path>`
- 数据入口：`<path>`
- 输出根目录：`<path>`

## 数据语义

- 相机外参约定：`<world-to-camera/camera-to-world>`
- 深度语义：`<camera-z/ray-distance>`
- 点云单位：`<meters/scene units>`
- mask 有效值：`<black/white/alpha>`
- 固定 train/val/test 划分：`<manifest>`
- 固定验证相机：`<list and hash>`

## 实验习惯

- 首选 GPU：L4。
- 云端数据先复制到本地盘。
- 先做 200 步冒烟，再做正式训练。
- Val：iteration 0、每 5000 步、最终步。
- Checkpoint：每 50000 步。
- 每次运行使用新目录，不覆盖历史产物。
- 完成后执行完整 test、最差 10 相机诊断、结果表和持久存储回读。

## 当前状态

- 最近已验证提交：`<commit>`
- 最近冒烟：`<status and evidence path>`
- 正式实验：`<status and evidence path>`
- 阻塞项：`<blocker>`
- 下一步：`<next action>`

## 恢复指针

- 实验登记：`<path>`
- 当前状态：`<path>`
- 环境记录：`<path>`
- 数据 manifest：`<path>`
- 最近运行 summary：`<path>`
