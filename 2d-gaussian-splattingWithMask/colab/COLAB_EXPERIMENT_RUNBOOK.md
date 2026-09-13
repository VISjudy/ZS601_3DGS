# ZS601 2DGS Colab 实验运行与排错手册

本文件记录可复用的运行流程和已经验证过的故障修复。目标是让新的 Colab 会话或新的 Codex 对话能直接恢复实验，不重复踩坑。

## 固定实验约束

- GitHub 分支：`2dgs-zs601-mask-init`
- Notebook：`colab/ZS601_2DGS_LiDAR_B_then_C_L4.ipynb`
- L4 架构：`TORCH_CUDA_ARCH_LIST=8.9`
- B：LiDAR 初始化，`lambda_dist=0`
- C：LiDAR 初始化，`lambda_dist=1000`
- 全分辨率、mask、同一训练/测试划分、随机种子和训练参数
- 150k 轮；50k、100k、150k 保存 checkpoint
- 每 5k 使用同一 manifest 的固定 10 相机，输出 RGB、depth、normal、Gaussian ellipsoid 与 metadata
- 所有断点、日志、预览和指标保存在 Drive 独立目录

## 标准启动顺序

1. 检查 L4、CUDA、Python 与显存。
2. 挂载 Drive。
3. 拉取最新分支，执行 `py_compile` 校验启动脚本。
4. 检查依赖；干净运行时只在独立“安装依赖”Cell 中安装。
5. 读取 Drive 的 `status.json` 和 checkpoint，决定新跑、断点恢复或只补后处理。
6. B 完成评测后才进入 C 冒烟测试。
7. C 冒烟测试通过且峰值显存不超过 13.5GB 后进入正式 150k。
8. 最后验证指标文件与每个 5k 节点的四类固定视角预览。

## 已解决问题

### 1. GitHub Notebook 更新后 Colab 仍显示旧内容

- 症状：刷新分支 URL 后仍看到旧 Cell。
- 原因：Colab 对 GitHub 分支 URL 有缓存。
- 修复：使用精确提交 URL 打开，例如 `.../blob/<commit-sha>/...ipynb`；Cell 内仍从目标分支拉取最新代码。
- 验证：确认 Notebook 中出现最新的小标题或参数，并打印当前 Git commit。

### 2. 启动脚本语法损坏，`py_compile` 失败

- 症状：checkpoint 正常但主 Cell 在训练前立即报 `CalledProcessError`。
- 原因：自动字符串替换把 JavaScript replacement string 中的 `$'` 当成特殊替换标记，导致 Python 代码段被拼接错位。
- 修复：从已知正确提交恢复文件，用 replacement callback 做精确替换；提交前检查关键函数数量、结尾评测代码，并执行 `python -m py_compile`。
- 保护：Notebook 的同步 Cell 每次训练前强制运行语法检查。

### 3. B 已训练到 150k，但渲染因 `mediapy` 缺失失败

- 症状：`render.py` 导入 `utils/render_utils.py` 时出现 `ModuleNotFoundError: mediapy`。
- 原因：`mediapy` 只用于视频路径，但被静态图片评测在模块导入阶段强制加载。
- 修复：将 `mediapy` 改为可选导入；静态测试渲染不再要求该包。复用 `chkpnt150000.pth` 并用 `--postprocess-only` 补跑渲染与指标，禁止重训覆盖。
- B Drive 目录：`zs601_2dgs_B_lidar_parallel_20260912_162033`

### 4. 新 L4 运行时依赖缺失 / Python 3.13 不兼容

- 症状：语法检查通过，但缺少 `plyfile`、`laspy`、`trimesh`、`open3d`、`diff_surfel_rasterization`、`simple_knn`。普通包可安装，`open3d` 返回 `No matching distribution found`，两个 2DGS CUDA 扩展构建 wheel 失败。
- 原因：新的 Colab 后端是干净环境；当前 Python 3.13 没有可用的 `open3d` wheel，且与仓库中的 2DGS CUDA 扩展构建链不兼容。Drive 数据和 Python/CUDA 环境是两回事。
- 已验证：`plyfile`、`laspy[lazrs]`、`trimesh`、`scikit-image`、`gdown` 在 Python 3.13 安装成功；失败项仅为 `open3d`、`diff-surfel-rasterization`、`simple-knn`。
- 修复：Colab 选择过去的 Python 3.11/3.12 运行时，同时保留 L4 GPU；再逐项安装依赖和两个 submodule。Notebook 在安装前主动拒绝 Python 3.13，并打印明确提示。
- 保护：依赖检测与安装拆为独立 Cell，安装完成后再次逐项检查。后续 B/C 命令使用 `--skip-install`，避免训练 Cell 中静默安装。
- 注意：CUDA 扩展必须从仓库 submodule 构建，且 `TORCH_CUDA_ARCH_LIST=8.9` 匹配 L4。

### 5. Colab 页面显示未使用 GPU，但 Drive 文件仍变化

- 原因候选：打开的是另一个 notebook/runtime 标签页，或页面已与实际运行后端断开。
- 检查：以当前 Cell 的 `nvidia-smi`、运行状态栏和 `status.json.updated` 三者为准；不要只看浏览器标签页是否滚动。
- 恢复：在目标 Notebook 重新连接 L4，挂载 Drive，读取 checkpoint 和 `status.json` 后恢复。

## 断点恢复决策

- `stage=complete`：跳过该组。
- 存在 `chkpnt150000.pth` 且评测失败：使用 `--postprocess-only --resume-output <run_dir>`。
- 存在较早 `chkpnt*.pth`：使用 `--resume-training --resume-output <run_dir>`。
- C 的 `stage=preflight_complete`：使用 `--skip-preflight --resume-output <run_dir>` 开始正式训练。
- 无 checkpoint 且状态失败：保留旧目录，新建运行目录；不要覆盖现场。
- 坐标一致性失败或出现 NaN/Inf：停止后续阶段，不猜测变换。

## 日志与验收

- `status.json`：阶段与最后更新时间
- `train.log`：loss、PSNR、distortion、normal、Gaussian 数量、速度
- `preflight.log`：冒烟测试与显存峰值
- `render.log` / `metrics.log`：测试渲染与 PSNR/SSIM/LPIPS
- `geometry_metrics.json`：LiDAR 几何误差与覆盖率
- `results_summary.json`：单组最终汇总

每遇到新的可复现故障，都在本文件追加“症状、原因、修复、验证、恢复方式”，并同步修改 Notebook 的保护检查。
