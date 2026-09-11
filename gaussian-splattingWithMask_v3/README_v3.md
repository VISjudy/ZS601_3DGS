# LiDAR D v3

基于 main `d45646bf3944599d0470e72eca7365dcd420191d` 的 `gaussian-splattingWithMask`。
这是独立的 `gaussian-splattingWithMask_v3` 目录，包含 v3 训练入口及其实际依赖。`v3-ab` 分支根目录只保留本文件夹；main/v2 分支历史不变。原主基线工作目录在本地保留。

[打开 Colab](https://colab.research.google.com/github/VISjudy/ZS601_3DGS/blob/v3-ab/gaussian-splattingWithMask_v3/colab/ZS601_D_v3_150k.ipynb)

进入本目录后执行以下命令。CUDA源码、GLM头文件和许可证随目录保留；不包含旧训练入口、旧notebook、GLM文档/测试和实验产物。不要使用v2-dev扩展。

## 运行

```bash
python train_mask_v3.py --experiment D \
  -s /content/dataset -m /content/drive/MyDrive/results/A_unique \
  --point_cloud /content/dataset/ZS601_3cm_sample.las \
  --cameras_file /content/dataset/sparse/cameras.txt \
  --train_file /content/work/train_v3.txt \
  --val_file /content/dataset/sparse/images-val10.txt \
  --test_file /content/dataset/sparse/images_test.txt \
  --iterations 150000 --position_lr_max_steps 150000
```

`--experiment D` 组合开启初始化、五项几何约束、尺度上限和贴面受控增密。例如 `--experiment D --surface_densify off` 单独关闭增密并退化为C。每个功能只有 `on/off` 一种覆盖表达；不接受旧 `--init_2d` / `--freeze_2d_z` 等混合控制。

| 功能参数 | baseline A | D |
|---|---|---|
| init_normal / init_flatten / orient_cameras / pruning | on | on |
| surface_loss / tangent_loss / normal_loss / flatten_loss / size_loss | off | on |
| scale_bounds / surface_densify | off | on |

标准增密、深度训练、opacity reset和硬厚度复位固定关闭；D只允许surface_densify_v3.py中的贴面受控增密。
通过 `python train_mask_v3.py --help` 查看容差、权重和调度。所有默认几何权重只是起始配置，尚未通过云端实验调优。

## 固定相机和输入安全

- 直接读取显式 txt，不回退 bin。验证 train 与 val/test 的图像名字互斥。
- val 必须正好十个唯一相机，按给定文件顺序导出；不重新随机挑选。
- 若输入 images.txt 为全量，运行 `prepare_v3.py` 从中排除给定 val/test，写入一个新的训练列表；不修改原始文件。不提供固定 val 时不会自动造一组替代。
- 几何参考由当前 LAS/PLY 一次性建立：固定点 ID、局部平面、法向、间距、置信度。法向参考通过附近训练相机投票定向；投票不代表完成可见性判断。
- LAS scale 不足以证明物理单位，默认 `--units scene`。确认以米为单位后改用 `--units meters`；该参数只声明单位，不缩放数据。
- 现有主基线使用居中相机投影，因此 v3 拒绝明显非居中主点和未去畸变相机。
- 不执行旧 preprocess.py 的删除步骤，也不使用旧 notebook 的清理命令。所有运行和恢复必须指定新输出目录。

## 几何定义

局部第三轴固定定义为厚度轴，法向为其旋转方向，不随最小尺度轴切换。
surface 是固定局部平面距离的 Huber；tangent 是超出切向范围的惩罚；normal 为 `1-dot²`（不区分轴正负）；flatten/size 仅惩罚超过间距比例上限的尺度。surface/normal 使用平面置信度；尺寸限制覆盖低置信度点。关闭某项不会留下硬复位。

原主基线四元数转换有漏赋值分支，v3 绕过该函数，以覆盖所有旋转分支的 SciPy 转换初始化。原文件不修改。

剪枝在预热后，依据持续低 opacity、每个无重复采样 epoch 中的不同投影视角数决定，并限制单次比例。`radii>0` 只是视锥候选，不是遮挡或像素贡献真值。剪枝会同步几何引用、点 ID 和优化器动量；不删除磁盘文件。

## 每5000步的输出

默认在初始化、每5000步及最后一步输出十个固定视角，每个包括：

- `valXX_rgb.png`：RGB。
- `valXX_normal.png`：朝当前相机的相机系法向，RGB=(normal+1)/2，无效像素黑色。
- `valXX_depth.png`：固定显示范围的灰度深度，默认0–15场景单位；不逐帧自动拉伸。
- 默认不生成 `geometry.npz`；只有显式设置 `--val_npz on` 才保存原始几何数组。
- `manifest.json`：相机名、R/T、显示范围、有效区PSNR及输出语义。

深度使用额外属性通道计算 `sum(alpha*T*z_center)/sum(alpha*T)`，不是原 renderer 的未归一化逆深度，也不是无偏射线—表面深度。它只用于诊断，不加入 A/B loss。法向在合成前按视角定向，合成后归一化并过滤无效像素。颜色曝光变换不应用于几何通道。

`train_log.jsonl` 每100步记录各 loss 开关状态/原始值/实际权重/加权值。
`geometry_log.jsonl` 记录离面、偏移、角度、尺度分位数与超限比例。
`prune_log.jsonl` 记录剪枝事件。尺寸是高斯标准差，不是查看器椭球直径。

## 恢复和版本

每50000步及结束保存完整 checkpoint 和 Gaussian PLY。恢复命令保持全部原参数，仅增加 `--resume <checkpoint>` 并使用新的 `-m`。不允许把旧模型 PLY 当作完整恢复。

checkpoint 包括模型、优化器、曝光状态、几何参考、剪枝计数、相机采样栈、随机状态、输入身份及 v3 源码哈希。严格检查源码和配置一致；核心输入、训练与val图像及mask均计算内容SHA256，重新解压后的mtime变化不影响数据校验。

发生非有限参数/梯度时停止并写 failure.json，不通过删坏点掩盖错误；使用最后完整 checkpoint 恢复。每个实验一个独立目录，所有原文件保留。

## 本地验证与云端边界

```bash
python -m unittest -v test_geometry_v3
```

CPU测试覆盖旋转、几何梯度、开关、相机定向和剪枝引用。Colab notebook 包含环境检查、主基线扩展构建、测试、200步冒烟和正式A/B入口。CPU测试通过不代表CUDA扩展编译、完整训练和恢复已经在Colab验证；请先运行冒烟单元。安装使用Colab自带torch，并记录实际版本，不盲目安装旧environment.yml。
# 正式 A/B 输出补充（2026-09-10）

`--val_ellipsoids on|off` 独立控制每次固定验证输出的 `valXX_ellipsoid.png`，默认开启。
显示固定1σ、真实三轴尺度的实体椭球，DC颜色加方向光，opacity过滤阈值0.05。
这是诊断可视化，不改变训练或模型参数；需要CuPy CUDA，避免Colab EGL落到CPU软件渲染。

`loss_log.csv` 每步记录RGB L1/DSSIM、总损失、五项几何loss的raw/weight/weighted、点数和累计耗时，每个log_interval刷新。
`val_metrics.csv` 每个验证时刻包含十个相机及MEAN行，记录有效像素PSNR/MAE；`ssim_zero_mask_full_image`是置零mask后的整图SSIM。
恢复运行仍使用新目录，CSV只包含续跑段；不要将恢复段当作从第一步开始的完整日志。

完整D流程 notebook：`colab/ZS601_D_v3_150k.ipynb`；代码固定提交，先200步冒烟，再运行D 150000步，输入先复制到`/content`。

## 云端存储策略

默认 `--val_npz off`：验证仅保存 PNG 和 CSV，不生成 geometry.npz。`--checkpoint_interval 50000`：正式训练在50000、100000、150000步保存checkpoint及PLY，最终步总会保存（包括200步冒烟）。初始PLY和reference_v3.npz用于初始化记录，仍保留。已有产物不删除。运行中的旧进程不会自动加载新版源码，需单独完成可验证的切换。

## D 组：贴面受控增密

`--experiment D` 继承 C 的全部功能，并独立开启 `--surface_densify on`。标准 `densify_and_prune` 保持关闭。

D 在 10000–100000 步、每 10000 步检查一次候选。候选必须同时满足：屏幕空间位置梯度达到阈值、opacity 达标、被足够多相机视锥覆盖、LiDAR 平面置信度有效、中心位于局部平面容差内且未超过原锚点的切向范围。每次最多新增当前点数的 1%，总点数最多为初始点数的 1.25 倍。

子高斯中心沿固定 LiDAR 平面的切向产生，再投影回该平面；它继承父高斯外观、旋转和 LiDAR 引用，尺度缩小到父高斯的 0.7 倍并立即受 C 的 XY/厚度上限约束。父子按 `1-(1-q)^2=p` 分配原父高斯的 alpha 质量并清理父 opacity 的 Adam 动量，避免每次复制直接增加约 50% 局部不透明度。候选使用最近一个完整无重复采样 epoch 的视角数，每个原始 LiDAR seed 默认最多生成2个子高斯。新增时同步优化器状态、几何 reference、剪枝计数和 checkpoint 状态。`surface_densify_log.jsonl` 记录每次候选数、新增数、增长预算和平均梯度。

各项参数均可单独覆盖，例如 `--experiment D --surface_densify off` 应退化到 C。D 的关键新增参数为：

- `--surface_densify_grad_threshold 0.0002`
- `--surface_densify_plane_ratio 0.25`
- `--surface_densify_offset_ratio 0.35`
- `--surface_densify_child_scale 0.7`
- `--surface_densify_max_fraction 0.01`
- `--surface_densify_max_points_ratio 1.25`
- `--surface_densify_max_children_per_seed 2`

## 15万步正式实验验收规则

一次正式实验只有在150000步模型、完整test评估和总结文档全部写入输出目录后才标记完成。

- `--final_test on` 默认开启；150000步正式运行必须提供非空的 `--test_file`。
- 完整test集逐相机计算 `masked_psnr`、`masked_mae` 和 `ssim_zero_mask_full_image`，保存到 `test_final/iteration_150000/test_metrics.csv`；CSV同时包含MEAN、MEDIAN、MIN和MAX汇总行。
- 最差相机按 `masked_psnr` 升序排序，同分按 `image_name`，取前10张。
- 只对这10张保存 RGB、1σ彩色椭球、深度和法向量PNG。test不生成NPZ，也不为全部test相机保存重型诊断图。
- `test_summary.json` 保存指标定义、汇总值、最差10张及对应文件名。
- `experiment_summary.md` 自动记录相对baseline A的功能变化、关键超参数、相对原始默认值的变化、最终test结果、最差10张、最终val与几何日志。给出 `--baseline_result <baseline输出目录>` 时，还会计算同协议test均值差。
- `completed.json` 中的 `final_test_complete` 必须为true，正式实验才通过验收。评估或总结失败会写入 `failure.json`，不会提前标记完成。
- 200步冒烟显式使用 `--final_test off`。

指标协议固定：PSNR和MAE按有效mask像素数归一化；SSIM为置零mask后的整图SSIM。不同test列表、mask协议或旧指标不得直接作数值对比。
