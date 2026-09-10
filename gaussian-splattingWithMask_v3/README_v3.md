# LiDAR A/B v3

基于 main `d45646bf3944599d0470e72eca7365dcd420191d` 的 `gaussian-splattingWithMask`。
这是独立的 `gaussian-splattingWithMask_v3` 目录，包含 v3 训练入口及其实际依赖。`v3-ab` 分支根目录只保留本文件夹；main/v2 分支历史不变。原主基线工作目录在本地保留。

[打开 Colab](https://colab.research.google.com/github/VISjudy/ZS601_3DGS/blob/v3-ab/gaussian-splattingWithMask_v3/colab/ZS601_AB_v3.ipynb)

进入本目录后执行以下命令。CUDA源码、GLM头文件和许可证随目录保留；不包含旧训练入口、旧notebook、GLM文档/测试和实验产物。不要使用v2-dev扩展。

## 运行

```bash
python train_mask_v3.py --experiment A \
  -s /content/dataset -m /content/drive/MyDrive/results/A_unique \
  --point_cloud /content/dataset/ZS601_3cm_sample.las \
  --cameras_file /content/dataset/sparse/cameras.txt \
  --train_file /content/work/train_v3.txt \
  --val_file /content/dataset/sparse/images-val10.txt \
  --test_file /content/dataset/sparse/images_test.txt \
  --iterations 30000 --position_lr_max_steps 30000
```

将 `--experiment A` 改成 `B` 开启五项几何约束。例如 `--experiment B --normal_loss off` 只关闭法向 loss。每个功能只有 `on/off` 一种覆盖表达；不接受旧 `--init_2d` / `--freeze_2d_z` 等混合控制，原入口仍支持原参数。

| 功能参数 | A | B |
|---|---|---|
| init_normal / init_flatten / orient_cameras / pruning | on | on |
| surface_loss / tangent_loss / normal_loss / flatten_loss / size_loss | off | on |

增密、深度训练、opacity reset、硬厚度复位在这个 A/B 版本中固定关闭，不暴露无效开关。
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

## 每1000步的输出

默认在初始化、每1000步及最后一步输出十个固定视角，每个包括：

- `valXX_rgb.png`：RGB。
- `valXX_normal.png`：朝当前相机的相机系法向，RGB=(normal+1)/2，无效像素黑色。
- `valXX_depth.png`：固定显示范围的灰度深度，默认0–15场景单位；不逐帧自动拉伸。
- `valXX_geometry.npz`：float深度、法向、透明度和有效 mask。
- `manifest.json`：相机名、R/T、显示范围、有效区PSNR及输出语义。

深度使用额外属性通道计算 `sum(alpha*T*z_center)/sum(alpha*T)`，不是原 renderer 的未归一化逆深度，也不是无偏射线—表面深度。它只用于诊断，不加入 A/B loss。法向在合成前按视角定向，合成后归一化并过滤无效像素。颜色曝光变换不应用于几何通道。

`train_log.jsonl` 每100步记录各 loss 开关状态/原始值/实际权重/加权值。
`geometry_log.jsonl` 记录离面、偏移、角度、尺度分位数与超限比例。
`prune_log.jsonl` 记录剪枝事件。尺寸是高斯标准差，不是查看器椭球直径。

## 恢复和版本

每1000步及结束保存完整 checkpoint 和 Gaussian PLY。恢复命令保持全部原参数，仅增加 `--resume <checkpoint>` 并使用新的 `-m`。不允许把旧模型 PLY 当作完整恢复。

checkpoint 包括模型、优化器、曝光状态、几何参考、剪枝计数、相机采样栈、随机状态、输入身份及 v3 源码哈希。严格检查源码和配置一致；核心输入、训练与val图像及mask均计算内容SHA256，重新解压后的mtime变化不影响数据校验。

发生非有限参数/梯度时停止并写 failure.json，不通过删坏点掩盖错误；使用最后完整 checkpoint 恢复。每个实验一个独立目录，所有原文件保留。

## 本地验证与云端边界

```bash
python -m unittest -v test_geometry_v3
```

CPU测试覆盖旋转、几何梯度、开关、相机定向和剪枝引用。Colab notebook 包含环境检查、主基线扩展构建、测试、200步冒烟和正式A/B入口。CPU测试通过不代表CUDA扩展编译、完整训练和恢复已经在Colab验证；请先运行冒烟单元。安装使用Colab自带torch，并记录实际版本，不盲目安装旧environment.yml。
