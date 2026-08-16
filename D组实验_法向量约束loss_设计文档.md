# D 组实验（法向量约束 loss）代码修改逻辑设计文档

> 适用目录：`gaussian-splattingWithMask_v2/`（阶段 2 开发目录；v1 目录保持冻结）
> 本文档为 D 组实验的代码修改逻辑设计说明，逐文件、逐函数梳理修改点，供审阅与报告撰写使用。
> 说明：下述全部设计已在 v2 目录完整落地，本文档同时作为该实现的逐点说明与审阅依据。

---

## 0. 实验定义与控制口径

### 0.1 D 组定位

| 实验变量 | A 组 (baseline) | B 组 | C 组 | **D 组** |
|---|---|---|---|---|
| `--init_2d`（2D 椭球初始化） | 关 | 开 | 开 | **关（baseline 默认 3DGS 初始化）** |
| `--freeze_2d_z`（训练中 z 轴形状冻结） | 关 | 开 | 关 | **关** |
| `--lambda_scale`（z 轴厚度正则） | 0 | 0 | 0 | **0.1** |
| `--lambda_normal`（深度-法向一致性） | 0 | 0 | 0 | **0.2** |
| `--lambda_size`（形状约束，防大椭球，可选） | 0 | 0 | 0 | 默认 0，需要时显式开启（建议 0.01 起调） |

D 组只修改 loss，参考 2DGS 增加两项：

1. **scale 正则**（压薄 z 轴厚度）：
   `loss_scale = mean(|exp(scaling)[:, 2:3]|)`，权重 `--lambda_scale`（默认 0.1，可调）
2. **深度-法向一致性**：渲染法向图 vs 深度导出法向图，
   `loss_normal = mean(1 - cos(n_rendered, n_depth))`，权重 `--lambda_normal`（默认 0.1，可调）；
   对齐 2DGS 原版延迟启用：`--normal_start_iter`（默认 7000）轮后才开启；不依赖外部 GT 法向量图。
3. **形状约束（可选，防大椭球）**：逐高斯惩罚最长尺度轴（exp 后真实尺度）的均值，
   `loss_size = mean(max(exp(scaling), dim=1))`，权重 `--lambda_size`（默认 0 关闭）。

**表面贴合机制说明**：法向一致性 loss 会驱动每个高斯的第 3 轴转向表面法向并压扁（倾斜/肥厚高斯的合成法向会与深度法向不一致），
配合 scale 正则使高斯趋向"贴合表面的扁平圆盘"（2DGS 机制移植）。注意 D 组 baseline 初始化下第 3 轴初始并非最小尺度轴，
需训练过程逐步对齐，收敛比 B 组（init_2d 预对齐）慢。前两项只约束法向轴厚度，不限制切向铺展，
故另设 `--lambda_size` 防止"又大又扁的煎饼状椭球"（常见于墙面/地面弱纹理区域）。

### 0.2 控制口径（单一变量原则）

- 两项权重为 0 时对应项**完全跳过**（不计算、不打印、csv 记录 0）。
- 由于 v2 中 `lambda_scale/lambda_normal` 默认非 0，**重跑 A/B/C 组必须显式传 `--lambda_scale 0 --lambda_normal 0`**。
- `lambda_size` 默认 0，仅需要防大椭球时显式开启，不影响既有组口径。

---

## 第一部分 2-A：光栅化器输出法向量图（submodules/diff-gaussian-rasterization）

### 1.1 每高斯法向的定义（数学口径）

- 法向 = 旋转矩阵第 3 列（即最小尺度轴方向）。设四元数分量 `r=q.x, x=q.y, y=q.z, z=q.w`，则：

  n = ( 2(xz + ry), 2(yz - rx), 1 - 2(x² + y²) )

- **朝向相机翻转**：若 `n · (p_orig - cam_pos) > 0`（背向相机）则取反，保证渲染法向恒朝向相机。
  - 注意：设计目标文字"点积为负则翻转"的准确表述应为"**点积为正（背向相机）才翻转**"——法向朝向相机要求 `n·(p-c) < 0`，两者语义目标一致。
- 法向只依赖四元数，**与 scale 无关**（反向不产生 scale 梯度，这是与协方差路径的关键区别）。

### 1.2 cuda_rasterizer/forward.cu（前向）

**`preprocessCUDA`（约 L269-290）**

- 签名新增输出参数 `float* normals`（大小 3*P）。
- 每高斯按 1.1 公式直接由四元数计算旋转矩阵第 3 列（不构造完整旋转矩阵，仅 6 次乘加）；
- 与 `(p_orig - cam_pos)` 点积 > 0 时翻转三个分量，写入 `normals[idx*3 + 0/1/2]`。

**`renderCUDA`（约 L349, L406-412, L434-440）**

- 签名新增输入 `normals` / 输出 `out_normal`；
- 与 invdepth 完全同构：每像素累计 `expected_normal[3] += n * alpha * T`（alpha 加权、透射率 T 同 depth 通道）；
- 写回时**背景为 0、不混 bg_color**（`out_normal[ch*H*W + pix_id] = expected_normal[ch]`），避免背景色污染法向方向；无贡献像素保持初始化的全 0。

### 1.3 cuda_rasterizer/rasterizer_impl.h / rasterizer_impl.cu（管线与缓冲）

- `GeometryState` 新增成员 `float* normals`（rasterizer_impl.h L41）；
- 几何缓冲分配：`obtain(chunk, geom.normals, P * 3, 128)`（rasterizer_impl.cu L166）——每高斯法向存入 geomBuffer，反向时随 buffer 原样恢复指针，**无需重算**；
- `forward()`：签名新增 `float* out_normal`，串联传入 preprocessCUDA 与 render；
- `backward()`：签名新增 `dL_dout_normals`（法向图像素梯度输入）与 `dL_dnormals`（每高斯法向梯度输出），并把 `geomState.normals` 传入反向 render 与 preprocess 反向。

### 1.4 cuda_rasterizer/backward.cu（反向，两条梯度路径）

**路径 A：像素 → 每高斯法向（反向 `renderCUDA`，与 invdepth 同构，约 L544-661）**

- 共享内存 `collected_normals[3 * BLOCK_SIZE]` 倒序收集每高斯法向；
- 每像素读取 `dL_dnormal_pix[3]`，维护 `accum_normal_rec[3]` / `last_normal[3]` 递推量；
- 逐高斯：
  - `dL_dalpha += (n_c - accum_rec) * dL_dnormal_pix`（法向对 alpha 的贡献并入既有 alpha 梯度链，与颜色/深度项相加）；
  - `atomicAdd(dL_dnormals[global_id*3 + ch], alpha*T * dL_dnormal_pix[ch])`（每高斯法向梯度，跨像素原子累加）。

**路径 B：每高斯法向 → 四元数（preprocess 反向，解析求导，约 L452-475）**

- 直接使用 geom 缓冲中**已翻转**的法向值推导梯度：前向翻转记号 s = ±1 在存储值与梯度上同时作用（s² = 1）相消，无需重存翻转标志；
- 法向对四元数四分量的解析偏导：
  - `dn/dr = (2y, -2x, 0)`
  - `dn/dx = (2z, 0, -4x)`
  - `dn/dy = (0, 2z, -4y)`
  - `dn/dz = (2x, 2y, 0)`
- 以 `atomicAdd` **累加**到 `dL_drot`（与协方差路径的旋转梯度叠加，不覆盖）；法向梯度不进入 scales。

### 1.5 头文件与绑定层（签名同步）

| 文件 | 修改点 |
|---|---|
| `cuda_rasterizer/forward.h` | preprocess / render 声明加 normals / out_normal 参数 |
| `cuda_rasterizer/backward.h` | backward 声明加 dL_dout_normals / dL_dnormals 参数 |
| `cuda_rasterizer/rasterizer.h` | Rasterizer::forward / backward 声明同步 |
| `rasterize_points.cu` | 见下 |
| `rasterize_points.h` | 两个 CUDA 入口函数声明同步 |
| `ext.cpp` | 无需改绑定名（仍是 `rasterize_gaussians` / `rasterize_gaussians_backward`） |

**`rasterize_points.cu` 细节**

- 前向（L76-128）：分配 `out_normal = zeros(3, H, W)`，返回值改为 8 元组 `(rendered, out_color, radii, geomBuffer, binningBuffer, imgBuffer, out_invdepth, out_normal)`；
- 反向（L131-243）：入参新增 `dL_dout_normal`；**仅当 `dL_dout_normal.size(0) != 0`（法向图参与 loss）才分配 `dL_dnormals(P, 3)`**，否则传 nullptr、路径 A/B 均被指针判空跳过（零开销兜底）；返回 9 元组（末尾加 `dL_dnormals`）。

### 1.6 diff_gaussian_rasterization/__init__.py（Python 绑定）

- `_RasterizeGaussians.forward`：解包 C++ 8 元组，`return color, radii, invdepths, normal`（**4 元组**）；
- `_RasterizeGaussians.backward(ctx, grad_out_color, _, grad_out_depth, grad_out_normal)`：`grad_out_normal` 传入 C++；解包 9 个梯度；法向图梯度已并入 `grad_rotations`（`grad_normals` 是每高斯中间量，不对应任何输入张量，无需作为输入梯度返回）。

### 1.7 gaussian_renderer/__init__.py

- 两个 `separate_sh` 分支均按 4 元组解包 `rendered_image, radii, depth_image, normal_image = rasterizer(...)`；
- 返回字典新增 `"normal": normal_image`——**不做 clamp**（分量理论范围 [-1,1]，供 loss 与可视化使用），区别于 `"render"` 的 clamp(0,1)。

---

## 第二部分 2-B：法向量 loss 融入训练（train_mask.py + gaussian_model.py + arguments）

### 2.1 arguments/__init__.py（ModelParams，L77-85）

- 新增 `self.lambda_scale = 0.1`：z 轴厚度正则权重；
- 新增 `self.lambda_normal = 0.1`：深度-法向一致性 loss 权重；
- 新增 `self.normal_start_iter = 7000`：法向 loss 延迟启用轮次（对齐 2DGS：前期深度不可靠，过早启用向旋转注入噪声梯度）；
- 新增 `self.lambda_size = 0.0`：形状约束（防大椭球）权重，默认关闭；
- 为 0 时对应项跳过；注释注明重跑 A/B 组需显式传 0。

### 2.2 scene/gaussian_model.py

- 新增 `get_normal` property（L178-181）：`build_rotation(self.get_rotation)[:, :, 2]`——每高斯几何法向（旋转矩阵第 3 列），**不做朝向翻转**（相机无关），供统计/可视化导出；渲染法向图的翻转在光栅化器内部完成；
- scale 正则**不需要**模型改动：直接用既有 `get_scaling[:, 2:3]`（已 exp），梯度经 exp 自然回传 `_scaling`。

### 2.3 train_mask.py

**深度导出法向 `normal_from_meddepth`（对齐 2DGS，无外部 GT）**

深度靶标为光栅化器新输出的 **median depth**（`med_depth` 通道：逐像素累计 alpha 首次越过 0.5 时贡献高斯的深度，
比 alpha 加权期望逆深度锐利得多；背景/无贡献像素为 0；反向不传梯度，仅作靶标）：

1. 深度 = `med_depth[0]`（已是深度单位，无需取倒数）；
2. 由 FoV 推内参：`fx = W / (2 tan(FoVx/2))`、`fy = H / (2 tan(FoVy/2))`、`cx=(W-1)/2`、`cy=(H-1)/2`；
3. 相机系反投影：`P = ((x-cx)/fx * d, (y-cy)/fy * d, d)`；
4. 中心差分 `dPdx = (P[:, 2:] - P[:, :-2]) / 2`、`dPdy = (P[2:, :] - P[:-2, :]) / 2`（内部像素 H-2 × W-2）；
5. `n = cross(dPdy, dPdx)`（叉积顺序保证朝向相机，与光栅化器口径一致）→ normalize，返回 `3 x (H-2) x (W-2)`。

**法向一致性 loss（仅 `lambda_normal > 0` 且 `iteration > normal_start_iter`，默认 7000）**

```python
n_rendered = render_pkg["normal"][:, 1:-1, 1:-1]          # 裁边对齐，3 x (H-2) x (W-2)
n_depth = normal_from_meddepth(render_pkg["med_depth"].detach(), viewpoint_cam)  # detach：median 选取不可微
cos = sum(n_rendered * n_depth, dim=0) / (|n_rendered| * |n_depth| + 1e-6)
valid = (med_depth > 1e-4) & (|n_rendered| > 0.5) & (alpha_mask > 0.5)  # 无效/弱/mask 外像素剔除
loss_normal = mean(1 - cos[valid])                        # 对齐 2DGS：不加绝对值（两侧法向均已朝向相机）
loss += lambda_normal * loss_normal
```

- **延迟启用（对齐 2DGS 原版，重要）**：训练前期深度不可靠、深度导出法向是噪声，过早启用会向旋转注入噪声梯度
  （实测：第 0 轮启用 λ=0.2 时 normal_loss 停滞在随机水平 ~0.42，5000 轮法向图呈彩虹噪点、渲染质量被抑制）；
  2DGS 官方为 7000 轮（COLMAP 稀疏初始化）；本项目激光点云初始化几何已准，正式训练建议传 1000、冒烟传 0；
- **弱像素过滤**：合成法向模长 < 0.5（累计不透明度低/近背景）的像素 cos 是纯噪声，剔除；
- 有效像素为 0 时整项跳过，避免空张量 mean 出 NaN；
- mask 语义遵循项目约定（加载时已取反，1 = 有效保留区域）。

**scale 正则（L224-230，仅 `lambda_scale > 0`）**

```python
loss_scale = mean(|gaussians.get_scaling[:, 2:3]|)        # exp 后的 z 轴厚度
loss += lambda_scale * loss_scale
```

**形状约束防大椭球（scale 正则之后，仅 `lambda_size > 0`）**

```python
loss_size = gaussians.get_scaling.max(dim=1, keepdim=True)[0].mean()  # 逐高斯最长轴（exp 后真实尺度）
loss += lambda_size * loss_size
```

- 与 `lambda_scale` 互补：后者只压法向轴厚度，此项限制切向铺展，直接针对"大椭球"；
- 梯度只作用于当前最长轴，轴间长短变化时自动切换目标轴；
- 致密化会分裂/克隆补偿覆盖，表现为高斯数变多、单个变小；权重过大可能导致点数暴涨/欠拟合，建议从 0.01 起调，观察 `val_metrics.csv` 的高斯数与 PSNR 变化；
- 尺度为世界单位，不同数据集场景尺度不同时需重新调参。

**日志与监控**

| 项 | 实现 |
|---|---|
| csv 表头 | 8 列：`iteration,L1,SSIM_loss,depth_loss,normal_loss,scale_reg,size_reg,total`，每 10 轮追加 |
| 每 10 轮 print | `[loss] iter {}: Normal Loss ... / Scale Reg ... / Size Reg ... / total ...`（仅有权重开启时打印） |
| 启动确认 | `[法向约束配置] lambda_scale=... / lambda_normal=... / lambda_size=...`，确认实验变量生效 |
| 曲线图 | `plot_loss_curves` 兼容 5/7/8 列 csv（按列数自适应） |
| val 可视化 | `render_val_cameras` 直接用光栅化器 normal 通道输出方向颜色贴图 `RGB=(n+1)/2`（L390-395），与训练 loss 同口径、无需额外渲染一次 |

---

## 第三部分 编译验证与训练执行（Colab）

1. **重新编译光栅化器**（CUDA 改动必须重编，否则 Python 侧 4 元组解包直接报错）：

   ```bash
   cd submodules/diff-gaussian-rasterization
   pip install .
   ```

2. **导入验证**：

   ```bash
   python -c "from diff_gaussian_rasterization import GaussianRasterizer; print('ok')"
   ```

3. **冒烟短训**（少量迭代）：确认 loss 有限不发散、`loss_log.csv` 为 8 列、`val_render/iter_*/val*_normal.png` 生成且法向分布合理。

4. **D 组正式训练**：baseline 初始化命令不变（不传 `--init_2d` / `--freeze_2d_z`），lambda 用默认值或显式 `--lambda_scale 0.1 --lambda_normal 0.2`，输出到独立目录（如 `output/D`）：

   ```bash
   python train_mask.py -s <数据路径> -m output/D --val_file <images-val10.txt> --lazy_load
   ```

5. **对照**：A/B/C 组重跑需补 `--lambda_scale 0 --lambda_normal 0`；评估口径与既有实验一致（图像指标 PSNR/L1/SSIM + 几何指标高斯点与激光点云 cloud2cloud 距离）。

---

## 验证要点（审查清单）

- [x] 翻转条件为 `dot > 0 才翻转`（朝向相机），与设计目标一致；
- [x] 法向反向梯度只进 rotations（atomicAdd 叠加，不覆盖协方差梯度），不进 scales；
- [x] `dL_dout_normal` 为空张量时反向零开销（指针判空跳过）；
- [x] normal loss 剔除 mask 外像素（`alpha_mask > 0.5`）、无效深度（`> 1e-4`）与弱像素（法向模长 ≤ 0.5）；
- [x] 法向 loss 延迟启用（`normal_start_iter` 默认 7000，对齐 2DGS），权重为 0 时完全不计算、不打印（含 `lambda_size`）；
- [x] 法向图背景为 0（不混 bg_color），normal 通道不 clamp。

## 相关文件索引

| 文件 | 角色 |
|---|---|
| `submodules/diff-gaussian-rasterization/cuda_rasterizer/forward.cu` | 每高斯法向计算 + 法向图合成（前向） |
| `submodules/diff-gaussian-rasterization/cuda_rasterizer/backward.cu` | 法向图反向（路径 A/B） |
| `submodules/diff-gaussian-rasterization/cuda_rasterizer/rasterizer_impl.h/.cu` | geom 缓冲 normals、前后向串联 |
| `submodules/diff-gaussian-rasterization/cuda_rasterizer/forward.h / backward.h / rasterizer.h` | 声明同步 |
| `submodules/diff-gaussian-rasterization/rasterize_points.cu/.h` | C++ 入口：8 元组前向 / 9 元组反向 |
| `submodules/diff-gaussian-rasterization/diff_gaussian_rasterization/__init__.py` | Python 绑定：4 元组 forward / backward |
| `gaussian_renderer/__init__.py` | render() 返回字典新增 "normal" |
| `arguments/__init__.py` | `--lambda_scale` / `--lambda_normal` / `--lambda_size` |
| `scene/gaussian_model.py` | `get_normal` property |
| `train_mask.py` | `normal_from_invdepth`、三项 loss（normal/scale/size）、日志/打印/曲线/val 法向图 |
