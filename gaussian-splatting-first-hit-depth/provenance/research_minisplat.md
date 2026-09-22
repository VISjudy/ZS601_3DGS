# Mini-Splatting / Mini-Splatting2 深度公式核查

核查日期：2026-09-22  
范围：作者论文与作者官方 GitHub；只读研究，不运行训练。  
固定源码版本：Mini-Splatting2 `5ec9202c2db7d900728bfc09733e85467cff885e`；Mini-Splatting `c0d55811930dec3bfd3d61cad927093d166aaf82`。

## 结论

1. **Mini-Splatting2 原版不是“首个贡献者深度”。**论文定义每像素选择
   \(i_{max}=\arg\max_i w_i\)，其中 \(w_i=T_i\alpha_i\mathcal G_i^{2D}(x)\)，即最大 alpha-blending 合成贡献；随后取该高斯沿射线的椭球交点中点。v1 的公式 (2) 明确如此；v2 延续同一定义。
2. **深度重初始化来自 Mini-Splatting1。**Mini-Splatting2 v1 明说保留 Mini-Splatting 的 depth reinitialization，并在 2K iteration 执行；Mini-Splatting1 论文和官方代码已经具有相同的最大贡献选择、射线峰值点与重初始化流程。
3. **固定椭球的进/出交点中点，严格等于高斯沿射线的最大密度位置。**对任意对称正定各向异性协方差都成立，不只对初始各向同性高斯成立。椭球等值面半径只影响“是否相交”和两个根的间距，不影响中点。
4. **发布 CUDA 实际没有要求有限 3σ 椭球真相交。**代码以 `scales * 3.0f` 构造 3σ 椭球并计算判别式，但没有检查 `discriminant >= 0`；只要峰值在相机前方，就可参与最大 `alpha*T` 选择。因此发布实现的有效语义是“所选高斯沿射线的最大密度点”，判别式只是输出诊断。
5. **“同一首击 ID：中心 Z vs 峰值/中点 Z”是合理的受控消融，但不是论文原版。**它固定选择变量，只比较取点公式。若同时改成最大贡献 ID、椭球 entry 排序或 `Δ>=0` 过滤，会混入选择、覆盖或有效性变化。
6. **scale 0.1 不会单调优于 0.5。**对同一高斯、同一射线、各轴统一缩放，\(t_{mid}=t_{peak}\) 完全不变。scale 会改变 2D footprint、alpha、有效覆盖、所选 Gaussian ID、训练动态，以及有限椭球的相交判定。0.1 更容易产生空洞和 ID 不稳定；只有在这些变量被明确控制时才能解释结果。

## 原始证据

| 问题 | 论文/源码事实 | 精确来源 |
|---|---|---|
| MSv2 深度公式 | v1 Eq. (2)：\(d(x)=d^{mid}_{i_{max}}(x)\)，\(i_{max}=\arg\max_i w_i\) | [Mini-Splatting2 v1 §3, Eq. 2](https://arxiv.org/html/2411.12788v1#S3) |
| MSv2 是否继承 MS1 | v1 写明保留 depth reinitialization；2K 时重初始化，采样点数等于当前高斯数 | [Mini-Splatting2 v1 §4](https://arxiv.org/html/2411.12788v1#S4) |
| MS1 的原始定义 | 椭球、两交点中点、最大密度点数值相同、最大贡献高斯选择均已在 MS1 出现 | [Mini-Splatting §4.1](https://arxiv.org/html/2403.14166#S4.SS1) |
| 当前论文版本 | 同一 arXiv ID 的 v2（2026-02-05）已改题为 *Efficient Scene Modeling via Structure-Aware and Region-Prioritized 3D Gaussians*，但仍把该框架称为 Mini-Splatting2 | [arXiv 版本记录](https://arxiv.org/abs/2411.12788) |
| v2 的中点定义 | 椭球 \(g=1\)，射线 \(o+td\)，取两交点中点；论文称其与 ray maximum-density 数值相同 | [v2 §IV-A](https://arxiv.org/html/2411.12788v2#S4.SS1) |
| v2 的高斯选择 | 多高斯时选择 \(\arg\max_i w_i\)，不是 first contributor、最近 entry、最大 opacity 或 median | [v2 §IV-A](https://arxiv.org/html/2411.12788v2#S4.SS1) |
| 论文 center vs mid | Table IV：Center 与 Mid 的 NVS 指标几乎相同；作者称 Mid 的稠密点云更好 | [v2 Table IV / §VI-E](https://arxiv.org/html/2411.12788v2#S6.SS5) |
| CUDA 排序 | tile key 的低 32 位是 `depths[idx]`；预处理把它设成 Gaussian center 的 `p_view.z`，所以 contributor 顺序按中心相机 Z，而非椭球 entry | [`rasterizer_impl.cu` L119-L188](https://github.com/fatPeter/mini-splatting2/blob/5ec9202c2db7d900728bfc09733e85467cff885e/submodules/diff-gaussian-rasterization_ms/cuda_rasterizer/rasterizer_impl.cu#L119-L188), [`forward.cu` L206-L277](https://github.com/fatPeter/mini-splatting2/blob/5ec9202c2db7d900728bfc09733e85467cff885e/submodules/diff-gaussian-rasterization_ms/cuda_rasterizer/forward.cu#L206-L277) |
| CUDA 贡献阈值 | `alpha=min(0.99, opacity*exp(power))`；`alpha<1/255` 跳过；若更新后的 `T<1e-4` 则终止且当前项不写入 | [`forward.cu` L740-L758](https://github.com/fatPeter/mini-splatting2/blob/5ec9202c2db7d900728bfc09733e85467cff885e/submodules/diff-gaussian-rasterization_ms/cuda_rasterizer/forward.cu#L740-L758) |
| CUDA 椭球尺度 | 二次式用 `scales * 3.0f`，所以源码的诊断等值面是 3σ 椭球，不是论文文字中的 1σ `g=1` | [`forward.cu` L763-L799](https://github.com/fatPeter/mini-splatting2/blob/5ec9202c2db7d900728bfc09733e85467cff885e/submodules/diff-gaussian-rasterization_ms/cuda_rasterizer/forward.cu#L763-L799) |
| CUDA 实际取点 | 计算 `discriminant` 后未检查其符号；直接用 `-b/(2*a)` 得到峰值点，只检查正深度 | [`forward.cu` L789-L817](https://github.com/fatPeter/mini-splatting2/blob/5ec9202c2db7d900728bfc09733e85467cff885e/submodules/diff-gaussian-rasterization_ms/cuda_rasterizer/forward.cu#L789-L817) |
| CUDA 选 ID | 比较 `weight_max < alpha*T`，记录最大贡献 ID、点和判别式 | [`forward.cu` L810-L843](https://github.com/fatPeter/mini-splatting2/blob/5ec9202c2db7d900728bfc09733e85467cff885e/submodules/diff-gaussian-rasterization_ms/cuda_rasterizer/forward.cu#L810-L843) |
| 实际消费的数据 | Python 重初始化消费 `out_pts` 世界点；标量 `rendered_depth` 没有用于该流程 | [`gaussian_model.py` L561-L599](https://github.com/fatPeter/mini-splatting2/blob/5ec9202c2db7d900728bfc09733e85467cff885e/scene/gaussian_model.py#L561-L599), [`__init__.py` L263-L271](https://github.com/fatPeter/mini-splatting2/blob/5ec9202c2db7d900728bfc09733e85467cff885e/submodules/diff-gaussian-rasterization_ms/diff_gaussian_rasterization_ms/__init__.py#L263-L271) |
| MS1 源码同源 | MS1 同样使用 3σ 二次式、`-b/(2*a)`、最大 `alpha*T`，且未以判别式过滤 | [`Mini-Splatting forward.cu` L583-L621](https://github.com/fatPeter/mini-splatting/blob/c0d55811930dec3bfd3d61cad927093d166aaf82/submodules/diff-gaussian-rasterization_ms/cuda_rasterizer/forward.cu#L583-L621) |

注意：论文中的 `w_i` 包含 2D Gaussian 值；CUDA 已把该值并入 `alpha`，所以代码比较的 `alpha*T` 与论文的合成权重一致。这里的“最大贡献”也不是 Mini-Splatting2 critical-Gaussian 的 0.99 quantile 筛选；quantile 属于后续 critical set 构造，不是逐像素深度 ID 的选择规则。

## 公式审查

令高斯中心为 \(\mu\)，旋转到高斯局部坐标后的射线为

\[
u=R(o-\mu),\qquad v=Rd,\qquad r_{local}(t)=u+tv.
\]

取固定等值椭球半径 \(\kappa\)：

\[
\sum_k\frac{(u_k+t v_k)^2}{(\kappa s_k)^2}=1.
\]

定义

\[
A=\sum_k\frac{v_k^2}{(\kappa s_k)^2},\quad
B=2\sum_k\frac{u_kv_k}{(\kappa s_k)^2},\quad
C=\sum_k\frac{u_k^2}{(\kappa s_k)^2}-1.
\]

则

\[
\Delta=B^2-4AC,\qquad
t_{\pm}=\frac{-B\pm\sqrt{\Delta}}{2A},\qquad
t_{mid}=\frac{t_-+t_+}{2}=-\frac{B}{2A}.
\]

沿射线的未截断高斯密度正比于

\[
\exp\left[-\frac12\sum_k\frac{(u_k+t v_k)^2}{s_k^2}\right].
\]

令指数中的二次型对 \(t\) 求导，最大密度点为

\[
t_{peak}=-\frac{\sum_k u_kv_k/s_k^2}{\sum_kv_k^2/s_k^2}
=-\frac{B}{2A}=t_{mid}.
\]

因此：

- 只要讨论同一高斯与同一射线，中点与 ray maximum-density point 精确相同。
- \(\kappa\) 或各轴统一 scale factor 会从 \(A\)、\(B\) 中约掉，不改变 \(t_{mid}\)。
- \(\Delta\) 决定该固定等值面有没有实数交点。即使 \(\Delta<0\)，无限支撑的 Gaussian 仍有合法 \(t_{peak}\)。这正是发布代码能够“不检查判别式仍输出点”的数学解释。
- `entry = t_-` 才是固定等值椭球的前表面。中点不是可见前表面，也不是物理表面；它是所选 Gaussian 的射线方向峰值。

### 有效性与阈值的两种清晰契约

**发布代码兼容契约**：pixel 在图像内；contributor 通过 `power<=0`、`alpha>=1/255`；合成没有因 `T<1e-4` 提前结束；\(t_{peak}>0\)。记录 \(\Delta\)，但不以 \(\Delta\) 掩码。

**有限椭球真相交契约**：在上述条件外要求 \(\Delta\ge\epsilon_\Delta\)，并要求至少 \(t_+>t_{near}\)。若要“前表面首击”，还必须用 \(t_-\)（相机在椭球内时需定义取 \(t_+\) 还是 near）重新排序。这会改变 ID 和覆盖，不能称为论文发布实现的等价替换。

浮点实现可用 `disc >= -eps` 后 `disc=max(disc,0)`，但 `eps` 应按 \(B^2\) 与 \(|4AC|\) 的尺度设相对容差；不能随意用固定世界单位阈值。

## camera-Z、ray distance 与单位

- rasterizer 的 contributor 排序量是 Gaussian center 的 `p_view.z`，即相机坐标 Z，单位与场景坐标一致。
- 源码用单位世界射线构造 `point_rec=o+t_peak*d_unit`，所以 \(t_{peak}\) 是沿射线的世界距离。重初始化直接消费这个世界点。
- 源码另写出的 `out_depth=(-b/(2a))/length(ray_direction)`，没有在重初始化中使用，也不应直接解释成 camera-Z。
- 若训练监督约定是 camera-Z，应显式把峰值世界点变换到相机坐标并取 `z`。等价地，若相机空间射线写成 \((x_n,y_n,1)\)，其参数 \(t\) 才直接等于 camera-Z；若使用单位射线，则 `z_cam=t_ray*d_cam.z`。
- 论文与代码没有毫米或米的固定承诺。输出继承 COLMAP/场景尺度；不要把它当作传感器毫米深度。

## 初始各向同性 sigma：何时相同、何时不同

若 \(\Sigma=\sigma^2I\)，则

\[
t_{peak}=\frac{d^T(\mu-o)}{d^Td}.
\]

这是 Gaussian center 到射线的欧氏正交投影参数。下列情况 center depth 与 midpoint/peak depth相同：

- center 恰好在该像素射线上；或
- 更一般地，中心与峰值点在相机 Z 上的投影恰好相同。

它们通常不同的情况：Gaussian center 投影位于相邻像素，但其 2D footprint 在当前像素仍有贡献。sigma 越大，这种 off-ray contributor 越常见；此时 peak 点落在当前像素射线上，而 center 不在射线上。

对固定 ID 和固定射线，把各向同性 \(\sigma\) 从 0.5 改成 0.1 不改变 \(t_{peak}\)。整体渲染仍会变化，因为 sigma 改变屏幕协方差、alpha、tile 覆盖和贡献顺序中的有效项。若初始点稀疏，0.1 可能让更多像素没有 contributor；若 sigma 太大，0.5 又可能产生跨边界贡献。应通过覆盖率和误差实测选择，不能预设越小越好。

## 对 v008 的具体实现建议

为保证当前实验只比较“取点公式”，建议把两个轴明确分开：

| 轴 | 方式 1 | 方式 2 | 说明 |
|---|---|---|---|
| Gaussian ID | 同一个 first-contributor ID | 同一个 first-contributor ID | 保持现有 RGB 前向、中心 Z 排序、alpha 阈值与 T 终止逻辑完全一致 |
| 深度位置 | 该 ID 的 center camera-Z | 该 ID 的 \(p_{peak}=o+t_{peak}d\) 再变换为 camera-Z | 这是受控 `first-center` vs `same-first-peak` 消融 |

同时输出以下诊断，不要让它们改变 RGB 或选 ID：

- `selected_id`、`alpha_at_select`、`T_before_select`；
- `z_center`、`z_peak`、`z_peak/z_center`、`abs(z_peak-z_center)`；
- 以明确的 \(\kappa\)（若复现作者 CUDA则 \(\kappa=3\)）计算 `discriminant` 与 `finite_ellipsoid_hit`；
- 正深度覆盖率、无 contributor 比例、`Δ<0` 比例、NaN/Inf 数；
- scale=0.5 与 0.1 各自的同 ID 覆盖率，以及 ID 发生变化的比例。

另设一个单独命名的 `paper-maxweight-peak` profile 才能接近论文/官方实现：遍历所有有效 contributor，选择最大 `alpha*T`，输出该 ID 的峰值点。不要把它与 `same-first-peak` 合并命名为“Mini-Splatting2 原版”。

如果实验确实要比较有限椭球表面，应再增设 `first-entry` 或 `maxweight-entry`，明确 \(\kappa\)、相机位于椭球内部时的规则、`Δ` 容差以及按 center-Z 还是 entry-Z 排序。前后交点的“中点”仍然是 peak，不提供额外的表面偏移。

