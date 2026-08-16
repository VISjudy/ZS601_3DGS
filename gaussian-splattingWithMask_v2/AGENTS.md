# AGENTS.md — ZS601 3DGS 几何质量提升项目 · 共享经验手册

> 本文档汇总本项目所有开发阶段(阶段 0/1/2)中各 agent 沉淀的经验与约定,
> 供所有参与本项目的 agent(含子 agent)阅读对齐。**修改代码/流程前先读本文档;
> 产生新的通用经验时,请同步更新本文档并告知用户。**
> 最后更新:阶段 2(法向量约束 loss)完成后。

---

## 1. 项目总览

- **目标**:在 ZS601 会议室室内场景上提升 3D Gaussian Splatting 的几何质量(对照激光点云),
  通过初始化方式与形状/法向约束 loss 的消融实验(A/B/C/D 组)验证。
- **数据**:2694 张图 + 二值 mask,22.8 万激光点云初始化,室内场景(无天空背景)。
- **训练入口**:`train_mask.py`;评估:`evaluate.py`(图像指标 + cloud2cloud 几何指标);
  预处理:`preprocess.py`(点云转换/法向量/训练-val 切分/验证渲染,产物在 `intermediate/`)。
- **通用约定**:seed 42、中文注释(仅限 Python)、先冒烟(2000 轮)后正式(150000 轮)、
  每功能增量验证、反馈机制(print + 可视化产物)必须到位、向后兼容(新参数默认不改变原行为时优先)。

## 2. 目录结构与分工

| 路径 | 说明 |
|---|---|
| `gaussian-splattingWithMask/`(v1) | 阶段 0/1 代码,**已冻结**,勿改动;A/B 组正式训练产物基于它 |
| `gaussian-splattingWithMask_v2/`(v2) | 阶段 2 起开发目录;光栅化器有 CUDA 改动(normal 通道) |
| `gs_code.zip` / `gs_code_v2.zip` | 上传 Colab 的代码包,由根目录 `pack_code.py` 生成 |
| `colab/ZS601_3DGS_stagesv2.ipynb` | v2 的主 notebook(注意文件名带 v2 后缀) |
| `D组实验_*.md` | 实验计划文档(用户维护) |

## 3. 数据集关键语义(极易踩坑,务必牢记)

1. **mask 极性**:masks 目录为 mode L 严格二值(0/255),**黑(0)=有效像素(静态主体,保留),
   白(255)=动态物体(人,剔除)**。与原版 WithMask 仓库假设相反!
   代码在 `scene/cameras.py` 的 `_load_mask` 中统一取反(1-mask),训练/val/test 口径一致。
   曾因此搞反导致 PSNR 卡 13~15,是历史最大坑。
2. **三连拍 `_1` 帧失效**:图像命名 `时间戳_0/_1/_2`,约 85% 的 `_1` 帧 mask 全黑
   (全局 761/2694=28.2%)。mask 取反后这些帧全帧有效,无需剔除;
   训练循环有"无有效像素视图跳过"逻辑(sum=0 即原图 mask 全白),重抽上限 100 次。
3. **mask 抢救**:`_rescue_mask` 处理 0/1 编码被按 0/255 读入的情况(max<0.5 时重新二值化)。
4. **GT 与 mask 路径独立**:GT 走 `images/`,mask 走 `masks/`(有 `map_0/camera_0` 两级子目录,
   审计脚本需递归 glob)。val/test 评估相机必须同样加载 mask,否则指标被系统性压低。

## 4. 运行环境(Colab)

- **GPU**:L4;CUDA 12.8;Python 3.12;torch 官方 whl。
- **数据**:Drive 路径 `我的云端硬盘/LCCDataset/ZS601meetingroom/ZS601meetingroom_data.zip`,
  代码包 `gs_code(_v2).zip` 同目录;结果回传 `.../results`。
- **工作目录**:`/content/gs_work`(临时盘!见 6.5);`CODE_DIR=/content/gs_work/gaussian-splattingWithMask_v2`,
  `DATA_DIR=/content/gs_work/dataset`。
- **必带参数**:`--data_device cpu --lazy_load --lazy_cache 100 --alpha_masks masks --disable_viewer`。
  - `--lazy_cache`:100 ≈ 1.1GB(免费版安全);1000 ≈ 11GB、提速约 10%,仅 25GB+ 内存实例可用。
- **依赖安装**:simple-knn / diff-gaussian-rasterization 用 `pip install ./submodules/...` 编译;
  **fused-ssim 不随 zip 分发**,notebook 里 `git clone https://github.com/rahul-goel/fused-ssim.git`
  后 `pip install . --no-build-isolation`,失败自动回退普通 ssim(可选组件)。
- **`/content` 是临时盘**:运行时重启即清空。重连后必须按顺序重跑:挂载 Drive → 解压代码/数据
  → 编译 CUDA 子模块,否则报 `Could not recognize scene type!` 或模块缺失。

## 5. v2 架构要点(阶段 2)

### 5.1 光栅化器 normal 通道(submodules/diff-gaussian-rasterization)
- `preprocessCUDA`:每高斯法向 = 旋转矩阵第 3 列(最小尺度轴),与 `point - campos` 点积为正
  则翻转,**始终朝向相机**;存入 geom 缓冲 `geom.normals`(3·P,前后向同一布局)。
- `renderCUDA`:按与 invdepth 相同的 `alpha*T` 加权合成 3×H×W 法向图(背景 0,不混 bg_color)。
- 反向:backward render 同 invdepth 模式累计每高斯法向梯度(含 alpha 项);
  `preprocessCUDA` 用 n=(2(xz+ry), 2(yz−rx), 1−2(x²+y²)) 的解析导数把梯度 atomicAdd 回四元数
  (**法向与 scale 无关**;翻转符号前后向相消)。
- Python 绑定:forward 返回 4 元组 `(color, radii, invdepths, normal)`;
  `gaussian_renderer.render()` 返回字典新增 `"normal"` 键(不 clamp,分量 ∈ [-1,1])。

### 5.2 训练 loss 与参数(ModelParams)
| 参数 | 默认 | 含义 |
|---|---|---|
| `--lambda_normal` | 0.2 | 渲染法向 vs 深度导出法向一致性:`mean(1-|cos|)`,mask==0 与无效深度像素不参与 |
| `--lambda_scale` | 0.1 | z 轴厚度正则:`|exp(scaling)[:,2]|` 均值(参考 2DGS) |
| `--lambda_size` | 0 | 防大椭球:逐高斯最长尺度轴 exp 后均值正则(建议 0.01 起调) |
| `--init_2d` / `--init_2d_z` | False / -10.0 | 2D 椭球初始化(z≈4.5e-5,法向对齐点云法线) |
| `--freeze_2d_z` | False | 训练中 z 轴缩放复位保持扁平;**必须配合 --init_2d**,单开报错 |

- **重跑 A/B 组 baseline 必须显式传 `--lambda_scale 0 --lambda_normal 0`**(默认非 0!)。
- 深度导出法向:`train_mask.py::normal_from_invdepth`(逆深度→深度→反投影→中心差分叉积,
  朝相机方向,内部 H-2×W-2 像素)。
- `loss_log.csv` 为 8 列:iteration,L1,SSIM_loss,depth_loss,normal_loss,scale_reg,size_reg,total;
  每 10 轮 print 一次 Normal Loss / Scale Reg(仅开启时)。
- val 渲染**每 100 轮**(iteration % 100 == 1),输出 RGB + 光栅化器 normal 通道 `(n+1)/2` 贴图,
  GT 仅首轮保存;`val_metrics.csv` 含 num_gaussians / elapsed_sec。

### 5.3 实验组定义(超参完全一致,仅开关不同)
| 组 | 开关 | 输出目录 |
|---|---|---|
| A baseline | 无 | `output/zs601_baseline` |
| B init2d+freeze | `--init_2d --freeze_2d_z` | `output/zs601_init2d` |
| C init2d+法向约束 | `--init_2d --lambda_scale 0.1 --lambda_normal 0.2` | `output/expC` |
| D | 见 `D组实验_*.md` | 按文档 |

### 5.4 断点续训
- 训练命令必须带 `--checkpoint_iterations 50000 100000 150000`(chkpnt 含模型+优化器+致密化统计+轮数);
  `--save_iterations` 的 ply 不能用于续训。
- 续训:同命令 + `--start_checkpoint output/expC/chkpnt100000.pth`,从下一轮继续。
- chkpnt 在 /content 临时盘,**训练中要定期 cp 到 Drive**(notebook 有同步/拉回/续训三个单元格)。

## 6. 已知坑与修复模式(血泪教训)

1. **mask 极性搞反**:见 3.1。任何涉及 mask 的改动,先用 `Image.composite` 合成图目视验证语义。
2. **设备不一致**:`--data_device cpu` 时相机矩阵在 CPU,光栅化器 CUDA 核直接解引用指针 →
   illegal memory access / numel 溢出等离奇报错。修复:`gaussian_renderer/__init__.py` 中
   `world_view_transform / full_proj_transform / camera_center` 强制 `.cuda()`(数据在 GPU 时为空操作)。
   同理 `evaluate.py` 中 `cam.alpha_mask.to("cuda")`。
3. **致密化 Inf 梯度**:`densify_and_prune` 中 denom=0 产生 Inf,原代码只清 NaN → Inf 通过阈值
   → clone/split 出非有限坐标。修复:`torch.where(torch.isfinite(...))` 清零 + prune 剔除 bad_xyz 兜底;
   `evaluate.py` 几何统计同样过滤非有限中心。
4. **CUDA 部署版本混搭(两次编译报错的真根因)**:Colab 上 `forward.h` 停留在旧版(无 normals/out_normal
   参数),而 .cu 实现文件是新版,导致:①定义处报 `should have been declared inside 'FORWARD'`
   (新签名定义在旧头文件的命名空间里找不到匹配声明);②调用处报 `too many arguments in function call`。
   曾误判为“中文注释导致”,实为旧目录未清空就解压/多来源同步造成的文件新旧混搭。
   **约定:部署前先 `rm -rf` 旧代码目录再解压;编译前用标记 grep 校验关键文件版本
   (forward.h 含 out_normal、backward.cu 含 dL_dnormals 等);头文件与实现文件必须同源更新。**
5. **CUDA 源码建议全 ASCII 注释**:非 ASCII 注释不是上述报错的根因(已澄清),但为规避
   宿主编译器/locale 相关的潜在风险仍约定 .cu/.h/.cpp 用英文注释;文件名不得含非 ASCII
   (`forward.cu（原）.cu` 等备份仅本地留存,不进包)。
6. **/content 临时盘**:见 4.5。产物(ply/chkpnt/日志/中间图)要及时同步 Drive。
7. **限定的命名空间外定义**:forward.cu 的 FORWARD::render/preprocess 已改为 namespace 块内定义,
   新增包装函数请沿用块内写法。
8. **SearchReplace/Grep 工具显示会去掉命中行缩进**:匹配失败时用 python `repr()` 输出真实缩进再重试。
9. **PowerShell**:不支持 `&&`(用 `;`);内联 `-c` 命令会吞 `$VAR`;含中文输出乱码 →
   写 utf-8 脚本文件执行。Colab heredoc 补丁用 `<<'EOF'` 防变量展开。
10. **notebook 单元格编辑**:用户会自行合并/改名单元格,编辑前先 dump 现状比对,用断言定位,
    失败后重新 dump 确认结构再改。

## 7. 打包规则(pack_code.py,已固化)

1. 目录排除:`__pycache__`、`.ipynb_checkpoints`、`build`、`.git`、`doc`(glm 文档)、
   `*.egg-info`、`fused-ssim`(git clone 安装)、`fused-ssim000`(备份)。
2. 文件排除:`*.pyc/.obj/.lib/.exp/.pyd/.so`、`loss_log.csv`、`val_metrics.csv`、
   文件名含非 ASCII 的备份文件。
3. `third_party` 下图片(.jpg/.png/.tiff 等)排除;glm 头文件必须保留(编译依赖)。
4. ZIP_DEFLATED compresslevel=9;顶层目录 `gaussian-splattingWithMask_v2/`。
5. 打包后自检(脚本内置):fused-ssim 未混入、关键文件齐全(含 colab notebook 与 AGENTS.md)、
   光栅化器 .cu/.h/.cpp 全 ASCII。

## 8. 工作流约定

- **增量开发**:每功能先冒烟(2000 轮)验证跑通 → 正式训练;每步有 print + 可视化产物 +
  csv 指标(loss 曲线、val_metrics),验证有效再叠加下一功能。
- **交付习惯**:只改代码并汇报变更文件清单 + Colab 操作指引;不主动打包以外的额外文件;
  用户负责 Colab 执行并把日志/报错贴回。
- **编译问题诊断**:pip 报错只显示尾部,用
  `python setup.py bdist_wheel > /tmp/build.log 2>&1; tail -n 60 /tmp/build.log` 拿真实错误。
- **消融实验必须含 cloud2cloud 几何指标**(高斯中心 vs 激光参考点云,evaluate.py 输出
  mean/median/rmse/p90,剔除 99 分位外离群与有限性过滤)。
- **内嵌浏览器无法登录 Google**(安全策略拦截),涉及 Colab/Drive 操作由用户在系统浏览器执行。

## 9. 当前状态速览

- 阶段 0(预处理)/阶段 1(A/B/C 组训练与评估)已完成,产物在 Drive results。
- 阶段 2(法向约束 loss):代码与 notebook 就绪,**C 组正式训练(output/expC)进行中/待执行**;
  冒烟验收点:Normal Loss/Scale Reg 打印、loss_log.csv 8 列、val_render 法向图与阶段 0 点云法向投影图趋势一致。
- 三组汇总:notebook 末单元格生成 `intermediate/ablation/summary.md` + 同视角 RGB/法向三组拼图。
