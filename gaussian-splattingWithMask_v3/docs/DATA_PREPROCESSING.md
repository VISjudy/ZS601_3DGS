# ZS601 v3 数据预处理

预处理分为 SfM、LiDAR 和监督图生成三条可审计流程。所有结果写入 `processed_v3/`；原始 RGB、mask、标定、LAS 和 PLY 均保持只读。

## 1. 输出目录

```text
processed_v3/
├── images/
├── masks/
├── sparse/0/
├── geometry/
│   ├── lidar_static_rgb.ply
│   ├── lidar_static_rgb_normal_oriented.ply
│   ├── normal_orientation.csv
│   └── normal_orientation_summary.json
├── supervision/
│   ├── depth/
│   ├── depth_preview/
│   ├── depth_valid/
│   ├── normal/
│   ├── normal_float/
│   └── normal_valid/
├── previews/
│   └── normal_overlay/
└── manifests/
```

重复运行前读取 stage manifest。除非用户显式指定新的输出目录或允许覆盖，不覆盖既有产物。

## 2. SfM 阶段

1. 读取 RGB，记录名称、尺寸和格式。
2. 读取内参，验证模型、焦距、主点、畸变参数和图像尺寸。
3. 把外参统一为 COLMAP world-to-camera 表达，并计算 `C = -R^T t`。
4. 按名称关联 RGB、mask 和相机记录，检查重复与缺失。
5. 生成或核对 `cameras.*`、`images.*` 和 `points3D.*`。
6. 固化 train/val/test，验证 `images-val10.txt` 恰好有 10 个有效、唯一相机。
7. 若 bin/txt 同时存在，比较内容并记录训练读取优先级。
8. 输出相机中心包围盒、内参范围、划分数量、异常清单和相机坐标轴预览。

## 3. LiDAR 清理与法向估计

输入为上游已经去除动态目标的彩色点云。本阶段只删除无法参与计算的 NaN、Inf 或无法解析点。重复点、孤立点和离群点仅统计，额外过滤默认关闭。

对每个点使用 KNN/PCA 估计局部表面法向：计算邻域协方差矩阵，取最小特征值对应的特征向量。记录邻居数、曲率和可靠度。邻域大小、最大半径、最少邻居数和有效阈值均写入 stage manifest。低可靠度点保留在点云中，但 `normal_valid=0`。

## 4. 使用训练相机确定法向可见侧

PCA 只能确定法向轴，不能确定符号。`--orient-normals-camera auto|on|off` 控制相机辅助定向；`auto` 由实验预设决定。

对每个法向有效点：

1. 仅从训练相机筛选候选相机；点必须在相机前方、投影位于图像内且没有被训练 mask 排除。
2. 若启用遮挡检查，用与监督图相同的 LiDAR z-buffer 排除被遮挡候选。
3. 选择距离最近的最多 8 个有效相机。
4. 对每个候选计算 `v_i = normalize(C_i - p)`。
5. 以距离、观察夹角和可见性置信度形成归一化权重，计算 `score = sum(w_i * dot(n, v_i))`。
6. `score < 0` 时翻转法向；以 `abs(score)` 参与定向置信度计算。
7. 候选或置信度不足时，可由独立开关执行邻域符号传播。
8. 仍无法判断的点保留原 PCA 符号，标记 `unresolved=1` 且不进入强法向监督。

这里的方向表示传感器可见侧，不代表封闭物体的严格语义内外。

定向 PLY 至少保存：`x/y/z`、RGB、`nx/ny/nz`、`normal_confidence`、`orientation_confidence`、`curvature` 和 `normal_valid`。汇总需包含翻转率、可靠定向率、unresolved 比例、法向长度、相机方向点积分位数、邻域一致性和每相机可见点数。

## 5. 逐相机法向与深度监督

深度和法向必须共用一次投影和 z-buffer 结果，保证同一像素对应同一表面：

1. 用 world-to-camera 外参变换点，过滤相机后方和近远范围外的点。
2. 投影到像素，以相机坐标 `z` 做 z-buffer，仅保留最近点。
3. 应用图像 mask 和点级 `normal_valid`。
4. 世界法向用 `R` 转到相机坐标，并再次使其朝向相机可见侧。
5. 浮点法向以 `float32`、范围 `[-1,1]` 保存为 NPY。
6. 预览图用 `(n + 1) / 2` 映射到 `[0,255]`；无效性只由独立 valid mask 表达。
7. 深度保存为相机坐标 `z-depth`，单位与场景长度单位一致。

每个相机输出：

```text
supervision/normal/<image_name>.png
supervision/normal_float/<image_name>.npy
supervision/normal_valid/<image_name>.png
previews/normal_overlay/<image_name>.png
supervision/depth/<image_name>.npy
supervision/depth_preview/<image_name>.png
supervision/depth_valid/<image_name>.png
```

原始监督图保持稀疏。小孔洞插值只能写入独立目录并显式标记，不能替换原始监督。

## 6. “大片黑色”诊断

预览黑色不能直接解释为法向错误。报告应分别统计：有效覆盖率、无 LiDAR 投影比例、mask 排除比例、低置信度排除比例、有效像素中接近纯黑比例，以及翻转前后的邻域方向一致性。

`(0,0,0)` 法向映射后应接近中灰而非黑色。有效区出现大量纯黑通常意味着编码、数据类型、有效 mask 或渲染链路错误。稀疏未覆盖区域应保持无效，不能用无依据法向填充。

## 7. 深度反投影验证

使用内参把每个有效深度像素反投影到相机坐标，再用外参逆变换到世界坐标，与 z-buffer 选中的源 LiDAR 点比较。报告均值、中位数、P90、P95 和最大误差，并保存抽样相机的点云对照预览。误差阈值应结合输入分辨率和场景单位配置，不能写死为未经验证的数值。

## 8. 阶段入口与验收

统一入口应支持：

```text
python scripts/preprocess_dataset_v3.py --experiment-group E --stage sfm
python scripts/preprocess_dataset_v3.py --experiment-group E --stage lidar
python scripts/preprocess_dataset_v3.py --experiment-group E --stage supervision
python scripts/preprocess_dataset_v3.py --experiment-group E --stage validate
python scripts/preprocess_dataset_v3.py --experiment-group E --stage all
```

正式训练前必须通过名称对应、val10、坐标与单位、投影对齐、法向统计和深度反投影检查。每个阶段记录输入哈希、参数、代码提交、开始/结束时间、输出清单和状态。
