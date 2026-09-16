# ZS601 输入数据集规范

本文定义 v3 预处理、训练和评估共同使用的数据契约。原始数据来自 L2 PRO 采集和 LCC 处理流程；原始文件按只读输入管理，所有派生文件写入场景目录下的 `processed_v3/`。

## 1. 逻辑路径

Google Drive 中的场景逻辑路径为：

```text
MyDrive/LCCDataset/zs601_output/<scene_name>/
```

推荐结构如下。已有数据无需迁移，可在预处理配置中把实际路径映射到这些逻辑角色。

```text
<scene_name>/
├── raw/
│   ├── images/              # 透视 RGB 图像
│   ├── masks/               # 与 RGB 同名或可确定映射的 mask
│   ├── camera_metadata/     # 内参、外参和图像姿态对应关系
│   └── lidar/               # 已去除动态目标的彩色 LiDAR 点云
└── processed_v3/            # 可重复生成，不写回 raw
```

Colab 运行时应先把选定场景复制到 `/content/datasets/<scene_name>/`，训练只读取本地副本。运行日志需记录 Drive 源路径、本地目标路径、复制耗时、文件数和磁盘可用空间。

## 2. SfM 与相机输入

每幅 RGB 图像必须具备唯一名称，并能关联到一个相机内参和一条外参记录。输入至少包括：

- 透视 RGB 图像及其宽、高、格式；
- 相机模型、焦距、主点和畸变参数；
- 相机外参；
- 图像 mask；
- 图像名称、相机 ID 和图像 ID 的映射。

COLMAP 外参采用 world-to-camera 约定：

```text
x_camera = R * x_world + t
C_world = -R^T * t
```

若 `cameras.bin/images.bin` 与对应 `.txt` 同时存在，预处理必须比较二者并在 manifest 中记录训练实际读取的格式。不能默认两者一致，也不能让旧 `.bin` 绕过新划分。

## 3. mask 语义

本项目当前基线的 mask 语义为黑色（0）有效、白色（255）剔除，加载时取反为训练有效 mask。manifest 必须记录该黑白编码、训练有效区域的布尔定义以及是否取反；新场景仍需用代表性样本确认。训练和监督图都使用显式有效 mask，不能根据像素是否为黑色猜测有效性。

## 4. LiDAR 输入

LiDAR 输入是已经完成动态目标去除的静态彩色点云。本流程不再次执行动态目标删除。必须记录：

- 采集设备和上游动态目标去除流程的说明或来源指针；
- 原始 LAS、转换 PLY 和最终训练 PLY 各自的路径与用途；
- 文件大小、修改时间和 SHA-256；
- 点字段，至少包括 `x/y/z`，通常包括 `red/green/blue`；
- 点数、包围盒、颜色范围、非有限值数量和相邻点距离分布；
- 坐标系、轴方向和长度单位。

单位必须结合相机中心分布、点云包围盒和相邻点间距验证，不能只根据文件名假定为米。

## 5. 数据划分

训练、验证和测试名单必须固化并写入 manifest。`images-val10.txt` 必须包含恰好 10 个有效且不重复的相机；这些相机不能参与基于相机位置的法向定向。测试相机同样不能用于预处理中的相机投票。

## 6. 数据身份与最小 manifest

`processed_v3/manifests/dataset_manifest.json` 至少记录：

```json
{
  "schema_version": "1.0",
  "scene_name": "<scene_name>",
  "source_root": "<actual-source-root>",
  "coordinate_convention": "COLMAP world-to-camera",
  "length_unit": "待校验",
  "depth_type": "camera_z",
  "images": {"count": null, "path": "<mapped-path>"},
  "masks": {"count": null, "path": "<mapped-path>", "valid_semantics": "待校验"},
  "cameras": {"train": null, "val": 10, "test": null},
  "lidar": {"dynamic_objects_removed_upstream": true, "sha256": "待计算"},
  "generated_at": "<ISO-8601>"
}
```

`null` 和“待校验”表示缺少已核验证据；生成工具不得用估计值替换。

## 7. 阻断条件

出现下列情况时停止正式训练：图像、mask 与相机记录无法一一对应；ID 重复；val10 无效；相机或点云单位不一致；LiDAR 投影存在明显系统偏移；深度反投影异常；输入哈希与实验登记不符。
