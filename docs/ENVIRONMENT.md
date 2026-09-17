# 环境规范

## 必须记录

每次冒烟和正式实验都保存 `environment.json`，至少包含：

- 操作系统、Python 版本与可执行文件路径。
- GPU 名称、显存、驱动版本和 `nvidia-smi` 原始输出。
- PyTorch、PyTorch CUDA runtime、CUDA 是否可用。
- 关键 CUDA 扩展能否导入。
- Git commit、工作区是否干净。
- 安装的关键包版本。

使用 `scripts/check_environment.py` 生成记录。正式实验优先 L4；项目需要其他 GPU 时必须在运行配置中写明。

## CUDA 扩展

每个新运行时都重新编译项目 CUDA 扩展。编译使用详细输出：

```bash
python -m pip install -v --no-build-isolation <extension-path>
```

保存第一条 nvcc/C++ 编译错误。只有扩展成功导入并完成目标测试，才能登记为环境验收通过。

## 版本变化

Python、PyTorch、CUDA、编译器或驱动任一变化，都视为新环境。旧环境的成功记录只能作为参考，不能替代本次验证。

## 通用评估工具依赖

安装 `requirements-tools.txt` 中的 NumPy、SciPy 和 plyfile。SciPy 用于大规模最近邻查询；缺少 SciPy 时，几何脚本只对小点集使用低速回退。项目自有数据格式通过 adapter 转成 PLY、NPY 或 NPZ 后再评估。
