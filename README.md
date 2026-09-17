# 3DGS Experiment Standard

这是一个独立的 3D Gaussian Splatting 实验规范分支，用于把同一套环境检查、数据契约、训练记录、验证渲染和结果汇总习惯复用到不同 3DGS 项目。

本分支不包含具体 3DGS 训练源码。接入项目通过 adapter 提供相机加载、模型加载和诊断渲染能力；其余目录结构、CSV、manifest、验收和分析协议保持一致。

## 包含内容

- `AGENTS.md`：可直接放入新项目的 Agent 规则。
- `MEMORY_TEMPLATE.md`：项目记忆模板，只保存稳定语义和恢复指针。
- `docs/`：环境、数据、实验、输出和 adapter 规范；包含 ZS601 接入示例。
- `scripts/check_environment.py`：记录 Python、GPU、PyTorch、CUDA 和依赖。
- `scripts/create_run.py`：创建不可覆盖的新实验目录和初始 manifest。
- `scripts/render_val_diagnostics.py`：统一 Val RGB、1σ 椭球、深度和法向渲染入口。
- `scripts/evaluate_geometry.py`：对最终高斯中心或表面点与参考点云计算双向几何指标。
- `scripts/verify_run_outputs.py`：按统一产物契约验收冒烟或正式训练。
- `scripts/summarize_experiment.py`：从真实 CSV/JSON 生成结果表和简要总结。
- `scripts/adapters/zs601_v3_adapter.py`：当前 ZS601 v3 项目的接入示例。
- `templates/`：运行配置、实验登记和总结模板。
- `colab/3DGS_EXPERIMENT_TEMPLATE.ipynb`：显式逐阶段 Colab 模板。

## 使用方式

1. 将本分支的文件复制到目标 3DGS 项目，保留原项目训练代码。
2. 根据 `docs/ADAPTER_GUIDE.md` 编写该项目的诊断 adapter。
3. 在训练入口写出 `run_config.json`，并按 `docs/OUTPUT_SCHEMA.md` 保存 CSV、checkpoint 和验证图。
4. 先运行环境检查和 200 步冒烟，通过后再运行正式实验。
5. 正式结束后执行验收和汇总脚本，再把结果持久化到云端并回读。

```bash
python scripts/check_environment.py --require-gpu-substring L4 --output environment.json
python scripts/create_run.py --root output --run-id EXP_A_001 --group A \
  --git-commit <commit> --seed 42
python scripts/verify_run_outputs.py output/EXP_A_001 --profile smoke --iterations 200
python scripts/evaluate_geometry.py --prediction output/EXP_A_150K/point_cloud/iteration_150000/point_cloud.ply --reference /data/reference.ply --reference-role heldout_lidar --unit meters --output output/EXP_A_150K/geometry_test
python scripts/verify_run_outputs.py output/EXP_A_150K --profile formal --iterations 150000
python scripts/summarize_experiment.py output/EXP_A_150K
```

统一规范不规定某一种 3DGS 算法。不同项目可以增加 loss、初始化、增密或监督，但必须独立开关、记录最终功能矩阵，并保持输出文件可比较。
