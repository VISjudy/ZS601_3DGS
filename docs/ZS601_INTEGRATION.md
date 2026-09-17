# ZS601 v3 接入说明

当前 `v3-experiment-standard/gaussian-splattingWithMask_v3` 已实现大部分统一产物。使用本模板时：

1. 保持 ZS601 训练源码位于独立 checkout。
2. 在运行配置中加入：

```json
{
  "project_root": "/content/ZS601_3DGS/gaussian-splattingWithMask_v3"
}
```

3. 补渲染验证诊断：

```bash
python scripts/render_val_diagnostics.py \
  --adapter scripts/adapters/zs601_v3_adapter.py \
  --run-config <run>/run_config.json \
  --model <run>/point_cloud/iteration_150000/point_cloud.ply \
  --iteration 150000 \
  --sigma 1 \
  --output <new-diagnostic-output>
```

4. 验收现有 ZS601 输出时指定其目录名：

```bash
python scripts/verify_run_outputs.py <run> \
  --profile formal --iterations 150000 --val-dir val_v3 \
  --output verification_standard.json
```

当前项目的训练脚本继续负责生成模型、loss、指标和原生诊断；本模板负责跨项目统一规则、验收和汇总。新项目优先直接采用 `val/iteration_XXXXXX/<camera>/...` 的标准目录。
