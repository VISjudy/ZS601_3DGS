# Val 诊断 Adapter 接口

`scripts/render_val_diagnostics.py` 不直接依赖某个 3DGS 仓库。目标项目提供一个 Python adapter 文件：

```python
def render_validation(*, run_config, model_path, output_dir, iteration, sigma):
    ...
    return {
        "camera_count": 10,
        "artifacts": ["relative/path.png", ...],
        "notes": {"depth": "camera-z", "normal": "camera coordinates"},
    }
```

要求：

1. 从 `run_config` 恢复固定验证相机和渲染设置。
2. 从 `model_path` 只读加载模型，不修改源运行。
3. 输出目录由驱动脚本创建，adapter 不能写到其他位置。
4. 每个相机至少输出 RGB、1σ 椭球、深度和法向；有 GT 时同时输出 GT 与 valid mask。
5. 返回 JSON 可序列化结果；驱动脚本写入顶层 `manifest.json`。
6. `sigma=1.0` 表示沿高斯三个主轴显示一个标准差。不要用固定球体代替真实尺度和旋转。

当前项目示例位于 `scripts/adapters/zs601_v3_adapter.py`。其他项目复制该文件并替换三处：相机加载、模型加载、诊断导出。
