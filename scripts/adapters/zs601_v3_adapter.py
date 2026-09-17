#!/usr/bin/env python3
"""Example adapter for gaussian-splattingWithMask_v3.

The target project must be checked out separately. Set project_root in
run_config.json or THREEDGS_PROJECT_ROOT in the environment.
"""
from __future__ import annotations
import os, sys
from pathlib import Path
from types import SimpleNamespace


def render_validation(*, run_config, model_path, output_dir, iteration, sigma):
    if abs(float(sigma) - 1.0) > 1e-9:
        raise ValueError("ZS601 v3 adapter currently supports only 1-sigma ellipsoids")
    project_root = Path(run_config.get("project_root") or os.environ.get("THREEDGS_PROJECT_ROOT", ""))
    if not project_root.is_dir():
        raise NotADirectoryError("Set run_config.project_root or THREEDGS_PROJECT_ROOT to gaussian-splattingWithMask_v3")
    sys.path.insert(0, str(project_root.resolve()))
    from data_v3 import load_data
    from scene.gaussian_model import GaussianModel
    from render_v3 import export_val
    cfg = dict(run_config); cfg["model_path"] = str(output_dir)
    cfg.setdefault("val_rgb", "on"); cfg.setdefault("val_depth", "on")
    cfg.setdefault("val_normal", "on"); cfg.setdefault("val_ellipsoids", "on")
    cfg.setdefault("val_npz", "off")
    args = SimpleNamespace(**cfg)
    _, _, _, cameras, _, _, _, _ = load_data(args)
    model = GaussianModel(args.sh_degree, "default", False, -10.0, False)
    model.load_ply(str(model_path), False)
    pipe = SimpleNamespace(
        debug=bool(cfg.get("debug", False)), antialiasing=bool(cfg.get("antialiasing", False)),
        convert_SHs_python=bool(cfg.get("convert_SHs_python", False)),
        compute_cov3D_python=bool(cfg.get("compute_cov3D_python", False)),
    )
    export_val(args, int(iteration), cameras, model, pipe)
    artifacts = [str(path.relative_to(output_dir)) for path in output_dir.rglob("*") if path.is_file()]
    return {
        "camera_count": len(cameras), "cameras": [getattr(c, "image_name", str(i)) for i, c in enumerate(cameras)],
        "artifacts": artifacts,
        "notes": {"ellipsoid": "1-sigma, current Gaussian scale and rotation", "depth": "camera-z", "normal": "renderer convention"},
    }
