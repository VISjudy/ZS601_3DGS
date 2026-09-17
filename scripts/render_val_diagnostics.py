#!/usr/bin/env python3
"""Render standardized validation diagnostics through a project adapter."""
from __future__ import annotations
import argparse, importlib.util, json
from datetime import datetime, timezone
from pathlib import Path


def load_adapter(path):
    spec = importlib.util.spec_from_file_location("experiment_adapter", path)
    if spec is None or spec.loader is None: raise ImportError(f"Cannot load adapter: {path}")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    if not callable(getattr(module, "render_validation", None)):
        raise AttributeError("Adapter must define render_validation(...)")
    return module


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--adapter", type=Path, required=True)
    p.add_argument("--run-config", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--iteration", type=int, required=True)
    p.add_argument("--sigma", type=float, default=1.0)
    a = p.parse_args(argv)
    if a.sigma <= 0: p.error("--sigma must be positive")
    for path in (a.adapter, a.run_config, a.model):
        if not path.is_file(): raise FileNotFoundError(path)
    if a.output.exists(): raise FileExistsError(f"Refusing to overwrite {a.output}")
    cfg = json.loads(a.run_config.read_text(encoding="utf-8"))
    adapter = load_adapter(a.adapter)
    a.output.mkdir(parents=True, exist_ok=False)
    result = adapter.render_validation(run_config=cfg, model_path=a.model.resolve(),
                                       output_dir=a.output.resolve(), iteration=a.iteration,
                                       sigma=a.sigma)
    if not isinstance(result, dict): raise TypeError("Adapter result must be a dict")
    manifest = {
        "schema_version": "3dgs-experiment-standard/1", "created_at": datetime.now(timezone.utc).isoformat(),
        "adapter": str(a.adapter.resolve()), "model": str(a.model.resolve()),
        "iteration": a.iteration, "ellipsoid_sigma": a.sigma, "result": result,
    }
    (a.output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))

if __name__ == "__main__": main()
