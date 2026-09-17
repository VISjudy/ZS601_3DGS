#!/usr/bin/env python3
"""Create a new standardized 3DGS run directory without overwriting history."""
from __future__ import annotations
import argparse, hashlib, json
from datetime import datetime, timezone
from pathlib import Path


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""): h.update(chunk)
    return h.hexdigest()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--run-id", required=True)
    p.add_argument("--group", default="custom")
    p.add_argument("--git-commit", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dataset-manifest", type=Path)
    p.add_argument("--feature", action="append", default=[], metavar="NAME=auto|on|off")
    a = p.parse_args(argv)
    run = a.root / a.run_id
    if run.exists(): raise FileExistsError(f"Refusing to overwrite existing run: {run}")
    features = {}
    for item in a.feature:
        if "=" not in item: p.error(f"invalid --feature {item!r}")
        key, value = item.split("=", 1)
        if value not in {"auto", "on", "off"}: p.error(f"invalid feature value: {item!r}")
        features[key] = value
    identity = None
    if a.dataset_manifest:
        if not a.dataset_manifest.is_file(): raise FileNotFoundError(a.dataset_manifest)
        identity = {"path": str(a.dataset_manifest.resolve()), "sha256": sha256(a.dataset_manifest)}
    for name in ("val", "checkpoints", "point_cloud", "test_final", "logs"):
        (run / name).mkdir(parents=True, exist_ok=False)
    cfg = {
        "schema_version": "3dgs-experiment-standard/1", "run_id": a.run_id,
        "experiment_group": a.group, "feature_overrides": features,
        "git_commit": a.git_commit, "seed": a.seed, "dataset_manifest": identity,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "validation": {"start_iteration": 0, "interval": 5000, "camera_count": 10, "ellipsoid_sigma": 1.0},
        "checkpoint_interval": 50000,
    }
    (run / "run_config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (run / "run_manifest.json").write_text(json.dumps({"status": "created", "artifacts": []}, indent=2) + "\n", encoding="utf-8")
    print(run.resolve())

if __name__ == "__main__": main()
