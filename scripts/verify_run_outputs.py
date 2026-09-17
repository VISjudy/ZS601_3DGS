#!/usr/bin/env python3
"""Verify a smoke or formal run against the reusable 3DGS output contract."""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path

CORE_CSV = ("loss_log.csv", "training_progress.csv", "val_metrics.csv", "geometry_metrics.csv")


def rows(path):
    with path.open(newline="", encoding="utf-8") as f: return list(csv.DictReader(f))


def expected_iterations(final_iteration, interval):
    values = list(range(0, final_iteration + 1, interval))
    if not values or values[-1] != final_iteration: values.append(final_iteration)
    return values


def find_final_dir(run, name, iteration):
    direct = run / name / f"iteration_{iteration:06d}"
    if direct.is_dir(): return direct
    plain = run / name / f"iteration_{iteration}"
    return plain


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run", type=Path)
    p.add_argument("--profile", choices=("smoke", "formal"), required=True)
    p.add_argument("--iterations", type=int, required=True)
    p.add_argument("--val-interval", type=int, default=5000)
    p.add_argument("--val-cameras", type=int, default=10)
    p.add_argument("--checkpoint-interval", type=int, default=50000)
    p.add_argument("--val-dir", default="val")
    p.add_argument("--test-dir", default="test_final")
    p.add_argument("--worst-count", type=int, default=10)
    p.add_argument("--output", type=Path)
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args(argv)
    run = a.run.resolve(); errors = []; warnings = []
    if not run.is_dir(): raise NotADirectoryError(run)
    for name in ("run_config.json", "run_manifest.json", "environment.json", *CORE_CSV):
        if not (run / name).is_file(): errors.append({"missing": name})
    cfg = None
    if (run / "run_config.json").is_file():
        try: cfg = json.loads((run / "run_config.json").read_text(encoding="utf-8-sig"))
        except Exception as exc: errors.append({"invalid_run_config": repr(exc)})
    expected = expected_iterations(a.iterations, a.val_interval)
    actual = []
    vm = run / "val_metrics.csv"
    if vm.is_file():
        try:
            data = rows(vm)
            actual = sorted({int(r["iteration"]) for r in data if r.get("image_name") == "MEAN"})
            if actual != expected: errors.append({"val_iterations": {"expected": expected, "actual": actual}})
        except Exception as exc: errors.append({"invalid_val_metrics": repr(exc)})
    for iteration in expected:
        folder = run / a.val_dir / f"iteration_{iteration:06d}"
        if not folder.is_dir(): errors.append({"missing_val_dir": str(folder)}); continue
        manifest = folder / "manifest.json"
        if not manifest.is_file(): errors.append({"missing_val_manifest": str(manifest)}); continue
        try:
            item = json.loads(manifest.read_text(encoding="utf-8-sig"))
            cameras = item.get("cameras") or item.get("result", {}).get("cameras")
            count = item.get("camera_count") or item.get("result", {}).get("camera_count")
            count = len(cameras) if cameras is not None else count
            if count is not None and int(count) != a.val_cameras:
                errors.append({"val_camera_count": {"iteration": iteration, "expected": a.val_cameras, "actual": count}})
        except Exception as exc: errors.append({"invalid_val_manifest": {"path": str(manifest), "error": repr(exc)}})
        if list(folder.rglob("*.npz")): warnings.append({"large_npz_present": str(folder)})
    checkpoint_iterations = [a.iterations] if a.profile == "smoke" else expected_iterations(a.iterations, a.checkpoint_interval)[1:]
    for iteration in checkpoint_iterations:
        checkpoint = run / "checkpoints" / f"iteration_{iteration}.pth"
        model_candidates = [run / "point_cloud" / f"iteration_{iteration}" / "point_cloud.ply",
                            run / "point_cloud" / f"iteration_{iteration:06d}" / "point_cloud.ply"]
        if not checkpoint.is_file(): errors.append({"missing_checkpoint": str(checkpoint)})
        if not any(x.is_file() for x in model_candidates): errors.append({"missing_model": [str(x) for x in model_candidates]})
    if a.profile == "formal":
        final = find_final_dir(run, a.test_dir, a.iterations)
        for name in ("test_metrics_per_camera.csv", "test_summary.json"):
            if not (final / name).is_file(): errors.append({"missing_test_artifact": str(final / name)})
        worst_rgb = list(final.rglob("worst_*_rgb.png")) + list((final / "worst10").rglob("*rgb*.png")) if final.exists() else []
        if len({x.resolve() for x in worst_rgb}) != a.worst_count:
            errors.append({"worst_rgb_count": {"expected": a.worst_count, "actual": len({x.resolve() for x in worst_rgb})}})
        for name in ("results_table.csv", "results_table.md", "results_table.tex", "experiment_summary.md"):
            if not (run / name).is_file(): errors.append({"missing_summary": name})
    report = {
        "schema_version": "3dgs-experiment-standard/1", "run": str(run), "profile": a.profile,
        "experiment_group": (cfg or {}).get("experiment_group", (cfg or {}).get("experiment")),
        "iterations": a.iterations, "expected_val_iterations": expected, "actual_val_iterations": actual,
        "errors": errors, "warnings": warnings, "verified": not errors,
    }
    payload = json.dumps(report, ensure_ascii=False, indent=2); print(payload)
    if a.output:
        output = a.output if a.output.is_absolute() else run / a.output
        if output.exists() and not a.overwrite: raise FileExistsError(f"Refusing to overwrite {output}")
        output.parent.mkdir(parents=True, exist_ok=True); output.write_text(payload + "\n", encoding="utf-8")
    if errors: raise SystemExit(1)

if __name__ == "__main__": main()
