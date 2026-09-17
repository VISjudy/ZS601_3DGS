#!/usr/bin/env python3
"""Verify a smoke or formal run against the reusable 3DGS output contract."""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path

CORE_CSV = ("loss_log.csv", "training_progress.csv", "val_metrics.csv", "geometry_metrics.csv")

def skip_reason(status, section):
    item = status.get(section, {}) if isinstance(status, dict) else {}
    reason = item.get("reason") if isinstance(item, dict) and item.get("status") == "skipped" else None
    return str(reason).strip() if reason else None


def rows(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def expected_iterations(final_iteration, interval):
    values = list(range(0, final_iteration + 1, interval))
    if not values or values[-1] != final_iteration:
        values.append(final_iteration)
    return values


def find_final_dir(run, name, iteration):
    padded = run / name / f"iteration_{iteration:06d}"
    return padded if padded.is_dir() else run / name / f"iteration_{iteration}"


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
    p.add_argument("--geometry-dir", default="geometry_test")
    p.add_argument("--worst-count", type=int, default=10)
    p.add_argument("--output", type=Path)
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args(argv)
    run = a.run.resolve(); errors = []; warnings = []; geometry_summary = None
    status_path = run / "evaluation_status.json"
    try:
        evaluation_status = json.loads(status_path.read_text(encoding="utf-8-sig")) if status_path.is_file() else {}
    except Exception as exc:
        evaluation_status = {}; errors.append({"invalid_evaluation_status": repr(exc)})
    if not run.is_dir():
        raise NotADirectoryError(run)
    for name in ("run_config.json", "run_manifest.json", "environment.json", *CORE_CSV):
        if not (run / name).is_file():
            errors.append({"missing": name})
    cfg = None
    if (run / "run_config.json").is_file():
        try:
            cfg = json.loads((run / "run_config.json").read_text(encoding="utf-8-sig"))
        except Exception as exc:
            errors.append({"invalid_run_config": repr(exc)})
    expected = expected_iterations(a.iterations, a.val_interval); actual = []
    vm = run / "val_metrics.csv"
    if vm.is_file():
        try:
            data = rows(vm)
            actual = sorted({int(r["iteration"]) for r in data if r.get("image_name") == "MEAN"})
            if actual != expected:
                errors.append({"val_iterations": {"expected": expected, "actual": actual}})
        except Exception as exc:
            errors.append({"invalid_val_metrics": repr(exc)})
    for iteration in expected:
        folder = run / a.val_dir / f"iteration_{iteration:06d}"
        if not folder.is_dir():
            errors.append({"missing_val_dir": str(folder)}); continue
        manifest = folder / "manifest.json"
        if not manifest.is_file():
            errors.append({"missing_val_manifest": str(manifest)}); continue
        try:
            item = json.loads(manifest.read_text(encoding="utf-8-sig"))
            cameras = item.get("cameras") or item.get("result", {}).get("cameras")
            count = item.get("camera_count") or item.get("result", {}).get("camera_count")
            count = len(cameras) if cameras is not None else count
            if count is not None and int(count) != a.val_cameras:
                errors.append({"val_camera_count": {"iteration": iteration, "expected": a.val_cameras, "actual": count}})
        except Exception as exc:
            errors.append({"invalid_val_manifest": {"path": str(manifest), "error": repr(exc)}})
        if list(folder.rglob("*.npz")):
            warnings.append({"large_npz_present": str(folder)})
    checkpoint_iterations = [a.iterations] if a.profile == "smoke" else expected_iterations(a.iterations, a.checkpoint_interval)[1:]
    for iteration in checkpoint_iterations:
        checkpoint = run / "checkpoints" / f"iteration_{iteration}.pth"
        models = [run / "point_cloud" / f"iteration_{iteration}" / "point_cloud.ply",
                  run / "point_cloud" / f"iteration_{iteration:06d}" / "point_cloud.ply"]
        if not checkpoint.is_file(): errors.append({"missing_checkpoint": str(checkpoint)})
        if not any(x.is_file() for x in models): errors.append({"missing_model": [str(x) for x in models]})
    if a.profile == "formal":
        final = find_final_dir(run, a.test_dir, a.iterations)
        visual_skip = skip_reason(evaluation_status, "visual_test")
        missing_test = [str(final / name) for name in ("test_metrics_per_camera.csv", "test_summary.json") if not (final / name).is_file()]
        if missing_test:
            target = warnings if visual_skip else errors
            target.append({"skipped_visual_test" if visual_skip else "missing_test_artifacts": {"files": missing_test, "reason": visual_skip}})
        worst = (list(final.rglob("worst_*_rgb.png")) + list((final / "worst10").rglob("*rgb*.png"))) if final.exists() else []
        if len({x.resolve() for x in worst}) != a.worst_count:
            target = warnings if visual_skip else errors
            target.append({"skipped_worst_diagnostics" if visual_skip else "worst_rgb_count": {"expected": a.worst_count, "actual": len({x.resolve() for x in worst}), "reason": visual_skip}})
        geometry = run / a.geometry_dir
        geometry_skip = skip_reason(evaluation_status, "geometry_test")
        missing_geometry = [str(geometry / name) for name in ("geometry_metrics.json", "geometry_metrics.csv", "manifest.json") if not (geometry / name).is_file()]
        if missing_geometry:
            target = warnings if geometry_skip else errors
            target.append({"skipped_geometry_test" if geometry_skip else "missing_geometry_artifacts": {"files": missing_geometry, "reason": geometry_skip}})
        geometry_json = geometry / "geometry_metrics.json"
        if geometry_json.is_file():
            try:
                geometry_summary = json.loads(geometry_json.read_text(encoding="utf-8-sig"))
                for key in ("unit", "reference_role", "metrics"):
                    if key not in geometry_summary: errors.append({"invalid_geometry_metrics": f"missing key: {key}"})
            except Exception as exc:
                errors.append({"invalid_geometry_metrics": repr(exc)})
        for name in ("results_table.csv", "results_table.md", "results_table.tex", "experiment_summary.md"):
            if not (run / name).is_file(): errors.append({"missing_summary": name})
    report = {
        "schema_version": "3dgs-experiment-standard/1", "run": str(run), "profile": a.profile,
        "experiment_group": (cfg or {}).get("experiment_group", (cfg or {}).get("experiment")),
        "iterations": a.iterations, "expected_val_iterations": expected, "actual_val_iterations": actual,
        "geometry_evaluation": geometry_summary, "evaluation_status": evaluation_status,
        "errors": errors, "warnings": warnings, "verified": not errors,
    }
    payload = json.dumps(report, ensure_ascii=False, indent=2); print(payload)
    if a.output:
        output = a.output if a.output.is_absolute() else run / a.output
        if output.exists() and not a.overwrite: raise FileExistsError(f"Refusing to overwrite {output}")
        output.parent.mkdir(parents=True, exist_ok=True); output.write_text(payload + "\n", encoding="utf-8")
    if errors: raise SystemExit(1)


if __name__ == "__main__":
    main()
