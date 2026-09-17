#!/usr/bin/env python3
"""Generate comparison-ready Markdown/CSV/LaTeX summaries from real run artifacts."""
from __future__ import annotations
import argparse, csv, json, math
from pathlib import Path


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as f: return list(csv.DictReader(f))


def number(value):
    try:
        result = float(value); return result if math.isfinite(result) else None
    except (TypeError, ValueError): return None


def find_test_csv(run, iteration):
    choices = [run / "test_final" / f"iteration_{iteration:06d}" / "test_metrics_per_camera.csv",
               run / "test_final" / f"iteration_{iteration}" / "test_metrics_per_camera.csv"]
    return next((x for x in choices if x.is_file()), None)


def mean_metric(rows, names):
    for name in names:
        values = [number(r.get(name)) for r in rows]; values = [x for x in values if x is not None]
        if values: return sum(values) / len(values)
    return None


def fmt(value): return "null" if value is None else f"{value:.6g}"


def nested(obj, *keys):
    for key in keys:
        if not isinstance(obj, dict): return None
        obj = obj.get(key)
    return obj


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run", type=Path)
    p.add_argument("--iterations", type=int, default=150000)
    p.add_argument("--geometry-dir", default="geometry_test")
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args(argv); run = a.run.resolve()
    cfg_path = run / "run_config.json"
    if not cfg_path.is_file(): raise FileNotFoundError(cfg_path)
    cfg = json.loads(cfg_path.read_text(encoding="utf-8-sig"))
    test_path = find_test_csv(run, a.iterations); test_rows = read_csv(test_path) if test_path else []
    progress_path = run / "training_progress.csv"; progress = read_csv(progress_path) if progress_path.is_file() else []
    last = progress[-1] if progress else {}
    geometry_path = run / a.geometry_dir / "geometry_metrics.json"
    geometry = json.loads(geometry_path.read_text(encoding="utf-8-sig")) if geometry_path.is_file() else {}
    status_path = run / "evaluation_status.json"
    evaluation_status = json.loads(status_path.read_text(encoding="utf-8-sig")) if status_path.is_file() else {}
    missing = []
    if not test_rows: missing.append("visual_test: " + str((evaluation_status.get("visual_test") or {}).get("reason", "no test metrics found")))
    if not geometry: missing.append("geometry_test: " + str((evaluation_status.get("geometry_test") or {}).get("reason", "no geometry metrics found")))
    fscores = nested(geometry, "metrics", "fscore") or []
    first_fscore = fscores[0] if fscores else {}
    result = {
        "run_id": cfg.get("run_id", run.name),
        "experiment_group": cfg.get("experiment_group", cfg.get("experiment")),
        "git_commit": cfg.get("git_commit") or cfg.get("code", {}).get("commit"),
        "iterations": a.iterations,
        "test_camera_count": len(test_rows),
        "psnr": mean_metric(test_rows, ("masked_psnr", "psnr")),
        "ssim": mean_metric(test_rows, ("masked_ssim", "ssim", "ssim_zero_mask_full_image")),
        "mae": mean_metric(test_rows, ("masked_mae", "mae")),
        "geometry_unit": geometry.get("unit"),
        "geometry_reference_role": geometry.get("reference_role"),
        "geometry_independent_gt": geometry.get("independent_geometry_gt"),
        "chamfer_l1": number(nested(geometry, "metrics", "chamfer_l1")),
        "chamfer_l2": number(nested(geometry, "metrics", "chamfer_l2")),
        "accuracy_p95": number(nested(geometry, "metrics", "accuracy_pred_to_ref", "p95")),
        "completeness_p95": number(nested(geometry, "metrics", "completeness_ref_to_pred", "p95")),
        "fscore_threshold": number(first_fscore.get("threshold")),
        "fscore": number(first_fscore.get("fscore")),
        "gaussian_count": number(last.get("gaussian_count") or last.get("count")),
        "elapsed_seconds": number(last.get("elapsed_seconds") or last.get("elapsed")),
        "missing_or_skipped_metrics": "; ".join(missing) if missing else None,
    }
    requested = cfg.get("feature_overrides", {}); resolved = cfg.get("resolved_feature_flags", {})
    enabled = sorted(k for k, v in resolved.items() if v is True)
    markdown = [
        f"# 实验 {result['run_id']} 总结", "",
        f"- 实验组：{result['experiment_group']}", f"- Git commit：{result['git_commit']}",
        f"- 目标迭代：{a.iterations}", f"- test 相机数：{len(test_rows)}",
        f"- 已启用功能：{', '.join(enabled) if enabled else '未记录'}", "", "## 独立覆盖项", "",
    ]
    markdown += [f"- `{k}` = `{v}`" for k, v in sorted(requested.items())] or ["- 未记录。"]
    markdown += ["", "## 最终测试集视觉结果", "", "| PSNR | SSIM | MAE |", "|---:|---:|---:|",
                 f"| {fmt(result['psnr'])} | {fmt(result['ssim'])} | {fmt(result['mae'])} |", "",
                 "## 最终几何结果", "",
                 f"- 单位：{result['geometry_unit'] or 'null'}",
                 f"- 参考几何角色：{result['geometry_reference_role'] or 'null'}",
                 f"- 独立几何真值：{result['geometry_independent_gt'] if result['geometry_independent_gt'] is not None else 'null'}",
                 f"- 限制：{geometry.get('limitation') or '未记录'}", "",
                 "| Chamfer-L1 | Chamfer-L2 | Accuracy P95 | Completeness P95 |", "|---:|---:|---:|---:|",
                 f"| {fmt(result['chamfer_l1'])} | {fmt(result['chamfer_l2'])} | {fmt(result['accuracy_p95'])} | {fmt(result['completeness_p95'])} |", ""]
    if fscores:
        markdown += ["| 阈值 | Precision | Recall | F-score |", "|---:|---:|---:|---:|"]
        markdown += [f"| {fmt(number(x.get('threshold')))} | {fmt(number(x.get('precision')))} | {fmt(number(x.get('recall')))} | {fmt(number(x.get('fscore')))} |" for x in fscores]
        markdown.append("")
    markdown += ["## 训练过程", "", f"- 最终高斯数量：{fmt(result['gaussian_count'])}",
                 f"- 训练时间（秒）：{fmt(result['elapsed_seconds'])}",
                 "- 完整曲线保存在 `loss_log.csv`、`training_progress.csv`、`val_metrics.csv` 和 `geometry_metrics.csv`。", "",
                 "## 未运行或缺失的指标", "",
                 *([f"- {item}" for item in missing] if missing else ["- 无。"]), "",
                 "缺失值保持为 `null`；本文件只汇总实际存在的 CSV/JSON，不推断未记录结果。", ""]
    fields = list(result)
    from io import StringIO
    csv_buffer = StringIO(); writer = csv.DictWriter(csv_buffer, fieldnames=fields); writer.writeheader(); writer.writerow(result)
    csv_text = csv_buffer.getvalue()
    md_table = "| " + " | ".join(fields) + " |\n|" + "---|" * len(fields) + "\n| " + " | ".join(str(result[k]) if result[k] is not None else "null" for k in fields) + " |\n"
    tex = "\\begin{tabular}{" + "l" * len(fields) + "}\n" + " & ".join(fields) + " \\\\\n" + " & ".join(str(result[k]) if result[k] is not None else "--" for k in fields) + " \\\\\n\\end{tabular}\n"
    outputs = {"experiment_summary.md": "\n".join(markdown), "results_table.csv": csv_text,
               "results_table.md": md_table, "results_table.tex": tex}
    for name, content in outputs.items():
        path = run / name
        if path.exists() and not a.overwrite: raise FileExistsError(f"Refusing to overwrite {path}")
        path.write_text(content, encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
