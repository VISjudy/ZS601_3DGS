#!/usr/bin/env python3
"""Generate concise Markdown/CSV/LaTeX summaries from real run artifacts."""
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


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run", type=Path)
    p.add_argument("--iterations", type=int, default=150000)
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args(argv); run = a.run.resolve()
    cfg_path = run / "run_config.json"
    if not cfg_path.is_file(): raise FileNotFoundError(cfg_path)
    cfg = json.loads(cfg_path.read_text(encoding="utf-8-sig"))
    test_path = find_test_csv(run, a.iterations); test_rows = read_csv(test_path) if test_path else []
    progress_path = run / "training_progress.csv"; progress = read_csv(progress_path) if progress_path.is_file() else []
    last = progress[-1] if progress else {}
    result = {
        "run_id": cfg.get("run_id", run.name),
        "experiment_group": cfg.get("experiment_group", cfg.get("experiment")),
        "git_commit": cfg.get("git_commit") or cfg.get("code", {}).get("commit"),
        "iterations": a.iterations,
        "psnr": mean_metric(test_rows, ("masked_psnr", "psnr")),
        "ssim": mean_metric(test_rows, ("masked_ssim", "ssim", "ssim_zero_mask_full_image")),
        "mae": mean_metric(test_rows, ("masked_mae", "mae")),
        "gaussian_count": number(last.get("gaussian_count") or last.get("count")),
        "elapsed_seconds": number(last.get("elapsed_seconds") or last.get("elapsed")),
    }
    requested = cfg.get("feature_overrides", {}); resolved = cfg.get("resolved_feature_flags", {})
    enabled = sorted(k for k, v in resolved.items() if v is True)
    markdown = [
        f"# 实验 {result['run_id']} 总结", "",
        f"- 实验组：{result['experiment_group']}", f"- Git commit：{result['git_commit']}",
        f"- 目标迭代：{a.iterations}", f"- test 相机数：{len(test_rows)}",
        f"- 已启用功能：{', '.join(enabled) if enabled else '未记录'}", "",
        "## 独立覆盖项", "",
    ]
    markdown += [f"- `{k}` = `{v}`" for k, v in sorted(requested.items())] or ["- 未记录。"]
    markdown += ["", "## 最终结果", "", "| PSNR | SSIM | MAE | 高斯数量 | 训练秒数 |",
                 "|---:|---:|---:|---:|---:|",
                 f"| {fmt(result['psnr'])} | {fmt(result['ssim'])} | {fmt(result['mae'])} | {fmt(result['gaussian_count'])} | {fmt(result['elapsed_seconds'])} |", "",
                 "缺失值保持为 `null`；本文件只汇总实际存在的 CSV/JSON，不推断未记录结果。", ""]
    table_fields = list(result)
    csv_text = ",".join(table_fields) + "\n" + ",".join(str(result[k]) if result[k] is not None else "" for k in table_fields) + "\n"
    md_table = "| " + " | ".join(table_fields) + " |\n|" + "---|" * len(table_fields) + "\n| " + " | ".join(str(result[k]) if result[k] is not None else "null" for k in table_fields) + " |\n"
    tex = "\\begin{tabular}{" + "l" * len(table_fields) + "}\n" + " & ".join(table_fields) + " \\\\\n" + " & ".join(str(result[k]) if result[k] is not None else "--" for k in table_fields) + " \\\\\n\\end{tabular}\n"
    outputs = {"experiment_summary.md": "\n".join(markdown), "results_table.csv": csv_text,
               "results_table.md": md_table, "results_table.tex": tex}
    for name, content in outputs.items():
        path = run / name
        if path.exists() and not a.overwrite: raise FileExistsError(f"Refusing to overwrite {path}")
        path.write_text(content, encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == "__main__": main()
