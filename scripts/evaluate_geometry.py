#!/usr/bin/env python3
"""Evaluate final predicted points or Gaussian centers against a reference point cloud."""
from __future__ import annotations
import argparse, csv, hashlib, json, math
from pathlib import Path
import numpy as np


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""): h.update(chunk)
    return h.hexdigest()


def load_cloud(path):
    extras = {}
    if path.suffix.lower() == ".npy":
        xyz = np.asarray(np.load(path), dtype=np.float64)
    elif path.suffix.lower() == ".npz":
        data = np.load(path)
        key = "xyz" if "xyz" in data else "points"
        xyz = np.asarray(data[key], dtype=np.float64)
        extras = {name: np.asarray(data[name]) for name in data.files if name != key}
    elif path.suffix.lower() == ".ply":
        try: from plyfile import PlyData
        except ImportError as exc: raise RuntimeError("PLY input requires: pip install plyfile") from exc
        vertex = PlyData.read(str(path))["vertex"].data
        names = set(vertex.dtype.names or ())
        xyz = np.column_stack([vertex[name] for name in ("x", "y", "z")]).astype(np.float64)
        extras = {name: np.asarray(vertex[name]) for name in names - {"x", "y", "z"}}
    else:
        raise ValueError("Supported point clouds: .ply, .npy, .npz")
    if xyz.ndim != 2 or xyz.shape[1] != 3: raise ValueError(f"Expected Nx3 points, got {xyz.shape}")
    valid = np.isfinite(xyz).all(axis=1); xyz = xyz[valid]
    extras = {key: value[valid] for key, value in extras.items() if value.ndim > 0 and value.shape[0] == len(valid)}
    if not len(xyz): raise ValueError(f"No finite points in {path}")
    normals = None
    if all(name in extras for name in ("nx", "ny", "nz")):
        normals = np.column_stack([extras["nx"], extras["ny"], extras["nz"]]).astype(np.float64)
    scales = None
    if all(name in extras for name in ("scale_0", "scale_1", "scale_2")):
        scales = np.column_stack([extras["scale_0"], extras["scale_1"], extras["scale_2"]]).astype(np.float64)
    return xyz, normals, scales


def sample(points, normals, scales, maximum, seed):
    if maximum <= 0 or len(points) <= maximum: return points, normals, scales
    idx = np.random.default_rng(seed).choice(len(points), size=maximum, replace=False)
    return points[idx], None if normals is None else normals[idx], None if scales is None else scales[idx]


def nearest(query, reference):
    try:
        from scipy.spatial import cKDTree
        distance, index = cKDTree(reference).query(query, k=1, workers=-1)
        return np.asarray(distance), np.asarray(index)
    except ImportError:
        if len(query) * len(reference) > 20_000_000:
            raise RuntimeError("Large geometry evaluation requires: pip install scipy")
        best_d = np.empty(len(query), dtype=np.float64); best_i = np.empty(len(query), dtype=np.int64)
        for start in range(0, len(query), 1024):
            block = query[start:start + 1024]
            squared = ((block[:, None, :] - reference[None, :, :]) ** 2).sum(axis=2)
            idx = squared.argmin(axis=1); best_i[start:start + len(block)] = idx
            best_d[start:start + len(block)] = np.sqrt(squared[np.arange(len(block)), idx])
        return best_d, best_i


def distance_stats(values):
    return {"mean": float(values.mean()), "median": float(np.median(values)),
            "p90": float(np.quantile(values, .90)), "p95": float(np.quantile(values, .95)),
            "rmse": float(np.sqrt(np.mean(values ** 2))), "max": float(values.max())}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prediction", type=Path, required=True)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--prediction-type", choices=("gaussian_centers", "surface_points"), default="gaussian_centers")
    p.add_argument("--reference-role", choices=("initialization_lidar", "heldout_lidar", "mesh_gt", "other"), required=True)
    p.add_argument("--unit", choices=("meters", "scene"), required=True)
    p.add_argument("--thresholds", type=float, nargs="+", default=(.02, .04, .08))
    p.add_argument("--max-points", type=int, default=500000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--linear-scales", action="store_true", help="scale_0..2 are linear; default assumes Gaussian log-scales")
    p.add_argument("--max-scale-threshold", type=float)
    p.add_argument("--alignment-description", default="already aligned; no transform applied")
    a = p.parse_args(argv)
    if a.output.exists(): raise FileExistsError(f"Refusing to overwrite {a.output}")
    if any(x <= 0 for x in a.thresholds): p.error("thresholds must be positive")
    pred, pred_n, scales = load_cloud(a.prediction); ref, ref_n, _ = load_cloud(a.reference)
    original_counts = {"prediction": len(pred), "reference": len(ref)}
    pred, pred_n, scales = sample(pred, pred_n, scales, a.max_points, a.seed)
    ref, ref_n, _ = sample(ref, ref_n, None, a.max_points, a.seed + 1)
    accuracy, pred_to_ref = nearest(pred, ref); completeness, _ = nearest(ref, pred)
    metrics = {
        "accuracy_pred_to_ref": distance_stats(accuracy),
        "completeness_ref_to_pred": distance_stats(completeness),
        "chamfer_l1": float(.5 * (accuracy.mean() + completeness.mean())),
        "chamfer_l2": float(.5 * (np.mean(accuracy ** 2) + np.mean(completeness ** 2))),
        "fscore": [],
    }
    for threshold in sorted(set(a.thresholds)):
        precision = float(np.mean(accuracy <= threshold)); recall = float(np.mean(completeness <= threshold))
        fscore = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        metrics["fscore"].append({"threshold": threshold, "precision": precision, "recall": recall, "fscore": fscore})
    if pred_n is not None and ref_n is not None:
        pn = pred_n / np.clip(np.linalg.norm(pred_n, axis=1, keepdims=True), 1e-12, None)
        rn = ref_n / np.clip(np.linalg.norm(ref_n, axis=1, keepdims=True), 1e-12, None)
        dot = np.einsum("ij,ij->i", pn, rn[pred_to_ref])
        metrics["normal_consistency"] = {"oriented_mean": float(dot.mean()), "absolute_mean": float(np.abs(dot).mean())}
    if scales is not None:
        scales = scales if a.linear_scales else np.exp(scales)
        smin = scales.min(axis=1); smax = scales.max(axis=1); aspect = smax / np.clip(smin, 1e-12, None)
        metrics["gaussian_shape"] = {
            "min_scale_median": float(np.median(smin)), "min_scale_p95": float(np.quantile(smin, .95)),
            "max_scale_median": float(np.median(smax)), "max_scale_p95": float(np.quantile(smax, .95)),
            "max_scale_max": float(smax.max()), "aspect_ratio_median": float(np.median(aspect)),
            "aspect_ratio_p95": float(np.quantile(aspect, .95)),
        }
        if a.max_scale_threshold is not None:
            metrics["gaussian_shape"]["fraction_over_max_scale_threshold"] = float(np.mean(smax > a.max_scale_threshold))
    report = {
        "schema_version": "3dgs-experiment-standard/1", "unit": a.unit,
        "prediction_type": a.prediction_type, "reference_role": a.reference_role,
        "independent_geometry_gt": a.reference_role != "initialization_lidar",
        "limitation": "Initialization LiDAR measures adherence to the seed geometry, not independent reconstruction accuracy." if a.reference_role == "initialization_lidar" else None,
        "alignment": a.alignment_description,
        "inputs": {"prediction": {"path": str(a.prediction.resolve()), "sha256": sha256(a.prediction), "count": original_counts["prediction"]},
                   "reference": {"path": str(a.reference.resolve()), "sha256": sha256(a.reference), "count": original_counts["reference"]}},
        "sampled_counts": {"prediction": len(pred), "reference": len(ref)}, "metrics": metrics,
    }
    a.output.mkdir(parents=True, exist_ok=False)
    (a.output / "geometry_metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    flat = {"unit": a.unit, "prediction_type": a.prediction_type, "reference_role": a.reference_role,
            "accuracy_mean": metrics["accuracy_pred_to_ref"]["mean"], "accuracy_p95": metrics["accuracy_pred_to_ref"]["p95"],
            "completeness_mean": metrics["completeness_ref_to_pred"]["mean"], "completeness_p95": metrics["completeness_ref_to_pred"]["p95"],
            "chamfer_l1": metrics["chamfer_l1"], "chamfer_l2": metrics["chamfer_l2"]}
    for item in metrics["fscore"]: flat[f"fscore_{item['threshold']:g}"] = item["fscore"]
    with (a.output / "geometry_metrics.csv").open("x", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(flat)); writer.writeheader(); writer.writerow(flat)
    (a.output / "manifest.json").write_text(json.dumps({"inputs": report["inputs"], "sampled_counts": report["sampled_counts"], "alignment": report["alignment"]}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))

if __name__ == "__main__": main()
