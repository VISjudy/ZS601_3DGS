#!/usr/bin/env python3
"""Stage-based ZS601 dataset preprocessing entry point."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from preprocess_v3 import (  # noqa: E402
    NormalSettings,
    generate_supervision,
    load_cameras,
    prepare_sfm,
    preprocess_lidar,
    read_name_list,
    read_ply,
    validate_processed,
)
from experiment_presets_v3 import PREPROCESS_FEATURES, resolve_experiment_config  # noqa: E402


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Preprocess ZS601 SfM and static LiDAR inputs")
    result.add_argument("--stage", choices=("sfm", "lidar", "supervision", "validate", "all"), required=True)
    result.add_argument("--experiment-group", choices=("original", "A", "B", "C", "D", "E", "custom"),
                        default="custom")
    result.add_argument("--scene-root", type=Path, required=True)
    result.add_argument("--output", type=Path, help="Defaults to <scene-root>/processed_v3")
    result.add_argument("--images", type=Path, help="Raw perspective image directory")
    result.add_argument("--masks", type=Path, help="Raw mask directory; white means valid")
    result.add_argument("--sparse", type=Path, help="Raw COLMAP sparse/0 directory")
    result.add_argument("--lidar", type=Path, help="Dynamic-object-removed colored LiDAR PLY")
    result.add_argument("--val-list", type=Path, help="Fixed validation list; otherwise inherit sparse/0/images-val10.txt")
    result.add_argument("--train-list", type=Path, help="Fixed training list; otherwise inherit sparse/0/images-train.txt")
    result.add_argument("--test-list", type=Path, help="Fixed test list; otherwise inherit sparse/0/images-test.txt")
    result.add_argument("--mask-valid-when", choices=("white", "black"), default="white")
    result.add_argument("--overwrite", action="store_true", help="Explicitly allow replacing generated files")
    result.add_argument("--knn", type=int, default=24)
    result.add_argument("--max-radius", type=float)
    result.add_argument("--min-neighbors", type=int, default=8)
    result.add_argument("--max-curvature", type=float, default=.15)
    result.add_argument("--camera-count", type=int, default=8)
    result.add_argument("--min-orientation-confidence", type=float, default=.05)
    result.add_argument("--estimate-normals", choices=("auto", "on", "off"), default="auto")
    result.add_argument("--orient-normals-camera", choices=("auto", "on", "off"), default="auto")
    result.add_argument("--propagate-unresolved", choices=("auto", "on", "off"), default="auto")
    result.add_argument("--filter-outliers", choices=("auto", "on", "off"), default="auto")
    result.add_argument("--outlier-neighbors", type=int, default=16)
    result.add_argument("--outlier-sigma", type=float, default=3.0)
    result.add_argument("--orientation-occlusion", choices=("auto", "on", "off"), default="auto")
    result.add_argument("--orientation-block-size", type=int, default=250000)
    result.add_argument("--near", type=float, default=.01)
    result.add_argument("--far", type=float, default=1e6)
    result.add_argument("--backprojection-samples", type=int, default=3)
    result.add_argument("--generate-supervision", choices=("auto", "on", "off"), default="auto")
    return result


def _required(value: Path | None, option: str) -> Path:
    if value is None:
        raise SystemExit(f"{option} is required for this stage")
    value = value.resolve()
    if not value.exists():
        raise SystemExit(f"Input does not exist: {value}")
    return value


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    scene_root = args.scene_root.resolve()
    if not scene_root.exists():
        raise SystemExit(f"Scene root does not exist: {scene_root}")
    output = (args.output or scene_root / "processed_v3").resolve()
    if output == scene_root:
        raise SystemExit("Output must be a dedicated directory, not the scene root")
    raw_root = (scene_root / "raw").resolve()
    if output == raw_root or raw_root in output.parents:
        raise SystemExit(f"Output must not be inside the read-only raw directory: {raw_root}")
    for candidate in (args.images, args.masks, args.sparse):
        if candidate is not None:
            raw_input = candidate.resolve()
            if output == raw_input or raw_input in output.parents:
                raise SystemExit(f"Output must not be inside raw input directory: {raw_input}")
    if args.near <= 0 or args.far <= args.near or args.backprojection_samples < 0:
        raise SystemExit("Require 0 < near < far and backprojection_samples >= 0")
    output.mkdir(parents=True, exist_ok=True)
    raw_overrides = {
        "lidar_outlier_filter": args.filter_outliers,
        "pca_normal_estimation": args.estimate_normals,
        "camera_normal_orientation": args.orient_normals_camera,
        "neighbor_normal_propagation": args.propagate_unresolved,
        "supervision_generation": args.generate_supervision,
        "visibility_occlusion_check": args.orientation_occlusion,
    }
    feature_params = {
        "lidar_outlier_filter": {"neighbors": args.outlier_neighbors, "sigma": args.outlier_sigma},
        "pca_normal_estimation": {"knn": args.knn, "max_radius": args.max_radius,
                                  "min_neighbors": args.min_neighbors, "max_curvature": args.max_curvature},
        "camera_normal_orientation": {"camera_count": args.camera_count,
                                       "min_confidence": args.min_orientation_confidence,
                                       "block_size": args.orientation_block_size},
        "neighbor_normal_propagation": {"neighbors": 8},
        "supervision_generation": {"near": args.near, "far": args.far,
                                    "backprojection_samples": args.backprojection_samples,
                                    "mask_valid_when": args.mask_valid_when},
        "visibility_occlusion_check": {"method": "per_pixel_nearest_z"},
    }
    experiment = resolve_experiment_config(
        args.experiment_group, raw_overrides, feature_params, PREPROCESS_FEATURES)
    flags = experiment["resolved_feature_flags"]
    experiment["all_feature_params"] = feature_params
    experiment["inactive_feature_params"] = {
        name: values for name, values in feature_params.items() if not flags[name]}
    experiment["stage"] = args.stage
    experiment["paths"] = {"scene_root": str(scene_root), "output": str(output),
                           "images": str(args.images.resolve()) if args.images else None,
                           "masks": str(args.masks.resolve()) if args.masks else None,
                           "sparse": str(args.sparse.resolve()) if args.sparse else None,
                           "lidar": str(args.lidar.resolve()) if args.lidar else None}
    settings = NormalSettings(
        knn=args.knn, max_radius=args.max_radius, min_neighbors=args.min_neighbors,
        max_curvature=args.max_curvature, camera_count=args.camera_count,
        min_orientation_confidence=args.min_orientation_confidence,
        orient_camera=flags["camera_normal_orientation"],
        propagate_unresolved=flags["neighbor_normal_propagation"],
        filter_outliers=flags["lidar_outlier_filter"],
        outlier_neighbors=args.outlier_neighbors, outlier_sigma=args.outlier_sigma,
        orientation_occlusion=flags["visibility_occlusion_check"],
        orientation_block_size=args.orientation_block_size,
    )
    settings.validate()
    manifest_path = output / "manifests" / f"preprocess_run_config_{args.stage}.json"
    if manifest_path.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite existing run config: {manifest_path}")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(experiment, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Resolved preprocessing experiment configuration:")
    print(json.dumps(experiment, ensure_ascii=False, indent=2))
    stages = ["sfm", "lidar", "supervision", "validate"] if args.stage == "all" else [args.stage]
    result: dict[str, object] = {"scene_root": str(scene_root), "output": str(output),
                                 "experiment": experiment, "stages": {}}
    for stage in stages:
        if stage == "sfm":
            result["stages"][stage] = prepare_sfm(
                _required(args.images, "--images"), args.masks.resolve() if args.masks else None,
                _required(args.sparse, "--sparse"), output,
                args.val_list.resolve() if args.val_list else None,
                args.train_list.resolve() if args.train_list else None,
                args.test_list.resolve() if args.test_list else None,
                args.mask_valid_when, args.overwrite)
        elif stage == "lidar":
            cameras = []
            if flags["camera_normal_orientation"]:
                sparse = output / "sparse" / "0"
                train_names = read_name_list(sparse / "images-train.txt")
                if not train_names:
                    raise SystemExit("Camera normal orientation requires processed sparse/0/images-train.txt")
                cameras = load_cameras(sparse, train_names)
            result["stages"][stage] = preprocess_lidar(
                _required(args.lidar, "--lidar"), output, settings, cameras,
                output / "masks" if (output / "masks").exists() else None,
                args.mask_valid_when, args.overwrite, flags["pca_normal_estimation"])
        elif stage == "supervision":
            if not flags["supervision_generation"]:
                result["stages"][stage] = {"status": "skipped", "reason": "supervision_generation is off"}
                continue
            sparse = output / "sparse" / "0"
            cameras = load_cameras(sparse)
            points, _, extras = read_ply(output / "geometry" / "lidar_static_rgb_normal_oriented.ply")
            required = {"nx", "ny", "nz", "normal_valid"}
            missing = required - extras.keys()
            if missing:
                raise SystemExit(f"Oriented PLY is missing fields: {sorted(missing)}")
            normals = np.column_stack([extras["nx"], extras["ny"], extras["nz"]])
            result["stages"][stage] = generate_supervision(
                points, normals, extras["normal_valid"].astype(bool), cameras, output,
                output / "images", output / "masks" if (output / "masks").exists() else None,
                args.near, args.far, args.mask_valid_when, args.backprojection_samples, args.overwrite)
        else:
            report, errors = validate_processed(output, args.overwrite)
            result["stages"][stage] = report
            if errors:
                print(json.dumps(result, ensure_ascii=False, indent=2))
                return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
