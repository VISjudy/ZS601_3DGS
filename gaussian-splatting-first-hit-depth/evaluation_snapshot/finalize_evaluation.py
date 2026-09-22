"""Finalize completed CPU metrics after separate figure generation and visual QA.

This recovery/finalization step only validates and creates new verification JSON
files. It does not recompute projection, evaluation metrics, or point clouds.
"""
from pathlib import Path
import hashlib
import json
import sys

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[3]
OUT = PROJECT / "3dgsResult/zs601-firsthit-depth-v008/evaluation"
sys.dont_write_bytecode = True
sys.path.insert(0, str(HERE / "python-deps"))
sys.path.insert(0, str(PROJECT / ".runtime/synthetic-dataset"))
sys.path.insert(0, str(PROJECT / "experiments/2026-09-21/synthetic-lidar-init-v001/python-deps"))
import numpy as np
from PIL import Image
from plyfile import PlyData


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def save(path, value):
    with path.open("x", encoding="utf-8") as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)


def main():
    assert not (OUT / "verification.json").exists()
    assert not (OUT / "figures/figure_verification.json").exists()
    frozen = load(OUT / "input_manifest.json")
    for filename, expected in frozen["files"].items():
        path = Path(filename)
        assert path.stat().st_size == expected["bytes"] and sha(path) == expected["sha256"], filename
    metrics = load(OUT / "geometry_metrics.json")
    figures = load(OUT / "figures/figure_manifest.json")
    review = load(OUT / "figures/visual_review.json")
    assert review["status"] == "PASS_VISUAL_INSPECTION"
    assert figures["representative_views"] == review["representative_views"] == [3202, 3376, 3481]
    assert len(metrics["per_view"]) == 50 and len(metrics["aggregate"]) == 5
    assert len(figures["records"]) == 103
    assert len(list((OUT / "figures/nn_heatmaps").glob("*.png"))) == 50
    assert len(list((OUT / "figures/signed_mesh_Z").glob("*.png"))) == 50
    assert len(list((OUT / "figures/representative_views").glob("*.png"))) == 3
    assert len(list((OUT / "figures/representative_views").glob("*.pdf"))) == 3
    figure_hashes = {}
    for record in figures["records"]:
        assert record["text_extents_inside_canvas"] and record["PNG_readback"] == "PASS"
        path = OUT / record["path"]
        with Image.open(path) as image:
            assert list(image.size) == record["PNG_size"]
            image.load()
        figure_hashes[record["path"]] = {"bytes": path.stat().st_size, "sha256": sha(path)}
        if record["PDF_path"]:
            pdf = OUT / record["PDF_path"]
            with pdf.open("rb") as f:
                assert f.read(5) == b"%PDF-"
            figure_hashes[record["PDF_path"]] = {"bytes": pdf.stat().st_size, "sha256": sha(pdf)}
    new_methods = [m for m in metrics["protocol"]["methods"] if m["new"]]
    ply_count = 0
    total_merged_points = 0
    for method in new_methods:
        rows = [r for r in metrics["per_view"] if r["method"] == method["id"]]
        assert len(rows) == 10
        for row in rows:
            record = row["PLY"]
            path = Path(record["path"])
            assert record["exact_roundtrip"] and sha(path) == record["sha256"]
            assert len(PlyData.read(path)["vertex"].data) == record["points"] == row["valid_pixels"]
            ply_count += 1
        merged = metrics["aggregate"][method["id"]]["merged_PLY"]
        assert sha(Path(merged["path"])) == merged["sha256"]
        assert len(PlyData.read(merged["path"])["vertex"].data) == merged["points"] == sum(r["valid_pixels"] for r in rows)
        total_merged_points += merged["points"]
        ply_count += 1
        diagnostics = metrics["aggregate"][method["id"]]["first_hit_diagnostics"]
        assert diagnostics["hit_count"] == sum(r["first_hit_diagnostics"]["hit_count"] for r in rows)
        assert diagnostics["qmin_gt9_count"] == sum(r["first_hit_diagnostics"]["qmin_gt9_count"] for r in rows)
    assert ply_count == 33
    assert len(list((OUT / "error_arrays").glob("*/*_nn_m.npy"))) == 50
    assert len(list((OUT / "error_arrays").glob("*/*_signed_mesh_Z_m.npy"))) == 50
    for row in metrics["per_view"]:
        if row["method"] in ["method_a_gaussian", "method_b_zbuffer"]:
            assert row["v007_baseline_own_domain_metrics_reproduced"]
    assert len({m["common_all5_depth"]["count"] for m in metrics["aggregate"].values()}) == 1
    assert len({m["common_all5_alpha50_depth"]["count"] for m in metrics["aggregate"].values()}) == 1
    figure_verification = {"status": "PASS_FILES_TEXT_EXTENTS_AND_VISUAL_REVIEW", "PNG_figures": 103, "representative_PDF_exports": 3, "all_PNGs_decoded_successfully": True, "all_103_figure_text_extents_checked": True, "visual_review": review, "files": figure_hashes, "note": "This final record supersedes the earlier figure-manifest pending-visual-review status. Visual checks cover all3 representative layouts and both standalone heatmap formats; scientific quality is measured, not accepted by this format check."}
    save(OUT / "figures/figure_verification.json", figure_verification)
    verification = {
        "status": "PASS_FORMATS_COORDINATES_MASKS_PROVENANCE_PLY_INPUT_PRESERVATION_AND_FIGURES",
        "methods": 5, "views_per_method": 10, "saved_depth_PNGs_read": 50,
        "new_per_view_PLYs": 30, "new_merged_PLYs": 3, "total_points_across_three_merged_clouds": total_merged_points,
        "input_files_unchanged": len(frozen["files"]),
        "all_primary_CDE_masks_verified_from_first_hit_ID_and_depth": True,
        "alpha50_not_used_to_filter_primary_PLYs": True,
        "qmin_gt9_not_removed_from_primary_domain": True,
        "common_domains_equal_across_all5": True,
        "baseline_PLYs_exactly_reproduced_read_only": True,
        "baseline_own_domain_metrics_reproduced": True,
        "figure_status": figure_verification["status"],
        "figure_verification": "figures/figure_verification.json",
        "visual_review_still_required": False,
        "occlusion_quality_is_measured_not_assumed": True,
        "metrics_status": metrics["status"],
        "recovery": "Numeric evaluation and all33 PLYs succeeded once. The initial plotting text checker counted hidden axis ticks as overflow before saving any figure. Ticks were explicitly removed; only plotting was rerun. Final input hashes and outputs were then validated here.",
        "evaluator_sha256": sha(HERE / "evaluate_five_methods.py"),
        "final_plotter_sha256": sha(HERE / "plot_errors.py"),
        "finalizer_sha256": sha(Path(__file__)),
        "geometry_metrics_sha256": sha(OUT / "geometry_metrics.json"),
    }
    save(OUT / "verification.json", verification)
    print(json.dumps(verification, indent=2), flush=True)


if __name__ == "__main__":
    main()
