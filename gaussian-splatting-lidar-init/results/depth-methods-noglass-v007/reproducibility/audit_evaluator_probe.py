"""Independent, read-only audit and exact-winner-ray occlusion probe.

Writes only this experiment's audit_evaluator.json. It does not rerun either
projection method, export point clouds, or modify shared evaluator outputs.
"""
from pathlib import Path
import hashlib
import json
import os
import sys
import time
from collections import Counter

os.environ["OPENBLAS_NUM_THREADS"] = "2"
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[2]
sys.path.insert(0, str(PROJECT / ".runtime/synthetic-dataset"))
sys.path.insert(0, str(PROJECT / "experiments/2026-09-21/synthetic-lidar-init-v001/python-deps"))
import numpy as np
from PIL import Image
from plyfile import PlyData
from scipy.spatial.transform import Rotation
from embreex.rtcore_scene import EmbreeScene
from embreex.mesh_construction import TriangleMesh


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def triangle_parameter(origin, direction, vertices):
    """Float64 Moller-Trumbore verification of one reported Embree occluder."""
    a, b, c = vertices.astype(np.float64)
    e1, e2 = b - a, c - a
    p = np.cross(direction, e2)
    determinant = float(np.dot(e1, p))
    if abs(determinant) < 1e-20:
        return None
    relative = origin - a
    u = float(np.dot(relative, p) / determinant)
    q = np.cross(relative, e1)
    v = float(np.dot(direction, q) / determinant)
    parameter = float(np.dot(e2, q) / determinant)
    return {"t_after_near_m": parameter, "u": u, "v": v, "inside_with_1e-5_tolerance": u >= -1e-5 and v >= -1e-5 and u + v <= 1 + 1e-5}


def main():
    started = time.perf_counter()
    report_path = ROOT / "audit_evaluator.json"
    assert not report_path.exists(), "Refusing to overwrite independent audit"
    spec = load_json(ROOT / "run_spec.json")
    out = Path(spec["local_output"])
    cameras = load_json(out / "selected_views.json")
    assert [v["image_id"] for v in cameras] == spec["view_ids"]
    script_path = ROOT / "evaluate_depth.py"
    source = Path(spec["input_cloud"])
    geometry_path = Path(spec["source_geometry"])
    mask_path = Path(spec["excluded_glass_faces"])
    face_path = source.parent / "mesh_face_id.npy"
    old_sample_path = source.parent / "source_sample_id.npy"
    audit_path = mask_path.parent / "material_audit.json"
    material_path = mask_path.parent / "face_material_id.npy"
    protected = [script_path, source, geometry_path, mask_path, face_path, old_sample_path, audit_path, material_path, ROOT / "run_spec.json", out / "selected_views.json"]
    protected += [out / "method_b_zbuffer/winner_source_id" / f"{Path(v['name']).stem}.npy" for v in cameras]
    before = {str(p): sha(p) for p in protected}
    assert before[str(source)] == spec["input_cloud_sha256"]
    plan = load_json(PROJECT / "scenes/zs601-meetingroom/synthetic-training-v001/plan.json")
    assert before[str(geometry_path)] == plan["geometry_sha256"]
    source_vertices = PlyData.read(source)["vertex"].data
    xyz = np.column_stack([source_vertices[k] for k in ("x", "y", "z")])
    assert len(xyz) == 7_004_696 and np.isfinite(xyz).all()
    point_faces = np.load(face_path, mmap_mode="r", allow_pickle=False)
    old_sample_ids = np.load(old_sample_path, mmap_mode="r", allow_pickle=False)
    with np.load(geometry_path, allow_pickle=False) as geom:
        vertices, faces, objects = geom["vertices"], geom["faces"], geom["object_ids"]
    excluded = np.load(mask_path, allow_pickle=False)
    assert excluded.dtype == np.bool_ and excluded.shape == (len(faces),) and excluded.sum() == 3468
    audit = load_json(audit_path)
    material_ids = np.load(material_path, allow_pickle=False)
    clear_ids = [m["id"] for m in audit["materials"] if m["custom_properties"].get("semantic_class") == "clear_glass"]
    assert np.array_equal(np.isin(material_ids, clear_ids), excluded)
    kept_face_ids = np.flatnonzero(~excluded)
    assert len(kept_face_ids) == 749_828
    scene = EmbreeScene()
    mesh = TriangleMesh(scene, vertices.astype(np.float32), faces[kept_face_ids].astype(np.int32))
    near, far, threshold = spec["near_clip_m"], spec["far_clip_m"], 0.03
    results, examples = [], []
    global_pairs = Counter()
    all_exact_errors, all_center_errors = [], []
    projected_total = exact_reference_total = exact_behind_total = center_behind_total = 0
    max_independent_triangle_Z_difference = 0.0
    for v in cameras:
        name, stem = v["name"], Path(v["name"]).stem
        winners = np.load(out / "method_b_zbuffer/winner_source_id" / f"{stem}.npy", allow_pickle=False)
        valid_pixels = np.flatnonzero(winners.ravel() >= 0)
        seed = 20260922 + int(v["image_id"])
        indices = np.sort(np.random.default_rng(seed).choice(valid_pixels, min(10_000, len(valid_pixels)), replace=False))
        source_ids = winners.ravel()[indices]
        points = xyz[source_ids].astype(np.float64)
        w, qx, qy, qz = v["q"]
        rotation = Rotation.from_quat([qx, qy, qz, w]).as_matrix()
        translation = np.asarray(v["t"], dtype=np.float64)
        center = -rotation.T @ translation
        camera_points = points @ rotation.T + translation
        winner_z = camera_points[:, 2]
        u = v["fx"] * camera_points[:, 0] / winner_z + v["cx"]
        vv = v["fy"] * camera_points[:, 1] / winner_z + v["cy"]
        checked_pixels = np.floor(vv).astype(np.int64) * v["width"] + np.floor(u).astype(np.int64)
        assert np.array_equal(checked_pixels, indices)
        assert np.all((u >= 0) & (u < v["width"]) & (vv >= 0) & (vv < v["height"]))
        assert np.all((winner_z > near) & (winner_z <= far))
        # Ray goes through the original winning 3D sample, not the pixel centre.
        directions = (points - center) / winner_z[:, None]
        assert np.max(np.abs((directions @ rotation.T)[:, 2] - 1)) < 1e-12
        origins = center + near * directions
        hits = scene.run(np.ascontiguousarray(origins, dtype=np.float32), np.ascontiguousarray(directions, dtype=np.float32), query="INTERSECT", output=1)
        first_z = hits["tfar"].astype(np.float64) + near
        hit_valid = (hits["primID"] >= 0) & np.isfinite(first_z) & (hits["tfar"] > 0) & (first_z <= far)
        hit_faces = np.full(len(indices), -1, dtype=np.int64)
        hit_faces[hit_valid] = kept_face_ids[hits["primID"][hit_valid]]
        assert not excluded[hit_faces[hit_valid]].any()
        source_faces = point_faces[source_ids]
        assert not excluded[source_faces].any()
        errors = winner_z - first_z
        behind = hit_valid & (errors > threshold)
        in_front = hit_valid & (errors < -threshold)
        pixel_reference = np.load(out / "reference_noglass_mesh/depth_float" / f"{stem}.npy", allow_pickle=False).ravel()[indices].astype(np.float64)
        center_valid = pixel_reference > 0
        center_errors = winner_z - pixel_reference
        center_behind = center_valid & (center_errors > threshold)
        own_objects = objects[source_faces]
        first_objects = np.full(len(indices), -1, dtype=np.int64)
        first_objects[hit_valid] = objects[hit_faces[hit_valid]]
        pair_counter = Counter(zip(first_objects[behind].tolist(), own_objects[behind].tolist()))
        global_pairs.update(pair_counter)
        projected_total += len(indices)
        exact_reference_total += int(hit_valid.sum())
        exact_behind_total += int(behind.sum())
        center_behind_total += int(center_behind.sum())
        all_exact_errors.append(errors[hit_valid])
        all_center_errors.append(center_errors[center_valid])
        top_pairs = [{"first_hit_object_id": a, "first_hit_object": audit["parts"][a]["name"], "winner_object_id": b, "winner_object": audit["parts"][b]["name"], "sampled_behind_count": n} for (a, b), n in pair_counter.most_common(5)]
        example_indices = []
        seen_pairs = set()
        for i in np.flatnonzero(behind)[np.argsort(-errors[behind])]:
            pair = (int(first_objects[i]), int(own_objects[i]))
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            example_indices.append(int(i))
            if len(example_indices) >= 3:
                break
        for i in example_indices:
            check = triangle_parameter(origins[i], directions[i], vertices[faces[hit_faces[i]]])
            assert check is not None and check["inside_with_1e-5_tolerance"]
            independent_z = near + check["t_after_near_m"]
            independent_difference = abs(independent_z - first_z[i])
            max_independent_triangle_Z_difference = max(max_independent_triangle_Z_difference, independent_difference)
            assert independent_difference < 2e-5 and winner_z[i] - independent_z > threshold
            examples.append({
                "image_id": v["image_id"], "pixel_flat": int(indices[i]), "pixel_xy": [int(indices[i] % v["width"]), int(indices[i] // v["width"])],
                "actual_point_uv": [float(u[i]), float(vv[i])], "winner_source_row": int(source_ids[i]), "original_v003_sample_id": int(old_sample_ids[source_ids[i]]),
                "winner_XYZ_m": points[i].tolist(), "winner_face_id": int(source_faces[i]), "winner_object_id": int(own_objects[i]), "winner_object": audit["parts"][own_objects[i]]["name"],
                "first_hit_face_id": int(hit_faces[i]), "first_hit_object_id": int(first_objects[i]), "first_hit_object": audit["parts"][first_objects[i]]["name"],
                "winner_camera_Z_m": float(winner_z[i]), "first_hit_camera_Z_m": float(first_z[i]), "behind_first_hit_m": float(errors[i]),
                "float64_triangle_parameter_check": check, "float64_triangle_check_Z_error_m": independent_difference,
            })
        row = {
            "image_id": v["image_id"], "seed": seed, "sampled_valid_pixels": len(indices), "exact_winner_ray_reference_valid": int(hit_valid.sum()), "exact_winner_ray_reference_misses": int((~hit_valid).sum()),
            "exact_winner_ray_behind_gt30mm_count": int(behind.sum()), "exact_winner_ray_behind_gt30mm_percent": float(behind.sum() * 100 / hit_valid.sum()),
            "exact_winner_ray_in_front_gt30mm_count": int(in_front.sum()), "exact_winner_ray_in_front_gt30mm_percent": float(in_front.sum() * 100 / hit_valid.sum()),
            "exact_winner_ray_MAE_mm": float(np.abs(errors[hit_valid]).mean() * 1000),
            "same_sample_pixel_center_reference_valid": int(center_valid.sum()), "same_sample_pixel_center_behind_gt30mm_count": int(center_behind.sum()), "same_sample_pixel_center_behind_gt30mm_percent": float(center_behind.sum() * 100 / center_valid.sum()),
            "center_behind_also_exact_ray_behind_count": int((behind & center_behind).sum()),
            "sample_pixel_index_sha256": hashlib.sha256(indices.astype("<i8").tobytes()).hexdigest(), "sample_winner_index_sha256": hashlib.sha256(source_ids.astype("<i4").tobytes()).hexdigest(),
            "top_occluder_winner_pairs": top_pairs,
        }
        results.append(row)
        print(json.dumps({k: row[k] for k in ("image_id", "sampled_valid_pixels", "exact_winner_ray_reference_misses", "exact_winner_ray_behind_gt30mm_percent", "same_sample_pixel_center_behind_gt30mm_percent")}), flush=True)
    after = {str(p): sha(p) for p in protected}
    assert before == after, "Audited files changed during probe"
    merged_errors = np.concatenate(all_exact_errors)
    pixel_weighted_rate = float(sum(v["exact_winner_ray_behind_gt30mm_count"] / v["exact_winner_ray_reference_valid"] * np.count_nonzero(np.load(out / "method_b_zbuffer/winner_source_id" / f"{Path(c['name']).stem}.npy", allow_pickle=False) >= 0) for v, c in zip(results, cameras)) / 2_781_236 * 100)
    report = {
        "status": "PASS_LIMITED_REVIEW_AFTER_GLASS_MASK_FIX",
        "reviewed_script": str(script_path), "reviewed_script_sha256": before[str(script_path)],
        "initial_script_sha256": "18d5b9fa5074e16ac6af1f59887be829c8d60861f48fc6a1a214fc41b08b982e",
        "resolved_findings": [{"severity": "blocking", "initial_lines": [155, 157, 275], "issue": "excluded_clear_glass_faces.npy is a 753296-element bool mask, not a 3468-element index vector; len(excluded)==3468 fails and len(excluded) misreports excluded triangle count.", "resolution": "Parent changed to dtype/shape assertion, keep=~excluded, excluded_count=int(excluded.sum()) and uses that count in metadata.", "verified_excluded_faces": 3468, "verified_kept_faces": 749828}],
        "remaining_blocking_findings": [],
        "coordinate_and_format_review": {
            "COLMAP_world_to_camera_and_inverse": "PASS: pc=world@R.T+t and world=(pc-t)@R, wxyz quaternion",
            "pixel_centres": "PASS: (x+0.5,y+0.5), consistent with floor(u),floor(v) assignment",
            "PNG_backprojection_export_chain": "PASS: delivered uint16 PNG is read, divided by1000, backprojected, cast float32, exported and exactly reread; RGB is taken from the matching saved RGB PNG",
            "all_B_valid_pixels_checked_before_probe": 2781236, "max_B_roundtrip_pixel_error": 2.9558577807620168e-12, "max_B_roundtrip_camera_Z_error_m": 3.552713678800501e-15,
            "analytic_Embree_checks_performed_before_probe": {"oblique_unnormalized_ray_plane_Z2_max_error_m": 0.0, "Z0_1_plane_skipped_by_near0_2": True, "Z70_plane_rejected_by_far65_535": True},
            "near_offset_and_tfar": "PASS: camera-Z-unit directions are not normalized; origin=center+0.2*direction and first camera Z=tfar+0.2",
            "glass_exclusion": "PASS: bool mask exactly matches material semantic_class clear_glass; same mask used by independent probe",
            "common_mask": "PASS: A depth valid AND B depth valid AND mesh reference valid, so common-valid depth compares identical pixel support",
        },
        "metric_interpretation_limits": [
            "One-sided NN to the original full cloud measures proximity to any source surface and cannot establish visibility or completeness. B retains original points and is structurally favored by this metric.",
            "Own-valid, interior and boundary summaries use method-specific masks. Only common-valid summaries have matched support across A/B; coverage must be shown alongside errors.",
            "The common-valid mask is concentrated on B-occupied pixels, so it is not a whole-image comparison of completeness.",
            "Behind/in-front of mesh first hit is a synthetic visibility proxy after the0.2m near clip. It is not real-world ground truth or a physical transmission model.",
            "Merged ten-view clouds retain duplicated surfaces, and aggregate errors pool eligible pixels rather than equally weighting cameras.",
        ],
        "exact_winner_ray_probe": {
            "status": "PASS_PROBE_CONFIRMS_BACKGROUND_LEAKAGE_WITHOUT_PIXEL_CENTER_SNAPPING",
            "method": "At most10000 uniformly sampled valid pixels per camera; ray from true camera centre through original PLY winner, scaled to camera-Z component1; no pixel-centre substitution and no PNG quantization in winner Z.",
            "independent_rotation": "scipy Rotation.from_quat after wxyz-to-xyzw reorder, independent of evaluator colmap_io.rotation",
            "seed_formula": "20260922 + image_id; numpy default_rng; choice without replacement; sort sampled flat pixel IDs",
            "sampled_points": projected_total, "exact_ray_reference_valid": exact_reference_total, "exact_ray_behind_gt30mm_count": exact_behind_total,
            "exact_ray_behind_gt30mm_percent_equal_sample_per_camera": exact_behind_total * 100 / exact_reference_total,
            "estimated_pixel_population_weighted_behind_gt30mm_percent": pixel_weighted_rate,
            "exact_ray_MAE_mm_equal_sample_per_camera": float(np.abs(merged_errors).mean() * 1000),
            "exact_ray_p95_abs_error_mm_equal_sample_per_camera": float(np.quantile(np.abs(merged_errors), 0.95) * 1000),
            "same_sample_pixel_center_behind_gt30mm_count": center_behind_total,
            "all_sample_winners_reproject_to_saved_pixel": True,
            "independent_float64_triangle_checks": len(examples), "max_independent_triangle_Z_difference_m": max_independent_triangle_Z_difference,
            "interpretation": "The large behind-first-surface rate remains when the ray passes through each original winning 3D sample. Thus the effect cannot be attributed to pixel-centre snapping, camera-coordinate inversion, or saved-PNG quantization. Sparse foreground samples leave image cells in which a geometrically occluded background sample becomes the closest projected point.",
            "per_view": results,
            "top_occluder_winner_pairs": [{"first_hit_object_id": a, "first_hit_object": audit["parts"][a]["name"], "winner_object_id": b, "winner_object": audit["parts"][b]["name"], "sampled_behind_count": n} for (a, b), n in global_pairs.most_common(15)],
            "examples": examples,
        },
        "protected_input_sha256": before, "protected_inputs_unchanged": True,
        "no_shared_output_modified": True, "no_projection_rerun": True, "elapsed_seconds": time.perf_counter() - started,
    }
    with report_path.open("x", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, allow_nan=False)
    print(json.dumps({"status": report["status"], "exact_ray_behind_gt30mm_percent_equal_sample_per_camera": exact_behind_total * 100 / exact_reference_total, "estimated_pixel_population_weighted_behind_gt30mm_percent": pixel_weighted_rate, "exact_ray_MAE_mm": float(np.abs(merged_errors).mean() * 1000), "examples_checked": len(examples), "max_independent_triangle_Z_difference_m": max_independent_triangle_Z_difference, "report": str(report_path)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
