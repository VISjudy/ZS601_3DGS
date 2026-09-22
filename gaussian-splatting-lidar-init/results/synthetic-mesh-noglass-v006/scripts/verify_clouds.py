"""Independent local verification of the two clear-glass-excluded mesh clouds.

Reads all exported PLY vertices and checks their provenance against the frozen
geometry and original PLY. Does not invoke Blender, Colab, or training. Writes
only the two explicitly named verification artifacts and refuses to replace them.
"""
from pathlib import Path
import hashlib
import json
import os
import sys
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[2]
sys.path.insert(0, str(PROJECT / ".runtime/synthetic-dataset"))
sys.path.insert(0, str(PROJECT / "experiments/2026-09-21/synthetic-lidar-init-v001/python-deps"))
import numpy as np
from scipy.spatial import cKDTree
from plyfile import PlyData

DATA = PROJECT / "scenes/zs601-meetingroom/synthetic-training-v001"
OLD = PROJECT / "3dgsResult/zs601-mesh1cm-scale1-v003/cloud"
OUT = PROJECT / "3dgsResult/zs601-mesh-noglass-v006"
AUDIT = OUT / "audit"
REPORT = OUT / "cloud_verification.json"
REFERENCE = ROOT / "knn_reference.npz"
CHUNK = 200_000
FIELDS = ("x", "y", "z", "nx", "ny", "nz", "red", "green", "blue")


def require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def snapshot(paths):
    return {
        p.relative_to(PROJECT).as_posix(): {
            "bytes": p.stat().st_size,
            "sha256": sha256(p),
        }
        for p in paths
    }


def columns(array, names):
    return np.column_stack([array[name] for name in names])


def voxel_keys(xyz, object_ids, resolution, object_count):
    cells = np.floor(xyz.astype(np.float64) / resolution).astype(np.int64) + 4096
    require(np.all((cells >= 0) & (cells < 8192)), "Voxel key domain exceeded")
    return np.ravel_multi_index(
        (object_ids, cells[:, 0], cells[:, 1], cells[:, 2]),
        (object_count, 8192, 8192, 8192),
    )


def distance_to_triangles(points, triangle_vertices):
    """Plane projection inside the triangle, else nearest of its three segments."""
    a, b, c = triangle_vertices[:, 0], triangle_vertices[:, 1], triangle_vertices[:, 2]
    ab, ac, ap = b - a, c - a, points - a
    cross = np.cross(ab, ac)
    twice_area = np.linalg.norm(cross, axis=1)
    require(np.all(twice_area > 1e-12), "A sample refers to a degenerate source face")
    unit_normal = cross / twice_area[:, None]
    plane_distance = np.einsum("ij,ij->i", ap, unit_normal)
    projected = points - plane_distance[:, None] * unit_normal
    delta = projected - a
    den = twice_area * twice_area
    u = np.einsum("ij,ij->i", np.cross(delta, ac), cross) / den
    v = np.einsum("ij,ij->i", np.cross(ab, delta), cross) / den
    inside = (u >= 0) & (v >= 0) & (u + v <= 1)
    distances = np.where(inside, np.abs(plane_distance), np.inf)
    outside = np.flatnonzero(~inside)
    if len(outside):
        for left, right in ((a, b), (b, c), (c, a)):
            edge = (right - left)[outside]
            relative = (points - left)[outside]
            t = np.clip(
                np.einsum("ij,ij->i", relative, edge)
                / np.maximum(np.einsum("ij,ij->i", edge, edge), 1e-30),
                0,
                1,
            )
            segment_distance = np.linalg.norm(relative - t[:, None] * edge, axis=1)
            distances[outside] = np.minimum(distances[outside], segment_distance)
    return distances, unit_normal, len(outside)


def main():
    started = time.perf_counter()
    require(not REPORT.exists(), f"Refusing to replace {REPORT}")
    require(not REFERENCE.exists(), f"Refusing to replace {REFERENCE}")
    manifest = read_json(OUT / "cloud_manifest.json")
    old_manifest = read_json(OLD / "cloud_manifest.json")
    decision = read_json(AUDIT / "semantic_decision.json")
    audit = read_json(AUDIT / "material_audit.json")
    plan = read_json(DATA / "plan.json")
    protected_paths = sorted(p for p in OLD.iterdir() if p.is_file())
    protected_paths += [DATA / "source/scene.blend", DATA / "source/geometry.npz", DATA / "plan.json"]
    protected_paths += sorted(p for p in AUDIT.iterdir() if p.is_file())
    protected_paths += [OUT / "cloud_manifest.json"]
    for label in ("1cm", "3cm"):
        protected_paths += sorted(p for p in (OUT / f"cloud_{label}").iterdir() if p.is_file())
    before = snapshot(protected_paths)

    def current_record(path):
        return before[path.relative_to(PROJECT).as_posix()]

    for filename, expected in old_manifest["files"].items():
        require(current_record(OLD / filename) == expected, f"Original cloud changed: {filename}")
    require(current_record(OLD / "points_mesh_1cm.ply")["sha256"] == manifest["source_cloud_sha256"], "Source cloud SHA mismatch")
    scene_sha = current_record(DATA / "source/scene.blend")["sha256"]
    require(scene_sha == plan["source_sha256"] == audit["source_scene_sha256"] == manifest["source_scene_sha256"] == old_manifest["source_scene_sha256"], "Frozen source scene SHA mismatch")
    require(current_record(DATA / "source/geometry.npz")["sha256"] == plan["geometry_sha256"], "Frozen geometry SHA mismatch")
    require(audit["source_geometry_exact"], "Audit does not establish scene-to-geometry equality")

    with np.load(DATA / "source/geometry.npz", allow_pickle=False) as archive:
        vertices = archive["vertices"]
        faces = archive["faces"]
        face_objects = archive["object_ids"]
    material_ids = np.load(AUDIT / "face_material_id.npy", allow_pickle=False)
    excluded_faces = np.load(AUDIT / "excluded_clear_glass_faces.npy", allow_pickle=False)
    require(excluded_faces.dtype == np.bool_ and excluded_faces.shape == (len(faces),), "Excluded face mask shape/type invalid")
    require(material_ids.shape == (len(faces),), "Material face map shape invalid")
    clear_materials = [m for m in audit["materials"] if m["custom_properties"].get("semantic_class") == "clear_glass"]
    require(len(clear_materials) == 1, "Unexpected clear_glass material count")
    require({m["name"] for m in clear_materials} == set(decision["excluded_materials"]), "Semantic material decision mismatch")
    semantic_mask = np.isin(material_ids, [m["id"] for m in clear_materials])
    require(np.array_equal(semantic_mask, excluded_faces), "Excluded face mask differs from clear_glass semantics")
    for material in clear_materials:
        require(not material["unresolved"], "Glass material shader remains unresolved")
        require(any(node.get("Transmission Weight", {}).get("value") == 1.0 and not node.get("Transmission Weight", {}).get("linked", True) for node in material["active_nodes"]), "Glass shader is not verified constant full transmission")
    excluded_parts = []
    seen_faces = np.zeros(len(faces), dtype=np.uint8)
    for object_id, part in enumerate(audit["parts"]):
        first, stop = part["f0"], part["f0"] + part["nf"]
        require(np.all(face_objects[first:stop] == object_id), "Audit object order differs from source geometry")
        seen_faces[first:stop] += 1
        if excluded_faces[first:stop].any():
            excluded_parts.append(part["name"])
    require(np.all(seen_faces == 1), "Audit face ranges do not partition geometry")
    require(len(excluded_parts) == 33 and set(excluded_parts) == set(decision["excluded_parts"]), "Excluded component decision mismatch")
    require(int(excluded_faces.sum()) == decision["excluded_triangles"] == 3468, "Excluded triangle count mismatch")

    source_ply = PlyData.read(OLD / "points_mesh_1cm.ply")
    source = source_ply["vertex"].data
    old_faces = np.load(OLD / "mesh_face_id.npy", mmap_mode="r", allow_pickle=False)
    old_views = np.load(OLD / "color_source_view_id.npy", mmap_mode="r", allow_pickle=False)
    require(len(source) == len(old_faces) == len(old_views) == 7_180_820, "Original cloud count invalid")
    require(old_faces.min() >= 0 and old_faces.max() < len(faces), "Original face IDs out of range")
    keep_expected = np.flatnonzero(~excluded_faces[old_faces])
    require(len(source) - len(keep_expected) == decision["excluded_source_points"] == 176_124, "Excluded source point count mismatch")
    train_ids = np.array([view["id"] for view in plan["views"] if view["subset"] == "train"], dtype=np.int64)
    require(len(train_ids) == len(np.unique(train_ids)) == 2418, "Training view ID list invalid")
    object_count = len(audit["parts"])
    reports = {}
    source_indices = {}
    clouds = {}
    object_maps = {}
    check_hashes = {}

    for label, expected_n, resolution in (("1cm", 7_004_696, 0.01), ("3cm", 797_520, 0.03)):
        folder = OUT / f"cloud_{label}"
        expected = manifest["outputs"][label]
        for filename, record in expected["files"].items():
            require(current_record(folder / filename) == record, f"Export hash mismatch: {label}/{filename}")
            check_hashes[(folder / filename).relative_to(OUT).as_posix()] = record
        ply = PlyData.read(OUT / expected["path"])
        require(not ply.text and ply.byte_order == "<", "Expected binary little-endian PLY")
        require(len(ply.elements) == 1 and ply.elements[0].name == "vertex", "Unexpected PLY elements")
        cloud = ply["vertex"].data
        require(cloud.dtype.names == FIELDS and cloud.dtype == source.dtype, "PLY vertex schema differs from source")
        sample_ids = np.load(folder / "source_sample_id.npy", mmap_mode="r", allow_pickle=False)
        face_ids = np.load(folder / "mesh_face_id.npy", mmap_mode="r", allow_pickle=False)
        view_ids = np.load(folder / "color_source_view_id.npy", mmap_mode="r", allow_pickle=False)
        require(len(cloud) == len(sample_ids) == len(face_ids) == len(view_ids) == expected_n == expected["points"], f"{label}: count mismatch")
        require(np.issubdtype(sample_ids.dtype, np.integer) and sample_ids.min() >= 0 and sample_ids.max() < len(source), "Invalid source sample IDs")
        require(np.all(sample_ids[1:] > sample_ids[:-1]), "Source sample IDs are not unique and increasing")
        require(np.issubdtype(face_ids.dtype, np.integer) and face_ids.min() >= 0 and face_ids.max() < len(faces), "Invalid face IDs")
        require(not excluded_faces[face_ids].any(), "Export retains excluded clear_glass faces")
        require(np.isin(view_ids, np.r_[-1, train_ids]).all(), "Color provenance includes non-training views or invalid fallback")
        if label == "1cm":
            require(np.array_equal(sample_ids, keep_expected), "1cm cloud is not exactly every non-glass source point")

        max_distance = 0.0
        distance_sum = 0.0
        max_normal_error = 0.0
        max_normal_unit_error = 0.0
        outside_projections = 0
        for start in range(0, len(cloud), CHUNK):
            stop = min(start + CHUNK, len(cloud))
            block = cloud[start:stop]
            ids = sample_ids[start:stop]
            original = source[ids]
            require(np.array_equal(block, original), f"{label}: XYZ/RGB/normal source attributes changed at block {start}")
            require(np.array_equal(face_ids[start:stop], old_faces[ids]), "Face provenance changed")
            require(np.array_equal(view_ids[start:stop], old_views[ids]), "Color provenance changed")
            points = columns(block, FIELDS[:3]).astype(np.float64)
            normals = columns(block, FIELDS[3:6]).astype(np.float64)
            rgb = columns(block, FIELDS[6:])
            require(np.isfinite(points).all() and np.isfinite(normals).all() and np.isfinite(rgb).all(), "Non-finite PLY value")
            require(rgb.dtype == np.uint8 and rgb.min() >= 0 and rgb.max() <= 255, "RGB outside uint8 range")
            triangle_vertices = vertices[faces[face_ids[start:stop]]].astype(np.float64)
            distances, unit_normal, n_outside = distance_to_triangles(points, triangle_vertices)
            require(np.isfinite(distances).all(), "Point-to-triangle computation failed")
            max_distance = max(max_distance, float(distances.max()))
            distance_sum += float(distances.sum())
            max_normal_error = max(max_normal_error, float(np.abs(normals - unit_normal).max()))
            max_normal_unit_error = max(max_normal_unit_error, float(np.abs(np.linalg.norm(normals, axis=1) - 1).max()))
            outside_projections += n_outside
        require(max_distance < 2e-6, f"{label}: source surface distance exceeds 2 micrometres: {max_distance}")
        require(max_normal_error < 1e-6, f"{label}: normal error exceeds 1e-6: {max_normal_error}")
        require(max_normal_unit_error < 1e-6, f"{label}: non-unit normals")
        xyz = columns(cloud, FIELDS[:3])
        objects = face_objects[face_ids]
        keys = voxel_keys(xyz, objects, resolution, object_count)
        unique_count = len(np.unique(keys))
        require(unique_count == len(cloud), f"{label}: duplicate per-component voxel")
        n_components = len(np.unique(objects))
        require(n_components == 2055 == expected["components"], "Component count mismatch")
        bounds = [xyz.min(axis=0).tolist(), xyz.max(axis=0).tolist()]
        require(bounds == expected["bounds"], "Cloud bounds differ from manifest")
        projected = int((view_ids >= 0).sum())
        fallback = int((view_ids == -1).sum())
        require(projected == expected["projected_points"] and fallback == expected["fallback_points"], "Color point counts differ from manifest")
        reports[label] = {
            "status": "PASS",
            "points": len(cloud),
            "components": n_components,
            "ply_readback": "binary_little_endian_float32_XYZ_float32_normals_uint8_RGB",
            "all_points_checked": True,
            "finite_XYZ_normals_RGB": True,
            "valid_uint8_RGB": True,
            "all_XYZ_RGB_normals_exact_original_source_values": True,
            "source_sample_face_and_color_view_ids_exact": True,
            "excluded_clear_glass_points": 0,
            "max_point_to_source_triangle_distance_m": max_distance,
            "mean_point_to_source_triangle_distance_m": distance_sum / len(cloud),
            "distance_tolerance_m": 2e-6,
            "max_normal_component_error": max_normal_error,
            "max_normal_unit_length_error": max_normal_unit_error,
            "projections_outside_triangle_count_checked_by_segments": outside_projections,
            "voxel_size_m": resolution,
            "unique_occupied_voxels_per_component": unique_count,
            "one_representative_per_component_voxel": True,
            "minimum_pairwise_spacing_guaranteed": False,
            "bounds_m": bounds,
            "train_view_ids_allowed": len(train_ids),
            "all_color_views_train_or_minus_one": True,
            "unique_train_views_actually_used": len(np.unique(view_ids[view_ids >= 0])),
            "projected_points": projected,
            "material_fallback_points": fallback,
        }
        source_indices[label] = sample_ids
        clouds[label] = xyz
        object_maps[label] = objects
        del keys
        print(f"ALL_POINT_CHECKS_PASS {label} {len(cloud)} max_surface_distance_m={max_distance:.12g}", flush=True)

    locations = np.searchsorted(source_indices["1cm"], source_indices["3cm"])
    require(np.all(locations < len(source_indices["1cm"])), "3cm source IDs outside 1cm source IDs")
    require(np.array_equal(source_indices["1cm"][locations], source_indices["3cm"]), "3cm cloud is not a source-ID subset of 1cm cloud")
    require(np.array_equal(clouds["1cm"][locations], clouds["3cm"]), "3cm subset XYZ differs")
    fine_to_coarse_keys = np.unique(voxel_keys(clouds["1cm"], object_maps["1cm"], 0.03, object_count))
    coarse_keys = np.sort(voxel_keys(clouds["3cm"], object_maps["3cm"], 0.03, object_count))
    require(np.array_equal(fine_to_coarse_keys, coarse_keys), "3cm cloud does not represent every occupied 3cm source voxel")
    del fine_to_coarse_keys, coarse_keys
    print("COARSE_SUBSET_AND_VOXEL_COVERAGE_PASS", flush=True)

    retained_material_checks = {}
    for name in decision["retained_transmissive_plastics"] + decision["retained_opaque_glass_named_materials"]:
        ids = [m["id"] for m in audit["materials"] if m["name"] == name]
        require(len(ids) == 1, f"Retained material missing or ambiguous: {name}")
        material_face_mask = material_ids == ids[0]
        require(not np.any(material_face_mask & excluded_faces), f"Retained material was excluded: {name}")
        old_count = int(material_face_mask[old_faces].sum())
        fine_faces = np.load(OUT / "cloud_1cm/mesh_face_id.npy", mmap_mode="r", allow_pickle=False)
        fine_count = int(material_face_mask[fine_faces].sum())
        require(old_count == fine_count and fine_count > 0, f"Retained material points lost: {name}")
        retained_material_checks[name] = {"original_points": old_count, "retained_1cm_points": fine_count, "all_preserved": True}

    sample_indices = np.random.default_rng(601006).choice(len(clouds["1cm"]), size=20_000, replace=False)
    tree = cKDTree(clouds["1cm"])
    distances, neighbor_indices = tree.query(clouds["1cm"][sample_indices], k=4, eps=0, p=2, workers=2)
    require(np.isfinite(distances).all() and np.all(distances[:, 0] == 0), "Exact neighbor query failed")
    # Removing one zero self-distance also gives the correct distance multiset
    # if coincident distinct samples tie with the queried point.
    rms = np.sqrt(np.mean(np.square(distances[:, 1:]), axis=1))
    require(np.all(rms > 0), "3NN RMS contains zero scales")
    knn_stats = {
        "sampled_points": len(sample_indices),
        "seed": 601006,
        "query": "scipy.spatial.cKDTree exact Euclidean query, eps=0, k=4 including one self-distance",
        "knn_k_excluding_self": 3,
        "rms_definition": "sqrt(mean(squared distances to 3 nearest other points))",
        "nearest_neighbor_m": {"min": float(distances[:, 1].min()), "median": float(np.median(distances[:, 1])), "p95": float(np.quantile(distances[:, 1], 0.95))},
        "rms_distance_m": {"min": float(rms.min()), "median": float(np.median(rms)), "p95": float(np.quantile(rms, 0.95)), "max": float(rms.max())},
        "render_scale_multiplier": 0.5,
        "expected_sigma_scale05_m": {"min": float(rms.min() * 0.5), "median": float(np.median(rms) * 0.5), "p95": float(np.quantile(rms, 0.95) * 0.5), "max": float(rms.max() * 0.5)},
        "reference_path": REFERENCE.relative_to(PROJECT).as_posix(),
        "reference_keys": ["point_indices", "rms_distance_m"],
    }
    after = snapshot(protected_paths)
    require(before == after, "An input, audit, original cloud, frozen scene, or exported cloud changed during verification")
    with REFERENCE.open("xb") as handle:
        np.savez(handle, point_indices=sample_indices, rms_distance_m=rms)
    with np.load(REFERENCE, allow_pickle=False) as reread:
        require(np.array_equal(reread["point_indices"], sample_indices) and np.array_equal(reread["rms_distance_m"], rms), "3NN reference roundtrip mismatch")
    knn_stats["reference_sha256"] = sha256(REFERENCE)
    report = {
        "schema": "zs601-mesh-noglass-cloud-verification-v1",
        "status": "PASS",
        "scope": "Independent local PLY/provenance/surface/voxel validation; no Gaussian rendering or training result",
        "units": "metres",
        "source_scene_sha256": scene_sha,
        "original_1cm_source_cloud_sha256": manifest["source_cloud_sha256"],
        "original_source_scene_matches_frozen_plan": True,
        "original_cloud_files_match_preexisting_manifest": True,
        "all_protected_inputs_unchanged_during_verification": True,
        "protected_input_files": before,
        "all_output_files_match_export_manifest": True,
        "export_file_hashes": check_hashes,
        "semantic_exclusion": {
            "criterion": "material custom_properties.semantic_class == clear_glass",
            "excluded_materials": decision["excluded_materials"],
            "excluded_components": len(excluded_parts),
            "excluded_triangles": int(excluded_faces.sum()),
            "removed_source_points": len(source) - len(keep_expected),
            "one_cm_exactly_original_cloud_minus_clear_glass": True,
            "final_face_mask_exactly_matches_semantic_class": True,
            "retained_material_checks": retained_material_checks,
            "broad_transmission_candidate_mask_used_as_final_exclusion": False,
        },
        "clouds": reports,
        "three_cm_exact_subset_of_one_cm": True,
        "three_cm_covers_all_one_cm_occupied_three_cm_voxels_per_component": True,
        "minimum_pairwise_spacing_guaranteed": False,
        "spacing_note": "1cm/3cm are world-axis voxel resolutions per component. Representatives in adjacent cells or different components may be closer than that resolution; this is not a strict minimum-distance or Poisson-disk guarantee.",
        "knn_reference": knn_stats,
        "limitations": [
            "The frozen Blender material audit is read as input; this verifier does not reopen Blender.",
            "Train-only view IDs and exact preservation of prior RGB are checked; per-point color visibility is inherited from the original cloud process, not recomputed here.",
            "Unobserved opaque surfaces retain original material fallback colors; removing clear glass does not bake missing textures, reflections, or illumination.",
            "Semantic image masks were not added and Gaussian rendering/training is outside this report.",
        ],
        "elapsed_seconds": time.perf_counter() - started,
    }
    with REPORT.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({"status": report["status"], "clouds": {k: {"points": v["points"], "max_point_to_source_triangle_distance_m": v["max_point_to_source_triangle_distance_m"]} for k, v in reports.items()}, "knn_reference": knn_stats, "report": str(REPORT)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
