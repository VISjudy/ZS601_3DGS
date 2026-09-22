"""Pure point-cloud z-buffer projection of the fixed ten v007 cameras.

No mesh, Gaussian splats, interpolation, hole filling, HPR, or generated PLY.
Depth selection uses float64 camera Z. Each point lands in floor(u), floor(v).
The only outputs are under method_b_zbuffer; existing outputs are protected.
"""
from pathlib import Path
import hashlib
import json
import os
import struct
import sys
import time

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

EXPECTED_IDS = [3202, 3340, 3376, 3388, 3409, 3481, 3505, 3520, 3571, 3583]
EXPECTED_CLOUD_SHA = "68f26fca61b4c1424c1f45f2f4983c4c42cf389abc23d6e0afadd9d48819de5b"
CHUNK = 250_000


def require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def qvec_to_rotation(qvec):
    w, x, y, z = np.asarray(qvec, dtype=np.float64)
    require(abs(np.linalg.norm(qvec) - 1) < 1e-10, "COLMAP quaternion is not unit length")
    rotation = np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y],
    ])
    independent = Rotation.from_quat([x, y, z, w]).as_matrix()
    require(np.max(np.abs(rotation - independent)) < 1e-12, "Independent quaternion conversion mismatch")
    require(np.max(np.abs(rotation @ rotation.T - np.eye(3))) < 1e-12, "Invalid camera rotation")
    return rotation


def projected_points(xyz, camera, rotation, near, far):
    """Bound working arrays by chunks; only in-frame candidates are collected."""
    translation = np.asarray(camera["t"], dtype=np.float64)
    width, height = camera["width"], camera["height"]
    ids_list, pixels_list, depths_list = [], [], []
    near_rejected = far_rejected = outside_rejected = 0
    for first in range(0, len(xyz), CHUNK):
        last = min(first + CHUNK, len(xyz))
        pc = xyz[first:last].astype(np.float64) @ rotation.T + translation
        require(np.isfinite(pc).all(), "Non-finite camera coordinates")
        z = pc[:, 2]
        near_rejected += int((z <= near).sum())
        far_rejected += int((z > far).sum())
        z_ok = np.flatnonzero((z > near) & (z <= far))
        camera_valid = pc[z_ok]
        u = camera["fx"] * camera_valid[:, 0] / camera_valid[:, 2] + camera["cx"]
        v = camera["fy"] * camera_valid[:, 1] / camera_valid[:, 2] + camera["cy"]
        inside = (u >= 0) & (u < width) & (v >= 0) & (v < height)
        outside_rejected += int((~inside).sum())
        selected = z_ok[inside]
        ids_list.append((selected + first).astype(np.int32))
        pixels_list.append((np.floor(v[inside]).astype(np.int32) * width + np.floor(u[inside]).astype(np.int32)))
        depths_list.append(z[selected])
    ids = np.concatenate(ids_list)
    pixels = np.concatenate(pixels_list)
    depths = np.concatenate(depths_list)
    require(len(ids) + near_rejected + far_rejected + outside_rejected == len(xyz), "Projection point accounting mismatch")
    return ids, pixels, depths, {
        "source_points": len(xyz),
        "rejected_Z_le_near": near_rejected,
        "rejected_Z_gt_far": far_rejected,
        "rejected_outside_image_with_valid_Z": outside_rejected,
        "projected_points_in_image": len(ids),
    }


def save_npy(path, array):
    with path.open("xb") as f:
        np.save(f, array, allow_pickle=False)


def save_png(path, array):
    with path.open("xb") as f:
        Image.fromarray(array).save(f, format="PNG", compress_level=6)


def check_png(path, expected, bit_depth, color_type):
    with path.open("rb") as f:
        header = f.read(33)
    require(header[:8] == b"\x89PNG\r\n\x1a\n" and header[12:16] == b"IHDR", "Invalid PNG header")
    width, height, depth, color, compression, filtering, interlace = struct.unpack(">IIBBBBB", header[16:29])
    require((height, width) == expected.shape[:2], "PNG dimensions differ")
    require(depth == bit_depth and color == color_type, "PNG channel/bit-depth encoding differs")
    with Image.open(path) as image:
        image.verify()
    with Image.open(path) as image:
        loaded = np.asarray(image)
    require(loaded.dtype == expected.dtype and loaded.shape == expected.shape, "PNG dtype or shape roundtrip failed")
    require(np.array_equal(loaded, expected) and np.isfinite(loaded).all(), "PNG values roundtrip failed")


def process_view(xyz, colors, camera, spec, out):
    started = time.perf_counter()
    width, height = camera["width"], camera["height"]
    require([width, height] == spec["resolution"], "Camera resolution differs from spec")
    rotation = qvec_to_rotation(camera["q"])
    ids, pixels, depths, counts = projected_points(xyz, camera, rotation, spec["near_clip_m"], spec["far_clip_m"])
    require(len(ids) > 0, "No projected point candidates")
    total_pixels = width * height
    minimum_z = np.full(total_pixels, np.inf, dtype=np.float64)
    np.minimum.at(minimum_z, pixels, depths)
    candidates = depths == minimum_z[pixels]
    winners = np.full(total_pixels, np.iinfo(np.int32).max, dtype=np.int32)
    np.minimum.at(winners, pixels[candidates], ids[candidates])
    valid = np.isfinite(minimum_z)
    winners[~valid] = -1
    n_valid = int(valid.sum())
    require(n_valid > 0 and np.all(winners[valid] >= 0) and np.all(winners[valid] < len(xyz)), "Winner source IDs invalid")

    # Independent all-candidate verification: sort by pixel, Z, then source ID.
    # It does not use the minimum.at reduction to identify reference winners.
    order = np.lexsort((ids, depths, pixels))
    sorted_pixels = pixels[order]
    first_per_pixel = np.r_[True, sorted_pixels[1:] != sorted_pixels[:-1]]
    reference_indices = order[first_per_pixel]
    reference_pixels = pixels[reference_indices]
    require(np.array_equal(np.flatnonzero(valid), reference_pixels), "Independent occupied pixel set differs")
    require(np.array_equal(winners[reference_pixels], ids[reference_indices]), "Independent nearest/tie winner IDs differ")
    require(np.array_equal(minimum_z[reference_pixels], depths[reference_indices]), "Independent nearest depth differs")
    require(np.all(depths >= minimum_z[pixels]), "A closer projected point was discarded")
    del order, sorted_pixels, first_per_pixel, reference_indices

    winner_ids = winners[valid]
    check_pc = xyz[winner_ids].astype(np.float64) @ rotation.T + np.asarray(camera["t"], dtype=np.float64)
    check_u = camera["fx"] * check_pc[:, 0] / check_pc[:, 2] + camera["cx"]
    check_v = camera["fy"] * check_pc[:, 1] / check_pc[:, 2] + camera["cy"]
    check_pixel = np.floor(check_v).astype(np.int64) * width + np.floor(check_u).astype(np.int64)
    require(np.array_equal(check_pixel, np.flatnonzero(valid)), "Winner points reproject to different pixels")
    require(np.max(np.abs(check_pc[:, 2] - minimum_z[valid])) < 1e-12, "Winner depth differs on recomputation")
    require(np.all(check_pc[:, 2] > spec["near_clip_m"]) and np.all(check_pc[:, 2] <= spec["far_clip_m"]), "Winner clipping invalid")

    depth_float = np.where(valid, minimum_z, 0).reshape(height, width).astype(np.float32)
    depth_png = np.zeros(total_pixels, dtype=np.uint16)
    depth_png[valid] = np.rint(minimum_z[valid] * 1000).astype(np.uint16)
    depth_png = depth_png.reshape(height, width)
    mask = (valid.reshape(height, width).astype(np.uint8) * 255)
    image = np.zeros((total_pixels, 3), dtype=np.uint8)
    image[valid] = colors[winner_ids]
    image = image.reshape(height, width, 3)
    winner_image = winners.reshape(height, width)
    require(np.array_equal(image.reshape(-1, 3)[valid], colors[winner_ids]), "RGB is not exact winning point RGB")
    require(np.all(image.reshape(-1, 3)[~valid] == 0), "Invalid pixel RGB not black")
    require(np.array_equal(depth_png > 0, mask > 0) and np.array_equal(depth_float > 0, mask > 0), "Depth/mask validity differs")
    require(np.isfinite(depth_float).all(), "Non-finite saved float depth")
    quantization_error = float(np.max(np.abs(depth_png.reshape(-1)[valid].astype(np.float64) / 1000 - minimum_z[valid])))
    require(quantization_error <= 0.000500000001, "Millimetre depth quantization error exceeds half millimetre")

    basename = camera["name"]
    require(Path(basename).name == basename and Path(basename).suffix == ".png", "Unexpected image filename")
    stem = Path(basename).stem
    files = {
        "depth": out / "depth" / basename,
        "mask": out / "masks" / basename,
        "RGB": out / "images" / basename,
        "depth_float": out / "depth_float" / f"{stem}.npy",
        "winner_source_id": out / "winner_source_id" / f"{stem}.npy",
    }
    save_png(files["depth"], depth_png)
    save_png(files["mask"], mask)
    save_png(files["RGB"], image)
    save_npy(files["depth_float"], depth_float)
    save_npy(files["winner_source_id"], winner_image)
    check_png(files["depth"], depth_png, 16, 0)
    check_png(files["mask"], mask, 8, 0)
    check_png(files["RGB"], image, 8, 2)
    for label, expected in (("depth_float", depth_float), ("winner_source_id", winner_image)):
        loaded = np.load(files[label], allow_pickle=False)
        require(loaded.dtype == expected.dtype and loaded.shape == expected.shape and np.array_equal(loaded, expected), f"{label} NPY roundtrip failed")
        require(np.isfinite(loaded).all(), f"{label} NPY is non-finite")
    dropped = len(ids) - n_valid
    far_discarded = int((depths > minimum_z[pixels] + spec["occlusion_error_threshold_m"]).sum())
    tied_discarded = int(candidates.sum()) - n_valid
    require(0 <= tied_discarded <= dropped and 0 <= far_discarded <= dropped, "Discard counts inconsistent")
    result = {
        "image_id": camera["image_id"],
        "name": basename,
        "status": "PASS",
        **counts,
        "total_pixels": total_pixels,
        "valid_pixels": n_valid,
        "empty_pixels": total_pixels - n_valid,
        "coverage_fraction": n_valid / total_pixels,
        "coverage_percent": n_valid * 100 / total_pixels,
        "same_pixel_discarded_points": dropped,
        "same_pixel_farther_than_3cm_discarded_points": far_discarded,
        "same_pixel_0_to_3cm_inclusive_discarded_points": dropped - far_discarded,
        "same_pixel_equal_minimum_Z_ties_discarded": tied_discarded,
        "min_valid_camera_Z_m": float(minimum_z[valid].min()),
        "max_valid_camera_Z_m": float(minimum_z[valid].max()),
        "max_depth_png_quantization_error_m": quantization_error,
        "independent_all_candidate_lexicographic_nearest_check": "PASS",
        "all_winners_reproject_correctly": True,
        "all_RGB_exact_winner_values": True,
        "all_PNG_values_dtype_channels_and_CRC_roundtrip": "PASS",
        "all_NPY_values_dtype_shape_roundtrip": "PASS",
        "finite_values": True,
        "files": {k: {"path": p.relative_to(out).as_posix(), "bytes": p.stat().st_size, "sha256": digest(p)} for k, p in files.items()},
        "elapsed_seconds": time.perf_counter() - started,
    }
    print(json.dumps({k: result[k] for k in ("image_id", "status", "projected_points_in_image", "valid_pixels", "coverage_percent", "same_pixel_discarded_points", "same_pixel_farther_than_3cm_discarded_points")}), flush=True)
    return result


def main():
    started = time.perf_counter()
    spec_path = ROOT / "run_spec.json"
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    base = Path(spec["local_output"])
    selected_path = base / "selected_views.json"
    cameras = json.loads(selected_path.read_text(encoding="utf-8"))
    cloud_path = Path(spec["input_cloud"])
    source_hashes = {str(p): digest(p) for p in (spec_path, selected_path, cloud_path)}
    require(source_hashes[str(cloud_path)] == spec["input_cloud_sha256"] == EXPECTED_CLOUD_SHA, "Input cloud SHA differs")
    require(spec["view_ids"] == [v["image_id"] for v in cameras] == EXPECTED_IDS, "Fixed ten-view IDs/order differ")
    require(spec["views"] == len(cameras) == 10, "View count differs")
    require(spec["near_clip_m"] == 0.2 and spec["far_clip_m"] == 65.535, "Unexpected clipping limits")
    require(spec["occlusion_error_threshold_m"] == 0.03, "Unexpected occlusion threshold")
    require(base.resolve() == (PROJECT / "3dgsResult/zs601-depth-methods-v007").resolve(), "Unexpected output root")
    out = base / "method_b_zbuffer"
    require(not out.exists(), f"Refusing to repeat or overwrite existing experiment outputs: {out}")
    print(json.dumps({"input_cloud_sha256": source_hashes[str(cloud_path)], "view_ids": EXPECTED_IDS, "input_agreement": "PASS"}), flush=True)
    vertices = PlyData.read(cloud_path)["vertex"].data
    require(len(vertices) == 7_004_696, "Point count differs from validated no-glass cloud")
    xyz = np.column_stack([vertices[name] for name in ("x", "y", "z")])
    colors = np.column_stack([vertices[name] for name in ("red", "green", "blue")])
    require(np.isfinite(xyz).all() and colors.dtype == np.uint8, "Invalid source XYZ/RGB")
    out.mkdir()
    for child in ("depth", "masks", "images", "depth_float", "winner_source_id"):
        (out / child).mkdir()
    results = [process_view(xyz, colors, camera, spec, out) for camera in cameras]
    after_hashes = {str(p): digest(p) for p in (spec_path, selected_path, cloud_path)}
    require(source_hashes == after_hashes, "Protected inputs changed during projection")
    for child in ("depth", "masks", "images", "depth_float", "winner_source_id"):
        require(len(list((out / child).iterdir())) == 10, f"Unexpected output count in {child}")
    total_pixels = sum(v["total_pixels"] for v in results)
    total_valid = sum(v["valid_pixels"] for v in results)
    report = {
        "schema": "zs601-depth-method-b-zbuffer-v007",
        "status": "PASS",
        "method": "Pure point projection with strict nearest camera Z per pixel",
        "input_cloud": str(cloud_path),
        "input_cloud_sha256": source_hashes[str(cloud_path)],
        "input_points": len(xyz),
        "view_ids": EXPECTED_IDS,
        "selected_views_sha256": source_hashes[str(selected_path)],
        "run_spec_sha256": source_hashes[str(spec_path)],
        "protected_inputs_unchanged": True,
        "world_to_camera": "pc = xyz @ R.T + t; q is COLMAP wxyz; matrix checked independently with scipy Rotation",
        "projection": "u = fx * X/Z + cx; v = fy * Y/Z + cy; pixel = floor(u),floor(v)",
        "pixel_centres": "COLMAP top-left pixel centre (0.5,0.5); no extra projection offset",
        "near_clip_m": 0.2,
        "far_clip_m": 65.535,
        "clipping": "Z > near and Z <= far; 0 <= u < width and 0 <= v < height",
        "selection_precision": "float64 camera coordinates and depth; np.minimum.at reduction",
        "ties": "Exact equal float64 camera Z: lowest zero-based input PLY vertex index wins",
        "winner_source_id_semantics": "int32 zero-based row in input_cloud; -1 invalid. Prior-cloud sample ID is obtainable via the adjacent source_sample_id.npy mapping.",
        "depth_PNG_encoding": "uint16 single-channel camera-Z millimetres, np.rint(Z*1000), 0 invalid",
        "depth_float_encoding": "float32 camera-Z metres, 0 invalid",
        "mask_encoding": "uint8 single-channel, 255 valid and 0 empty",
        "RGB_encoding": "uint8 RGB copied exactly from nearest original point, black invalid pixels",
        "point_radius": 0,
        "interpolation": False,
        "hole_filling": False,
        "HPR": False,
        "mesh_or_GT_used_to_remove_points": False,
        "Gaussian_splats_used": False,
        "PLY_generated": False,
        "independent_verification": "All projected candidates independently sorted by pixel, Z, source ID; every winning source ID and depth exactly matches reduction. Every winner is reprojected, RGB checked, and PNG/NPY files reread with value/dtype/channel checks.",
        "aggregate": {
            "views": len(results),
            "total_pixels": total_pixels,
            "valid_pixels": total_valid,
            "coverage_fraction": total_valid / total_pixels,
            "coverage_percent": total_valid * 100 / total_pixels,
            "min_view_coverage_percent": min(v["coverage_percent"] for v in results),
            "max_view_coverage_percent": max(v["coverage_percent"] for v in results),
            "projected_points_in_images": sum(v["projected_points_in_image"] for v in results),
            "same_pixel_discarded_points": sum(v["same_pixel_discarded_points"] for v in results),
            "same_pixel_farther_than_3cm_discarded_points": sum(v["same_pixel_farther_than_3cm_discarded_points"] for v in results),
        },
        "views": results,
        "elapsed_seconds": time.perf_counter() - started,
    }
    with (out / "projection_report.json").open("x", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, allow_nan=False)
    print(json.dumps({"status": report["status"], "aggregate": report["aggregate"], "report": str(out / "projection_report.json"), "elapsed_seconds": report["elapsed_seconds"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
