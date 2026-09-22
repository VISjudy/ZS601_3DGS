"""CPU-only v008 five-method depth evaluation, run after renderer download.

Saved uint16 PNG depth is the source for all backprojection and geometric metrics.
Auxiliary float depth/first-hit records validate encoding and provenance only.
The evaluator never filters generated points by the mesh reference or alpha50.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import os
import struct
import sys
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[3]
sys.path.insert(0, str(HERE / "python-deps"))
sys.path.insert(0, str(PROJECT / ".runtime/synthetic-dataset"))
sys.path.insert(0, str(PROJECT / "experiments/2026-09-21/synthetic-lidar-init-v001/python-deps"))
import numpy as np
from PIL import Image
from scipy.spatial import cKDTree
from plyfile import PlyData, PlyElement

VIEW_IDS = [3202, 3340, 3376, 3388, 3409, 3481, 3505, 3520, 3571, 3583]
CLOUD_SHA = "68f26fca61b4c1424c1f45f2f4983c4c42cf389abc23d6e0afadd9d48819de5b"
PLY_DTYPE = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")])
AUX_ALIASES = {
    "first_id": ("first_id", "first_hit_id", "hit_id", "first_splat_id", "id"),
    "center_z": ("center_z", "first_center_z", "z_center"),
    "peak_z": ("peak_z", "first_peak_z", "z_peak"),
    "entry_z": ("entry_z", "first_entry_z", "z_entry"),
    "exit_z": ("exit_z", "first_exit_z", "z_exit"),
    "qmin": ("qmin", "q_min", "first_qmin"),
    "first_alpha": ("first_alpha", "hit_alpha"),
}


def require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)


def save_npy(path, array):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as f:
        np.save(f, array, allow_pickle=False)


def save_mask(path, mask):
    path.parent.mkdir(parents=True, exist_ok=True)
    expected = mask.astype(np.uint8) * 255
    with path.open("xb") as f:
        Image.fromarray(expected).save(f, format="PNG")
    require(np.array_equal(read_png(path, 8, 1, expected.shape), expected), "Mask output roundtrip mismatch")


def read_png(path, bits, channels, shape):
    with path.open("rb") as f:
        header = f.read(33)
    require(header[:8] == b"\x89PNG\r\n\x1a\n" and header[12:16] == b"IHDR", f"Invalid PNG: {path}")
    width, height, actual_bits, color_type, _, _, _ = struct.unpack(">IIBBBBB", header[16:29])
    require((height, width) == tuple(shape) and actual_bits == bits and color_type == {1: 0, 3: 2}[channels], f"PNG encoding mismatch: {path}")
    with Image.open(path) as image:
        image.verify()
    with Image.open(path) as image:
        array = np.asarray(image).copy()
    require(array.dtype == {8: np.uint8, 16: np.uint16}[bits], f"PNG dtype mismatch: {path}")
    require(array.shape == tuple(shape) + (() if channels == 1 else (3,)), f"PNG channel shape mismatch: {path}")
    return array


def read_mask(path, shape):
    mask = read_png(path, 8, 1, shape)
    require(np.isin(mask, [0, 255]).all(), f"Mask must be 0/255: {path}")
    return mask > 0


def rotation(q):
    q = np.asarray(q, dtype=np.float64)
    require(np.isfinite(q).all() and abs(np.linalg.norm(q) - 1) < 1e-8, "Invalid COLMAP quaternion")
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([[1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)], [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)], [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)]])


def backproject(camera, depth, valid):
    y, x = np.indices(depth.shape, dtype=np.float64)
    rays = np.stack([(x + .5 - camera["cx"]) / camera["fx"], (y + .5 - camera["cy"]) / camera["fy"], np.ones_like(x)], axis=-1)[valid]
    pc = rays * depth[valid, None]
    R, t = rotation(camera["q"]), np.asarray(camera["t"], dtype=np.float64)
    world = (pc - t) @ R
    if len(world):
        inverse = np.linalg.inv(np.block([[R, t[:, None]], [np.zeros((1, 3)), np.ones((1, 1))]]))
        require(np.max(np.abs(world - (pc @ inverse[:3, :3].T + inverse[:3, 3]))) < 1e-10, "Independent inverse transform mismatch")
        check = world @ R.T + t
        yy, xx = np.nonzero(valid)
        px = camera["fx"] * check[:, 0] / check[:, 2] + camera["cx"]
        py = camera["fy"] * check[:, 1] / check[:, 2] + camera["cy"]
        require(max(np.max(np.abs(px - xx - .5)), np.max(np.abs(py - yy - .5))) < 1e-8, "Pixel-centre roundtrip failed")
        require(np.max(np.abs(check[:, 2] - depth[valid])) < 1e-10, "Camera-Z roundtrip failed")
    return world


def stats(values, signed=False):
    values = np.asarray(values, dtype=np.float64)
    require(np.isfinite(values).all(), "Non-finite eligible metric values")
    keys = ["mean_mm", "median_mm", "p95_mm", "rmse_mm", "max_mm", "above_10mm_percent", "above_30mm_percent"]
    if signed:
        keys += ["bias_mm", "behind_first_surface_gt30mm_percent", "in_front_of_surface_gt30mm_percent"]
    if not len(values):
        return {"count": 0, **{k: None for k in keys}}
    absolute = np.abs(values)
    result = {"count": len(values), "mean_mm": float(absolute.mean() * 1000), "median_mm": float(np.median(absolute) * 1000), "p95_mm": float(np.quantile(absolute, .95) * 1000), "rmse_mm": float(np.sqrt(np.mean(values * values)) * 1000), "max_mm": float(absolute.max() * 1000), "above_10mm_percent": float((absolute > .01).mean() * 100), "above_30mm_percent": float((absolute > .03).mean() * 100)}
    if signed:
        result.update(bias_mm=float(values.mean() * 1000), behind_first_surface_gt30mm_percent=float((values > .03).mean() * 100), in_front_of_surface_gt30mm_percent=float((values < -.03).mean() * 100))
    return result


def first_hit_stats(qmin, alpha):
    qmin = np.asarray(qmin, dtype=np.float64)
    alpha = np.asarray(alpha, dtype=np.float64)
    require(qmin.shape == alpha.shape and np.isfinite(qmin).all() and np.isfinite(alpha).all(), "Invalid pooled first-contributor diagnostics")
    count = len(qmin)
    outside = int((qmin > 9).sum())
    strong = int((alpha >= .5).sum())
    return {"hit_count": count, "qmin_gt9_count": outside, "qmin_gt9_percent": outside * 100 / count if count else None,
            "qmin_median": float(np.median(qmin)) if count else None, "qmin_p95": float(np.quantile(qmin, .95)) if count else None,
            "first_alpha_median": float(np.median(alpha)) if count else None, "first_alpha_p05": float(np.quantile(alpha, .05)) if count else None, "first_alpha_p95": float(np.quantile(alpha, .95)) if count else None,
            "first_alpha_ge_0_5_count": strong, "first_alpha_ge_0_5_percent": strong * 100 / count if count else None, "alpha50_percent": strong * 100 / count if count else None,
            "alpha50_percent_definition": "Compatibility alias of first_alpha_ge_0_5_percent; single first-contributor alpha, NOT accumulated alpha and NOT depth_mask_alpha50",
            "population": "All pixels with decoded zero-based first_id>=0; pooled over pixels, not mean of per-view quantiles",
            "qmin_gt9_meaning": "No finite 3-sigma ellipsoid intersection; first-contributor halo is retained by renderer and separately diagnosed"}


def ply_array(xyz, rgb):
    result = np.empty(len(xyz), dtype=PLY_DTYPE)
    for i, key in enumerate(("x", "y", "z")):
        result[key] = xyz[:, i]
    for i, key in enumerate(("red", "green", "blue")):
        result[key] = rgb[:, i]
    return result


def write_ply(path, xyz, rgb):
    path.parent.mkdir(parents=True, exist_ok=True)
    result = ply_array(xyz, rgb)
    with path.open("xb") as f:
        PlyData([PlyElement.describe(result, "vertex")], text=False, byte_order="<", comments=["World metres; backprojected saved uint16-mm camera-Z PNG.", "Primary valid first hit; no alpha50 or mesh filtering; no ICP; no hole filling."]).write(f)
    loaded = PlyData.read(path)["vertex"].data
    require(np.array_equal(result, loaded), "PLY reread differs")
    return {"path": str(path), "points": len(result), "bytes": path.stat().st_size, "sha256": sha256(path), "exact_roundtrip": True}


def merge_plys(path, paths):
    arrays = [PlyData.read(p)["vertex"].data for p in paths]
    require(all(a.dtype == PLY_DTYPE for a in arrays), "Unexpected PLY schema for merge")
    count = sum(len(a) for a in arrays)
    header = ("ply\nformat binary_little_endian 1.0\ncomment World metres; concatenated per-view saved-PNG backprojections; duplicates retained.\n" + f"element vertex {count}\n" + "property float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n").encode("ascii")
    with path.open("xb") as f:
        f.write(header)
        for array in arrays:
            f.write(array.tobytes())
    loaded = PlyData.read(path)["vertex"].data
    offset = 0
    for array in arrays:
        require(np.array_equal(loaded[offset:offset + len(array)], array), "Merged PLY reread differs")
        offset += len(array)
    require(offset == len(loaded) == count, "Merged point count differs")
    return {"path": str(path), "points": count, "bytes": path.stat().st_size, "sha256": sha256(path), "exact_roundtrip": True, "duplicate_surfaces_retained": True}


def method_definitions(baseline, new_root):
    return [
        {"id": "method_a_gaussian", "label": "A: alpha-normalized harmonic Z, scale 0.5", "plot_label": "A | Harmonic Z\nGaussian scale 0.5", "input_root": str(baseline / "method_a_gaussian"), "mask_folder": "depth_mask", "new": False},
        {"id": "method_b_zbuffer", "label": "B: strict nearest projected point Z", "plot_label": "B | Point Z buffer\nNo point radius", "input_root": str(baseline / "method_b_zbuffer"), "mask_folder": "masks", "new": False},
        {"id": "method_c_first_center_s05", "label": "C: first-hit Gaussian centre Z, scale 0.5", "plot_label": "C | First centre Z\nGaussian scale 0.5", "input_root": str(new_root / "method_c_first_center_s05"), "mask_folder": "depth_mask", "new": True, "selected_aux_depth": "center_z"},
        {"id": "method_d_first_peak_s05", "label": "D: first-hit density-peak Z, scale 0.5", "plot_label": "D | First peak Z\nGaussian scale 0.5", "input_root": str(new_root / "method_d_first_peak_s05"), "mask_folder": "depth_mask", "new": True, "selected_aux_depth": "peak_z"},
        {"id": "method_e_first_peak_s01", "label": "E: first-hit density-peak Z, scale 0.1", "plot_label": "E | First peak Z\nGaussian scale 0.1", "input_root": str(new_root / "method_e_first_peak_s01"), "mask_folder": "depth_mask", "new": True, "selected_aux_depth": "peak_z"},
    ]


def resolve_aux_keys(keys):
    result = {}
    for canonical, alternatives in AUX_ALIASES.items():
        found = [key for key in alternatives if key in keys]
        require(len(found) == 1, f"Auxiliary {canonical}: expected one recognized key, got {found}; available={sorted(keys)}")
        result[canonical] = found[0]
    return result


def input_paths(methods, cameras, baseline, aux_folder, cloud):
    paths = [baseline / "selected_views.json", cloud]
    for camera in cameras:
        name, stem = camera["name"], Path(camera["name"]).stem
        paths += [baseline / "reference_noglass_mesh/depth_float" / f"{stem}.npy", baseline / "reference_noglass_mesh/discontinuity_masks" / name]
        for method in methods:
            root = Path(method["input_root"])
            paths += [root / "depth" / name, root / method["mask_folder"] / name, root / "images" / name, root / "depth_float" / f"{stem}.npy"]
            if method["new"]:
                paths += [root / "depth_mask_alpha50" / name, root / aux_folder / f"{stem}.npz"]
            else:
                paths += [root / "backprojected" / f"{stem}.ply"]
    return sorted(set(paths))


def validate_aux(path, raw, mask, method, camera, source_xyz):
    with np.load(path, allow_pickle=False) as archive:
        mapping = resolve_aux_keys(archive.files)
        data = {canonical: archive[key] for canonical, key in mapping.items()}
    require(all(a.shape == raw.shape for a in data.values()), f"Auxiliary image shape mismatch: {path}")
    ids = data["first_id"]
    require(np.issubdtype(ids.dtype, np.signedinteger), "First-hit IDs must be signed integer with -1 invalid")
    require(np.all(ids >= -1) and np.all(ids < len(source_xyz)), "First-hit source ID out of range")
    expected = (ids >= 0) & np.isfinite(raw) & (raw > 0)
    require(np.array_equal(mask, expected), "Primary mask differs from first-ID-present AND finite positive depth; alpha50 truncation is not allowed")
    field = data[method["selected_aux_depth"]]
    if mask.any():
        require(np.isfinite(field[mask]).all() and np.max(np.abs(raw[mask].astype(np.float64) - field[mask].astype(np.float64))) < 2e-5, "Raw selected depth differs from auxiliary centre/peak Z")
    hit = ids >= 0
    diagnostics = {"key_mapping": mapping, "first_id_present_pixels": int(hit.sum()), "primary_valid_pixels": int(mask.sum()), "ID_present_but_nonpositive_or_nonfinite_depth": int((hit & ~mask).sum()), "raw_nonfinite_outside_primary": int((~np.isfinite(raw) & ~mask).sum())}
    diagnostics["first_hit_diagnostics"] = first_hit_stats(data["qmin"][hit], data["first_alpha"][hit])
    diagnostics["first_hit_diagnostics"]["primary_valid_hit_count"] = int(mask.sum())
    if hit.any():
        for canonical in ("center_z", "peak_z", "entry_z", "exit_z", "qmin", "first_alpha"):
            require(np.isfinite(data[canonical][hit]).all(), f"Non-finite auxiliary {canonical} at first hit")
        R = rotation(camera["q"])
        expected_center = source_xyz[ids[hit]] @ R[2] + camera["t"][2]
        center_error = float(np.max(np.abs(expected_center - data["center_z"][hit])))
        require(center_error < 2e-5, "Auxiliary first-hit centre Z disagrees with source Gaussian/point index and COLMAP camera")
        require(np.all(data["entry_z"][hit] <= data["exit_z"][hit] + 1e-5), "Auxiliary entry exceeds exit")
        require(np.all(data["qmin"][hit] >= -1e-5), "Auxiliary Mahalanobis qmin is negative")
        require(np.all((data["first_alpha"][hit] >= -1e-6) & (data["first_alpha"][hit] <= 1 + 1e-6)), "Auxiliary first alpha outside [0,1]")
        halo = hit & (data["qmin"] > 9)
        require(np.all(data["entry_z"][halo] == 0) and np.all(data["exit_z"][halo] == 0), "qmin>9 must use the declared zero entry/exit sentinel")
        diagnostics.update(max_source_center_Z_difference_m=center_error, peak_outside_entry_exit_count=int(((data["peak_z"] < data["entry_z"] - 1e-5) | (data["peak_z"] > data["exit_z"] + 1e-5))[hit].sum()), qmin_range=[float(data["qmin"][hit].min()), float(data["qmin"][hit].max())], first_alpha_range=[float(data["first_alpha"][hit].min()), float(data["first_alpha"][hit].max())])
    return diagnostics


def read_method(method, camera, source_xyz, aux_folder):
    root = Path(method["input_root"])
    name, stem = camera["name"], Path(camera["name"]).stem
    shape = (camera["height"], camera["width"])
    png = read_png(root / "depth" / name, 16, 1, shape)
    mask = read_mask(root / method["mask_folder"] / name, shape)
    require(np.array_equal(png > 0, mask), "Saved PNG depth validity differs from delivered primary mask")
    rgb = read_png(root / "images" / name, 8, 3, shape)
    raw = np.load(root / "depth_float" / f"{stem}.npy", allow_pickle=False)
    require(raw.dtype == np.float32 and raw.shape == shape, "Raw float depth dtype/shape mismatch")
    if method["new"]:
        aux = validate_aux(root / aux_folder / f"{stem}.npz", raw, mask, method, camera, source_xyz)
        alpha50 = read_mask(root / "depth_mask_alpha50" / name, shape)
        require(np.all(~alpha50 | mask), "Alpha50 mask contains pixels outside primary first-hit validity")
    else:
        require(np.isfinite(raw).all() and np.array_equal(raw > 0, mask), "Baseline float validity mismatch")
        aux, alpha50 = None, mask.copy()
    z = png.astype(np.float64) / 1000
    quant = z[mask] - raw[mask].astype(np.float64)
    require(np.isfinite(quant).all() and (not len(quant) or np.max(np.abs(quant)) <= .000505), "Saved depth differs by more than float32-mm encoding tolerance")
    return {"z": z, "rgb": rgb, "mask": mask, "alpha50": alpha50, "quantization": quant, "auxiliary_validation": aux}


def metric_groups(nn, error, alpha50, common, common_alpha50, interior, boundary):
    own = np.isfinite(error)
    return {
        "point_to_input": (nn[np.isfinite(nn)], False),
        "common_all5_point_to_input": (nn[common], False),
        "common_all5_alpha50_point_to_input": (nn[common_alpha50], False),
        "own_valid_depth": (error[own], True),
        "common_all5_depth": (error[common], True),
        "alpha50_own_valid_depth": (error[own & alpha50], True),
        "common_all5_alpha50_depth": (error[common_alpha50], True),
        "interior_own_valid_depth": (error[own & interior], True),
        "interior_common_all5_depth": (error[common & interior], True),
        "boundary_own_valid_depth": (error[own & boundary], True),
        "boundary_common_all5_depth": (error[common & boundary], True),
    }


def write_csv(path, rows):
    with path.open("x", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def flat_metric_row(row):
    result = {k: row[k] for k in ("method", "coverage_percent", "coverage_alpha50_percent", "valid_pixels", "total_pixels")}
    if "image_id" in row:
        result["image_id"] = row["image_id"]
    for domain in ("point_to_input", "common_all5_point_to_input", "own_valid_depth", "common_all5_depth", "alpha50_own_valid_depth", "common_all5_alpha50_depth", "interior_common_all5_depth"):
        for field in ("count", "mean_mm", "median_mm", "p95_mm", "rmse_mm", "above_30mm_percent", "behind_first_surface_gt30mm_percent", "in_front_of_surface_gt30mm_percent"):
            if field in row[domain]:
                result[f"{domain}__{field}"] = row[domain][field]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", type=Path, default=PROJECT / "3dgsResult/zs601-depth-methods-v007")
    parser.add_argument("--new-root", type=Path, default=PROJECT / "3dgsResult/zs601-firsthit-depth-v008")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--aux-folder", default="aux")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--preflight", action="store_true", help="Check input existence and auxiliary keys without creating evaluation outputs")
    parser.add_argument("--representatives", nargs=3, type=int, default=[3202, 3376, 3481])
    parser.add_argument("--nn-max-mm", type=float, default=30)
    parser.add_argument("--signed-z-limit-mm", type=float, default=150)
    parser.add_argument("--no-figures", action="store_true", help="CPU metrics/PLY only; figures may be generated later with plot_errors.py")
    args = parser.parse_args()
    started = time.perf_counter()
    baseline, new_root = args.baseline_root.resolve(), args.new_root.resolve()
    output = (args.output or new_root / "evaluation").resolve()
    require(output.is_relative_to((new_root / "evaluation").resolve()), "Outputs must stay under the v008 evaluation directory")
    require(Path(args.aux_folder).name == args.aux_folder, "Auxiliary folder must be a single directory name")
    cameras = load_json(baseline / "selected_views.json")
    require([c["image_id"] for c in cameras] == VIEW_IDS, "The fixed ten cameras differ")
    if (new_root / "selected_views.json").exists():
        require(load_json(new_root / "selected_views.json") == cameras, "v008 camera metadata differs from v007")
    methods = method_definitions(baseline, new_root)
    cloud = PROJECT / "3dgsResult/zs601-mesh-noglass-v006/cloud_1cm/points_mesh_1cm_noglass.ply"
    paths = input_paths(methods, cameras, baseline, args.aux_folder, cloud)
    paths += [p for p in (new_root / "run_spec.json", new_root / "selected_views.json", baseline / "metrics/geometry_metrics.json") if p.exists()]
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        print(json.dumps({"status": "WAITING_FOR_RENDER_DOWNLOAD", "missing": missing}, indent=2), flush=True)
        raise SystemExit(2)
    aux_schemas = {}
    for method in methods:
        if method["new"]:
            for camera in cameras:
                p = Path(method["input_root"]) / args.aux_folder / f"{Path(camera['name']).stem}.npz"
                with np.load(p, allow_pickle=False) as archive:
                    aux_schemas[str(p)] = resolve_aux_keys(archive.files)
    if args.preflight:
        print(json.dumps({"status": "PASS_INPUT_FILES_AND_AUXILIARY_KEYS", "input_files": len(paths), "view_ids": VIEW_IDS, "aux_schemas": aux_schemas}, indent=2), flush=True)
        return
    require(not output.exists(), f"Refusing to overwrite existing evaluation: {output}")
    hashes = {str(p): {"bytes": p.stat().st_size, "sha256": sha256(p)} for p in paths}
    require(hashes[str(cloud)]["sha256"] == CLOUD_SHA, "Source point cloud hash differs from verified no-glass 1cm input")
    source = PlyData.read(cloud)["vertex"].data
    xyz = np.column_stack([source[k] for k in ("x", "y", "z")]).astype(np.float64)
    require(len(xyz) == 7_004_696 and np.isfinite(xyz).all(), "Source point data invalid")
    tree = cKDTree(xyz, balanced_tree=True, compact_nodes=True)
    output.mkdir(parents=True, exist_ok=False)
    save_json(output / "input_manifest.json", {"status": "FROZEN_FOR_CPU_EVALUATION", "files": hashes, "evaluator_sha256": sha256(Path(__file__)), "plotter_sha256": sha256(HERE / "plot_errors.py")})
    rows = []
    group_keys = None
    new_plys = {m["id"]: [] for m in methods if m["new"]}
    baseline_rows = {}
    if (baseline / "metrics/geometry_metrics.json").exists():
        baseline_rows = {(r["method"], r["image_id"]): r for r in load_json(baseline / "metrics/geometry_metrics.json")["per_view"]}
    for camera in cameras:
        name, stem = camera["name"], Path(camera["name"]).stem
        shape = (camera["height"], camera["width"])
        reference = np.load(baseline / "reference_noglass_mesh/depth_float" / f"{stem}.npy", allow_pickle=False)
        require(reference.dtype == np.float32 and reference.shape == shape and np.isfinite(reference).all(), "Reference depth invalid")
        reference_valid = reference > 0
        boundary = read_mask(baseline / "reference_noglass_mesh/discontinuity_masks" / name, shape)
        require(np.all(~boundary | reference_valid), "Reference boundary mask outside valid reference")
        interior = reference_valid & ~boundary
        data = {m["id"]: read_method(m, camera, xyz, args.aux_folder) for m in methods}
        common = reference_valid.copy()
        common_alpha50 = reference_valid.copy()
        for value in data.values():
            common &= value["mask"]
            common_alpha50 &= value["alpha50"]
        save_mask(output / "masks/common_all5" / name, common)
        save_mask(output / "masks/common_all5_alpha50" / name, common_alpha50)
        for method in methods:
            value = data[method["id"]]
            z, mask, rgb = value["z"], value["mask"], value["rgb"]
            world = backproject(camera, z, mask)
            world_float = world.astype(np.float32)
            distances, nearest = tree.query(world_float, k=1, workers=args.workers)
            require(np.isfinite(distances).all(), "Non-finite source NN distances")
            nn = np.full(shape, np.nan, dtype=np.float64)
            nn[mask] = distances
            error = np.full(shape, np.nan, dtype=np.float64)
            eligible = mask & reference_valid
            error[eligible] = z[eligible] - reference[eligible].astype(np.float64)
            array_root = output / "error_arrays" / method["id"]
            save_npy(array_root / f"{stem}_nn_m.npy", nn)
            save_npy(array_root / f"{stem}_signed_mesh_Z_m.npy", error)
            save_mask(output / "masks" / method["id"] / "primary" / name, mask)
            save_mask(output / "masks" / method["id"] / "alpha50" / name, value["alpha50"])
            if method["new"]:
                ply_path = output / "backprojected" / method["id"] / f"{stem}.ply"
                ply_record = write_ply(ply_path, world_float, rgb[mask])
                new_plys[method["id"]].append(ply_path)
            else:
                baseline_path = Path(method["input_root"]) / "backprojected" / f"{stem}.ply"
                loaded = PlyData.read(baseline_path)["vertex"].data
                require(np.array_equal(loaded, ply_array(world_float, rgb[mask])), "Current PNG backprojection differs from existing baseline PLY")
                ply_record = {"path": str(baseline_path), "points": len(loaded), "reused_read_only": True, "exact_match_to_PNG_backprojection": True}
            groups = metric_groups(nn, error, value["alpha50"], common, common_alpha50, interior, boundary)
            group_keys = list(groups)
            row = {"method": method["id"], "image_id": camera["image_id"], "name": name, "total_pixels": int(mask.size), "valid_pixels": int(mask.sum()), "valid_alpha50_pixels": int(value["alpha50"].sum()), "coverage_percent": float(mask.mean() * 100), "coverage_alpha50_percent": float(value["alpha50"].mean() * 100), "valid_without_reference_pixels": int((mask & ~reference_valid).sum()), "common_all5_pixels": int(common.sum()), "common_all5_alpha50_pixels": int(common_alpha50.sum()), "PNG_quantization": stats(value["quantization"]), "float32_PLY_storage_max_error_m": float(np.max(np.abs(world_float.astype(np.float64) - world))) if len(world) else 0.0, "auxiliary_validation": value["auxiliary_validation"], "PLY": ply_record, **{key: stats(values, signed) for key, (values, signed) in groups.items()}}
            if method["new"]:
                row["first_hit_diagnostics"] = value["auxiliary_validation"]["first_hit_diagnostics"]
            if not method["new"] and (method["id"], camera["image_id"]) in baseline_rows:
                old = baseline_rows[(method["id"], camera["image_id"])]
                for domain in ("point_to_input", "own_valid_depth"):
                    require(row[domain]["count"] == old[domain]["count"], "Baseline eligible pixel count changed")
                    if row[domain]["count"]:
                        require(abs(row[domain]["mean_mm"] - old[domain]["mean_mm"]) < 1e-7, "Baseline numeric evaluation changed")
                row["v007_baseline_own_domain_metrics_reproduced"] = True
            rows.append(row)
            del world, world_float, distances, nearest, nn, error, groups
        print(json.dumps({"view": camera["image_id"], "common_all5_pixels": int(common.sum()), "common_all5_alpha50_pixels": int(common_alpha50.sum()), "coverage_percent": {m["id"]: float(data[m["id"]]["mask"].mean() * 100) for m in methods}}), flush=True)
        del data
    aggregate = {}
    for method in methods:
        method_rows = [r for r in rows if r["method"] == method["id"]]
        total = sum(r["total_pixels"] for r in method_rows)
        count = sum(r["valid_pixels"] for r in method_rows)
        count_alpha = sum(r["valid_alpha50_pixels"] for r in method_rows)
        pooled = {key: [] for key in group_keys}
        signed_flags = {}
        for camera in cameras:
            name, stem = camera["name"], Path(camera["name"]).stem
            shape = (camera["height"], camera["width"])
            folder = output / "error_arrays" / method["id"]
            nn = np.load(folder / f"{stem}_nn_m.npy", mmap_mode="r", allow_pickle=False)
            error = np.load(folder / f"{stem}_signed_mesh_Z_m.npy", mmap_mode="r", allow_pickle=False)
            reference = np.load(baseline / "reference_noglass_mesh/depth_float" / f"{stem}.npy", mmap_mode="r", allow_pickle=False)
            boundary = read_mask(baseline / "reference_noglass_mesh/discontinuity_masks" / name, shape)
            alpha50 = read_mask(output / "masks" / method["id"] / "alpha50" / name, shape)
            common = read_mask(output / "masks/common_all5" / name, shape)
            common_alpha50 = read_mask(output / "masks/common_all5_alpha50" / name, shape)
            for key, (values, signed) in metric_groups(nn, error, alpha50, common, common_alpha50, (reference > 0) & ~boundary, boundary).items():
                pooled[key].append(values)
                signed_flags[key] = signed
        summary = {key: stats(np.concatenate(parts), signed_flags[key]) for key, parts in pooled.items()}
        del pooled
        summary.update(method=method["id"], label=method["label"], views=10, total_pixels=total, valid_pixels=count, valid_alpha50_pixels=count_alpha, coverage_percent=count * 100 / total, coverage_alpha50_percent=count_alpha * 100 / total, valid_without_reference_pixels=sum(r["valid_without_reference_pixels"] for r in method_rows))
        if method["new"]:
            summary["merged_PLY"] = merge_plys(output / "backprojected" / method["id"] / "merged_10views.ply", new_plys[method["id"]])
            qmin_parts, alpha_parts = [], []
            for camera in cameras:
                path = Path(method["input_root"]) / args.aux_folder / f"{Path(camera['name']).stem}.npz"
                with np.load(path, allow_pickle=False) as archive:
                    mapping = resolve_aux_keys(archive.files)
                    hit = archive[mapping["first_id"]] >= 0
                    qmin_parts.append(archive[mapping["qmin"]][hit])
                    alpha_parts.append(archive[mapping["first_alpha"]][hit])
            summary["first_hit_diagnostics"] = first_hit_stats(np.concatenate(qmin_parts), np.concatenate(alpha_parts))
            summary["first_hit_diagnostics"]["primary_valid_hit_count"] = count
            del qmin_parts, alpha_parts
        aggregate[method["id"]] = summary
    require(len({r["common_all5_depth"]["count"] for r in aggregate.values()}) == 1, "All-five common supports differ")
    require(len({r["common_all5_alpha50_depth"]["count"] for r in aggregate.values()}) == 1, "All-five alpha50 common supports differ")
    protocol = {
        "cameras": cameras, "methods": methods, "source_cloud": str(cloud), "source_cloud_sha256": CLOUD_SHA,
        "depth_source_for_metrics_and_PLY": "Saved single-channel uint16 PNG decoded as camera Z metres by dividing by1000; no float-depth substitution",
        "primary_CDE_validity": "First-hit ID >=0 AND raw depth positive and finite; verified identical to saved PNG/mask; no alpha50 cutoff",
        "alpha50_policy": "C/D/E depth_mask_alpha50 is primary mask AND accumulated float alpha>=0.5; A primary depth already alpha50; B occupied pixels are retained (no alpha model). first_alpha is single-contributor alpha and is NOT accumulated alpha.",
        "qmin_gt9_policy": "Retained by actual first-contributor raypeak logic, with entry_z=exit_z=0; no finite 3-sigma ellipsoid intersection. Counts and first_alpha distributions are reported separately, never silently gated out.",
        "world_to_camera": "pc=world@R.T+t, COLMAP Hamilton wxyz",
        "backprojection": "pc=[(x+0.5-cx)/fx, (y+0.5-cy)/fy, 1]*Z; world=(pc-t)@R",
        "pixel_centres": [0.5, 0.5], "world_units": "metres", "depth_PNG_units": "integer millimetres,0 invalid",
        "PLY_storage": "float32 XYZ, uint8 RGB sampled from matching saved RGB PNG; new C/D/E each10 views plus unfiltered merged file",
        "NN": "Exact one-sided Euclidean nearest neighbour from float32 exported/backprojected points to all7004696 original no-glass source points; no ICP",
        "mesh_reference": "Unchanged v007 no-glass first-hit camera-Z float32 arrays, near0.2m and unnormalized rays already independently audited; no mesh used to filter exported points",
        "own_valid_domain": "method primary validity AND mesh first-hit validity for Z errors; NN uses every primary valid point",
        "common_all5_domain": "All five primary depth masks AND mesh reference validity, identical across methods",
        "common_all5_alpha50_domain": "All five alpha50 eligibility masks AND mesh reference validity, identical across methods",
        "interior_boundary": "Reuse fixed v0073x3 mesh-depth discontinuity masks; both own-domain and matched all-five slices reported",
        "aggregation": "Pool eligible pixels/points over10 views; unequal coverage is not hidden by per-view means",
        "metric_precision": "Float64 metric computation and saved error arrays; NN queries use float32 PLY coordinates and float64 source cloud",
        "error_array_invalid": "NaN outside the stated eligible domain; all evaluated values must be finite",
        "representative_views": args.representatives, "representative_selection": "Fixed before v008 results; v007 B coverage spans a broad range; no v008 result-based selection",
        "limitations": ["NN proximity to any source surface cannot establish visibility, completeness, or correct first-hit ordering; B is structurally favored because it selects original samples.", "Own-valid scores have different support; all-five common scores and coverage must be read together.", "The common domain is restricted by sparse methods and does not measure errors in their missing regions.", "A first Gaussian centre or density peak is not guaranteed to lie on the original mesh surface.", "Signed mesh error is a synthetic first-surface proxy after near clipping, not independent real-world ground truth.", "Both positive and negative errors beyond30mm are reported; low background leakage alone is not an occlusion pass.", "Merged clouds preserve repeated surfaces from multiple views; no fusion or deduplication is performed."],
    }
    metrics = {"status": "EVALUATED_FIVE_DEPTH_METHODS_NOT_TRAINING_OR_OCCLUSION_PASS", "protocol": protocol, "aggregate": aggregate, "per_view": rows, "elapsed_seconds_before_figures": time.perf_counter() - started}
    save_json(output / "geometry_metrics.json", metrics)
    write_csv(output / "per_view.csv", [flat_metric_row(r) for r in rows])
    write_csv(output / "aggregate.csv", [flat_metric_row(aggregate[m["id"]]) for m in methods])
    figure_manifest = None
    if not args.no_figures:
        from plot_errors import render_figures
        figure_manifest = render_figures(output, output / "figures", args.representatives, args.nn_max_mm, args.signed_z_limit_mm)
    after = {str(p): {"bytes": p.stat().st_size, "sha256": sha256(p)} for p in paths}
    require(after == hashes, "Input files changed during evaluation")
    save_json(output / "verification.json", {"status": "PASS_FORMATS_COORDINATES_MASKS_PROVENANCE_PLY_AND_INPUT_PRESERVATION", "methods": 5, "views_per_method": 10, "saved_depth_PNGs_read": 50, "new_per_view_PLYs": 30, "new_merged_PLYs": 3, "input_files_unchanged": len(paths), "all_primary_CDE_masks_verified_from_first_hit_ID_and_depth": True, "alpha50_not_used_to_filter_primary_PLYs": True, "common_domains_equal_across_all5": True, "baseline_PLYs_exactly_reproduced_read_only": True, "figure_status": figure_manifest["status"] if figure_manifest else "NOT_GENERATED", "visual_review_still_required": bool(figure_manifest), "occlusion_quality_is_measured_not_assumed": True, "elapsed_seconds": time.perf_counter() - started})
    print(json.dumps({"status": metrics["status"], "output": str(output), "aggregate": {m: {"coverage_percent": r["coverage_percent"], "NN_mean_mm": r["point_to_input"]["mean_mm"], "common_all5_MAE_mm": r["common_all5_depth"]["mean_mm"], "common_all5_behind_gt30mm_percent": r["common_all5_depth"]["behind_first_surface_gt30mm_percent"]} for m, r in aggregate.items()}}, indent=2), flush=True)


if __name__ == "__main__":
    main()
