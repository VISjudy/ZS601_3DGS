"""Reproducible SfM/LiDAR preprocessing for the ZS601 v3 experiments.

The module deliberately has no dependency on the training entry point.  Raw inputs are
read-only; every generated artifact lives below ``processed_v3`` and existing files are
rejected unless ``overwrite`` is explicitly enabled.
"""
from __future__ import annotations

import csv
import hashlib
import json
import shutil
import struct
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image

try:
    from scipy.spatial import cKDTree
except ImportError:  # pragma: no cover
    cKDTree = None

try:
    from plyfile import PlyData, PlyElement
except ImportError:  # pragma: no cover
    PlyData = PlyElement = None


PREPROCESS_VERSION = 3
COLMAP_MODELS = {
    0: ("SIMPLE_PINHOLE", 3), 1: ("PINHOLE", 4), 2: ("SIMPLE_RADIAL", 4),
    3: ("RADIAL", 5), 4: ("OPENCV", 8), 5: ("OPENCV_FISHEYE", 8),
    6: ("FULL_OPENCV", 12), 7: ("FOV", 5), 8: ("SIMPLE_RADIAL_FISHEYE", 4),
    9: ("RADIAL_FISHEYE", 5), 10: ("THIN_PRISM_FISHEYE", 12),
}


@dataclass
class Camera:
    image_id: int
    camera_id: int
    name: str
    width: int
    height: int
    model: str
    params: np.ndarray
    rotation: np.ndarray  # world -> camera
    translation: np.ndarray

    @property
    def center(self) -> np.ndarray:
        return -(self.rotation.T @ self.translation)

    @property
    def intrinsics(self) -> tuple[float, float, float, float]:
        if self.model in {"SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL", "SIMPLE_RADIAL_FISHEYE", "RADIAL_FISHEYE"}:
            f, cx, cy = self.params[:3]
            return float(f), float(f), float(cx), float(cy)
        if self.model in {"PINHOLE", "OPENCV", "OPENCV_FISHEYE", "FULL_OPENCV", "FOV", "THIN_PRISM_FISHEYE"}:
            fx, fy, cx, cy = self.params[:4]
            return float(fx), float(fy), float(cx), float(cy)
        raise ValueError(f"Unsupported COLMAP camera model: {self.model}")


@dataclass
class NormalSettings:
    knn: int = 24
    max_radius: float | None = None
    min_neighbors: int = 8
    max_curvature: float = 0.15
    camera_count: int = 8
    min_orientation_confidence: float = 0.05
    orient_camera: bool = True
    propagate_unresolved: bool = False
    filter_outliers: bool = False
    outlier_neighbors: int = 16
    outlier_sigma: float = 3.0
    orientation_occlusion: bool = False
    orientation_block_size: int = 250000

    def validate(self) -> None:
        if self.knn < 3 or self.min_neighbors < 3 or self.min_neighbors > self.knn:
            raise ValueError("Require 3 <= min_neighbors <= knn")
        if self.max_radius is not None and self.max_radius <= 0:
            raise ValueError("max_radius must be positive")
        if not 0 <= self.max_curvature <= 1:
            raise ValueError("max_curvature must be in [0, 1]")
        if self.camera_count < 1 or self.orientation_block_size < 1:
            raise ValueError("camera_count and orientation_block_size must be positive")
        if not 0 <= self.min_orientation_confidence <= 1:
            raise ValueError("min_orientation_confidence must be in [0, 1]")
        if self.outlier_neighbors < 2 or self.outlier_sigma <= 0:
            raise ValueError("Invalid statistical outlier settings")


def _json_dump(path: Path, value: Any, overwrite: bool = False) -> None:
    _guard_output(path, overwrite)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _guard_output(path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing output: {path}")


def sha256_file(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def file_identity(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns,
            "sha256": sha256_file(path)}


def qvec_to_rotmat(qvec: Sequence[float]) -> np.ndarray:
    w, x, y, z = map(float, qvec)
    return np.array([
        [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
        [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
        [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y],
    ], dtype=np.float64)


def _read_next_bytes(fid: Any, count: int, fmt: str) -> tuple[Any, ...]:
    return struct.unpack("<" + fmt, fid.read(count))


def read_colmap_cameras(sparse_dir: Path, source: str = "auto") -> dict[int, tuple[str, int, int, np.ndarray]]:
    txt = sparse_dir / "cameras.txt"
    if source not in {"auto", "text", "binary"}:
        raise ValueError("source must be auto, text or binary")
    if txt.exists() and source != "binary":
        result = {}
        for line in txt.read_text(encoding="utf-8").splitlines():
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            fields = line.split()
            result[int(fields[0])] = (fields[1], int(fields[2]), int(fields[3]), np.asarray(fields[4:], dtype=np.float64))
        return result
    if source == "text":
        raise FileNotFoundError(txt)
    binary = sparse_dir / "cameras.bin"
    if not binary.exists():
        raise FileNotFoundError(f"Neither cameras.txt nor cameras.bin exists in {sparse_dir}")
    result = {}
    with binary.open("rb") as fid:
        count = _read_next_bytes(fid, 8, "Q")[0]
        for _ in range(count):
            camera_id, model_id, width, height = _read_next_bytes(fid, 24, "iiQQ")
            if model_id not in COLMAP_MODELS:
                raise ValueError(f"Unknown COLMAP camera model id: {model_id}")
            model, nparams = COLMAP_MODELS[model_id]
            params = np.asarray(_read_next_bytes(fid, 8 * nparams, "d" * nparams))
            result[camera_id] = (model, width, height, params)
    return result


def read_colmap_images(sparse_dir: Path, source: str = "auto") -> dict[int, tuple[np.ndarray, np.ndarray, int, str]]:
    txt = sparse_dir / "images.txt"
    if source not in {"auto", "text", "binary"}:
        raise ValueError("source must be auto, text or binary")
    if txt.exists() and source != "binary":
        # Keep blank point2D lines: every image record occupies two physical lines.
        lines = [line for line in txt.read_text(encoding="utf-8").splitlines()
                 if not line.lstrip().startswith("#")]
        result = {}
        index = 0
        while index < len(lines):
            if not lines[index].strip():
                index += 1
                continue
            fields = lines[index].split()
            if len(fields) < 10:
                raise ValueError(f"Malformed COLMAP image pose line {index + 1} in {txt}")
            result[int(fields[0])] = (
                np.asarray(fields[1:5], dtype=np.float64), np.asarray(fields[5:8], dtype=np.float64),
                int(fields[8]), " ".join(fields[9:]),
            )
            index += 2
        return result
    if source == "text":
        raise FileNotFoundError(txt)
    binary = sparse_dir / "images.bin"
    if not binary.exists():
        raise FileNotFoundError(f"Neither images.txt nor images.bin exists in {sparse_dir}")
    result = {}
    with binary.open("rb") as fid:
        count = _read_next_bytes(fid, 8, "Q")[0]
        for _ in range(count):
            props = _read_next_bytes(fid, 64, "idddddddi")
            image_id, qvec, tvec, camera_id = props[0], props[1:5], props[5:8], props[8]
            name = bytearray()
            while True:
                char = fid.read(1)
                if char == b"\x00":
                    break
                name.extend(char)
            point_count = _read_next_bytes(fid, 8, "Q")[0]
            fid.seek(24 * point_count, 1)
            result[image_id] = (np.asarray(qvec), np.asarray(tvec), camera_id, name.decode("utf-8"))
    return result


def load_cameras(sparse_dir: Path, image_names: set[str] | None = None) -> list[Camera]:
    calibrations = read_colmap_cameras(sparse_dir)
    poses = read_colmap_images(sparse_dir)
    cameras = []
    for image_id, (qvec, tvec, camera_id, name) in sorted(poses.items()):
        if image_names is not None and name not in image_names:
            continue
        model, width, height, params = calibrations[camera_id]
        if model not in {"SIMPLE_PINHOLE", "PINHOLE"}:
            raise ValueError(
                f"Camera {camera_id} uses {model}; preprocess_v3 requires already-undistorted "
                "images with SIMPLE_PINHOLE or PINHOLE intrinsics")
        cameras.append(Camera(image_id, camera_id, name, width, height, model, params, qvec_to_rotmat(qvec), tvec))
    return cameras


def read_name_list(path: Path | None) -> set[str] | None:
    if path is None or not path.exists():
        return None
    names = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        value = line.strip()
        if value and not value.startswith("#"):
            names.add(value.split()[-1])
    return names


def read_ply(path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    if PlyData is None:
        raise RuntimeError("Reading PLY requires the optional dependency 'plyfile'")
    vertex = PlyData.read(str(path))["vertex"].data
    names = set(vertex.dtype.names or ())
    for required in ("x", "y", "z"):
        if required not in names:
            raise ValueError(f"{path} has no '{required}' vertex field")
    xyz = np.column_stack([vertex[k] for k in ("x", "y", "z")]).astype(np.float64)
    if {"red", "green", "blue"}.issubset(names):
        raw_rgb = np.column_stack([vertex[k] for k in ("red", "green", "blue")])
        if raw_rgb.dtype.kind == "f" and len(raw_rgb) and np.nanmax(raw_rgb) <= 1.0:
            raw_rgb = raw_rgb * 255.0
        if len(raw_rgb) and (np.nanmin(raw_rgb) < 0 or np.nanmax(raw_rgb) > 255):
            raise ValueError(f"{path} RGB fields must be in [0,1] or [0,255]")
        rgb = np.rint(raw_rgb).astype(np.uint8)
    else:
        rgb = np.full((len(xyz), 3), 127, dtype=np.uint8)
    extras = {name: np.asarray(vertex[name]) for name in names - {"x", "y", "z", "red", "green", "blue"}}
    return xyz, rgb, extras


def write_ply(path: Path, fields: dict[str, np.ndarray], overwrite: bool = False) -> None:
    if PlyData is None or PlyElement is None:
        raise RuntimeError("Writing PLY requires the optional dependency 'plyfile'")
    _guard_output(path, overwrite)
    path.parent.mkdir(parents=True, exist_ok=True)
    lengths = {len(np.asarray(value)) for value in fields.values()}
    if len(lengths) != 1:
        raise ValueError("PLY fields must have the same length")
    dtype = []
    normalized = {}
    for name, value in fields.items():
        array = np.asarray(value)
        if array.dtype.kind == "b":
            array = array.astype(np.uint8)
        if array.dtype.kind in "iu" and name not in {"red", "green", "blue"}:
            array = array.astype(np.int32)
        if array.dtype.kind == "f":
            array = array.astype(np.float32)
        normalized[name] = array
        dtype.append((name, array.dtype.str))
    data = np.empty(next(iter(lengths), 0), dtype=dtype)
    for name, value in normalized.items():
        data[name] = value
    PlyData([PlyElement.describe(data, "vertex")], text=False).write(str(path))


def estimate_normals(points: np.ndarray, settings: NormalSettings) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return normals, curvature, confidence, neighbor_count and valid mask."""
    if cKDTree is None:
        raise RuntimeError("PCA KNN normal estimation requires scipy")
    settings.validate()
    if len(points) < 3:
        raise ValueError("At least three points are required for normal estimation")
    k = min(max(settings.knn, 3), len(points))
    distances, indices = cKDTree(points).query(points, k=k)
    if k == 1:
        distances, indices = distances[:, None], indices[:, None]
    normals = np.zeros_like(points, dtype=np.float64)
    curvature = np.ones(len(points), dtype=np.float64)
    counts = np.zeros(len(points), dtype=np.int32)
    for row in range(len(points)):
        keep = np.isfinite(distances[row])
        if settings.max_radius is not None:
            keep &= distances[row] <= settings.max_radius
        neighbors = points[indices[row][keep]]
        counts[row] = len(neighbors)
        if len(neighbors) < 3:
            continue
        centered = neighbors - neighbors.mean(axis=0)
        eigenvalues, eigenvectors = np.linalg.eigh(centered.T @ centered / len(neighbors))
        normals[row] = eigenvectors[:, 0]
        total = max(float(eigenvalues.sum()), np.finfo(float).eps)
        curvature[row] = max(float(eigenvalues[0]), 0.0) / total
    confidence = np.clip(1.0 - curvature / max(settings.max_curvature, 1e-12), 0.0, 1.0)
    valid = (counts >= settings.min_neighbors) & (curvature <= settings.max_curvature) & (np.linalg.norm(normals, axis=1) > 0.5)
    return normals, curvature, confidence, counts, valid


def project(camera: Camera, points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    camera_xyz = points @ camera.rotation.T + camera.translation
    fx, fy, cx, cy = camera.intrinsics
    z = camera_xyz[:, 2]
    uv = np.column_stack((fx * camera_xyz[:, 0] / np.where(z == 0, 1, z) + cx,
                          fy * camera_xyz[:, 1] / np.where(z == 0, 1, z) + cy))
    inside = (z > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < camera.width) & (uv[:, 1] >= 0) & (uv[:, 1] < camera.height)
    return camera_xyz, uv, inside


def _camera_output_key(name: str) -> Path:
    value = Path(name)
    if value.is_absolute() or ".." in value.parts:
        raise ValueError(f"Unsafe camera image name: {name}")
    return value.with_suffix("")


def _mask_for_camera(mask_dir: Path | None, camera: Camera, uv: np.ndarray, indices: np.ndarray,
                     valid_when: str = "white") -> np.ndarray:
    if mask_dir is None:
        return np.ones(len(indices), dtype=bool)
    candidates = [mask_dir / camera.name, mask_dir / Path(camera.name).with_suffix(".png")]
    path = next((item for item in candidates if item.exists()), None)
    if path is None:
        return np.ones(len(indices), dtype=bool)
    mask = np.asarray(Image.open(path).convert("L"))
    if mask.shape != (camera.height, camera.width):
        raise ValueError(f"Mask size {mask.shape[::-1]} does not match camera "
                         f"{camera.name} size {(camera.width, camera.height)}")
    x = np.rint(uv[indices, 0]).astype(int)
    y = np.rint(uv[indices, 1]).astype(int)
    values = mask[y, x]
    if valid_when == "white":
        return values >= 128
    if valid_when == "black":
        return values < 128
    raise ValueError("mask_valid_when must be 'white' or 'black'")


def orient_normals_to_cameras(points: np.ndarray, normals: np.ndarray, cameras: Sequence[Camera],
                              valid: np.ndarray, settings: NormalSettings, mask_dir: Path | None = None,
                              mask_valid_when: str = "white"
                              ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    """Resolve PCA signs by weighted votes from the nearest visible training cameras."""
    npoints = len(points)
    k = min(settings.camera_count, max(len(cameras), 1))
    best_distance = np.full((npoints, k), np.inf)
    best_score = np.zeros((npoints, k), dtype=np.float64)
    visibility = np.zeros(npoints, dtype=np.int32)
    per_camera: dict[str, int] = {}
    for camera in cameras:
        camera_visible = 0
        occlusion_visible = None
        if settings.orientation_occlusion:
            occlusion_visible = np.zeros(npoints, dtype=bool)
            occlusion_visible[_zbuffer(camera, points, 1e-6, np.inf)[0]] = True
        for start in range(0, npoints, settings.orientation_block_size):
            stop = min(start + settings.orientation_block_size, npoints)
            _, uv, inside = project(camera, points[start:stop])
            local_indices = np.flatnonzero(inside & valid[start:stop])
            if not len(local_indices):
                continue
            indices = local_indices + start
            if occlusion_visible is not None:
                keep = occlusion_visible[indices]
                local_indices, indices = local_indices[keep], indices[keep]
            if len(indices):
                keep = _mask_for_camera(mask_dir, camera, uv, local_indices, mask_valid_when)
                indices = indices[keep]
            if not len(indices):
                continue
            camera_visible += len(indices)
            vectors = camera.center[None, :] - points[indices]
            distance = np.linalg.norm(vectors, axis=1)
            direction = vectors / np.maximum(distance[:, None], 1e-12)
            signed = np.einsum("ij,ij->i", normals[indices], direction)
            weight = np.maximum(np.abs(signed), 0.05) / np.maximum(distance, 1e-6) ** 2
            scores = signed * weight
            visibility[indices] += 1
            slots = np.argmax(best_distance[indices], axis=1)
            rows = np.arange(len(indices))
            replace = distance < best_distance[indices, slots]
            best_distance[indices[replace], slots[replace]] = distance[replace]
            best_score[indices[replace], slots[replace]] = scores[replace]
        per_camera[camera.name] = int(camera_visible)
    total_weight = np.sum(np.abs(best_score), axis=1)
    vote = np.sum(best_score, axis=1)
    confidence = np.divide(np.abs(vote), total_weight, out=np.zeros_like(vote), where=total_weight > 0)
    resolved = valid & (visibility > 0) & (confidence >= settings.min_orientation_confidence)
    flipped = resolved & (vote < 0)
    oriented = normals.copy()
    oriented[flipped] *= -1
    return oriented, confidence, resolved, flipped, per_camera


def propagate_normal_signs(points: np.ndarray, normals: np.ndarray, resolved: np.ndarray,
                           valid: np.ndarray, neighbors: int = 8) -> tuple[np.ndarray, np.ndarray]:
    if cKDTree is None:
        raise RuntimeError("Normal sign propagation requires scipy")
    oriented, resolved = normals.copy(), resolved.copy()
    if not resolved.any():
        return oriented, resolved
    _, indices = cKDTree(points).query(points, k=min(max(neighbors + 1, 2), len(points)))
    for _ in range(8):
        changed = 0
        for point_index in np.flatnonzero(valid & ~resolved):
            adjacent = np.atleast_1d(indices[point_index])[1:]
            reference = adjacent[resolved[adjacent]]
            if not len(reference):
                continue
            mean = oriented[reference].mean(axis=0)
            if np.linalg.norm(mean) <= 1e-12:
                continue
            if np.dot(oriented[point_index], mean) < 0:
                oriented[point_index] *= -1
            resolved[point_index] = True
            changed += 1
        if not changed:
            break
    return oriented, resolved


def _filter_statistical(points: np.ndarray, settings: NormalSettings) -> np.ndarray:
    if cKDTree is None:
        raise RuntimeError("Statistical outlier filtering requires scipy")
    k = min(max(settings.outlier_neighbors, 2), len(points))
    distances, _ = cKDTree(points).query(points, k=k)
    mean_distance = distances[:, 1:].mean(axis=1)
    limit = np.median(mean_distance) + settings.outlier_sigma * np.std(mean_distance)
    return mean_distance <= limit


def preprocess_lidar(lidar_path: Path, output_dir: Path, settings: NormalSettings,
                     cameras: Sequence[Camera], mask_dir: Path | None = None,
                     mask_valid_when: str = "white", overwrite: bool = False,
                     estimate_normals_enabled: bool = True) -> dict[str, Any]:
    settings.validate()
    geometry = output_dir / "geometry"
    targets = [geometry / "lidar_static_rgb.ply", geometry / "lidar_static_rgb_normal_oriented.ply",
               geometry / "normal_orientation.csv", geometry / "normal_orientation_summary.json"]
    for target in targets:
        _guard_output(target, overwrite)
    points, colors, extras = read_ply(lidar_path)
    finite = np.isfinite(points).all(axis=1)
    clean_points, clean_colors = points[finite], colors[finite]
    filter_keep = np.ones(len(clean_points), dtype=bool)
    if settings.filter_outliers:
        filter_keep = _filter_statistical(clean_points, settings)
        clean_points, clean_colors = clean_points[filter_keep], clean_colors[filter_keep]
    if estimate_normals_enabled:
        normals, curvature, normal_confidence, counts, normal_valid = estimate_normals(clean_points, settings)
        normal_source = "pca_knn"
    else:
        missing = {"nx", "ny", "nz"} - extras.keys()
        if missing:
            raise ValueError(
                "PCA normal estimation is disabled, but the input PLY has no complete "
                f"nx/ny/nz fields (missing {sorted(missing)})")
        normals = np.column_stack([extras["nx"], extras["ny"], extras["nz"]]).astype(np.float64)[finite]
        if settings.filter_outliers:
            normals = normals[filter_keep]
        lengths = np.linalg.norm(normals, axis=1)
        normal_valid = np.isfinite(normals).all(axis=1) & (lengths > 1e-12)
        normals[normal_valid] /= lengths[normal_valid, None]
        curvature = np.zeros(len(clean_points), dtype=np.float64)
        normal_confidence = normal_valid.astype(np.float64)
        counts = np.zeros(len(clean_points), dtype=np.int32)
        normal_source = "input_ply"
    if settings.orient_camera:
        normals, orientation_confidence, resolved, flipped, per_camera = orient_normals_to_cameras(
            clean_points, normals, cameras, normal_valid, settings, mask_dir, mask_valid_when)
    else:
        orientation_confidence = np.zeros(len(clean_points))
        resolved = normal_valid.copy()
        flipped = np.zeros(len(clean_points), dtype=bool)
        per_camera = {}
    camera_resolved = resolved.copy()
    if settings.propagate_unresolved:
        normals, resolved = propagate_normal_signs(clean_points, normals, resolved, normal_valid)
    final_valid = normal_valid & resolved
    write_ply(geometry / "lidar_static_rgb.ply", {
        "x": clean_points[:, 0], "y": clean_points[:, 1], "z": clean_points[:, 2],
        "red": clean_colors[:, 0], "green": clean_colors[:, 1], "blue": clean_colors[:, 2],
    }, overwrite)
    write_ply(geometry / "lidar_static_rgb_normal_oriented.ply", {
        "x": clean_points[:, 0], "y": clean_points[:, 1], "z": clean_points[:, 2],
        "red": clean_colors[:, 0], "green": clean_colors[:, 1], "blue": clean_colors[:, 2],
        "nx": normals[:, 0], "ny": normals[:, 1], "nz": normals[:, 2],
        "normal_confidence": normal_confidence, "orientation_confidence": orientation_confidence,
        "curvature": curvature, "neighbor_count": counts, "normal_valid": final_valid,
    }, overwrite)
    csv_path = geometry / "normal_orientation.csv"
    _guard_output(csv_path, overwrite)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["point_index", "normal_confidence", "orientation_confidence", "flipped", "resolved", "normal_valid"])
        writer.writerows(zip(range(len(clean_points)), normal_confidence, orientation_confidence,
                             flipped.astype(int), resolved.astype(int), final_valid.astype(int)))
    norm_length = np.linalg.norm(normals, axis=1)
    summary = {
        "version": PREPROCESS_VERSION, "source": str(lidar_path), "source_sha256": sha256_file(lidar_path),
        "normal_source": normal_source, "pca_normal_estimation": estimate_normals_enabled,
        "input_count": int(len(points)), "finite_count": int(finite.sum()), "output_count": int(len(clean_points)),
        "filter_outliers": settings.filter_outliers, "filter_removed": int((~filter_keep).sum()),
        "flipped_count": int(flipped.sum()), "flipped_ratio": float(flipped.mean()),
        "camera_resolved_count": int(camera_resolved.sum()),
        "propagated_count": int((resolved & ~camera_resolved).sum()),
        "resolved_count": int(resolved.sum()), "resolved_ratio": float(resolved.mean()),
        "unresolved_count": int((~resolved).sum()), "unresolved_ratio": float((~resolved).mean()),
        "valid_count": int(final_valid.sum()), "valid_ratio": float(final_valid.mean()),
        "normal_length": _stats(norm_length), "orientation_confidence": _stats(orientation_confidence[resolved]),
        "curvature": _stats(curvature), "visible_points_per_camera": per_camera, "settings": asdict(settings),
        "mask_valid_when": mask_valid_when,
        "orientation_weighting": {"distance": "inverse_square", "angle": "max(abs(dot), 0.05)",
                                    "nearest_camera_count": settings.camera_count},
        "inputs": {"lidar": file_identity(lidar_path)},
    }
    _json_dump(geometry / "normal_orientation_summary.json", summary, overwrite)
    return summary


def _stats(values: np.ndarray) -> dict[str, float | None]:
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    if not len(values):
        return {key: None for key in ("min", "mean", "median", "p90", "p95", "max")}
    return {"min": float(values.min()), "mean": float(values.mean()), "median": float(np.median(values)),
            "p90": float(np.quantile(values, .90)), "p95": float(np.quantile(values, .95)), "max": float(values.max())}


def _zbuffer(camera: Camera, points: np.ndarray, near: float, far: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    camera_xyz, uv, inside = project(camera, points)
    rounded = np.rint(uv).astype(np.int64)
    valid = inside & (camera_xyz[:, 2] >= near) & (camera_xyz[:, 2] <= far)
    valid &= (rounded[:, 0] >= 0) & (rounded[:, 0] < camera.width) & (rounded[:, 1] >= 0) & (rounded[:, 1] < camera.height)
    point_indices = np.flatnonzero(valid)
    pixels = rounded[point_indices]
    linear = pixels[:, 1] * camera.width + pixels[:, 0]
    order = np.lexsort((camera_xyz[point_indices, 2], linear))
    sorted_linear = linear[order]
    first = np.r_[True, sorted_linear[1:] != sorted_linear[:-1]]
    selected = point_indices[order[first]]
    return selected, rounded[selected], camera_xyz[selected], uv


def generate_supervision(points: np.ndarray, normals: np.ndarray, normal_valid: np.ndarray,
                         cameras: Sequence[Camera], output_dir: Path, image_dir: Path | None = None,
                         mask_dir: Path | None = None, near: float = 0.01, far: float = 1e6,
                         mask_valid_when: str = "white", backprojection_samples: int = 3,
                         overwrite: bool = False) -> dict[str, Any]:
    if near <= 0 or far <= near:
        raise ValueError("Require 0 < near < far")
    if len(points) != len(normals) or len(points) != len(normal_valid):
        raise ValueError("points, normals and normal_valid must have equal lengths")
    planned: list[Path] = [output_dir / "manifests" / "supervision_report.json"]
    camera_paths: list[dict[str, Path]] = []
    for camera_index, camera in enumerate(cameras):
        key = _camera_output_key(camera.name)
        paths = {
            "normal_png": output_dir / "supervision" / "normal" / key.with_suffix(".png"),
            "normal_npy": output_dir / "supervision" / "normal_float" / key.with_suffix(".npy"),
            "normal_valid": output_dir / "supervision" / "normal_valid" / key.with_suffix(".png"),
            "depth_npy": output_dir / "supervision" / "depth" / key.with_suffix(".npy"),
            "depth_preview": output_dir / "supervision" / "depth_preview" / key.with_suffix(".png"),
            "depth_valid": output_dir / "supervision" / "depth_valid" / key.with_suffix(".png"),
        }
        if image_dir is not None and (image_dir / camera.name).exists():
            if Image.open(image_dir / camera.name).size != (camera.width, camera.height):
                raise ValueError(f"Image size does not match camera {camera.name}")
            paths["overlay"] = output_dir / "previews" / "normal_overlay" / key.with_suffix(".png")
        _mask_for_camera(mask_dir, camera, np.empty((0, 2)), np.empty(0, dtype=int), mask_valid_when)
        if camera_index < backprojection_samples:
            paths["backproject_source"] = output_dir / "previews" / "backprojection" / key.parent / f"{key.name}_source.ply"
            paths["backproject_reconstructed"] = output_dir / "previews" / "backprojection" / key.parent / f"{key.name}_reconstructed.ply"
        camera_paths.append(paths)
        planned.extend(paths.values())
    if len(set(planned)) != len(planned):
        raise ValueError("Camera names map to duplicate supervision output paths")
    for target in planned:
        _guard_output(target, overwrite)
    camera_reports = []
    for camera_index, camera in enumerate(cameras):
        paths = camera_paths[camera_index]
        selected, pixels, camera_xyz, all_uv = _zbuffer(camera, points, near, far)
        zbuffer_count = len(selected)
        mask_keep = (_mask_for_camera(mask_dir, camera, all_uv, selected, mask_valid_when)
                     if len(selected) else np.empty(0, dtype=bool))
        mask_excluded = int((~mask_keep).sum())
        depth_selected = selected[mask_keep]
        depth_pixels = pixels[mask_keep]
        depth_camera_xyz = camera_xyz[mask_keep]
        normal_keep = normal_valid[depth_selected]
        low_confidence = int((~normal_keep).sum())
        normal_selected = depth_selected[normal_keep]
        normal_pixels = depth_pixels[normal_keep]
        normal_camera_xyz = depth_camera_xyz[normal_keep]
        depth = np.zeros((camera.height, camera.width), dtype=np.float32)
        normal_map = np.zeros((camera.height, camera.width, 3), dtype=np.float32)
        depth_valid_map = np.zeros((camera.height, camera.width), dtype=np.uint8)
        normal_valid_map = np.zeros((camera.height, camera.width), dtype=np.uint8)
        if len(depth_selected):
            dx, dy = depth_pixels[:, 0], depth_pixels[:, 1]
            depth[dy, dx] = depth_camera_xyz[:, 2].astype(np.float32)
            depth_valid_map[dy, dx] = 255
        if len(normal_selected):
            x, y = normal_pixels[:, 0], normal_pixels[:, 1]
            camera_normals = normals[normal_selected] @ camera.rotation.T
            view = -normal_camera_xyz / np.maximum(np.linalg.norm(normal_camera_xyz, axis=1, keepdims=True), 1e-12)
            flip = np.einsum("ij,ij->i", camera_normals, view) < 0
            camera_normals[flip] *= -1
            normal_map[y, x] = camera_normals.astype(np.float32)
            normal_valid_map[y, x] = 255
        encoded = np.clip(np.rint((normal_map + 1.0) * 127.5), 0, 255).astype(np.uint8)
        encoded[normal_valid_map == 0] = 0
        depth_preview = np.zeros_like(depth_valid_map)
        positive = depth > 0
        if positive.any():
            lo, hi = np.quantile(depth[positive], [.02, .98])
            depth_preview[positive] = np.clip((depth[positive] - lo) / max(hi - lo, 1e-12) * 255, 0, 255).astype(np.uint8)
        for path in paths.values():
            path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(encoded).save(paths["normal_png"])
        np.save(paths["normal_npy"], normal_map)
        Image.fromarray(normal_valid_map).save(paths["normal_valid"])
        np.save(paths["depth_npy"], depth)
        Image.fromarray(depth_preview).save(paths["depth_preview"])
        Image.fromarray(depth_valid_map).save(paths["depth_valid"])
        if "overlay" in paths:
            source = image_dir / camera.name
            image = Image.open(source).convert("RGB")
            if image.size != (camera.width, camera.height):
                raise ValueError(f"Image size {image.size} does not match camera {camera.name}")
            rgb = np.asarray(image)
            overlay = rgb.copy()
            has_normal = normal_valid_map > 0
            overlay[has_normal] = (.55 * rgb[has_normal] + .45 * encoded[has_normal]).astype(np.uint8)
            Image.fromarray(overlay).save(paths["overlay"])
        fx, fy, cx, cy = camera.intrinsics
        if len(depth_selected):
            dx, dy = depth_pixels[:, 0], depth_pixels[:, 1]
            reconstructed_camera = np.column_stack(((dx - cx) * depth_camera_xyz[:, 2] / fx,
                                                     (dy - cy) * depth_camera_xyz[:, 2] / fy,
                                                     depth_camera_xyz[:, 2]))
            reconstructed_world = (reconstructed_camera - camera.translation) @ camera.rotation
            backprojection_error = np.linalg.norm(reconstructed_world - points[depth_selected], axis=1)
            if "backproject_source" in paths:
                sample = np.arange(min(len(depth_selected), 10000))
                for key_name, values in (("backproject_source", points[depth_selected][sample]),
                                         ("backproject_reconstructed", reconstructed_world[sample])):
                    write_ply(paths[key_name], {"x": values[:, 0], "y": values[:, 1], "z": values[:, 2]}, overwrite)
        else:
            backprojection_error = np.empty(0)
        valid_pixels = normal_valid_map > 0
        black_valid = valid_pixels & (encoded.max(axis=2) <= 5)
        pixel_count = camera.width * camera.height
        camera_reports.append({
            "camera": camera.name, "zbuffer_points": zbuffer_count,
            "depth_valid_pixels": int((depth_valid_map > 0).sum()), "normal_valid_pixels": int(valid_pixels.sum()),
            "depth_coverage": float((depth_valid_map > 0).sum() / pixel_count),
            "normal_coverage": float(valid_pixels.sum() / pixel_count),
            "no_lidar_ratio": float(1.0 - zbuffer_count / pixel_count),
            "mask_excluded_points": mask_excluded, "low_confidence_normal_points": low_confidence,
            "near_black_valid_ratio": float(black_valid.sum() / max(valid_pixels.sum(), 1)),
            "depth": _stats(depth[positive]), "backprojection_error": _stats(backprojection_error),
        })
    report = {"version": PREPROCESS_VERSION, "near": near, "far": far, "depth_type": "camera_z",
              "mask_valid_when": mask_valid_when, "camera_count": len(cameras),
              "depth_weighting": {"applied": False, "suggested_distance_bins": [0, 2, 5, 10, far],
                                    "note": "Preprocessing records z-depth; training chooses bin weights."},
              "cameras": camera_reports}
    _json_dump(output_dir / "manifests" / "supervision_report.json", report, overwrite)
    return report


def _records_equal(left: dict[int, tuple[Any, ...]], right: dict[int, tuple[Any, ...]]) -> bool:
    if left.keys() != right.keys():
        return False
    for key in left:
        if len(left[key]) != len(right[key]):
            return False
        for first, second in zip(left[key], right[key]):
            if isinstance(first, np.ndarray):
                if not np.allclose(first, second, rtol=1e-10, atol=1e-12):
                    return False
            elif first != second:
                return False
    return True


def prepare_sfm(source_images: Path, source_masks: Path | None, source_sparse: Path,
                output_dir: Path, val_list: Path | None = None, train_list: Path | None = None,
                test_list: Path | None = None, mask_valid_when: str = "white",
                overwrite: bool = False) -> dict[str, Any]:
    list_sources = {
        "images-val10.txt": val_list or (source_sparse / "images-val10.txt"),
        "images-train.txt": train_list or (source_sparse / "images-train.txt"),
        "images-test.txt": test_list or (source_sparse / "images-test.txt"),
    }
    if not list_sources["images-val10.txt"].exists():
        raise ValueError("A fixed --val-list or source sparse/0/images-val10.txt is required")
    if not list_sources["images-train.txt"].exists():
        raise ValueError("A fixed --train-list or source sparse/0/images-train.txt is required")
    if not list_sources["images-test.txt"].exists():
        raise ValueError("A fixed --test-list or source sparse/0/images-test.txt is required")
    copy_plan: dict[Path, Path] = {}
    for source, destination in ((source_images, output_dir / "images"),
                                (source_sparse, output_dir / "sparse" / "0")):
        for item in source.rglob("*"):
            if item.is_file():
                copy_plan[destination / item.relative_to(source)] = item
    if source_masks is not None:
        for item in source_masks.rglob("*"):
            if item.is_file():
                copy_plan[output_dir / "masks" / item.relative_to(source_masks)] = item
    for name, source in list_sources.items():
        if source.exists():
            copy_plan[output_dir / "sparse" / "0" / name] = source
    report_path = output_dir / "manifests" / "sfm_report.json"
    for target in [*copy_plan, report_path]:
        _guard_output(target, overwrite)
    for target, source in copy_plan.items():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    sparse_out = output_dir / "sparse" / "0"
    cameras = load_cameras(sparse_out)
    image_files = {path.relative_to(output_dir / "images").as_posix(): path
                   for path in (output_dir / "images").rglob("*") if path.is_file()}
    missing = sorted(camera.name for camera in cameras if camera.name not in image_files)
    val_names = read_name_list(sparse_out / "images-val10.txt") or set()
    train_names = read_name_list(sparse_out / "images-train.txt") or set()
    test_names = read_name_list(sparse_out / "images-test.txt") or set()
    invalid_val = sorted(val_names - {camera.name for camera in cameras})
    invalid_train = sorted(train_names - {camera.name for camera in cameras})
    invalid_test = sorted(test_names - {camera.name for camera in cameras})
    overlap = sorted((val_names & train_names) | (val_names & test_names) | (train_names & test_names))
    unassigned = sorted({camera.name for camera in cameras} - val_names - train_names - test_names)
    duplicates = len({camera.name for camera in cameras}) != len(cameras)
    image_size_errors = []
    mask_errors = []
    for camera in cameras:
        image_path = image_files.get(camera.name)
        if image_path is not None and Image.open(image_path).size != (camera.width, camera.height):
            image_size_errors.append(camera.name)
        if source_masks is not None:
            candidates = [output_dir / "masks" / camera.name,
                          output_dir / "masks" / Path(camera.name).with_suffix(".png")]
            mask_path = next((path for path in candidates if path.exists()), None)
            if mask_path is None or Image.open(mask_path).size != (camera.width, camera.height):
                mask_errors.append(camera.name)
    bin_text_consistent = True
    if (sparse_out / "cameras.txt").exists() and (sparse_out / "cameras.bin").exists():
        bin_text_consistent &= _records_equal(read_colmap_cameras(sparse_out, "text"),
                                              read_colmap_cameras(sparse_out, "binary"))
    if (sparse_out / "images.txt").exists() and (sparse_out / "images.bin").exists():
        bin_text_consistent &= _records_equal(read_colmap_images(sparse_out, "text"),
                                              read_colmap_images(sparse_out, "binary"))
    centers = np.asarray([camera.center for camera in cameras])
    report = {
        "version": PREPROCESS_VERSION, "camera_count": len(cameras), "image_count": len(image_files),
        "missing_images": missing, "duplicate_camera_names": duplicates,
        "val_count": len(val_names), "train_count": len(train_names), "test_count": len(test_names),
        "invalid_val": invalid_val, "invalid_train": invalid_train, "invalid_test": invalid_test,
        "split_overlap": overlap, "unassigned_cameras": unassigned,
        "image_size_errors": image_size_errors, "mask_errors": mask_errors,
        "mask_valid_when": mask_valid_when, "bin_text_consistent": bin_text_consistent,
        "camera_center_min": centers.min(axis=0).tolist() if len(centers) else None,
        "camera_center_max": centers.max(axis=0).tolist() if len(centers) else None,
        "inputs": {str(target.relative_to(output_dir)): file_identity(source)
                   for target, source in copy_plan.items()},
    }
    blocking = missing or invalid_val or invalid_train or invalid_test or overlap or unassigned or duplicates \
        or image_size_errors or mask_errors or not bin_text_consistent or len(val_names) != 10
    _json_dump(report_path, report, overwrite)
    if blocking:
        raise ValueError(f"SfM validation failed; inspect {report_path}")
    return report


def validate_processed(output_dir: Path, overwrite: bool = False) -> tuple[dict[str, Any], list[str]]:
    manifest_path = output_dir / "manifests" / "dataset_manifest.json"
    md_path = output_dir / "manifests" / "dataset_validation_report.md"
    _guard_output(manifest_path, overwrite)
    _guard_output(md_path, overwrite)
    errors: list[str] = []
    warnings: list[str] = []
    sparse = output_dir / "sparse" / "0"
    try:
        cameras = load_cameras(sparse)
    except Exception as exc:
        cameras = []
        errors.append(f"COLMAP data cannot be read: {exc}")
    image_dir = output_dir / "images"
    image_names = {path.name for path in image_dir.iterdir()} if image_dir.exists() else set()
    camera_names = {camera.name for camera in cameras}
    missing_images = sorted(camera_names - image_names)
    if missing_images:
        errors.append(f"{len(missing_images)} camera images are missing")
    val_names = read_name_list(sparse / "images-val10.txt") or set()
    if len(val_names) != 10:
        errors.append(f"images-val10.txt must contain exactly 10 unique names, found {len(val_names)}")
    invalid_val = sorted(val_names - camera_names)
    if invalid_val:
        errors.append(f"Validation names absent from COLMAP: {invalid_val[:5]}")
    train_names = read_name_list(sparse / "images-train.txt") or set()
    test_names = read_name_list(sparse / "images-test.txt") or set()
    if not train_names:
        errors.append("images-train.txt is missing or empty")
    if not test_names:
        errors.append("images-test.txt is missing or empty")
    split_overlap = (val_names & train_names) | (val_names & test_names) | (train_names & test_names)
    if split_overlap:
        errors.append(f"Dataset splits overlap: {sorted(split_overlap)[:5]}")
    unassigned = camera_names - val_names - train_names - test_names
    if unassigned:
        errors.append(f"Cameras absent from every split: {sorted(unassigned)[:5]}")
    oriented = output_dir / "geometry" / "lidar_static_rgb_normal_oriented.ply"
    point_count = 0
    if oriented.exists():
        points, _, extras = read_ply(oriented)
        point_count = len(points)
        expected = {"nx", "ny", "nz", "normal_confidence", "orientation_confidence", "curvature", "normal_valid"}
        absent = sorted(expected - extras.keys())
        if absent:
            errors.append(f"Oriented PLY fields missing: {absent}")
    else:
        errors.append(f"Missing oriented LiDAR PLY: {oriented}")
    supervision_report = output_dir / "manifests" / "supervision_report.json"
    if not supervision_report.exists():
        warnings.append("Supervision report is absent")
    report = {
        "version": PREPROCESS_VERSION, "status": "failed" if errors else "passed", "errors": errors,
        "warnings": warnings, "camera_count": len(cameras), "image_count": len(image_names),
        "point_count": point_count, "val_count": len(val_names), "train_count": len(train_names),
        "test_count": len(test_names),
        "artifacts": {str(path.relative_to(output_dir)): file_identity(path)
                      for path in [oriented, sparse / "images-val10.txt", sparse / "images-train.txt",
                                   sparse / "images-test.txt"]
                      if path.exists()},
    }
    configs = {}
    for name in ("sfm_report.json", "supervision_report.json"):
        path = output_dir / "manifests" / name
        if path.exists():
            configs[name] = json.loads(path.read_text(encoding="utf-8"))
    for path in sorted((output_dir / "manifests").glob("preprocess_run_config_*.json")):
        configs[path.name] = json.loads(path.read_text(encoding="utf-8"))
    orientation_path = output_dir / "geometry" / "normal_orientation_summary.json"
    if orientation_path.exists():
        configs[orientation_path.name] = json.loads(orientation_path.read_text(encoding="utf-8"))
    report["preprocess_records"] = configs
    _json_dump(manifest_path, report, overwrite)
    md_path.write_text(
        "# Dataset validation report\n\n"
        f"Status: **{report['status']}**\n\n"
        f"- Cameras: {len(cameras)}\n- Images: {len(image_names)}\n- LiDAR points: {point_count}\n- Val cameras: {len(val_names)}\n\n"
        "## Blocking errors\n\n" + ("\n".join(f"- {item}" for item in errors) or "None") +
        "\n\n## Warnings\n\n" + ("\n".join(f"- {item}" for item in warnings) or "None") + "\n",
        encoding="utf-8")
    return report, errors
