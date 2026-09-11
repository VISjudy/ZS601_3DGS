"""E-only occlusion-aware LiDAR pseudo-depth and distance-weighted loss."""
from collections import OrderedDict
from pathlib import Path
import csv
import hashlib
import json
import re

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def camera_intrinsics(cam):
    width, height = int(cam.image_width), int(cam.image_height)
    fx = width / (2.0 * np.tan(float(cam.FoVx) / 2.0))
    fy = height / (2.0 * np.tan(float(cam.FoVy) / 2.0))
    return fx, fy, width / 2.0, height / 2.0


def world_to_camera_np(points, rotation, translation):
    """Project world rows with the convention used by this COLMAP loader."""
    return np.asarray(points, dtype=np.float64) @ np.asarray(rotation, dtype=np.float64) + np.asarray(translation, dtype=np.float64)


def camera_to_world_np(points, rotation, translation):
    """Inverse of world_to_camera_np for orthonormal COLMAP rotations."""
    return (np.asarray(points, dtype=np.float64) - np.asarray(translation, dtype=np.float64)) @ np.asarray(rotation, dtype=np.float64).T


def project_camera_np(camera, fx, fy, cx, cy):
    """Map camera points to pixels whose centers are (u+0.5,v+0.5)."""
    camera = np.asarray(camera, dtype=np.float64)
    u = np.floor(fx * camera[:, 0] / camera[:, 2] + cx).astype(np.int64)
    v = np.floor(fy * camera[:, 1] / camera[:, 2] + cy).astype(np.int64)
    return u, v


def pixels_to_camera_np(u, v, z, fx, fy, cx, cy):
    """Backproject integer pixels through their centers."""
    return np.stack((
        (np.asarray(u) + .5 - cx) * z / fx,
        (np.asarray(v) + .5 - cy) * z / fy,
        z,
    ), axis=1)


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_depth_png(path, expected_shape=None):
    with Image.open(path) as image:
        if image.mode not in ('I;16', 'I'):
            raise ValueError(f'Expected 16-bit grayscale depth PNG, got {image.mode}: {path}')
        raw = np.asarray(image)
    if raw.size and (raw.min() < 0 or raw.max() > 65535):
        raise ValueError(f'Depth PNG contains values outside uint16: {path}')
    encoded = raw.astype(np.uint16)
    if expected_shape is not None and encoded.shape != tuple(expected_shape):
        raise ValueError(
            f'Depth PNG shape {encoded.shape} != expected {tuple(expected_shape)}: {path}'
        )
    return encoded


def scatter_zbuffer_min(zbuffer, pixel_index, camera_z):
    """Update a flat z-buffer; multiple points at one pixel keep the nearest z."""
    zbuffer.scatter_reduce_(0, pixel_index, camera_z, reduce='amin', include_self=True)
    return zbuffer


def conservative_fill(depth, hit, radius, min_neighbors, edge_absolute, edge_relative):
    """Fill only neighborhoods supported by coherent foreground LiDAR samples."""
    if radius == 0:
        return torch.where(hit, depth, torch.zeros_like(depth)), hit
    kernel = 2 * radius + 1
    depth4, hit4 = depth[None, None], hit[None, None]
    negative_inf = torch.full_like(depth4, -torch.inf)
    local_min = -F.max_pool2d(torch.where(hit4, -depth4, negative_inf),
                              kernel, stride=1, padding=radius)
    local_max = F.max_pool2d(torch.where(hit4, depth4, negative_inf),
                             kernel, stride=1, padding=radius)
    neighbors = F.avg_pool2d(hit4.float(), kernel, stride=1, padding=radius) * kernel * kernel
    fill_valid = (
        torch.isfinite(local_min) &
        torch.isfinite(local_max) &
        (neighbors >= min_neighbors)
    )
    spread = local_max - local_min
    fill_valid &= spread <= edge_absolute + edge_relative * local_min
    # Exact z-buffer hits retain their own nearest depth, including isolated hits.
    valid = hit4 | ((~hit4) & fill_valid)
    filled = torch.where(hit4, depth4, torch.where(fill_valid, local_min, torch.zeros_like(local_min)))
    return filled[0, 0], valid[0, 0]


def distance_weights(depth, power, minimum, maximum):
    """Inverse-distance weights anchored at the current valid median depth."""
    reference = depth.detach().median()
    weights = (reference / depth.detach().clamp_min(1e-6)).pow(power)
    return weights.clamp(minimum, maximum)


def encode_depth_u16(depth, valid, depth_min, depth_max):
    depth = np.asarray(depth, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    normalized = np.clip((depth - depth_min) / (depth_max - depth_min), 0.0, 1.0)
    encoded = np.zeros(depth.shape, dtype=np.uint16)
    encoded[valid] = np.rint(normalized[valid] * 65534.0 + 1.0).astype(np.uint16)
    return encoded


def decode_depth_u16(encoded, depth_min, depth_max):
    encoded = np.asarray(encoded, dtype=np.uint16)
    valid = encoded > 0
    depth = np.zeros(encoded.shape, dtype=np.float32)
    depth[valid] = depth_min + (encoded[valid].astype(np.float64) - 1.0) / 65534.0 * (depth_max - depth_min)
    return depth, valid


class LidarDepthProvider:
    def __init__(self, points, args, input_identity):
        self.args = args
        self.points_cpu = np.asarray(points, dtype=np.float64)
        if self.points_cpu.ndim != 2 or self.points_cpu.shape[1] != 3 or not np.isfinite(self.points_cpu).all():
            raise ValueError('LiDAR points must be a finite Nx3 array')
        self.points = torch.as_tensor(self.points_cpu, dtype=torch.float32, device='cuda')
        self.memory = OrderedDict()
        self.persistent_index = {}
        self.generated = 0
        self.identity = {
            'version': 'lidar_camera_z_v3',
            'inputs': dict(sorted(input_identity.items())),
            'depth_min': args.lidar_depth_min,
            'depth_max': args.lidar_depth_max,
            'projection': 'camera = world @ R + T; centered FoV intrinsics; rounded pixel',
            'zbuffer': 'nearest positive camera-z per pixel',
            'splat_radius': args.lidar_depth_splat_radius,
            'min_neighbors': args.lidar_depth_min_neighbors,
            'edge_relative': args.lidar_depth_edge_relative,
            'edge_absolute': args.lidar_depth_edge_absolute,
        }
        self.root = Path(args.lidar_depth_cache).expanduser().resolve()
        if '/drive/' in self.root.as_posix().lower():
            raise ValueError('LiDAR cache must stay on Colab local disk, not Google Drive')
        manifest = self.root / 'cache_manifest.json'
        if self.root.exists():
            if manifest.is_file():
                existing = json.loads(manifest.read_text(encoding='utf-8'))
                if existing.get('identity') != self.identity:
                    raise ValueError('Existing LiDAR depth cache belongs to different inputs or parameters')
            elif any(self.root.iterdir()):
                raise FileExistsError(f'Nonempty unverified LiDAR depth cache: {self.root}')
        else:
            self.root.mkdir(parents=True)
        if not manifest.exists():
            manifest.write_text(json.dumps({
                'identity': self.identity,
                'storage': 'temporary local uint16 PNG using the same encoding as the Drive dataset',
                'persistent_to_drive': False,
                'occlusion': 'nearest-depth z-buffer before conservative hole filling',
                'invalid': 'nonfinite, behind, too near, too far, outside image, or a depth discontinuity',
            }, indent=2), encoding='utf-8')
        print('[LIDAR DEPTH CACHE]', json.dumps({
            'path': str(self.root), 'points': len(self.points), 'persistent_to_drive': False,
            **self.identity,
        }, indent=2), flush=True)

    def _key(self, cam):
        camera = {
            'name': cam.image_name,
            'width': int(cam.image_width),
            'height': int(cam.image_height),
            'fovx': float(cam.FoVx),
            'fovy': float(cam.FoVy),
            'R': np.asarray(cam.R).round(12).tolist(),
            'T': np.asarray(cam.T).round(12).tolist(),
        }
        payload = json.dumps({'input': self.identity, 'camera': camera},
                             sort_keys=True, separators=(',', ':'))
        return hashlib.sha256(payload.encode()).hexdigest()

    @torch.no_grad()
    def _generate(self, cam):
        height, width = int(cam.image_height), int(cam.image_width)
        fx, fy, cx, cy = camera_intrinsics(cam)
        rotation = torch.as_tensor(np.asarray(cam.R), device='cuda', dtype=torch.float32)
        translation = torch.as_tensor(np.asarray(cam.T), device='cuda', dtype=torch.float32)
        zbuffer = torch.full((height * width,), float('inf'), device='cuda')
        for start in range(0, len(self.points), self.args.lidar_depth_chunk):
            point = self.points[start:start + self.args.lidar_depth_chunk]
            camera = point @ rotation + translation
            z = camera[:, 2]
            valid = (torch.isfinite(camera).all(1) &
                     (z >= self.args.lidar_depth_min) &
                     (z <= self.args.lidar_depth_max))
            if not valid.any():
                continue
            camera, z = camera[valid], z[valid]
            u = torch.floor(fx * camera[:, 0] / z + cx).to(torch.long)
            v = torch.floor(fy * camera[:, 1] / z + cy).to(torch.long)
            inside = (u >= 0) & (u < width) & (v >= 0) & (v < height)
            if inside.any():
                scatter_zbuffer_min(zbuffer, v[inside] * width + u[inside], z[inside])
        depth = zbuffer.view(height, width)
        hit = torch.isfinite(depth)
        depth, valid = conservative_fill(
            depth, hit,
            self.args.lidar_depth_splat_radius,
            self.args.lidar_depth_min_neighbors,
            self.args.lidar_depth_edge_absolute,
            self.args.lidar_depth_edge_relative,
        )
        valid &= ((depth >= self.args.lidar_depth_min) &
                  (depth <= self.args.lidar_depth_max) &
                  torch.isfinite(depth))
        depth = torch.where(valid, depth, torch.zeros_like(depth))
        return depth.cpu(), valid.cpu()

    def _get_cpu(self, cam):
        key = self._key(cam)
        if key in self.memory:
            depth, valid = self.memory.pop(key)
            self.memory[key] = (depth, valid)
            return depth, valid
        path = self.root / (key + '_depth_u16.png')
        generated_now = False
        if path.is_file():
            encoded = read_depth_png(
                path, (int(cam.image_height), int(cam.image_width))
            )
        elif cam.image_name in self.persistent_index:
            encoded = read_depth_png(
                self.persistent_index[cam.image_name],
                (int(cam.image_height), int(cam.image_width)),
            )
        else:
            raw_depth, raw_valid = self._generate(cam)
            encoded = encode_depth_u16(
                raw_depth.numpy(), raw_valid.numpy(),
                self.args.lidar_depth_min, self.args.lidar_depth_max,
            )
            self.generated += 1
            generated_now = True
        depth_np, valid_np = decode_depth_u16(
            encoded, self.args.lidar_depth_min, self.args.lidar_depth_max
        )
        depth = torch.from_numpy(depth_np)
        valid = torch.from_numpy(valid_np)
        if not path.is_file():
            temporary = self.root / (key + '.tmp.png')
            Image.fromarray(encoded).save(temporary, format='PNG')
            temporary.replace(path)
        if generated_now and (self.generated <= 5 or self.generated % 100 == 0):
            values = depth[valid]
            print('[LIDAR DEPTH]', json.dumps({
                'generated': self.generated,
                'camera': cam.image_name,
                'valid_pixels': int(valid.sum()),
                'coverage': float(valid.float().mean()),
                'min': float(values.min()) if values.numel() else None,
                'median': float(values.median()) if values.numel() else None,
                'max': float(values.max()) if values.numel() else None,
            }), flush=True)
        self.memory[key] = (depth, valid)
        while len(self.memory) > self.args.lidar_depth_cache_memory:
            self.memory.popitem(last=False)
        return depth, valid

    def get(self, cam):
        depth, valid = self._get_cpu(cam)
        return depth.cuda(non_blocking=True), valid.cuda(non_blocking=True)

    def _backproject_validation(self, cam, depth, valid, tree):
        pixel = torch.nonzero(valid, as_tuple=False)
        if not len(pixel):
            raise ValueError(f'No valid LiDAR depth for {cam.image_name}')
        if len(pixel) > self.args.lidar_depth_backproject_samples:
            choose = torch.linspace(
                0, len(pixel) - 1, self.args.lidar_depth_backproject_samples
            ).round().long()
            pixel = pixel[choose]
        v = pixel[:, 0].numpy().astype(np.float64)
        u = pixel[:, 1].numpy().astype(np.float64)
        z = depth[pixel[:, 0], pixel[:, 1]].numpy().astype(np.float64)
        fx, fy, cx, cy = camera_intrinsics(cam)
        camera = pixels_to_camera_np(u, v, z, fx, fy, cx, cy)
        world = camera_to_world_np(camera, cam.R, cam.T)
        distance, source_index = tree.query(world, k=1, workers=-1)
        source_camera = world_to_camera_np(
            self.points_cpu[source_index], cam.R, cam.T
        )
        source_u, source_v = project_camera_np(source_camera, fx, fy, cx, cy)
        reprojection = np.maximum(np.abs(source_u - u), np.abs(source_v - v))
        source_depth_error = np.abs(source_camera[:, 2] - z)
        tolerance = self.args.lidar_depth_backproject_tolerance
        quantile = self.args.lidar_depth_backproject_quantile
        pixel_tolerance = self.args.lidar_depth_reprojection_tolerance_px
        return {
            'samples': int(len(distance)),
            'mean': float(np.mean(distance)),
            'median': float(np.median(distance)),
            'p95': float(np.quantile(distance, .95)),
            'quantile': quantile,
            'quantile_distance': float(np.quantile(distance, quantile)),
            'max': float(np.max(distance)),
            'pass_fraction': float(np.mean(distance <= tolerance)),
            'tolerance': tolerance,
            'reprojection_quantile_px': float(np.quantile(reprojection, quantile)),
            'reprojection_pass_fraction': float(np.mean(reprojection <= pixel_tolerance)),
            'reprojection_tolerance_px': pixel_tolerance,
            'source_depth_error_quantile': float(np.quantile(source_depth_error, quantile)),
        }

    def _save_and_reload_depth(self, path, depth, valid):
        encoded = encode_depth_u16(
            depth.numpy(), valid.numpy(),
            self.args.lidar_depth_min, self.args.lidar_depth_max,
        )
        Image.fromarray(encoded).save(path)
        stored = read_depth_png(path, depth.shape)
        decoded, decoded_valid = decode_depth_u16(
            stored, self.args.lidar_depth_min, self.args.lidar_depth_max,
        )
        if not np.array_equal(decoded_valid, valid.numpy()):
            raise ValueError(f'Depth PNG valid-mask round trip failed: {path}')
        quantization = float(np.max(np.abs(decoded[decoded_valid] - depth.numpy()[decoded_valid])))
        allowed = (self.args.lidar_depth_max - self.args.lidar_depth_min) / 65534.0 + 1e-6
        if quantization > allowed:
            raise ValueError(f'Depth PNG quantization exceeded bound: {quantization} > {allowed}')
        return decoded, decoded_valid, quantization

    @staticmethod
    def _save_color(path, depth, valid, depth_min, depth_max):
        normalized = np.clip((depth - depth_min) / (depth_max - depth_min), 0.0, 1.0)
        rgb = np.stack((
            np.clip(1.5 - np.abs(4 * normalized - 3), 0, 1),
            np.clip(1.5 - np.abs(4 * normalized - 2), 0, 1),
            np.clip(1.5 - np.abs(4 * normalized - 1), 0, 1),
        ), axis=-1)
        rgb[~valid] = 0
        Image.fromarray(np.rint(rgb * 255).astype(np.uint8)).save(path)

    def _reuse_verified_export(self, export):
        summary_path = export / 'dataset_summary.json'
        verification_path = export / 'verification.json'
        manifest_path = export / 'depth_manifest.csv'
        if not (summary_path.is_file() and verification_path.is_file() and manifest_path.is_file()):
            raise FileExistsError(
                f'Existing depth export is incomplete and will not be overwritten: {export}'
            )
        summary = json.loads(summary_path.read_text(encoding='utf-8'))
        verification = json.loads(verification_path.read_text(encoding='utf-8'))
        if summary.get('identity') != self.identity or verification.get('passed') is not True:
            raise ValueError('Existing depth export identity or verification does not match this run')
        index = {}
        with manifest_path.open(newline='', encoding='utf-8') as handle:
            for row in csv.DictReader(handle):
                name = row['image_name']
                candidate = (export / row['depth_u16']).resolve()
                if not candidate.is_relative_to(export) or not candidate.is_file():
                    raise FileNotFoundError(f'Invalid persisted depth path for {name}: {candidate}')
                if name in index:
                    raise ValueError(f'Duplicate image name in persisted depth manifest: {name}')
                index[name] = candidate
        if len(index) != summary.get('camera_count'):
            raise ValueError('Persisted depth manifest count differs from dataset summary')
        self.persistent_index = index
        print('[LIDAR DEPTH DATASET REUSED]', json.dumps({
            'path': str(export), 'camera_count': len(index),
            'backprojection': summary.get('backprojection'),
        }, indent=2), flush=True)
        return summary

    def prepare_and_export(self, camera_groups, export_path):
        """Create or read a verified persistent pseudo-GT dataset before training."""
        from scipy.spatial import cKDTree

        export = Path(export_path).expanduser().resolve()
        if export.exists():
            return self._reuse_verified_export(export)
        export.mkdir(parents=True)
        tree = cKDTree(self.points_cpu)
        fields = [
            'role', 'camera_index', 'image_name', 'depth_u16', 'color_preview',
            'valid_pixels', 'coverage', 'depth_min', 'depth_median', 'depth_max',
            'depth_quantization_max', 'backproject_samples', 'backproject_mean',
            'backproject_median', 'backproject_p95', 'backproject_max',
            'backproject_pass_fraction',
        ]
        manifest = export / 'depth_manifest.csv'
        records = []
        total = sum(len(cameras) for cameras in camera_groups.values())
        done = 0
        with manifest.open('x', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for role, cameras in camera_groups.items():
                raw_dir = export / role
                raw_dir.mkdir()
                preview_dir = export / (role + '_preview') if role in ('val', 'test') else None
                if preview_dir:
                    preview_dir.mkdir()
                for index, cam in enumerate(cameras):
                    depth, valid = self._get_cpu(cam)
                    safe = re.sub(r'[^A-Za-z0-9_.-]+', '_', Path(cam.image_name).stem)
                    stem = f'{index:05d}_{safe}_{self._key(cam)[:10]}'
                    depth_rel = Path(role) / (stem + '_depth_u16.png')
                    decoded, decoded_valid, quantization = self._save_and_reload_depth(
                        export / depth_rel, depth, valid
                    )
                    decoded_t = torch.from_numpy(decoded)
                    decoded_valid_t = torch.from_numpy(decoded_valid)
                    check = self._backproject_validation(
                        cam, decoded_t, decoded_valid_t, tree
                    )
                    if check['p95'] > self.args.lidar_depth_backproject_tolerance:
                        raise ValueError(
                            f'Backprojection validation failed for {cam.image_name}: {check}'
                        )
                    preview_rel = ''
                    if preview_dir:
                        preview_rel = Path(role + '_preview') / (stem + '_depth_color.png')
                        self._save_color(
                            export / preview_rel, decoded, decoded_valid,
                            self.args.lidar_depth_min, self.args.lidar_depth_max,
                        )
                    values = decoded[decoded_valid]
                    row = {
                        'role': role,
                        'camera_index': index,
                        'image_name': cam.image_name,
                        'depth_u16': str(depth_rel),
                        'color_preview': str(preview_rel),
                        'valid_pixels': int(decoded_valid.sum()),
                        'coverage': float(decoded_valid.mean()),
                        'depth_min': float(values.min()),
                        'depth_median': float(np.median(values)),
                        'depth_max': float(values.max()),
                        'depth_quantization_max': quantization,
                        **{
                            'backproject_' + key: value
                            for key, value in check.items()
                            if key != 'tolerance'
                        },
                    }
                    writer.writerow(row)
                    handle.flush()
                    records.append(row)
                    done += 1
                    if done <= 5 or done % 100 == 0 or done == total:
                        print(
                            f'[LIDAR DEPTH EXPORT] {done}/{total} {role} {cam.image_name} '
                            f'coverage={row["coverage"]:.4f} '
                            f'backproject_p95={row["backproject_p95"]:.5f}',
                            flush=True,
                        )
        by_role = {}
        for role in camera_groups:
            selected = [record for record in records if record['role'] == role]
            by_role[role] = {
                'camera_count': len(selected),
                'coverage_mean': float(np.mean([r['coverage'] for r in selected])) if selected else None,
                'backproject_p95_max': max((r['backproject_p95'] for r in selected), default=None),
            }
        summary = {
            'identity': self.identity,
            'camera_count': len(records),
            'roles': by_role,
            'depth_encoding': {
                'file': '16-bit PNG',
                'invalid_code': 0,
                'valid_code_range': [1, 65535],
                'decode': (
                    f'depth={self.args.lidar_depth_min}+(code-1)/65534*'
                    f'{self.args.lidar_depth_max-self.args.lidar_depth_min}'
                ),
                'maximum_quantization_error': (
                    self.args.lidar_depth_max - self.args.lidar_depth_min
                ) / 65534.0,
            },
            'occlusion': (
                'nearest camera-z z-buffer; hole filling requires coherent nearby '
                'LiDAR hits and rejects depth discontinuities'
            ),
            'filters': {
                'depth_range': [self.args.lidar_depth_min, self.args.lidar_depth_max],
                'nonfinite_and_outside_image': 'invalid',
                'splat_radius': self.args.lidar_depth_splat_radius,
                'minimum_neighbors': self.args.lidar_depth_min_neighbors,
                'edge_threshold': (
                    'spread <= edge_absolute + edge_relative * nearest_depth'
                ),
            },
            'backprojection': {
                'world_formula': '(camera_xyz - T) @ R.T',
                'source': 'reloaded persisted 16-bit PNG',
                'nearest_lidar_p95_max': max(r['backproject_p95'] for r in records),
                'required_p95_max': self.args.lidar_depth_backproject_tolerance,
                'all_cameras_passed': True,
            },
            'coverage_mean': float(np.mean([r['coverage'] for r in records])),
            'manifest': 'depth_manifest.csv',
            'previews': 'val_preview and test_preview; raw train/val/test PNGs are all retained',
        }
        (export / 'dataset_summary.json').write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8'
        )
        verification = {
            'passed': True,
            'camera_count': len(records),
            'checked_saved_png_round_trip': True,
            'checked_camera_to_world_backprojection': True,
            'backproject_p95_max': summary['backprojection']['nearest_lidar_p95_max'],
            'tolerance': self.args.lidar_depth_backproject_tolerance,
            'dataset_summary': 'dataset_summary.json',
            'manifest': 'depth_manifest.csv',
        }
        (export / 'verification.json').write_text(
            json.dumps(verification, indent=2), encoding='utf-8'
        )
        print('[LIDAR DEPTH DATASET READY]', json.dumps(summary, indent=2), flush=True)
        return summary


def render_expected_center_depth(cam, g, pipe):
    """Differentiable opacity-normalized expected camera-z of Gaussian centers."""
    from render_v3 import attributes

    transform = cam.world_view_transform.cuda()
    z = (g.get_xyz @ transform[:3, :3] + transform[3, :3])[:, 2]
    accum = attributes(
        cam, g, pipe,
        torch.stack((z, torch.ones_like(z), torch.zeros_like(z)), dim=-1),
    )
    alpha = accum[1]
    return accum[0] / alpha.clamp_min(1e-8), alpha


def lidar_depth_term(cam, g, pipe, provider, args, iteration):
    configured = args.lambda_lidar_depth
    empty = {
        'raw': 0.0,
        'weight': 0.0,
        'weighted': 0.0,
        'valid_pixels': 0,
        'lidar_pixels': 0,
        'rendered_fraction': 0.0,
        'mean_target_depth': 0.0,
        'distance_weight_mean': 0.0,
    }
    if not args.lidar_depth_loss or iteration < args.lidar_depth_start or configured == 0:
        state = 'OFF' if not args.lidar_depth_loss else 'WAITING_OR_ZERO_WEIGHT'
        return g.get_xyz.new_zeros(()), {**empty, 'state': state}
    ramp = min(
        1.0,
        max(0.0, (iteration - args.lidar_depth_start) / max(1, args.lidar_depth_warmup)),
    )
    weight = configured * ramp
    if weight == 0:
        return g.get_xyz.new_zeros(()), {**empty, 'state': 'WARMUP'}

    target, target_valid = provider.get(cam)
    predicted, alpha = render_expected_center_depth(cam, g, pipe)
    valid = (
        target_valid &
        torch.isfinite(predicted) &
        (predicted >= args.lidar_depth_min) &
        (predicted <= args.lidar_depth_max) &
        (alpha >= args.lidar_depth_alpha_min)
    )
    if cam.alpha_mask is not None:
        valid &= cam.alpha_mask.cuda()[0] > .5
    count = int(valid.sum())
    lidar_count = int(target_valid.sum())
    fraction = count / max(lidar_count, 1)
    if count < args.lidar_depth_min_pixels:
        return g.get_xyz.new_zeros(()), {
            **empty,
            'valid_pixels': count,
            'lidar_pixels': lidar_count,
            'rendered_fraction': fraction,
            'state': 'INSUFFICIENT_VALID_PIXELS',
        }

    target_values = target[valid]
    predicted_values = predicted[valid]
    relative = (
        predicted_values - target_values
    ) / target_values.clamp_min(1e-6)
    per_pixel = F.smooth_l1_loss(
        relative,
        torch.zeros_like(relative),
        beta=args.lidar_depth_huber_beta,
        reduction='none',
    )
    per_distance = distance_weights(
        target_values,
        args.lidar_depth_distance_power,
        args.lidar_depth_weight_min,
        args.lidar_depth_weight_max,
    )
    raw = (per_pixel * per_distance).sum() / per_distance.sum().clamp_min(1e-8)
    return weight * raw, {
        'raw': float(raw.detach()),
        'weight': weight,
        'weighted': float(raw.detach()) * weight,
        'valid_pixels': count,
        'lidar_pixels': lidar_count,
        'rendered_fraction': fraction,
        'mean_target_depth': float(target_values.mean()),
        'distance_weight_mean': float(per_distance.mean()),
        'state': 'ACTIVE',
    }
