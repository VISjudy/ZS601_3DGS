"""Sparse LiDAR supervision; no mesh targets, filling, or Gaussian-axis loss.

Integration: construct PrecomputedSupervision(root, {image_name: six_digit_id},
skip_images=explicit_virtual_names). Call supervision_terms after RGB rendering
and add its returned scalar to RGB loss. All weights are already applied.
The map must contain every supervised train image, with exact image names.
Targets: arrays/000001.npz contains metric camera-Z depth, source point_ids,
camera-coordinate PCA normal (toward camera), and independent normal_valid.
Geometry gradients flow through the renderer and depth-derived normal only.
"""
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


class PrecomputedSupervision:
    def __init__(self, root, image_to_id, cache_size=8, skip_images=()):
        self.root = Path(root)
        if not (self.root / 'arrays').is_dir():
            raise FileNotFoundError(self.root / 'arrays')
        self.image_to_id = dict(image_to_id)
        self.skip_images = frozenset(skip_images)
        if self.skip_images & self.image_to_id.keys():
            raise ValueError('An image cannot be both supervised and explicitly skipped')
        if int(cache_size) != cache_size or cache_size < 0:
            raise ValueError('cache_size must be a nonnegative integer')
        self.cache_size = int(cache_size)
        self.cache = OrderedDict()

    def get(self, image_name, device, dtype=torch.float32, expected_shape=None):
        if image_name in self.skip_images:
            return None
        if image_name not in self.image_to_id:
            raise KeyError(f'Missing supervision mapping for real training image: {image_name}')
        view_id = int(self.image_to_id[image_name])
        if not 0 <= view_id <= 999999:
            raise ValueError('Supervision ID must fit six digits')
        key = (view_id, str(torch.device(device)), dtype)
        if key in self.cache:
            result = self.cache.pop(key)
            self.cache[key] = result
        else:
            path = self.root / 'arrays' / f'{view_id:06d}.npz'
            with np.load(path, allow_pickle=False) as archive:
                depth = np.array(archive['depth'], copy=True)
                ids = np.array(archive['point_ids'], copy=True)
                normal = np.array(archive['normal'], copy=True)
                normal_valid = np.array(archive['normal_valid'], copy=True)
            if depth.ndim != 2 or depth.dtype != np.float64:
                raise ValueError(f'{path}: depth must be float64 HxW metric camera-Z')
            if ids.shape != depth.shape or not np.issubdtype(ids.dtype, np.integer):
                raise ValueError(f'{path}: invalid source point_ids')
            if normal.shape != (*depth.shape, 3) or normal_valid.shape != depth.shape:
                raise ValueError(f'{path}: inconsistent normal shapes')
            if normal_valid.dtype != np.bool_:
                raise ValueError(f'{path}: normal_valid must be boolean')
            if not np.isfinite(depth).all() or (depth < 0).any():
                raise ValueError(f'{path}: depth must be finite and nonnegative')
            depth_valid = depth > 0
            if (ids[depth_valid] < 0).any():
                raise ValueError(f'{path}: valid depth lacks source point ID')
            if (normal_valid & ~depth_valid).any():
                raise ValueError(f'{path}: normal target without valid source depth')
            selected = normal[normal_valid]
            if not np.isfinite(selected).all() or (np.linalg.norm(selected, axis=-1) < 1e-8).any():
                raise ValueError(f'{path}: invalid valid-area PCA normals')
            normal[~normal_valid] = 0
            result = {
                'depth': torch.as_tensor(depth, device=device, dtype=dtype),
                'depth_valid': torch.as_tensor(depth_valid, device=device),
                'normal': F.normalize(torch.as_tensor(normal, device=device, dtype=dtype), dim=-1),
                'normal_valid': torch.as_tensor(normal_valid, device=device),
                'point_ids': torch.as_tensor(ids.astype(np.int64), device=device),
            }
            if self.cache_size:
                self.cache[key] = result
                while len(self.cache) > self.cache_size:
                    self.cache.popitem(last=False)
        if expected_shape is not None and tuple(result['depth'].shape) != tuple(expected_shape):
            raise ValueError(f'{image_name}: supervision resolution mismatch; no implicit resizing')
        return result


def depth_to_camera_normals(depth, alpha, intrinsics, alpha_min=.05, pixel_mask=None):
    """Central differences over a valid five-pixel cross; outer border invalid.

    Pixel rays use (u+.5,v+.5). Flip cross products toward the camera origin.
    Only the predicted dense depth needs a neighborhood; sparse GT is untouched.
    Returns HxWx3 unit normals and HxW bool validity.
    """
    if depth.ndim != 2 or alpha.shape != depth.shape:
        raise ValueError('depth and alpha must be matching HxW tensors')
    fx, fy, cx, cy = intrinsics
    if fx <= 0 or fy <= 0 or not 0 <= alpha_min <= 1:
        raise ValueError('invalid intrinsics or alpha threshold')
    h, w = depth.shape
    if h < 3 or w < 3:
        raise ValueError('normal reconstruction needs at least 3x3 pixels')
    valid = torch.isfinite(depth) & (depth > 0) & torch.isfinite(alpha) & (alpha >= alpha_min)
    if pixel_mask is not None:
        if pixel_mask.shape != depth.shape:
            raise ValueError('pixel mask resolution mismatch')
        valid = valid & pixel_mask.bool()
    safe_depth = torch.where(valid, depth, torch.zeros_like(depth))
    v, u = torch.meshgrid(torch.arange(h, device=depth.device, dtype=depth.dtype),
                          torch.arange(w, device=depth.device, dtype=depth.dtype), indexing='ij')
    ray = torch.stack(((u+.5-cx)/fx, (v+.5-cy)/fy, torch.ones_like(u)), -1)
    points = ray * safe_depth[..., None]
    dx = points[1:-1, 2:] - points[1:-1, :-2]
    dy = points[2:, 1:-1] - points[:-2, 1:-1]
    cross = torch.linalg.cross(dx, dy, dim=-1)
    length = torch.linalg.vector_norm(cross, dim=-1)
    inside = (valid[1:-1, 1:-1] & valid[1:-1, :-2] & valid[1:-1, 2:]
              & valid[:-2, 1:-1] & valid[2:, 1:-1]
              & torch.isfinite(cross).all(-1) & (length > 1e-10))
    normal = F.normalize(cross, dim=-1, eps=1e-10)
    # The camera lies at the origin in camera coordinates.
    sign = torch.where((normal * points[1:-1, 1:-1]).sum(-1) > 0, -1., 1.)
    normal = torch.where(inside[..., None], normal * sign[..., None], torch.zeros_like(normal))
    return F.pad(normal.permute(2, 0, 1), (1, 1, 1, 1)).permute(1, 2, 0), F.pad(inside, (1, 1, 1, 1))


def losses_from_maps(predicted, alpha, target, intrinsics, *, depth_weight=0.,
                     normal_weight=0., beta=.02, alpha_min=.05, min_pixels=1,
                     pixel_mask=None):
    """Return weighted total and flat scalar log, using absolute metric Huber.

    smooth_l1_loss is PyTorch's Huber/beta form: quadratic e^2/(2 beta)
    near zero, absolute residual minus beta/2 outside; beta is in meters.
    """
    if depth_weight < 0 or normal_weight < 0 or beta <= 0 or min_pixels < 1:
        raise ValueError('invalid loss weights, beta or min_pixels')
    if predicted.ndim != 2 or alpha.shape != predicted.shape:
        raise ValueError('prediction and alpha must be matching HxW tensors')
    if pixel_mask is not None and pixel_mask.shape != predicted.shape:
        raise ValueError('pixel mask resolution mismatch')
    if not 0 <= alpha_min <= 1:
        raise ValueError('invalid alpha threshold')
    zero = torch.where(torch.isfinite(predicted), predicted, 0.).sum() * 0.
    log = {}
    if target is None:
        for name in ('depth', 'normal'):
            log.update({f'{name}_raw': 0., f'{name}_weight': 0., f'{name}_weighted': 0.,
                        f'{name}_valid_pixels': 0, f'{name}_state': 'EXPLICIT_SKIP'})
        return zero, log
    if target['depth'].shape != predicted.shape:
        raise ValueError('target/prediction resolution mismatch')
    valid = torch.isfinite(predicted) & (predicted > 0) & torch.isfinite(alpha) & (alpha >= alpha_min)
    if pixel_mask is not None:
        valid = valid & pixel_mask.bool()
    total = zero
    for name, weight in (('depth', depth_weight), ('normal', normal_weight)):
        raw, count, state = zero, 0, 'OFF'
        if weight > 0:
            if name == 'depth':
                mask = valid & target['depth_valid']
                count = int(mask.sum().item())
                if count >= min_pixels:
                    raw = F.smooth_l1_loss(predicted[mask], target['depth'][mask], beta=beta)
            else:
                normals, normal_valid = depth_to_camera_normals(predicted, alpha, intrinsics, alpha_min, pixel_mask)
                mask = normal_valid & target['normal_valid']
                count = int(mask.sum().item())
                if count >= min_pixels:
                    cosine = (normals[mask] * target['normal'][mask]).sum(-1).clamp(-1, 1)
                    raw = (1. - cosine).mean()
            state = 'ACTIVE' if count >= min_pixels else 'INSUFFICIENT_VALID_PIXELS'
            total = total + weight * raw
        log.update({f'{name}_raw': float(raw.detach()), f'{name}_weight': weight,
                    f'{name}_weighted': float(raw.detach()) * weight,
                    f'{name}_valid_pixels': count, f'{name}_state': state})
    return total, log


def supervision_terms(cam, g, pipe, provider, *, depth_weight=0., normal_weight=0.,
                      beta=.02, alpha_min=.05, min_pixels=1):
    """Trainer entry point; no renderer call for an explicitly skipped image."""
    from lidar_depth_v3 import camera_intrinsics, render_expected_center_depth
    if depth_weight == 0 and normal_weight == 0:
        zero = g.get_xyz.sum() * 0.
        log = {}
        for name in ('depth', 'normal'):
            log.update({f'{name}_raw': 0., f'{name}_weight': 0., f'{name}_weighted': 0.,
                        f'{name}_valid_pixels': 0, f'{name}_state': 'OFF'})
        return zero, log
    target = provider.get(cam.image_name, g.get_xyz.device, g.get_xyz.dtype,
                          (cam.image_height, cam.image_width))
    if target is None:
        zero_map = g.get_xyz.sum().reshape(1, 1) * 0.
        return losses_from_maps(zero_map, zero_map, None, (1., 1., 0., 0.))
    predicted, alpha = render_expected_center_depth(cam, g, pipe)
    pixel_mask = None if cam.alpha_mask is None else cam.alpha_mask.to(predicted.device)[0] > .5
    return losses_from_maps(predicted, alpha, target, camera_intrinsics(cam),
                           depth_weight=depth_weight, normal_weight=normal_weight,
                           beta=beta, alpha_min=alpha_min, min_pixels=min_pixels, pixel_mask=pixel_mask)
