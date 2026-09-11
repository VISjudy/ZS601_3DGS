"""Optional per-reference hard bounds for Gaussian log scales."""
import torch


def _scalar(value, like, dtype=None):
    return torch.as_tensor(value, device=like.device, dtype=dtype or like.dtype)


@torch.no_grad()
def apply_scale_bounds(g, reference, a):
    """Project local XY/Z log scales and clear outward Adam momentum.

    All returned values are tensors on the scale parameter's device. Callers can
    decide when to synchronize them for logging.
    """
    log_scale = g._scaling
    if log_scale.ndim != 2 or log_scale.shape[1] != 3:
        raise ValueError("Expected Gaussian log scales with shape [N, 3]")

    total_points = _scalar(log_scale.shape[0], log_scale, torch.long)
    zero_count = _scalar(0, log_scale, torch.long)
    zero_value = _scalar(0.0, log_scale)
    if not bool(a.scale_bounds):
        return {
            "enabled": _scalar(False, log_scale, torch.bool),
            "total_points": total_points,
            "clipped_points": zero_count,
            "clipped_coordinates": zero_count,
            "clipped_xy_coordinates": zero_count,
            "clipped_z_coordinates": zero_count,
            "max_log_excess_before": zero_value,
            "max_log_excess_after": zero_value,
        }

    spacing = torch.as_tensor(
        reference["spacing"], device=log_scale.device, dtype=log_scale.dtype
    ).reshape(-1)
    if spacing.shape[0] != log_scale.shape[0]:
        raise ValueError("Reference spacing and Gaussian counts differ")

    tiny = torch.finfo(log_scale.dtype).tiny
    negative_infinity = torch.full_like(spacing, -torch.inf)
    # exp(log(linear_limit)) can round slightly above linear_limit in float32,
    # which makes the strict diagnostics report false exceedances. One log-space
    # ULP inward keeps the activated scale at or below the requested bound.
    xy_limit = torch.nextafter(
        (spacing * float(a.size_ratio)).clamp_min(tiny).log(), negative_infinity
    )
    z_limit = torch.nextafter(
        (spacing * float(a.thickness_ratio)).clamp_min(tiny).log(), negative_infinity
    )
    bounds = torch.cat((xy_limit[:, None].expand(-1, 2), z_limit[:, None]), dim=1)
    clipped = log_scale > bounds
    excess_before = torch.relu(log_scale - bounds)

    projected = torch.minimum(log_scale, bounds)
    log_scale.copy_(projected)

    optimizer = getattr(g, "optimizer", None)
    if optimizer is not None:
        state = optimizer.state.get(log_scale, {})
        for name in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
            value = state.get(name)
            if isinstance(value, torch.Tensor) and value.shape == clipped.shape:
                value.masked_fill_(clipped, 0)

    return {
        "enabled": _scalar(True, log_scale, torch.bool),
        "total_points": total_points,
        "clipped_points": clipped.any(dim=1).sum(),
        "clipped_coordinates": clipped.sum(),
        "clipped_xy_coordinates": clipped[:, :2].sum(),
        "clipped_z_coordinates": clipped[:, 2].sum(),
        "max_log_excess_before": excess_before.amax(),
        "max_log_excess_after": torch.relu(log_scale - bounds).amax(),
    }
