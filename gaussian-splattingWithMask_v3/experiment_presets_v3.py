"""Central, serializable feature presets for ZS601 v3 experiments.

The training and preprocessing CLIs use the same auto/on/off semantics. A
feature set is resolved once, printed, and written to run_config.json. No
training module branches directly on the experiment-group letter.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

TRI_STATE_VALUES = ("auto", "on", "off")

TRAINING_FEATURES = (
    "init_normal", "init_flatten", "orient_cameras", "surface_loss",
    "tangent_loss", "normal_loss", "flatten_loss", "size_loss", "pruning",
    "scale_bounds", "surface_densify", "lidar_depth_loss",
    "validation_diagnostics",
)

PREPROCESS_FEATURES = (
    "lidar_outlier_filter", "pca_normal_estimation",
    "camera_normal_orientation", "neighbor_normal_propagation",
    "supervision_generation", "visibility_occlusion_check",
)

FEATURE_NAMES = TRAINING_FEATURES + PREPROCESS_FEATURES


def _features(*enabled: str) -> dict[str, bool]:
    unknown = sorted(set(enabled) - set(FEATURE_NAMES))
    if unknown:
        raise ValueError(f"Unknown preset features: {', '.join(unknown)}")
    selected = set(enabled)
    return {name: name in selected for name in FEATURE_NAMES}


_A = ("init_normal", "init_flatten", "orient_cameras", "pruning",
      "validation_diagnostics")
_GEOMETRY = ("surface_loss", "tangent_loss", "normal_loss", "flatten_loss",
             "size_loss")
_PREPROCESS = ("pca_normal_estimation", "camera_normal_orientation",
               "supervision_generation", "visibility_occlusion_check")

EXPERIMENT_PRESETS = {
    "original": _features("validation_diagnostics"),
    "A": _features(*_A, *_PREPROCESS),
    "B": _features(*_A, *_GEOMETRY, *_PREPROCESS),
    "C": _features(*_A, *_GEOMETRY, "scale_bounds", *_PREPROCESS),
    "D": _features(*_A, *_GEOMETRY, "scale_bounds", "surface_densify", *_PREPROCESS),
    "E": _features(*_A, *_GEOMETRY, "scale_bounds", "lidar_depth_loss", *_PREPROCESS),
    "custom": _features(),
}


def normalize_group(group: str) -> str:
    if not isinstance(group, str):
        raise TypeError("experiment_group must be a string")
    value = group.strip()
    value = value.upper() if len(value) == 1 else value.lower()
    if value not in EXPERIMENT_PRESETS:
        raise ValueError(f"Unknown experiment group {group!r}")
    return value


def normalize_override(value: Any) -> str:
    if isinstance(value, bool):
        return "on" if value else "off"
    if not isinstance(value, str):
        raise TypeError("feature override must be auto/on/off or bool")
    value = value.strip().lower()
    if value not in TRI_STATE_VALUES:
        raise ValueError(f"Invalid feature override {value!r}")
    return value


def resolve_feature_flags(group: str,
                          overrides: Mapping[str, Any] | None = None,
                          feature_names: tuple[str, ...] = FEATURE_NAMES
                          ) -> dict[str, bool]:
    group = normalize_group(group)
    requested = dict(overrides or {})
    unknown = sorted(set(requested) - set(feature_names))
    if unknown:
        raise ValueError(f"Unknown feature overrides: {', '.join(unknown)}")
    result = {name: EXPERIMENT_PRESETS[group][name] for name in feature_names}
    for name, raw in requested.items():
        value = normalize_override(raw)
        if value != "auto":
            result[name] = value == "on"
    return result


def resolve_experiment_config(group: str,
                              overrides: Mapping[str, Any] | None = None,
                              feature_params: Mapping[str, Mapping[str, Any]] | None = None,
                              feature_names: tuple[str, ...] = FEATURE_NAMES
                              ) -> dict[str, Any]:
    raw = dict(overrides or {})
    normalized = {name: normalize_override(value) for name, value in raw.items()}
    flags = resolve_feature_flags(group, normalized, feature_names)
    params = dict(feature_params or {})
    unknown = sorted(set(params) - set(feature_names))
    if unknown:
        raise ValueError(f"Parameters supplied for unknown features: {', '.join(unknown)}")
    enabled_params = {}
    for name, value in params.items():
        if not isinstance(value, Mapping):
            raise TypeError(f"Parameters for {name!r} must be a mapping")
        if flags[name]:
            enabled_params[name] = deepcopy(dict(value))
    return {
        "requested_preset": group,
        "resolved_preset": normalize_group(group),
        "raw_overrides": deepcopy(raw),
        "normalized_overrides": normalized,
        "resolved_feature_flags": flags,
        "enabled_feature_params": enabled_params,
    }


__all__ = ["EXPERIMENT_PRESETS", "FEATURE_NAMES", "TRAINING_FEATURES",
           "PREPROCESS_FEATURES", "TRI_STATE_VALUES", "normalize_group",
           "normalize_override", "resolve_experiment_config",
           "resolve_feature_flags"]
