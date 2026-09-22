# Five-method CPU depth evaluation

`evaluate_five_methods.py` evaluates the same ten fixed v007 cameras for A/B and
the three downloaded v008 methods. It does not invoke CUDA, Colab, Blender, Git,
or the previous exact-winner-ray probe. All previous inputs are read-only.

The depth used for metrics and PLY export is the saved single-channel uint16 PNG,
decoded as camera-Z millimetres. COLMAP uses `pc = world @ R.T + t`, with inverse
`world = (pc-t) @ R`; pixel centres are `(x+0.5, y+0.5)`.

## Commands

Use the host Python supplied for the experiment:

```powershell
& 'C:/Users/Lenovo/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe' 'D:/codex/blenderProject/experiments/2026-09-22/first-hit-depth-v008/analysis/evaluate_five_methods.py' --preflight
& 'C:/Users/Lenovo/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe' 'D:/codex/blenderProject/experiments/2026-09-22/first-hit-depth-v008/analysis/evaluate_five_methods.py'
```

The output is `3dgsResult/zs601-firsthit-depth-v008/evaluation/`. The evaluator
refuses to replace an existing evaluation directory. Figures can be regenerated
without repeating metrics by running `plot_errors.py` with a fresh `--figures`
directory under the evaluation directory.

`python-deps/` contains the local Matplotlib environment, including Matplotlib
3.11.2. It is execution support, not experimental data. Shared NumPy/SciPy and
PLY dependencies remain unchanged. `matplotlib-config/` is an experiment-local
font/config cache. No network service is used during evaluation.

## Inputs and first-contributor semantics

- A/B come from `zs601-depth-methods-v007`.
- C: `method_c_first_center_s05`; D: `method_d_first_peak_s05`;
  E: `method_e_first_peak_s01` under `zs601-firsthit-depth-v008`.
- Each new method supplies `depth`, `depth_mask`, `depth_mask_alpha50`, `images`,
  `depth_float`, and `aux/<six-digit-view>.npz`.
- `first_id` is a zero-based signed int32 source index, with `-1` invalid.
  Auxiliary `center_z`, `peak_z`, `entry_z`, and `exit_z` are camera-Z metres.
  `qmin` and `first_alpha` are dimensionless.
- Primary validity is first ID present and selected raw depth finite and
  positive. Alpha50 never filters the primary point clouds.
- `depth_mask_alpha50` is primary validity AND accumulated float alpha >= 0.5.
  It is independent of the single first-contributor `first_alpha` diagnostic.
- `qmin > 9` has no finite 3-sigma ellipsoid intersection; `entry_z` and `exit_z`
  are zero sentinels. These pixels are retained by the renderer's actual
  first-contributor ray-peak logic and separately counted, without an added gate.

## Outputs

`geometry_metrics.json` contains `protocol`, `aggregate` keyed by the five stable
method directory names, and 50 `per_view` rows. `per_view.csv` and `aggregate.csv`
provide flattened core metrics. `verification.json` describes the scope of
format, coordinate, mask, provenance, PLY, and input-preservation checks.

Error objects include count, mean absolute / median absolute / P95 absolute /
RMS / maximum error in millimetres. Signed mesh-Z objects also report bias and
both behind/in-front percentages beyond 30 mm. The following domains are distinct:

- `point_to_input`: all primary valid points, exact one-sided NN to the original
  7,004,696-point cloud, using float32 exported point coordinates.
- `own_valid_depth`: a method's primary mask intersected with reference validity.
- `common_all5_depth`: all five primary masks intersected with reference validity.
- `alpha50_own_valid_depth` and `common_all5_alpha50_depth`: corresponding
  accumulated-alpha50 comparisons (B retains its occupied-pixel mask).
- Both own-domain and common-domain interior/boundary Z errors are retained.

Each new method has `first_hit_diagnostics` in per-view and aggregate results:
`hit_count`, `qmin_gt9_count/percent`, `qmin_median/p95`,
`first_alpha_p05/median/p95`, and `first_alpha_ge_0_5_count/percent`.
`alpha50_percent` is only a compatibility alias of the last percentage and is
explicitly not the cumulative-alpha mask. Aggregate quantiles pool all eligible
pixels rather than averaging per-view quantiles.

New C/D/E PLYs are under `backprojected/<method>/`: ten views plus one merged
file each, with exact PLY reread checks. Merges preserve duplicate surfaces.
The existing A/B PLYs are matched to the new PNG backprojection read-only.

Float64 `error_arrays/<method>/<view>_nn_m.npy` and
`<view>_signed_mesh_Z_m.npy` use NaN only outside the evaluated domain.
`masks/` records both five-method common domains and each method's primary and
alpha50 masks.

## Figures and interpretation

`figures/nn_heatmaps/` and `figures/signed_mesh_Z/` each contain 50 PNGs, with
shared default colour limits of 0–30 mm NN and -150 to +150 mm signed Z.
Saturation counts are recorded; numeric metrics never exclude saturated values.
Gray means not evaluated. Positive signed Z is behind the first mesh surface.

Fixed representative views 3202, 3376, and 3481 each have a five-column,
three-row RGB / NN / signed-Z scientific comparison in PNG and PDF. Views were
fixed before v008 results. Text extents are checked before figure saving, and
PNGs are reopened. Human visual inspection remains an explicit final check.

One-sided NN measures proximity to any source surface. It cannot establish
visibility or completeness and structurally favours B's original-point choices.
Read coverage and both signs of mesh-Z error alongside NN. Common-mask metrics
do not evaluate the regions missing from the sparsest method. These results are
synthetic depth comparisons, not training or blanket occlusion acceptance.
