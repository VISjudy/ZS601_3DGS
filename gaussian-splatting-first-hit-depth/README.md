# Controlled first-contributor depth

This branch adds forward-only first-contributor depth to `gaussian-splattingWithMask/submodules/diff-gaussian-rasterization/cuda_rasterizer/forward.cu`. It preserves the existing RGB compositor and weighted inverse-depth mode. No Gaussian optimisation is performed.

The user-supplied Drive `forward.cu` already recorded the first contributing Gaussian's inverse center Z. Its exact source is retained under `provenance/`. This extension makes the mode explicit and adds ray-peak geometry and diagnostics. The original `synthetic-lidar-init-v4` branch is preserved.

| Mode | Depth returned in channel 0 | Python decoding |
|---|---|---|
| 0 default | Sum of `T * alpha / center_Z` | `Z = accumulated_alpha / raw` |
| 1 | `1 / first_center_Z` | `Z = 1 / raw` |
| 2 | `1 / first_ray_peak_Z` | `Z = 1 / raw` |

Modes 1 and 2 return eight channels: raw inverse Z, Gaussian ID plus one, center Z, ray-peak Z, 3-sigma entry Z, 3-sigma exit Z, first fragment alpha, and minimum Mahalanobis squared distance. ID zero means no contributor. The Python wrapper rejects point counts at or above 2^24 so packed float IDs remain exact. The generic training renderer continues to use mode 0. `run_first_hit.py` explicitly passes the requested mode and applies its matching decoding. Backpropagation in modes 1 and 2 raises an error; the old weighted-depth backward formula must not be reused for them.

## Selection and geometry

The first contributor is the first Gaussian in the existing center-Z sorted tile list that survives the compositor's support/alpha tests and early termination logic. Opacity is float32 sigmoid(20) = 1, but fragment alpha is still capped at 0.99, decreases away from the projected center, and must reach 1/255. RGB and transmittance updates are unchanged. This is not exact ray entry-surface ordering.

Let the world ray be `x(t) = camera_center + t * R_transpose * [(x+.5-cx)/fx, (y+.5-cy)/fy, 1]`, and let `Q = covariance_inverse`. For `m = camera_center - gaussian_center`, the ray quadratic is `A t^2 + 2 B t + C`, where `A = d^T Q d`, `B = m^T Q d`. Its peak is `t = -B/A`. Because the camera ray has Z component one, t is camera-Z in metres, not Euclidean range. The implementation uses double precision for inverse covariance and the closest-point calculation, then exports float32 diagnostics.

If the ray intersects the finite 3-sigma ellipsoid, the peak is the midpoint of its two intersections. If `qmin > 9`, no such intersections exist; entry/exit are zero while peak depth remains defined for the infinite Gaussian. This mirrors the published formula's lack of a discriminant gate, but is explicitly diagnosed. Mini-Splatting2 selects the maximum `alpha*T` contributor, whereas this experiment deliberately keeps the first contributor. This is a controlled variant, not a full reproduction of that paper. See the primary-source research notes for exact commits and lines.

This runner requires centered principal points; it asserts `cx = width/2` and `cy = height/2`. It does not silently accept off-center cameras. Singular covariance produces invalid peak depth and the supplied runner fails if any first contributor has an invalid positive depth.

## Reproduce in Colab

The verified run used Python 3.13, PyTorch 2.11.0+cu128, CUDA 12.8, and an NVIDIA L4. Use the official Colab CLI from WSL2 on Windows. Keep user authentication local; no token belongs in the notebook or repository.

```bash
git clone --branch synthetic-first-hit-depth-v008 https://github.com/VISjudy/ZS601_3DGS.git
cd ZS601_3DGS
python -m pip install plyfile==1.1.3 ninja
MAX_JOBS=2 TORCH_CUDA_ARCH_LIST=8.9 python -m pip install --no-build-isolation --no-deps ./gaussian-splattingWithMask/submodules/diff-gaussian-rasterization
MAX_JOBS=2 TORCH_CUDA_ARCH_LIST=8.9 python -m pip install --no-build-isolation --no-deps ./gaussian-splattingWithMask/submodules/simple-knn
python gaussian-splatting-first-hit-depth/run_first_hit.py \
  --package "$PWD/gaussian-splatting-lidar-init" --root /content/firsthit-input
```

Prepare `/content/firsthit-input` with `config/run_spec.json`, `config/selected_views.json` copied to its root, the exact v007 Gaussian PLY at `input/gaussian_opacity1_scale05.ply`, and the ten v007 `images/*.png` / `depth/*.png` copied to `baseline/` / `baseline_depth/`. The input PLY SHA256 is checked before use. All these data are available in the delivered local v007 folder; large point clouds and renders are not committed to Git. `results/` must not already exist.

The runner first executes analytical tests using an off-axis anisotropic Gaussian, a rotated/translated camera, overlapping front/back Gaussians, no-hit pixels, background-invariant first depth, and the backward guard. It then renders three variants on the same fixed ten cameras, checks every PNG by reading it back, verifies mode-0 RGB/depth against the historical PNGs bit-for-bit, and verifies modes 1 and 2 share IDs/RGB. The actual successful notebook including installation/build/test output is under `notebooks/`; it used a SHA256-verified local archive supplied through the CLI rather than a mutable Git checkout.

## Delivered formats

- `depth/*.png`: uint16, one channel, camera Z in millimetres; zero invalid. Decode as `Z_metres = value / 1000`. Values outside 0–65.535 m are rejected. No visualization colour map is stored in these metric images.
- `depth_mask/*.png`: uint8, one channel, 255 for finite positive first-contributor depth, zero invalid. This primary mask has no alpha50 cutoff.
- `depth_mask_alpha50/*.png`: primary mask AND accumulated alpha at least 0.5.
- `masks/*.png`: uint8, 255 where accumulated alpha is at least 0.95; zero elsewhere. The user's RGB gap metric is the zero fraction of this mask.
- `alpha/*.png`: uint16, one channel, rounded accumulated alpha times 65535. Its zero fraction is a separate strict diagnostic.
- `images/*.png`: RGB uint8 with black background; the RGB compositor is unchanged.
- `depth_float/*.npy`: float32 camera Z in metres before PNG quantization.
- `aux/*.npz`: zero-based `first_id` (-1 for no contribution), `center_z`, `peak_z`, `entry_z`, `exit_z`, `first_alpha`, `qmin`. Z values are float32 metres; qmin and alpha are dimensionless. Test qmin, not an arbitrary entry/exit sentinel, to classify finite-ellipsoid intersection.
- Scale0.1 Gaussian PLY: the exact same positions, colours, rotations, opacity and SH as scale0.5; only the three log-scales receive `log(0.1/0.5)`.

Backprojection/evaluation reads the saved PNGs, uses pixel centers `(x+.5,y+.5)`, and reports nearest-input-point error separately from signed depth error against the frozen glass-free mesh's first surface. Neither a low nearest-point error nor successful file verification proves correct visibility or successful 3DGS training. Newly recolored v009 data are a separate experiment and are not mixed into these five depth conditions.

## Measured comparison

All five methods use the same ten cameras and the same 1 cm no-glass geometry. The intersection of their valid masks and the reference contains 2,781,085 pixels.

| Method | Primary coverage percent | NN mean on own valid points mm | Z MAE on shared pixels mm |
|---|---:|---:|---:|
| A harmonic scale0.5 | 99.9723 | 6.7195 | 22.0188 |
| B pure point z-buffer | 39.2209 | 1.8522 | 100.7400 |
| C first center scale0.5 | 99.9793 | 12.5364 | 46.9591 |
| D same-first peak scale0.5 | 99.9793 | 10.0522 | 43.6515 |
| E first peak scale0.1 | 97.5069 | 4.1721 | 23.0348 |

The NN column has different supports and structurally favors the point z-buffer. Use the matched-support and signed-occlusion metrics under `verified/evaluation/` to interpret it. First-contributor center/peak at scale0.5 eliminate deep-background blending in the shared domain but introduce stronger front-surface bias. Scale0.1 is not unconditionally better: shared MAE remains slightly worse than harmonic, behind-surface errors above 30 mm increase to 2.5225%, and RGB alpha95-mask gaps reach 70.1037%. For C/D, 57.9625% of first contributors have no real 3-sigma intersection; for E this is 73.3808%.

`evaluation_snapshot/` retains the exact locally executed CPU evaluator and plotter, with their original workspace-relative dependency/data defaults. Supply the indicated baseline/new-root arguments and reproduce the documented workspace layout when using that snapshot; it is not an automatic data downloader. Full-resolution images, point clouds, HTML and DOCX reports remain in the local versioned delivery folders rather than this code repository.
