# Colored LiDAR initialization and virtual-view rendering

Latest: [mesh 1 cm, scale 0.5, full 200-view run](results/synthetic-mesh1cm-full-v005/README.md)
and [four-configuration Black RGB / mask comparison](results/synthetic-four-config-mask-v001/README.md).
The user selected scale 0.5 after the four-camera comparison; all 200 renders and
2000 channel PNGs are locally verified. The actual executed notebook is included.
The full run exposes opaque white glass artifacts: lower mask-gap ratios do not
guarantee better appearance, and full-scene PSNR is lower than the old 3 cm run.
Read the quality review before using these pseudo-images for training. The user
defines grain as zero-valued valid-mask ratio (alpha < 0.95), with strict alpha=0
reported separately. Formal training retains the original 3 cm initialization.

Earlier four-camera records: [scale 1.0 baseline](results/synthetic-mesh1cm-smoke-v003/README.md),
[paired scale 0.5 smoke](results/synthetic-mesh1cm-scale05-v004/README.md). Their
review gates describe the historical smoke stage, now followed by the authorized
full run. The original v001 baseline is documented below.

This folder adapts the supplied `render_from_sparse_v4.py` for the ZS601 synthetic
dataset. It creates and reloads an **iteration-0** Gaussian PLY, then renders the
exact 200 `virtual_near` COLMAP cameras. **No optimizer or training step runs.**

The source input is the 309,240-point `cloud/points_3cm.ply` from
`synthetic-training-v001`. It is a **3 cm voxel cloud fused only from Blender
training RGB-D views**, not a raw measured LAS scan. Rendering needs only colored
PLY and COLMAP TXT; held-out RGB/depth are opened only by the evaluation script.

## Provenance

- User source: [Drive folder](https://drive.google.com/drive/folders/1DPzzWWKEyHQBqgsH_x8u5taBOa3y4mUf).
- Original entrypoint SHA256: `8a74e8b175cf3173d6572c6959e192ef1f766f7ab48d706d528f0a2e96ade490`.
- Exact original: `provenance/render_from_sparse_v4.original.py`; dependency hashes:
  `provenance/drive_source_manifest.json`.
- `scene/`, `utils/`, `arguments/`, `gaussian_renderer/` retain the downloaded code.
- CUDA extensions use the vendored sources at
  `../gaussian-splattingWithMask/submodules/{diff-gaussian-rasterization,simple-knn}`
  from base commit `c28d66eaee6b988c09a6ac6e6f34b2d48fd91f8f`, with one compile-only
  addition: `#include <cstdint>` in `rasterizer_impl.h`. CUDA 12.8 reported undefined
  `std::uintptr_t`, `uint32_t` and `uint64_t`; no rendering math or alpha cap changed.
- Keep the included upstream research license; this adaptation grants no new rights.

## Initialization

One isotropic Gaussian per input point. XYZ stays in original COLMAP world metres.
RGB is encoded into SH degree 0 (the supplied image values are not gamma-converted).
Rotation is identity. No densification, pruning, exposure fitting or optimization.

Opacity target is **0.999999**, represented as a finite float32 logit; actual decoded
values are recorded. Exact 1 would require an infinite logit and is deliberately
avoided. The unchanged stock rasterizer still caps each fragment alpha at **0.99**.
High model opacity does not guarantee complete pixel coverage between points.

Default scale is **0.5 times the square root of the mean squared distance to the
three nearest neighbours**, isotropic in metres. The original uniform 0.001 m
default made its KNN scaling branch unreachable; the uniform override now defaults
to `None`. `--init-scale-uniform` remains available explicitly. No scale parameter
is selected using held-out image quality.

`point_cloud/iteration_0/point_cloud.ply` is a standard binary Gaussian parameter PLY:
float32 XYZ, zero placeholder normals, SH0 `f_dc_0..2`, **logit** `opacity`, **log**
`scale_0..2`, quaternion `rot_0..3` (wxyz). It is not an RGB point PLY. Use
`sparse/0/points3D.ply` for the original colored point cloud. Gaussian PLY reload is
checked bit for bit before rendering. This SH0 checkpoint has no `f_rest_*` fields;
load it with `GaussianModel(sh_degree=0)`.

## PNG and camera contract

Every view keeps its original image ID, filename, resolution and calibration.
Current data is 640 x 1108 pixels. World-to-camera rotation uses COLMAP Hamilton
wxyz quaternions, camera axes right/down/forward. Top-left pixel centre is (0.5,0.5).
No automatic train/test split or reordering by the old Scene loader is used.
Camera near projection plane is 0.01 m; the unchanged CUDA kernel rejects Gaussian
centres at camera Z <= 0.2 m. Distortion models must be undistorted beforehand.

| Folder | PNG format | Meaning |
|---|---|---|
| `images/` | uint8 RGB, 3 channels | Native standard 3DGS alpha-composited render over black; zero outside splat support. |
| `color/` | uint8 RGB, 3 channels | Straight RGB, native RGB divided by accumulated alpha; matches the original v4 color output intent. |
| `rgba/` | uint8 RGBA, 4 channels | Straight RGB and accumulated alpha, not premultiplied. |
| `alpha/` | uint16 grayscale, 1 channel | Accumulated opacity: `alpha = value / 65535`. |
| `masks/` | uint8 grayscale, 1 channel | 255 where accumulated alpha >= 0.95, otherwise 0. White means valid. |
| `depth/` | uint16 grayscale, 1 channel | Alpha-normalized harmonic camera-Z depth in **millimetres**, `Z_m = value / 1000`. 0 invalid. |
| `depth_mask/` | uint8 grayscale, 1 channel | 255 where alpha >= 0.5 and depth is positive; otherwise 0. |

The rasterizer returns `Dinv = sum(T_i * alpha_i / Z_i)`. Depth is
`Z = alpha_accumulated / Dinv`, not `1/Dinv`. A single-Gaussian CUDA analytic check
verifies depth is independent of coverage. Depth is a compositing statistic of
Gaussian centres, not exact surface intersection or LiDAR range. It is valid up to
65.535 m; larger valid depths raise an error instead of clipping. Background is 0.
No visualization colormap is stored as numeric depth. Normal maps are not inferred
from isotropic Gaussians; the existing Blender dataset retains its normal GT.

`sparse/0/` contains `cameras.txt`, `images.txt`, `points3D.txt` and original colored
`points3D.ply`. Point tracks are empty; ERROR=0 is a placeholder, not a measured SfM
reprojection error. This is the standard camera/point layout accepted by common
3DGS loaders, not a fully reconstructed SfM feature database.

For training augmentation, use `color/` with the explicit valid mask if avoiding
low-coverage darkening. A stock trainer does not automatically honor these masks.
Keep pseudo-images train-only; do not replace the Blender RGB/depth/normal GT or
mix these 200 near views into held-out robustness tests.

## Run on Colab

Use `notebooks/ZS601_LiDAR_Init_Colab.ipynb`. The executed counterpart records the
actual GPU, environment, checks, smoke, full render and verification outputs.
Inputs/large PNGs/point clouds are not committed to GitHub. The input bundle must
contain `points_3cm.ply`, `sparse/0/{cameras,images}.txt`, and optional `gt/` folders
for post-render comparison. Use the official Colab CLI upload and run the notebook
via `nbclient`; interactive Drive mount is unnecessary.

For the delivered run, reuse `reproducibility/source_bundle.tar.gz` and
`reproducibility/source_manifest.json`, plus the experiment's
`inputs/input_bundle.zip`. Upload these three files into a **new** remote directory.
The reusable `cli/deploy_and_run.py` assembles that immutable source/input pair and
starts one notebook worker. Its paths are controlled by `ZS601_RUN_ROOT`:

```bash
colab --auth=oauth2 exec -s YOUR_SESSION -f /local/path/cli/deploy_and_run.py \
  --env ZS601_RUN_ROOT=/content/zs601-lidar-reproduce --timeout 60
```

On Windows, run this CLI in WSL2 and use `/mnt/d/...` local paths. Allocate one GPU
session first and use CLI upload/download for transfers. Inspect `launch.json`,
`notebook_status.json` and `artifacts_ready.json`; download and hash-check the
artifact archive before stopping the session. The delivered executed notebook is
the primary evidence of the successful run; the launcher is a packaging helper.

Source download hashes refer to original CRLF files; see
`provenance/LINE_ENDINGS.md` for Git byte-normalization details.

Direct entrypoints after CUDA extension installation:

```bash
python check_contract.py --sparse /path/to/input/sparse/0
python render_from_sparse_v4.py --point-cloud /path/to/input/points_3cm.ply \
  --sparse /path/to/input/sparse/0 --output /path/to/new-output \
  --opacity 0.999999 --init-scale-factor 0.5
python verify_outputs.py --output /path/to/new-output \
  --point-cloud /path/to/input/points_3cm.ply --sparse /path/to/input/sparse/0 \
  --ground-truth /path/to/input/gt --expected-views 200
```

Output directories must not already exist. Failure artifacts are retained.
`COMPLETE.json` means parameter/render/package checks passed, **not trained quality**.
Per-view PSNR/SSIM and depth errors compare initialization renders to fixed Blender
GT after rendering. They measure this synthetic scene and are not real-world accuracy.
