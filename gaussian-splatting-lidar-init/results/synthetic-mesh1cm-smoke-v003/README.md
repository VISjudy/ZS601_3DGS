# Full mesh 1cm cloud: four-view smoke

This follow-up uses 7,180,820 points sampled from all 2,088 render-visible Blender
mesh components, including camera-unobserved surfaces. Geometry is not a
backprojection or an upsample of the old 3 cm cloud. Four uniform area candidates
per cm² are reduced to one true surface point per occupied 1 cm world voxel within
each component. Independent components are kept separate; this is not a strict
minimum-spacing guarantee. All-point distance to the source triangle is below
3.32e-7m. The 15 inherited zero-area triangles were excluded.

Visible point colors use training images only, with mesh ray occlusion tests and
bilinear RGB selection. 2,605,400 points receive observed RGB. Unobserved or
transparent surfaces retain constant material base colors; those colors do not
include baked textures/lighting. No test, validation or virtual-view RGB enters
color assignment. Frozen BlenderGT is used only for smoke evaluation.

**Roles stay separate:** the 1 cm cloud initializes Gaussians for virtual-view
rendering. Formal training still initializes from the original 309,240-point 3 cm
cloud, whose hash is unchanged. No training input file or default is replaced.

Run settings: k=3, scale multiplier 1.0, opacity target 0.999999, SH0, zero
optimization steps. Gaussian sigma median 8.795322 mm. CPU 3-neighbor checks on
20,000 samples match the actual GPU scales within 3.46e-9 m.

| Same 4 views, 640×1108 | Old 3 cm, scale 1 | Mesh 1 cm, scale 1 |
|---|---:|---:|
| Black PSNR |21.8708dB|24.0372dB|
| Black SSIM |0.846062|0.852090|
| Straight PSNR |21.8552dB|24.0301dB|
| Alpha≥0.95 coverage |99.9848%|100%|
| Depth MAE |97.7160mm|36.8787mm|

Depth uses each result's valid intersection. This is a smoke comparison, not a
200-view result or a trained model. Visual inspection found sharper fixture and
equipment edges, with remaining edge grain and view-dependent highlight errors.

The actual executed notebook is
[ZS601_Mesh1cm_Smoke.executed.ipynb](../../notebooks/ZS601_Mesh1cm_Smoke.executed.ipynb).
Its four code cells ran on Colab L4, Python 3.13.15, torch 2.11.0+cu128/CUDA12.8,
using renderer commit 4c7186e363f14050c4977bb192f12ed237112764 and verified reused
CUDA wheels. It expects the hashed source/input/wheels deployed by the official
Colab CLI. Local preparation scripts and input/archive manifests accompany the
local delivery. It does not perform interactive Drive authentication.

Gaussian PLY, ordinary 1 cm PLY, 28 PNGs and four-view comparison images are saved in
the user's local 3dgsResult/zs601-mesh1cm-scale1-v003 delivery, not committed here.
The notebook contains no full 200-view rendering cell. User image review and
confirmation are required before that next phase. The GPU session was released.
