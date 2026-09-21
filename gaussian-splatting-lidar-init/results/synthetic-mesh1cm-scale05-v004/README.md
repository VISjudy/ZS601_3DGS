# Same mesh 1 cm cloud: scale 1.0 versus 0.5

Four-view smoke only, using exactly the same 7,180,820 points, colors, opacity,
rotations and cameras as the scale 1.0 baseline. Every non-scale Gaussian field
was checked for exact equality. The scale 0.5 sigma median is
4.397661 mm, half of the baseline. k=3, opacity target
0.999999, SH0, zero optimizer steps. Formal training still uses the original
3 cm cloud; it remains unchanged. No full 200-view run was launched.

| Same four views | Scale 1.0 | Scale 0.5 |
|---|---:|---:|
| Black PSNR |24.0372 dB|24.4738 dB|
| Black SSIM |0.852090|0.831103|
| Straight PSNR |24.0301 dB|24.4935 dB|
| Straight SSIM, identical GT windows |0.852399|0.834120|
| Empty pixels |0.000000%|0.000000%|
| Alpha below 0.95 |0.0000%|0.4855%|
| Depth MAE |36.8787 mm|14.7009 mm|

Assess blur in Straight-to-Straight images as well as native Black images;
partial-opacity dark seams can resemble sharp detail. PSNR improvements do not
imply that SSIM or grain improved. Metrics above cover four smoke views only.

Visual inspection of the four-view GT sheet and full-resolution Black/Straight
pairs found clearer equipment, table and plant edges at scale 0.5, but more
visible grain on walls, ceilings and equipment. The grain persists in Straight
images, so it is not only darkening from partial alpha over black. Scale 1.0 is
smoother and blurrier. No scale has been selected for the full 200-view run.

## Explicit pixel masks saved locally

Both scales have four masks in each family, uint8 grayscale PNG, 640x1108:

- `no_coverage_masks/scale1` and `scale05`: white 255 means rendered alpha is zero;
  black 0 means some Gaussian contribution. This is not a point-center occupancy map.
- `low_coverage_masks/scale1` and `scale05`: white 255 means alpha<0.95, including
  partial coverage and holes. It is the inverse of the original valid mask.
- Existing `smoke/masks` uses white 255 for valid alpha>=0.95, the opposite polarity.
- `smoke/alpha` stores uint16 alpha, decoded by dividing by 65535.

The exact derivation, applied independently to both scale folders, is:

```python
alpha16 = np.asarray(Image.open(alpha_png))
valid95 = np.asarray(Image.open(original_valid_mask_png)) == 255
empty = (alpha16 == 0).astype(np.uint8) * 255
low_coverage = (~valid95).astype(np.uint8) * 255
Image.fromarray(empty).save(empty_output_png)
Image.fromarray(low_coverage).save(low_coverage_output_png)
```

All 16 derived masks were reloaded and checked against those exact conditions.
Individual pixel counts and fixed-GT/common-coverage metrics are in the CSV.

The [executed notebook](../../notebooks/ZS601_Mesh1cm_Scale05_Smoke.executed.ipynb)
ran four code cells on Colab L4, torch 2.11.0+cu128, Python 3.13.15. It uses the
identical hashed input archive and renderer commit 4c7186e363f14050c4977bb192f12ed237112764.
It expects source, input and CUDA wheels deployed by the official CLI, as in
the scale 1 baseline. Render artifacts and Gaussian PLY remain in the local
3dgsResult/zs601-mesh1cm-scale05-v004 folder. User review is required before 200 views.
