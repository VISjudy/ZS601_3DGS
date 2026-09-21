# Four-camera Black RGB comparison

Order: 3 cm scale 0.5; 3 cm scale 1; 1 cm scale 0.5; 1 cm scale 1.
Same four cameras 003001, 003238, 003301, 003478. The user defines grain as
the proportion of zero-valued pixels in the existing valid masks, which use
alpha >= 0.95 as valid. Strict alpha=0 is a separate auxiliary statistic.
No high-frequency noise score is used in this report.

| Configuration | Black PSNR dB | SSIM | Mask gap % |
|---|---:|---:|---:|
| 3 cm scale 0.5 |20.1033|0.737293|33.033196|
| 3 cm scale 1 |21.8708|0.846062|0.015230|
| 1 cm scale 0.5 |24.4738|0.831103|0.485461|
| 1 cm scale 1 |24.0372|0.852090|0.000000|

Metrics use Black RGB on the same GT-valid pixels/windows. Depth in the CSV
uses the common valid intersection of all four configurations. The 3 cm cloud
was fused from training RGB-D; the 1 cm cloud samples the entire mesh with
training-only visibility coloring and material fallback. This changes more
than density; only the within-cloud scale comparison isolates scale.

The selected 1 cm scale 0.5 [full 200-view run](../synthetic-mesh1cm-full-v005/README.md)
is complete. It exposes opaque white glass artifacts: low gap ratios do not
guarantee correct appearance. See that report before using pseudo-images for
training. Formal training initialization remains the original 3 cm cloud.

Interactive HTML with all 16 original images, GT comparisons, mask figures,
per-camera tables and the full-run failure case is saved locally at
`D:/codex/blenderProject/3dgsResult/comparison-four-config-v001/index.html`.
