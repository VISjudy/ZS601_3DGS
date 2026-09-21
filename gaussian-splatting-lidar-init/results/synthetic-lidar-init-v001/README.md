# Verified run evidence

`REPORT.md` is a copy of the local delivery report. Its image, PLY and
`reproducibility/` paths are relative to the local artifact directory:

`D:/codex/blenderProject/scenes/zs601-meetingroom/gaussian-initialization-v001/`

Large inputs, point clouds and rendered images are not committed. The local
format specification is also available as the [package README](../../README.md).
The [executed notebook](../../notebooks/ZS601_LiDAR_Init_Colab.executed.ipynb)
contains the actual five executed code cells and outputs.

Executed commit: `4c7186e363f14050c4977bb192f12ed237112764`.
The later publication commit adds evidence, documentation, a CLI packaging helper,
and preserves the unused original reference's CRLF bytes. Rendering code is unchanged.

Verified: 200 views, 309240 Gaussians, 1400 PNGs, exact input XYZ, finite opacity,
PLY reload equality, COLMAP matrix equality, PNG headers and decoded data, archive
and artifact hashes. Independent local 3-NN scale check: 1024 points, maximum error
7.8799755e-9 m. Optimization steps: **0**.
