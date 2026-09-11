# ZS601 mask-aware 2DGS fork

This directory is a project-local fork of the official 2D Gaussian Splatting source from `hbb1/2d-gaussian-splatting`, prepared for the ZS601 image layout and mask-aware training.

Imported upstream commit:

```text
f3e3b9fa67bbd1c75e05167ff37391d8dab2a678
```

## What changed

- COLMAP image paths may stay nested, for example `images/map_0/camera_0/*.jpg`.
- A mask directory can be passed with `--masks`.
- Masks are matched from the COLMAP image relative path, using `.png` mask files.
- `--mask_valid_value black` treats black pixels as valid ZS601 static-scene pixels.
- RGB L1, SSIM, validation L1, validation PSNR, normal regularization, and distortion regularization are masked when a mask is available.
- If `--masks` is omitted, behavior falls back to the upstream 2DGS behavior.

## ZS601 COLMAP initialization run

Run from this directory after installing the official 2DGS environment:

```bash
python train.py \
  -s /content/ZS601meetingroom_data \
  --images images \
  --masks masks \
  --mask_valid_value black \
  -m /content/outputs/zs601_2dgs_colmap_sparse \
  --iterations 150000 \
  --sh_degree 2 \
  --position_lr_init 0.000016 \
  --position_lr_final 0.00000016 \
  --position_lr_max_steps 150000 \
  --scaling_lr 0.0015 \
  --densification_interval 10000 \
  --densify_until_iter 100000 \
  --opacity_reset_interval 150000 \
  --densify_grad_threshold 0.0002 \
  --lambda_normal 0.05 \
  --lambda_dist 0.0 \
  --depth_ratio 0 \
  --test_iterations 50000 100000 150000 \
  --save_iterations 50000 100000 150000 \
  --checkpoint_iterations 50000 100000 150000
```

## LiDAR initialization run

Do not modify the original ZS601 dataset in place. Create a copied dataset directory and replace only the copied `sparse/0/points3D.*` with the LiDAR-derived initialization point cloud, then run the same command with a different output directory:

```bash
python train.py \
  -s /content/ZS601meetingroom_data_lidar_init \
  --images images \
  --masks masks \
  --mask_valid_value black \
  -m /content/outputs/zs601_2dgs_lidar_init \
  --iterations 150000 \
  --sh_degree 2 \
  --position_lr_init 0.000016 \
  --position_lr_final 0.00000016 \
  --position_lr_max_steps 150000 \
  --scaling_lr 0.0015 \
  --densification_interval 10000 \
  --densify_until_iter 100000 \
  --opacity_reset_interval 150000 \
  --densify_grad_threshold 0.0002 \
  --lambda_normal 0.05 \
  --lambda_dist 0.0 \
  --depth_ratio 0 \
  --test_iterations 50000 100000 150000 \
  --save_iterations 50000 100000 150000 \
  --checkpoint_iterations 50000 100000 150000
```

The two arms should write to separate output folders and be compared only after both have saved checkpoints, logs, validation renders, and metrics.
