"""Ten-view initialized Gaussian depth smoke; no optimization and no GT access.

Reuses the verified k=3 RMS * 0.5 initialization. Only opacity changes to a
finite logit whose float32 sigmoid is exactly one. The unchanged rasterizer
still caps each fragment alpha at 0.99. Its depth is alpha-normalized harmonic
camera Z, not a first-surface depth.
"""
import argparse
import json
from pathlib import Path
import sys
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--package', required=True)
    p.add_argument('--source-gaussian', required=True)
    p.add_argument('--input-cloud', required=True)
    p.add_argument('--spec', required=True)
    p.add_argument('--views', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    sys.path.insert(0, args.package)
    import numpy as np
    from PIL import Image
    import torch
    from torch import nn
    from render_from_sparse_v4 import (GaussianModel, camera, render,
                                      check_analytic_cuda, stats, load_cloud)
    from colmap_io import View, sha256, write_json, encode_depth

    spec = json.loads(Path(args.spec).read_text())
    views = [View(**r) for r in json.loads(Path(args.views).read_text())]
    assert [v.image_id for v in views] == spec['view_ids'] and len(views) == 10
    assert sha256(args.input_cloud) == spec['input_cloud_sha256']
    assert sha256(args.source_gaussian) == spec['source_gaussian_sha256']
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    write_json(out/'config.json', dict(spec=spec, actual_arguments=vars(args),
        torch=torch.__version__, gpu=torch.cuda.get_device_name(), optimization_steps=0))
    write_json(out/'analytic_cuda_check.json', check_analytic_cuda())
    with torch.no_grad():
        pc = GaussianModel(0)
        pc.load_ply(args.source_gaussian)
        xyz, _ = load_cloud(args.input_cloud)
        assert np.array_equal(pc.get_xyz.cpu().numpy(), xyz)
        del xyz
        pc._opacity = nn.Parameter(torch.full_like(pc._opacity, 20.0), requires_grad=False)
        assert torch.all(pc.get_opacity == 1.0).item()
        gauss = out/'point_cloud/iteration_0/point_cloud.ply'
        pc.save_ply(str(gauss))
        reloaded = GaussianModel(0)
        reloaded.load_ply(str(gauss))
        for attr in ['_xyz', '_features_dc', '_features_rest', '_opacity', '_scaling', '_rotation']:
            assert torch.equal(getattr(pc, attr), getattr(reloaded, attr)), attr
        write_json(out/'initialization.json', dict(
            point_count=len(pc.get_xyz), sh_degree=0, optimization_steps=0,
            source_gaussian_sha256=spec['source_gaussian_sha256'],
            input_cloud_sha256=spec['input_cloud_sha256'],
            gaussian_ply_sha256=sha256(gauss), changed_parameter='opacity only',
            knn_k=3, scale_factor=0.5, sigma_formula='0.5 * sqrt(mean(d_nearest3 ** 2))',
            nominal_opacity=1.0, actual_float32_opacity=stats(reloaded.get_opacity),
            opacity_logits=stats(reloaded._opacity),
            mathematical_sigmoid20=1/(1+np.exp(-20.0)),
            kernel_per_fragment_alpha_cap=0.99, kernel_near_cutoff_m=0.2,
            scales_metres=stats(pc.get_scaling), ply_reload_exact=True,
            xyz_equal_input=True, source_other_fields_preserved=True))
        del pc
        for folder in ['images','alpha','masks','depth','depth_mask','depth_float']:
            (out/folder).mkdir()
        rows=[]
        started=time.time()
        with (out/'render_progress.jsonl').open('x') as log:
            for i,v in enumerate(views):
                ts=time.time()
                rgb,_,alpha,depth,visible=render(camera(v),reloaded)
                valid=(alpha >= 0.5)&np.isfinite(depth)&(depth>0)
                depth=np.where(valid,depth,0).astype(np.float32)
                arrays=dict(images=np.rint(rgb*255).astype(np.uint8),
                    alpha=np.rint(alpha*65535).astype(np.uint16),
                    masks=(alpha>=0.95).astype(np.uint8)*255,
                    depth=encode_depth(depth,valid), depth_mask=valid.astype(np.uint8)*255)
                for folder,array in arrays.items():
                    Image.fromarray(array).save(out/folder/v.name)
                    assert np.array_equal(np.array(Image.open(out/folder/v.name)),array)
                np.save(out/'depth_float'/Path(v.name).with_suffix('.npy'),depth)
                row=dict(image_id=v.image_id,name=v.name,visible_gaussians=visible,
                    valid_depth_pixels=int(valid.sum()),
                    depth_coverage=float(valid.mean()),
                    rgb_coverage_alpha95=float((alpha>=0.95).mean()),
                    strictly_empty_alpha16_fraction=float((arrays['alpha']==0).mean()),
                    quantization_max_error_mm=float(np.max(np.abs(arrays['depth'][valid]/1000-depth[valid]))*1000),
                    seconds=time.time()-ts)
                rows.append(row);log.write(json.dumps(row)+'\n');log.flush()
                print(json.dumps(row),flush=True)
        write_json(out/'render_summary.json',dict(status='PASS',views=len(rows),rows=rows,
            seconds=time.time()-started,depth='alpha / sum(T_i * alpha_i / Z_i)',
            camera_Z_metres=True,depth_mask_alpha_min=0.5,rgb_mask_alpha_min=0.95))
        write_json(out/'RENDER_COMPLETE.json',dict(status='SMOKE_10_NOT_TRAINED',views=10,
            optimization_steps=0,gaussian_ply_sha256=sha256(gauss)))


if __name__ == '__main__':
    main()
