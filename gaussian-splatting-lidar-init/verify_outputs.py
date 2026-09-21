"""Verify iteration-0 parameters, every PNG and pose; optionally compare Blender GT.

Ground truth is used only here after rendering. It never enters initialization.
"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from plyfile import PlyData
from scipy.ndimage import gaussian_filter, binary_erosion
from colmap_io import read_views, rotation, sha256, write_json


def ssim_map(a, b):
    def smooth(v):
        return gaussian_filter(v, sigma=(1.5, 1.5, 0), truncate=10/3, mode='reflect')
    ma, mb = smooth(a), smooth(b)
    va, vb = smooth(a*a)-ma*ma, smooth(b*b)-mb*mb
    cov = smooth(a*b)-ma*mb
    return (((2*ma*mb+0.01**2)*(2*cov+0.03**2)) /
            ((ma*ma+mb*mb+0.01**2)*(va+vb+0.03**2))).mean(axis=2)


def psnr(a, b, m):
    if not m.any():
        return None
    mse = float(np.mean((a[m]-b[m])**2))
    return float(-10*np.log10(max(mse, 1e-12)))


def verify(args):
    root, sparse = Path(args.output), Path(args.sparse)
    source = read_views(sparse)
    result = read_views(root/'sparse/0')
    assert len(result) == args.expected_views
    src = {v.image_id: v for v in source}
    for v in result:
        a = src[v.image_id]
        assert v.name == a.name and v.camera_id == a.camera_id
        assert np.array_equal(v.matrices()[0], a.matrices()[0])
        assert np.array_equal(v.matrices()[1], a.matrices()[1])
    cloud = PlyData.read(args.point_cloud)['vertex'].data
    gauss = PlyData.read(root/'point_cloud/iteration_0/point_cloud.ply')['vertex'].data
    names = list(gauss.dtype.names)
    assert len(gauss) == len(cloud)
    assert names == ['x','y','z','nx','ny','nz','f_dc_0','f_dc_1','f_dc_2','opacity','scale_0','scale_1','scale_2','rot_0','rot_1','rot_2','rot_3']
    for n in names:
        assert np.isfinite(gauss[n]).all(), n
    for n in ['x','y','z']:
        assert np.array_equal(gauss[n], cloud[n]), n
    dc = np.column_stack([gauss[f'f_dc_{i}'] for i in range(3)])
    rgb = np.column_stack([cloud[n] for n in ['red','green','blue']])/255
    color_error = float(np.max(np.abs(dc*0.28209479177387814+0.5-rgb)))
    assert color_error < 1e-6
    opacity = 1/(1+np.exp(-gauss['opacity'].astype(np.float64)))
    assert np.all((opacity > 0.99999) & (opacity < 1))
    assert np.all(gauss['rot_0'] == 1) and all(np.all(gauss[n] == 0) for n in ['rot_1','rot_2','rot_3'])
    assert np.array_equal(gauss['scale_0'], gauss['scale_1']) and np.array_equal(gauss['scale_0'], gauss['scale_2'])
    formats = dict(images=(8,2,3), color=(8,2,3), rgba=(8,6,4), alpha=(16,0,1), masks=(8,0,1), depth=(16,0,1), depth_mask=(8,0,1))
    expected_names = {v.name for v in result}
    for folder in formats:
        assert {p.name for p in (root/folder).glob('*.png')} == expected_names, folder
    rows = []
    gtroot = Path(args.ground_truth) if args.ground_truth else None
    for i, v in enumerate(result):
        arrays = {}
        for folder, (bits, color_type, channels) in formats.items():
            p = root/folder/v.name
            with p.open('rb') as f:
                head = f.read(26)
            assert head[24:26] == bytes([bits, color_type]), str(p)
            with Image.open(p) as im:
                assert im.size == (v.width, v.height), str(p)
                arrays[folder] = np.array(im)
            if channels > 1:
                assert arrays[folder].shape[2] == channels
        assert arrays['images'].max() > 0
        assert set(np.unique(arrays['masks'])).issubset({0,255})
        assert set(np.unique(arrays['depth_mask'])).issubset({0,255})
        assert np.array_equal(arrays['depth'] > 0, arrays['depth_mask'] == 255)
        # Straight RGB and alpha must reconstruct native RGB within quantization.
        reconstructed = arrays['color'].astype(np.float32)*(arrays['alpha'].astype(np.float32)/65535)[...,None]
        assert float(np.abs(reconstructed-arrays['images']).max()) <= 1.6
        row = dict(image_id=v.image_id, name=v.name,
            alpha95_coverage=float((arrays['masks'] == 255).mean()),
            depth_coverage=float((arrays['depth_mask'] == 255).mean()))
        if gtroot:
            gt = np.array(Image.open(gtroot/'images'/v.name)).astype(np.float32)/255
            gm = np.array(Image.open(gtroot/'masks'/v.name)) == 255
            native = arrays['images'].astype(np.float32)/255
            straight = arrays['color'].astype(np.float32)/255
            covered = gm & (arrays['masks'] == 255)
            ssim_valid = binary_erosion(covered, structure=np.ones((11,11)), border_value=0)
            gt_window = binary_erosion(gm, structure=np.ones((11,11)), border_value=0)
            sn, sc = ssim_map(native, gt), ssim_map(straight, gt)
            gd = np.array(Image.open(gtroot/'depth'/v.name)).astype(np.float64)/1000
            geometry = np.array(Image.open(gtroot/'geometry_masks'/v.name)) == 255
            dmask = geometry & (gd > 0) & (arrays['depth_mask'] == 255)
            delta = arrays['depth'].astype(np.float64)/1000 - gd
            row.update(native_psnr_gt_valid_db=psnr(native,gt,gm), straight_psnr_gt_valid_db=psnr(straight,gt,gm),
                straight_psnr_covered_db=psnr(straight,gt,covered),
                native_ssim_gt_valid=float(sn[gt_window].mean()) if gt_window.any() else None,
                straight_ssim_covered=float(sc[ssim_valid].mean()) if ssim_valid.any() else None,
                ssim_covered_window_fraction=float(ssim_valid.mean()),
                depth_mae_mm=float(np.abs(delta[dmask]).mean()*1000) if dmask.any() else None,
                depth_rmse_mm=float(np.sqrt(np.mean(delta[dmask]**2))*1000) if dmask.any() else None,
                depth_absrel=float(np.mean(np.abs(delta[dmask])/gd[dmask])) if dmask.any() else None,
                depth_eval_coverage=float(dmask.mean()))
        rows.append(row)
        if i % 25 == 0:
            print('verified', i+1, '/', len(result), flush=True)
    csvpath = root/'per_view_metrics.csv'
    with csvpath.open('x', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    means = {k: float(np.mean([r[k] for r in rows if r[k] is not None]))
        for k in rows[0] if k not in ['image_id','name'] and any(r[k] is not None for r in rows)}
    report = dict(status='VERIFIED_INITIALIZATION_NOT_TRAINED', views=len(result), points=len(gauss),
        optimization_steps=0, exact_xyz=True, max_sh0_rgb_error=color_error,
        opacity_decoded_min=float(opacity.min()), opacity_decoded_max=float(opacity.max()),
        png_count=len(result)*len(formats), camera_pose_and_intrinsics_exact=True,
        means_over_views=means,
        evaluation='Blender synthetic GT, fixed near views; no real-image reconstruction or optimized 3DGS claims',
        ssim='RGB [0,1], Gaussian 11x11 sigma1.5, C1=0.01^2 C2=0.03^2, fully valid windows')
    write_json(root/'verification.json', report)
    if gtroot:
        selected = [result[i] for i in np.linspace(0, len(result)-1, min(6,len(result)), dtype=int)]
        thumb_w, thumb_h, label_h = 220, 381, 36
        sheet = Image.new('RGB', (thumb_w*4, (thumb_h+label_h)*len(selected)), '#202020')
        draw = ImageDraw.Draw(sheet)
        for r,v in enumerate(selected):
            paths = [gtroot/'images'/v.name, root/'images'/v.name, root/'color'/v.name, root/'alpha'/v.name]
            for c,(p,label) in enumerate(zip(paths,['Blender GT','3DGS black','3DGS straight','Alpha'])):
                with Image.open(p) as im:
                    if c==3:
                        im=Image.fromarray(np.rint(np.array(im)/257).astype(np.uint8)).convert('RGB')
                    im=im.convert('RGB').resize((thumb_w,thumb_h))
                    x,y=c*thumb_w,r*(thumb_h+label_h)
                    sheet.paste(im,(x,y+label_h))
                    draw.text((x+5,y+4),v.name+' '+label,fill='white')
        sheet.save(root/'comparison_contact_sheet.jpg', quality=92)
    records = []
    for p in sorted(root.rglob('*')):
        if p.is_file():
            records.append(dict(path=p.relative_to(root).as_posix(), bytes=p.stat().st_size, sha256=sha256(p)))
    write_json(root/'artifact_manifest.json', records)
    write_json(root/'COMPLETE.json', dict(status='VERIFIED_INITIALIZATION_NOT_TRAINED', optimization_steps=0,
        views=len(result), points=len(gauss), files=len(records), manifest_sha256=sha256(root/'artifact_manifest.json')))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--output', required=True)
    p.add_argument('--point-cloud', required=True)
    p.add_argument('--sparse', required=True)
    p.add_argument('--ground-truth')
    p.add_argument('--expected-views', type=int, default=200)
    verify(p.parse_args())
