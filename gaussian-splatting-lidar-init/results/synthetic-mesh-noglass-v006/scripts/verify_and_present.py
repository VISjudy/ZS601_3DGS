"""Verify retrieved no-glass outputs, evaluate fixed views, and write delivery docs."""
from pathlib import Path
import csv
import hashlib
import json
import os
import shutil
import sys
os.environ['OPENBLAS_NUM_THREADS']='2'
ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parents[2]
PKG=PROJECT/'experiments/2026-09-21/synthetic-lidar-init-v001/repository/gaussian-splatting-lidar-init'
sys.path[:0]=[str(PROJECT/'.runtime/synthetic-dataset'),str(PKG.parent.parent/'python-deps'),str(PKG)]
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from plyfile import PlyData
from scipy.ndimage import binary_erosion
from colmap_io import read_views
from verify_outputs import psnr,ssim_map

SPEC=json.loads((ROOT/'run_spec.json').read_text())
DEST=Path(SPEC['local_output'])
RET=ROOT/'colab/retrieved'
FULL=RET/'run/full'
GT=Path(SPEC['source_scene']).parents[1]/'virtual_near'
BASE=Path(SPEC['baseline_output'])

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def save(p,data):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(data,f,ensure_ascii=False,indent=2,allow_nan=False)

def read(p):
    with Image.open(p) as im:im.load();return np.array(im)

def copy_new(a,b):
    if b.exists():
        assert sha(a)==sha(b),str(b)
    else:
        b.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(a,b)

cloud_check=json.loads((DEST/'cloud_verification.json').read_text(encoding='utf-8'))
assert cloud_check['status']=='PASS'
for key in ['input_cloud','training_cloud','previous_training_cloud','source_scene']:
    assert sha(SPEC[key])==SPEC[key+'_sha256'],key
status=json.loads((RET/'notebook_status.json').read_text())
assert status['success'] and status['formal_200_authorized']
nbpath=RET/'ZS601_NoGlass_1cm_Scale05_Full200.executed.ipynb'
nb=json.loads(nbpath.read_text())
cells=[c for c in nb['cells'] if c['cell_type']=='code']
assert [c['execution_count'] for c in cells]==list(range(1,len(cells)+1))
assert all(not any(o['output_type']=='error' for o in c.get('outputs',[])) for c in cells)
for row in json.loads((FULL/'artifact_manifest.json').read_text()):
    p=FULL/row['path'];assert p.stat().st_size==row['bytes'] and sha(p)==row['sha256'],row['path']
config=json.loads((FULL/'config.json').read_text())
assert config['view_count']==200 and config['point_count']==SPEC['point_count']
assert config['arguments']['init_scale_factor']==.5 and config['arguments']['view_ids']==''
assert config['optimization_steps']==0
gauss=PlyData.read(FULL/'point_cloud/iteration_0/point_cloud.ply')['vertex'].data
cloud=PlyData.read(SPEC['input_cloud'])['vertex'].data
assert len(gauss)==len(cloud)==SPEC['point_count']
for name in gauss.dtype.names:assert np.isfinite(gauss[name]).all(),name
for name in ['x','y','z']:assert np.array_equal(gauss[name],cloud[name]),name
rgb=np.column_stack([cloud[k] for k in ['red','green','blue']])/255
dc=np.column_stack([gauss[f'f_dc_{i}'] for i in range(3)])
assert np.max(np.abs(dc*.28209479177387814+.5-rgb))<1e-6
opacity=1/(1+np.exp(-gauss['opacity'].astype('f8')))
assert np.all((opacity>.99999)&(opacity<1))
assert np.array_equal(gauss['scale_0'],gauss['scale_1']) and np.array_equal(gauss['scale_0'],gauss['scale_2'])
assert np.all(gauss['rot_0']==1) and all(np.all(gauss[k]==0) for k in ['rot_1','rot_2','rot_3'])
knn=np.load(ROOT/'knn_reference.npz')
reference=np.sqrt(np.maximum(knn['rms_distance_m']**2,1e-7))*.5
observed=np.exp(gauss['scale_0'][knn['point_indices']].astype('f8'))
assert np.allclose(observed,reference,rtol=5e-4,atol=2e-7),np.max(np.abs(observed-reference))
sigma_median_mm=float(np.median(np.exp(gauss['scale_0'].astype('f8')))*1000)
sources={v.image_id:v for v in read_views(GT/'sparse/0')}
views=read_views(FULL/'sparse/0')
assert len(views)==len(sources)==200 and {v.image_id for v in views}==set(sources)
for v in views:
    b=sources[v.image_id]
    assert v.name==b.name and np.array_equal(v.matrices()[0],b.matrices()[0]) and np.array_equal(v.matrices()[1],b.matrices()[1])
formats=dict(images=(8,2),color=(8,2),rgba=(8,6),alpha=(16,0),masks=(8,0),depth=(16,0),depth_mask=(8,0),
             no_coverage_masks=(8,0),low_coverage_masks=(8,0),training_masks_nonempty=(8,0))
for folder in formats:assert {p.name for p in (FULL/folder).glob('*.png')}=={v.name for v in views}
for iid in SPEC['smoke_image_ids']:
    for folder in ['images','color','rgba','alpha','masks','depth','depth_mask']:
        assert np.array_equal(read(FULL/folder/f'{iid:06d}.png'),read(RET/'run/preflight'/folder/f'{iid:06d}.png'))
for p in FULL.rglob('*'):
    if p.is_file():copy_new(p,DEST/'full'/p.relative_to(FULL))
repro=DEST/'reproducibility';repro.mkdir(exist_ok=True)
for p in RET.iterdir():
    if p.is_file():copy_new(p,repro/p.name)
for p in (RET/'run').iterdir():
    if p.is_file():copy_new(p,repro/p.name)
for n in ['run_spec.json','inspect_glass_blender.py','export_noglass_clouds.py','verify_clouds.py','prepare_run.py','verify_and_present.py']:
    copy_new(ROOT/n,repro/n)
print('Identity, all Gaussian fields, 20000 exact 3NN values and all cameras verified',flush=True)
metricdir=DEST/'metrics';metricdir.mkdir(exist_ok=True)
rows=[]
for i,v in enumerate(views):
    arrays={}
    for folder,fmt in formats.items():
        p=DEST/'full'/folder/v.name
        with p.open('rb') as f:assert f.read(26)[24:26]==bytes(fmt)
        a=read(p);assert a.shape[:2]==(1108,640)
        if folder not in ['images','color','rgba','alpha','depth']:assert set(np.unique(a)).issubset({0,255})
        arrays[folder]=a
    alpha=arrays['alpha'];valid=arrays['masks']==255;empty=alpha==0
    assert np.array_equal(arrays['no_coverage_masks']==255,empty)
    assert np.array_equal(arrays['low_coverage_masks']==255,~valid)
    assert np.array_equal(arrays['training_masks_nonempty']==255,~empty)
    assert np.array_equal(arrays['depth']>0,arrays['depth_mask']==255)
    reconstructed=arrays['color'].astype('f4')*(alpha.astype('f4')/65535)[...,None]
    assert np.abs(reconstructed-arrays['images']).max()<=1.6
    gt=read(GT/'images'/v.name).astype('f4')/255
    gm=read(GT/'masks'/v.name)==255
    gtwin=binary_erosion(gm,structure=np.ones((11,11)),border_value=0)
    keep=gm&valid;keepwin=binary_erosion(keep,structure=np.ones((11,11)),border_value=0)
    native=arrays['images'].astype('f4')/255;sm=ssim_map(native,gt)
    gd=read(GT/'depth'/v.name).astype('f8');geom=read(GT/'geometry_masks'/v.name)==255
    dkeep=geom&(gd>0)&(arrays['depth_mask']==255)
    delta=arrays['depth'].astype('f8')-gd
    row=dict(image_id=v.image_id,name=v.name,native_psnr_gt_valid_db=psnr(native,gt,gm),
        native_ssim_gt_valid=float(sm[gtwin].mean()),native_psnr_masked_db=psnr(native,gt,keep),
        native_ssim_masked=float(sm[keepwin].mean()),gap_pixels=int((~valid).sum()),gap_pct=float((~valid).mean()*100),
        strict_alpha_zero_pixels=int(empty.sum()),strict_alpha_zero_pct=float(empty.mean()*100),
        depth_mae_mm=float(np.abs(delta[dkeep]).mean()),depth_rmse_mm=float(np.sqrt(np.mean(delta[dkeep]**2))),
        depth_eval_coverage_pct=float(dkeep.mean()*100),gt_valid_pct=float(gm.mean()*100))
    assert all(np.isfinite(value) for key,value in row.items() if key!='name')
    rows.append(row)
    if i%25==0:print('EVALUATED',i+1,'/200',flush=True)
means={k:float(np.mean([r[k] for r in rows])) for k in rows[0] if k not in ['image_id','name']}
with (metricdir/'per_view_metrics.csv').open('x',newline='',encoding='utf-8-sig') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
save(metricdir/'metrics.json',dict(means=means,per_view=rows,gap_definition='count(masks==0)/all pixels*100; unchanged alpha<0.95',
                                 strict_empty_definition='count(alpha16==0)/all pixels*100',evaluation='Blender virtual_near GT; no image fitting or training'))
oldmetrics=json.loads((BASE/'metrics/metrics.json').read_text())
oldby={r['name']:r for r in oldmetrics['per_view']}
comparison=[]
for r in rows:
    b=oldby[r['name']]
    comparison.append(dict(name=r['name'],old_psnr=b['native_psnr_gt_valid_db'],new_psnr=r['native_psnr_gt_valid_db'],
        delta_psnr=r['native_psnr_gt_valid_db']-b['native_psnr_gt_valid_db'],old_ssim=b['native_ssim_gt_valid'],new_ssim=r['native_ssim_gt_valid'],
        old_gap_pct=b['gap_pct'],new_gap_pct=r['gap_pct']))
with (metricdir/'versus_glass_v005.csv').open('x',newline='',encoding='utf-8-sig') as f:
    w=csv.DictWriter(f,fieldnames=list(comparison[0]));w.writeheader();w.writerows(comparison)
figs=DEST/'comparisons';figs.mkdir(exist_ok=True)
font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',21)
for name in ['003193.png','003301.png','003001.png','003238.png','003478.png']:
    tilew=480;tileh=831;canvas=Image.new('RGB',(tilew*3,tileh+76),(245,245,245));draw=ImageDraw.Draw(canvas)
    for j,(label,p) in enumerate([('Blender GT',GT/'images'/name),('v005: glass sampled',BASE/'full/images'/name),('v006: no glass points',DEST/'full/images'/name)]):
        with Image.open(p) as im:canvas.paste(im.convert('RGB').resize((tilew,tileh),Image.Resampling.LANCZOS),(tilew*j,76))
        draw.text((tilew*j+12,10),label,font=font,fill=(25,25,25))
    c=next(r for r in comparison if r['name']==name)
    draw.text((tilew+12,42),f"PSNR {c['old_psnr']:.2f} dB",font=font,fill=(40,40,40))
    draw.text((2*tilew+12,42),f"PSNR {c['new_psnr']:.2f} dB",font=font,fill=(40,40,40))
    p=figs/(Path(name).stem+'_three_way.png');assert not p.exists();canvas.save(p)
report=dict(status='VERIFIED_COMPLETE_INITIALIZATION_NOT_TRAINED',cloud_verification='PASS',views=200,png_count=2000,
    points=SPEC['point_count'],training_cloud_points=797520,excluded_glass_parts=33,excluded_source_points=176124,
    init_scale_factor=.5,k=3,sigma_median_mm=sigma_median_mm,knn_reference_max_error_m=float(np.max(np.abs(observed-reference))),
    opacity_target=.999999,optimization_steps=0,cameras_exact=True,preflight_repeat_14_pngs_exact=True,
    semantic_image_masks_added=False,mask_logic_unchanged=True,means=means,old_v005_means=oldmetrics['means'],
    psnr_improved_views=sum(r['delta_psnr']>0 for r in comparison),psnr_worse_views=sum(r['delta_psnr']<0 for r in comparison),
    case_003193=next(r for r in comparison if r['name']=='003193.png'),executed_notebook_code_cells=len(cells),
    input_cloud_sha256=SPEC['input_cloud_sha256'],training_cloud_sha256=SPEC['training_cloud_sha256'],
    previous_source_files_preserved=True,remote_archive_sha256=json.loads((ROOT/'colab/download/download_manifest.json').read_text())['sha256'])
save(DEST/'verification.json',report)
save(ROOT/'local_verification.json',report)
oldmean=oldmetrics['means'];case=report['case_003193']
doc=f'''# 去玻璃点云与 200 张虚拟视图 v006

已核验：33 个透明玻璃部件的 176,124 个表面点全部排除；1 cm 点云 7,004,696 点，3 cm 点云 797,520 点。两个文件均为普通彩色 PLY，单位米，含 XYZ、单位法向 nx/ny/nz（float32）及 sRGB red/green/blue（uint8）。

| 用途 | 文件 |
|---|---|
| 生成虚拟图像 | [1 cm 点云](cloud_1cm/points_mesh_1cm_noglass.ply) |
| 后续正式训练初始化 | [3 cm 点云](cloud_3cm/points_mesh_3cm_noglass.ply) |
| 本轮初始化高斯参数 | [Gaussian PLY](full/point_cloud/iteration_0/point_cloud.ply) |

新 3 cm 文件是完整非玻璃 mesh 表面点的体素子集，与旧的训练 RGB-D 融合 3 cm 云来源和点数不同。旧文件保留。本轮没有执行正式训练，也没有自动改写旧数据集的 sparse 初始化文件；训练时须显式选择这里的新 3 cm 文件。

## 玻璃识别与采样

冻结源场景中的 `ZS601_Semantic_CabinetGlass` 具有真实自定义属性 `semantic_class=clear_glass`，Principled Transmission Weight=1、IOR≈1.45。根据实际材质槽逐三角面映射，排除 16 个柜门玻璃、16 个玻璃搁板和 1 个门窗玻璃部件。半透塑料、电视屏幕、控制面板和发光涂层仍保留，未仅凭名字含 Glass 排除。

1 cm 使用上一轮确定性 mesh 采样中的非玻璃点，全部保留点的坐标、颜色、法向逐值一致；3 cm 在同一非玻璃集合中按每部件世界坐标体素选择最接近体素中心的真实表面代表点。3 cm 是 1 cm 的子集。体素尺寸不是严格点间最小距离。此为面采样近似，不模拟 LiDAR 扫描遮挡、回波和材质响应。

审计见 [材质判定](audit/semantic_decision.json)、[实际 Blender 材质](audit/material_audit.json) 和 [全点独立检查](cloud_verification.json)。`audit/glass_face_mask.npy` 是初步宽泛透射候选；最终排除依据为 `excluded_clear_glass_faces.npy`。这些是三维导出审计数组，没有新增图像语义 mask。

## 200 张虚拟视图

沿用原 virtual_near 的 200 个相机、640×1108 分辨率、k=3、scale=0.5、opacity=0.999999、SH0；优化步数为 0。移除玻璃后重新计算剩余点的 3NN 尺度，sigma 中位数 {sigma_median_mm:.6f} mm。20,000 个独立精确 3NN 参考检查通过。

| 全 200 张指标 | 原 v005 | 去玻璃 v006 |
|---|---:|---:|
| Black RGB PSNR / dB | {oldmean['native_psnr_gt_valid_db']:.6f} | {means['native_psnr_gt_valid_db']:.6f} |
| SSIM | {oldmean['native_ssim_gt_valid']:.6f} | {means['native_ssim_gt_valid']:.6f} |
| 原有效 mask 的无效像素比例 / % | {oldmean['gap_pct']:.6f} | {means['gap_pct']:.6f} |
| depth MAE / mm | {oldmean['depth_mae_mm']:.6f} | {means['depth_mae_mm']:.6f} |

003193 玻璃柜案例 PSNR：{case['old_psnr']:.6f} → {case['new_psnr']:.6f} dB。完整逐视角数据见 [CSV](metrics/per_view_metrics.csv) 与 [配对对比](metrics/versus_glass_v005.csv)。PSNR 使用同一 GT 有效区域，SSIM 使用同一有效窗口；深度排除原 GT 无效/透明几何，仅为被评估不透明区域精度。

![玻璃柜对照](comparisons/003193_three_way.png)

## 图像格式及 mask

所有图像文件名与 COLMAP `full/sparse/0/images.txt` 对应。相机 txt 与原数据一致；`full/sparse/0/points3D.ply`、`points3D.txt` 记录本轮 1 cm 投影输入，正式训练初始化使用上方独立 3 cm 文件。

| full/ 子目录 | 格式 | 说明 |
|---|---|---|
| images | uint8 RGB | 黑底合成 RGB，虚拟训练图 |
| color | uint8 RGB | 除以累计 alpha 的 Straight RGB |
| rgba | uint8 RGBA | Straight RGB 与 alpha |
| alpha | uint16 单通道 | 累计 alpha=值/65535 |
| masks | uint8 单通道 0/255 | 白色有效，原浮点 alpha≥0.95 |
| low_coverage_masks | uint8 单通道 0/255 | masks 反相，白色为覆盖不足 |
| no_coverage_masks | uint8 单通道 0/255 | 白色仅表示 alpha16=0 |
| training_masks_nonempty | uint8 单通道 0/255 | 只忽略 alpha16=0 的替代有效 mask |
| depth | uint16 单通道 | camera-Z 毫米，0 无效；alpha 归一化的调和深度 |
| depth_mask | uint8 单通道 0/255 | 白色表示正深度且 alpha≥0.5 |

按用户定义，颗粒感/间隙比例为 `count(masks==0)/总像素数`；严格 alpha16=0 比例另列。原 mask 逻辑保留，没有玻璃语义 mask。PLY 附带 mesh 法向，但各向同性初始化高斯不提供确定表面法向图；同相机 Blender normal 真值仍在原 `synthetic-training-v001/virtual_near/normal/`。

## 限制与复现

本轮只改变采样是否包含玻璃，没有重新拟合颜色。非玻璃点沿用原训练视图取色及未观测表面的材质底色回退；柜内遮挡区域仍可能缺少纹理、光照和反射。移除玻璃不会重现玻璃的物理反射与折射。以上是初始化投影与数据检查，不是 3DGS 训练收敛结果。

两种点云回读、全部表面点距离/体素唯一性/来源、200 个相机、2000 张 PNG、Gaussian PLY、14 张预检重复图、notebook 的全部 {len(cells)} 个代码单元均已检查。源 .blend、旧点云与旧结果保留。实际执行 notebook、代码与日志见 `reproducibility/`。
'''
with (DEST/'README.md').open('x',encoding='utf-8') as f:f.write(doc)
print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)
