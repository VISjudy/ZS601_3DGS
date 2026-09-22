"""Prepare matched ten-view depth experiment; no changes to earlier results."""
from pathlib import Path
from dataclasses import asdict
import hashlib
import json
import sys
ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parents[2]
sys.path.insert(0,str(PROJECT/'.runtime/synthetic-dataset'))
sys.path.insert(0,str(PROJECT/'experiments/2026-09-21/synthetic-lidar-init-v001/repository/gaussian-splatting-lidar-init'))
import numpy as np
from colmap_io import read_views,write_sparse
DATA=PROJECT/'scenes/zs601-meetingroom/synthetic-training-v001'
PRIOR=PROJECT/'3dgsResult/zs601-mesh-noglass-v006'
OUT=PROJECT/'3dgsResult/zs601-depth-methods-v007'
OUT.mkdir(exist_ok=False)
for n in ['colab','logs']:(ROOT/n).mkdir(exist_ok=False)

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def save(p,d):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(d,f,indent=2,ensure_ascii=False)

views=read_views(DATA/'virtual_near/sparse/0')
seed=20260922
indices=sorted(np.random.default_rng(seed).choice(len(views),10,replace=False).tolist())
selected=[views[i] for i in indices]
write_sparse(OUT/'sparse/0',selected,np.empty((0,3)),np.empty((0,3),dtype='u1'))
save(OUT/'selected_views.json',[asdict(v) for v in selected])
cloud=PRIOR/'cloud_1cm/points_mesh_1cm_noglass.ply'
gaussian=PRIOR/'full/point_cloud/iteration_0/point_cloud.ply'
spec=dict(run_id='zs601-depth-methods-v007',random_seed=seed,selection='10 without replacement from fixed 200 virtual_near cameras; no quality-based selection',
          view_ids=[v.image_id for v in selected],views=10,resolution=[640,1108],input_cloud=str(cloud),input_cloud_sha256=sha(cloud),
          source_gaussian=str(gaussian),source_gaussian_sha256=sha(gaussian),source_run='zs601-mesh-noglass-v006',
          source_geometry=str(DATA/'source/geometry.npz'),excluded_glass_faces=str(PRIOR/'audit/excluded_clear_glass_faces.npy'),
          source_scene_sha256=sha(DATA/'source/scene.blend'),local_output=str(OUT),
          comparison_input='same noglass synthetic1cm cloud for both methods; optional clarification offered to user; chosen to isolate generation method',
          method_a='Graphdeco isotropic SH0 3NN scale0.5, nominal opacity1, alpha-normalized harmonic camera-Z, savedPNG then unproject',
          method_b='Pure point projection and nearest camera-Z per occupied pixel; no splat radius, hole fill, interpolation or mesh-aided occlusion removal',
          opacity_target=1.,opacity_finite_logit=20.,opacity_runtime_dtype='float32',kernel_fragment_alpha_cap=.99,
          near_clip_m=.2,far_clip_m=65.535,depth_png='uint16 single-channel camera-Z millimetres;0 invalid',
          depth_mask_a_alpha_min=.5,rgb_coverage_mask_a_alpha_min=.95,occlusion_error_threshold_m=.03,
          pixel_centres='COLMAP top-left pixel centre (0.5,0.5)',
          evaluation='Point-to-original-cloud NN plus same-mask depth and foreground-leakage errors against independent no-glass mesh first hits; no mesh used by either rendering method',
          colab_session='zs601-noglass-0922-v006',session_reused_for_authorized_followup=True,
          remote_root='/content/zs601-depth-methods-v007',remote_source_root='/content/zs601-mesh-noglass-v006',
          renderer_commit='4c7186e363f14050c4977bb192f12ed237112764',optimization_steps=0)
save(ROOT/'run_spec.json',spec)
save(OUT/'run_spec.json',spec)
print(json.dumps(dict(prepared=True,view_ids=spec['view_ids'],input_cloud_sha256=spec['input_cloud_sha256'])))
