"""Create immutable new Colab run from the verified renderer and glass-free cloud."""
from pathlib import Path
import datetime
import hashlib
import json
import shutil
import zipfile

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[2]
BASE = PROJECT / 'experiments/2026-09-21/synthetic-mesh1cm-scale05-full-v005'
DATA = PROJECT / 'scenes/zs601-meetingroom/synthetic-training-v001'
OUT = PROJECT / '3dgsResult/zs601-mesh-noglass-v006'
OLD = json.loads((BASE/'run_spec.json').read_text())
CLOUDS = json.loads((OUT/'cloud_manifest.json').read_text(encoding='utf-8'))

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def save(p,d):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(d,f,indent=2,ensure_ascii=False)

for n in ['colab','inputs','logs']:(ROOT/n).mkdir(exist_ok=False)
spec={**OLD,'run_id':'zs601-mesh-noglass-v006','parent_run':OLD['run_id'],
      'scope':'Known clear-glass surfaces excluded; new 1cm virtual-render and 3cm formal-training initialization clouds; unchanged 200 virtual poses and renderer; no training',
      'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
      'source_scene':str(DATA/'source/scene.blend'),'source_scene_sha256':CLOUDS['source_scene_sha256'],
      'input_cloud':str(OUT/CLOUDS['outputs']['1cm']['path']),
      'input_cloud_sha256':CLOUDS['outputs']['1cm']['files']['points_mesh_1cm_noglass.ply']['sha256'],
      'training_cloud':str(OUT/CLOUDS['outputs']['3cm']['path']),
      'training_cloud_sha256':CLOUDS['outputs']['3cm']['files']['points_mesh_3cm_noglass.ply']['sha256'],
      'previous_training_cloud':OLD['training_cloud'],'previous_training_cloud_sha256':OLD['training_cloud_sha256'],
      'point_count':CLOUDS['outputs']['1cm']['points'],'smoke_image_ids':[3193,3301],
      'colab_session':'zs601-noglass-0922-v006','remote_root':'/content/zs601-mesh-noglass-v006',
      'local_output':str(OUT),'baseline_output':OLD['local_output'],
      'gt_evaluation':'Local frozen Blender virtual_near RGB/depth after rendering; no GT uploaded or optimized',
      'smoke_gate':'Two-view structural/finite output preflight; unchanged already validated renderer. Full200 user-authorized.',
      'release_gate':'Both clouds independently verified; 200 cameras/2000 PNGs/finite Gaussian PLY; local metrics and visual comparison; actual notebook; release GPU',
      'semantic_image_masks_added':False,'color_policy':'Preserve prior train-only colors on non-glass sample points',
      'sampling_policy':'1cm retained exact non-glass surface samples;3cm per-component voxel subset'}
save(ROOT/'run_spec.json',spec)
stage=ROOT/'inputs/stage';(stage/'sparse/0').mkdir(parents=True)
shutil.copyfile(spec['input_cloud'],stage/'points_mesh_1cm_noglass.ply')
for n in ['cameras.txt','images.txt']:shutil.copyfile(DATA/'virtual_near/sparse/0'/n,stage/'sparse/0'/n)
plan=json.loads((DATA/'plan.json').read_text())
ids=sorted(r['id'] for r in plan['views'] if r['subset']=='virtual_near')
assert len(ids)==200
files=[dict(path=p.relative_to(stage).as_posix(),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(stage.rglob('*')) if p.is_file()]
save(stage/'input_manifest.json',dict(files=files,view_ids=ids,units='metres',points=spec['point_count'],
    source_scene_sha256=spec['source_scene_sha256'],glass_excluded=True,train_only_color=True,gt_images_uploaded=0))
archive=ROOT/'inputs/input_bundle.zip'
with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as z:
    for p in sorted(stage.rglob('*')):
        if p.is_file():z.write(p,p.relative_to(stage).as_posix())
partdir=ROOT/'inputs/upload_parts';partdir.mkdir()
parts=[]
with archive.open('rb') as f:
    for i,b in enumerate(iter(lambda:f.read(8*1024*1024),b'')):
        p=partdir/f'input.part{i:03d}'
        with p.open('xb') as o:o.write(b)
        parts.append(dict(name=p.name,bytes=len(b),sha256=sha(p)))
save(ROOT/'inputs/input_parts.json',dict(bytes=archive.stat().st_size,sha256=sha(archive),parts=parts))
nb=json.loads((BASE/'colab/ZS601_Mesh1cm_Scale05_Full200.ipynb').read_text())
for i,c in enumerate(nb['cells']):
    s=''.join(c['source']).replace(OLD['remote_root'],spec['remote_root']).replace('points_mesh_1cm.ply','points_mesh_1cm_noglass.ply')
    if c['cell_type']=='markdown':
        s='# ZS601 clear-glass-free mesh initialization, 200 virtual views\n\nKnown Blender clear_glass faces are excluded.1cm cloud:7,004,696 points;3cm training cloud:797,520 points.Original200COLMAP poses,k=3,scale0.5,opacity0.999999,SH0,zero optimization steps.Original coverage masks retained;no semantic image mask.\n'
    else:
        c['execution_count']=None;c['outputs']=[]
        s=s.replace("assert inputs['view_ids']==[3001,3238,3301,3478]","assert len(inputs['view_ids'])==200 and inputs['points']==7004696\nassert sha(INPUT/'points_mesh_1cm_noglass.ply')=='"+spec['input_cloud_sha256']+"'")
        s=s.replace('Source and smoke inputs verified.','Immutable renderer and glass-free input verified.')
        compile(s,'<notebook>','exec')
    c['source']=s.splitlines(True);c['id']=f'noglass-v006-{i:02d}'
smoke="""preflight=RUN/'preflight'
run([sys.executable,'render_from_sparse_v4.py','--point-cloud',INPUT/'points_mesh_1cm_noglass.ply',
     '--sparse',INPUT/'sparse/0','--output',preflight,'--opacity','0.999999',
     '--init-scale-factor','0.5','--view-ids','3193,3301'],'preflight_render.log')
run([sys.executable,'verify_outputs.py','--point-cloud',INPUT/'points_mesh_1cm_noglass.ply',
     '--sparse',INPUT/'sparse/0','--output',preflight,'--expected-views','2'],'preflight_verify.log')
assert json.loads((preflight/'verification.json').read_text())['views']==2
save('PREFLIGHT_COMPLETE.json',dict(views=2,structural_checks_passed=True,visual_evaluation_pending=True))
"""
compile(smoke,'<preflight>','exec')
nb['cells'].insert(4,dict(cell_type='code',id='noglass-preflight',metadata={},source=smoke.splitlines(True),execution_count=None,outputs=[]))
fullcell=nb['cells'][5]
fullcell['source'] += ["assert sha(preflight/'point_cloud/iteration_0/point_cloud.ply')==sha(full/'point_cloud/iteration_0/point_cloud.ply')\n"]
name='ZS601_NoGlass_1cm_Scale05_Full200.ipynb'
save(ROOT/'colab'/name,nb)
for n in ['source_bundle.tar.gz','source_manifest.json']:shutil.copyfile(BASE/'colab'/n,ROOT/'colab'/n)
for n in ['colab_call.py','upload_files.py']:shutil.copyfile(BASE/n,ROOT/n)
for n in ['runtime_probe.py','progress.py','inspect_before_launch.py']:
    s=(BASE/'colab'/n).read_text().replace(OLD['remote_root'],spec['remote_root'])
    with (ROOT/'colab'/n).open('x') as f:f.write(s)
s=(BASE/'colab/start_full.py').read_text().replace(OLD['remote_root'],spec['remote_root']).replace('ZS601_Mesh1cm_Scale05_Full200','ZS601_NoGlass_1cm_Scale05_Full200')
s=s.replace("if p.is_file():z.write(p,p.relative_to(root).as_posix())", "if p.is_file() and not any(p.is_relative_to(root/q) for q in ['run/preflight/point_cloud','run/preflight/sparse']):z.write(p,p.relative_to(root).as_posix())")
compile(s,'<worker-launch>','exec')
with (ROOT/'colab/start_full.py').open('x') as f:f.write(s)
s=(BASE/'download_full.py').read_text().replace(OLD['remote_root'],spec['remote_root']).replace(OLD['colab_session'],spec['colab_session'])
with (ROOT/'download_full.py').open('x') as f:f.write(s)
oldrows=json.loads((BASE/'inputs/upload_manifest.json').read_text())
wheels=next(Path(r['local']) for r in oldrows if Path(r['local']).name=='wheels.zip')
save(ROOT/'inputs/upload_identity.json',dict(input_bytes=archive.stat().st_size,input_sha256=sha(archive),notebook_sha256=sha(ROOT/'colab'/name),wheels_sha256=sha(wheels)))
upload=[ROOT/'colab/source_bundle.tar.gz',ROOT/'colab/source_manifest.json',ROOT/'colab'/name,ROOT/'inputs/upload_identity.json',ROOT/'inputs/input_parts.json',wheels]+sorted(partdir.iterdir())
rows=[dict(local=str(p),remote=spec['remote_root']+'/'+p.name,bytes=p.stat().st_size,sha256=sha(p)) for p in upload]
save(ROOT/'inputs/upload_manifest.json',rows)
save(ROOT/'phase_planned.done.json',dict(state='planned',session=spec['colab_session'],views=200,notebook_sha256=sha(ROOT/'colab'/name)))
save(ROOT/'phase_cli_auth_verified.done.json',dict(state='auth_verified',cli_version='0.7.0',active_sessions_before_allocation=0))
print(json.dumps(dict(prepared=True,upload_files=len(rows),archive_bytes=archive.stat().st_size,views=200)))
