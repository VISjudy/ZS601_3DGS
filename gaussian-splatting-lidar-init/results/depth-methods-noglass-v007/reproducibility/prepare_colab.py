"""Create a small, inspectable notebook and one-shot CLI worker launcher."""
from pathlib import Path
import json

ROOT=Path(__file__).resolve().parent
R='/content/zs601-depth-methods-v007'
OLD='/content/zs601-mesh-noglass-v006'
def code(s):
    return dict(cell_type='code',execution_count=None,metadata={},outputs=[],source=s.splitlines(keepends=True))
nb=dict(nbformat=4,nbformat_minor=5,metadata=dict(kernelspec=dict(name='python3',display_name='Python 3',language='python')),
    cells=[dict(cell_type='markdown',metadata={},source=[
        '# ZS601: ten-view no-glass depth comparison\n',
        'Method A: verified 1 cm point-cloud initialization, k=3 RMS × 0.5, nominal opacity 1, no training.\n',
        'Reuses the authenticated v006 Colab runtime and pinned CUDA renderer. Exact input hashes are checked.\n',
        'The companion CPU z-buffer and PNG backprojection evaluator run locally from the same ten cameras.\n',
        'Saved depth: uint16 PNG, single-channel camera Z in millimetres, 0 invalid.\n']),
    code(f"""from pathlib import Path
import json,sys,subprocess,torch
ROOT=Path('{R}')
OLD=Path('{OLD}')
assert torch.cuda.is_available()
print(dict(torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),
           sigmoid20_float32=float(torch.sigmoid(torch.tensor(20.,device='cuda')).cpu())))
print(json.loads((ROOT/'run_spec.json').read_text()))
"""),
    code("""subprocess.run([sys.executable,'-u',str(ROOT/'run_gaussian_depth.py'),
 '--package',str(OLD/'source/gaussian-splatting-lidar-init'),
 '--source-gaussian',str(OLD/'run/full/point_cloud/iteration_0/point_cloud.ply'),
 '--input-cloud',str(OLD/'input/points_mesh_1cm_noglass.ply'),
 '--spec',str(ROOT/'run_spec.json'),'--views',str(ROOT/'selected_views.json'),
 '--output',str(ROOT/'method_a_gaussian')],check=True)
"""),
    code("""from PIL import Image
import numpy as np
out=ROOT/'method_a_gaussian'
views=json.loads((ROOT/'selected_views.json').read_text())
assert len(views)==10
for folder in ['images','alpha','masks','depth','depth_mask']:
    assert len(list((out/folder).glob('*.png')))==10
for v in views:
    d=np.array(Image.open(out/'depth'/v['name']))
    m=np.array(Image.open(out/'depth_mask'/v['name']))>0
    assert d.dtype==np.uint16 and np.array_equal(d>0,m)
    assert d.shape==(v['height'],v['width'])
print((out/'initialization.json').read_text())
print((out/'render_summary.json').read_text())
print('TEN_VIEW_GAUSSIAN_DEPTH_PASS_NO_TRAINING')
""")])
with (ROOT/'colab/ZS601_NoGlass_Depth_Methods_Smoke10.ipynb').open('x',encoding='utf-8') as f:json.dump(nb,f,indent=1)

worker=f'''from pathlib import Path
import json,traceback,time,hashlib,zipfile
import nbformat
from nbclient import NotebookClient
root=Path({R!r})
nb=nbformat.read(root/'ZS601_NoGlass_Depth_Methods_Smoke10.ipynb',as_version=4)
error=None
try:
    NotebookClient(nb,timeout=None,kernel_name='python3',resources={{'metadata':{{'path':str(root)}}}}).execute()
except Exception:
    error=traceback.format_exc();print(error,flush=True)
finally:
    executed=root/'ZS601_NoGlass_Depth_Methods_Smoke10.executed.ipynb'
    assert not executed.exists()
    nbformat.write(nb,executed)
status=dict(success=error is None,error=error,finished_unix=time.time(),views=10,optimization_steps=0)
with (root/'notebook_status.json').open('x') as f:json.dump(status,f,indent=2)
with zipfile.ZipFile(root/'depth_artifacts.zip','x',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as z:
    for p in sorted((root/'method_a_gaussian').rglob('*')):
        if p.is_file():z.write(p,p.relative_to(root).as_posix())
    for n in ['ZS601_NoGlass_Depth_Methods_Smoke10.executed.ipynb','notebook_status.json','launch.json','run_spec.json','selected_views.json','run_gaussian_depth.py','upload_identity.json']:
        z.write(root/n,n)
archive=root/'depth_artifacts.zip';parts=root/'download_parts';parts.mkdir()
records=[]
with archive.open('rb') as f:
    for i,data in enumerate(iter(lambda:f.read(8*1024*1024),b'')):
        name=f'depth.part{{i:03d}}'
        with (parts/name).open('xb') as out:out.write(data)
        records.append(dict(name=name,bytes=len(data),sha256=hashlib.sha256(data).hexdigest()))
ready=dict(bytes=archive.stat().st_size,sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),parts=records)
with (root/'download_manifest.json').open('x') as f:json.dump(ready,f,indent=2)
print(json.dumps(status),flush=True)
'''
launcher=f'''from pathlib import Path
import hashlib,json,subprocess,sys,time
root=Path({R!r})
identity=json.loads((root/'upload_identity.json').read_text())
for name,digest in identity.items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,name
for name in ['worker.py','launch.json','method_a_gaussian','notebook_status.json','depth_artifacts.zip']:
    if (root/name).exists():raise FileExistsError(root/name)
worker={worker!r}
with (root/'worker.py').open('x') as f:f.write(worker)
log=(root/'worker.log').open('x')
p=subprocess.Popen([sys.executable,'-u',str(root/'worker.py')],cwd=root,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
record=dict(pid=p.pid,started_unix=time.time(),worker_sha256=hashlib.sha256(worker.encode()).hexdigest(),expected_views=10,optimization_steps=0)
with (root/'launch.json').open('x') as f:json.dump(record,f,indent=2)
print(json.dumps(record))
'''
with (ROOT/'colab/start_depth.py').open('x',encoding='utf-8') as f:f.write(launcher)
print('PREPARED_NOTEBOOK_AND_LAUNCHER')
