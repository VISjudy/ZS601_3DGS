"""Run via `colab exec -f` after uploading the three archives/manifest.

Environment: ZS601_RUN_ROOT (new remote directory containing uploaded files).
Required: input_bundle.zip, source_bundle.tar.gz, source_manifest.json.
The delivered source archive and manifest identify the actual tested commit.
"""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
import tarfile
import time
import zipfile

root = Path(os.environ.get('ZS601_RUN_ROOT', '/content/zs601-lidar-reproduce'))
manifest = json.loads((root/'source_manifest.json').read_text())
archive = root/'source_bundle.tar.gz'
assert hashlib.sha256(archive.read_bytes()).hexdigest() == manifest['archive_sha256']
for target in ['source', 'input', 'run-001', 'worker.py', 'launch.json']:
    if (root/target).exists():
        raise FileExistsError('Use a new run directory: ' + str(root/target))
(root/'source').mkdir()
with tarfile.open(archive) as z:
    z.extractall(root/'source', filter='data')
(root/'input').mkdir()
with zipfile.ZipFile(root/'input_bundle.zip') as z:
    for n in z.namelist():
        if not (root/'input'/n).resolve().is_relative_to((root/'input').resolve()):
            raise ValueError('Unsafe input archive member')
    z.extractall(root/'input')
os.environ.update(ZS601_RUN_ROOT=str(root), ZS601_CODE_COMMIT=manifest['code_commit'], ZS601_ATTEMPT='run-001')
worker = '''from pathlib import Path
import json, os, time, traceback, zipfile, hashlib
import nbformat
from nbclient import NotebookClient
root=Path(os.environ['ZS601_RUN_ROOT'])
pkg=root/'source/gaussian-splatting-lidar-init'
nb=nbformat.read(pkg/'notebooks/ZS601_LiDAR_Init_Colab.ipynb',as_version=4)
executed=root/'ZS601_LiDAR_Init_Colab.executed.ipynb'
if executed.exists(): raise FileExistsError(executed)
error=None
try:
    NotebookClient(nb,timeout=None,kernel_name='python3',resources={'metadata':{'path':str(pkg)}}).execute()
except Exception:
    error=traceback.format_exc()
finally:
    nbformat.write(nb,executed)
status=dict(success=error is None,error=error,finished_unix=time.time())
with (root/'notebook_status.json').open('x') as f: json.dump(status,f,indent=2)
with zipfile.ZipFile(root/'artifacts.zip','x',compression=zipfile.ZIP_STORED) as z:
    for p in sorted((root/'run-001').rglob('*')):
        if p.is_file(): z.write(p,p.relative_to(root).as_posix())
    for n in ['ZS601_LiDAR_Init_Colab.executed.ipynb','notebook_status.json','source_manifest.json']:
        z.write(root/n,n)
artifact=root/'artifacts.zip'
with (root/'artifacts_ready.json').open('x') as f:
    json.dump(dict(bytes=artifact.stat().st_size,sha256=hashlib.sha256(artifact.read_bytes()).hexdigest()),f,indent=2)
print(json.dumps(status),flush=True)
'''
with (root/'worker.py').open('x') as f:
    f.write(worker)
log = (root/'worker.log').open('x')
proc = subprocess.Popen([sys.executable, '-u', str(root/'worker.py')], cwd=root,
    env=os.environ.copy(), stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
launch = dict(pid=proc.pid, source_commit=manifest['code_commit'], started_unix=time.time(),
              command_sha256=hashlib.sha256(worker.encode()).hexdigest())
with (root/'launch.json').open('x') as f:
    json.dump(launch,f,indent=2)
print(json.dumps(launch))
