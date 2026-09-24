"""Rebuild the exact vendored CUDA extensions, never fetch a floating upstream."""
import os,sys,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
subprocess.run([sys.executable,'-m','pip','install','torch==2.11.0','torchvision==0.26.0','--index-url','https://download.pytorch.org/whl/cu128'],check=True)
subprocess.run([sys.executable,'-m','pip','install','-r',str(ROOT/'requirements-colab.txt')],check=True)
import torch
assert torch.cuda.is_available(),'Select a Colab GPU runtime first'
major,minor=torch.cuda.get_device_capability();os.environ['TORCH_CUDA_ARCH_LIST']=f'{major}.{minor}';os.environ['MAX_JOBS']='2'
for name in ['diff-gaussian-rasterization','simple-knn']:
    subprocess.run([sys.executable,'-m','pip','install','--no-build-isolation',str(ROOT/'submodules'/name)],check=True)
print(dict(python=sys.version,torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name()))
