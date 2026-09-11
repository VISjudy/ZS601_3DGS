"""Explicit text camera lists: no binary fallback and no source mutation."""
from pathlib import Path
import hashlib
import numpy as np

def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()

def load_data(a):
    from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text
    from scene.dataset_readers import readColmapCameras,getNerfppNorm
    from utils.camera_utils import cameraList_from_camInfos
    from utils.graphics_utils import BasicPointCloud
    intr=read_intrinsics_text(a.cameras_file)
    train=read_extrinsics_text(a.train_file); val=read_extrinsics_text(a.val_file)
    test=read_extrinsics_text(a.test_file) if a.test_file else {}
    train_names=[e.name for e in train.values()]; val_names=[e.name for e in val.values()]
    if not train or len(val)!=10 or len(set(val_names))!=10:
        raise ValueError('Need nonempty train set and exactly ten unique fixed val cameras')
    if len(set(train_names))!=len(train_names): raise ValueError('Duplicate training image names')
    if set(train_names)&set(val_names): raise ValueError('Train/val overlap: fix explicit input lists')
    test_names=[e.name for e in test.values()]
    if len(set(test_names))!=len(test_names): raise ValueError('Duplicate test image names')
    if set(train_names)&set(test_names): raise ValueError('Train/test overlap')
    if set(val_names)&set(test_names): raise ValueError('Val/test overlap')
    # Baseline renderer uses a centered projection; reject unsupported calibration silently lost before.
    for c in intr.values():
        if c.model=='PINHOLE': cx,cy=c.params[2:4]
        elif c.model=='SIMPLE_PINHOLE': cx,cy=c.params[1:3]
        else: raise ValueError('Only undistorted PINHOLE/SIMPLE_PINHOLE supported')
        if abs(cx-c.width/2)>.51 or abs(cy-c.height/2)>.51:
            raise ValueError('Off-center principal point needs a custom projection matrix')
    root=Path(a.source_path)
    def infos(extr):
        for e in extr.values():
            image=root/a.images/e.name
            mask=root/a.alpha_masks/Path(e.name).with_suffix('.png')
            if not image.is_file() or (a.alpha_masks and not mask.is_file()):
                raise FileNotFoundError(str(image)+' / '+str(mask))
        return readColmapCameras(extr,intr,None,str(root/a.images),
                str(root/a.alpha_masks) if a.alpha_masks else '', '', [])
    ti,vi,xi=infos(train),infos(val),infos(test)
    p=Path(a.point_cloud)
    if p.suffix.lower()=='.las':
        import laspy
        las=laspy.read(p); xyz=np.asarray(las.xyz)
        color=np.stack([las.red,las.green,las.blue],1)
        if color.max()>255: color=color>>8
        color=np.clip(color,0,255).astype('f4')/255
    elif p.suffix.lower()=='.ply':
        from plyfile import PlyData
        v=PlyData.read(p)['vertex']
        xyz=np.stack([v[n] for n in ('x','y','z')],1)
        color=np.stack([v[n] for n in ('red','green','blue')],1).astype('f4')/255
    else: raise ValueError('Expected LAS or RGB PLY')
    if not np.isfinite(xyz).all() or not np.isfinite(color).all(): raise ValueError('Nonfinite cloud')
    centers=np.stack([-c.R@c.T for c in ti])
    identity={k:sha256(getattr(a,k)) for k in ('point_cloud','train_file','val_file','cameras_file')}
    if a.test_file: identity['test_file']=sha256(a.test_file)
    # Content identity survives re-extraction on a new Colab runtime (mtime does not).
    metadata=[]
    for c in ti+vi+xi:
        for f in (c.image_path,c.mask_path):
            if f:
                metadata.append((str(Path(f).relative_to(root)),sha256(f)))
    identity['image_content_sha256']=hashlib.sha256(repr(metadata).encode()).hexdigest()
    from scene.cameras import set_lazy_cache_size
    set_lazy_cache_size(a.lazy_cache)
    cams=cameraList_from_camInfos(ti,1.,a,False,False)
    vals=cameraList_from_camInfos(vi,1.,a,False,True)
    tests=cameraList_from_camInfos(xi,1.,a,False,True)
    print(f'[DATA] explicit TEXT train={len(ti)} val={len(vi)} test={len(xi)} overlap=0 points={len(xyz)} units={a.units}')
    return BasicPointCloud(xyz.astype('f4'),color,np.zeros_like(xyz,dtype='f4')),ti,cams,vals,tests,centers,max(float(getNerfppNorm(ti)['radius']),1e-3),identity
