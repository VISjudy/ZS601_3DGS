#!/usr/bin/env python3
"""Run one isolated Colab experiment: ZS601 LiDAR-B or official DTU scan105.
Outputs are durable on Drive; source/config live in GitHub.
"""
import argparse, json, os, re, shutil, subprocess, sys, tarfile, threading, time, zipfile
from datetime import datetime
from pathlib import Path
import numpy as np

PARSER=argparse.ArgumentParser()
PARSER.add_argument('--mode',required=True,choices=['b','c','official'])
PARSER.add_argument('--drive-root',default='/content/drive/MyDrive/LCCDataset/zs601_output')
ARGS=PARSER.parse_args()
DRIVE=Path(ARGS.drive_root)
STAMP=datetime.now().strftime('%Y%m%d_%H%M%S')
prefix={'b':'zs601_2dgs_B_lidar_parallel_','c':'zs601_2dgs_C_lidar_distortion_parallel_','official':'official_2dgs_DTU_scan105_parallel_'}[ARGS.mode]
OUT=DRIVE/(prefix+STAMP)
OUT.mkdir(parents=True,exist_ok=True)
STATUS=OUT/'status.json'

def state(stage,**kw):
    data={'mode':ARGS.mode,'stage':stage,'updated':datetime.now().isoformat(),**kw}
    STATUS.write_text(json.dumps(data,indent=2,ensure_ascii=False)); print('STATUS',json.dumps(data,ensure_ascii=False),flush=True)

def run(cmd,cwd=None,log='run.log',check=True):
    cmd=[str(x) for x in cmd]; print('RUN',' '.join(cmd),flush=True)
    with (OUT/log).open('a',encoding='utf-8') as f:
        f.write(chr(10)+'$ '+' '.join(cmd)+chr(10)); f.flush()
        p=subprocess.Popen(cmd,cwd=cwd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
        for line in p.stdout:
            print(line,end='',flush=True); f.write(line); f.flush()
        rc=p.wait()
    if check and rc: raise subprocess.CalledProcessError(rc,cmd)
    return rc

os.environ.setdefault('TORCH_CUDA_ARCH_LIST', '7.5')

def pip_install(repo=None):
    run([sys.executable,'-m','pip','install','-q','plyfile','laspy[lazrs]','lpips','open3d','trimesh','scikit-image','gdown'],log='install.log')
    if repo:
        for sub in ['submodules/diff-surfel-rasterization','submodules/simple-knn']:
            run([sys.executable,'-m','pip','install','-q',str(repo/sub)],log='install.log')

def patch_cuda(repo):
    for rel,token,inc in [('submodules/diff-surfel-rasterization/cuda_rasterizer/rasterizer_impl.h','#include <cuda_runtime.h>','#include <cstdint>'),('submodules/simple-knn/simple_knn.cu','#include <cuda_runtime.h>','#include <cfloat>')]:
        p=repo/rel
        if p.exists():
            s=p.read_text()
            if inc not in s:
                p.write_text(s.replace(token,token+chr(10)+inc) if token in s else inc+chr(10)+s)

def gpu_name():
    return subprocess.run(['nvidia-smi','--query-gpu=name,memory.total','--format=csv,noheader'],capture_output=True,text=True).stdout.strip()

def monitor_run(cmd,cwd,log):
    peak=0
    with (OUT/log).open('a',encoding='utf-8') as f:
        p=subprocess.Popen([str(x) for x in cmd],cwd=cwd,stdout=f,stderr=subprocess.STDOUT,text=True)
        while p.poll() is None:
            q=subprocess.run(['nvidia-smi','--query-compute-apps=used_memory','--format=csv,noheader,nounits'],capture_output=True,text=True).stdout
            vals=[int(x.strip()) for x in q.splitlines() if x.strip().isdigit()]
            peak=max([peak]+vals); time.sleep(2)
        rc=p.returncode
    print('RC',rc,'PEAK_MIB',peak,flush=True)
    if rc: raise subprocess.CalledProcessError(rc,cmd)
    return peak

def find_dataset(base):
    cams=list(base.rglob('cameras.bin'))
    for cam in cams:
        root=cam.parent.parent if cam.parent.name=='0' else cam.parent
        if (root/'images').is_dir(): return root
    raise FileNotFoundError('No COLMAP dataset with images and cameras.bin')

def make_lidar_ply(las_path,ply_path):
    import laspy
    from scipy.spatial import cKDTree
    from plyfile import PlyData,PlyElement
    las=laspy.read(las_path); xyz=np.c_[las.x,las.y,las.z].astype(np.float32)
    rgb=np.zeros((len(xyz),3),np.uint8)
    if all(hasattr(las,n) for n in ('red','green','blue')):
        raw=np.c_[las.red,las.green,las.blue]
        rgb=np.clip(raw/(257 if raw.max()>255 else 1),0,255).astype(np.uint8)
    _,nn=cKDTree(xyz).query(xyz,k=min(16,len(xyz)),workers=-1)
    d=xyz[nn]-xyz[:,None,:]; cov=np.einsum('nki,nkj->nij',d,d)/d.shape[1]
    _,vec=np.linalg.eigh(cov); normals=vec[:,:,0].astype(np.float32); normals[normals[:,2]<0]*=-1
    v=np.empty(len(xyz),dtype=[('x','f4'),('y','f4'),('z','f4'),('red','u1'),('green','u1'),('blue','u1'),('nx','f4'),('ny','f4'),('nz','f4')])
    for i,n in enumerate(('x','y','z')): v[n]=xyz[:,i]
    for i,n in enumerate(('red','green','blue')): v[n]=rgb[:,i]
    for i,n in enumerate(('nx','ny','nz')): v[n]=normals[:,i]
    PlyData([PlyElement.describe(v,'vertex')],text=False).write(ply_path)
    return xyz,rgb

def colmap_alignment(repo,data,lidar_xyz):
    sys.path.insert(0,str(repo)); from scene.colmap_loader import read_points3D_binary,read_extrinsics_binary,qvec2rotmat
    s=data/'sparse'/'0'; s=s if s.exists() else data/'sparse'
    pts,_,_=read_points3D_binary(str(s/'points3D.bin')); ex=read_extrinsics_binary(str(s/'images.bin'))
    centers=np.array([-qvec2rotmat(e.qvec).T@e.tvec for e in ex.values()])
    lo,hi=lidar_xyz.min(0),lidar_xyz.max(0); margin=.1*(hi-lo)
    inside=np.mean(np.all((centers>=lo-margin)&(centers<=hi+margin),axis=1))
    from scipy.spatial import cKDTree
    med=float(np.median(cKDTree(lidar_xyz).query(pts,workers=-1)[0])); radius=float(np.linalg.norm(hi-lo)/2)
    report={'lidar_points':len(lidar_xyz),'camera_inside_expanded_bbox':float(inside),'sfm_to_lidar_median_nn':med,'scene_radius':radius,'normalized_median_nn':med/radius}
    (OUT/'lidar_alignment.json').write_text(json.dumps(report,indent=2));
    if inside<.8 or med/radius>.02: raise RuntimeError('LiDAR/COLMAP coordinate validation failed: '+json.dumps(report))
    return report

def geometry_against_lidar(model,lidar_xyz):
    from plyfile import PlyData
    from scipy.spatial import cKDTree
    candidates=list(model.glob('point_cloud/iteration_*/point_cloud.ply'))
    if not candidates: return {'error':'point cloud not found'}
    p=max(candidates,key=lambda x:int(re.search(r'iteration_(\d+)',str(x)).group(1)))
    v=PlyData.read(p)['vertex']; pred=np.c_[v['x'],v['y'],v['z']].astype(np.float32)
    lo,hi=lidar_xyz.min(0),lidar_xyz.max(0); pred=pred[np.all((pred>=lo)&(pred<=hi),axis=1)]
    rng=np.random.default_rng(601); pred=pred[rng.choice(len(pred),min(len(pred),300000),replace=False)]; gt=lidar_xyz[rng.choice(len(lidar_xyz),min(len(lidar_xyz),300000),replace=False)]
    a=cKDTree(gt).query(pred,workers=-1)[0]; c=cKDTree(pred).query(gt,workers=-1)[0]
    return {'source':str(p),'pred_points':len(pred),'lidar_points':len(gt),'accuracy_mean':float(a.mean()),'accuracy_rmse':float(np.sqrt(np.mean(a*a))),'completeness_mean':float(c.mean()),'completeness_rmse':float(np.sqrt(np.mean(c*c))),'chamfer_l1':float((a.mean()+c.mean())/2),'coverage_3cm':float(np.mean(c<=.03))}

def run_zs601():
    state('setup',gpu=gpu_name())
    repo=Path(__file__).resolve().parents[1]; patch_cuda(repo); pip_install(repo)
    z=Path('/content/drive/MyDrive/LCCDataset/ZS601meetingroom/ZS601meetingroom_data.zip')
    base=Path('/content/zs601_b_dataset');
    if not base.exists(): zipfile.ZipFile(z).extractall(base)
    data=find_dataset(base); las=next(base.rglob('ZS601_3cm_sample.las'))
    init=data/'lidar_init_3cm_normals.ply'; xyz,_=make_lidar_ply(las,init); align=colmap_alignment(repo,data,xyz)
    manifest=DRIVE/'zs601_2dgs_A_sfm_colored_fullres_20260912_045340'/'preview_cameras.json'
    common=['-s',data,'--images','images','--masks','masks','--mask_valid_value','black','--resolution','1','--eval','--sh_degree','2','--position_lr_init','0.000016','--position_lr_final','0.00000016','--position_lr_max_steps','150000','--scaling_lr','0.0015','--densification_interval','10000','--densify_until_iter','100000','--opacity_reset_interval','150000','--densify_grad_threshold','0.0002','--lambda_normal','0.05','--lambda_dist',('1000' if ARGS.mode=='c' else '0.0'),'--depth_ratio','0','--init_ply',init,'--preview_interval','5000','--preview_view_count','10','--preview_manifest',manifest,'--quiet']
    pre=OUT/'preflight_model'; pre_iters=4000 if ARGS.mode=='c' else 200; state('preflight',alignment=align,iterations=pre_iters)
    peak=monitor_run([sys.executable,'train.py','-m',pre,'--iterations',str(pre_iters),'--save_iterations',str(pre_iters),*common],repo,'preflight.log')
    if peak>13824: raise RuntimeError(f'preflight peak {peak} MiB exceeds 13.5GB')
    model=OUT/'model'; state('training',preflight_peak_mib=peak)
    monitor_run([sys.executable,'train.py','-m',model,'--iterations','150000','--test_iterations','50000','100000','150000','--save_iterations','50000','100000','150000','--checkpoint_iterations','50000','100000','150000',*common],repo,'train.log')
    state('rendering')
    run([sys.executable,'render.py','-s',data,'-m',model,'--iteration','150000','--skip_train','--depth_ratio','0','--quiet'],repo,'render.log')
    run([sys.executable,'metrics.py','-m',model],repo,'metrics.log')
    geom=geometry_against_lidar(model,xyz); (OUT/'geometry_metrics.json').write_text(json.dumps(geom,indent=2))
    result={'mode':ARGS.mode,'gpu':gpu_name(),'alignment':align,'preflight_peak_mib':peak,'image_metrics':json.loads((model/'results.json').read_text()) if (model/'results.json').exists() else None,'geometry_metrics':geom}
    (OUT/'results_summary.json').write_text(json.dumps(result,indent=2)); state('complete',summary=str(OUT/'results_summary.json'))

def download(url,dest):
    dest.parent.mkdir(parents=True,exist_ok=True); run(['wget','-c',url,'-O',dest],log='download.log')

def run_official():
    state('setup',gpu=gpu_name())
    repo=Path('/content/official_2dgs')
    if not repo.exists(): run(['git','clone','--recursive','https://github.com/hbb1/2d-gaussian-splatting.git',repo],log='install.log')
    commit=subprocess.run(['git','rev-parse','HEAD'],cwd=repo,capture_output=True,text=True,check=True).stdout.strip(); (OUT/'official_commit.txt').write_text(commit+chr(10))
    patch_cuda(repo); pip_install(repo)
    dl=Path('/content/dtu_download'); dl.mkdir(exist_ok=True)
    archive=next(dl.rglob('dtu.tar.gz'),None)
    if archive is None:
        run(['gdown','--folder','https://drive.google.com/drive/folders/1SJFgt8qhQomHX55Q4xSvYE2C6-8tFll9','-O',dl],log='download.log'); archive=next(dl.rglob('dtu.tar.gz'))
    data_base=Path('/content/dtu_scan105');
    if not list(data_base.rglob('cameras.npz')):
        data_base.mkdir(exist_ok=True)
        with tarfile.open(archive) as t:
            members=[m for m in t.getmembers() if 'scan105' in m.name]; t.extractall(data_base,members=members)
    scan=next(p.parent for p in data_base.rglob('cameras.npz') if p.parent.name=='scan105')
    gt=Path('/content/dtu_official_gt'); gt.mkdir(exist_ok=True)
    if not (gt/'Points').exists(): download('https://roboimagedata2.compute.dtu.dk/data/MVS/Points.zip',gt/'Points.zip'); zipfile.ZipFile(gt/'Points.zip').extractall(gt)
    if not (gt/'SampleSet').exists(): download('https://roboimagedata2.compute.dtu.dk/data/MVS/SampleSet.zip',gt/'SampleSet.zip'); zipfile.ZipFile(gt/'SampleSet.zip').extractall(gt)
    model=OUT/'model'; state('training',official_commit=commit)
    monitor_run([sys.executable,'train.py','-s',scan,'-m',model,'-r','2','--depth_ratio','1','--lambda_dist','1000','--eval','--quiet','--test_iterations','7000','15000','30000','--save_iterations','30000'],repo,'train.log')
    state('rendering')
    run([sys.executable,'render.py','--iteration','30000','-s',scan,'-m',model,'-r','2','--depth_ratio','1','--skip_train','--num_cluster','1','--voxel_size','0.004','--sdf_trunc','0.016','--depth_trunc','3.0','--quiet'],repo,'render.log')
    run([sys.executable,'metrics.py','-m',model],repo,'metrics.log')
    mesh=model/'train'/'ours_30000'/'fuse_post.ply'
    ev=repo/'scripts/eval_dtu/evaluate_single_scene.py'; eval_out=OUT/'geometry_eval'
    run([sys.executable,ev,'--input_mesh',mesh,'--scan_id','105','--output_dir',eval_out,'--mask_dir',scan.parent,'--DTU',gt],repo,'geometry.log')
    text=(OUT/'geometry.log').read_text(errors='ignore'); nums=re.findall(r'(?i)(?:overall|chamfer|mean)[^\n]*?([0-9]+(?:\.[0-9]+)?)',text)
    result={'mode':'official','gpu':gpu_name(),'official_commit':commit,'config':{'scene':'scan105','resolution':2,'depth_ratio':1,'lambda_dist':1000,'eval_split':True,'iterations':30000},'image_metrics':json.loads((model/'results.json').read_text()) if (model/'results.json').exists() else None,'geometry_metric_candidates':nums[-10:],'mesh':str(mesh)}
    (OUT/'results_summary.json').write_text(json.dumps(result,indent=2)); state('complete',summary=str(OUT/'results_summary.json'))

try:
    run_zs601() if ARGS.mode in ('b','c') else run_official()
except Exception as exc:
    import traceback; traceback.print_exc(); state('failed',error=repr(exc)); raise

