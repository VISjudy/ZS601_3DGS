"""Topology bookkeeping, logs and versioned checkpoints for v3."""
import json, random, subprocess, time
from pathlib import Path
import numpy as np
import torch

def append_csv(path,record):
    import csv
    path=Path(path)
    exists=path.exists() and path.stat().st_size>0
    with path.open('a',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(record))
        if not exists: writer.writeheader()
        writer.writerow(record)

def append_json(path,value):
    with open(path,'a',encoding='utf-8') as f: f.write(json.dumps(value,ensure_ascii=False)+'\n')

def signature(a):
    # Input contents are checked separately, allowing a new Colab extraction path.
    paths={'resume','model_path','source_path','point_cloud','train_file','val_file','cameras_file','test_file','baseline_result'}
    return {k:v for k,v in vars(a).items() if k not in paths}

def provenance():
    root=Path(__file__).resolve().parent
    def git(*args):
        try: return subprocess.check_output(['git','-C',str(root),*args],text=True,stderr=subprocess.DEVNULL).strip()
        except (OSError,subprocess.CalledProcessError): return 'unavailable'
    from data_v3 import sha256
    return {'git_commit':git('rev-parse','HEAD'),'git_status':git('status','--porcelain'),
            'v3_source_sha256':{p.name:sha256(p) for p in sorted(root.glob('*_v3.py'))},
            'torch':torch.__version__,'cuda':torch.version.cuda,
            'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}

def fresh_topology(g):
    n=len(g.get_xyz); dev=g.get_xyz.device
    return {k:torch.zeros(n,dtype=torch.int32,device=dev) for k in ('epoch_views','max_epoch_views','low_streak','densify_count')}

def finish_epoch(state):
    state['max_epoch_views']=torch.maximum(state['max_epoch_views'],state['epoch_views'])
    state['epoch_views'].zero_()

@torch.no_grad()
def prune(g,r,state,a,iteration):
    if not a.pruning or iteration<a.prune_start or iteration%a.prune_interval: return r,state,None
    views=torch.maximum(state['epoch_views'],state['max_epoch_views'])
    low=(g.get_opacity.flatten()<a.prune_opacity)&(views>=a.prune_min_views)
    state['low_streak']=torch.where(low,state['low_streak']+1,0)
    eligible=state['low_streak']>=a.prune_patience
    candidates=eligible.nonzero().flatten()
    cap=min(max(0,int(len(low)*a.prune_max_fraction)),len(low)-1)
    chosen=candidates[torch.argsort(g.get_opacity.flatten()[candidates])[:cap]]
    remove=torch.zeros_like(low); remove[chosen]=True; keep=~remove
    if len(chosen):
        # Baseline prune_points expects this buffer even without densification.
        g.tmp_radii=torch.zeros(len(low),device=g.get_xyz.device)
        g.prune_points(remove)
        g.tmp_radii=None
        r={k:v[keep] for k,v in r.items()}
        state={k:v[keep] for k,v in state.items()}
    event={'iteration':iteration,'candidates':len(candidates),'removed':len(chosen),
           'remaining':len(g.get_xyz),'criterion':'persistent low opacity with distinct frustum views per sampling epoch'}
    return r,state,event

def check_finite(g,gradients=False):
    for name in ('_xyz','_scaling','_rotation','_opacity','_features_dc','_features_rest'):
        x=getattr(g,name)
        if not torch.isfinite(x).all(): raise FloatingPointError('Nonfinite '+name)
        if gradients and x.grad is not None and not torch.isfinite(x.grad).all():
            raise FloatingPointError('Nonfinite gradient '+name)
    if (torch.linalg.vector_norm(g._rotation,dim=-1)<1e-8).any(): raise FloatingPointError('Zero quaternion')
    if not torch.isfinite(g.get_scaling).all() or (g.get_scaling<=0).any():
        raise FloatingPointError('Invalid activated scale')

def save_checkpoint(path,g,reference,state,sampler,iteration,a,identity,code,elapsed):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists(): raise FileExistsError(path)
    torch.save({'version':3,'iteration':iteration,'config':signature(a),'identity':identity,
        'source_sha256':code['v3_source_sha256'],'model':g.capture(),
        'exposure':g._exposure.detach(),'exposure_optimizer':g.exposure_optimizer.state_dict(),
        'reference':reference,'topology':state,'sampler':sampler,'elapsed':elapsed,
        'rng':{'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),
               'cuda':torch.cuda.get_rng_state_all()}},path)
    print('[CHECKPOINT]',path,flush=True)

def restore_checkpoint(path,g,a,identity,code):
    # Only resume your own trusted checkpoints (torch pickle).
    ck=torch.load(path,map_location='cpu',weights_only=False)
    if ck.get('version')!=3: raise ValueError('Not a v3 full checkpoint; legacy hot-start is unsupported')
    if ck['config']!=signature(a): raise ValueError('Resume configuration differs (paths may change, training settings may not)')
    if ck['identity']!=identity: raise ValueError('Input data identity differs')
    if ck['source_sha256']!=code['v3_source_sha256']: raise ValueError('v3 source differs; use original version to resume')
    model=list(ck['model'])
    for i,v in enumerate(model):
        if isinstance(v,torch.Tensor): model[i]=torch.nn.Parameter(v.cuda(),requires_grad=True) if i in range(1,7) else v.cuda()
    g.restore(tuple(model),a)
    with torch.no_grad(): g._exposure.copy_(ck['exposure'].cuda())
    g.exposure_optimizer.load_state_dict(ck['exposure_optimizer'])
    r={k:v.cuda() for k,v in ck['reference'].items()}
    state={k:v.cuda() for k,v in ck['topology'].items()}
    random.setstate(ck['rng']['python']); np.random.set_state(ck['rng']['numpy'])
    torch.set_rng_state(ck['rng']['torch']); torch.cuda.set_rng_state_all(ck['rng']['cuda'])
    return r,state,ck['sampler'],ck['iteration'],ck['elapsed']
