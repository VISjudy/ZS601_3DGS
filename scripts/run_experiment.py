"""One shared entrypoint; presets only change paper_arm and explicit input paths."""
import argparse,hashlib,json,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def read_config(arm):
    common=json.loads((ROOT/'configs/common.json').read_text())
    variant=json.loads((ROOT/'configs'/f'{arm}.json').read_text())
    return common|variant

def build_command(config,inputs,output):
    opts=dict(config['training'])
    opts.update(experiment='original',paper_arm=config['paper_arm'],source_path=inputs['source_path'],
        point_cloud=inputs['point_cloud'],train_file=inputs['train_file'],val_file=inputs['val_file'],
        test_file=inputs['test_file'],cameras_file=inputs['cameras_file'],alpha_masks=inputs.get('alpha_masks',''),
        precomputed_root=inputs.get('supervision_root',''),supervision_root=inputs.get('supervision_root',''),
        supervision_manifest=inputs.get('supervision_manifest',''),data_manifest=inputs.get('data_manifest',''),model_path=str(output))
    cmd=[sys.executable,str(ROOT/'train_mask_v3.py')]
    for name,value in opts.items():cmd += ['--'+name,str(value)]
    return cmd

def verify_code():
    expected=json.loads((ROOT/'provenance/trainer-source-sha256.json').read_text())
    for name,digest in expected.items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,'Source differs: '+name

def preflight(config,inputs):
    for key in ['source_path','point_cloud','train_file','val_file','test_file','cameras_file','supervision_manifest','data_manifest']:
        if inputs.get(key):assert Path(inputs[key]).exists(),f'Missing {key}: {inputs[key]}'
    def names(path):
        # COLMAP text records alternate pose / feature line, including empty feature lines.
        lines=Path(path).read_text().splitlines();result=[];i=0
        while i<len(lines):
            line=lines[i].strip();i+=1
            if not line or line.startswith('#'):continue
            result.append(' '.join(line.split()[9:]));i+=1
        return result
    train,val,test=[names(inputs[k]) for k in ['train_file','val_file','test_file']]
    assert len(val)==10 and all(len(n)==len(set(n)) for n in [train,val,test])
    assert not (set(train)&set(val) or set(train)&set(test) or set(val)&set(test))
    virtual=set(json.loads(Path(inputs['supervision_manifest']).read_text()).get('virtual_names',[]))
    if config['virtual_views']:assert len(virtual)==200 and virtual.issubset(set(train))
    for name in train+val+test:
        assert (Path(inputs['source_path'])/'images'/name).is_file(),name
        if inputs.get('alpha_masks'):assert (Path(inputs['source_path'])/inputs['alpha_masks']/Path(name).with_suffix('.png')).is_file(),name
    if config['depth_loss'] or config['normal_loss']:
        for name in set(train)-virtual:
            assert (Path(inputs['supervision_root'])/'arrays'/f'{int(Path(name).stem):06d}.npz').exists(),f'Missing supervision: {name}'
    if config['dataset']=='real':
        protocol=json.loads(Path(inputs['supervision_manifest']).read_text())
        assert protocol.get('dataset_kind')=='real' and protocol.get('source_cloud_sha256'),'Real data must use real-LiDAR supervision, never synthetic targets'
    return dict(train=len(train),validation=len(val),test=len(test),virtual=len(virtual))

def main():
    p=argparse.ArgumentParser();p.add_argument('--arm',required=True,choices=['A','B','C','D','E','F-real'])
    p.add_argument('--inputs',required=True);p.add_argument('--output',required=True);p.add_argument('--notebook');p.add_argument('--execute',action='store_true')
    a=p.parse_args();verify_code();cfg=read_config(a.arm);inp=json.loads(Path(a.inputs).read_text());out=Path(a.output)
    assert not out.exists(),'Choose a fresh output directory; existing training/results are protected'
    counts=preflight(cfg,inp);cmd=build_command(cfg,inp,out)
    print(json.dumps(dict(config=cfg,counts=counts,command=cmd),indent=2))
    if not a.execute:return
    proc=subprocess.Popen(cmd,cwd=ROOT)
    while proc.poll() is None and not (out/'run_config.json').exists():time.sleep(.2)
    if out.exists():
        (out/'reproduction-config.json').write_text(json.dumps(dict(config=cfg,inputs=inp,command=cmd),indent=2))
        if a.notebook:
            with (out/'reproduce.ipynb').open('xb') as f:f.write(Path(a.notebook).read_bytes())
    if proc.wait()!=0:raise RuntimeError(f'Training failed: exit {proc.returncode}')
    done=json.loads((out/'completed.json').read_text());assert done['iteration']==50000 and done['final_test_complete']

if __name__=='__main__':main()
