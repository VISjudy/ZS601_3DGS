"""Stage shared Drive data locally without writing back duplicates to Drive."""
import argparse,json,shutil,hashlib
from pathlib import Path

def pose_names(path):
    rows=Path(path).read_text().splitlines();names=[];i=0
    while i<len(rows):
        row=rows[i].strip();i+=1
        if not row or row.startswith('#'):continue
        names.append(' '.join(row.split()[9:]));i+=1
    return names

def copy_once(src,dst):
    assert src.is_file(),f'Missing input: {src}'
    dst.parent.mkdir(parents=True,exist_ok=True)
    if dst.exists():
        assert hashlib.sha256(src.read_bytes()).digest()==hashlib.sha256(dst.read_bytes()).digest(),f'Conflicting cached input: {dst}'
    else:shutil.copyfile(src,dst)

def main():
    p=argparse.ArgumentParser();p.add_argument('--drive-root',required=True);p.add_argument('--local-root',required=True);p.add_argument('--arm',required=True);a=p.parse_args()
    root=Path(a.drive_root);local=Path(a.local_root);real=a.arm=='F-real';virtual=a.arm in ['E','F-real']
    # Real inputs require their own prepared LiDAR targets and virtual RGB. Synthetic data cannot substitute.
    base=root/'01-datasets'/('real-prepared' if real else 'synthetic-base')
    supervision=root/'01-datasets'/('real-supervision' if real else 'supervision')
    virtual_root=root/'01-datasets'/('real-virtual-frozen-200' if real else 'virtual-frozen-200')
    if not base.exists():raise FileNotFoundError(f'Prepared dataset required: {base}; see DATA_CONTRACT.md')
    source=local/('real-merged' if real else ('synthetic-merged' if virtual else 'synthetic'))
    camera_local=local/('real-cameras' if real else 'cameras')
    for subset in ['train','validation','test']+(['virtual_near'] if virtual else []):
        for name in ['images.txt','cameras.txt']:copy_once(base/'cameras'/subset/name,camera_local/subset/name)
    native_source=base/('real' if real else 'synthetic')
    all_names=[]
    for subset in ['train','validation','test']:all_names+=pose_names(camera_local/subset/'images.txt')
    for name in all_names:
        copy_once(native_source/'images'/name,source/'images'/name)
        if real:copy_once(native_source/'masks'/Path(name).with_suffix('.png'),source/'masks'/Path(name).with_suffix('.png'))
    virtual_names=[]
    if virtual:
        from PIL import Image
        virtual_names=pose_names(camera_local/'virtual_near/images.txt');assert len(virtual_names)==200
        for name in virtual_names:
            copy_once(virtual_root/'images'/name,source/'images'/name)
            copy_once(virtual_root/'masks'/Path(name).with_suffix('.png'),source/'masks'/Path(name).with_suffix('.png'))
        if not real:
            for name in all_names:
                q=source/'masks'/Path(name).with_suffix('.png');q.parent.mkdir(exist_ok=True)
                if not q.exists():
                    with Image.open(source/'images'/name) as image:Image.new('L',image.size,0).save(q)
        merged=camera_local/'train-virtual';merged.mkdir(exist_ok=True)
        (merged/'images.txt').write_text((camera_local/'train/images.txt').read_text()+'\n'+(camera_local/'virtual_near/images.txt').read_text())
        # Deduplicate shared calibration IDs and reject conflicts.
        records={}
        for subset in ['train','virtual_near']:
            for line in (camera_local/subset/'cameras.txt').read_text().splitlines():
                if not line.strip() or line.startswith('#'):continue
                key=line.split()[0];assert key not in records or records[key]==line;records[key]=line
        (merged/'cameras.txt').write_text('\n'.join(records.values())+'\n')
    cloud=base/'pointclouds'/('points_3cm.las' if real else 'points_3cm.ply');cloud_local=local/('real-'+cloud.name if real else cloud.name);copy_once(cloud,cloud_local)
    sup_local=local/('real-supervision' if real else 'supervision')
    if a.arm!='A':
        for name in pose_names(camera_local/'train/images.txt'):
            n=f'{int(Path(name).stem):06d}.npz';copy_once(supervision/'arrays'/n,sup_local/'arrays'/n)
        for kind in ['depth','valid','normal_preview','normal_valid']:
            for q in (supervision/kind).glob('*.png'):copy_once(q,sup_local/kind/q.name)
    protocol=dict(dataset_kind='real' if real else 'synthetic',virtual_names=virtual_names,method='lp_radius2',depth='camera-Z meters',normal='camera-space PCA normal',interpolation=False)
    if real:
        verified=json.loads((supervision/'protocol.json').read_text());assert verified['dataset_kind']=='real' and verified['source_cloud_sha256'];protocol.update(verified);protocol['virtual_names']=virtual_names
    local.mkdir(exist_ok=True,parents=True);protocol_path=local/(a.arm+'-supervision-protocol.json');protocol_path.write_text(json.dumps(protocol,indent=2))
    train_subset='train-virtual' if virtual else 'train'
    result=dict(source_path=str(source),point_cloud=str(cloud_local),train_file=str(camera_local/train_subset/'images.txt'),val_file=str(camera_local/'validation/images.txt'),test_file=str(camera_local/'test/images.txt'),cameras_file=str(camera_local/train_subset/'cameras.txt'),alpha_masks='masks' if real or virtual else '',supervision_root=str(sup_local),supervision_manifest=str(protocol_path),data_manifest='')
    output=local/(a.arm+'-inputs.json');output.write_text(json.dumps(result,indent=2));print(output)

if __name__=='__main__':main()
