"""Derive train text by excluding supplied held-out lists; never invent val cameras."""
import argparse,json
from pathlib import Path

def records(path):
    lines=Path(path).read_text(encoding='utf-8-sig').splitlines(); result=[]; i=0
    while i<len(lines):
        line=lines[i].strip(); i+=1
        if not line or line.startswith('#'): continue
        p=line.split()
        if len(p)!=10: raise ValueError(f'Invalid pose header in {path}: {line[:80]}')
        if i>=len(lines): raise ValueError('Missing POINTS2D row (can be blank)')
        points=lines[i]; i+=1
        result.append((p[9],line,points))
    names=[r[0] for r in result]
    if len(set(names))!=len(names): raise ValueError('Duplicate camera names')
    return result

def prepare(images_file,val_file,test_file,output_train):
    source=records(images_file); val=records(val_file)
    if len(val)!=10: raise ValueError('Supply exactly ten fixed val cameras')
    test=records(test_file) if test_file else []
    excluded={r[0] for r in val+test}; train=[r for r in source if r[0] not in excluded]
    if not train: raise ValueError('No training cameras remain')
    out=Path(output_train); out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('x',encoding='utf-8') as f:
        f.write('# v3 explicit train list; supplied val/test excluded; original files retained.\n')
        for _,pose,points in train: f.write(pose+'\n'+points+'\n')
    print(json.dumps({'input':len(source),'train':len(train),'val':len(val),'test':len(test),
        'removed_from_train':len(source)-len(train),'output':str(out)}))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--images_file',required=True); p.add_argument('--val_file',required=True)
    p.add_argument('--test_file',default=''); p.add_argument('--output_train',required=True)
    a=p.parse_args(); prepare(a.images_file,a.val_file,a.test_file,a.output_train)
