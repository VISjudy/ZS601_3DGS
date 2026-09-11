"""Read a trusted v3 checkpoint on CPU and export bounded scale diagnostics."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import torch
from geometry_v3 import diagnostics


def analyze(checkpoint):
    ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
    assert ck['version']==3
    # GaussianModel.capture(): active SH, xyz, DC, rest, scaling, rotation, opacity.
    _,xyz,_,_,log_scale,rotation,log_opacity,*_=ck['model']
    scales=log_scale.exp(); opacity=log_opacity.sigmoid().flatten()
    r=ck['reference']; a=SimpleNamespace(**ck['config'])
    ratio=scales[:,:2].amax(1)/(a.size_ratio*r['spacing'])
    visible=opacity>=.05
    result={'iteration':ck['iteration'],'units':a.units,
            'ellipsoid_opacity_threshold':.05,
            'all':diagnostics(xyz,scales,rotation,r,a),
            'opaque_candidate_count':int(visible.sum()),
            'opaque_candidates':diagnostics(xyz[visible],scales[visible],rotation[visible],
                {k:v[visible] for k,v in r.items()},a) if visible.any() else None,
            'visible_definition':'opacity threshold only, not camera occlusion visibility',
            'largest':[]}
    for i in torch.argsort(scales[:,:2].amax(1),descending=True)[:30].tolist():
        result['largest'].append({'index':i,'source_id':int(r['source_id'][i]),
            'xyz':xyz[i].tolist(),'anchor':r['anchor'][i].tolist(),
            'scales_1sigma':scales[i].tolist(),'opacity':float(opacity[i]),
            'spacing':float(r['spacing'][i]),'size_ratio_to_limit':float(ratio[i]),
            'normal_confidence':float(r['confidence'][i])})
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--output',required=True)
    a=p.parse_args()
    result=analyze(a.checkpoint)
    path=Path(a.output); path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as f: json.dump(result,f,indent=2)
    print(json.dumps(result,indent=2))
