"""Decode delivered PNGs, export colored point PLYs, and evaluate both methods.

Mesh first hits are an independent evaluation reference only. They never fill,
filter, or otherwise alter either method's pixels or exported point clouds.
"""
from pathlib import Path
from dataclasses import asdict
import csv
import json
import os
import sys
import time

os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parents[2]
sys.path.insert(0,str(PROJECT/'.runtime/synthetic-dataset'))
sys.path.insert(0,str(PROJECT/'experiments/2026-09-21/synthetic-lidar-init-v001/python-deps'))
sys.path.insert(0,str(PROJECT/'experiments/2026-09-21/synthetic-lidar-init-v001/repository/gaussian-splatting-lidar-init'))
import numpy as np
from PIL import Image
from scipy.spatial import cKDTree
from scipy.ndimage import maximum_filter,minimum_filter,binary_erosion
from plyfile import PlyData,PlyElement
from embreex.rtcore_scene import EmbreeScene
from embreex.mesh_construction import TriangleMesh
from colmap_io import rotation,read_views,sha256,encode_depth

OUT=PROJECT/'3dgsResult/zs601-depth-methods-v007'
METHODS=['method_a_gaussian','method_b_zbuffer']


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as f:json.dump(value,f,indent=2,allow_nan=False)


def npy(path,array):
    with path.open('xb') as f:np.save(f,array,allow_pickle=False)


def write_ply(path,xyz,rgb):
    array=np.empty(len(xyz),dtype=[('x','<f4'),('y','<f4'),('z','<f4'),
        ('red','u1'),('green','u1'),('blue','u1')])
    for i,k in enumerate(['x','y','z']):array[k]=xyz[:,i]
    for i,k in enumerate(['red','green','blue']):array[k]=rgb[:,i]
    with path.open('xb') as f:
        PlyData([PlyElement.describe(array,'vertex')],text=False,byte_order='<',
            comments=['World metres; backprojected SAVED uint16-mm camera-Z PNG.',
                      'No ICP, no mesh filtering, no hole filling.']).write(f)
    loaded=PlyData.read(path)['vertex'].data
    assert np.array_equal(loaded,array)
    return dict(points=len(xyz),bytes=path.stat().st_size,sha256=sha256(path),roundtrip_exact=True)


def summary(values,signed=False):
    a=np.asarray(values,dtype=np.float64)
    if not len(a):return dict(count=0)
    assert np.isfinite(a).all()
    ab=np.abs(a)
    result=dict(count=len(a),mean_mm=float(ab.mean()*1000),
        median_mm=float(np.median(ab)*1000),p95_mm=float(np.quantile(ab,.95)*1000),
        rmse_mm=float(np.sqrt(np.mean(a*a))*1000),max_mm=float(ab.max()*1000),
        above_10mm_percent=float((ab>.01).mean()*100),above_30mm_percent=float((ab>.03).mean()*100))
    if signed:result.update(bias_mm=float(a.mean()*1000),
        behind_first_surface_gt30mm_percent=float((a>.03).mean()*100),
        in_front_of_surface_gt30mm_percent=float((a<-.03).mean()*100))
    return result


def rays_for_view(v):
    y,x=np.indices((v['height'],v['width']),dtype=np.float64)
    return np.stack([(x+.5-v['cx'])/v['fx'],(y+.5-v['cy'])/v['fy'],np.ones_like(x)],-1)


def reference_depth(scene,v,near,far):
    R=rotation(v['q']);t=np.asarray(v['t'])
    center=-R.T@t
    dirs=rays_for_view(v).reshape(-1,3)@R
    z=np.zeros(len(dirs),dtype=np.float32)
    ids=np.full(len(dirs),-1,dtype=np.int32)
    for lo in range(0,len(dirs),100000):
        dr=dirs[lo:lo+100000]
        origins=center+near*dr
        hit=scene.run(np.ascontiguousarray(origins,dtype='f4'),
            np.ascontiguousarray(dr,dtype='f4'),query='INTERSECT',output=1)
        valid=(hit['primID']>=0)&np.isfinite(hit['tfar'])&(hit['tfar']>0)&(hit['tfar']+near<=far)
        z[lo:lo+len(dr)]=np.where(valid,hit['tfar']+near,0)
        ids[lo:lo+len(dr)]=np.where(valid,hit['primID'],-1)
    return z.reshape(v['height'],v['width']),ids.reshape(v['height'],v['width'])


def backproject(v,depth_m,mask):
    rays=rays_for_view(v)[mask]
    pc=rays*depth_m[mask,None]
    R=rotation(v['q']);t=np.asarray(v['t'])
    world=(pc-t)@R
    # Independent inverse matrix route, plus full pixel and depth roundtrip.
    extrinsic=np.eye(4);extrinsic[:3,:3]=R;extrinsic[:3,3]=t
    inverse=np.linalg.inv(extrinsic)
    independent=pc@inverse[:3,:3].T+inverse[:3,3]
    assert np.max(np.abs(world-independent))<1e-10
    recovered=world@R.T+t
    y,x=np.nonzero(mask)
    px=v['fx']*recovered[:,0]/recovered[:,2]+v['cx']
    py=v['fy']*recovered[:,1]/recovered[:,2]+v['cy']
    assert max(np.abs(px-x-.5).max(),np.abs(py-y-.5).max())<1e-8
    assert np.abs(recovered[:,2]-depth_m[mask]).max()<1e-10
    return world


def check_png(path,bit_depth,channels,width,height):
    header=path.read_bytes()[:33]
    assert header[:8]==b'\x89PNG\r\n\x1a\n'
    assert header[24]==bit_depth and header[25]=={1:0,3:2}[channels]
    with Image.open(path) as im:im.verify()
    a=np.asarray(Image.open(path))
    assert a.shape[:2]==(height,width)
    assert a.dtype==({8:np.uint8,16:np.uint16}[bit_depth])
    return a


def main():
    start=time.time()
    spec=json.loads((OUT/'run_spec.json').read_text())
    views=json.loads((OUT/'selected_views.json').read_text())
    assert [asdict(v) for v in read_views(OUT/'sparse/0')]==views
    source_views=read_views(PROJECT/'scenes/zs601-meetingroom/synthetic-training-v001/virtual_near/sparse/0')
    selected=sorted(np.random.default_rng(spec['random_seed']).choice(len(source_views),10,replace=False))
    assert [asdict(source_views[i]) for i in selected]==views
    assert sha256(spec['input_cloud'])==spec['input_cloud_sha256']
    assert sha256(spec['source_gaussian'])==spec['source_gaussian_sha256']
    geom_sha=sha256(spec['source_geometry']);excluded_sha=sha256(spec['excluded_glass_faces'])
    metrics=OUT/'metrics';metrics.mkdir(exist_ok=False)
    refout=OUT/'reference_noglass_mesh';refout.mkdir(exist_ok=False)
    for name in ['depth','depth_float','valid_masks','discontinuity_masks']:(refout/name).mkdir()
    # Check the new Gaussian's complete parameter file; all fields except opacity
    # must be bit-exact to the independently checked v006 initialization.
    old=PlyData.read(spec['source_gaussian'])['vertex'].data
    current=PlyData.read(OUT/'method_a_gaussian/point_cloud/iteration_0/point_cloud.ply')['vertex'].data
    assert old.dtype==current.dtype and len(old)==len(current)
    preserved=[]
    for k in current.dtype.names:
        if k=='opacity':assert np.all(current[k]==20.0)
        else:
            assert np.array_equal(old[k],current[k]),k
            preserved.append(k)
    del old,current
    p=PlyData.read(spec['input_cloud'])['vertex']
    xyz=np.column_stack([p[k] for k in ['x','y','z']]).astype(np.float64)
    tree=cKDTree(xyz,balanced_tree=True,compact_nodes=True)
    print('INPUT_KDTREE_READY',len(xyz),flush=True)
    geo=np.load(spec['source_geometry'])
    excluded=np.load(spec['excluded_glass_faces'])
    assert excluded.dtype==np.bool_ and excluded.shape==(len(geo['faces']),)
    keep=~excluded
    excluded_count=int(excluded.sum())
    assert excluded_count==3468
    scene=EmbreeScene();mesh=TriangleMesh(scene,geo['vertices'].astype(np.float32),geo['faces'][keep].astype(np.int32))
    # Analytic oblique rays prove Embree tfar uses our unnormalized camera-Z ray.
    test=EmbreeScene();testmesh=TriangleMesh(test,np.array([[-100,-100,2],[100,-100,2],[0,100,2]],dtype='f4'),np.array([[0,1,2]],dtype='i4'))
    rr=np.array([[0,0,1],[.3,-.2,1]],dtype='f4');oo=.2*rr
    analytic=test.run(oo,rr,query='INTERSECT',output=1)
    assert np.max(np.abs(analytic['tfar']+.2-2))<1e-6 and np.all(analytic['primID']==0)
    agg={m:dict(nn=[],own=[],common=[],interior=[],edge=[],strict=[],world=[],rgb=[],
        count=0,coverage_rgb=0,total=0,matched_source=[],quantization=[],png_world_error=[],rows=[],ply=[]) for m in METHODS}
    rows=[]
    for v in views:
        ts=time.time();name=v['name'];stem=Path(name).stem;H=v['height'];W=v['width']
        reference,face=reference_depth(scene,v,spec['near_clip_m'],spec['far_clip_m'])
        refvalid=reference>0
        # 3x3 range identifies depth boundaries; only full-valid neighborhoods
        # qualify as interior. Used only for diagnostic slicing.
        mn=minimum_filter(np.where(refvalid,reference,np.inf),size=3,mode='nearest')
        mx=maximum_filter(reference,size=3,mode='nearest')
        interior=refvalid&binary_erosion(refvalid,structure=np.ones((3,3)))&((mx-mn)<=.03)
        edge=refvalid&~interior
        npy(refout/'depth_float'/f'{stem}.npy',reference)
        Image.fromarray(encode_depth(reference,refvalid)).save(refout/'depth'/name)
        Image.fromarray(refvalid.astype('u1')*255).save(refout/'valid_masks'/name)
        Image.fromarray(edge.astype('u1')*255).save(refout/'discontinuity_masks'/name)
        data={}
        for method in METHODS:
            base=OUT/method
            depth=check_png(base/'depth'/name,16,1,W,H)
            maskfolder='depth_mask' if method==METHODS[0] else 'masks'
            mask8=check_png(base/maskfolder/name,8,1,W,H)
            assert set(np.unique(mask8)).issubset({0,255})
            mask=mask8>0
            assert np.array_equal(depth>0,mask)
            rgb=check_png(base/'images'/name,8,3,W,H)
            raw=np.load(base/'depth_float'/f'{stem}.npy')
            assert raw.shape==depth.shape and raw.dtype==np.float32 and np.isfinite(raw).all()
            assert np.array_equal(raw>0,mask)
            z=depth.astype(np.float64)/1000
            world=backproject(v,z,mask)
            rawworld=backproject(v,raw.astype(np.float64),mask)
            quant=z[mask]-raw[mask].astype(np.float64)
            # np.rint on float32 adds only sub-micron representation error.
            assert np.abs(quant).max()<.000501
            pixelquant=np.linalg.norm(world-rawworld,axis=1)
            # All metrics use the exported float32 world coordinates, reloaded
            # losslessly by write_ply; not the higher precision temporary array.
            wf=world.astype(np.float32)
            d,nearest=tree.query(wf,k=1,workers=4)
            plydir=base/'backprojected';plydir.mkdir(exist_ok=True)
            record=write_ply(plydir/f'{stem}.ply',wf,rgb[mask])
            errors=(z-reference.astype(np.float64))
            rr=dict(method=method,image_id=v['image_id'],name=name,
                pixels=H*W,valid_pixels=int(mask.sum()),coverage_percent=float(mask.mean()*100),
                gap_percent=float((~mask).mean()*100),reference_valid_pixels=int(refvalid.sum()),
                own_valid_reference_pixels=int((mask&refvalid).sum()),
                valid_without_reference_pixels=int((mask&~refvalid).sum()),
                point_to_input=summary(d),own_valid_depth=summary(errors[mask&refvalid],True),
                interior_depth=summary(errors[mask&interior],True),boundary_depth=summary(errors[mask&edge],True),
                png_quantization=summary(quant,True),png_quantization_world_displacement=summary(pixelquant),
                float32_PLY_storage_max_error_m=float(np.abs(wf.astype(float)-world).max()),
                PLY=record)
            if method==METHODS[0]:
                rgbmask=check_png(base/'masks'/name,8,1,W,H)>0
                alpha=check_png(base/'alpha'/name,16,1,W,H)
                assert np.all(~rgbmask|mask)
                rr['rgb_alpha95_gap_percent']=float((~rgbmask).mean()*100)
                rr['strictly_empty_alpha16_percent']=float((alpha==0).mean()*100)
                strict=errors[rgbmask&refvalid]
                rr['alpha95_depth']=summary(strict,True)
                agg[method]['strict'].append(strict)
            else:
                rgbmask=mask
                winners=np.load(base/'winner_source_id'/f'{stem}.npy')
                assert winners.shape==mask.shape and np.array_equal(winners>=0,mask)
                matched=np.linalg.norm(wf.astype(float)-xyz[winners[mask]],axis=1)
                rr['backprojection_to_exact_source_point']=summary(matched)
                # Pixel-center snapping can move an original point sideways.
                # Exact source differences must stay inside the half-pixel plus
                # depth-quantization bound; no assertion of zero point error.
                rays=rays_for_view(v)[mask]
                bound=z[mask]*np.sqrt((.5/v['fx'])**2+(.5/v['fy'])**2)+.000502*np.linalg.norm(rays,axis=1)+2e-6
                assert np.all(matched<=bound)
                rr['all_matched_source_errors_within_pixel_quantization_bound']=True
                agg[method]['matched_source'].append(matched)
            a=agg[method]
            for key,value in [('nn',d),('own',errors[mask&refvalid]),('interior',errors[mask&interior]),
                ('edge',errors[mask&edge]),('world',wf),('rgb',rgb[mask]),('quantization',quant),('png_world_error',pixelquant)]:a[key].append(value)
            a['count']+=int(mask.sum());a['total']+=H*W;a['coverage_rgb']+=int(rgbmask.sum());a['ply'].append(record)
            data[method]=(z,mask,errors,rr)
        common=data[METHODS[0]][1]&data[METHODS[1]][1]&refvalid
        for method in METHODS:
            z,mask,errors,rr=data[method]
            rr['common_valid_depth']=summary(errors[common],True)
            agg[method]['common'].append(errors[common]);agg[method]['rows'].append(rr);rows.append(rr)
        print(json.dumps(dict(view=v['image_id'],seconds=time.time()-ts,common_pixels=int(common.sum()),
            A_nn_mean_mm=data[METHODS[0]][3]['point_to_input']['mean_mm'],
            B_nn_mean_mm=data[METHODS[1]][3]['point_to_input']['mean_mm'],
            A_common_MAE_mm=data[METHODS[0]][3]['common_valid_depth']['mean_mm'],
            B_common_MAE_mm=data[METHODS[1]][3]['common_valid_depth']['mean_mm'])),flush=True)
    aggregate={}
    for method in METHODS:
        a=agg[method]
        merge=write_ply(OUT/method/'backprojected/merged_10views.ply',np.concatenate(a['world']),np.concatenate(a['rgb']))
        r=dict(views=10,total_pixels=a['total'],valid_pixels=a['count'],
            coverage_percent=100*a['count']/a['total'],gap_percent=100*(1-a['count']/a['total']),
            rgb_mask_gap_percent=100*(1-a['coverage_rgb']/a['total']),
            merged_PLY=merge,merged_contains_duplicate_surfaces=True)
        for key,label,signed in [('nn','point_to_input',False),('own','own_valid_depth',True),
            ('common','common_valid_depth',True),('interior','interior_depth',True),('edge','boundary_depth',True),
            ('quantization','png_quantization',True),('png_world_error','png_quantization_world_displacement',False)]:
            r[label]=summary(np.concatenate(a[key]),signed)
        if a['strict']:r['alpha95_depth']=summary(np.concatenate(a['strict']),True)
        if a['matched_source']:r['backprojection_to_exact_source_point']=summary(np.concatenate(a['matched_source']))
        aggregate[method]=r
    result=dict(status='EVALUATED_SMOKE_NOT_TRAINING_OR_OCCLUSION_PASS',views=10,
        aggregate=aggregate,per_view=rows,
        protocol=dict(input_confirmed_by_user='same simulated noglass 1 cm cloud for both methods',
            source_cloud_sha256=spec['input_cloud_sha256'],source_geometry_sha256=geom_sha,
            excluded_faces_sha256=excluded_sha,excluded_triangles=excluded_count,kept_triangles=int(keep.sum()),
            world_units='metres',depth_units='mm integer camera Z; not radial range',
            error_aggregation='pool all eligible pixels/points over 10 views, not mean of per-view means',
            point_error='one-sided Euclidean NN from every saved float32 backprojected PLY point to original noglass 1cm cloud; no ICP',
            depth_error='saved decoded PNG camera Z minus independent unnormalized-ray no-glass mesh first-hit camera Z',
            common_mask='A depth alpha>=0.5 AND B occupied AND mesh reference valid',
            occlusion_proxy='signed depth difference >30 mm behind or <-30 mm in front of mesh first hit; boundary sampling can contribute',
            interior='3x3 mesh-depth range<=30mm and complete valid 3x3 neighborhood',
            mesh_used_for_generation=False,mesh_used_to_filter_exported_points=False,
            completeness_not_claimed=True,renderer_unchanged=True,source_nonopacity_gaussian_fields_exact=preserved,
            PNG_backprojection_pixel_centres=[0.5,0.5],PNG_roundtrip_verified=True,PLY_roundtrip_verified=True,
            analytic_unnormalized_ray_error_m=float(np.max(np.abs(analytic['tfar']+.2-2))),
            finite_logit=20.0,float32_sigmoid_opacity=1.0,fragment_alpha_cap=0.99),seconds=time.time()-start)
    save(metrics/'geometry_metrics.json',result)
    flat=[]
    for r in rows:
        flat.append(dict(method=r['method'],image_id=r['image_id'],coverage_percent=r['coverage_percent'],gap_percent=r['gap_percent'],
            point_mean_mm=r['point_to_input']['mean_mm'],point_median_mm=r['point_to_input']['median_mm'],
            point_p95_mm=r['point_to_input']['p95_mm'],point_rmse_mm=r['point_to_input']['rmse_mm'],
            own_depth_MAE_mm=r['own_valid_depth']['mean_mm'],common_depth_MAE_mm=r['common_valid_depth']['mean_mm'],
            common_depth_RMSE_mm=r['common_valid_depth']['rmse_mm'],
            common_behind_gt30mm_percent=r['common_valid_depth']['behind_first_surface_gt30mm_percent'],
            own_behind_gt30mm_percent=r['own_valid_depth']['behind_first_surface_gt30mm_percent'],
            interior_behind_gt30mm_percent=r['interior_depth']['behind_first_surface_gt30mm_percent']))
    with (metrics/'per_view.csv').open('x',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
    save(OUT/'verification.json',dict(status='PASS_FORMATS_COORDINATES_PROVENANCE_AND_ROUNDTRIPS',
        checked_views=10,depth_PNGs=20,backprojected_PLYs=22,cloud_sha256_unchanged=sha256(spec['input_cloud'])==spec['input_cloud_sha256'],
        source_geometry_unchanged=sha256(spec['source_geometry'])==geom_sha,
        source_gaussian_unchanged=sha256(spec['source_gaussian'])==spec['source_gaussian_sha256'],
        Gaussian_fields_other_than_opacity_exact=True,random_camera_selection_reproduced=True,
        occlusion_correctness_is_measured_not_assumed=True))
    print(json.dumps(dict(status=result['status'],aggregate=aggregate,seconds=time.time()-start)),flush=True)


if __name__=='__main__':main()
