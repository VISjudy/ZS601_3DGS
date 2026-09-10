"""A/B training derived from the main baseline; all v3 code is additive."""
import csv,json,random,time
from pathlib import Path
import numpy as np
import torch
from arguments_v3 import parse_args,FEATURES
from geometry_v3 import build_reference,tensor_reference,normals_to_quaternions,geometry_losses,diagnostics
from runtime_v3 import (append_json,provenance,fresh_topology,finish_epoch,prune,check_finite,
                        save_checkpoint,restore_checkpoint)

def main(argv=None):
    a,opt,pipe=parse_args(argv)
    if not torch.cuda.is_available(): raise RuntimeError('Training requires a Colab/NVIDIA CUDA GPU')
    out=Path(a.model_path)
    # Every run/resume gets a new output directory; original artifacts are retained.
    out.mkdir(parents=True,exist_ok=False)
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed)
    code=provenance()
    print('[RUN]',json.dumps({'experiment':a.experiment,'features':{n:getattr(a,n) for n in FEATURES},
          'overrides':a.overrides,'growth':'OFF (v3 A/B)','opacity_reset':'OFF','depth_loss':'OFF',
          'hard_thickness_reset':'OFF','seed':a.seed,'val_ellipsoids':a.val_ellipsoids,
          'ellipsoid_sigma':1,'ellipsoid_color':'SH DC + lighting','loss_csv':'every iteration',
          'val_npz':a.val_npz,'checkpoint_interval':a.checkpoint_interval},indent=2),flush=True)
    (out/'run_config.json').write_text(json.dumps(vars(a),indent=2),encoding='utf-8')
    from data_v3 import load_data
    from scene.gaussian_model import GaussianModel
    from gaussian_renderer import render
    from utils.loss_utils import l1_loss,ssim
    from render_v3 import export_val
    pcd,infos,cameras,val_cameras,centers,extent,identity=load_data(a)
    (out/'run_manifest.json').write_text(json.dumps({'code':code,'inputs':identity,
        'train_names':[c.image_name for c in cameras],'val_names':[c.image_name for c in val_cameras]},indent=2),encoding='utf-8')
    g=GaussianModel(a.sh_degree,'default',False,-10.,False)
    g.create_from_pcd(pcd,infos,extent)
    ref_np=build_reference(pcd.points,centers,a)
    ref=tensor_reference(ref_np,'cuda')
    with torch.no_grad():
        if a.init_normal: g._rotation.copy_(torch.from_numpy(normals_to_quaternions(ref_np['normal'])).cuda())
        if a.init_flatten: g._scaling[:,2]=a.init_log_thickness
    g.training_setup(opt)
    state=fresh_topology(g); sampler=[]; first=0; prior_elapsed=0.
    if a.resume:
        ref,state,sampler,first,prior_elapsed=restore_checkpoint(a.resume,g,a,identity,code)
    print('[GEOMETRY CONFIG]',json.dumps({k:v for k,v in vars(a).items() if
        k.startswith('lambda_') or k.endswith(('_ratio','_start','_warmup'))},indent=2),flush=True)
    if (a.surface_loss or a.normal_loss) and not (ref['confidence']>0).any():
        raise ValueError('No reliable references for enabled surface/normal loss')
    check_finite(g)
    stat=diagnostics(g.get_xyz,g.get_scaling,g.get_rotation,ref,a)
    append_json(out/'geometry_log.jsonl',{'iteration':first,**stat}); print('[INIT]',stat,flush=True)
    if not a.resume:
        np.savez_compressed(out/'reference_v3.npz',**ref_np)
        g.save_ply(str(out/'point_cloud'/'iteration_0'/'point_cloud.ply'))
        export_val(a,0,val_cameras,g,pipe)
    background=torch.zeros(3,device='cuda'); started=time.monotonic()
    iteration=first
    csv_fields=['iteration','rgb_l1','rgb_dssim','total','count','elapsed']
    csv_fields += [name+'_'+kind for name in ('surface','tangent','normal','flatten','size')
                   for kind in ('raw','weight','weighted')]
    loss_file=(out/'loss_log.csv').open('x',newline='',encoding='utf-8')
    loss_writer=csv.DictWriter(loss_file,fieldnames=csv_fields); loss_writer.writeheader()
    try:
        for iteration in range(first+1,a.iterations+1):
            g.update_learning_rate(iteration)
            if iteration%1000==0: g.oneupSHdegree()
            for retry in range(max(100,len(cameras)*2)):
                if not sampler:
                    finish_epoch(state); sampler=list(range(len(cameras))); random.shuffle(sampler)
                cam=cameras[sampler.pop()]
                mask=cam.alpha_mask.cuda() if cam.alpha_mask is not None else None
                if mask is None or float(mask.sum())>0: break
            else: raise ValueError('Could not sample a camera with valid image pixels')
            pkg=render(cam,g,pipe,background)
            image=pkg['render'] if mask is None else pkg['render']*mask
            gt=cam.original_image.cuda()
            rgb_l1=l1_loss(image,gt); rgb_ssim=1-ssim(image,gt)
            rgb_loss=(1-opt.lambda_dssim)*rgb_l1+opt.lambda_dssim*rgb_ssim
            geo,terms=geometry_losses(g.get_xyz,g.get_scaling,g.get_rotation,ref,a,iteration)
            loss=rgb_loss+geo
            if not torch.isfinite(loss): raise FloatingPointError('Nonfinite total loss')
            loss.backward()
            check_finite(g,gradients=True)
            with torch.no_grad():
                state['epoch_views']+=(pkg['radii']>0).to(torch.int32)
                g.optimizer.step(); g.optimizer.zero_grad(set_to_none=True)
                # Exposure is deliberately disabled for both groups, consistent with baseline default.
                check_finite(g)
                ref,state,event=prune(g,ref,state,a,iteration)
                if event: append_json(out/'prune_log.jsonl',event); print('[PRUNE]',event,flush=True)
            record={'iteration':iteration,'rgb_l1':float(rgb_l1.detach()),'rgb_dssim':float(rgb_ssim.detach()),
                    'total':float(loss.detach()),'losses':terms,'count':len(g.get_xyz),
                    'elapsed':prior_elapsed+time.monotonic()-started}
            csv_record={k:v for k,v in record.items() if k!='losses'}
            csv_record.update({name+'_'+kind:terms[name][kind] for name in terms
                               for kind in ('raw','weight','weighted')})
            loss_writer.writerow(csv_record)
            if iteration==1 or iteration%a.log_interval==0:
                loss_file.flush()
                append_json(out/'train_log.jsonl',record); print('[TRAIN]',json.dumps(record),flush=True)
            if iteration%a.val_interval==0 or iteration==a.iterations:
                stat=diagnostics(g.get_xyz,g.get_scaling,g.get_rotation,ref,a)
                append_json(out/'geometry_log.jsonl',{'iteration':iteration,**stat}); print('[GEOMETRY]',stat,flush=True)
                export_val(a,iteration,val_cameras,g,pipe)
            if iteration%a.checkpoint_interval==0 or iteration==a.iterations:
                g.save_ply(str(out/'point_cloud'/f'iteration_{iteration}'/'point_cloud.ply'))
                save_checkpoint(out/'checkpoints'/f'iteration_{iteration}.pth',g,ref,state,sampler,
                    iteration,a,identity,code,prior_elapsed+time.monotonic()-started)
    except BaseException as error:
        (out/'failure.json').write_text(json.dumps({'iteration':iteration,'error':repr(error),
            'note':'Resume only a completed checkpoint; partial step is not saved as valid.'},indent=2),encoding='utf-8')
        raise
    finally:
        loss_file.close()
    (out/'completed.json').write_text(json.dumps({'iteration':iteration,'count':len(g.get_xyz)}),encoding='utf-8')
    print('[DONE]',out,flush=True)

if __name__=='__main__': main()
