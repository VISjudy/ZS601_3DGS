"""A/B training derived from the main baseline; all v3 code is additive."""
import csv,json,random,time
from pathlib import Path
import numpy as np
import torch
from arguments_v3 import parse_args,FEATURES
from scale_bounds_v3 import apply_scale_bounds
from surface_densify_v3 import accumulate_surface_gradient,surface_densify
from geometry_v3 import build_reference,tensor_reference,normals_to_quaternions,geometry_losses,diagnostics
from runtime_v3 import (append_csv,append_json,provenance,fresh_topology,finish_epoch,prune,check_finite,
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
          'overrides':a.overrides,'resolved_config':a.experiment_config,
          'active_feature_params':a.active_feature_params,
          'growth':('UPSTREAM 3DGS' if a.experiment=='original' else
                    ('SURFACE CONTROLLED' if a.surface_densify else 'OFF')),
          'opacity_reset':('UPSTREAM 3DGS' if a.experiment=='original' else 'OFF'),
          'depth_loss':'LIDAR CAMERA-Z' if a.lidar_depth_loss else 'OFF',
          'hard_scale_bounds':a.scale_bounds,'seed':a.seed,'val_ellipsoids':a.val_ellipsoids,
          'ellipsoid_sigma':1,'ellipsoid_color':'SH DC + lighting','loss_csv':'every iteration',
          'val_npz':a.val_npz,'checkpoint_interval':a.checkpoint_interval,
          'final_test':a.final_test,'final_test_iteration':150000,'test_worst_count':10},indent=2),flush=True)
    (out/'run_config.json').write_text(json.dumps(vars(a),indent=2),encoding='utf-8')
    (out/'environment.json').write_text(json.dumps(code,indent=2),encoding='utf-8')
    from data_v3 import load_data
    from scene.gaussian_model import GaussianModel
    from gaussian_renderer import render
    from utils.loss_utils import l1_loss,ssim
    from render_v3 import export_val,export_final_test
    from experiment_summary_v3 import write_experiment_summary
    pcd,infos,cameras,val_cameras,test_cameras,centers,extent,identity=load_data(a)
    from lidar_depth_v3 import LidarDepthProvider,lidar_depth_term
    depth_provider=LidarDepthProvider(pcd.points,a,identity) if a.lidar_depth_loss else None
    depth_dataset=None
    if depth_provider is not None and a.lidar_depth_export:
        try:
            depth_dataset=depth_provider.prepare_and_export(
                {'train':cameras,'val':val_cameras,'test':test_cameras},
                a.lidar_depth_export)
        except BaseException as error:
            (out/'failure.json').write_text(json.dumps({
                'iteration':0,'stage':'lidar_depth_preflight','error':repr(error),
                'note':'Training did not start. Use a new export folder after fixing the input or thresholds.'
            },indent=2),encoding='utf-8')
            raise
    run_manifest={'code':code,'inputs':identity,
        'train_names':[c.image_name for c in cameras],
        'val_names':[c.image_name for c in val_cameras],
        'test_names':[c.image_name for c in test_cameras],
        'lidar_depth_dataset':{
            'path':str(Path(a.lidar_depth_export).expanduser().resolve()) if a.lidar_depth_export else None,
            'summary':depth_dataset}}
    (out/'run_manifest.json').write_text(json.dumps(run_manifest,indent=2),encoding='utf-8')
    g=GaussianModel(a.sh_degree,'default',False,-10.,False)
    g.create_from_pcd(pcd,infos,extent)
    ref_np=build_reference(pcd.points,centers,a)
    seed_ref_np=ref_np
    seed_xyz_np=np.asarray(pcd.points,dtype=np.float32)
    ref=tensor_reference(ref_np,'cuda')
    with torch.no_grad():
        if a.init_normal: g._rotation.copy_(torch.from_numpy(normals_to_quaternions(ref_np['normal'])).cuda())
        if a.init_flatten: g._scaling[:,2]=a.init_log_thickness
    if a.scale_bounds and not a.resume:
        bounds=apply_scale_bounds(g,ref,a)
        event={'iteration':0,'phase':'initialization',**{k:v.item() for k,v in bounds.items()}}
        append_json(out/'scale_bounds_log.jsonl',event); print('[SCALE BOUNDS]',event,flush=True)
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
        if a.validation_diagnostics:
            export_val(a,0,val_cameras,g,pipe)
    background=torch.zeros(3,device='cuda'); started=time.monotonic()
    iteration=first
    csv_fields=['iteration','rgb_l1','rgb_dssim','total','count','elapsed']
    csv_fields += [name+'_'+kind for name in ('surface','tangent','normal','flatten','size','lidar_depth')
                   for kind in ('raw','weight','weighted')]
    csv_fields += ['lidar_depth_valid_pixels','lidar_depth_lidar_pixels',
                   'lidar_depth_rendered_fraction','lidar_depth_mean_target_depth',
                   'lidar_depth_distance_weight_mean','lidar_depth_unweighted_raw',
                   'lidar_depth_distance_weighted_raw','lidar_depth_state']
    loss_file=(out/'loss_log.csv').open('x',newline='',encoding='utf-8')
    loss_writer=csv.DictWriter(loss_file,fieldnames=csv_fields); loss_writer.writeheader()
    final_test_summary=None; final_test_dir=None
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
            depth_loss,depth_stats=lidar_depth_term(
                cam,g,pipe,depth_provider,a,iteration)
            terms['lidar_depth']=depth_stats
            loss=rgb_loss+geo+depth_loss
            if not torch.isfinite(loss): raise FloatingPointError('Nonfinite total loss')
            loss.backward()
            if a.surface_densify: accumulate_surface_gradient(g,pkg)
            check_finite(g,gradients=True)
            with torch.no_grad():
                state['epoch_views']+=(pkg['radii']>0).to(torch.int32)
                upstream_topology_changed=False
                if a.experiment=='original' and iteration<opt.densify_until_iter:
                    visible=pkg['visibility_filter']
                    g.max_radii2D[visible]=torch.maximum(g.max_radii2D[visible],pkg['radii'][visible])
                    g.add_densification_stats(pkg['viewspace_points'],visible)
                    if iteration>opt.densify_from_iter and iteration%opt.densification_interval==0:
                        size_threshold=20 if iteration>opt.opacity_reset_interval else None
                        g.densify_and_prune(opt.densify_grad_threshold,0.005,extent,size_threshold,pkg['radii'])
                        upstream_topology_changed=True
                        print(f'[UPSTREAM DENSIFY] iteration={iteration} count={len(g.get_xyz)}',flush=True)
                    if iteration%opt.opacity_reset_interval==0:
                        g.reset_opacity()
                        print(f'[UPSTREAM OPACITY RESET] iteration={iteration}',flush=True)
                # Upstream skips the final optimizer step. Existing A-E behavior is preserved.
                if a.experiment!='original' or iteration<a.iterations:
                    if a.experiment=='original':
                        g.exposure_optimizer.step(); g.exposure_optimizer.zero_grad(set_to_none=True)
                    g.optimizer.step()
                g.optimizer.zero_grad(set_to_none=True)
                if upstream_topology_changed:
                    from scipy.spatial import cKDTree
                    current=g.get_xyz.detach().cpu().numpy()
                    nearest=cKDTree(seed_xyz_np).query(current,k=1,workers=-1)[1]
                    mapped={k:(v[nearest] if len(v)==len(seed_xyz_np) else v) for k,v in seed_ref_np.items()}
                    ref=tensor_reference(mapped,'cuda')
                    state=fresh_topology(g)
                if a.scale_bounds:
                    bounds=apply_scale_bounds(g,ref,a)
                    if iteration==1 or iteration%a.log_interval==0:
                        event={'iteration':iteration,'phase':'post_optimizer',**{k:v.item() for k,v in bounds.items()}}
                        append_json(out/'scale_bounds_log.jsonl',event); print('[SCALE BOUNDS]',event,flush=True)
                # Exposure is deliberately disabled for both groups, consistent with baseline default.
                check_finite(g)
                ref,state,event=prune(g,ref,state,a,iteration)
                if event: append_json(out/'prune_log.jsonl',event); print('[PRUNE]',event,flush=True)
                ref,state,dense_event=surface_densify(g,ref,state,a,iteration)
                if dense_event:
                    append_json(out/'surface_densify_log.jsonl',dense_event)
                    print('[SURFACE DENSIFY]',dense_event,flush=True)
                    if a.scale_bounds and dense_event['added']:
                        bounds=apply_scale_bounds(g,ref,a)
                        bound_event={'iteration':iteration,'phase':'post_surface_densify',
                            **{k:v.item() for k,v in bounds.items()}}
                        append_json(out/'scale_bounds_log.jsonl',bound_event)
                        print('[SCALE BOUNDS]',bound_event,flush=True)
                check_finite(g)
            record={'iteration':iteration,'rgb_l1':float(rgb_l1.detach()),'rgb_dssim':float(rgb_ssim.detach()),
                    'total':float(loss.detach()),'losses':terms,'count':len(g.get_xyz),
                    'elapsed':prior_elapsed+time.monotonic()-started}
            csv_record={k:v for k,v in record.items() if k!='losses'}
            csv_record.update({name+'_'+kind:terms[name][kind] for name in terms
                               for kind in ('raw','weight','weighted')})
            csv_record.update({'lidar_depth_valid_pixels':depth_stats['valid_pixels'],
                'lidar_depth_lidar_pixels':depth_stats['lidar_pixels'],
                'lidar_depth_rendered_fraction':depth_stats['rendered_fraction'],
                'lidar_depth_mean_target_depth':depth_stats['mean_target_depth'],
                'lidar_depth_distance_weight_mean':depth_stats['distance_weight_mean'],
                'lidar_depth_unweighted_raw':depth_stats['unweighted_raw'],
                'lidar_depth_distance_weighted_raw':depth_stats['distance_weighted_raw'],
                'lidar_depth_state':depth_stats['state']})
            loss_writer.writerow(csv_record)
            if iteration==1 or iteration%a.log_interval==0:
                loss_file.flush()
                append_json(out/'train_log.jsonl',record)
                append_csv(out/'training_progress.csv',{'iteration':iteration,'count':len(g.get_xyz),
                    'elapsed_seconds':record['elapsed'],
                    'gpu_memory_allocated_mb':float(torch.cuda.memory_allocated()/1048576),
                    'gpu_memory_reserved_mb':float(torch.cuda.memory_reserved()/1048576)})
                print('[TRAIN]',json.dumps(record),flush=True)
            if iteration%a.val_interval==0 or iteration==a.iterations:
                stat=diagnostics(g.get_xyz,g.get_scaling,g.get_rotation,ref,a)
                append_json(out/'geometry_log.jsonl',{'iteration':iteration,**stat})
                append_csv(out/'geometry_metrics.csv',{'iteration':iteration,'count':stat['count'],
                    'reference_valid_fraction':stat['reference_valid_fraction'],
                    'abs_plane_distance_p50':stat['abs_plane_distance_valid'][0] if stat['abs_plane_distance_valid'] else '',
                    'abs_plane_distance_p95':stat['abs_plane_distance_valid'][1] if stat['abs_plane_distance_valid'] else '',
                    'normal_angle_deg_p50':stat['normal_angle_deg_valid'][0] if stat['normal_angle_deg_valid'] else '',
                    'normal_angle_deg_p95':stat['normal_angle_deg_valid'][1] if stat['normal_angle_deg_valid'] else '',
                    'thickness_p95':stat['thickness'][1] if stat['thickness'] else '',
                    'max_tangent_scale_p95':stat['max_tangent_scale'][1] if stat['max_tangent_scale'] else '',
                    'size_exceed_fraction':stat['size_exceed_fraction'],
                    'thickness_exceed_fraction':stat['thickness_exceed_fraction']})
                print('[GEOMETRY]',stat,flush=True)
                if a.validation_diagnostics:
                    export_val(a,iteration,val_cameras,g,pipe)
            if iteration%a.checkpoint_interval==0 or iteration==a.iterations:
                g.save_ply(str(out/'point_cloud'/f'iteration_{iteration}'/'point_cloud.ply'))
                save_checkpoint(out/'checkpoints'/f'iteration_{iteration}.pth',g,ref,state,sampler,
                    iteration,a,identity,code,prior_elapsed+time.monotonic()-started)
        if a.final_test=='on' and iteration==150000:
            final_test_summary,final_test_dir=export_final_test(
                a,iteration,test_cameras,g,pipe,identity)
            write_experiment_summary(a,iteration,final_test_summary,final_test_dir)
        else:
            print(f'[FINAL TEST] skipped: enabled={a.final_test} completed_iteration={iteration}; formal contract requires 150000',flush=True)
    except BaseException as error:
        (out/'failure.json').write_text(json.dumps({'iteration':iteration,'error':repr(error),
            'note':'Resume only a completed checkpoint; partial step is not saved as valid.'},indent=2),encoding='utf-8')
        raise
    finally:
        loss_file.close()
    (out/'completed.json').write_text(json.dumps({'iteration':iteration,'count':len(g.get_xyz),
        'final_test_complete':final_test_summary is not None,
        'final_test_dir':str(final_test_dir) if final_test_dir else None,
        'summary':str(out/'experiment_summary.md') if final_test_summary is not None else None,
        'lidar_depth_dataset':run_manifest['lidar_depth_dataset']},indent=2),encoding='utf-8')
    print('[DONE]',out,flush=True)

if __name__=='__main__': main()
