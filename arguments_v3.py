"""One A/B preset plus tri-state overrides. No redundant enable/disable flags."""
import argparse
from experiment_presets_v3 import resolve_experiment_config

FEATURES = ('init_normal', 'init_flatten', 'orient_cameras', 'surface_loss',
            'tangent_loss', 'normal_loss', 'flatten_loss', 'size_loss', 'pruning',
            'scale_bounds', 'surface_densify', 'lidar_depth_loss',
            'validation_diagnostics')
LOSSES = ('surface', 'tangent', 'normal', 'flatten', 'size')

def parse_args(argv=None):
    from arguments import OptimizationParams, PipelineParams
    p = argparse.ArgumentParser(description='LiDAR 3DGS v3 E (C geometry plus occlusion-aware LiDAR depth)')
    op, pp = OptimizationParams(p), PipelineParams(p)
    p.add_argument('--experiment','--experiment_group',dest='experiment',
                   choices=['original','A','B','C','D','E','custom'],default='A',
                   help='Unified group preset; per-feature auto/on/off overrides it')
    for name in FEATURES:
        p.add_argument('--'+name, choices=['auto','on','off'], default='auto')
    p.add_argument('-s', '--source_path', required=True)
    p.add_argument('-m', '--model_path', required=True)
    p.add_argument('--point_cloud', required=True, help='LAS or PLY; never modified')
    p.add_argument('--train_file', required=True, help='Explicit COLMAP images text')
    p.add_argument('--val_file', required=True, help='Exactly ten fixed cameras, never resampled')
    p.add_argument('--cameras_file', required=True)
    p.add_argument('--test_file', default='', help='Explicit final test list; required for a 150k formal run')
    p.add_argument('--final_test', choices=['on','off'], default='on',
                   help='At a completed 150k run, evaluate every test camera and export worst ten')
    p.add_argument('--baseline_result', default='',
                   help='Optional completed baseline output used for metric deltas in the summary')
    p.add_argument('--data_manifest', default='',
                   help='Optional processed_v3 dataset manifest; content hash is recorded')
    p.add_argument('--preprocess_version', default='processed_v3-1',
                   help='Human-readable preprocessing protocol/version recorded in run_config')
    p.add_argument('--supervision_root', default='',
                   help='processed_v3 root containing supervision; default: source_path')
    p.add_argument('--images', default='images')
    p.add_argument('--alpha_masks', default='masks')
    p.add_argument('--resolution', type=int, default=1)
    p.add_argument('--sh_degree', type=int, default=3)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--lazy_cache', type=int, default=32)
    p.add_argument('--units', choices=['meters', 'scene'], default='scene')
    p.add_argument('--init_log_thickness', type=float, default=-10.)
    p.add_argument('--knn', type=int, default=16)
    p.add_argument('--neighbor_radius_ratio', type=float, default=3.)
    p.add_argument('--planarity_min', type=float, default=.2)
    p.add_argument('--surface_tolerance_ratio', type=float, default=.25)
    p.add_argument('--tangent_radius_ratio', type=float, default=1.)
    p.add_argument('--thickness_ratio', type=float, default=.1)
    p.add_argument('--size_ratio', type=float, default=2.)
    for name, weight in zip(LOSSES, [.1, .01, .05, .01, .01]):
        p.add_argument('--lambda_'+name, type=float, default=weight)
        p.add_argument('--'+name+'_start', type=int, default=0)
        p.add_argument('--'+name+'_warmup', type=int, default=500)
    p.add_argument('--surface_densify_start', type=int, default=10000)
    p.add_argument('--surface_densify_until', type=int, default=100000)
    p.add_argument('--surface_densify_interval', type=int, default=10000)
    p.add_argument('--surface_densify_grad_threshold', type=float, default=.0002)
    p.add_argument('--surface_densify_plane_ratio', type=float, default=.25)
    p.add_argument('--surface_densify_offset_ratio', type=float, default=.35)
    p.add_argument('--surface_densify_child_scale', type=float, default=.7)
    p.add_argument('--surface_densify_max_fraction', type=float, default=.01)
    p.add_argument('--surface_densify_max_points_ratio', type=float, default=1.25)
    p.add_argument('--surface_densify_min_opacity', type=float, default=.01)
    p.add_argument('--surface_densify_min_views', type=int, default=20)
    p.add_argument('--surface_densify_max_children_per_seed', type=int, default=2)
    p.add_argument('--prune_start', type=int, default=3000)
    p.add_argument('--prune_interval', type=int, default=1000)
    p.add_argument('--prune_opacity', type=float, default=.005)
    p.add_argument('--prune_min_views', type=int, default=20)
    p.add_argument('--prune_patience', type=int, default=3)
    p.add_argument('--prune_max_fraction', type=float, default=.01)
    p.add_argument('--log_interval', type=int, default=100)
    p.add_argument('--val_interval', type=int, default=5000)
    p.add_argument('--checkpoint_interval', type=int, default=50000)
    p.add_argument('--val_npz', choices=['on','off'], default='off',
                   help='Optional raw validation arrays; PNG and CSV output is unaffected')
    p.add_argument('--depth_visual_max', type=float, default=15.)
    p.add_argument('--val_rgb', choices=['auto','on','off'], default='auto')
    p.add_argument('--val_depth', choices=['auto','on','off'], default='auto')
    p.add_argument('--val_normal', choices=['auto','on','off'], default='auto')
    p.add_argument('--val_ellipsoids', choices=['auto','on','off'], default='auto',
                   help='Diagnostic opaque 1-sigma DC-color ellipsoids; no training effect')
    p.add_argument('--lidar_depth_cache', default='',
                   help='Local temporary cache directory; E requires it and it must not be on Drive')
    p.add_argument('--lidar_depth_export', default='',
                   help='New or already-verified Drive folder for pseudo-GT PNGs and validation reports')
    p.add_argument('--lambda_lidar_depth', type=float, default=.05)
    p.add_argument('--lidar_depth_start', type=int, default=1000)
    p.add_argument('--lidar_depth_warmup', type=int, default=4000)
    p.add_argument('--lidar_depth_min', type=float, default=.1)
    p.add_argument('--lidar_depth_max', type=float, default=15.)
    p.add_argument('--lidar_depth_splat_radius', type=int, default=1)
    p.add_argument('--lidar_depth_edge_relative', type=float, default=.02)
    p.add_argument('--lidar_depth_edge_absolute', type=float, default=.02)
    p.add_argument('--lidar_depth_min_neighbors', type=int, default=2)
    p.add_argument('--lidar_depth_alpha_min', type=float, default=.05)
    p.add_argument('--lidar_depth_min_pixels', type=int, default=64)
    p.add_argument('--lidar_depth_min_coverage', type=float, default=.001)
    p.add_argument('--lidar_depth_chunk', type=int, default=250000)
    p.add_argument('--lidar_depth_huber_beta', type=float, default=.02)
    p.add_argument('--lidar_depth_distance_power', type=float, default=1.)
    p.add_argument('--lidar_depth_weight_min', type=float, default=.25)
    p.add_argument('--lidar_depth_weight_max', type=float, default=4.)
    p.add_argument('--lidar_depth_backproject_samples', type=int, default=4096)
    p.add_argument('--lidar_depth_backproject_tolerance', type=float, default=.06)
    p.add_argument('--lidar_depth_backproject_quantile', type=float, default=.99)
    p.add_argument('--lidar_depth_backproject_min_fraction', type=float, default=.99)
    p.add_argument('--lidar_depth_reprojection_tolerance_px', type=float, default=2.)
    p.add_argument('--lidar_depth_cache_memory', type=int, default=32)
    p.add_argument('--resume', default='')
    # Keep upstream densification controls for the original compatibility group.
    # A-E ignore them unless surface_densify is explicitly enabled.
    p.add_argument('--paper_arm', choices=['A','B','C','D','E'], default='A')
    p.add_argument('--precomputed_root', default='')
    p.add_argument('--supervision_manifest', default='')
    p.add_argument('--supervision_start', type=int, default=1000)
    p.add_argument('--supervision_warmup', type=int, default=4000)
    p.add_argument('--metric_depth_weight', type=float, default=.1)
    p.add_argument('--external_normal_weight', type=float, default=.05)
    a = p.parse_args(argv)
    if a.paper_arm!='A' and not a.precomputed_root: p.error('Missing precomputed supervision')
    if min(a.supervision_start,a.supervision_warmup,a.metric_depth_weight,a.external_normal_weight)<0: p.error('Negative supervision parameter')
    if a.final_test=='on' and not a.test_file:
        p.error('--test_file is required when --final_test on')
    raw_overrides={n:getattr(a,n) for n in FEATURES}
    resolved=resolve_experiment_config(a.experiment,raw_overrides,{},FEATURES)
    a.overrides={n:v for n,v in raw_overrides.items() if v!='auto'}
    a.experiment_config=resolved
    for n,value in resolved['resolved_feature_flags'].items():
        setattr(a,n,value)
    for n in ('val_rgb','val_depth','val_normal','val_ellipsoids'):
        value=getattr(a,n)
        setattr(a,n,('on' if a.validation_diagnostics else 'off') if value=='auto' else value)
    for n in LOSSES:
        if getattr(a,'lambda_'+n)<0 or getattr(a,n+'_start')<0 or getattr(a,n+'_warmup')<0:
            p.error('loss weights and schedules must be nonnegative')
    for n in ('iterations','log_interval','val_interval','checkpoint_interval','prune_interval',
              'neighbor_radius_ratio','surface_tolerance_ratio','tangent_radius_ratio',
              'thickness_ratio','size_ratio','depth_visual_max','prune_patience','prune_min_views','lazy_cache',
              'lidar_depth_min','lidar_depth_max','lidar_depth_min_pixels','lidar_depth_chunk',
              'lidar_depth_huber_beta','lidar_depth_cache_memory','lidar_depth_min_neighbors',
              'lidar_depth_weight_min','lidar_depth_weight_max','lidar_depth_backproject_samples',
              'lidar_depth_backproject_tolerance','lidar_depth_reprojection_tolerance_px',
              'surface_densify_interval','surface_densify_plane_ratio','surface_densify_offset_ratio',
              'surface_densify_child_scale','surface_densify_max_fraction','surface_densify_max_points_ratio',
              'surface_densify_min_opacity','surface_densify_min_views','surface_densify_max_children_per_seed'):
        if getattr(a,n)<=0: p.error(n+' must be positive')
    if a.knn<3 or not 0<a.planarity_min<1 or not 0<a.prune_max_fraction<1:
        p.error('invalid neighborhood, confidence or pruning fraction')
    if not 0<a.prune_opacity<1 or a.prune_start<0: p.error('invalid pruning settings')
    if a.surface_densify_start<0 or a.surface_densify_until<a.surface_densify_start:
        p.error('invalid surface densification schedule')
    if a.surface_densify_grad_threshold<0 or not 0<a.surface_densify_max_fraction<1:
        p.error('invalid surface densification threshold/fraction')
    if a.surface_densify_max_points_ratio<1 or not 0<a.surface_densify_min_opacity<1:
        p.error('invalid surface densification growth/opacity limit')
    if a.lambda_lidar_depth<0 or a.lidar_depth_start<0 or a.lidar_depth_warmup<0:
        p.error('invalid LiDAR depth loss weight/schedule')
    if a.lidar_depth_max<=a.lidar_depth_min or a.lidar_depth_splat_radius<0:
        p.error('invalid LiDAR depth range/splat radius')
    if a.lidar_depth_edge_relative<0 or a.lidar_depth_edge_absolute<0 or a.lidar_depth_distance_power<0:
        p.error('invalid LiDAR depth edge thresholds')
    if not 0<a.lidar_depth_alpha_min<1:
        p.error('invalid LiDAR depth alpha threshold')
    if a.lidar_depth_weight_max<a.lidar_depth_weight_min:
        p.error('invalid LiDAR depth distance weight bounds')
    if not 0<a.lidar_depth_min_coverage<1:
        p.error('lidar_depth_min_coverage must be in (0,1)')
    if not 0<a.lidar_depth_backproject_quantile<=1:
        p.error('lidar_depth_backproject_quantile must be in (0,1]')
    if not 0<a.lidar_depth_backproject_min_fraction<=1:
        p.error('lidar_depth_backproject_min_fraction must be in (0,1]')
    if a.lidar_depth_loss and not a.lidar_depth_cache:
        p.error('--lidar_depth_cache is required when lidar_depth_loss is on')
    if a.lidar_depth_loss and a.iterations==150000 and not a.lidar_depth_export:
        p.error('--lidar_depth_export is required for a 150000-iteration E run')
    feature_params={
        'init_normal':{'knn':a.knn,'neighbor_radius_ratio':a.neighbor_radius_ratio,
                       'planarity_min':a.planarity_min},
        'init_flatten':{'init_log_thickness':a.init_log_thickness},
        'orient_cameras':{'camera_vote_count':8,'training_cameras_only':True},
        'surface_loss':{'lambda':a.lambda_surface,'start':a.surface_start,'warmup':a.surface_warmup,
                        'tolerance_ratio':a.surface_tolerance_ratio},
        'tangent_loss':{'lambda':a.lambda_tangent,'start':a.tangent_start,'warmup':a.tangent_warmup,
                        'radius_ratio':a.tangent_radius_ratio},
        'normal_loss':{'lambda':a.lambda_normal,'start':a.normal_start,'warmup':a.normal_warmup},
        'flatten_loss':{'lambda':a.lambda_flatten,'start':a.flatten_start,'warmup':a.flatten_warmup,
                        'thickness_ratio':a.thickness_ratio},
        'size_loss':{'lambda':a.lambda_size,'start':a.size_start,'warmup':a.size_warmup,
                     'size_ratio':a.size_ratio},
        'pruning':{'start':a.prune_start,'interval':a.prune_interval,'opacity':a.prune_opacity,
                   'min_views':a.prune_min_views,'patience':a.prune_patience,
                   'max_fraction':a.prune_max_fraction},
        'scale_bounds':{'thickness_ratio':a.thickness_ratio,'size_ratio':a.size_ratio},
        'surface_densify':{'start':a.surface_densify_start,'until':a.surface_densify_until,
            'interval':a.surface_densify_interval,'grad_threshold':a.surface_densify_grad_threshold,
            'plane_ratio':a.surface_densify_plane_ratio,'offset_ratio':a.surface_densify_offset_ratio,
            'child_scale':a.surface_densify_child_scale,'max_fraction':a.surface_densify_max_fraction,
            'max_points_ratio':a.surface_densify_max_points_ratio,
            'min_opacity':a.surface_densify_min_opacity,'min_views':a.surface_densify_min_views,
            'max_children_per_seed':a.surface_densify_max_children_per_seed},
        'lidar_depth_loss':{'lambda':a.lambda_lidar_depth,'start':a.lidar_depth_start,
            'warmup':a.lidar_depth_warmup,'min':a.lidar_depth_min,'max':a.lidar_depth_max,
            'distance_power':a.lidar_depth_distance_power,
            'weight_min':a.lidar_depth_weight_min,'weight_max':a.lidar_depth_weight_max},
        'validation_diagnostics':{'interval':a.val_interval,'rgb':a.val_rgb,
            'depth':a.val_depth,'normal':a.val_normal,'ellipsoid_1sigma':a.val_ellipsoids,
            'npz':a.val_npz},
    }
    a.active_feature_params={name:feature_params[name] for name in FEATURES
                             if getattr(a,name) and name in feature_params}
    a.experiment_config['enabled_feature_params']=a.active_feature_params
    a.optimizer_type='default'
    a.data_device='cpu'; a.lazy_load=True; a.train_test_exp=False
    # Deterministic comparison: black background, no opacity resets or growth.
    a.white_background=False
    return a, op.extract(a), pp.extract(a)
