"""One A/B preset plus tri-state overrides. No redundant enable/disable flags."""
import argparse

FEATURES = ('init_normal', 'init_flatten', 'orient_cameras', 'surface_loss',
            'tangent_loss', 'normal_loss', 'flatten_loss', 'size_loss', 'pruning', 'scale_bounds',
            'lidar_depth_loss')
LOSSES = ('surface', 'tangent', 'normal', 'flatten', 'size')

def preset_features(experiment):
    enabled={'init_normal','init_flatten','orient_cameras','pruning'}
    if experiment in ('B','C','E'): enabled.update(name+'_loss' for name in LOSSES)
    if experiment in ('C','E'): enabled.add('scale_bounds')
    if experiment=='E': enabled.add('lidar_depth_loss')
    return {name:name in enabled for name in FEATURES}

def parse_args(argv=None):
    from arguments import OptimizationParams, PipelineParams
    p = argparse.ArgumentParser(description='LiDAR 3DGS v3 E (C geometry plus occlusion-aware LiDAR depth)')
    op, pp = OptimizationParams(p), PipelineParams(p)
    p.add_argument('--experiment', choices=['A', 'B', 'C', 'E'], default='A')
    for name in FEATURES:
        p.add_argument('--'+name, choices=['on', 'off'], default=None)
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
    p.add_argument('--val_ellipsoids', choices=['on','off'], default='on',
                   help='Diagnostic opaque 1-sigma DC-color ellipsoids; no training effect')
    p.add_argument('--lidar_depth_cache', default='',
                   help='Local temporary cache directory; E requires it and it must not be on Drive')
    p.add_argument('--lidar_depth_export', default='',
                   help='Brand-new Drive dataset folder for formal pseudo-GT PNGs and validation reports')
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
    p.add_argument('--lidar_depth_chunk', type=int, default=250000)
    p.add_argument('--lidar_depth_huber_beta', type=float, default=.02)
    p.add_argument('--lidar_depth_distance_power', type=float, default=1.)
    p.add_argument('--lidar_depth_weight_min', type=float, default=.25)
    p.add_argument('--lidar_depth_weight_max', type=float, default=4.)
    p.add_argument('--lidar_depth_backproject_samples', type=int, default=4096)
    p.add_argument('--lidar_depth_backproject_tolerance', type=float, default=.06)
    p.add_argument('--lidar_depth_cache_memory', type=int, default=32)
    p.add_argument('--resume', default='')
    # Remove inherited controls unused by the v3 fixed-population runner.
    removed = {'densification_interval','opacity_reset_interval','densify_from_iter',
               'densify_until_iter','densify_grad_threshold','depth_l1_weight_init',
               'depth_l1_weight_final','random_background','optimizer_type'}
    for action in list(p._actions):
        if action.dest in removed:
            p._remove_action(action)
            for group in p._action_groups:
                if action in group._group_actions: group._group_actions.remove(action)
            for option in action.option_strings: p._option_string_actions.pop(option, None)
    a = p.parse_args(argv)
    if a.final_test=='on' and a.iterations==150000 and not a.test_file:
        p.error('--test_file is required when --final_test on for a 150000-iteration formal run')
    a.overrides = {n:getattr(a,n) for n in FEATURES if getattr(a,n) is not None}
    preset=preset_features(a.experiment)
    for n in FEATURES:
        setattr(a,n,preset[n] if getattr(a,n) is None else getattr(a,n)=='on')
    for n in LOSSES:
        if getattr(a,'lambda_'+n)<0 or getattr(a,n+'_start')<0 or getattr(a,n+'_warmup')<0:
            p.error('loss weights and schedules must be nonnegative')
    for n in ('iterations','log_interval','val_interval','checkpoint_interval','prune_interval',
              'neighbor_radius_ratio','surface_tolerance_ratio','tangent_radius_ratio',
              'thickness_ratio','size_ratio','depth_visual_max','prune_patience','prune_min_views','lazy_cache',
              'lidar_depth_min','lidar_depth_max','lidar_depth_min_pixels','lidar_depth_chunk',
              'lidar_depth_huber_beta','lidar_depth_cache_memory','lidar_depth_min_neighbors',
              'lidar_depth_weight_min','lidar_depth_weight_max','lidar_depth_backproject_samples',
              'lidar_depth_backproject_tolerance'):
        if getattr(a,n)<=0: p.error(n+' must be positive')
    if a.knn<3 or not 0<a.planarity_min<1 or not 0<a.prune_max_fraction<1:
        p.error('invalid neighborhood, confidence or pruning fraction')
    if not 0<a.prune_opacity<1 or a.prune_start<0: p.error('invalid pruning settings')
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
    if a.lidar_depth_loss and not a.lidar_depth_cache:
        p.error('--lidar_depth_cache is required when lidar_depth_loss is on')
    if a.lidar_depth_loss and a.iterations==150000 and not a.lidar_depth_export:
        p.error('--lidar_depth_export is required for a 150000-iteration E run')
    a.optimizer_type='default'
    a.data_device='cpu'; a.lazy_load=True; a.train_test_exp=False
    # Deterministic comparison: black background, no opacity resets or growth.
    a.white_background=False
    return a, op.extract(a), pp.extract(a)
