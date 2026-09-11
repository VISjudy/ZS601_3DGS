"""Generate the concise, evidence-backed report required after a formal run."""
from pathlib import Path
import csv
import json
from arguments_v3 import preset_features

FEATURES=('init_normal','init_flatten','orient_cameras','surface_loss','tangent_loss',
          'normal_loss','flatten_loss','size_loss','pruning','scale_bounds','surface_densify')
UPSTREAM_DEFAULTS={
    'iterations':30000,'sh_degree':3,'position_lr_init':0.00016,
    'position_lr_final':0.0000016,'position_lr_max_steps':30000,
    'feature_lr':0.0025,'opacity_lr':0.025,'scaling_lr':0.005,
    'rotation_lr':0.001,'lambda_dssim':0.2,
}
KEYS=list(UPSTREAM_DEFAULTS)+[
    'seed','init_log_thickness','knn','planarity_min','neighbor_radius_ratio',
    'surface_tolerance_ratio','tangent_radius_ratio','thickness_ratio','size_ratio',
    'lambda_surface','lambda_tangent','lambda_normal','lambda_flatten','lambda_size',
    'prune_start','prune_interval','prune_opacity','prune_min_views',
    'prune_patience','prune_max_fraction','surface_densify_start',
    'surface_densify_until','surface_densify_interval',
    'surface_densify_grad_threshold','surface_densify_plane_ratio',
    'surface_densify_offset_ratio','surface_densify_child_scale',
    'surface_densify_max_fraction','surface_densify_max_points_ratio',
    'surface_densify_min_opacity','surface_densify_min_views',
    'surface_densify_max_children_per_seed','val_interval','checkpoint_interval',
]

def _fmt(value):
    return f'{value:.8g}' if isinstance(value,float) else str(value)

def _jsonl(path):
    path=Path(path)
    if not path.is_file(): return []
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]

def _val_rows(path):
    path=Path(path)
    if not path.is_file(): return []
    with path.open(newline='',encoding='utf-8') as f:
        return [r for r in csv.DictReader(f) if r.get('image_name')=='MEAN']

def _baseline_artifacts(path):
    if not path: return None
    given=Path(path)
    if (given/'test_summary.json').is_file():
        test_dir=given; root=given.parents[1]
    else:
        root=given; test_dir=root/'test_final'/'iteration_150000'
    summary=test_dir/'test_summary.json'; config=root/'run_config.json'
    if not summary.is_file() or not config.is_file():
        raise FileNotFoundError('baseline_result must contain run_config.json and test_final/iteration_150000/test_summary.json')
    return root,json.loads(config.read_text(encoding='utf-8')),json.loads(summary.read_text(encoding='utf-8'))

def write_experiment_summary(a,iteration,test_summary,test_dir):
    out=Path(a.model_path)
    active={k:bool(getattr(a,k)) for k in FEATURES}
    baseline=_baseline_artifacts(a.baseline_result)
    baseline_cfg=baseline_summary=None
    baseline_features=preset_features('A')
    feature_delta={k:(baseline_features[k],active[k]) for k in FEATURES
                   if baseline_features[k]!=active[k]}
    parameter_delta={}
    if baseline:
        _,baseline_cfg,baseline_summary=baseline
        if baseline_summary.get('comparison_identity')!=test_summary.get('comparison_identity'):
            raise ValueError('baseline_result uses a different input identity, ordered test list, or metric protocol')
        feature_delta={k:(bool(baseline_cfg.get(k)),active[k]) for k in FEATURES
                       if bool(baseline_cfg.get(k))!=active[k]}
        parameter_delta={k:(baseline_cfg.get(k),getattr(a,k)) for k in KEYS
                         if k in baseline_cfg and baseline_cfg.get(k)!=getattr(a,k)}
    mean=test_summary['aggregates']['mean']
    val_rows=_val_rows(out/'val_metrics.csv')
    final_val=val_rows[-1] if val_rows else None
    best_val=max(val_rows,key=lambda r:float(r['masked_psnr'])) if val_rows else None
    geometry_rows=_jsonl(out/'geometry_log.jsonl')
    final_geometry=geometry_rows[-1] if geometry_rows else None
    dense_rows=_jsonl(out/'surface_densify_log.jsonl')
    added=sum(int(r.get('added',0)) for r in dense_rows)
    lines=[
        f'# 实验 {a.experiment} 总结','',
        f'- 状态：完成 {iteration} 次迭代，并完成完整 test 集评估。',
        f'- 高斯数量：{mean["count"]}；test 相机数：{test_summary["camera_count"]}。',
        '- 最差相机排序：masked PSNR 升序，同分按 image_name；保存前10张的RGB、1σ彩色椭球、深度和法向量。',
        f'- val_npz={a.val_npz}；test固定不保存NPZ。','',
        '## 相对 baseline 的修改','',
    ]
    if baseline:
        lines.append(f'- baseline目录：{baseline[0]}。')
        if feature_delta:
            for key,(before,after) in feature_delta.items():
                lines.append(f'- 功能 {key}：{str(before).lower()} → {str(after).lower()}。')
        else:
            lines.append('- 功能开关与给定baseline相同。')
        if parameter_delta:
            lines.append('- 数值超参数变化：'+', '.join(
                f'{k}={_fmt(v[0])}→{_fmt(v[1])}' for k,v in parameter_delta.items())+'。')
    else:
        lines.append('- 未提供 baseline_result；本报告不猜测历史baseline配置，也不生成数值差。')
        lines.append('- D preset相对C只新增surface_densify；实际跨实验差异应以各自run_config.json复核。')
    lines += ['','当前启用功能：'+', '.join(k for k,v in active.items() if v)+'。','',
              '## 关键超参数','',
              '| 参数 | 本实验 | 原始默认值 | 说明 |','|---|---:|---:|---|']
    for key in KEYS:
        if not hasattr(a,key): continue
        value=getattr(a,key); default=UPSTREAM_DEFAULTS.get(key,'—')
        changed=key in UPSTREAM_DEFAULTS and value!=default
        note='与原始默认值不同' if changed else ('与原始默认值相同' if key in UPSTREAM_DEFAULTS else 'v3/D参数')
        lines.append(f'| {key} | {_fmt(value)} | {_fmt(default)} | {note} |')
    behavior='LiDAR贴面受控增密开启' if a.surface_densify else '增密关闭'
    lines += ['',
        f'固定运行行为：{behavior}；标准densify_and_prune和opacity reset关闭；checkpoint/PLY每50000步；固定val每{a.val_interval}步。','',
        '## 最终 test 结果','',
        '| 指标 | mean | median | min | max |','|---|---:|---:|---:|---:|']
    for key in ('masked_psnr','masked_mae','ssim_zero_mask_full_image'):
        ag=test_summary['aggregates']
        lines.append(f'| {key} | {_fmt(ag["mean"][key])} | {_fmt(ag["median"][key])} | {_fmt(ag["min"][key])} | {_fmt(ag["max"][key])} |')
    if baseline_summary:
        b=baseline_summary['aggregates']['mean']
        lines += ['','与同一输入及同一指标协议baseline的test mean差值：',
                  f'- masked PSNR：{mean["masked_psnr"]-b["masked_psnr"]:+.4f} dB。',
                  f'- masked MAE：{mean["masked_mae"]-b["masked_mae"]:+.6f}。',
                  f'- zero-mask full-image SSIM：{mean["ssim_zero_mask_full_image"]-b["ssim_zero_mask_full_image"]:+.6f}。']
    lines += ['','### 指标最差的10个相机','',
              '| 排名 | 相机 | masked PSNR | masked MAE | SSIM |','|---:|---|---:|---:|---:|']
    for row in test_summary['worst10']:
        lines.append(f'| {row["rank"]} | {row["image_name"]} | {row["masked_psnr"]:.4f} | {row["masked_mae"]:.6f} | {row["ssim_zero_mask_full_image"]:.6f} |')
    lines += ['','## 结果分析','',
              f'- 完整test平均masked PSNR为{mean["masked_psnr"]:.4f} dB；最差相机是{test_summary["worst10"][0]["image_name"]}（{test_summary["worst10"][0]["masked_psnr"]:.4f} dB）。',
              '- masked PSNR/MAE按有效mask像素数归一化；SSIM是置零mask后的整图SSIM。',
              '- 深度图是opacity归一化的高斯中心相机z，仅用于诊断，不能作为LiDAR深度真值。',
              '- PSNR不能单独证明贴面；需同时检查1σ椭球图和geometry_log.jsonl。']
    if final_val:
        lines.append(f'- 最终固定val：PSNR={float(final_val["masked_psnr"]):.4f} dB；最佳固定val：PSNR={float(best_val["masked_psnr"]):.4f} dB@iter {best_val["iteration"]}。')
    if final_geometry:
        lines.append(f'- 最终几何：size_exceed_fraction_tol={final_geometry.get("size_exceed_fraction_tol")}，thickness_exceed_fraction_tol={final_geometry.get("thickness_exceed_fraction_tol")}。')
        lines.append(f'- 最终平面距离分位数={final_geometry.get("abs_plane_distance_valid")}；法向角分位数={final_geometry.get("normal_angle_deg_valid")}。')
    if a.surface_densify:
        initial=int(dense_rows[0].get('count',mean['count'])) if dense_rows else mean['count']
        coverage=dense_rows[-1].get('seed_source_coverage') if dense_rows else None
        lines.append(f'- 受控增密事件={len(dense_rows)}，累计新增={added}，最终种子覆盖率={coverage}；逐事件证据见surface_densify_log.jsonl。')
    rel=Path(test_dir).relative_to(out)
    lines += ['','## 产物',
              f'- 完整test指标：{rel / "test_metrics.csv"}',
              f'- 最差10相机诊断：{rel}',
              '- 训练损失：loss_log.csv',
              '- 固定验证指标：val_metrics.csv',
              '- 几何日志：geometry_log.jsonl',
              '- 受控增密日志：surface_densify_log.jsonl']
    report=out/'experiment_summary.md'
    report.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(f'[SUMMARY] {report}',flush=True)
    return report
