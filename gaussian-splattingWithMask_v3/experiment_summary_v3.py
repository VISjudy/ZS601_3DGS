"""Generate the concise, source-grounded report required after a formal run."""
from pathlib import Path
import csv
import json

UPSTREAM_DEFAULTS={
    'iterations':30000,'sh_degree':3,
    'position_lr_init':0.00016,'position_lr_final':0.0000016,
    'position_lr_max_steps':30000,'feature_lr':0.0025,'opacity_lr':0.025,
    'scaling_lr':0.005,'rotation_lr':0.001,'lambda_dssim':0.2,
}
KEYS=list(UPSTREAM_DEFAULTS)+[
    'seed','init_log_thickness','knn','planarity_min','neighbor_radius_ratio',
    'surface_tolerance_ratio','tangent_radius_ratio','thickness_ratio','size_ratio',
    'lambda_surface','lambda_tangent','lambda_normal','lambda_flatten','lambda_size',
    'prune_start','prune_interval','prune_opacity','prune_min_views',
    'prune_patience','prune_max_fraction','surface_densify_start','surface_densify_until',
    'surface_densify_interval','surface_densify_grad_threshold','surface_densify_plane_ratio',
    'surface_densify_offset_ratio','surface_densify_child_scale','surface_densify_max_fraction',
    'surface_densify_max_points_ratio','surface_densify_min_opacity','surface_densify_min_views',
    'val_interval','checkpoint_interval',
]
A_FEATURES={
    'init_normal':True,'init_flatten':True,'orient_cameras':True,
    'surface_loss':False,'tangent_loss':False,'normal_loss':False,
    'flatten_loss':False,'size_loss':False,'pruning':True,'scale_bounds':False,
    'surface_densify':False,
}

def _fmt(v):
    if isinstance(v,float): return f'{v:.8g}'
    return str(v)

def _last_csv_mean(path):
    path=Path(path)
    if not path.is_file(): return None
    answer=None
    with path.open(newline='',encoding='utf-8') as f:
        for row in csv.DictReader(f):
            if row.get('image_name')=='MEAN': answer=row
    return answer

def _last_jsonl(path):
    path=Path(path)
    if not path.is_file(): return None
    answer=None
    with path.open(encoding='utf-8') as f:
        for line in f:
            if line.strip(): answer=json.loads(line)
    return answer

def _baseline_summary(path):
    if not path: return None
    direct=Path(path)/'test_final'/'iteration_150000'/'test_summary.json'
    if direct.is_file(): return json.loads(direct.read_text(encoding='utf-8'))
    return None

def write_experiment_summary(a,iteration,test_summary,test_dir):
    out=Path(a.model_path)
    active={k:bool(getattr(a,k)) for k in A_FEATURES}
    feature_delta={k:(A_FEATURES[k],active[k]) for k in A_FEATURES if A_FEATURES[k]!=active[k]}
    baseline=_baseline_summary(a.baseline_result)
    mean=test_summary['aggregates']['mean']
    val=_last_csv_mean(out/'val_metrics.csv')
    geometry=_last_jsonl(out/'geometry_log.jsonl')
    lines=[
        f'# 实验 {a.experiment} 总结','',
        f'- 状态：完成 {iteration} 次迭代，并完成完整 test 集评估。',
        f'- 高斯数量：{mean["count"]}。',
        f'- test 相机数：{test_summary["camera_count"]}。',
        '- 最差相机排序：masked PSNR 升序，同分按 image_name；保存前 10 张的 RGB、1σ 彩色椭球、深度和法向量。',
        '- test 与 val 均不保存 geometry.npz。','',
        '## 相对 baseline A 的修改','',
    ]
    if not feature_delta:
        lines.append('- 本组即 A preset；功能开关与 baseline A 相同。')
    else:
        for key,(before,after) in feature_delta.items():
            lines.append(f'- {key}：{str(before).lower()} → {str(after).lower()}。')
    lines += ['','当前启用功能：'+', '.join(k for k,v in active.items() if v)+'。','',
              '## 关键超参数','',
              '| 参数 | 本实验 | 原始默认值 | 说明 |','|---|---:|---:|---|']
    for key in KEYS:
        if not hasattr(a,key): continue
        value=getattr(a,key); default=UPSTREAM_DEFAULTS.get(key,'—')
        changed=key in UPSTREAM_DEFAULTS and value!=default
        note='与原始默认值不同' if changed else ('与原始默认值相同' if key in UPSTREAM_DEFAULTS else 'v3 几何/运行参数')
        lines.append(f'| {key} | {_fmt(value)} | {_fmt(default)} | {note} |')
    lines += ['','固定运行行为：D 仅允许贴面受控增密；标准 densify_and_prune 与 opacity reset 关闭；checkpoint/PLY 每 50000 步；固定 val 每 1000 步。','',
              '## 最终 test 结果','',
              '| 指标 | mean | median | min | max |','|---|---:|---:|---:|---:|']
    for key in ('masked_psnr','masked_mae','ssim_zero_mask_full_image'):
        ag=test_summary['aggregates']
        lines.append(f'| {key} | {_fmt(ag["mean"][key])} | {_fmt(ag["median"][key])} | {_fmt(ag["min"][key])} | {_fmt(ag["max"][key])} |')
    if baseline:
        b=baseline['aggregates']['mean']
        lines += ['','与给定 baseline 最终 test mean 的差值：',
                  f'- masked PSNR：{mean["masked_psnr"]-b["masked_psnr"]:+.4f} dB。',
                  f'- masked MAE：{mean["masked_mae"]-b["masked_mae"]:+.6f}。',
                  f'- zero-mask full-image SSIM：{mean["ssim_zero_mask_full_image"]-b["ssim_zero_mask_full_image"]:+.6f}。']
    else:
        lines += ['','未提供具有同一评估协议的 baseline_result，因此不生成数值差值，避免把不同 test 划分或旧指标混为一组。']
    lines += ['','### 指标最差的 10 个相机','',
              '| 排名 | 相机 | masked PSNR | masked MAE | SSIM |','|---:|---|---:|---:|---:|']
    for row in test_summary['worst10']:
        lines.append(f'| {row["rank"]} | {row["image_name"]} | {row["masked_psnr"]:.4f} | {row["masked_mae"]:.6f} | {row["ssim_zero_mask_full_image"]:.6f} |')
    lines += ['','## 结果分析','',
              f'- 完整 test 集平均 masked PSNR 为 {mean["masked_psnr"]:.4f} dB，最差相机为 {test_summary["worst10"][0]["image_name"]}（{test_summary["worst10"][0]["masked_psnr"]:.4f} dB）。',
              '- masked PSNR 和 masked MAE 按有效 mask 像素数归一化；SSIM 是置零 mask 后的整图 SSIM，字段名保留这一语义。',
              '- 深度图是 opacity 归一化的高斯中心相机 z，只用于诊断；不能当作 LiDAR 深度真值或无偏表面深度。',
              '- 几何优劣需结合 1σ 椭球图和 geometry_log.jsonl 判断；仅凭 PSNR 不能证明高斯贴面。']
    if val:
        lines.append(f'- 最终固定 val：masked PSNR={float(val["masked_psnr"]):.4f}，masked MAE={float(val["masked_mae"]):.6f}，SSIM={float(val["ssim_zero_mask_full_image"]):.6f}。')
    if geometry:
        lines.append(f'- 最终 geometry_log 已记录 iteration={geometry.get("iteration")}；详细数值保留在原始 JSONL 中。')
    rel=Path(test_dir).relative_to(out)
    lines += ['','## 产物',
              f'- 完整 test 指标：{rel / "test_metrics.csv"}',
              f'- 最差 10 相机诊断：{rel}',
              '- 训练损失：loss_log.csv',
              '- 固定验证指标：val_metrics.csv',
              '- 几何日志：geometry_log.jsonl']
    report=out/'experiment_summary.md'
    report.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(f'[SUMMARY] {report}',flush=True)
    return report
