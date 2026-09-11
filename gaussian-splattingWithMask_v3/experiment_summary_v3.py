"""Generate the concise, source-grounded report required after a formal run."""
from pathlib import Path
import csv
import json

from arguments_v3 import FEATURES, preset_features


UPSTREAM_DEFAULTS = {
    'iterations': 30000,
    'sh_degree': 3,
    'position_lr_init': 0.00016,
    'position_lr_final': 0.0000016,
    'position_lr_max_steps': 30000,
    'feature_lr': 0.0025,
    'opacity_lr': 0.025,
    'scaling_lr': 0.005,
    'rotation_lr': 0.001,
    'lambda_dssim': 0.2,
}
KEYS = list(UPSTREAM_DEFAULTS) + [
    'seed', 'init_log_thickness', 'knn', 'planarity_min',
    'neighbor_radius_ratio', 'surface_tolerance_ratio', 'tangent_radius_ratio',
    'thickness_ratio', 'size_ratio', 'lambda_surface', 'lambda_tangent',
    'lambda_normal', 'lambda_flatten', 'lambda_size', 'prune_start',
    'prune_interval', 'prune_opacity', 'prune_min_views', 'prune_patience',
    'prune_max_fraction', 'lambda_lidar_depth', 'lidar_depth_start',
    'lidar_depth_warmup', 'lidar_depth_min', 'lidar_depth_max',
    'lidar_depth_splat_radius', 'lidar_depth_edge_relative',
    'lidar_depth_edge_absolute', 'lidar_depth_min_neighbors',
    'lidar_depth_alpha_min', 'lidar_depth_min_pixels',
    'lidar_depth_huber_beta', 'lidar_depth_distance_power',
    'lidar_depth_weight_min', 'lidar_depth_weight_max',
    'lidar_depth_backproject_samples', 'lidar_depth_backproject_tolerance',
    'val_interval', 'checkpoint_interval',
]


def _fmt(value):
    if isinstance(value, float):
        return f'{value:.8g}'
    return str(value)


def _last_csv_mean(path):
    path = Path(path)
    if not path.is_file():
        return None
    answer = None
    with path.open(newline='', encoding='utf-8') as handle:
        for row in csv.DictReader(handle):
            if row.get('image_name') == 'MEAN':
                answer = row
    return answer


def _last_jsonl(path):
    path = Path(path)
    if not path.is_file():
        return None
    answer = None
    with path.open(encoding='utf-8') as handle:
        for line in handle:
            if line.strip():
                answer = json.loads(line)
    return answer


def _baseline_summary(path):
    if not path:
        return None
    direct = Path(path) / 'test_final' / 'iteration_150000' / 'test_summary.json'
    if direct.is_file():
        return json.loads(direct.read_text(encoding='utf-8'))
    return None


def _depth_loss_stats(path):
    path = Path(path)
    if not path.is_file():
        return None
    fields = [
        'lidar_depth_raw',
        'lidar_depth_weighted',
        'lidar_depth_valid_pixels',
        'lidar_depth_rendered_fraction',
        'lidar_depth_mean_target_depth',
        'lidar_depth_distance_weight_mean',
    ]
    sums = {field: 0.0 for field in fields}
    total = 0
    active = 0
    with path.open(newline='', encoding='utf-8') as handle:
        for row in csv.DictReader(handle):
            total += 1
            if row.get('lidar_depth_state') != 'ACTIVE':
                continue
            active += 1
            for field in fields:
                sums[field] += float(row[field])
    return {
        'total_rows': total,
        'active_rows': active,
        'active_fraction': active / max(total, 1),
        'active_means': {
            field: sums[field] / active for field in fields
        } if active else {},
    }


def write_experiment_summary(a, iteration, test_summary, test_dir):
    out = Path(a.model_path)
    baseline_features = preset_features('A')
    active = {key: bool(getattr(a, key)) for key in FEATURES}
    feature_delta = {
        key: (baseline_features[key], active[key])
        for key in FEATURES
        if baseline_features[key] != active[key]
    }
    baseline = _baseline_summary(a.baseline_result)
    comparison_ok = bool(
        baseline and
        baseline.get('comparison_identity', {}).get('sha256') ==
        test_summary.get('comparison_identity', {}).get('sha256')
    )
    mean = test_summary['aggregates']['mean']
    val = _last_csv_mean(out / 'val_metrics.csv')
    geometry = _last_jsonl(out / 'geometry_log.jsonl')
    depth_training = _depth_loss_stats(out / 'loss_log.csv')
    manifest = json.loads((out / 'run_manifest.json').read_text(encoding='utf-8'))
    depth_dataset = manifest.get('lidar_depth_dataset') or {}
    depth_summary = depth_dataset.get('summary')

    lines = [
        f'# 实验 {a.experiment} 总结',
        '',
        f'- 状态：完成 {iteration} 次迭代，并完成完整 test 集评估。',
        f'- 高斯数量：{mean["count"]}。',
        f'- test 相机数：{test_summary["camera_count"]}。',
        '- 最差相机排序：masked PSNR 升序，同分按 image_name；保存前 10 张的 RGB、1σ 彩色椭球、深度和法向量。',
        '- test 与 val 均不保存 geometry.npz。',
        '',
        '## 相对 baseline A 的修改',
        '',
    ]
    if not feature_delta:
        lines.append('- 本组功能开关与 baseline A 相同。')
    else:
        for key, (before, after) in feature_delta.items():
            lines.append(
                f'- {key}：{str(before).lower()} → {str(after).lower()}。'
            )
    if a.experiment == 'E':
        lines.append(
            '- E 继承 C 的几何损失和硬尺度上限；本组唯一新增训练因素是 '
            'occlusion-aware LiDAR camera-z depth loss。'
        )
    lines += [
        '',
        '当前启用功能：' + ', '.join(key for key, value in active.items() if value) + '。',
        '',
        '## 关键超参数',
        '',
        '| 参数 | 本实验 | 原始默认值 | 说明 |',
        '|---|---:|---:|---|',
    ]
    for key in KEYS:
        if not hasattr(a, key):
            continue
        value = getattr(a, key)
        default = UPSTREAM_DEFAULTS.get(key, '—')
        changed = key in UPSTREAM_DEFAULTS and value != default
        note = (
            '与原始默认值不同'
            if changed else
            ('与原始默认值相同' if key in UPSTREAM_DEFAULTS else 'v3 几何/运行参数')
        )
        lines.append(f'| {key} | {_fmt(value)} | {_fmt(default)} | {note} |')

    lines += [
        '',
        f'固定运行行为：关闭增密和 opacity reset；checkpoint/PLY 每 '
        f'{a.checkpoint_interval} 步；固定 val 每 {a.val_interval} 步。',
        '',
        '## LiDAR 伪深度与训练',
        '',
    ]
    if depth_summary:
        back = depth_summary['backprojection']
        lines += [
            f'- 数据集目录：{depth_dataset["path"]}。',
            f'- 已导出 {depth_summary["camera_count"]} 张 train/val/test 16-bit PNG；'
            '编码 0 表示无效，val/test 另有彩色预览。',
            f'- 平均有效覆盖率：{depth_summary["coverage_mean"]:.6f}。',
            f'- 从已保存 PNG 解码并反投影后，逐相机 P95 的最大值为 '
            f'{back["nearest_lidar_p95_max"]:.6f}；阈值为 '
            f'{back["required_p95_max"]:.6f}，全部通过。',
            '- 遮挡处理：同一像素保留最小 camera-z；补洞只接受局部深度一致的 '
            'LiDAR 邻域，深度跳变处不补。',
        ]
    else:
        lines.append('- 本次运行没有持久化 LiDAR 伪深度数据集。')
    if depth_training:
        means = depth_training['active_means']
        lines.append(
            f'- 深度 loss 生效 {depth_training["active_rows"]}/'
            f'{depth_training["total_rows"]} 步'
            f'（{depth_training["active_fraction"]:.4%}）。'
        )
        if means:
            lines += [
                f'- 生效步平均 raw depth loss：{means["lidar_depth_raw"]:.8f}；'
                f'平均 weighted depth loss：{means["lidar_depth_weighted"]:.8f}。',
                f'- 生效步平均监督像素：{means["lidar_depth_valid_pixels"]:.2f}；'
                f'平均 LiDAR 可渲染比例：'
                f'{means["lidar_depth_rendered_fraction"]:.6f}。',
                f'- 距离权重平均值：'
                f'{means["lidar_depth_distance_weight_mean"]:.6f}；'
                '权重按 median_depth/depth 的幂计算并截断。',
            ]

    lines += [
        '',
        '## 最终 test 结果',
        '',
        '| 指标 | mean | median | min | max |',
        '|---|---:|---:|---:|---:|',
    ]
    for key in ('masked_psnr', 'masked_mae', 'ssim_zero_mask_full_image'):
        aggregates = test_summary['aggregates']
        lines.append(
            f'| {key} | {_fmt(aggregates["mean"][key])} | '
            f'{_fmt(aggregates["median"][key])} | '
            f'{_fmt(aggregates["min"][key])} | '
            f'{_fmt(aggregates["max"][key])} |'
        )
    if comparison_ok:
        baseline_mean = baseline['aggregates']['mean']
        lines += [
            '',
            '与同一输入和 test 协议的 baseline 最终 test mean 差值：',
            f'- masked PSNR：{mean["masked_psnr"]-baseline_mean["masked_psnr"]:+.4f} dB。',
            f'- masked MAE：{mean["masked_mae"]-baseline_mean["masked_mae"]:+.6f}。',
            f'- zero-mask full-image SSIM：'
            f'{mean["ssim_zero_mask_full_image"]-baseline_mean["ssim_zero_mask_full_image"]:+.6f}。',
        ]
    else:
        lines += [
            '',
            '未生成 baseline 数值差值：缺少 baseline_result，或其输入、test 顺序和'
            '指标协议的 comparison_identity 与本次不完全一致。',
        ]

    lines += [
        '',
        '### 指标最差的 10 个相机',
        '',
        '| 排名 | 相机 | masked PSNR | masked MAE | SSIM |',
        '|---:|---|---:|---:|---:|',
    ]
    for row in test_summary['worst10']:
        lines.append(
            f'| {row["rank"]} | {row["image_name"]} | '
            f'{row["masked_psnr"]:.4f} | {row["masked_mae"]:.6f} | '
            f'{row["ssim_zero_mask_full_image"]:.6f} |'
        )
    worst = test_summary['worst10'][0]
    lines += [
        '',
        '## 结果分析',
        '',
        f'- 完整 test 集平均 masked PSNR 为 {mean["masked_psnr"]:.4f} dB；'
        f'最差相机为 {worst["image_name"]}（{worst["masked_psnr"]:.4f} dB）。',
        '- masked PSNR 和 masked MAE 按有效 mask 像素数归一化；SSIM 是置零 '
        'mask 后的整图 SSIM。',
        '- 训练深度预测是 opacity 归一化的高斯中心 camera-z。它与 LiDAR '
        'camera-z 在可靠像素做相对 Smooth L1，并额外使用可调的逆距离权重。',
        '- 几何优劣仍需结合 1σ 椭球图、geometry_log.jsonl 和 LiDAR '
        '反投影误差判断；PSNR 本身不能证明高斯贴面。',
    ]
    if val:
        lines.append(
            f'- 最终固定 val：masked PSNR={float(val["masked_psnr"]):.4f}，'
            f'masked MAE={float(val["masked_mae"]):.6f}，'
            f'SSIM={float(val["ssim_zero_mask_full_image"]):.6f}。'
        )
    if geometry:
        lines.append(
            f'- 最终 geometry_log 已记录 iteration={geometry.get("iteration")}；'
            '详细数值保留在原始 JSONL 中。'
        )

    relative_test = Path(test_dir).relative_to(out)
    lines += [
        '',
        '## 产物',
        '',
        f'- 完整 test 指标：{relative_test / "test_metrics.csv"}',
        f'- 最差 10 相机诊断：{relative_test}',
        '- 训练损失：loss_log.csv',
        '- 固定验证指标：val_metrics.csv',
        '- 几何日志：geometry_log.jsonl',
    ]
    if depth_dataset.get('path'):
        lines.append(
            f'- LiDAR 伪深度、manifest 和反投影验证：{depth_dataset["path"]}'
        )
    report = out / 'experiment_summary.md'
    report.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f'[SUMMARY] {report}', flush=True)
    return report
