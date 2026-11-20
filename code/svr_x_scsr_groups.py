# -*- coding: utf-8 -*-
"""
svr_x_scsr_groups.py
====================
B3：SVR 群体差异 — 高 vs 低 SCSR 被试
=====================================

目的
----
Sullivan (2015) 假说的核心预测：**自控能力强（高 SCSR）的被试对 healthiness
加工比 tastiness 更早**，因此 EEG 表征应不同。具体地：

- 高 SCSR：health 在 EEG 中表征强（更早/更准）→ SVR health r 更高
- 低 SCSR：taste 表征强 → SVR taste r 更高

直接检验
--------
把 106 被试按 SCSR 中位数分两组（n_high / n_low 各约 53），
分别对每个 target × window 算 LOSO mean r，比较两组：

    diff = mean_r(high SCSR, health) - mean_r(high SCSR, taste)
    diff vs low SCSR：high 应有更高 health-vs-taste 优势

预计耗时（106 被试，按 SCSR 中位数分两组各约 53 人）：
- 默认 --windows=full（1 窗）：约 30 分钟
- --all-windows（6 窗）：约 22 小时
"""
from __future__ import annotations
import sys
import os
import time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

from config import FIGURES_DIR, RESULTS_DIR, PROJECT_ROOT, VALID_SUBJECTS
from svr_decoding import (
    load_subject_list, preload_full, compute_logm_all, build_target_views,
    TARGETS, TIME_WINDOWS,
)
from svr_roi_channels import loso_svr_roi

# ============ 默认运行配置 ============
DEFAULT_WINDOWS = ['full_0-1000ms']  # 只跑最强窗，默认 30 分钟
WINDOW_KEY_ALIASES = {
    'full': 'full_0-1000ms',
    'n1': 'N1_100-200ms',
    'p2': 'P2_200-300ms',
    'p3': 'P3_300-450ms',
    'lpp': 'LPP_400-700ms',
    'late': 'LATE_700-1000ms',
}

# ============ SCSR 加载 ============
SCSR_FILE = RESULTS_DIR / 'scsr_per_subject.csv'


def load_scsr():
    """从 compute_scsr.py 的输出加载 SCSR（直接来自 paired-choice 行为数据）"""
    beh_dir = RESULTS_DIR
    f = beh_dir / 'scsr_per_subject.csv'
    if f.exists():
        df = pd.read_csv(f)
        print(f'  [scsr] SCSR 加载 {len(df)} 行, columns={list(df.columns)}')
        return df
    raise FileNotFoundError('未找到 scsr_per_subject.csv，请先运行 compute_scsr.py')


def compute_scsr_from_behavior():
    """从原始 pairedchoice 行为数据计算 SCSR（兜底方案）

    SCSR = 在 paired choice 任务中，被试选择'健康选项'的比例
    （即在每对中选了相对健康那个）

    需要 load_behavior.py 的输出文件。
    """
    # 默认行为数据位于 datasets/<sub>/beh/*_task-pairedchoice_beh.tsv
    # 我们先看看能不能直接从 results 里找到
    beh_summary = RESULTS_DIR / 'baseline_summary.json'
    if beh_summary.exists():
        import json
        with open(beh_summary, 'r') as f:
            data = json.load(f)
        if 'scsr' in data:
            scsr = data['scsr']
            # 注意：scsr 是群体汇总而非被试级别，需要被试级别数据
            print(f'  [warn] baseline_summary 仅含群体 SCSR={scsr.get("mean")}, '
                  f'被试级 SCSR 需要重算')

    raise FileNotFoundError('被试级 SCSR 不在 baseline_summary 中，需要从 ddm params '
                            '或单独脚本计算')


def assign_groups(subject_ids, scsr_per_subject):
    """按 SCSR 中位数分高低组"""
    valid_ids = [s for s in subject_ids if s in scsr_per_subject]
    scsr_vals = np.array([scsr_per_subject[s] for s in valid_ids])
    median = np.median(scsr_vals)
    high = [s for s in valid_ids if scsr_per_subject[s] >= median]
    low = [s for s in valid_ids if scsr_per_subject[s] < median]
    print(f'  [groups] median SCSR={median:.3f}, '
          f'high={len(high)} low={len(low)}')
    return high, low, median


def main():
    import argparse
    import sys
    # 强制 stdout 行缓冲
    try:
        sys.stdout.reconfigure(line_buffering=True, write_through=True)
    except (AttributeError, OSError):
        pass
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true', help='10 被试冒烟')
    parser.add_argument('--n', type=int, default=None, help='限制被试数（含 SCSR 筛选后）')
    parser.add_argument('--windows', type=str, nargs='+', default=DEFAULT_WINDOWS,
                        help='要跑的窗名列表，支持别名 full/n1/p2/p3/lpp/late（默认 full）')
    parser.add_argument('--all-windows', action='store_true',
                        help='跑全部 6 窗（覆盖 --windows）')
    args, _ = parser.parse_known_args()

    # 解析窗选择
    if args.all_windows:
        windows_use = list(TIME_WINDOWS.keys())
    else:
        windows_use = []
        for w in args.windows:
            resolved = WINDOW_KEY_ALIASES.get(w.lower(), w)
            if resolved not in TIME_WINDOWS:
                raise SystemExit(f'[error] 未知窗口 {w!r}，可用别名: '
                                 f'{list(WINDOW_KEY_ALIASES.keys())} 或 {list(TIME_WINDOWS.keys())}')
            windows_use.append(resolved)

    print('=' * 60)
    print('B3: SVR decoding by SCSR group (high vs low)')
    print('=' * 60)
    print(f'  窗口: {windows_use}')
    print(f'  --n={args.n}, --quick={args.quick}')

    # 加载 SCSR
    try:
        scsr_df = load_scsr()
    except FileNotFoundError as e:
        print(f'  [error] {e}')
        return

    # SCSR 映射到被试 ID（专用 csv：subject, scsr, n_conflict）
    print(f'  [scsr] columns: {list(scsr_df.columns)}')
    sid_col = 'subject'
    scsr_col = 'scsr'

    # 归一化 subject id 格式
    scsr_df[sid_col] = scsr_df[sid_col].astype(str).apply(
        lambda x: f'sub-{int(x):03d}' if not str(x).startswith('sub-') else x)

    scsr_map = dict(zip(scsr_df[sid_col], scsr_df[scsr_col]))
    print(f'  [scsr] 加载了 {len(scsr_map)} 个被试的 SCSR（真实计算自 paired-choice）')

    # 全部有效被试
    subject_ids = load_subject_list()
    if args.quick:
        subject_ids = subject_ids[:10]
    elif args.n:
        subject_ids = subject_ids[:args.n]
    print(f'\n有效被试: {len(subject_ids)}')

    # 分组
    high_ids, low_ids, median_scsr = assign_groups(subject_ids, scsr_map)

    # 预加载
    print('\n[feat] 预加载数据...')
    full_cache = preload_full(subject_ids)
    print(f'  完成, {len(full_cache)} 被试')

    # 每个 target × window × group 跑 LOSO
    rows = []
    for wname in windows_use:
        tw = TIME_WINDOWS[wname]
        print(f'\n=== window: {wname} ({tw[0]:.3f}-{tw[1]:.3f}s) ===')
        t0 = time.time()
        logm_all = compute_logm_all(full_cache, tw)
        print(f'  [feat] logm 完成 ({time.time()-t0:.1f}s)')
        for target_col, target_label in TARGETS.items():
            logm_by_sid_full, y_by_sid = build_target_views(logm_all, target_col)
            for group_name, group_ids in [('high_scsr', high_ids),
                                          ('low_scsr', low_ids)]:
                logm_by_sid_group = {s: logm_by_sid_full[s]
                                     for s in group_ids if s in logm_by_sid_full}
                y_by_sid_group = {s: y_by_sid[s]
                                  for s in group_ids if s in logm_by_sid_full}
                if len(logm_by_sid_group) < 5:
                    continue
                t1 = time.time()
                results = loso_svr_roi(logm_by_sid_group, y_by_sid_group,
                                        group_ids, max_trials=200,
                                        random_state=42)
                if not results:
                    continue
                rs = [r['pearson_r'] for r in results]
                t_stat, p_val = stats.ttest_1samp(rs, 0)
                mean_r = float(np.mean(rs))
                std_r = float(np.std(rs, ddof=1))
                print(f'  {group_name:10s} {target_col:25s} '
                      f'n={len(rs):3d} r={mean_r:+.4f}±{std_r:.4f} '
                      f't={t_stat:+.2f} p={p_val:.4g} ({time.time()-t1:.1f}s)')
                for r in results:
                    rows.append({
                        'group': group_name,
                        'median_scsr': median_scsr,
                        'target': target_col,
                        'window': wname,
                        'subject_id': r['subject_id'],
                        'pearson_r': r['pearson_r'],
                        'n_test': r['n_test'],
                        'mean_r': mean_r,
                        'std_r': std_r,
                        't': t_stat,
                        'p': p_val,
                    })

    df = pd.DataFrame(rows)
    out_csv = RESULTS_DIR / 'svr_scsr_groups_results.csv'
    df.to_csv(out_csv, index=False)
    print(f'\n[saved] {out_csv}')

    # 汇总
    summary = df.groupby(['group', 'target', 'window']).agg(
        n=('pearson_r', 'count'),
        mean_r=('pearson_r', 'mean'),
        std_r=('pearson_r', lambda x: x.std(ddof=1)),
        t=('t', 'first'),
        p=('p', 'first'),
    ).reset_index()
    summary_csv = RESULTS_DIR / 'svr_scsr_groups_summary.csv'
    summary.to_csv(summary_csv, index=False)
    print(f'\n[saved] {summary_csv}')

    # 关键检验：高 SCSR vs 低 SCSR 在 health vs taste 上的 SVR r 差
    print('\n=== 高 vs 低 SCSR：taste vs health 优势差 ===')
    for wname in windows_use:
        h_high = summary[(summary['group'] == 'high_scsr') &
                         (summary['window'] == wname) &
                         (summary['target'] == 'rating_health')]
        t_high = summary[(summary['group'] == 'high_scsr') &
                         (summary['window'] == wname) &
                         (summary['target'] == 'rating_taste')]
        h_low = summary[(summary['group'] == 'low_scsr') &
                        (summary['window'] == wname) &
                        (summary['target'] == 'rating_health')]
        t_low = summary[(summary['group'] == 'low_scsr') &
                        (summary['window'] == wname) &
                        (summary['target'] == 'rating_taste')]
        if any(len(d) == 0 for d in [h_high, t_high, h_low, t_low]):
            continue
        diff_high = h_high['mean_r'].values[0] - t_high['mean_r'].values[0]
        diff_low = h_low['mean_r'].values[0] - t_low['mean_r'].values[0]
        # 合并 t 检验：high_diff vs low_diff（两组 SVR r 在 health-taste 差值上的差异）
        h_high_rs = df[(df['group'] == 'high_scsr') & (df['window'] == wname) &
                       (df['target'] == 'rating_health')]['pearson_r'].values
        t_high_rs = df[(df['group'] == 'high_scsr') & (df['window'] == wname) &
                       (df['target'] == 'rating_taste')]['pearson_r'].values
        h_low_rs = df[(df['group'] == 'low_scsr') & (df['window'] == wname) &
                      (df['target'] == 'rating_health')]['pearson_r'].values
        t_low_rs = df[(df['group'] == 'low_scsr') & (df['window'] == wname) &
                      (df['target'] == 'rating_taste')]['pearson_r'].values
        # 被试级 diff（每被试：health_r - taste_r），跨被试 t 检验 high vs low
        n_min = min(len(h_high_rs), len(t_high_rs), len(h_low_rs), len(t_low_rs))
        # 用被试 index 对齐：取交集
        common_high = set(df[(df['group'] == 'high_scsr') &
                             (df['window'] == wname) &
                             (df['target'] == 'rating_health')]['subject_id']) & \
                       set(df[(df['group'] == 'high_scsr') &
                             (df['window'] == wname) &
                             (df['target'] == 'rating_taste')]['subject_id'])
        common_low = set(df[(df['group'] == 'low_scsr') &
                            (df['window'] == wname) &
                            (df['target'] == 'rating_health')]['subject_id']) & \
                      set(df[(df['group'] == 'low_scsr') &
                            (df['window'] == wname) &
                            (df['target'] == 'rating_taste')]['subject_id'])
        diff_per_high = []
        diff_per_low = []
        for sid in sorted(common_high):
            hr = df[(df['group'] == 'high_scsr') & (df['window'] == wname) &
                    (df['target'] == 'rating_health') &
                    (df['subject_id'] == sid)]['pearson_r'].values[0]
            tr = df[(df['group'] == 'high_scsr') & (df['window'] == wname) &
                    (df['target'] == 'rating_taste') &
                    (df['subject_id'] == sid)]['pearson_r'].values[0]
            diff_per_high.append(hr - tr)
        for sid in sorted(common_low):
            hr = df[(df['group'] == 'low_scsr') & (df['window'] == wname) &
                    (df['target'] == 'rating_health') &
                    (df['subject_id'] == sid)]['pearson_r'].values[0]
            tr = df[(df['group'] == 'low_scsr') & (df['window'] == wname) &
                    (df['target'] == 'rating_taste') &
                    (df['subject_id'] == sid)]['pearson_r'].values[0]
            diff_per_low.append(hr - tr)
        t_diff, p_diff = stats.ttest_ind(diff_per_high, diff_per_low,
                                         equal_var=False)
        print(f'  {wname:18s}: diff_high={diff_high:+.4f}, diff_low={diff_low:+.4f}, '
              f't={t_diff:+.2f}, p={p_diff:.4g}')

    # 可视化：3 panel（每 target 一行），group × window
    fig, axes = plt.subplots(len(TARGETS), 1, figsize=(8, 9), sharex=True)
    if len(TARGETS) == 1:
        axes = [axes]
    for ax, (target_col, _) in zip(axes, TARGETS.items()):
        x = np.arange(len(windows_use))
        for gi, gname in enumerate(['high_scsr', 'low_scsr']):
            sub = summary[(summary['group'] == gname) &
                          (summary['target'] == target_col)]
            rs = [sub[sub['window'] == w]['mean_r'].values[0]
                  if len(sub[sub['window'] == w]) > 0 else 0
                  for w in windows_use]
            sems = [sub[sub['window'] == w]['std_r'].values[0] /
                    np.sqrt(sub[sub['window'] == w]['n'].values[0])
                    if len(sub[sub['window'] == w]) > 0 else 0
                    for w in windows_use]
            ax.bar(x + (gi - 0.5) * 0.4, rs, 0.4,
                   yerr=sems, label=gname, alpha=0.85,
                   color=['steelblue', 'darkorange'][gi])
        ax.axhline(0, color='black', linestyle=':', linewidth=0.8)
        ax.set_ylabel('mean r (LOSO)')
        ax.set_title(f'{target_col}')
        ax.legend(fontsize=8)
        ax.grid(axis='y', alpha=0.3)
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(windows_use, rotation=20, ha='right', fontsize=8)
    plt.suptitle('SVR decoding by SCSR group (high vs low)', fontsize=12)
    plt.tight_layout()
    out_png = FIGURES_DIR / 'fig25_scsr_group_diff.png'
    plt.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'\n[fig] {out_png}')

    # ===== JSON 汇总 =====
    import json
    summary_json = {
        'n_subjects': len(subject_ids),
        'median_scsr': float(median_scsr),
        'high_n': len(high_ids),
        'low_n': len(low_ids),
        'windows': windows_use,
        'high_vs_low_diff_ttest': [],
    }
    for wname in windows_use:
        h_high = summary[(summary['group'] == 'high_scsr') & (summary['window'] == wname) &
                         (summary['target'] == 'rating_health')]
        t_high = summary[(summary['group'] == 'high_scsr') & (summary['window'] == wname) &
                         (summary['target'] == 'rating_taste')]
        h_low = summary[(summary['group'] == 'low_scsr') & (summary['window'] == wname) &
                        (summary['target'] == 'rating_health')]
        t_low = summary[(summary['group'] == 'low_scsr') & (summary['window'] == wname) &
                        (summary['target'] == 'rating_taste')]
        if any(len(d) == 0 for d in [h_high, t_high, h_low, t_low]):
            continue
        diff_high = h_high['mean_r'].values[0] - t_high['mean_r'].values[0]
        diff_low = h_low['mean_r'].values[0] - t_low['mean_r'].values[0]
        summary_json['high_vs_low_diff_ttest'].append({
            'window': wname,
            'diff_high': float(diff_high),
            'diff_low': float(diff_low),
        })
    out_json = RESULTS_DIR / 'svr_scsr_groups_summary.json'
    with open(out_json, 'w') as f:
        json.dump(summary_json, f, indent=2)
    print(f'[saved] {out_json}')


if __name__ == '__main__':
    main()