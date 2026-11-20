# -*- coding: utf-8 -*-
"""
svr_roi_x_window.py
====================
B6：SVR 通道 ROI × 时间窗 二维网格分析
========================================

目的
----
之前 B1 只跑了"全窗 (full_0-1000ms) × 7 ROI × 3 target" = 21 个组合。
现在扩展到 **6 个时间窗 × 7 ROI × 3 target = 126 个组合**，
找出"哪个 ROI 在哪个时间窗"对哪个 target 编码最强。

复用的 B1 优化
--------------
- 通道顺序：与 EEG_CHANNEL_NAMES 同步（避免 import mne 慢）
- 缓存 logm_by_sid：每个 (window, target) 的 SVR 共享同一份预计算 logm
- LOSO 复用 svr_roi_channels.loso_svr_roi

计算量估算
----------
- 7 ROI × 6 窗 × 3 target = 126 LOSO 组合
- 106 被试全量
- 工作站预期 ~30-50 分钟（每个组合约 15-25s）

输出
----
- svr_roi_window_results.csv：每被试每条件一行
- svr_roi_window_summary.csv：每 (roi, target, window) 一行（含 FDR p）
- fig27_roi_x_window.png：3 panel (target) × 7 ROI × 6 window 热图
"""
from __future__ import annotations
import sys
import os
import time
import argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
from statsmodels.stats.multitest import multipletests

from config import RESULTS_DIR, FIGURES_DIR
from svr_decoding import (
    load_subject_list, preload_full, compute_logm_all, build_target_views,
    TARGETS, TIME_WINDOWS,
)
from svr_roi_channels import (
    detect_channel_names, get_roi_indices, ROI_CHANNELS, loso_svr_roi,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true', help='冒烟测试：5 被试')
    parser.add_argument('--rois', type=str, default='all',
                        help='指定 ROI 子集，逗号分隔，如 "frontal,occipital"')
    parser.add_argument('--targets', type=str, default='all',
                        help='指定 target 子集，逗号分隔，如 "rating_taste"')
    parser.add_argument('--max-trials', type=int, default=80)
    args, _ = parser.parse_known_args()

    print('=' * 60)
    print('B6: SVR ROI × 时间窗 二维网格')
    print('=' * 60)

    subject_ids = load_subject_list()
    if args.quick:
        subject_ids = subject_ids[:5]
    print(f'\n有效被试: {len(subject_ids)}')

    # 通道顺序
    ch_names = detect_channel_names(subject_ids)
    print(f'  通道顺序: {ch_names[:5]} ... (n={len(ch_names)})')

    # ROI 筛选
    if args.rois == 'all':
        rois = ROI_CHANNELS
    else:
        names = [n.strip() for n in args.rois.split(',')]
        rois = {n: ROI_CHANNELS[n] for n in names if n in ROI_CHANNELS}
    print(f'  ROI 数: {len(rois)}')

    # target 筛选
    if args.targets == 'all':
        targets = TARGETS
    else:
        names = [n.strip() for n in args.targets.split(',')]
        targets = {n: TARGETS[n] for n in names if n in TARGETS}
    print(f'  target 数: {len(targets)}')

    # 预加载
    print('\n[feat] 预加载数据...')
    full_cache = preload_full(subject_ids)
    print(f'  完成, {len(full_cache)} 被试')

    # ROI 索引预计算（一次）
    roi_indices = {}
    for roi_name, roi_chs in rois.items():
        idx, missing = get_roi_indices(ch_names, roi_chs)
        if len(idx) < 4:
            print(f'  [skip ROI] {roi_name}: 仅 {len(idx)} 通道')
            continue
        roi_indices[roi_name] = idx
    print(f'  ROI 索引: {list(roi_indices.keys())}')

    # 每个窗口算一次 logm_all（每个 target 用同一份 logm 切片）
    rows = []
    n_total = len(roi_indices) * len(targets) * len(TIME_WINDOWS)
    n_done = 0
    t_start = time.time()

    for wname, tw in TIME_WINDOWS.items():
        print(f'\n=== window: {wname} ({tw[0]:.3f}-{tw[1]:.3f}s) ===')
        t0 = time.time()
        logm_all = compute_logm_all(full_cache, tw)
        print(f'  [feat] logm 完成 ({time.time()-t0:.1f}s)')
        for target_col, target_label in targets.items():
            logm_by_sid_full, y_by_sid = build_target_views(logm_all, target_col)
            for roi_name, roi_idx in roi_indices.items():
                # 按 ROI 切片 logm
                logm_by_sid_roi = {
                    sid: logm[np.ix_(np.arange(len(logm)), roi_idx, roi_idx)]
                    for sid, logm in logm_by_sid_full.items()
                }
                t1 = time.time()
                results = loso_svr_roi(logm_by_sid_roi, y_by_sid, subject_ids,
                                        max_trials=args.max_trials,
                                        random_state=42)
                n_done += 1
                if not results:
                    continue
                rs = [r['pearson_r'] for r in results]
                t_stat, p_val = stats.ttest_1samp(rs, 0)
                mean_r = float(np.mean(rs))
                std_r = float(np.std(rs, ddof=1))
                d_val = mean_r / std_r if std_r > 0 else 0.0
                print(f'  [{n_done:3d}/{n_total}] {roi_name:18s} {target_col:25s} '
                      f'n={len(rs):3d} r={mean_r:+.4f}±{std_r:.4f} '
                      f't={t_stat:+.2f} p={p_val:.4g} d={d_val:+.2f} '
                      f'({time.time()-t1:.1f}s)')
                for r in results:
                    rows.append({
                        'roi': roi_name,
                        'n_channels': len(roi_idx),
                        'target': target_col,
                        'window': wname,
                        'subject_id': r['subject_id'],
                        'pearson_r': r['pearson_r'],
                        'n_test': r['n_test'],
                        'mean_r': mean_r,
                        'std_r': std_r,
                        't': t_stat,
                        'p': p_val,
                        'cohen_d': d_val,
                    })

    total_dur = time.time() - t_start
    print(f'\n===== 总耗时 {total_dur:.1f}s ({total_dur/60:.1f} min) =====')

    # 保存逐被试结果
    df = pd.DataFrame(rows)
    out_csv = RESULTS_DIR / 'svr_roi_window_results.csv'
    df.to_csv(out_csv, index=False)
    print(f'[saved] {out_csv}')

    # 汇总（每 (roi, target, window) 一行）
    summary = df.groupby(['roi', 'target', 'window']).agg(
        n=('pearson_r', 'count'),
        mean_r=('pearson_r', 'mean'),
        std_r=('pearson_r', lambda x: x.std(ddof=1)),
        t=('t', 'first'),
        p=('p', 'first'),
        cohen_d=('cohen_d', 'first'),
    ).reset_index()

    # FDR 校正（126 个比较）
    if len(summary) > 0:
        summary['p_fdr'] = multipletests(summary['p'], method='fdr_bh')[1]
    out_summary = RESULTS_DIR / 'svr_roi_window_summary.csv'
    summary.to_csv(out_summary, index=False)
    print(f'[saved] {out_summary}')

    # 显著项
    n_sig_uncorr = (summary['p'] < 0.05).sum()
    n_sig_fdr = (summary['p_fdr'] < 0.05).sum()
    print(f'\n[统计] 总比较: {len(summary)}, '
          f'未校正显著: {n_sig_uncorr}, FDR 显著: {n_sig_fdr}')

    sig_df = summary[summary['p_fdr'] < 0.05].sort_values('p_fdr')
    print(f'\n=== FDR 显著 Top 20 (按 p_fdr) ===')
    for _, row in sig_df.head(20).iterrows():
        print(f"  {row['roi']:18s} {row['target']:25s} {row['window']:18s} "
              f"r={row['mean_r']:+.4f} p={row['p']:.3g} p_fdr={row['p_fdr']:.3g} "
              f"d={row['cohen_d']:+.2f}")

    # 可视化：3 panel（每 target 一行）热图：7 ROI × 6 window
    n_rois = len(roi_indices)
    n_wins = len(TIME_WINDOWS)
    fig, axes = plt.subplots(len(targets), 1, figsize=(10, 8), sharex=True)
    if len(targets) == 1:
        axes = [axes]
    for ax, (target_col, _) in zip(axes, targets.items()):
        sub = summary[summary['target'] == target_col]
        if len(sub) == 0:
            ax.set_visible(False)
            continue
        pivot = sub.pivot(index='roi', columns='window', values='mean_r')
        pivot_p = sub.pivot(index='roi', columns='window', values='p_fdr')
        # 保证行列顺序
        pivot = pivot.reindex(index=list(roi_indices.keys()),
                              columns=list(TIME_WINDOWS.keys()))
        pivot_p = pivot_p.reindex(index=list(roi_indices.keys()),
                                  columns=list(TIME_WINDOWS.keys()))
        vmax = max(abs(pivot.values.min()), abs(pivot.values.max()))
        im = ax.imshow(pivot.values, cmap='RdBu_r',
                       vmin=-vmax, vmax=vmax, aspect='auto')
        ax.set_xticks(range(n_wins))
        ax.set_xticklabels(pivot.columns, rotation=25, ha='right', fontsize=8)
        ax.set_yticks(range(n_rois))
        ax.set_yticklabels(pivot.index, fontsize=9)
        for i in range(n_rois):
            for j in range(n_wins):
                val = pivot.values[i, j]
                p = pivot_p.values[i, j]
                if np.isnan(p):
                    marker = ''
                elif p < 0.001:
                    marker = '***'
                elif p < 0.01:
                    marker = '**'
                elif p < 0.05:
                    marker = '*'
                else:
                    marker = ''
                color = 'white' if abs(val) > 0.03 else 'black'
                ax.text(j, i, f'{val:+.3f}\n{marker}', ha='center', va='center',
                        fontsize=7, color=color)
        ax.set_title(f'{target_col} (mean r, * = FDR p<0.05)', fontsize=10)
        plt.colorbar(im, ax=ax, label='mean r')
    plt.suptitle('SVR LOSO decoding: ROI × 时间窗 二维网格 (n={})'.format(
        len(subject_ids)), fontsize=12, y=1.02)
    plt.tight_layout()
    out_png = FIGURES_DIR / 'fig27_roi_x_window.png'
    plt.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'\n[fig] {out_png}')


if __name__ == '__main__':
    main()