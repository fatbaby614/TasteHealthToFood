# -*- coding: utf-8 -*-
"""
svr_x_ddm_ind_diff.py
=====================
B5：SVR × DDM 个体差异关联
==========================

目的
----
检验：**被试的 SVR 解码能力（mean Pearson r across LOSO folds）是否与其 DDM 决策
模型参数相关？**

预测（基于 Hare 2009, 2011）：
- v_taste 越高的被试 → taste SVR r 应该越强（更多 taste-driven 加工 → 更强的脑表征）
- v_health 越高的被试 → health SVR r 应该越强
- a（决策阈值）越高的被试 → 三 target SVR 都应该更强（更"精细"加工）
- t（非决策时间）越高的被试 → SVR 可能更弱（注意早 = 没加工完）

输入
----
- outputs/results/svr_decoding_results.csv （每被试每 target 每 window 的 r）
- outputs/results/ddm_regression_params.csv （每被试 v_taste, v_health, a, t）

输出
----
- svr_ddm_correlations.csv：每 (target, window) × DDM 参数 的相关
- svr_ddm_significance.json：summary
- fig26_svr_x_ddm_corr.png：热图

计算量
------
极小：join 两个 csv + scipy 相关。
"""
from __future__ import annotations
import sys
import os
import json
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

SVR_CSV = RESULTS_DIR / 'svr_decoding_results.csv'
DDM_CSV = RESULTS_DIR / 'ddm_regression_params.csv'
SCSR_OPTIONAL = RESULTS_DIR / 'individual_metrics.csv'


def load_ddm_params():
    """加载 DDM 参数，确保 subject_id 格式统一"""
    df = pd.read_csv(DDM_CSV)
    print(f'  [ddm] 加载 {len(df)} 行, columns={list(df.columns)}')

    # 找 subject id 列
    sid_col = None
    for c in ['subject_id', 'sub', 'sub_id', 'subid', 'subject']:
        if c in df.columns:
            sid_col = c
            break
    if sid_col is None:
        sid_col = df.columns[0]
    df[sid_col] = df[sid_col].astype(str).apply(
        lambda x: f'sub-{int(x):03d}' if not x.startswith('sub-') else x)
    df = df.rename(columns={sid_col: 'subject_id'})
    print(f'  [ddm] subject_id 列 OK, n={len(df)}')

    # 标准化 DDM 列名（兼容旧版）
    rename_map = {
        'v_taste': 'v_taste',
        'v.health': 'v_health',
        'v_health': 'v_health',
        'v_taste_minus_v_health': 'v_diff',
        'a': 'a',
        'bound': 'a',
        't': 't',
        'ndt': 't',
    }
    df = df.rename(columns=rename_map)

    return df


def load_svr_results():
    df = pd.read_csv(SVR_CSV)
    print(f'  [svr] 加载 {len(df)} 行')
    # 确保 subject_id 格式
    df['subject_id'] = df['subject_id'].astype(str)
    return df


def main(quick=False):
    print('=' * 60)
    print('B5: SVR × DDM individual differences correlation')
    print('=' * 60)

    svr = load_svr_results()
    ddm = load_ddm_params()

    # merge
    merged = svr.merge(ddm, on='subject_id', how='inner')
    print(f'  [merge] 共有 {len(merged)} 行')

    # 公共被试过少 → 无法做个体差异相关分析
    # 不写任何结果文件，避免部分运行（如 --quick）覆盖全量结果
    n_common = merged['subject_id'].nunique()
    if n_common < 10:
        print('\n[warn] SVR 与 DDM 的公共被试不足 10 人，无法做个体差异相关分析')
        print(f'  svr_decoding_results.csv : {len(svr)} 行, '
              f'{svr["subject_id"].nunique()} 被试')
        print(f'  ddm_regression_params.csv: {len(ddm)} 行, '
              f'{ddm["subject_id"].nunique()} 被试')
        print(f'  公共被试: {n_common}')
        print('  可能原因: --quick 冒烟模式只处理少量被试，或全量结果文件缺失/被覆盖。')
        print('  解决: 先完成全量阶段 1d (ddm_model) 与 5c (svr_decoding) 再运行本分析。')
        if quick:
            print('  [quick] 冒烟模式下跳过本分析（退出码 0）')
            return
        sys.exit(1)

    # DDM 参数列
    ddm_cols = [c for c in ['v_taste', 'v_health', 'v_diff', 'a', 't']
                if c in merged.columns]
    print(f'  [ddm params] {ddm_cols}')

    # 加载 SCSR（可选）
    scsr_map = {}
    if SCSR_OPTIONAL.exists():
        sdf = pd.read_csv(SCSR_OPTIONAL)
        if 'scsr' in sdf.columns:
            sid_col = [c for c in sdf.columns if 'sub' in c.lower()]
            if sid_col:
                sdf[sid_col[0]] = sdf[sid_col[0]].astype(str).apply(
                    lambda x: f'sub-{int(x):03d}' if not str(x).startswith('sub-') else x)
                scsr_map = dict(zip(sdf[sid_col[0]], sdf['scsr']))
                merged['scsr'] = merged['subject_id'].map(scsr_map)
                ddm_cols.append('scsr')
                print(f'  [scsr] 加载 {len(scsr_map)} 个被试')

    # 计算每个 (target, window) × DDM 参数的 Pearson r + Spearman rho
    rows = []
    for target_col in svr['target'].unique():
        for wname in svr['window_name'].unique():
            sub = merged[(merged['target'] == target_col) &
                         (merged['window_name'] == wname)]
            if len(sub) < 10:
                continue
            for ddm_c in ddm_cols:
                d = sub[ddm_c].dropna()
                s = sub.loc[d.index, 'pearson_r']
                if len(d) < 10:
                    continue
                r_p, p_p = stats.pearsonr(d, s)
                r_s, p_s = stats.spearmanr(d, s)
                rows.append({
                    'target': target_col,
                    'window': wname,
                    'ddm_param': ddm_c,
                    'n': len(d),
                    'pearson_r': r_p,
                    'pearson_p': p_p,
                    'spearman_r': r_s,
                    'spearman_p': p_s,
                })

    df = pd.DataFrame(rows)
    # FDR 校正（无论 df 是否为空都生成列，避免下游 KeyError）
    df['pearson_p_fdr'] = np.nan
    df['spearman_p_fdr'] = np.nan
    if len(df) > 0:
        df['pearson_p_fdr'] = multipletests(df['pearson_p'].values,
                                             method='fdr_bh')[1]
        df['spearman_p_fdr'] = multipletests(df['spearman_p'].values,
                                              method='fdr_bh')[1]

    out_csv = RESULTS_DIR / 'svr_ddm_correlations.csv'
    df.to_csv(out_csv, index=False)
    print(f'\n[saved] {out_csv}')

    # 显著项
    sig_df = df[(df['pearson_p_fdr'] < 0.05) | (df['spearman_p_fdr'] < 0.05)]
    if len(sig_df) > 0:
        print(f'\n=== FDR 显著项 (n={len(sig_df)}) ===')
        for _, row in sig_df.iterrows():
            print(f"  {row['target']:30s} {row['window']:20s} {row['ddm_param']:12s} "
                  f"r={row['pearson_r']:+.3f} (p={row['pearson_p']:.4g}, "
                  f"p_fdr={row['pearson_p_fdr']:.4g})")
    else:
        print('\n=== 无 FDR 显著项 ===')
        # 列出未校正最强
        if len(df) > 0:
            top = df.nsmallest(10, 'pearson_p')
            print(top[['target', 'window', 'ddm_param', 'pearson_r', 'pearson_p']].to_string())
        else:
            print('  （无可用相关结果：各组合的有效被试均不足或 DDM 参数缺失）')

    # 汇总 json
    summary = {
        'n_subjects': int(merged['subject_id'].nunique()),
        'n_tests': len(df),
        'n_significant_uncorrected': int((df['pearson_p'] < 0.05).sum()) if len(df) > 0 else 0,
        'n_significant_fdr': int((df['pearson_p_fdr'] < 0.05).sum()) if len(df) > 0 else 0,
        'ddm_params_used': ddm_cols,
    }
    if len(sig_df) > 0:
        summary['significant_findings'] = sig_df.to_dict('records')
    with open(RESULTS_DIR / 'svr_ddm_correlations.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f'\n[summary] {RESULTS_DIR / "svr_ddm_correlations.json"}')

    # 可视化：3 panel（每 target 一行），每个 cell = Pearson r
    fig, axes = plt.subplots(3, 1, figsize=(9, 8))
    for ax, target_col in zip(axes, svr['target'].unique()):
        sub = df[df['target'] == target_col]
        if len(sub) == 0:
            ax.set_visible(False)
            continue
        pivot = sub.pivot(index='ddm_param', columns='window', values='pearson_r')
        # 标记 FDR 显著
        pivot_p = sub.pivot(index='ddm_param', columns='window', values='pearson_p_fdr')
        im = ax.imshow(pivot.values, cmap='RdBu_r',
                       vmin=-abs(pivot.values).max(),
                       vmax=abs(pivot.values).max(), aspect='auto')
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, rotation=30, ha='right', fontsize=8)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index, fontsize=9)
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                val = pivot.values[i, j]
                p = pivot_p.values[i, j]
                marker = '*' if p < 0.05 else ''
                color = 'white' if abs(val) > 0.15 else 'black'
                ax.text(j, i, f'{val:+.2f}{marker}', ha='center', va='center',
                        fontsize=8, color=color)
        ax.set_title(f'{target_col}: Pearson r(SVR mean r, DDM param)', fontsize=10)
        plt.colorbar(im, ax=ax)
    plt.suptitle('SVR × DDM cross-task correlations (* = FDR p<0.05)', fontsize=11)
    plt.tight_layout()
    out_png = FIGURES_DIR / 'fig26_svr_x_ddm_corr.png'
    plt.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'[fig] {out_png}')


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(
        description='B5: SVR × DDM individual differences correlation')
    ap.add_argument('--quick', action='store_true',
                    help='冒烟模式：公共被试不足 10 时警告并跳过（退出码 0）')
    args = ap.parse_args()
    main(quick=args.quick)