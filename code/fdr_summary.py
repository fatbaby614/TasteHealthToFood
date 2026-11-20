# -*- coding: utf-8 -*-
"""FDR 校正 + 全阳性结果汇总（基于工作站 106 被试补充实验）"""
import pandas as pd
from statsmodels.stats.multitest import multipletests

print('=' * 60)
print('B1: ROI × target FDR 校正（21 比较）')
print('=' * 60)
b1 = pd.read_csv('outputs/results/svr_roi_channels_summary.csv')
b1['p_fdr'] = multipletests(b1['p'], method='fdr_bh')[1]
print(f'未校正显著 p<0.05: {(b1["p"] < 0.05).sum()}/{len(b1)}')
print(f'FDR 显著 p_fdr<0.05: {(b1["p_fdr"] < 0.05).sum()}/{len(b1)}')
sig = b1[b1['p_fdr'] < 0.05].sort_values('p_fdr')
for _, r in sig.iterrows():
    print(f"  {r['roi']:18s} {r['target']:30s} "
          f"r={r['mean_r']:+.4f} p={r['p']:.3g} p_fdr={r['p_fdr']:.3g} d={r['cohen_d']:+.2f}")

print()
print('=' * 60)
print('B3: group × window × target FDR 校正（36 比较）')
print('=' * 60)
b3 = pd.read_csv('outputs/results/svr_scsr_groups_summary.csv')
b3['p_fdr'] = multipletests(b3['p'], method='fdr_bh')[1]
print(f'未校正显著 p<0.05: {(b3["p"] < 0.05).sum()}/{len(b3)}')
print(f'FDR 显著 p_fdr<0.05: {(b3["p_fdr"] < 0.05).sum()}/{len(b3)}')
sig = b3[b3['p_fdr'] < 0.05].sort_values('p_fdr')
for _, r in sig.iterrows():
    print(f"  {r['group']:10s} {r['target']:25s} {r['window']:18s} "
          f"r={r['mean_r']:+.4f} p={r['p']:.3g} p_fdr={r['p_fdr']:.3g}")

print()
print('=' * 60)
print('全部显著结果按 p_fdr 排序 Top 15')
print('=' * 60)
b1_mark = b1[['roi', 'target', 'mean_r', 'p', 'p_fdr', 'cohen_d']].copy()
b1_mark['source'] = 'B1_ROI'
b1_mark = b1_mark.rename(columns={'roi': 'factor1'})
b3_mark = b3[['group', 'target', 'window', 'mean_r', 'p', 'p_fdr']].copy()
b3_mark['source'] = 'B3_SCSR'
b3_mark = b3_mark.rename(columns={'group': 'factor1', 'window': 'factor2'})
all_sig = pd.concat([
    b1_mark[b1_mark['p_fdr'] < 0.05],
    b3_mark[b3_mark['p_fdr'] < 0.05],
], ignore_index=True)
all_sig = all_sig.sort_values('p_fdr').head(15)
print(all_sig.to_string(index=False))
