"""提取改稿所需的全部数值（本地执行，输出为写作事实清单）。"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

R = Path('outputs/results')

print('=' * 70)
print('1) 21-cell ROI 家族（roi_family_pooled）— 正式主口径')
print('=' * 70)
df = pd.read_csv(R / 'audit_posthoc_bh.csv')
r = df[df['family'] == 'roi_family_pooled'].copy()
r['pass_one'] = r['q_bh'] <= 0.05
r['pass_two'] = r['q_bh_two_sided'] <= 0.05
print(r[['roi', 'target', 'n_perm_B', 'n_seeds_S', 'T_obs',
          'p_agg', 'p_agg_two_sided_exact', 'q_bh', 'pass_one',
          'q_bh_two_sided', 'pass_two']].to_string(index=False))
print(f"\n单侧通过 {int(r['pass_one'].sum())}/{len(r)}  "
      f"双侧通过 {int(r['pass_two'].sum())}/{len(r)}")

print('\n' + '=' * 70)
print('2) ROI within（置换单位敏感性）')
print('=' * 70)
w = df[df['family'] == 'roi_family_within'].copy()
print(f"单侧通过 {int((w['q_bh'] <= 0.05).sum())}/{len(w)}  "
      f"双侧通过 {int((w['q_bh_two_sided'] <= 0.05).sum())}/{len(w)}")
print(w[w['q_bh'] <= 0.06][['roi', 'target', 'p_agg', 'q_bh',
                            'q_bh_two_sided']].to_string(index=False))

print('\n' + '=' * 70)
print('3) main_pooled 家族（full 窗主结果，5 行）')
print('=' * 70)
m = df[df['family'] == 'main_pooled'].copy()
print(m[['roi', 'target', 'window', 'n_perm_B', 'n_seeds_S', 'T_obs',
         'p_agg', 'p_agg_two_sided_exact', 'q_bh', 'q_bh_two_sided']].to_string(index=False))
print(f"单侧通过 {int((m['q_bh'] <= 0.05).sum())}/{len(m)}  "
      f"双侧通过 {int((m['q_bh_two_sided'] <= 0.05).sum())}/{len(m)}")

print('\n' + '=' * 70)
print('4) scalp18_pooled 家族（探索性）')
print('=' * 70)
s = df[df['family'] == 'scalp18_pooled'].copy()
print(s[['window', 'target', 'n_perm_B', 'n_seeds_S', 'T_obs', 'p_agg',
         'p_agg_two_sided_exact', 'q_bh', 'q_bh_two_sided']].to_string(index=False))
print(f"单侧通过 {int((s['q_bh'] <= 0.05).sum())}/{len(s)}  "
      f"双侧通过 {int((s['q_bh_two_sided'] <= 0.05).sum())}/{len(s)}")
print(f"单侧最小 q={s['q_bh'].min():.6f}  双侧最小 q={s['q_bh_two_sided'].min():.6f}")

print('\n' + '=' * 70)
print('5) 训练预算三配置（A/B/C）+ hib 变体')
print('=' * 70)
for fam in ['budgetA_pool200', 'budgetB_persubj200', 'budgetC_alltrials',
             'budgetB_persubj200_hib', 'budgetC_alltrials_hib',
             'budgetB_persubj200_scalp', 'budgetC_alltrials_scalp',
             'scalp_within']:
    g = df[df['family'] == fam]
    if len(g) == 0:
        print(f'[{fam}] 无记录')
        continue
    print(f'\n[{fam}] m={len(g)}  单侧通过 {int((g["q_bh"] <= 0.05).sum())}  '
          f'双侧通过 {int((g["q_bh_two_sided"] <= 0.05).sum())}')
    print(g[['roi', 'target', 'n_perm_B', 'n_seeds_S', 'max_trials', 'T_obs',
             'p_agg', 'q_bh', 'q_bh_two_sided']].to_string(index=False))

print('\n' + '=' * 70)
print('6) audit_posthoc_cells.csv（观测统计量/σ_null/MDE/功效）')
print('=' * 70)
c = pd.read_csv(R / 'audit_posthoc_cells.csv')
print('列：', list(c.columns))
print(c.head(30).to_string(index=False))

print('\n' + '=' * 70)
print('7) 校准 FPR（Type I error）')
print('=' * 70)
f = pd.read_csv(R / 'audit_posthoc_fpr.csv')
print('列：', list(f.columns))
print(f.to_string(index=False))
