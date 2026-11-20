"""提取 scalp18 家族逐 cell 完整数值（重写 Table 2 用）。"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

R = Path('outputs/results')
WIN_ORDER = ['N1_100-200ms', 'P2_200-300ms', 'P3_300-450ms',
             'LPP_400-700ms', 'LATE_700-1000ms', 'full_0-1000ms']
FILES = {
    'N1_100-200ms': 'seed_inference_audit_scalp_N1.json',
    'P2_200-300ms': 'seed_inference_audit_scalp_P2.json',
    'P3_300-450ms': 'seed_inference_audit_scalp_P3.json',
    'LPP_400-700ms': 'seed_inference_audit_scalp_LPP.json',
    'LATE_700-1000ms': 'seed_inference_audit_scalp_LATE.json',
    'full_0-1000ms': 'seed_inference_audit.json',
}
TGT = ['rating_health', 'rating_taste', 'rating_willingnessToEat']

bh = pd.read_csv(R / 'audit_posthoc_bh.csv')
s18 = bh[bh['family'] == 'scalp18_pooled'].set_index(['window', 'target'])

cells = pd.read_csv(R / 'audit_posthoc_cells.csv')
# scalp18_pooled 标签下还包含主结果文件中的 occipital cell（BH 表已按
# roi|target 白名单过滤，此处按 roi=='all' 过滤以对齐 18 行）
cc = cells[(cells['source'] == 'scalp18_pooled') & (cells['roi'] == 'all')] \
    .set_index(['window', 'target'])


def pick(frame, w, t, col):
    """按 (window, target) 取单列标量（MultiIndex → 单行 Series）。"""
    return frame.loc[(w, t), col] if col in frame.columns else np.nan

rows = []
for w in WIN_ORDER:
    d = json.load(open(R / FILES[w], encoding='utf-8'))
    cd = d.get('cell', {})
    for t in TGT:
        rec = cd.get(f'all|{t}')
        a = rec['agg']
        ps = np.array(a['per_seed_p'])
        b = s18.loc[(w, t)]
        c = cc.loc[(w, t)]
        rows.append({
            'window': w, 'target': t,
            'T_obs': a['T_obs'],
            'sigma_null': a['T_null_std'],
            'null_lo': a['T_null_95ci'][0], 'null_hi': a['T_null_95ci'][1],
            'p_one': a['p_agg_one_sided'],
            'p_two': a['p_agg_two_sided'],
            'q_one': float(b['q_bh']), 'q_two': float(b['q_bh_two_sided']),
            'seed_p_min': ps.min(), 'seed_p_max': ps.max(),
            'seed_p_mean': ps.mean(), 'seed_p_sd': ps.std(ddof=1),
            'MDE80': float(c['MDE80']), 'z_eff': float(c['z_eff']),
            'power': float(c['power_at_T_obs']),
            'p_fisher': float(c['p_fisher']), 'p_stouffer': float(c['p_stouffer']),
        })

df = pd.DataFrame(rows)
pd.set_option('display.width', 260)
print(df.to_string(index=False, float_format=lambda v: f'{v:.6g}'))
df.to_csv(R / 'scalp18_summary.csv', index=False)
print('\n[saved] outputs/results/scalp18_summary.csv')
