# -*- coding: utf-8 -*-
"""
audit_posthoc_analysis.py
=========================
Array 大修离线后处理（§3.1 / §3.3 / §4.1 / §4.2）——纯读取已有 JSON，
不重跑任何解码，秒级完成。

输入（默认 outputs/results/，存在才处理，缺失则提示后跳过）：
  seed_inference_audit.json                §1.2 主结果（pooled 置换；
                                           枕区 B=5000 S=20，全脑 B=2000 S=5）
  seed_inference_audit_within.json         §1.3 within-subject 置换单位
  budget_audit_pool.json                   §2 配置 A（总预算 200/fold）
  budget_audit_per_subject.json            §2 配置 B（200/被试）
  budget_audit_all.json                    §2 配置 C（全部训练 trial）
  seed_inference_audit_calib.json          §3.1 第一类错误率校准（pooled）
  seed_inference_audit_calib_within.json   §3.1（within）
  seed_inference_audit_scalp_{N1,P2,P3,LPP,LATE}.json
                                §1.4 全脑 18-cell 家族（s1d，5 时间窗 ×
                                3 target；full 窗 3 cell 复用主结果文件）

产出（同目录）：
  audit_posthoc_cells.csv       每 cell 一行：T_obs、σ_null、MDE80、
                                观测统计量对应功效、Fisher / Stouffer
                                合并 p、p 分辨率（§3.3 + §4.1）
  audit_posthoc_seedcount.csv   §4.2 seed 数敏感性（cell × S'）
  audit_posthoc_fpr.csv         §3.1 校准 FPR 表（unit × S' × 规则）
并打印控制台汇总表。

统计口径
--------
MDE80（80% 功效最小可检测效应）= (z_0.95 + z_0.80) * sigma_null
                               ≈ 2.4865 * sigma_null（单侧上尾，与 p_agg 口径一致）；
                               双侧参考值 = (z_0.975 + z_0.80) * sigma_null ≈ 2.8016 * sigma_null；
功效(T) = Phi( (T - mu_null)/sigma_null - z_0.95 )（单侧）；双侧参考用 z_0.975。
Fisher  X = -2 * sum_s ln p_s ~ chi2_{2S}；
Stouffer Z = sum_s Phi^{-1}(1 - p_s) / sqrt(S)（单侧）。
p 分辨率 = 1/(B+1)。
"""
from __future__ import annotations

import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

Z_95 = 1.644854    # Phi^{-1}(0.95)  —— 单侧上尾临界值（p_agg 用此口径）
Z_975 = 1.959964   # Phi^{-1}(0.975) —— 双侧参考
Z_80 = 0.841621    # Phi^{-1}(0.80)
MDE_K = Z_95 + Z_80         # ≈ 2.4865（单侧主口径，与 p_agg 上尾一致）
MDE_K_TWO = Z_975 + Z_80    # ≈ 2.8016（双侧参考）

CELL_FILES = [
    ('seed_inference_audit.json', 'main_pooled'),
    ('seed_inference_audit_within.json', 'within'),
    ('budget_audit_pool.json', 'budgetA_pool200'),
    ('budget_audit_per_subject.json', 'budgetB_persubj200'),
    ('budget_audit_all.json', 'budgetC_alltrials'),
    # Array 大修补跑：21-cell ROI 家族（§1.4 正式检验家族）
    ('seed_inference_audit_roi_family.json', 'roi_family_pooled'),
    ('seed_inference_audit_roi_family_within.json', 'roi_family_within'),
    ('seed_inference_audit_scalp_within.json', 'scalp_within'),
    ('budget_audit_per_subject_scalp.json', 'budgetB_persubj200_scalp'),
    ('budget_audit_all_scalp.json', 'budgetC_alltrials_scalp'),
    # 提高置换次数以突破 p 分辨率下限（§1.4）
    ('budget_audit_per_subject_hib.json', 'budgetB_persubj200_hib'),
    ('budget_audit_all_hib.json', 'budgetC_alltrials_hib'),
    # Array 大修第三轮：全脑 18-cell 时间窗家族（6 窗 × 3 target，§1.4）。
    # full 窗 3 cell 复用主结果文件，其余 5 窗各一个文件；
    # 共用 label 使 build_bh_table 按 source 聚合 → 18 行。
    ('seed_inference_audit.json', 'scalp18_pooled'),
    ('seed_inference_audit_scalp_N1.json', 'scalp18_pooled'),
    ('seed_inference_audit_scalp_P2.json', 'scalp18_pooled'),
    ('seed_inference_audit_scalp_P3.json', 'scalp18_pooled'),
    ('seed_inference_audit_scalp_LPP.json', 'scalp18_pooled'),
    ('seed_inference_audit_scalp_LATE.json', 'scalp18_pooled'),
]
# BH 家族定义：label → 该 label 下参与校正的 cell 名单
# None 表示"该 source 自身全部 cell"；显式列表表示指定子集。
BH_FAMILIES = {
    'roi_family_pooled': None,      # 21 个 ROI×target（原稿的 ROI 家族）
    'roi_family_within': None,
    'scalp_within': None,
    'main_pooled': ['occipital|rating_taste',
                    'occipital|rating_willingnessToEat',
                    'all|rating_taste', 'all|rating_health',
                    'all|rating_willingnessToEat'],
    # 全脑 18-cell 家族：6 时间窗 × 3 target（full 窗来自主结果文件，
    # 其余 5 窗来自 s1d；keep 只写 all|* 三键，跨 6 文件聚合）
    'scalp18_pooled': ['all|rating_taste',
                       'all|rating_health',
                       'all|rating_willingnessToEat'],
}
CALIB_FILES = [
    ('seed_inference_audit_calib.json', 'pooled'),
    ('seed_inference_audit_calib_within.json', 'within'),
]


def _norm_budget(v):
    if isinstance(v, str):
        return v
    return 'inf' if float(v) >= 10 ** 9 else int(v)


def load_cells(results_dir):
    """汇总所有 cell 结果文件 → [(label, rec)]。"""
    out = []
    for fname, label in CELL_FILES:
        p = results_dir / fname
        if not p.exists():
            print(f'  [skip] {fname} 不存在（对应阶段尚未运行）')
            continue
        d = json.load(open(p, encoding='utf-8'))
        cells = d.get('cell', {})
        if isinstance(cells, list):
            cells = {f"{r['roi']}|{r['target']}": r for r in cells}
        for rec in cells.values():
            # 旧格式记录缺省即原口径
            rec.setdefault('unit', 'pooled')
            rec.setdefault('semantics', 'pool')
            rec.setdefault('max_trials', 200)
            # 时间窗（§1.4 18-cell 家族需要区分 6 窗；旧文件缺省 full 窗）
            rec.setdefault('window',
                           d.get('meta', {}).get('window', 'full_0-1000ms'))
            out.append((label, rec))
        n = len(cells)
        print(f'  [load] {fname}: {n} cell')
    return out


def fisher_stouffer(per_seed_p):
    """由 per-seed p（correct 口径）计算 Fisher 与 Stouffer 合并检验。"""
    p = np.maximum(np.asarray(per_seed_p, float), 1e-300)
    S = len(p)
    x_fisher = -2.0 * np.sum(np.log(p))
    p_fisher = float(stats.chi2.sf(x_fisher, 2 * S))
    z = float(np.sum(stats.norm.ppf(1.0 - p)) / np.sqrt(S))
    p_stouffer = float(stats.norm.sf(z))
    return p_fisher, p_stouffer


def bh_adjust(pvals):
    """Benjamini–Hochberg 校正，返回与输入同序的 q 值。

    输入 p 必须全部有效（0<p<=1）。NaN 位置返回 NaN 并排除在
    家族规模 m 之外（无效检验不计入检验家族）。
    """
    p = np.asarray(pvals, float)
    q = np.full(p.shape, np.nan)
    valid = np.isfinite(p) & (p > 0)
    idx = np.flatnonzero(valid)
    if len(idx) == 0:
        return q
    pv = p[idx]
    m = len(pv)
    order = np.argsort(pv)
    ranked = pv[order]
    qv = ranked * m / np.arange(1, m + 1)
    # 自后向前取累计最小（保证 q 单调）
    qv = np.minimum.accumulate(qv[::-1])[::-1]
    qv = np.clip(qv, 0, 1)
    out = np.empty(m)
    out[order] = qv
    q[idx] = out
    return q


def build_bh_table(cells):
    """在预定义检验家族内做 BH 校正（§1.4 / §30 BH-FDR 要求）。

    家族划分见 BH_FAMILIES；每个家族内用正式 p 值（p_agg，按 --sided
    事先固定的方向口径）做 BH，输出 q 值与是否通过。
    """
    by_source = {}
    for label, rec in cells:
        by_source.setdefault(label, []).append(rec)

    rows = []
    for label, recs in by_source.items():
        keep = BH_FAMILIES.get(label, '__SELF__')
        if keep == '__SELF__':
            keep = None
        if keep is not None:
            keep_set = set(keep)
            recs = [r for r in recs
                    if f"{r['roi']}|{r['target']}" in keep_set]
        if not recs:
            continue
        # 家族完整性保护：scalp18 必须 6 窗 × 3 target = 18 齐全才输出，
        # 否则 m 不等于预注册家族规模，中间表会被误读为正式结果。
        if label == 'scalp18_pooled' and len(recs) != 18:
            print(f'  [skip-bh] {label}: 仅 {len(recs)}/18 cell'
                  '（s1d 尚未跑齐，不出中间 BH 表）')
            continue
        # 正式 p：agg['p_agg'] 已在产出阶段按 --sided 事先固定的方向口径
        # 写入（JSON 的 meta.sided_primary 记录了当时的选择）。
        # BH 一律用该正式 p，不在这里重新选择方向。
        pvals = [r['agg'].get('p_agg') for r in recs]
        if any(p is None for p in pvals):
            continue
        sided_used = {r['agg'].get('sided_primary', 'one') for r in recs}
        if len(sided_used) > 1:
            raise ValueError(
                f'家族 {label} 内方向口径不一致（{sided_used}）；'
                '正式 p 必须同口径，请用 --sided 统一重跑该家族')
        qvals = bh_adjust(pvals)
        # 双侧敏感性 q（§1.4 配套要求）：家族内全部 cell 都有精确双侧 p
        # 才计算，否则整列置 NaN —— 绝不用单侧 p 冒充双侧。
        p_two_list = [r['agg'].get('p_agg_two_sided') for r in recs]
        if all(p is not None and np.isfinite(p) for p in p_two_list):
            q_two_vals = bh_adjust(p_two_list)
        else:
            q_two_vals = [float('nan')] * len(recs)
        m = len(recs)
        for r, p, q, q2 in zip(recs, pvals, qvals, q_two_vals):
            rows.append({
                'family': label,
                'roi': r['roi'], 'target': r['target'],
                'window': r.get('window', 'full_0-1000ms'),
                'n_tests_in_family': m,
                'sided_primary': sorted(sided_used)[0],
                'n_perm_B': r.get('n_perm'),
                'n_seeds_S': len(r.get('seeds', [])),
                'unit': r.get('unit', 'pooled'),
                'semantics': r.get('semantics', 'pool'),
                'max_trials': _norm_budget(r.get('max_trials', 200)),
                'T_obs': round(r['agg']['T_obs'], 6),
                'p_agg': round(float(p), 8),
                # 精确置换双侧 p：仅新产出（含 p_agg_two_sided 字段）的
                # JSON 才有；旧 JSON 缺失时置 NaN，绝不用单侧值冒充。
                'p_agg_two_sided_exact': (
                    round(float(r['agg']['p_agg_two_sided']), 8)
                    if 'p_agg_two_sided' in r['agg'] else float('nan')),
                'p_agg_one_sided': (
                    round(float(r['agg']['p_agg_one_sided']), 8)
                    if 'p_agg_one_sided' in r['agg'] else float('nan')),
                'q_bh': round(float(q), 8),
                'significant_bh_0.05': int(q <= 0.05),
                # 双侧敏感性（§1.4 配套要求 2）：同一家族内用精确双侧 p
                # 重算的 q；缺双侧 p 时为 NaN
                'q_bh_two_sided': (round(float(q2), 8)
                                   if np.isfinite(q2) else float('nan')),
                'significant_bh_two_sided_0.05': (
                    int(q2 <= 0.05) if np.isfinite(q2) else ''),
            })
    return pd.DataFrame(rows)


def build_cells_table(cells):
    rows = []
    for label, rec in cells:
        agg = rec['agg']
        mu, sd = agg['T_null_mean'], agg['T_null_std']
        T_obs, B = agg['T_obs'], rec['n_perm']
        if sd and sd > 0:
            z_eff = (T_obs - mu) / sd
            power = float(stats.norm.cdf(z_eff - Z_95))
            power_two = float(stats.norm.cdf(z_eff - Z_975))
        else:
            z_eff, power, power_two = float('nan'), float('nan'), float('nan')
        p_fisher, p_stouffer = fisher_stouffer(agg['per_seed_p'])
        # 双侧参考（正态近似）：p_agg 用单侧上尾，双侧值由 z_eff 反推，
        # 用于评估"方向性假设"对边界 cell 结论的影响（§1.2 要求事先固定方向）
        p_two_norm = (float(2 * stats.norm.sf(abs(z_eff)))
                      if np.isfinite(z_eff) else float('nan'))
        rows.append({
            'source': label,
            'roi': rec['roi'], 'target': rec['target'],
            'window': rec.get('window', 'full_0-1000ms'),
            'unit': rec.get('unit', 'pooled'),
            'semantics': rec.get('semantics', 'pool'),
            'max_trials': _norm_budget(rec.get('max_trials', 200)),
            'n_subjects': rec.get('n_subjects'),
            'n_perm_B': B,
            'n_seeds_S': len(rec['seeds']),
            'T_obs': round(T_obs, 6),
            'mu_null': round(mu, 6), 'sigma_null': round(sd, 6),
            'MDE80': round(MDE_K * sd, 6) if sd else float('nan'),
            'MDE80_over_sigma': round(MDE_K, 4),
            'MDE80_two_sided_ref': round(MDE_K_TWO * sd, 6) if sd else float('nan'),
            'z_eff': round(z_eff, 4),
            'power_at_T_obs': round(power, 4),
            'power_at_T_obs_two_sided_ref': round(power_two, 4),
            'p_agg': round(agg['p_agg'], 6),
            'p_agg_two_sided_normal_ref': round(p_two_norm, 6),
            'sig_agg_two_sided_ref_0.05': (int(p_two_norm <= 0.05)
                                           if np.isfinite(p_two_norm) else ''),
            'p_fisher': round(p_fisher, 6),
            'p_stouffer': round(p_stouffer, 6),
            'p_single_seed1': round(agg['p_single'], 6),
            'p_min': round(agg['p_min'], 6),
            'p_mean': round(agg['p_mean'], 6),
            'p_resolution_1_over_B+1': round(1.0 / (B + 1), 8),
            'p_agg_censored_at_floor': int(agg['p_agg'] <= 1.0 / (B + 1) + 1e-12),
            'sig_agg_0.05': int(agg['p_agg'] <= 0.05),
            'sig_fisher_0.05': int(p_fisher <= 0.05),
            'sig_stouffer_0.05': int(p_stouffer <= 0.05),
        })
    return pd.DataFrame(rows)


def build_seedcount_table(cells):
    rows = []
    for label, rec in cells:
        for Ssize, d in sorted(rec.get('by_seed_count', {}).items(),
                               key=lambda kv: int(kv[0])):
            rows.append({
                'source': label,
                'roi': rec['roi'], 'target': rec['target'],
                'unit': rec.get('unit', 'pooled'),
                'semantics': rec.get('semantics', 'pool'),
                'max_trials': _norm_budget(rec.get('max_trials', 200)),
                'S': int(Ssize),
                'n_combinations': d.get('n_combinations'),
                'T_obs_mean': round(d['T_obs_mean'], 6),
                'T_obs_sd': round(d.get('T_obs_std', float('nan')), 6),
                'p_mean': round(d['p_mean'], 6),
                'p_sd': round(d.get('p_std', float('nan')), 6),
                'p_min': round(d['p_min'], 6),
                'p_max': round(d['p_max'], 6),
                'frac_p_le_0.05': round(d['frac_p_le_0.05'], 4),
            })
    return pd.DataFrame(rows)


def build_fpr_table(results_dir):
    rows = []
    for fname, unit_label in CALIB_FILES:
        p = results_dir / fname
        if not p.exists():
            print(f'  [skip] {fname} 不存在（对应阶段尚未运行）')
            continue
        d = json.load(open(p, encoding='utf-8'))
        c = d.get('calib', d)  # 兼容：校准记录可能嵌套在 'calib' 键下
        for Ssize, dd in sorted(c.get('by_seed_count', {}).items(),
                                key=lambda kv: int(kv[0])):
            for rule, fpr in dd['fpr'].items():
                rows.append({
                    'source': fname,
                    'permutation_unit': c.get('unit', unit_label),
                    'cell': f"{c['cell']['roi']}|{c['cell']['target']}",
                    'S': int(Ssize),
                    'rule': rule,
                    'fpr': round(float(fpr), 4),
                    'fpr_se': round(float(dd['fpr_se'][rule]), 4),
                    'mean_p': round(float(dd['mean_p'][rule]), 4),
                    'n_rep': c.get('n_rep'),
                    'nominal_alpha': c.get('alpha', 0.05),
                })
        print(f'  [load] {fname}: FPR 表')
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description='Array 大修离线后处理')
    ap.add_argument('--dir', default='outputs/results',
                    help='结果 JSON 所在目录')
    ap.add_argument('--out-dir', default='', help='CSV 输出目录（默认同 --dir）')
    args = ap.parse_args()

    results_dir = Path(args.dir)
    out_dir = Path(args.out_dir) if args.out_dir else results_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print('=' * 72)
    print('Array 大修离线后处理 (audit_posthoc_analysis)')
    print(f'  输入目录: {results_dir}')
    print('=' * 72)

    cells = load_cells(results_dir)
    if not cells:
        print('[warn] 没有任何 cell 结果可处理，请先运行审计主脚本。')
        return

    df_cells = build_cells_table(cells)
    df_sc = build_seedcount_table(cells)
    df_fpr = build_fpr_table(results_dir)
    df_bh = build_bh_table(cells)

    p1 = out_dir / 'audit_posthoc_cells.csv'
    p2 = out_dir / 'audit_posthoc_seedcount.csv'
    p3 = out_dir / 'audit_posthoc_fpr.csv'
    p4 = out_dir / 'audit_posthoc_bh.csv'
    df_cells.to_csv(p1, index=False, encoding='utf-8-sig')
    df_sc.to_csv(p2, index=False, encoding='utf-8-sig')
    if len(df_fpr):
        df_fpr.to_csv(p3, index=False, encoding='utf-8-sig')
    if len(df_bh):
        df_bh.to_csv(p4, index=False, encoding='utf-8-sig')

    show = ['source', 'roi', 'target', 'window', 'unit', 'semantics',
            'max_trials', 'n_perm_B', 'n_seeds_S', 'T_obs', 'sigma_null',
            'MDE80', 'power_at_T_obs', 'p_agg', 'p_agg_two_sided_normal_ref',
            'p_agg_censored_at_floor', 'p_fisher', 'p_stouffer', 'p_min',
            'p_mean']
    print('\n== §3.3 功效/MDE + §4.1 Fisher 对照（主表） ==')
    print(df_cells[show].to_string(index=False))

    if len(df_sc):
        print('\n== §4.2 seed 数敏感性 ==')
        print(df_sc[['source', 'roi', 'target', 'unit', 'S', 'T_obs_mean',
                     'p_mean', 'frac_p_le_0.05']].to_string(index=False))
    if len(df_fpr):
        print('\n== §3.1 第一类错误率校准 ==')
        print(df_fpr.to_string(index=False))
    if len(df_bh):
        print('\n== §1.4 BH 家族校正（正式 p 值 = p_agg）==')
        _show_bh = ['family', 'roi', 'target', 'window', 'n_tests_in_family',
                    'n_perm_B', 'n_seeds_S', 'T_obs', 'p_agg',
                    'sided_primary', 'p_agg_two_sided_exact', 'q_bh',
                    'significant_bh_0.05', 'q_bh_two_sided',
                    'significant_bh_two_sided_0.05']
        print(df_bh[_show_bh].to_string(index=False))

    print(f'\n[saved] {p1}')
    print(f'[saved] {p2}')
    if len(df_fpr):
        print(f'[saved] {p3}')
    if len(df_bh):
        print(f'[saved] {p4}')


if __name__ == '__main__':
    main()
