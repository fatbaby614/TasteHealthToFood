# -*- coding: utf-8 -*-
"""§3.3 零假设校准与检验性能图表（Array 大修 ARRAY-D-26-07014）。

对应修改计划 paper/Array/ARRAY_major_revision_plan.md §3.3 的四类图：

  fig28  Null p-value histogram / empirical CDF
         （零假设下经验 p 值的分布 + 经验累积分布 + 聚合统计量的零分布）
  fig29  名义 Type I error 与实际拒绝率（含 95% binomial CI）
  fig30  不同训练预算下的检验性能比较（T_obs、σ_null、MDE80、功效）
  fig31  不同种子数下的 observed statistic SD（§4.2 / §4.3 连续稳定性指标）

数据来源（全部为补实验产物，不做任何模拟或外推）：
  outputs/results/seed_inference_audit_calib.json        §3.1 pooled 校准（含 per_rep 原始 p 值）
  outputs/results/seed_inference_audit_calib_within.json  §3.1 within 校准
  outputs/results/audit_posthoc_cells.csv                §3.2 MDE / 功效 / 各配置 T_obs
  outputs/results/audit_posthoc_seedcount.csv            §4.2 种子数敏感性
  outputs/results/audit_posthoc_fpr.csv                  §3.1 拒绝率表（用于交叉核对）
  outputs/results/seed_inference_audit.json              §1.2 观测统计量（用于零分布标线）

用法：
  python code/make_calibration_figures.py
  python code/make_calibration_figures.py --fig-dir paper/Array/figures
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

from config import PROJECT_ROOT, RESULTS_DIR, FIGURES_DIR

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
    'font.size': 8,
    'axes.titlesize': 8.5,
    'axes.labelsize': 8,
    'xtick.labelsize': 7.5,
    'ytick.labelsize': 7.5,
    'legend.fontsize': 7.5,
    'figure.titlesize': 9,
    'axes.linewidth': 0.6,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
})

DPI = 300

# 配色（沿用仓库既有图件色系）
C_POOLED = '#33658A'
C_WITHIN = '#E07B39'
RULE_COLOR = {
    'single': '#7A7A7A',
    'min': '#C0504D',
    'mean': '#E07B39',
    'agg': '#33658A',
    'single_two': '#9C9C9C',
    'agg_two': '#6FA3C7',
}
RULE_LABEL = {
    'single': 'single-seed p',
    'min': 'min-p across seeds',
    'mean': 'mean-p across seeds',
    'agg': 'aggregated T (one-sided)',
    'single_two': 'single-seed p (two-sided)',
    'agg_two': 'aggregated T (two-sided)',
}

# §2.2 训练预算配置。与正文一致：预算扫描本身在 B=2000/S=5 下运行（A 行）；
# B/C 两行展示的是 B=20000/S=5 的高置换重跑（hib 变体，与正文 §3.5 的
# 「Configurations B and C were rerun with B=20000」一致）。
BUDGET_CONFIGS = [
    ('budgetA_pool200', 'A: pooled 200/fold'),
    ('budgetB_persubj200_hib', 'B: per-subject 200'),
    ('budgetC_alltrials_hib', 'C: all trials'),
    ('main_pooled', 'final audit: pooled 200'),
]
BUDGET_TARGETS = ['rating_taste', 'rating_willingnessToEat']
TARGET_LABEL = {
    'rating_taste': 'taste',
    'rating_willingnessToEat': 'willingness',
}
CELL_COLOR = {
    'occipital|rating_taste': '#33658A',
    'occipital|rating_willingnessToEat': '#E07B39',
    'all|rating_taste': '#6FA3C7',
    'all|rating_health': '#9C9C9C',
    'all|rating_willingnessToEat': '#C0504D',
}
CELL_LABEL = {
    'occipital|rating_taste': 'occipital · taste',
    'occipital|rating_willingnessToEat': 'occipital · willingness',
    'all|rating_taste': 'scalp · taste',
    'all|rating_health': 'scalp · health',
    'all|rating_willingnessToEat': 'scalp · willingness',
}


def save_fig(fig, name, fig_dir):
    out = Path(fig_dir) / name
    fig.savefig(out, dpi=DPI, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)
    print(f'[fig] saved: {out}')


# ----------------------------------------------------------------------
# 数据读取
# ----------------------------------------------------------------------
def load_calib(path):
    """读取 §3.1 校准 JSON，返回 per_rep / by_seed_count 等。"""
    p = Path(path)
    if not p.exists():
        print(f'  [skip] 校准文件不存在: {p.name}')
        return None
    d = json.load(open(p, encoding='utf-8'))
    c = dict(d.get('calib', d))
    per_rep = c.get('per_rep') or []
    if not per_rep:
        print(f'  [skip] {p.name} 无 per_rep 字段（旧的校准格式不产原始 p 值）')
        return None
    c['per_rep'] = per_rep
    c['_source'] = p.name
    return c


def clopper_pearson(k, n, alpha=0.05):
    """二项比例的精确（Clopper-Pearson）置信区间。"""
    if n <= 0:
        return (np.nan, np.nan)
    lo = 0.0 if k == 0 else float(stats.beta.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k >= n else float(stats.beta.ppf(1 - alpha / 2, k + 1, n - k))
    return lo, hi


def calib_rejection_table(calibs):
    """由 by_seed_count 还原拒绝计数并给 Clopper-Pearson 95% CI。"""
    rows = []
    for unit, cal in calibs:
        if cal is None:
            continue
        n_rep = int(cal.get('n_rep', 0))
        nominal = float(cal.get('nominal', cal.get('alpha', 0.05)))
        for S, dd in (cal.get('by_seed_count') or {}).items():
            for rule, val in (dd.get('fpr') or {}).items():
                k = int(round(float(val) * n_rep))
                lo, hi = clopper_pearson(k, n_rep)
                rows.append({
                    'source': cal['_source'],
                    'permutation_unit': unit,
                    'S': int(S),
                    'rule': rule,
                    'n_rep': n_rep,
                    'n_rejected': k,
                    'rejection_rate': k / n_rep if n_rep else np.nan,
                    'ci95_lo': lo,
                    'ci95_hi': hi,
                    'nominal_alpha': nominal,
                })
    return pd.DataFrame(rows)


def cross_check_rejection(calibs, df):
    """用 per_rep 直接重算拒绝率，与 posthoc 表核对（打印最大偏差）。"""
    rule_map = {'single': 'p_single', 'min': 'p_min', 'mean': 'p_mean',
                'agg': 'p_agg', 'agg_two': 'p_agg_two_sided'}
    worst, worst_tag = 0.0, ''
    for unit, cal in calibs:
        if cal is None:
            continue
        per_rep = cal['per_rep']
        for S in sorted({int(k) for r in per_rep for k in r}, key=int):
            for rule, key in rule_map.items():
                vals = np.array([r.get(str(S), {}).get(key, np.nan)
                                 for r in per_rep], float)
                vals = vals[np.isfinite(vals)]
                if vals.size == 0:
                    continue
                rate = float((vals <= 0.05).mean())
                sub = df[(df.permutation_unit == unit) & (df.S == S) &
                         (df.rule == rule)]
                if sub.empty:
                    continue
                d = abs(rate - float(sub['rejection_rate'].iloc[0]))
                if d > worst:
                    worst = d
                    worst_tag = (f'{unit} S={S} {rule}: per_rep={rate:.4f} '
                                 f'table={float(sub["rejection_rate"].iloc[0]):.4f}')
    print(f'  [核对] per_rep 重算拒绝率 vs audit_posthoc_fpr.csv '
          f'最大偏差 = {worst:.4f}' + (f'（{worst_tag}）' if worst_tag else ''))
    return worst


def observed_T(results_dir, cell_key, S=None):
    """从 §1.2 主结果 JSON 取观测聚合统计量 T_obs。

    S 给定时按实际 seed 顺序取前 S 个 seed 的 r 求均值，与 §3.1 校准
    （seeds = SEED_POOL 前 10 个、S=10）的零分布口径一致，便于同图对比。
    """
    p = Path(results_dir) / 'seed_inference_audit.json'
    if not p.exists():
        return None, None
    d = json.load(open(p, encoding='utf-8'))
    rec = (d.get('cell') or {}).get(cell_key)
    if not rec:
        return None, None
    if S is not None:
        rs = rec.get('actual_rs') or []
        if len(rs) < S:
            return None, None
        return float(np.mean(rs[:S])), S
    t = (rec.get('agg') or {}).get('T_obs')
    return (float(t) if t is not None else None), None


# ----------------------------------------------------------------------
# fig28：Null p-value histogram / empirical CDF
# ----------------------------------------------------------------------
def fig28_null_p(calibs, results_dir, fig_dir):
    """零假设下经验 p 值的直方图与经验累积分布。

    数据为 §3.1 校准的 per_rep：每次 Monte Carlo null replicate 内部都对
    「观测统计量 vs 该 replicate 自己的置换零分布」算一次 p 值，因此
    1000 个 replicate 的 p 值构成零假设下的经验 p 值分布。
    """
    pooled = dict(calibs)['pooled']
    if pooled is None:
        print('  [skip] fig28：缺少 pooled 校准数据')
        return
    per_rep = pooled['per_rep']
    all_S = {int(k) for r in per_rep for k in r}
    S = 10 if 10 in all_S else max(all_S)
    rules = [('p_single', 'single'), ('p_min', 'min'), ('p_mean', 'mean'),
             ('p_agg', 'agg')]

    fig, axes = plt.subplots(1, 3, figsize=(6.84, 2.45),
                             layout='constrained')

    # (a) 直方图：agg 规则零假设 p 值分布 + 均匀分布期望
    ax = axes[0]
    vals = np.array([r.get(str(S), {}).get('p_agg', np.nan)
                     for r in per_rep], float)
    vals = vals[np.isfinite(vals)]
    bins = np.linspace(0, 1, 21)
    cnt, edges = np.histogram(vals, bins=bins)
    exp = len(vals) * (edges[1] - edges[0])
    ax.bar(edges[:-1], cnt, width=np.diff(edges), align='edge',
           color=C_POOLED, alpha=0.75, edgecolor='white', linewidth=0.4)
    ax.axhline(exp, color='black', linestyle='--', linewidth=0.8,
               label='uniform expectation')
    ax.axvline(0.05, color=C_WITHIN, linewidth=1.0,
               label='nominal $\\alpha$ = 0.05')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, exp * 1.75)
    ax.set_xlabel('Null p-value')
    ax.set_ylabel('Count')
    fpr = float((vals <= 0.05).mean())
    ax.set_title('(a) Null p histogram (agg)', pad=14)
    ax.text(0.5, 1.015, f'S = {S}, n = {len(vals)}, rejection rate = {fpr:.3f}',
            transform=ax.transAxes, ha='center', va='bottom', fontsize=6.5)
    ax.legend(frameon=True, facecolor='white', edgecolor='none', framealpha=0.85,
              loc='upper right', fontsize=6.5)

    # (b) 经验 CDF：四条规则（pooled）
    ax = axes[1]
    for key, rule in rules:
        v = np.array([r.get(str(S), {}).get(key, np.nan) for r in per_rep],
                     float)
        v = v[np.isfinite(v)]
        if v.size == 0:
            continue
        v.sort()
        ax.step(v, np.arange(1, v.size + 1) / v.size, where='post',
                color=RULE_COLOR[rule], linewidth=1.0,
                label=RULE_LABEL[rule])
    ax.plot([0, 1], [0, 1], color='black', linestyle=':', linewidth=0.9,
            label='perfect calibration')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel('Null p-value')
    ax.set_ylabel('Empirical CDF')
    ax.set_title('(b) Empirical CDF of null p', pad=14)
    ax.text(0.5, 1.015, f'pooled, S = {S}', transform=ax.transAxes,
            ha='center', va='bottom', fontsize=6.5)
    # 图例移到图外底部，避免白底框遮挡曲线（与 fig30/fig31 风格一致）
    handles_b, labels_b = ax.get_legend_handles_labels()
    fig.legend(handles=handles_b, labels=labels_b, fontsize=6.8,
               frameon=False, loc='outside lower center', ncols=3)

    # (c) 聚合统计量 T 的零分布 + 观测 T_obs
    ax = axes[2]
    tv = np.array([r.get(str(S), {}).get('T_obs', np.nan) for r in per_rep],
                  float)
    tv = tv[np.isfinite(tv)]
    ax.hist(tv, bins=30, color='#A9C6DC', edgecolor='white', linewidth=0.4)
    # 观测 T_obs 取同一 cell、同一 seed 集前 S 个 seed 的均值（与校准口径一致）
    t_obs, _ = observed_T(results_dir, 'occipital|rating_taste', S=S)
    if t_obs is not None:
        ax.axvline(t_obs, color=C_WITHIN, linewidth=1.2)
        # 标注放右上空白区，避免压住右侧尾部小柱
        ax.text(t_obs, 0.58, f'observed\n$T_{{obs}}$ = {t_obs:.4f}',
                transform=ax.get_xaxis_transform(), ha='right', va='bottom',
                fontsize=6.5, color=C_WITHIN)
    ax.set_xlabel('Aggregated statistic $T$')
    ax.set_ylabel('Count')
    ax.set_title('(c) Null distribution of $T$ (pooled)', pad=14)
    ax.text(0.5, 1.015, f'S = {S}, null mean = {tv.mean():.5f}, '
            f'null SD = {tv.std(ddof=1):.5f}',
            transform=ax.transAxes, ha='center', va='bottom', fontsize=6.5)

    save_fig(fig, 'fig28_calibration_null_p.png', fig_dir)


# ----------------------------------------------------------------------
# fig29：名义 Type I error 与实际拒绝率（含 95% CI）
# ----------------------------------------------------------------------
def fig29_type1(df, fig_dir):
    """名义 α = 0.05 与各规则实际拒绝率的对比（含精确 95% CI）。"""
    if df is None or df.empty:
        print('  [skip] fig29：无拒绝率数据')
        return
    rules = ['single', 'min', 'mean', 'agg', 'single_two', 'agg_two']
    units = [('pooled', C_POOLED), ('within', C_WITHIN)]
    Ss = sorted(df['S'].unique())
    markers = {1: 'o', 2: 's', 3: '^', 5: 'D', 10: 'v', 20: 'P'}

    fig, axes = plt.subplots(1, 2, figsize=(6.84, 2.75), sharey=True,
                             layout='constrained')
    for k, (unit, color) in enumerate(units):
        ax = axes[k]
        sub = df[df.permutation_unit == unit]
        if sub.empty:
            ax.set_visible(False)
            continue
        x = np.arange(len(rules))
        for j, S in enumerate(Ss):
            off = (j - (len(Ss) - 1) / 2) * 0.15
            xs, ys, lo, hi = [], [], [], []
            for i, rule in enumerate(rules):
                r = sub[(sub.S == S) & (sub.rule == rule)]
                if r.empty:
                    continue
                xs.append(i + off)
                ys.append(float(r['rejection_rate'].iloc[0]))
                lo.append(float(r['ci95_lo'].iloc[0]))
                hi.append(float(r['ci95_hi'].iloc[0]))
            if not xs:
                continue
            ax.errorbar(xs, ys, yerr=[np.array(ys) - np.array(lo),
                                      np.array(hi) - np.array(ys)],
                        fmt=markers.get(S, 'o'), color=color,
                        markerfacecolor=color if S == Ss[0] else 'white',
                        markersize=4, capsize=2, linewidth=1.0,
                        label=f'S = {S}')
        ax.axhline(0.05, color='black', linestyle='--', linewidth=0.9)
        ax.axhspan(0.0, 0.05, color='gray', alpha=0.07, zorder=0)
        ax.set_xticks(x)
        ax.set_xticklabels([RULE_LABEL[r] for r in rules], rotation=25,
                           ha='right')
        ax.set_title(f'({"ab"[k]}) {unit} permutation')
        ax.set_ylim(-0.02, 0.55)
        ax.legend(frameon=False, ncol=3, loc='upper left', fontsize=6.8)
        ax.text(len(rules) - 0.45, 0.068, 'nominal 0.05', fontsize=6.5,
                ha='right', va='bottom', color='black')
    axes[0].set_ylabel('Rejection rate at $\\alpha$ = 0.05')
    save_fig(fig, 'fig29_calibration_type1_error.png', fig_dir)


# ----------------------------------------------------------------------
# fig30：不同训练预算下的检验性能比较
# ----------------------------------------------------------------------
def fig30_budget(cells_df, fig_dir):
    """配置 A/B/C 下的 T_obs、σ_null、MDE80 与功效对比（枕区两 cell）。"""
    if cells_df is None or cells_df.empty:
        print('  [skip] fig30：无 cells 表')
        return
    need = ['source', 'roi', 'target', 'T_obs', 'sigma_null', 'MDE80',
            'power_at_T_obs', 'n_perm_B', 'n_seeds_S']
    if any(c not in cells_df.columns for c in need):
        print('  [skip] fig30：cells 表缺列')
        return
    df = cells_df[cells_df['roi'] == 'occipital']
    df = df[df['source'].isin([c for c, _ in BUDGET_CONFIGS])]
    if df.empty:
        print('  [skip] fig30：cells 表无枕区行')
        return

    configs = [(c, lab) for c, lab in BUDGET_CONFIGS
               if c in set(df['source'])]
    y = np.arange(len(configs))[::-1]   # 第一个配置在顶部
    h = 0.34

    # 每个配置的 (B, S) 标注，直接从表中读，避免人工转录
    sub_labels = []
    for c, lab in configs:
        r = df[df.source == c]
        B = int(r['n_perm_B'].iloc[0]) if not r.empty else 0
        S = int(r['n_seeds_S'].iloc[0]) if not r.empty else 0
        sub_labels.append(f'{lab}\n(B={B}, S={S})')

    fig, axes = plt.subplots(1, 3, figsize=(6.84, 2.35), layout='constrained')

    def draw_bars(ax, col, title, xlab, xlim=None):
        for j, target in enumerate(BUDGET_TARGETS):
            ys, xs = [], []
            for i, (c, _) in enumerate(configs):
                r = df[(df.source == c) & (df.target == target)]
                if r.empty:
                    continue
                ys.append(y[i] + (0.5 - j) * h)
                xs.append(float(r[col].iloc[0]))
            if ys:
                ax.barh(ys, xs, height=h * 0.92,
                        color=CELL_COLOR[f'occipital|{target}'], alpha=0.85,
                        edgecolor='white', linewidth=0.4,
                        label=TARGET_LABEL.get(target, target))
        ax.set_yticks(y)
        ax.set_yticklabels(sub_labels, fontsize=6.6)
        ax.set_xlabel(xlab)
        ax.set_title(title)
        if xlim:
            ax.set_xlim(*xlim)

    # (a) T_obs（误差棒为 σ_null）
    ax = axes[0]
    for j, target in enumerate(BUDGET_TARGETS):
        ys, tobs, sig = [], [], []
        for i, (c, _) in enumerate(configs):
            r = df[(df.source == c) & (df.target == target)]
            if r.empty:
                continue
            ys.append(y[i] + (0.5 - j) * h)
            tobs.append(float(r['T_obs'].iloc[0]))
            sig.append(float(r['sigma_null'].iloc[0]))
        if not ys:
            continue
        ax.barh(ys, tobs, height=h * 0.92,
                color=CELL_COLOR[f'occipital|{target}'], alpha=0.85,
                edgecolor='white', linewidth=0.4,
                label=TARGET_LABEL.get(target, target))
        ax.errorbar(tobs, ys, xerr=sig, fmt='none', ecolor='black',
                    elinewidth=0.9, capsize=2.5)
    ax.set_yticks(y)
    ax.set_yticklabels(sub_labels, fontsize=6.6)
    ax.set_xlabel('Aggregated statistic (correlation units)')
    ax.set_title('(a) $T_{obs}$ (error bar = $\\sigma_{null}$)')

    # (b) MDE80
    draw_bars(axes[1], 'MDE80', '(b) MDE at 80% power',
              'MDE (correlation units)')

    # (c) 功效
    draw_bars(axes[2], 'power_at_T_obs',
              '(c) Power at observed $T_{obs}$', 'Power', xlim=(0, 1.05))

    # 共用图例（放在整幅图下方，避免压住条形与标题）
    handles = [plt.Rectangle((0, 0), 1, 1,
                             color=CELL_COLOR[f'occipital|{t}'], alpha=0.85)
               for t in BUDGET_TARGETS]
    fig.legend(handles, [TARGET_LABEL.get(t, t) for t in BUDGET_TARGETS],
               loc='outside lower center', ncol=2, frameon=False,
               fontsize=7.5)

    save_fig(fig, 'fig30_calibration_budget_performance.png', fig_dir)


# ----------------------------------------------------------------------
# fig31：不同种子数下的 observed statistic SD
# ----------------------------------------------------------------------
def fig31_seedcount(seed_df, fig_dir):
    """跨种子离散度与 p 值随种子数 S 的变化（§4.2 / §4.3）。"""
    if seed_df is None or seed_df.empty:
        print('  [skip] fig31：无 seedcount 表')
        return
    df = seed_df[seed_df['source'] == 'main_pooled']
    if df.empty:
        df = seed_df
    df = df.copy()
    df['cell'] = df['roi'] + '|' + df['target']
    # S=20 时只有 1 种种子组合，SD 退化（=0），不能作为稳定性证据，剔除
    df = df[df['n_combinations'] >= 2]

    fig, axes = plt.subplots(1, 3, figsize=(6.84, 2.5), layout='constrained')

    def series(ax, col, cell_list, logy=False):
        for cell in cell_list:
            sub = df[df.cell == cell].sort_values('S')
            if sub.empty:
                continue
            ax.plot(sub['S'], sub[col], marker='o', markersize=4,
                    color=CELL_COLOR.get(cell, '#7A7A7A'), linewidth=1.1,
                    linestyle='-' if cell.startswith('occipital') else '--',
                    label=CELL_LABEL.get(cell, cell))
        ax.set_xscale('log')
        ax.set_xticks([1, 2, 3, 5, 10, 20])
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        if logy:
            ax.set_yscale('log')

    occ = ['occipital|rating_taste', 'occipital|rating_willingnessToEat']
    scalp = ['all|rating_taste', 'all|rating_health',
             'all|rating_willingnessToEat']

    # (a) across-seed SD of observed statistic
    ax = axes[0]
    series(ax, 'T_obs_sd', occ + scalp)
    ax.axhline(0.006, color='black', linestyle='--', linewidth=0.9)
    ax.text(0.97, 0.0064, 'manuscript threshold 0.006', fontsize=6.5,
            color='black', ha='right', va='bottom',
            transform=ax.get_yaxis_transform())
    ax.set_xlabel('Number of random seeds S')
    ax.set_ylabel('Across-seed SD of $T_{obs}$')
    ax.set_title('(a) Stability of $T_{obs}$')

    # (b) 跨种子 p 均值
    ax = axes[1]
    series(ax, 'p_mean', occ + scalp, logy=True)
    ax.axhline(0.05, color='black', linestyle='--', linewidth=0.9)
    ax.set_xlabel('Number of random seeds S')
    ax.set_ylabel('Mean seed-level p-value')
    ax.set_title('(b) Seed-level p convergence')

    # (c) p ≤ 0.05 的比例
    ax = axes[2]
    series(ax, 'frac_p_le_0.05', occ + scalp)
    ax.set_xlabel('Number of random seeds S')
    ax.set_ylabel('Fraction with p $\\leq$ 0.05')
    ax.set_ylim(-0.03, 1.03)
    ax.set_title('(c) Seed subsets reaching $\\alpha$')

    # 共用图例放在整幅图下方
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside lower center', ncol=3,
               frameon=False, fontsize=7)

    save_fig(fig, 'fig31_calibration_seedcount_stability.png', fig_dir)


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results-dir', default=str(RESULTS_DIR))
    ap.add_argument('--fig-dir', default=str(FIGURES_DIR))
    args = ap.parse_args()
    results_dir = Path(args.results_dir)
    fig_dir = Path(args.fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    print(f'[fig] results = {results_dir}')
    print(f'[fig] figures = {fig_dir}')

    pooled = load_calib(results_dir / 'seed_inference_audit_calib.json')
    within = load_calib(results_dir / 'seed_inference_audit_calib_within.json')
    calibs = [('pooled', pooled), ('within', within)]

    # 拒绝率表 + 交叉核对
    rej_df = calib_rejection_table(calibs)
    if not rej_df.empty:
        cross_check_rejection(calibs, rej_df)
        out_csv = results_dir / 'audit_calibration_rejection_ci.csv'
        rej_df.to_csv(out_csv, index=False, encoding='utf-8-sig')
        print(f'[fig] 拒绝率 CI 表已写出: {out_csv}')

    cells_df = pd.read_csv(results_dir / 'audit_posthoc_cells.csv') \
        if (results_dir / 'audit_posthoc_cells.csv').exists() else None
    seed_df = pd.read_csv(results_dir / 'audit_posthoc_seedcount.csv') \
        if (results_dir / 'audit_posthoc_seedcount.csv').exists() else None

    fig28_null_p(calibs, results_dir, fig_dir)
    fig29_type1(rej_df, fig_dir)
    fig30_budget(cells_df, fig_dir)
    fig31_seedcount(seed_df, fig_dir)
    print('[fig] 完成')


if __name__ == '__main__':
    main()
