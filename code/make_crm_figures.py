# -*- coding: utf-8 -*-
"""生成 Cell Reports Methods 稿件主图（Figure 1–5）。

数据来源（全部为项目内可溯源产物，不臆造）：
  Figure 2 : outputs/results/seed_inference_audit.json
  Figure 3 : outputs/results/seed_inference_audit_calib.json
             + outputs/results/seed_inference_audit.json
  Figure 4 : outputs/results/budget_sensitivity.csv
  Figure 5 : outputs/results/svr_roi_channels_summary.json
             + outputs/results/scsr_per_subject.csv
  Figure 1 : 概念示意图（无数据）

输出：paper/CellReportsMethods/figures/Figure{1..5}.pdf 与 .png(300dpi)

用法：python code/make_crm_figures.py
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / 'outputs' / 'results'
FIGDIR = ROOT / 'paper' / 'CellReportsMethods' / 'figures'
FIGDIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    'font.size': 8,
    'axes.titlesize': 8.5,
    'axes.labelsize': 8,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 7,
    'axes.linewidth': 0.7,
    'font.family': 'DejaVu Sans',
    'pdf.fonttype': 42,
    'savefig.dpi': 300,
})

C_TASTE = '#C0504D'
C_WILL = '#4F81BD'
C_AGG = '#2E7D32'
C_GREY = '#7F7F7F'
C_RULES = {'single': '#E8A33D', 'min': '#B23A48',
           'mean': C_GREY, 'agg': C_AGG}
RULE_LABEL = {'single': 'First seed', 'min': 'Minimum $p$',
              'mean': 'Mean $p$', 'agg': 'Aggregated'}

AUDIT = RES / 'seed_inference_audit.json'
CALIB = RES / 'seed_inference_audit_calib.json'
BUDGET = RES / 'budget_sensitivity.csv'
ROI = RES / 'svr_roi_channels_summary.json'
SCSR = RES / 'scsr_per_subject.csv'

CELL_LABEL = {
    'occipital|rating_taste': 'Occipital / taste',
    'occipital|rating_willingnessToEat': 'Occipital / willingness',
    'all|rating_taste': 'Whole-scalp / taste',
    'all|rating_health': 'Whole-scalp / health',
    'all|rating_willingnessToEat': 'Whole-scalp / willingness',
}


def save(fig, name):
    for ext in ('pdf', 'png'):
        fig.savefig(FIGDIR / f'{name}.{ext}', bbox_inches='tight')
    plt.close(fig)
    print(f'[saved] {FIGDIR / (name + ".pdf")}')


# ============================================================
# Figure 1 — framework schematic
# ============================================================
def fig1():
    fig, ax = plt.subplots(figsize=(7.0, 4.3))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 62)
    ax.axis('off')

    def box(x, y, w, h, text, fc='#F2F2F2', ec='#404040', fs=7.2, bold=False):
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h, boxstyle='round,pad=0.6,rounding_size=1.6',
            linewidth=0.8, facecolor=fc, edgecolor=ec, zorder=2))
        ax.text(x + w / 2, y + h / 2, text, ha='center', va='center',
                fontsize=fs, zorder=3,
                fontweight='bold' if bold else 'normal')

    def arrow(x1, y1, x2, y2, style='-|>', color='#404040', ls='-'):
        ax.add_patch(FancyArrowPatch(
            (x1, y1), (x2, y2), arrowstyle=style, mutation_scale=9,
            linewidth=0.9, color=color, linestyle=ls, zorder=1))

    y1 = 47
    box(2, y1, 20, 11, 'Single-trial\ncovariance matrices\n$C_i \\in \\mathbb{R}^{64\\times64}$')
    box(27, y1, 21, 11, 'Riemannian tangent\nspace (log-Euclidean,\nshrinkage covariance)')
    box(53, y1, 21, 11, 'Leakage-safe LOSO\ndecoding (transforms\nfitted on train only)')
    box(79, y1, 19, 11, 'Randomised\ntraining-trial\nsubsampling', fc='#FDE9D9',
        ec='#C0504D')

    for x in (22.5, 48.5, 74.5):
        arrow(x, y1 + 5.5, x + 4.2, y1 + 5.5)

    # 回到 3 的循环箭头（重复 S 个 seed）
    ax.add_patch(FancyArrowPatch(
        (88.5, y1 - 0.6), (63.5, y1 - 0.6), arrowstyle='-|>',
        mutation_scale=9, linewidth=0.9, color='#C0504D',
        connectionstyle='arc3,rad=0.28', linestyle='--', zorder=1))
    ax.text(76, y1 - 6.0, 'repeat for $S$ documented seeds',
            fontsize=7, color='#C0504D', ha='center')

    y2 = 27
    box(20, y2, 26, 11, 'Seed-stability summary\n(across-seed SD / CV)\n-- separate quantity --',
        fc='#EDF3FA', ec=C_WILL)
    box(54, y2, 24, 11, 'Seed-aggregated statistic\n$T_{\\mathrm{obs}}=\\frac{1}{S}\\sum_s T_s$',
        fc='#E8F5E9', ec=C_AGG)
    arrow(63.5, y1, 33, y2 + 11.6)
    arrow(72, y1, 66, y2 + 11.6)

    y3 = 9
    box(30, y3, 26, 10, 'Shared-shuffle permutation\nnull of $T_{\\mathrm{obs}}$\n$p=\\frac{1+\\#\\{T^{(b)}\\geq T_{\\mathrm{obs}}\\}}{B+1}$',
        fc='#E8F5E9', ec=C_AGG)
    box(62, y3, 20, 10, 'BH-FDR control\nacross the\ncontrast family')
    arrow(66, y2, 55, y3 + 10.4)
    arrow(78, y2, 72, y3 + 10.4)
    arrow(56, y3 + 5, 62, y3 + 5)

    ax.text(50, 59, 'Auditable inference for stochastic single-trial EEG decoding',
            ha='center', fontsize=8.5, fontweight='bold')
    save(fig, 'Figure1')


# ============================================================
# Figure 2 — seed dependence (occipital cells)
# ============================================================
def cell_map(audit):
    """audit['cell'] 为 list；转成 'roi|target' -> 记录 的字典。"""
    return {f"{e['roi']}|{e['target']}": e for e in audit['cell']}


def fig2(audit):
    CM = cell_map(audit)
    cells = ['occipital|rating_taste', 'occipital|rating_willingnessToEat']
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.7))

    ax = axes[0]
    for i, ck in enumerate(cells):
        c = CM[ck]
        rs = np.array(c['actual_rs'])
        xs = np.full(len(rs), i) + np.linspace(-0.12, 0.12, len(rs))
        ax.plot(xs, rs, 'o', ms=3.2, color=[C_TASTE, C_WILL][i],
                label=['Taste', 'Willingness'][i])
        ax.hlines(rs.mean(), i - 0.25, i + 0.25, color='k', lw=1.0)
        ax.hlines(rs.mean() - rs.std(), i - 0.25, i + 0.25, color='k',
                  lw=0.6, ls=':')
        ax.hlines(rs.mean() + rs.std(), i - 0.25, i + 0.25, color='k',
                  lw=0.6, ls=':')
    ax.axhline(0, color='#AAA', lw=0.7, ls='--')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Taste', 'Willingness'])
    ax.set_ylabel('Observed statistic $r$')
    ax.set_title('A  Statistic per seed')
    ax.legend(frameon=False, loc='upper left')

    ax = axes[1]
    for i, ck in enumerate(cells):
        c = CM[ck]['agg']
        ps = np.array(c['per_seed_p'])
        xs = np.full(len(ps), i) + np.linspace(-0.12, 0.12, len(ps))
        ax.plot(xs, ps, 'o', ms=3.2, color=[C_TASTE, C_WILL][i])
        ax.hlines(ps.mean(), i - 0.25, i + 0.25, color='k', lw=1.0)
    ax.axhline(0.05, color=C_RULES['single'], lw=1.0, ls='--')
    ax.text(0.5, 0.085, r'$\alpha=0.05$', fontsize=7,
            color=C_RULES['single'], ha='center')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Taste', 'Willingness'])
    ax.set_ylabel(r'Per-seed permutation $p$')
    ax.set_ylim(-0.03, 0.65)
    ax.set_title(r'B  $p$-value per seed')

    ax = axes[2]
    for i, ck in enumerate(cells):
        c = CM[ck]['agg']
        lo, hi = c['T_null_95ci']
        ax.bar(i, hi - lo, bottom=lo, width=0.45, color='#CFCFCF',
               edgecolor='#707070', lw=0.7)
        ax.hlines(c['T_obs'], i - 0.3, i + 0.3, color=[C_TASTE, C_WILL][i],
                  lw=1.8)
    ax.axhline(0, color='#AAA', lw=0.7, ls='--')
    ax.set_ylim(-0.008, 0.023)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Taste', 'Willingness'])
    ax.set_ylabel(r'$T_{\mathrm{obs}}$ and null interval')
    ax.set_title(r'C  Aggregated vs null')
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    ax.legend(handles=[Line2D([], [], color=C_TASTE, lw=1.8, label='$T_{obs}$'),
                       Patch(facecolor='#CFCFCF', edgecolor='#707070',
                             label='null 95% CI')],
              frameon=False, loc='upper center', ncol=2,
              bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout()
    save(fig, 'Figure2')


# ============================================================
# Figure 3 — calibration + rule comparison + seed-number sensitivity
# ============================================================
def fig3(audit, calib):
    CM = cell_map(audit)
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.7))

    # A: FPR vs S
    ax = axes[0]
    bs = calib['by_seed_count'] if 'by_seed_count' in calib else calib
    Ss = sorted(int(k) for k in bs)
    for rule in ('single', 'min', 'mean', 'agg'):
        y = [bs[str(s)]['fpr'][rule] for s in Ss]
        e = [bs[str(s)]['fpr_se'][rule] for s in Ss]
        ax.errorbar(Ss, y, yerr=e, marker='o', ms=3.4, lw=1.0, capsize=2,
                    color=C_RULES[rule], label=RULE_LABEL[rule])
    nom = calib.get('nominal', 0.05)
    ax.axhline(nom, color='k', lw=0.9, ls='--')
    ax.text(Ss[-1], nom + 0.012, f'nominal {nom:.2f}', fontsize=6.6,
            ha='right')
    ax.set_xscale('log')
    ax.set_xticks(Ss)
    ax.set_xticklabels([str(s) for s in Ss])
    ax.set_xlabel('Number of seeds $S$')
    ax.set_ylabel('Empirical Type I error')
    ax.set_title('A  Calibration')
    ax.legend(frameon=False, loc='upper right')

    # B: rule comparison on the same permutation output
    ax = axes[1]
    cells = [ck for ck in CELL_LABEL if ck in CM]
    rules = ('single', 'min', 'mean', 'agg')
    SHORT = {'occipital|rating_taste': 'Occipital taste',
             'occipital|rating_willingnessToEat': 'Occipital will.',
             'all|rating_taste': 'Scalp taste',
             'all|rating_health': 'Scalp health',
             'all|rating_willingnessToEat': 'Scalp will.'}
    w = 0.2
    for j, rule in enumerate(rules):
        ys, xs = [], []
        for i, ck in enumerate(cells):
            a = CM[ck]['agg']
            ys.append(a[f'p_{rule}'] if rule != 'agg' else a['p_agg'])
            xs.append(i + (j - 1.5) * w)
        ax.bar(xs, ys, width=w, color=C_RULES[rule], label=RULE_LABEL[rule],
               edgecolor='white', lw=0.4)
    ax.axhline(0.05, color='k', lw=0.9, ls='--')
    ax.text(len(cells) - 0.55, 0.058, r'$\alpha=0.05$', fontsize=6.2, ha='right')
    ax.set_xticks(range(len(cells)))
    ax.set_xticklabels([SHORT[c] for c in cells], fontsize=6.0,
                       rotation=22, ha='right')
    ax.set_ylabel(r'Permutation $p$')
    ax.set_ylim(0, 0.33)
    ax.set_title('B  Same output, four rules')

    # C: seed-number sensitivity
    ax = axes[2]
    ax2 = ax.twinx()
    for i, ck in enumerate(['occipital|rating_taste',
                            'occipital|rating_willingnessToEat']):
        bs2 = CM[ck]['by_seed_count']
        Ss2 = sorted(int(k) for k in bs2)
        sd = [bs2[str(s)]['T_obs_std'] for s in Ss2]
        fr = [bs2[str(s)]['frac_p_le_0.05'] for s in Ss2]
        ax.plot(Ss2, sd, 'o-', ms=3.4, lw=1.0, color=[C_TASTE, C_WILL][i],
                label=['Taste', 'Willingness'][i])
        ax2.plot(Ss2, fr, 's--', ms=3.0, lw=0.9, alpha=0.75,
                 color=[C_TASTE, C_WILL][i])
    ax.set_xscale('log')
    ax.set_xticks([1, 2, 3, 5, 10])
    ax.set_xticklabels(['1', '2', '3', '5', '10'])
    ax.set_xlabel('Number of seeds $S$')
    ax.set_ylabel('SD of $T_{obs}$ across subsets', color='k')
    ax2.set_ylabel('Fraction of subsets with $p<0.05$', color='#555')
    ax2.set_ylim(0, 1.08)
    ax.set_title('C  Seed-number sensitivity')
    ax.legend(frameon=False, loc='upper right')
    fig.tight_layout()
    save(fig, 'Figure3')


# ============================================================
# Figure 4 — training-budget sensitivity
# ============================================================
def fig4():
    if not BUDGET.exists():
        print('[skip] Figure4: budget_sensitivity.csv 尚不存在')
        return False
    df = pd.read_csv(BUDGET)
    df['bud'] = df['budget'].replace({'inf': np.inf}).astype(float)
    df.loc[~np.isfinite(df['bud']), 'bud'] = 2600.0

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7))
    COL = {'rating_taste': (C_TASTE, 'o', '-'),
           'rating_willingnessToEat': (C_WILL, 's', '--'),
           'rating_health': ('#7F7F7F', '^', ':')}
    LBL = {'rating_taste': 'Taste', 'rating_willingnessToEat': 'Willingness',
           'rating_health': 'Health'}
    from matplotlib.lines import Line2D

    handles_a = []
    for sem, alpha in (('pool', 1.0), ('per_subject', 0.55)):
        sub = df[(df['semantics'] == sem) & (df['roi'] == 'full_montage')]
        for tgt, (col, mk, ls) in COL.items():
            s2 = sub[sub['target'] == tgt].sort_values('bud')
            if s2.empty:
                continue
            axes[0].plot(s2['bud'], s2['r_mean'], marker=mk, ls=ls, ms=3.2,
                         lw=1.0, color=col, alpha=alpha)
            axes[1].plot(s2['bud'], s2['r_std'], marker=mk, ls=ls, ms=3.2,
                         lw=1.0, color=col, alpha=alpha)
            if sem == 'pool':
                handles_a.append(Line2D([], [], color=col, marker=mk, ls=ls,
                                        label=LBL[tgt]))
    for ax in axes:
        ax.set_xscale('log')
        ax.set_xlabel('Pooled training-trial budget per fold')
        ax.axhline(0, color='#AAA', lw=0.7, ls='--')
    axes[0].set_ylabel('Mean observed statistic $r$')
    axes[0].set_title('A  Observed statistic')
    axes[0].legend(handles=handles_a, frameon=False, fontsize=6.6,
                   title='pooled budget', title_fontsize=6.6, loc='best')
    axes[1].set_ylabel('Across-seed SD of $r$')
    axes[1].set_title('B  Seed-induced dispersion')
    axes[1].legend(handles=handles_a, frameon=False, fontsize=6.6, loc='best')
    fig.tight_layout()
    save(fig, 'Figure4')
    return True


# ============================================================
# Figure 5 — FoodEEG application
# ============================================================
def fig5():
    from scipy import stats as _st
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.7))

    # A: SCSR distribution
    ax = axes[0]
    if SCSR.exists():
        s = pd.read_csv(SCSR)
        v = s['scsr'].astype(float).values
        v = v[np.isfinite(v)]
        ax.hist(v, bins=14, color='#B7C9E2', edgecolor='#4F81BD', lw=0.6)
        mu, sd = v.mean(), v.std(ddof=1)
        t, p = _st.ttest_1samp(v, 0.5)
        ax.axvline(mu, color='k', lw=1.2)
        ax.axvline(0.5, color=C_RULES['min'], lw=1.0, ls='--')
        ax.text(0.505, ax.get_ylim()[1] * 0.55, 'ref. 0.5', fontsize=6.3,
                color=C_RULES['min'], ha='left')
        ax.text(0.03, ax.get_ylim()[1] * 0.93,
                f'$n={len(v)}$\nmean $={mu:.3f}\\pm{sd:.3f}$',
                fontsize=6.4, va='top')
        ax.set_xlim(0, 1)
        ax.set_xlabel('Self-control success rate')
        ax.set_ylabel('Participants')
        ax.set_title('A  Behaviour')
    else:
        ax.set_title('A  Behaviour')

    # B: ROI-level descriptive parametric effects (Cohen's d)
    ax = axes[1]
    d = json.loads(ROI.read_text(encoding='utf-8'))
    res = sorted(d['results'], key=lambda r: r['cohen_d'])
    x = np.arange(len(res))
    cols = ['#4F81BD' if r['p'] < 0.05 else '#B7C9E2' for r in res]
    ax.bar(x, [r['cohen_d'] for r in res], color=cols, edgecolor='#3D6A99',
           lw=0.5, width=0.75)
    ax.axhline(0, color='#888', lw=0.7)
    ax.set_xlabel('Region-of-interest cell (sorted)')
    ax.set_ylabel("Cohen's $d$")
    ax.set_title(f"B  Descriptive effects ({len(res)} cells)")
    ax.set_xticks([])
    ax.text(0.02, 0.94, 'none FDR-significant', transform=ax.transAxes,
            fontsize=6.4, va='top', color='#555')

    # C: retraction
    ax = axes[2]
    e = [p for p in d['permutation']
         if p['roi'] == 'temporal' and p['target'] == 'rating_willingnessToEat']
    if e:
        e = e[0]
        ps = np.array(e['perm_ps'], float)
        xs = np.arange(1, len(ps) + 1)
        ax.plot(xs, ps, 'o', ms=4.0, color=C_WILL)
        ax.hlines(e['perm_p'], 0.6, len(ps) + 0.4, color=C_AGG, lw=1.4)
        ax.text(len(ps) + 0.35, e['perm_p'] + 0.05,
                f"across-seed mean $p={e['perm_p']:.2f}$",
                fontsize=6.4, color=C_AGG, ha='right')
        ax.plot(1, 0.0, 'x', ms=6, color=C_RULES['min'], mew=1.6)
        ax.annotate('originally flagged\nsingle-seed $p=0.000$',
                    xy=(1, 0.0), xytext=(1.4, 0.12), fontsize=6.3,
                    color=C_RULES['min'],
                    arrowprops=dict(arrowstyle='-', lw=0.6,
                                    color=C_RULES['min']))
        ax.set_xticks(xs)
        ax.set_xlabel('Seed')
        ax.set_ylabel(r'Permutation $p$')
        ax.set_ylim(-0.08, 1.05)
        ax.set_xlim(0.5, len(ps) + 0.6)
        ax.set_title('C  Retracted effect')
    fig.tight_layout()
    save(fig, 'Figure5')


def fig_ga():
    """Graphical Abstract：单张正方形，1200x1200 px @ 300 dpi（Cell Press 要求）。"""
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
    fig, ax = plt.subplots(figsize=(4, 4))
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')

    def box(x, y, w, h, text, fc='#F4F4F4', ec='#3F3F3F', fs=6.6,
            bold=False, tc='#1A1A1A'):
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h, boxstyle='round,pad=0.8,rounding_size=2.0',
            linewidth=0.9, facecolor=fc, edgecolor=ec, zorder=2))
        ax.text(x + w / 2, y + h / 2, text, ha='center', va='center',
                fontsize=fs, zorder=3, color=tc,
                fontweight='bold' if bold else 'normal')

    def arrow(x1, y1, x2, y2, color='#3F3F3F', ls='-', rad=0.0):
        ax.add_patch(FancyArrowPatch(
            (x1, y1), (x2, y2), arrowstyle='-|>', mutation_scale=8,
            linewidth=1.0, color=color, linestyle=ls, zorder=1,
            connectionstyle=f'arc3,rad={rad}'))

    box(2, 78, 27, 14,
        'Single-trial EEG\ncovariance $C_i$\n(64 channels)', fs=6.0)
    box(36, 78, 28, 14,
        'Riemannian tangent\nspace\n(log-Euclidean)', fs=6.0)
    box(71, 78, 27, 14,
        'LOSO decoding\nwith random\ntrial subsampling', fs=6.0,
        fc='#FDE9D9', ec='#C0504D')
    arrow(29.5, 85, 35.5, 85)
    arrow(64.5, 85, 70.5, 85)
    ax.text(50, 73.5, 'discarded seed draw', fontsize=6.0, ha='center',
            color='#C0504D', style='italic')

    box(3, 52, 42, 16,
        'One seed, one $p$\nunstable across\nnearly its whole range',
        fc='#FBE9E7', ec='#B23A48', fs=6.0)
    box(55, 52, 42, 16,
        'Aggregate over $S$ seeds\n$T_{\\mathrm{obs}}=\\frac{1}{S}\\sum_s T_s$',
        fc='#E8F5E9', ec='#2E7D32', fs=6.0)
    arrow(30, 78, 24, 69.0, ls=':', rad=0.0)
    arrow(88, 78, 76, 69.0)

    box(3, 30, 42, 15,
        'Single-seed permutation\ntest:\ninflated type I error',
        fc='#FBE9E7', ec='#B23A48', fs=6.0)
    box(55, 30, 42, 15,
        'Shared-shuffle\npermutation of $T_{\\mathrm{obs}}$:\ncalibrated at $\\alpha=0.05$',
        fc='#E8F5E9', ec='#2E7D32', fs=6.0)
    arrow(24, 52, 24, 45.5, color='#B23A48')
    arrow(76, 52, 76, 45.5, color='#2E7D32')

    ax.plot([12, 20, 28, 36], [22.0, 26.5, 19.0, 25.5], 'o', ms=3.0,
            color='#B23A48', zorder=3)
    ax.text(24, 12.5, 'per-seed $p$: scattered', fontsize=5.8,
            ha='center', color='#B23A48')
    ax.hlines(23.0, 72, 98, color='#2E7D32', lw=1.8)
    ax.text(85, 12.5, 'one calibrated verdict', fontsize=5.8,
            ha='center', color='#2E7D32')

    ax.text(50, 4.0,
            'Report seed stability and statistical significance separately',
            fontsize=6.2, ha='center', fontweight='bold')
    ax.text(50, 97.0, 'Auditable inference for stochastic EEG decoding',
            fontsize=7.0, ha='center', fontweight='bold')

    for ext in ('png', 'pdf', 'tif'):
        fig.savefig(FIGDIR / f'GraphicalAbstract.{ext}', dpi=300)
    plt.close(fig)
    print(f'[saved] {FIGDIR / "GraphicalAbstract.png"} (1200x1200 @300dpi)')


def main():
    audit = json.loads(AUDIT.read_text(encoding='utf-8')) if AUDIT.exists() else None
    calib = None
    if CALIB.exists():
        c = json.loads(CALIB.read_text(encoding='utf-8'))
        calib = c.get('calib', c)
    fig1()
    if audit:
        fig2(audit)
    else:
        print('[skip] Figure2: audit json 缺失')
    if audit and calib and 'by_seed_count' in calib:
        fig3(audit, calib)
    else:
        print('[skip] Figure3: 校准结果尚未生成')
    fig4()
    fig5()
    fig_ga()


if __name__ == '__main__':
    main()
