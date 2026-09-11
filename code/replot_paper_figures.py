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
from scipy.ndimage import gaussian_filter1d

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

from config import PROJECT_ROOT, RESULTS_DIR
from eeg_preprocessing import TMAX_STIM as TMAX, TMIN_STIM as TMIN
from erp_analysis import (CH_IDX as ERP_CH_IDX, KEY_CHANNELS as ERP_CHANNELS,
                          TIME_WINDOWS, TRIAL_TYPES, TYPE_COLORS,
                          build_time_axis, compute_individual_erps,
                          load_subject_list)
from sullivan_integration import (correlate_with_behavior,
                                  extract_individual_metrics, load_ddm_params,
                                  load_erp_window_means, load_scsr_per_subject,
                                  load_tfr_window_means)
from timefreq_analysis import (CH_IDX as TFR_CH_IDX, FREQS, FREQ_BANDS,
                               KEY_CHANNELS as TFR_CHANNELS, baseline_correct,
                               compute_group_tfr,
                               load_subject_list as tfr_load_subject_list)

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
PAPER_FIG_DIR = PROJECT_ROOT / 'paper' / 'figures'
WINDOW_NAME = 'full_0-1000ms'
TARGETS_ORDER = ['rating_health', 'rating_taste', 'rating_willingnessToEat']


def save_fig(fig, name):
    out = PAPER_FIG_DIR / name
    fig.savefig(out, dpi=DPI, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)
    print(f'[fig] saved: {out}')


def fig9_erp_group_average():
    sids = load_subject_list()
    group_erps, kept = compute_individual_erps(sids)
    n_times = group_erps['health'].shape[-1]
    t_axis = build_time_axis(n_times)
    n_subs = group_erps['health'].shape[0]
    fig, axes = plt.subplots(2, 3, figsize=(6.84, 3.65), sharex=True)
    for ax, ch in zip(axes.flat, ERP_CHANNELS):
        ci = ERP_CH_IDX[ch]
        for tt in TRIAL_TYPES:
            data = group_erps[tt][:, ci, :] * 1e6
            mean = data.mean(axis=0)
            sem = data.std(axis=0) / np.sqrt(data.shape[0])
            ax.plot(t_axis, mean, label=tt, color=TYPE_COLORS[tt], linewidth=1.2)
            ax.fill_between(t_axis, mean - sem, mean + sem,
                            color=TYPE_COLORS[tt], alpha=0.2)
        ax.axvline(0, color='black', linestyle='--', alpha=0.5, linewidth=0.8)
        ax.axhline(0, color='black', linewidth=0.5)
        for _, (w0, w1) in TIME_WINDOWS.items():
            ax.axvspan(w0, w1, color='gray', alpha=0.06)
        ax.set_title(f'Channel {ch}')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude (µV)')
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, [l.capitalize() for l in labels], loc='lower center',
               ncol=3, frameon=False, fontsize=8, columnspacing=1.6,
               handletextpad=0.5)
    fig.suptitle(f'Group ERP by trial type (stimulus-locked, n = {n_subs})',
                 fontsize=9)
    fig.tight_layout(rect=[0, 0.055, 1, 0.94])
    save_fig(fig, 'fig9_erp_group_average.png')


def fig13_tfr_group_average(recompute=False):
    db_path = RESULTS_DIR / 'tfr_group_db.npy'
    if recompute or not db_path.exists():
        print('[fig13] 重算全被试 TFR（约 1-2 小时）...')
        sids = tfr_load_subject_list()
        group_tfr, kept = compute_group_tfr(sids)
        n_times = next(iter(group_tfr.values())).shape[-1]
        t_axis = np.linspace(TMIN, TMAX, n_times)
        tfr_db = baseline_correct(group_tfr, t_axis)
        np.save(RESULTS_DIR / 'tfr_group_raw.npy',
                {tt: arr for tt, arr in group_tfr.items()}, allow_pickle=True)
        np.save(db_path, {tt: arr for tt, arr in tfr_db.items()},
                allow_pickle=True)
        n_subs = len(kept)
        print(f'[fig13] TFR 重算完成：{n_subs} 名被试')
    else:
        tfr_db = np.load(db_path, allow_pickle=True).item()
        n_times = next(iter(tfr_db.values())).shape[-1]
        t_axis = np.linspace(TMIN, TMAX, n_times)
        n_subs = next(iter(tfr_db.values())).shape[0]
    fig, axes = plt.subplots(len(TRIAL_TYPES), len(TFR_CHANNELS),
                             figsize=(6.84, 2.85))
    t_mask = (t_axis >= -0.2) & (t_axis <= 1.0)
    t_plot = t_axis[t_mask]
    for r, tt in enumerate(TRIAL_TYPES):
        for c, ch in enumerate(TFR_CHANNELS):
            ax = axes[r, c]
            ci = TFR_CH_IDX[ch]
            data = tfr_db[tt][:, ci, :, :].mean(axis=0)
            ax.imshow(data[:, t_mask], aspect='auto', origin='lower',
                      extent=[t_plot[0], t_plot[-1], FREQS[0], FREQS[-1]],
                      cmap='RdBu_r', vmin=-3, vmax=3)
            ax.axvline(0, color='black', linestyle='--', linewidth=0.8)
            for _, (f_lo, f_hi) in FREQ_BANDS.items():
                ax.axhline(f_lo, color='gray', linestyle=':', alpha=0.6,
                           linewidth=0.6)
                ax.axhline(f_hi, color='gray', linestyle=':', alpha=0.6,
                           linewidth=0.6)
            if r == len(TRIAL_TYPES) - 1:
                ax.set_xlabel('Time (s)')
            ax.set_title(ch)
    fig.suptitle(f'Group-average TF power (dB, n={n_subs})')
    fig.tight_layout(rect=[0.075, 0, 1, 0.90])
    fig.text(0.012, 0.43, 'Frequency (Hz)', rotation=90, ha='center',
             va='center', fontsize=7.5, color='#333333')
    for r, tt in enumerate(TRIAL_TYPES):
        pos = axes[r, 0].get_position()
        fig.text(0.036, (pos.y0 + pos.y1) / 2, tt.capitalize(),
                 rotation=90, ha='center', va='center', fontsize=7.5,
                 color='#333333')
    save_fig(fig, 'fig13_tfr_group_average.png')


def fig17_sullivan_summary():
    erp_df = load_erp_window_means()
    tfr_df = load_tfr_window_means()
    ddm_df = load_ddm_params()
    scsr_df = load_scsr_per_subject()
    metrics_df = extract_individual_metrics(erp_df, tfr_df)
    _, merged = correlate_with_behavior(metrics_df, ddm_df, scsr_df)

    def panel(ax, xcol, ycol, color, xlabel, ylabel, title, half_line):
        if xcol not in merged.columns or ycol not in merged.columns:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center')
            ax.set_title(title)
            return
        valid = merged[[xcol, ycol]].dropna()
        ax.scatter(valid[xcol], valid[ycol], alpha=0.6, color=color, s=12)
        if len(valid) > 5:
            r, p = stats.pearsonr(valid[xcol], valid[ycol])
            ax.set_title(f'{title}\nr={r:.3f}, p={p:.3f}')
            z = np.polyfit(valid[xcol], valid[ycol], 1)
            xs = np.linspace(valid[xcol].min(), valid[xcol].max(), 50)
            ax.plot(xs, np.polyval(z, xs), 'k--', alpha=0.5)
        else:
            ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if half_line:
            ax.axhline(0.5, color='black', linewidth=0.5, linestyle='--')
        else:
            ax.axhline(0, color='black', linewidth=0.5)
            ax.axvline(0, color='black', linewidth=0.5)

    fig, axes = plt.subplots(2, 2, figsize=(6.84, 5.70))
    panel(axes[0, 0], 'taste_early_advantage_erp_Pz', 'v_taste', '#E07B39',
          'taste_early_advantage_erp_Pz (µV)', 'v_taste (DDM)',
          'taste ERP advantage vs v_taste', False)
    panel(axes[0, 1], 'health_late_advantage_erp_Pz', 'v_health', '#3D7A8E',
          'health_late_advantage_erp_Pz (µV)', 'v_health (DDM)',
          'health ERP advantage vs v_health', False)
    panel(axes[1, 0], 'taste_mid_theta_advantage_Fz', 'scsr', '#E07B39',
          'taste_mid_theta_advantage_Fz (dB)', 'SCSR',
          'taste mid theta advantage vs SCSR', True)
    panel(axes[1, 1], 'health_late_alpha_advantage_Fz', 'scsr', '#3D7A8E',
          'health_late_alpha_advantage_Fz (dB)', 'SCSR',
          'health late alpha advantage vs SCSR', True)
    fig.suptitle('Sullivan hypothesis: neural timing advantage vs behavior')
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save_fig(fig, 'fig17_sullivan_summary.png')


def fig18_riemannian_decoding():
    df = pd.read_csv(RESULTS_DIR / 'riemannian_decoding_results.csv')
    if df['window'].nunique() > 1:
        df = df[df['window'] == '0.0-1.0s']
    contrasts = ['taste-health', 'taste-decision', 'health-decision']
    colors = {'taste-health': '#33658A', 'taste-decision': '#6FA3C7',
              'health-decision': '#A9C6DC'}
    labels = {'taste-health': 'Taste vs health',
              'taste-decision': 'Taste vs decision',
              'health-decision': 'Health vs decision'}
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(6.84, 2.6), gridspec_kw={'width_ratios': [1.5, 1]})
    rng = np.random.default_rng(0)
    ylo = df['balanced_acc'].min()
    yhi = df['balanced_acc'].max()
    ticklabels = []
    for i, c in enumerate(contrasts):
        vals = df[df['contrast'] == c]['balanced_acc'].values
        mean, sd = vals.mean(), vals.std(ddof=1)
        ci = stats.t.ppf(0.975, len(vals) - 1) * sd / np.sqrt(len(vals))
        ax1.scatter(i + rng.uniform(-0.16, 0.16, len(vals)), vals, s=10,
                    alpha=0.4, color=colors[c], edgecolors='none', zorder=2)
        ax1.errorbar(i, mean, yerr=ci, fmt='o', color='black', markersize=4,
                     capsize=3, linewidth=1.0, zorder=4)
        _, p_val = stats.ttest_1samp(vals, 0.5)
        sig = '' if p_val < 0.05 else ' (ns)'
        ax1.text(i, yhi + 0.018, f'p = {p_val:.3f}{sig}', ha='center',
                 va='bottom', fontsize=7.5, color='dimgray')
        ticklabels.append(f'{labels[c]}\n{mean:.3f} ± {sd:.3f}')
    ax1.axhline(0.5, color='black', linestyle='--', linewidth=0.8, zorder=1)
    ax1.text(2.54, 0.508, 'chance\n(0.5)', ha='right', va='bottom',
             fontsize=7.5, color='dimgray')
    ax1.set_xlim(-0.55, 2.55)
    ax1.set_ylim(ylo - 0.03, yhi + 0.075)
    ax1.set_xticks(range(len(contrasts)))
    ax1.set_xticklabels(ticklabels)
    ax1.set_ylabel('Balanced accuracy')
    n_sub = df['subject_id'].nunique()
    fig_note = (f'n = {n_sub} subjects · small points: individual LOSO '
                'accuracy · black circles: mean ± 95% CI · '
                'one-sample t vs 0.5')
    pivot = df.pivot(index='subject_id', columns='contrast',
                     values='balanced_acc')
    if 'taste-health' in pivot.columns and 'taste-decision' in pivot.columns:
        xa = pivot['taste-health']
        ya = pivot['taste-decision']
        lo = min(xa.min(), ya.min()) - 0.02
        hi = max(xa.max(), ya.max()) + 0.02
        ax2.plot([lo, hi], [lo, hi], color='black', linestyle='--',
                 linewidth=0.8, zorder=1)
        ax2.axvline(0.5, color='dimgray', linestyle=':', linewidth=0.8,
                    zorder=1)
        ax2.axhline(0.5, color='dimgray', linestyle=':', linewidth=0.8,
                    zorder=1)
        ax2.scatter(xa, ya, s=12, alpha=0.5, color='#33658A',
                    edgecolors='none', zorder=2)
        ax2.set_xlim(lo, hi)
        ax2.set_ylim(lo, hi)
        r, p_r = stats.pearsonr(xa, ya)
        p_txt = 'p < 0.001' if p_r < 0.001 else f'p = {p_r:.3f}'
        ax2.text(0.04, 0.96, f'r = {r:.2f}, {p_txt}', transform=ax2.transAxes,
                 ha='left', va='top', fontsize=7.5)
        ax2.set_xlabel('Taste vs health (balanced accuracy)')
        ax2.set_ylabel('Taste vs decision (balanced accuracy)')
    else:
        ax2.set_visible(False)
    fig.suptitle('Riemannian tangent-space decoding (LOSO)', fontsize=9)
    fig.text(0.5, 0.012, fig_note, ha='center', va='bottom', fontsize=7,
             color='dimgray')
    fig.tight_layout(rect=[0, 0.06, 1, 0.93])
    save_fig(fig, 'fig18_riemannian_decoding.png')


def fig22_temporal_decoding_curve():
    df = pd.read_csv(RESULTS_DIR / 'sliding_window_decoding_results.csv')
    summary = json.loads((RESULTS_DIR / 'sliding_window_summary.json')
                         .read_text(encoding='utf-8'))
    grouped = df.groupby('t_center_ms')['balanced_acc']
    means = grouped.mean()
    sems = grouped.sem()
    t_centers = means.index
    means_smooth = gaussian_filter1d(means.values, sigma=1.5)
    n_sub = int(df['subject_id'].nunique())
    ymax = float((means + sems).values.max())
    ymin = float((means - sems).values.min())
    rng_data = max(ymax - ymin, 0.001)
    ax_top = ymax + 0.45 * rng_data
    ax_bot = ymin - 0.18 * rng_data

    def win_ms(label):
        a, b = label.replace('ms', '').split('-')
        return float(a), float(b)

    peak = summary['peak_window']
    below = next((w for w in summary['windows']
                  if w['mean'] < 0.5 and w['sig_uncorrected']), None)
    fig = plt.figure(figsize=(3.30, 3.60))
    ax = fig.add_axes([0.145, 0.40, 0.825, 0.50])
    ax.plot(t_centers, means.values, 'o', color='#33658A', alpha=0.5,
            markersize=2.6, label='Per-window mean')
    ax.plot(t_centers, means_smooth, '-', color='#33658A', linewidth=1.2,
            label='Smoothed ($\\sigma$=1.5 windows)')
    ax.fill_between(t_centers, means - sems, means + sems, alpha=0.22,
                    color='#33658A', label='$\\pm$1 SEM')
    ax.axhline(0.5, color='black', linestyle='--', linewidth=0.7,
               label='chance (0.5)')
    p_lo, p_hi = win_ms(peak['label'])
    ax.axvspan(p_lo, p_hi, alpha=0.10, color='#C44E52')
    if below is not None:
        b_lo, b_hi = win_ms(below['window'])
        ax.axvspan(b_lo, b_hi, alpha=0.10, color='#8C8C8C')
    ax.set_xlim(t_centers.values[0] - 20, t_centers.values[-1] + 20)
    ax.set_ylim(ax_bot, ax_top)
    ax.text(peak['t_center_ms'], ax_top - 0.012 * rng_data,
            f"peak {peak['label']}\n{peak['mean']:.3f}, "
            f"p={peak['p_value']:.3f}",
            ha='center', va='top', fontsize=7, color='#8E2F33',
            linespacing=1.3)
    if below is not None:
        ax.text(below['t_center_ms'], ax_top - 0.012 * rng_data,
                f"below chance\n{below['window']}\n"
                f"p={below['p_value']:.3f}",
                ha='center', va='top', fontsize=7, color='#555555',
                linespacing=1.3)
    ax.set_xlabel('Window center (ms)', fontsize=7.5)
    ax.set_ylabel('Balanced accuracy (LOSO)', fontsize=7.5)
    ax.set_title('Taste-vs.-health decoding across time', fontsize=8)
    ax.tick_params(labelsize=7, pad=2)
    ax.grid(True, alpha=0.25, linewidth=0.4)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=7, loc='upper center',
               bbox_to_anchor=(0.5, 0.285), ncol=2, frameon=False,
               columnspacing=0.9, handletextpad=0.3)
    n_win = summary['window_params']['n_windows']
    fig.text(0.5, 0.05,
             f"windows: 100 ms, step 50 ms · n = {n_sub}\n"
             f"BH-FDR over {n_win} windows: "
             f"{summary['n_sig_fdr']} significant",
             fontsize=7, color='#444444', ha='center', va='center',
             linespacing=1.4)
    save_fig(fig, 'fig22_temporal_decoding_curve.png')


def fig24_svr_roi_channels():
    df = pd.read_csv(RESULTS_DIR / 'svr_roi_channels_summary.csv')
    roi_order = ['occipital', 'central', 'temporal', 'frontal',
                 'fronto-central', 'centro-parietal', 'parietal']
    targets = [t for t in TARGETS_ORDER if t in df['target'].unique()]
    piv = df.pivot(index='roi', columns='target',
                   values='mean_r').reindex(index=roi_order, columns=targets)
    piv_p = df.pivot(index='roi', columns='target',
                     values='p').reindex(index=roi_order, columns=targets)
    piv_pp = df.pivot(index='roi', columns='target',
                      values='perm_p').reindex(index=roi_order,
                                               columns=targets)
    tgt_short = {'rating_health': 'health', 'rating_taste': 'taste',
                 'rating_willingnessToEat': 'willingness'}
    fig = plt.figure(figsize=(3.30, 3.10))
    ax = fig.add_axes([0.24, 0.34, 0.575, 0.56])
    vmax = max(abs(piv.values).max(), 0.01)
    im = ax.imshow(piv.values, aspect='auto', cmap='RdBu_r',
                   vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(targets)))
    ax.set_xticklabels([tgt_short.get(t, t) for t in targets],
                       rotation=0, ha='center', fontsize=7)
    ax.set_yticks(range(len(roi_order)))
    ax.set_yticklabels(roi_order, fontsize=7)
    for i in range(len(roi_order)):
        for j in range(len(targets)):
            val = piv.values[i, j]
            p = piv_p.values[i, j]
            pp = piv_pp.values[i, j]
            if pp < 0.05:
                mark, mcolor = 'P', '#1a7a4a'
            elif p < 0.05:
                mark, mcolor = 'O', '#b3541e'
            else:
                mark, mcolor = '', None
            if mcolor is not None:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                           facecolor='none',
                                           edgecolor=mcolor, linewidth=1.3,
                                           zorder=2))
                ax.plot(j + 0.35, i + 0.35, marker='o', markersize=4.6,
                        markerfacecolor='none', markeredgecolor=mcolor,
                        markeredgewidth=1.2, zorder=4)
            ax.text(j, i - 0.1, f'{val:+.3f}', ha='center', va='center',
                    fontsize=7, zorder=3,
                    color='white' if abs(val) > 0.45 * vmax else '#222222')
    ax.set_xlim(-0.5, len(targets) - 0.5)
    ax.set_ylim(len(roi_order) - 0.5, -0.5)
    ax.tick_params(length=2, pad=1.5)
    cb = fig.colorbar(im, ax=ax, pad=0.03, aspect=14)
    cb.set_label('mean LOSO r', fontsize=7)
    cb.ax.tick_params(labelsize=7, length=2)
    ax.set_title('Tangent-space regression by ROI $\\times$ target',
                 fontsize=8, pad=6)
    n_sub = int(df['n'].dropna().unique()[0])
    min_perm = float(df['perm_p'].min())
    min_q = float(df['q_value'].min())
    n_sig = int((df['q_value'] < 0.05).sum())
    fig.text(0.02, 0.14,
             '\u25CB: parametric p < 0.05 (bordered)\n'
             f'no cell: seed-mean permutation p < 0.05 '
             f'(min = {min_perm:.3f})',
             fontsize=7, color='#444444', ha='left', va='center',
             linespacing=1.4)
    fig.text(0.02, 0.04,
             rf'n = {n_sub} · BH-FDR over {len(df)} tests: '
             rf'{n_sig}/{len(df)} (q $\geq$ {min_q:.2f})',
             fontsize=7, color='#444444', ha='left', va='center')
    save_fig(fig, 'fig24_svr_roi_channels.png')


def fig25_scsr_group_diff():
    sub = pd.read_csv(RESULTS_DIR / 'svr_scsr_groups_results.csv')
    sub = sub[sub['window'] == WINDOW_NAME]
    group_specs = [('high_scsr', '#2C5F8A'), ('low_scsr', '#8FB8D4')]
    target_labels = {'rating_health': 'Health', 'rating_taste': 'Taste',
                     'rating_willingnessToEat': 'Willingness\nto eat'}
    fig, ax = plt.subplots(figsize=(3.30, 2.85))
    rng = np.random.default_rng(1)
    n_groups = []
    for gi, (gname, color) in enumerate(group_specs):
        g = sub[sub['group'] == gname]
        n_groups.append(g['subject_id'].nunique())
        for ti, tgt in enumerate(TARGETS_ORDER):
            vals = g[g['target'] == tgt]['pearson_r'].values
            xpos = ti + (gi - 0.5) * 0.36
            ax.scatter(xpos + rng.uniform(-0.065, 0.065, len(vals)), vals,
                       s=8, alpha=0.35, color=color, edgecolors='none',
                       zorder=2)
            mean = vals.mean()
            ci = stats.t.ppf(0.975, len(vals) - 1) * vals.std(ddof=1) \
                / np.sqrt(len(vals))
            ax.errorbar(xpos, mean, yerr=ci, fmt='o', color=color,
                        markersize=4, capsize=2.5, linewidth=1.0, zorder=4)
    wide = sub.pivot_table(index=['group', 'subject_id'], columns='target',
                           values='pearson_r')
    d_hi = (wide.loc['high_scsr', 'rating_health']
            - wide.loc['high_scsr', 'rating_taste'])
    d_lo = (wide.loc['low_scsr', 'rating_health']
            - wide.loc['low_scsr', 'rating_taste'])
    t_stat, p_val = stats.ttest_ind(d_hi, d_lo, equal_var=False)
    sig = ' (ns)' if p_val >= 0.05 else ''
    note = (f'health − taste: high {d_hi.mean():+.4f}, '
            f'low {d_lo.mean():+.4f}\n'
            f'Welch t = {t_stat:.2f}, p = {p_val:.3f}{sig}')
    h = [ax.scatter([], [], s=25, color=c) for _, c in group_specs]
    ax.legend(h, [f'High SCSR, n = {n_groups[0]}',
                  f'Low SCSR, n = {n_groups[1]}'], frameon=False,
              loc='upper center', ncol=2, handletextpad=0.2,
              columnspacing=0.9)
    ax.axhline(0, color='black', linestyle=':', linewidth=0.8, zorder=1)
    ax.set_title('Tangent-space regression by SCSR group', fontsize=8)
    ax.set_xlim(-0.55, 2.55)
    ymin = min(sub['pearson_r'].min(), 0.0)
    ymax = sub['pearson_r'].max()
    ax.set_ylim(ymin - 0.11, ymax + 0.16)
    ax.set_xticks(range(len(TARGETS_ORDER)))
    ax.set_xticklabels([target_labels[t] for t in TARGETS_ORDER],
                       fontsize=7)
    ax.set_ylabel('Pearson r (LOSO)', fontsize=7.5)
    ax.tick_params(labelsize=7, pad=2)
    fig.text(0.5, 0.055, note, ha='center', va='center', fontsize=7,
             color='dimgray', linespacing=1.45)
    fig.tight_layout(rect=[0, 0.135, 1, 1])
    save_fig(fig, 'fig25_scsr_group_diff.png')


def fig26_svr_x_ddm_corr():
    df = pd.read_csv(RESULTS_DIR / 'svr_ddm_correlations.csv')
    targets = [t for t in ['rating_health', 'rating_taste',
                           'rating_willingnessToEat']
               if t in df['target'].unique()]
    row_lbl = {'rating_health': 'Health', 'rating_taste': 'Taste',
               'rating_willingnessToEat': 'Willingness'}
    win_order = [w for w in ['N1_100-200ms', 'P2_200-300ms', 'P3_300-450ms',
                             'LPP_400-700ms', 'LATE_700-1000ms',
                             'full_0-1000ms'] if w in df['window'].unique()]
    win_lbl = {'N1_100-200ms': 'N1', 'P2_200-300ms': 'P2',
               'P3_300-450ms': 'P3', 'LPP_400-700ms': 'LPP',
               'LATE_700-1000ms': 'Late', 'full_0-1000ms': 'Full'}
    ddm_order = [c for c in ['v_taste', 'v_health', 'a', 't']
                 if c in df['ddm_param'].unique()]
    ddm_lbl = {'v_taste': 'v$_{taste}$', 'v_health': 'v$_{health}$',
               'a': 'a', 't': 't'}
    n_sub = int(df['n'].dropna().unique()[0])
    n_tests = len(df)
    min_q = float(np.nanmin(df['pearson_p_fdr'].values))
    n_sig = int((df['pearson_p_fdr'] < 0.05).sum())
    fig = plt.figure(figsize=(3.30, 4.50))
    axes = [fig.add_axes([0.155, b, 0.770, 0.165])
            for b in [0.612, 0.396, 0.180]]
    absmax = max(abs(df['pearson_r'].max()), abs(df['pearson_r'].min()),
                 0.05)
    im0 = None
    for idx, (ax, target_col) in enumerate(zip(axes, targets)):
        sub = df[df['target'] == target_col]
        pivot = sub.pivot(index='ddm_param', columns='window',
                          values='pearson_r').reindex(index=ddm_order,
                                                      columns=win_order)
        piv_p = sub.pivot(index='ddm_param', columns='window',
                          values='pearson_p_fdr').reindex(index=ddm_order,
                                                          columns=win_order)
        im = ax.imshow(pivot.values, cmap='RdBu_r', vmin=-absmax,
                       vmax=absmax, aspect='auto', zorder=1)
        if idx == 0:
            im0 = im
        last = idx == len(axes) - 1
        ax.set_xticks(range(len(win_order)))
        ax.set_xticklabels(
            [win_lbl.get(c, c) for c in win_order] if last
            else [''] * len(win_order), fontsize=7)
        ax.set_yticks(range(len(ddm_order)))
        ax.set_yticklabels([ddm_lbl.get(r, r) for r in ddm_order],
                           fontsize=7)
        ax.set_xlim(-0.5, len(win_order) - 0.5)
        ax.set_ylim(len(ddm_order) - 0.5, -0.5)
        for i in range(len(ddm_order)):
            for j in range(len(win_order)):
                val = pivot.values[i, j]
                p = piv_p.values[i, j]
                sig = (not np.isnan(p)) and p < 0.05
                ax.text(j, i, f'{val:+.2f}',
                        ha='center', va='center',
                        fontsize=7, zorder=4,
                        color='white' if abs(val) > 0.55 * absmax
                        else '#222222',
                        fontweight='bold' if sig else 'normal')
        ax.set_title(row_lbl.get(target_col, target_col), fontsize=7.5,
                     loc='left', pad=4)
    cax = fig.add_axes([0.155, 0.100, 0.770, 0.016])
    cb = fig.colorbar(im0, cax=cax, orientation='horizontal')
    cb.set_label('Pearson r', fontsize=7, labelpad=10)
    cb.ax.tick_params(labelsize=7, length=2, pad=1.5)
    fig.suptitle('Tangent-space regression $\\times$ DDM parameters',
                 fontsize=8.5, y=0.988)
    fig.text(0.02, 0.862,
             f'{len(targets)} targets $\\times$ {len(ddm_order)} params '
             f'$\\times$ {len(win_order)} windows = {n_tests} tests,\n'
             f'n = {n_sub} subjects; BH-FDR: {n_sig}/{n_tests} survive '
             f'(min q = {min_q:.2f})',
             ha='left', va='center', fontsize=7, color='#444444',
             linespacing=1.4)
    save_fig(fig, 'fig26_svr_x_ddm_corr.png')


def main():
    ap = argparse.ArgumentParser(
        description='按论文排版宽度重绘论文图（有效字号>=7pt, dpi=300）')
    ap.add_argument('--only', default='',
                    help='逗号分隔图名，如 fig9,fig13；默认全部')
    ap.add_argument('--tfr-recompute', action='store_true',
                    help='fig13 强制重算全被试 TFR 并回写缓存')
    args = ap.parse_args()
    PAPER_FIG_DIR.mkdir(parents=True, exist_ok=True)
    registry = {
        'fig9': fig9_erp_group_average,
        'fig13': lambda: fig13_tfr_group_average(recompute=args.tfr_recompute),
        'fig17': fig17_sullivan_summary,
        'fig18': fig18_riemannian_decoding,
        'fig22': fig22_temporal_decoding_curve,
        'fig24': fig24_svr_roi_channels,
        'fig25': fig25_scsr_group_diff,
        'fig26': fig26_svr_x_ddm_corr,
    }
    only = [s.strip() for s in args.only.split(',') if s.strip()]
    names = only if only else list(registry.keys())
    for name in names:
        if name not in registry:
            print(f'[skip] 未知图名: {name}')
            continue
        print(f'===== {name} =====')
        registry[name]()


if __name__ == '__main__':
    main()
