# -*- coding: utf-8 -*-
"""
切入点 1-A1：ERP 分析
=====================

验证 Sullivan (2015) 属性加工时序假说：
  - tastiness 判断加工峰值 ~ 500ms（LPP/P3 区域）
  - healthiness 判断加工峰值 ~ 1000ms（晚期正成分）

分析内容：
1. 群体平均 ERP（按 trial_type: taste/health/decision）
2. 差异波（taste - health, taste - decision, health - decision）
3. 拓扑图（关键时间窗）
4. 统计检验：
   a. 选定 ROI × 时间窗做群体配对 t-test（Bonferroni 校正）
   b. cluster-based permutation test（spatio-temporal）
"""

import numpy as np
import pandas as pd
import mne
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import json
from scipy import stats
from mne.stats import fdr_correction

from config import FIGURES_DIR, RESULTS_DIR
from eeg_preprocessing import (
    load_subject_epochs, EEG_CHANNEL_NAMES, QC_DIR,
    TMIN_STIM as TMIN, TMAX_STIM as TMAX, BASELINE,
)

# BDF 默认采样率 512 Hz
FS = 512

mne.set_log_level('ERROR')

# ============ 参数 ============
TRIAL_TYPES = ['health', 'taste', 'decision']
TYPE_COLORS = {'health': '#3D7A8E', 'taste': '#E07B39', 'decision': '#5B9279'}

# 关键通道（基于 Hare 2009 + Sullivan 2015）
KEY_CHANNELS = ['O1', 'Pz', 'P3', 'P4', 'Fz', 'FCz']
CH_IDX = {ch: EEG_CHANNEL_NAMES.index(ch) for ch in KEY_CHANNELS}

# 时间窗（基于 Sullivan 2015 + Hare 2009 文献假说）
TIME_WINDOWS = {
    'N1_100-180ms':    (0.100, 0.180),   # 早期视觉注意
    'N2_200-280ms':    (0.200, 0.280),   # 分类/冲突检测
    'P3_300-450ms':    (0.300, 0.450),   # P3b — 价值/注意分配
    'LPP_450-650ms':   (0.450, 0.650),   # taste 加工峰值假说
    'LATE_800-1000ms': (0.800, 1.000),   # health 加工峰值假说
}

# 对比组合
CONTRASTS = [
    ('taste', 'health'),    # 核心：taste vs health 时序差异
    ('taste', 'decision'),
    ('health', 'decision'),
]

# 统计参数
ALPHA = 0.05
N_PERMUTATIONS = 1000  # cluster permutation
RANDOM_STATE = 42


# ============ 数据加载 ============
def load_subject_list():
    """从 QC summary 加载有效被试"""
    qc_df = pd.read_csv(QC_DIR / 'qc_summary.csv')
    valid = qc_df[qc_df['n_stim_kept'].notna()]
    # 至少 30 个 taste epochs
    valid = valid[(valid['n_taste_kept'] >= 30)
                  & (valid['n_health_kept'] >= 30)
                  & (valid['n_decision_kept'] >= 30)]
    return valid['subject_id'].tolist()


def build_time_axis(n_times):
    """构造时间轴"""
    return np.linspace(TMIN, TMAX, n_times)


def compute_individual_erps(subject_ids):
    """逐被试计算三个 trial_type 的 ERP 均值

    返回
    ----
    group_erps : dict, 每个键 trial_type 的形状 (n_subjects, n_channels, n_times)
    subject_ids_kept : list 实际保留的被试 ID
    """
    group_erps = {tt: [] for tt in TRIAL_TYPES}
    kept = []

    for i, sid in enumerate(subject_ids):
        try:
            X, labels = load_subject_epochs(sid, 'stim')
        except Exception as e:
            print(f'  [skip] {sid}: {e}')
            continue

        tt_arr = labels['trial_type']
        ok = True
        for tt in TRIAL_TYPES:
            mask = tt_arr == tt
            if mask.sum() < 30:
                ok = False
                break
        if not ok:
            continue

        for tt in TRIAL_TYPES:
            mask = tt_arr == tt
            erp = X[mask].mean(axis=0)  # (n_ch, n_times)
            group_erps[tt].append(erp)

        kept.append(sid)
        if (i + 1) % 20 == 0:
            print(f'  进度: {i+1}/{len(subject_ids)}')

    group_erps = {tt: np.stack(arrs) for tt, arrs in group_erps.items() if len(arrs) > 0}
    return group_erps, kept


def compute_window_means(group_erps, t_axis):
    """为每个被试 × trial_type × 时间窗 × 通道 计算平均幅值

    返回
    ----
    df : DataFrame, 每行一个被试，列为 {tt}_{window}_{channel} (单位 µV)
    """
    rows = []
    n_subs = next(iter(group_erps.values())).shape[0]
    for i in range(n_subs):
        row = {'subject_idx': i}
        for tt in TRIAL_TYPES:
            if tt not in group_erps:
                continue
            erp = group_erps[tt][i]  # (n_ch, n_times)
            for wname, (w0, w1) in TIME_WINDOWS.items():
                t_mask = (t_axis >= w0) & (t_axis <= w1)
                for ch in KEY_CHANNELS:
                    ci = CH_IDX[ch]
                    row[f'{tt}_{wname}_{ch}'] = erp[ci, t_mask].mean() * 1e6
        rows.append(row)
    return pd.DataFrame(rows)


# ============ 可视化 ============
def plot_group_erp(group_erps, group_sems, t_axis, save_path):
    """群体平均 ERP（关键通道）"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)

    for ax, ch in zip(axes.flat, KEY_CHANNELS):
        ci = CH_IDX[ch]
        for tt in TRIAL_TYPES:
            if tt not in group_erps:
                continue
            mean = group_erps[tt][:, ci, :] * 1e6
            sem = group_sems[tt][:, ci, :] * 1e6
            ax.plot(t_axis, mean, label=tt, color=TYPE_COLORS[tt], linewidth=1.8)
            ax.fill_between(t_axis, mean - sem, mean + sem,
                            color=TYPE_COLORS[tt], alpha=0.2)
        ax.axvline(0, color='black', linestyle='--', alpha=0.5, linewidth=0.8)
        ax.axhline(0, color='black', linewidth=0.5)
        # 标记关键时间窗
        for wname, (w0, w1) in TIME_WINDOWS.items():
            ax.axvspan(w0, w1, color='gray', alpha=0.06)
        ax.set_title(f'Channel {ch}', fontsize=11)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude (µV)')
        ax.legend(fontsize=8, loc='best')

    plt.suptitle('Group-level ERP by trial_type (stimulus-locked, n=%d)'
                 % next(iter(group_erps.values())).shape[0],
                 fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'[fig] saved: {save_path}')


def plot_difference_waves(group_erps, t_axis, save_path):
    """差异波：taste - health 等"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)

    for ax, ch in zip(axes.flat, KEY_CHANNELS):
        ci = CH_IDX[ch]
        for (a, b) in CONTRASTS:
            if a not in group_erps or b not in group_erps:
                continue
            diff = (group_erps[a] - group_erps[b])[:, ci, :] * 1e6
            mean = diff.mean(axis=0)
            sem = diff.std(axis=0) / np.sqrt(diff.shape[0])
            color = {'taste-health': '#A04000',
                     'taste-decision': '#6E4B9E',
                     'health-decision': '#1E5F74'}.get(f'{a}-{b}', 'gray')
            ax.plot(t_axis, mean, label=f'{a} − {b}',
                    color=color, linewidth=1.8)
            ax.fill_between(t_axis, mean - sem, mean + sem,
                            color=color, alpha=0.2)
        ax.axvline(0, color='black', linestyle='--', alpha=0.5, linewidth=0.8)
        ax.axhline(0, color='black', linewidth=0.5)
        for wname, (w0, w1) in TIME_WINDOWS.items():
            ax.axvspan(w0, w1, color='gray', alpha=0.06)
        ax.set_title(f'Channel {ch}', fontsize=11)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude (µV)')
        ax.legend(fontsize=8, loc='best')

    plt.suptitle('Difference waves (stimulus-locked)', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'[fig] saved: {save_path}')


def plot_topography(group_erps, t_axis, save_path):
    """关键时间窗的拓扑图（每行一个 trial_type，每列一个时间窗）"""
    montage = mne.channels.make_standard_montage('biosemi64')
    info = mne.create_info(EEG_CHANNEL_NAMES, FS, ch_types='eeg')
    info.set_montage(montage)

    windows = list(TIME_WINDOWS.items())
    n_rows = len(TRIAL_TYPES)
    n_cols = len(windows)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))

    for r, tt in enumerate(TRIAL_TYPES):
        if tt not in group_erps:
            continue
        erp = group_erps[tt].mean(axis=0)  # (n_ch, n_times)
        for c, (wname, (w0, w1)) in enumerate(windows):
            t_mask = (t_axis >= w0) & (t_axis <= w1)
            data = erp[:, t_mask].mean(axis=1) * 1e6  # µV
            ax = axes[r, c]
            im, _ = mne.viz.plot_topomap(data, info, axes=ax, show=False,
                                         cmap='RdBu_r', contours=6)
            ax.set_title(f'{tt}\n{wname}', fontsize=9)
            if c == 0:
                pass

    plt.suptitle('Topography by trial_type × time window (mean amplitude, µV)',
                 fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'[fig] saved: {save_path}')


# ============ 统计 ============
def roi_t_tests(window_df, save_path):
    """对预定义的 ROI × 时间窗做配对 t 检验

    返回
    ----
    results : DataFrame, 每行一个 (contrast, window, channel) 检验
    """
    results = []
    for (a, b) in CONTRASTS:
        for wname in TIME_WINDOWS.keys():
            for ch in KEY_CHANNELS:
                col_a = f'{a}_{wname}_{ch}'
                col_b = f'{b}_{wname}_{ch}'
                if col_a not in window_df or col_b not in window_df:
                    continue
                d = window_df[col_a] - window_df[col_b]
                t, p = stats.ttest_1samp(d, 0)
                # 效应量
                cohens_dz = d.mean() / d.std(ddof=1)
                results.append({
                    'contrast': f'{a}-{b}',
                    'window': wname,
                    'channel': ch,
                    'mean_diff': d.mean(),
                    'std_diff': d.std(ddof=1),
                    't': t,
                    'p_raw': p,
                    'cohen_dz': cohens_dz,
                    'n': len(d),
                })

    df = pd.DataFrame(results)
    # 多重比较校正：Bonferroni（总检验数）
    df['p_bonf'] = (df['p_raw'] * len(df)).clip(upper=1.0)
    # FDR
    _, p_fdr = fdr_correction(df['p_raw'].values)
    df['p_fdr'] = p_fdr
    df['sig_bonf'] = df['p_bonf'] < ALPHA
    df['sig_fdr'] = df['p_fdr'] < ALPHA

    df.to_csv(save_path, index=False)
    print(f'[stat] saved: {save_path}  ({len(df)} tests)')
    return df


def run_cluster_permutation(group_erps, contrast=('taste', 'health'),
                            save_path=None):
    """cluster-based spatio-temporal permutation test (1-sample on differences)

    在所有通道 × 后刺激时间窗（0-1s）上对 (a-b) 做群体 1-sample t-test
    H0: mean(diff) = 0
    """
    a, b = contrast
    if a not in group_erps or b not in group_erps:
        return None

    n_subs = group_erps[a].shape[0]
    # 差异矩阵 (n_subs, n_ch, n_times)
    diff = group_erps[a] - group_erps[b]  # V

    # 时间窗：刺激后 0–1 s
    t_axis = build_time_axis(diff.shape[2])
    t_mask = (t_axis >= 0) & (t_axis <= 1.0)
    diff = diff[:, :, t_mask]
    t_use = t_axis[t_mask]
    n_times = diff.shape[2]

    # 转置为 (n_subs, n_times, n_ch) —— spatio_temporal_cluster_1samp_test 期望
    X = np.transpose(diff, (0, 2, 1))

    # 邻接关系（仅空间维度，n_channels × n_channels）
    montage = mne.channels.make_standard_montage('biosemi64')
    info = mne.create_info(EEG_CHANNEL_NAMES, FS, ch_types='eeg')
    info.set_montage(montage)
    adjacency, _ = mne.channels.find_ch_adjacency(info, 'eeg')

    # cluster 阈值（1-sample t）
    t_thresh = stats.t.ppf(1 - ALPHA / 2, n_subs - 1)
    print(f'  [cluster] contrast={a}-{b}, t_thresh=±{t_thresh:.3f}, '
          f'n_subs={n_subs}, X.shape={X.shape}')

    cluster_stats = mne.stats.spatio_temporal_cluster_1samp_test(
        X,
        n_permutations=N_PERMUTATIONS,
        threshold=t_thresh,
        tail=0,
        adjacency=adjacency,
        seed=RANDOM_STATE,
        n_jobs=1,
        verbose='ERROR',
    )
    T_obs, clusters, p_values, H0 = cluster_stats
    # T_obs 形状 (n_times, n_ch)

    # FDR 校正
    _, p_fdr = fdr_correction(p_values)

    # 可视化：T_obs 跨通道的最大值
    fig, ax = plt.subplots(figsize=(12, 5))
    T_max = np.abs(T_obs).max(axis=1)  # (n_times,)
    ax.plot(t_use, T_max, color='black', linewidth=1, label='|T|_max')
    ax.axhline(t_thresh, color='red', linestyle='--',
               label=f'|T| threshold (p={ALPHA})')
    # 标记显著 cluster
    n_sig = 0
    for i, (cl, pval, pfdr) in enumerate(zip(clusters, p_values, p_fdr)):
        if pval < ALPHA:
            times_idx = cl[0]
            chans_idx = cl[1]
            t_start = float(t_use[times_idx[0]])
            t_end = float(t_use[times_idx[-1]])
            ax.axvspan(t_start, t_end, color='red', alpha=0.2)
            n_sig += 1
            print(f'    cluster {i}: t=[{t_start:.3f}, {t_end:.3f}]s, '
                  f'n_chans={len(chans_idx)}, p={pval:.4f}, p_fdr={pfdr:.4f}')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Max |T| across channels')
    ax.set_title(f'Cluster-based 1-sample permutation test: {a} − {b}\n'
                 f'{n_sig} significant cluster(s) at α={ALPHA}')
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'[fig] saved: {save_path}')

    return {
        'n_clusters_total': len(clusters),
        'n_clusters_significant': int((p_values < ALPHA).sum()),
        'n_clusters_significant_fdr': int((p_fdr < ALPHA).sum()),
        'min_p': float(np.min(p_values)) if len(p_values) > 0 else None,
        'clusters': [
            {
                'idx': i,
                'p': float(p),
                'p_fdr': float(pf),
                'n_channels': int(len(cl[1])),
                'time_range': [float(t_use[cl[0][0]]), float(t_use[cl[0][-1]])],
            }
            for i, (cl, p, pf) in enumerate(zip(clusters, p_values, p_fdr))
        ]
    }


# ============ 主入口 ============
def main():
    print('=' * 60)
    print('切入点 1-A1：ERP 分析')
    print('=' * 60)

    # 1. 加载被试
    subject_ids = load_subject_list()
    print(f'\n有效被试（保留率 + 各 trial_type ≥ 30）：{len(subject_ids)}')

    # 2. 计算个体 ERP
    print('\n[step 1] 计算个体 ERP...')
    group_erps, kept = compute_individual_erps(subject_ids)
    print(f'  保留被试：{len(kept)}')
    for tt in TRIAL_TYPES:
        if tt in group_erps:
            print(f'  {tt}: shape={group_erps[tt].shape}')

    if len(kept) == 0 or len(group_erps) == 0:
        raise RuntimeError(
            '无有效 ERP（保留被试为 0）：请检查 outputs/qc/qc_summary.csv 中 '
            'n_stim_kept / n_health_kept / n_taste_kept / n_decision_kept '
            '指标是否完整（先运行 eeg_preprocessing.py 重建 QC 汇总）')

    n_times = next(iter(group_erps.values())).shape[-1]
    t_axis = build_time_axis(n_times)

    # 3. 群体平均 + SEM
    group_mean = {tt: erp.mean(axis=0, keepdims=True).squeeze(0)
                  for tt, erp in group_erps.items()} if False else None
    # 实际我们想要 group_erps 本身作为 (n_subs, ch, time)
    # 平均：
    group_mean = {tt: erp.mean(axis=0) for tt, erp in group_erps.items()}
    group_sem = {tt: erp.std(axis=0) / np.sqrt(erp.shape[0])
                 for tt, erp in group_erps.items()}

    # 4. 可视化
    print('\n[step 2] 绘图...')
    plot_group_erp_v2(group_erps, t_axis,
                      FIGURES_DIR / 'fig9_erp_group_average.png')

    plot_difference_waves(group_erps, t_axis,
                          FIGURES_DIR / 'fig10_erp_difference_waves.png')

    plot_topography(group_erps, t_axis,
                    FIGURES_DIR / 'fig11_topography.png')

    # 5. 统计
    print('\n[step 3] 统计检验...')
    window_df = compute_window_means(group_erps, t_axis)
    window_df.to_csv(RESULTS_DIR / 'erp_window_means.csv', index=False)

    ttest_df = roi_t_tests(window_df, RESULTS_DIR / 'erp_roi_ttests.csv')
    print('\n[ROI t-test] 显著结果 (FDR)：')
    sig = ttest_df[ttest_df['sig_fdr']]
    if len(sig) > 0:
        print(sig[['contrast', 'window', 'channel', 'mean_diff', 't',
                   'p_fdr', 'cohen_dz']].to_string(index=False))
    else:
        print('  无 FDR 显著结果。')

    print(f'\n[ROI t-test] Bonferroni 显著：'
          f'{ttest_df["sig_bonf"].sum()}')

    # cluster permutation
    print('\n[step 4] cluster-based permutation test...')
    cluster_results = {}
    for (a, b) in CONTRASTS:
        print(f'\n--- {a} vs {b} ---')
        result = run_cluster_permutation(
            group_erps, (a, b),
            FIGURES_DIR / f'fig12_cluster_{a}_vs_{b}.png'
        )
        cluster_results[f'{a}-{b}'] = result

    # 6. 保存汇总
    summary = {
        'n_subjects_analyzed': len(kept),
        'subject_ids': kept,
        'time_windows': {k: list(v) for k, v in TIME_WINDOWS.items()},
        'key_channels': KEY_CHANNELS,
        'roi_ttest_summary': {
            'n_tests': int(len(ttest_df)),
            'n_significant_fdr': int(ttest_df['sig_fdr'].sum()),
            'n_significant_bonferroni': int(ttest_df['sig_bonf'].sum()),
        },
        'cluster_results': cluster_results,
        'sullivan_hypothesis_test': {
            'description': 'Sullivan (2015): tastiness ~ 500ms, healthiness ~ 1000ms',
            'taste_window': 'LPP_450-650ms',
            'health_window': 'LATE_800-1000ms',
        },
    }

    # 补充：直接检验 Sullivan 时序假说
    # 在 Pz 通道，对比 taste 在 LPP 窗 vs health 在 LATE 窗
    print('\n[step 5] Sullivan 时序假说直接检验...')
    sullivan_test = test_sullivan_temporal_hypothesis(window_df)
    summary['sullivan_direct_test'] = sullivan_test

    with open(RESULTS_DIR / 'erp_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    print(f'\n[summary] {RESULTS_DIR / "erp_summary.json"}')

    print('\n' + '=' * 60)
    print('A1 完成：ERP 分析')
    print('=' * 60)


def plot_group_erp_v2(group_erps, t_axis, save_path):
    """群体 ERP：直接传 (n_subs, n_ch, n_times)"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
    n_subs = next(iter(group_erps.values())).shape[0]

    for ax, ch in zip(axes.flat, KEY_CHANNELS):
        ci = CH_IDX[ch]
        for tt in TRIAL_TYPES:
            if tt not in group_erps:
                continue
            data = group_erps[tt][:, ci, :] * 1e6  # µV
            mean = data.mean(axis=0)
            sem = data.std(axis=0) / np.sqrt(data.shape[0])
            ax.plot(t_axis, mean, label=tt, color=TYPE_COLORS[tt], linewidth=1.8)
            ax.fill_between(t_axis, mean - sem, mean + sem,
                            color=TYPE_COLORS[tt], alpha=0.2)
        ax.axvline(0, color='black', linestyle='--', alpha=0.5, linewidth=0.8)
        ax.axhline(0, color='black', linewidth=0.5)
        for wname, (w0, w1) in TIME_WINDOWS.items():
            ax.axvspan(w0, w1, color='gray', alpha=0.06)
        ax.set_title(f'Channel {ch}', fontsize=11)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude (µV)')
        ax.legend(fontsize=8, loc='best')

    plt.suptitle(f'Group ERP by trial_type (stimulus-locked, n={n_subs})',
                 fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'[fig] saved: {save_path}')


def test_sullivan_temporal_hypothesis(window_df):
    """直接检验 Sullivan (2015) 时序假说

    H1: taste 加工峰值早于 health 加工峰值
    - taste LPP_450-650ms 应显著 ≠ baseline (0)
    - health LATE_800-1000ms 应显著 ≠ baseline
    - taste LPP > health LPP (taste 早期更强)
    - health LATE > taste LATE (health 晚期更强)

    ROI: Pz（顶叶 — LPP 主成分）
    """
    results = {}
    ch = 'Pz'

    # 1. taste 在 LPP 窗显著
    taste_lpp = window_df[f'taste_LPP_450-650ms_{ch}']
    health_late = window_df[f'health_LATE_800-1000ms_{ch}']

    t1, p1 = stats.ttest_1samp(taste_lpp, 0)
    t2, p2 = stats.ttest_1samp(health_late, 0)

    # 2. taste LPP > health LPP（taste 在早期 LPP 窗更强）
    d_early = taste_lpp - window_df[f'health_LPP_450-650ms_{ch}']
    t3, p3 = stats.ttest_1samp(d_early, 0)

    # 3. health LATE > taste LATE（health 在晚期窗更强）
    d_late = health_late - window_df[f'taste_LATE_800-1000ms_{ch}']
    t4, p4 = stats.ttest_1samp(d_late, 0)

    # 4. 交互：taste 早 - health 早 > taste 晚 - health 晚
    # 即 taste 在 LPP 优势 > taste 在 LATE 优势
    d_early_per_subj = window_df[f'taste_LPP_450-650ms_{ch}'] - window_df[f'health_LPP_450-650ms_{ch}']
    d_late_per_subj = window_df[f'taste_LATE_800-1000ms_{ch}'] - window_df[f'health_LATE_800-1000ms_{ch}']
    t5, p5 = stats.ttest_rel(d_early_per_subj, d_late_per_subj)

    results = {
        'test_1_taste_vs_zero_at_LPP_Pz': {
            'mean': float(taste_lpp.mean()), 't': float(t1), 'p': float(p1),
            'cohen_dz': float(taste_lpp.mean() / taste_lpp.std(ddof=1)),
            'n': int(len(taste_lpp)),
        },
        'test_2_health_vs_zero_at_LATE_Pz': {
            'mean': float(health_late.mean()), 't': float(t2), 'p': float(p2),
            'cohen_dz': float(health_late.mean() / health_late.std(ddof=1)),
            'n': int(len(health_late)),
        },
        'test_3_taste_gt_health_at_LPP_Pz': {
            'mean_diff': float(d_early.mean()), 't': float(t3), 'p': float(p3),
            'cohen_dz': float(d_early.mean() / d_early.std(ddof=1)),
        },
        'test_4_health_gt_taste_at_LATE_Pz': {
            'mean_diff': float(d_late.mean()), 't': float(t4), 'p': float(p4),
            'cohen_dz': float(d_late.mean() / d_late.std(ddof=1)),
        },
        'test_5_interaction_taste_advantage_early_vs_late_Pz': {
            'mean_diff': float((d_early_per_subj - d_late_per_subj).mean()),
            't': float(t5), 'p': float(p5),
        },
    }

    print('  Sullivan 时序假说直接检验（ROI=Pz）：')
    for k, v in results.items():
        sig = '***' if v['p'] < 0.001 else ('**' if v['p'] < 0.01
              else ('*' if v['p'] < 0.05 else 'ns'))
        print(f'    {k}: t={v["t"]:.2f}, p={v["p"]:.4f} {sig}')

    return results


if __name__ == '__main__':
    main()
