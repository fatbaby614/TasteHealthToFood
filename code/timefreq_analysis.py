# -*- coding: utf-8 -*-
"""
切入点 1-A2：时间频率分析
========================

验证 Sullivan (2015) 属性加工时序假说的时频动力学补充：
  - tastiness 判断：~ 4-8Hz (theta) 早期(300-600ms) 在 frontal 区域
  - healthiness 判断：~ 8-13Hz (alpha) 晚期(700-1000ms) 在 frontal/parietal 区域
  - 决策权衡：~ 13-30Hz (beta) 中期在 central/frontal 区域

分析内容：
1. 计算每个 trial_type 的 time-frequency power (Morlet wavelet)
2. 关键频段平均功率的时间动力学曲线
3. 频段拓扑图（每个 trial_type × 频段 × 时间窗）
4. 频段 power 的群体统计检验（验证 Sullivan 假说）
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

FS = 512
mne.set_log_level('ERROR')

# ============ 参数 ============
TRIAL_TYPES = ['health', 'taste', 'decision']
TYPE_COLORS = {'health': '#3D7A8E', 'taste': '#E07B39', 'decision': '#5B9279'}

# 频段定义
FREQ_BANDS = {
    'theta':   (4, 8),     # 情绪/记忆（前额）
    'alpha':   (8, 13),    # 注意抑制（顶枕）
    'beta':    (13, 30),   # 决策/认知控制（额中央）
}

# 关键通道
KEY_CHANNELS = ['Fz', 'FCz', 'Cz', 'Pz', 'O1', 'P3']
CH_IDX = {ch: EEG_CHANNEL_NAMES.index(ch) for ch in KEY_CHANNELS}

# 时间窗（与 ERP 分析一致，便于对照）
TIME_WINDOWS = {
    'early_150-350ms':   (0.150, 0.350),   # 早期注意 / 视觉处理
    'mid_350-600ms':    (0.350, 0.600),    # taste 加工峰值假说
    'late_700-1000ms':  (0.700, 1.000),    # health 加工峰值假说
}

# Morlet 参数
FREQS = np.logspace(np.log10(4), np.log10(45), 25)  # 4-45 Hz logspace
NCYCLES = FREQS / 2.0  # 4-45 Hz 时 cycles 在 2-22.5 之间，保证低频也至少 2 个 cycle

# Baseline 用刺激前（-0.2 ~ 0），mode='ratio' 常用于 ERP 学界
TF_BASELINE = (-0.2, 0)

# 对比组合
CONTRASTS = [
    ('taste', 'health'),
    ('taste', 'decision'),
    ('health', 'decision'),
]

ALPHA = 0.05
RANDOM_STATE = 42


# ============ 数据加载 ============
def load_subject_list():
    qc_df = pd.read_csv(QC_DIR / 'qc_summary.csv')
    valid = qc_df[qc_df['n_stim_kept'].notna()]
    valid = valid[(valid['n_taste_kept'] >= 30)
                  & (valid['n_health_kept'] >= 30)
                  & (valid['n_decision_kept'] >= 30)]
    return valid['subject_id'].tolist()


def compute_tfr_for_subject(subject_id, max_trials_per_type=60):
    """计算单个被试的 time-frequency power (单 trial 后平均)

    返回
    ----
    tfr_dict : {trial_type: ndarray (n_channels, n_freqs, n_times)}
    """
    try:
        X, labels = load_subject_epochs(subject_id, 'stim')
    except Exception as e:
        print(f'  [skip] {subject_id}: {e}')
        return None

    tt_arr = labels['trial_type']
    n_ch = X.shape[1]
    n_t = X.shape[2]
    times = np.linspace(TMIN, TMAX, n_t)

    # 准备 Epochs 对象以使用 mne.time_frequency.tfr_array_morlet
    info = mne.create_info(EEG_CHANNEL_NAMES, FS, ch_types='eeg')
    montage = mne.channels.make_standard_montage('biosemi64')
    info.set_montage(montage)

    rng = np.random.RandomState(RANDOM_STATE)
    tfr_dict = {}
    for tt in TRIAL_TYPES:
        mask = tt_arr == tt
        if mask.sum() < 10:
            return None
        # 限制 trial 数（计算成本）
        if mask.sum() > max_trials_per_type:
            idx = np.where(mask)[0]
            idx = rng.choice(idx, max_trials_per_type, replace=False)
            X_tt = X[idx]
        else:
            X_tt = X[mask]

        # 转为 (n_trials, n_ch, n_times)
        # 应用 Morlet wavelet
        # tfr_array_morlet 期望 (n_epochs, n_channels, n_times)
        power = mne.time_frequency.tfr_array_morlet(
            X_tt.astype(np.float64), sfreq=FS,
            freqs=FREQS, n_cycles=NCYCLES,
            output='power', n_jobs=1, verbose='ERROR'
        )
        # power: (n_trials, n_ch, n_freqs, n_times)
        # 跨 trial 平均
        tfr_dict[tt] = power.mean(axis=0)  # (n_ch, n_freqs, n_times)

    return tfr_dict


def compute_group_tfr(subject_ids):
    """计算群体 TF power

    返回
    ----
    group_tfr : {trial_type: ndarray (n_subs, n_ch, n_freqs, n_times)}
    kept_subs : list
    """
    group_tfr = {tt: [] for tt in TRIAL_TYPES}
    kept = []

    for i, sid in enumerate(subject_ids):
        tfr = compute_tfr_for_subject(sid)
        if tfr is None:
            continue
        for tt in TRIAL_TYPES:
            group_tfr[tt].append(tfr[tt])
        kept.append(sid)

        if (i + 1) % 10 == 0:
            print(f'  进度: {i+1}/{len(subject_ids)}')

    group_tfr = {tt: np.stack(arrs) for tt, arrs in group_tfr.items() if len(arrs) > 0}
    return group_tfr, kept


def baseline_correct(group_tfr, t_axis):
    """对每个被试每个通道每个频段做 baseline 校正（dB 变换 + 减基线）

    baseline: -0.2 ~ 0 (刺激前)
    返回 (n_subs, n_ch, n_freqs, n_times) - dB scale
    """
    bl_mask = (t_axis >= TF_BASELINE[0]) & (t_axis <= TF_BASELINE[1])

    corrected = {}
    for tt, data in group_tfr.items():
        # data: (n_subs, n_ch, n_freqs, n_times)
        bl_mean = data[..., bl_mask].mean(axis=-1, keepdims=True)
        # 10 * log10(power / baseline) —— dB
        # 防止除零
        bl_mean = np.maximum(bl_mean, 1e-30)
        corrected[tt] = 10 * np.log10(data / bl_mean)
    return corrected


def compute_band_power(tfr_db, t_axis):
    """计算每个频段的平均 power

    返回
    ----
    band_power : {trial_type: dict{band: ndarray (n_subs, n_ch, n_times)}}
    """
    band_power = {}
    for tt, data in tfr_db.items():
        # data: (n_subs, n_ch, n_freqs, n_times)
        bp = {}
        for band, (f_lo, f_hi) in FREQ_BANDS.items():
            f_mask = (FREQS >= f_lo) & (FREQS <= f_hi)
            bp[band] = data[:, :, f_mask, :].mean(axis=2)  # (n_subs, n_ch, n_times)
        band_power[tt] = bp
    return band_power


def compute_band_window_means(band_power, t_axis):
    """每个被试 × trial_type × 频段 × 时间窗 × 通道 的平均 power

    返回 DataFrame
    """
    rows = []
    n_subs = next(iter(next(iter(band_power.values())).values())).shape[0]
    for i in range(n_subs):
        row = {'subject_idx': i}
        for tt in TRIAL_TYPES:
            if tt not in band_power:
                continue
            for band in FREQ_BANDS.keys():
                data = band_power[tt][band][i]  # (n_ch, n_times)
                for wname, (w0, w1) in TIME_WINDOWS.items():
                    t_mask = (t_axis >= w0) & (t_axis <= w1)
                    for ch in KEY_CHANNELS:
                        ci = CH_IDX[ch]
                        row[f'{tt}_{band}_{wname}_{ch}'] = float(data[ci, t_mask].mean())
        rows.append(row)
    return pd.DataFrame(rows)


# ============ 可视化 ============
def plot_tfr_group_average(tfr_db, t_axis, save_path):
    """群体平均 TF 图（关键通道）"""
    n_subs = next(iter(tfr_db.values())).shape[0]
    fig, axes = plt.subplots(len(TRIAL_TYPES), len(KEY_CHANNELS),
                              figsize=(len(KEY_CHANNELS) * 3, len(TRIAL_TYPES) * 2.5))

    if len(TRIAL_TYPES) == 1:
        axes = axes[np.newaxis, :]

    freqs_for_plot = FREQS
    t_mask = (t_axis >= -0.2) & (t_axis <= 1.0)
    t_plot = t_axis[t_mask]

    for r, tt in enumerate(TRIAL_TYPES):
        if tt not in tfr_db:
            continue
        for c, ch in enumerate(KEY_CHANNELS):
            ci = CH_IDX[ch]
            ax = axes[r, c]
            # 群体平均 (n_subs, n_freqs, n_times)
            data = tfr_db[tt][:, ci, :, :].mean(axis=0)
            # 截取时间窗
            data_plot = data[:, t_mask]
            im = ax.imshow(data_plot, aspect='auto', origin='lower',
                          extent=[t_plot[0], t_plot[-1], freqs_for_plot[0], freqs_for_plot[-1]],
                          cmap='RdBu_r', vmin=-3, vmax=3)
            ax.axvline(0, color='black', linestyle='--', linewidth=0.8)
            # 频段标记
            for band, (f_lo, f_hi) in FREQ_BANDS.items():
                ax.axhline(f_lo, color='gray', linestyle=':', linewidth=0.5, alpha=0.6)
                ax.axhline(f_hi, color='gray', linestyle=':', linewidth=0.5, alpha=0.6)
            if c == 0:
                ax.set_ylabel(f'{tt}\nFrequency (Hz)')
            if r == len(TRIAL_TYPES) - 1:
                ax.set_xlabel('Time (s)')
            ax.set_title(ch, fontsize=10)

    plt.suptitle(f'Group-average TF power (dB, n={n_subs})', fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'[fig] saved: {save_path}')


def plot_band_dynamics(band_power, t_axis, save_path):
    """关键频段的时间动力学曲线（按 trial_type 对比）

    每行一个频段，每列一个通道
    """
    n_subs = next(iter(next(iter(band_power.values())).values())).shape[0]
    bands = list(FREQ_BANDS.keys())
    fig, axes = plt.subplots(len(bands), len(KEY_CHANNELS),
                              figsize=(len(KEY_CHANNELS) * 3, len(bands) * 2.5),
                              sharex=True)

    if len(bands) == 1:
        axes = axes[np.newaxis, :]

    t_mask = (t_axis >= -0.2) & (t_axis <= 1.0)
    t_plot = t_axis[t_mask]

    for r, band in enumerate(bands):
        for c, ch in enumerate(KEY_CHANNELS):
            ci = CH_IDX[ch]
            ax = axes[r, c]
            for tt in TRIAL_TYPES:
                if tt not in band_power:
                    continue
                data = band_power[tt][band][:, ci, t_mask]  # (n_subs, n_times)
                mean = data.mean(axis=0)
                sem = data.std(axis=0) / np.sqrt(data.shape[0])
                ax.plot(t_plot, mean, label=tt, color=TYPE_COLORS[tt], linewidth=1.5)
                ax.fill_between(t_plot, mean - sem, mean + sem,
                                color=TYPE_COLORS[tt], alpha=0.2)
            ax.axvline(0, color='black', linestyle='--', linewidth=0.8)
            ax.axhline(0, color='black', linewidth=0.5)
            # 标记关键时间窗
            for wname, (w0, w1) in TIME_WINDOWS.items():
                ax.axvspan(w0, w1, color='gray', alpha=0.05)
            if c == 0:
                ax.set_ylabel(f'{band}\nPower (dB)')
            if r == len(bands) - 1:
                ax.set_xlabel('Time (s)')
            ax.set_title(ch, fontsize=10)
            ax.legend(fontsize=8, loc='best')

    plt.suptitle(f'Band-specific power dynamics (n={n_subs})', fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'[fig] saved: {save_path}')


def plot_band_topography(band_power, t_axis, save_path):
    """频段 × 时间窗 拓扑图（每行一个 trial_type）"""
    montage = mne.channels.make_standard_montage('biosemi64')
    info = mne.create_info(EEG_CHANNEL_NAMES, FS, ch_types='eeg')
    info.set_montage(montage)

    bands = list(FREQ_BANDS.keys())
    windows = list(TIME_WINDOWS.items())
    n_rows = len(TRIAL_TYPES)
    n_cols = len(bands) * len(windows)
    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(n_cols * 2.5, n_rows * 2.5))

    if n_rows == 1:
        axes = axes[np.newaxis, :]

    for r, tt in enumerate(TRIAL_TYPES):
        if tt not in band_power:
            continue
        col_idx = 0
        for band in bands:
            data_band = band_power[tt][band].mean(axis=0)  # (n_ch, n_times)
            for wname, (w0, w1) in windows:
                t_mask = (t_axis >= w0) & (t_axis <= w1)
                topo = data_band[:, t_mask].mean(axis=1)
                ax = axes[r, col_idx]
                im, _ = mne.viz.plot_topomap(topo, info, axes=ax, show=False,
                                             cmap='RdBu_r', contours=6)
                ax.set_title(f'{band}\n{wname}', fontsize=8)
                col_idx += 1

    plt.suptitle('Band × time-window topography (mean power, dB)', fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'[fig] saved: {save_path}')


# ============ 统计 ============
def band_t_tests(window_df, save_path):
    """对预定义的 频段 × 时间窗 × 通道 做 paired t-test

    返回 DataFrame
    """
    results = []
    for (a, b) in CONTRASTS:
        for band in FREQ_BANDS.keys():
            for wname in TIME_WINDOWS.keys():
                for ch in KEY_CHANNELS:
                    col_a = f'{a}_{band}_{wname}_{ch}'
                    col_b = f'{b}_{band}_{wname}_{ch}'
                    if col_a not in window_df or col_b not in window_df:
                        continue
                    d = window_df[col_a] - window_df[col_b]
                    t, p = stats.ttest_1samp(d, 0)
                    cohens_dz = d.mean() / d.std(ddof=1)
                    results.append({
                        'contrast': f'{a}-{b}',
                        'band': band,
                        'window': wname,
                        'channel': ch,
                        'mean_diff_dB': d.mean(),
                        'std_diff': d.std(ddof=1),
                        't': t,
                        'p_raw': p,
                        'cohen_dz': cohens_dz,
                        'n': len(d),
                    })

    df = pd.DataFrame(results)
    df['p_bonf'] = (df['p_raw'] * len(df)).clip(upper=1.0)
    _, p_fdr = fdr_correction(df['p_raw'].values)
    df['p_fdr'] = p_fdr
    df['sig_bonf'] = df['p_bonf'] < ALPHA
    df['sig_fdr'] = df['p_fdr'] < ALPHA

    df.to_csv(save_path, index=False)
    print(f'[stat] saved: {save_path}  ({len(df)} tests)')
    return df


# ============ Sullivan 时序假说直接检验 ============
def test_sullivan_tf_hypothesis(window_df):
    """检验 Sullivan 时序假说的时频版本

    H1: taste 在 mid 窗（350-600ms）应引发 frontal theta 增强（vs health）
    H2: health 在 late 窗（700-1000ms）应引发 frontal alpha 抑制减少 或 frontal theta（vs taste）

    ROI: Fz (frontal)
    """
    ch = 'Fz'
    results = {}

    # H1: taste > health at mid window, theta band, Fz
    col_a = f'taste_theta_mid_350-600ms_{ch}'
    col_b = f'health_theta_mid_350-600ms_{ch}'
    if col_a in window_df and col_b in window_df:
        d = window_df[col_a] - window_df[col_b]
        t, p = stats.ttest_1samp(d, 0)
        results['H1_taste_gt_health_theta_mid_Fz'] = {
            'mean_diff_dB': float(d.mean()), 't': float(t), 'p': float(p),
            'cohen_dz': float(d.mean() / d.std(ddof=1)),
        }

    # H2: health > taste at late window, alpha band, Fz (less alpha suppression)
    col_a = f'health_alpha_late_700-1000ms_{ch}'
    col_b = f'taste_alpha_late_700-1000ms_{ch}'
    if col_a in window_df and col_b in window_df:
        d = window_df[col_a] - window_df[col_b]
        t, p = stats.ttest_1samp(d, 0)
        results['H2_health_gt_taste_alpha_late_Fz'] = {
            'mean_diff_dB': float(d.mean()), 't': float(t), 'p': float(p),
            'cohen_dz': float(d.mean() / d.std(ddof=1)),
        }

    # H3: taste > health at mid, beta, Fz (executive engagement)
    col_a = f'taste_beta_mid_350-600ms_{ch}'
    col_b = f'health_beta_mid_350-600ms_{ch}'
    if col_a in window_df and col_b in window_df:
        d = window_df[col_a] - window_df[col_b]
        t, p = stats.ttest_1samp(d, 0)
        results['H3_taste_gt_health_beta_mid_Fz'] = {
            'mean_diff_dB': float(d.mean()), 't': float(t), 'p': float(p),
            'cohen_dz': float(d.mean() / d.std(ddof=1)),
        }

    # H4: health > taste at late, beta, Fz
    col_a = f'health_beta_late_700-1000ms_{ch}'
    col_b = f'taste_beta_late_700-1000ms_{ch}'
    if col_a in window_df and col_b in window_df:
        d = window_df[col_a] - window_df[col_b]
        t, p = stats.ttest_1samp(d, 0)
        results['H4_health_gt_taste_beta_late_Fz'] = {
            'mean_diff_dB': float(d.mean()), 't': float(t), 'p': float(p),
            'cohen_dz': float(d.mean() / d.std(ddof=1)),
        }

    print('\n  Sullivan 时频假说直接检验（ROI=Fz）：')
    for k, v in results.items():
        sig = '***' if v['p'] < 0.001 else ('**' if v['p'] < 0.01
              else ('*' if v['p'] < 0.05 else 'ns'))
        print(f'    {k}: t={v["t"]:.2f}, p={v["p"]:.4f} {sig}, d={v["cohen_dz"]:.3f}')

    return results


# ============ 主入口 ============
def main():
    print('=' * 60)
    print('切入点 1-A2：时间频率分析')
    print('=' * 60)

    subject_ids = load_subject_list()
    print(f'\n有效被试：{len(subject_ids)}')

    # 1. 计算群体 TF
    print('\n[step 1] 计算 TF power（Morlet wavelet）...')
    print('  注意：此步较慢，每被试约 30-60 秒')
    group_tfr, kept = compute_group_tfr(subject_ids)
    print(f'  保留被试：{len(kept)}')
    for tt in TRIAL_TYPES:
        if tt in group_tfr:
            print(f'  {tt}: shape={group_tfr[tt].shape}')

    n_times = next(iter(group_tfr.values())).shape[-1]
    t_axis = np.linspace(TMIN, TMAX, n_times)

    # 2. Baseline 校正 (dB)
    print('\n[step 2] Baseline 校正（dB）...')
    tfr_db = baseline_correct(group_tfr, t_axis)

    # 3. 频段平均
    print('\n[step 3] 频段平均...')
    band_power = compute_band_power(tfr_db, t_axis)

    # 4. 保存原始 group TFR（用于后续复用）
    np.save(RESULTS_DIR / 'tfr_group_raw.npy',
            {tt: arr for tt, arr in group_tfr.items()}, allow_pickle=True)
    np.save(RESULTS_DIR / 'tfr_group_db.npy',
            {tt: arr for tt, arr in tfr_db.items()}, allow_pickle=True)
    print(f'  保存原始 TF 数据至 {RESULTS_DIR / "tfr_group_db.npy"}')

    # 5. 可视化
    print('\n[step 4] 可视化...')
    plot_tfr_group_average(tfr_db, t_axis,
                          FIGURES_DIR / 'fig13_tfr_group_average.png')
    plot_band_dynamics(band_power, t_axis,
                       FIGURES_DIR / 'fig14_band_dynamics.png')
    plot_band_topography(band_power, t_axis,
                         FIGURES_DIR / 'fig15_band_topography.png')

    # 6. 统计
    print('\n[step 5] 统计检验...')
    window_df = compute_band_window_means(band_power, t_axis)
    window_df.insert(0, 'subject_id', kept[:len(window_df)])
    window_df.to_csv(RESULTS_DIR / 'tfr_band_window_means.csv', index=False)

    ttest_df = band_t_tests(window_df, RESULTS_DIR / 'tfr_band_ttests.csv')
    sig = ttest_df[ttest_df['sig_fdr']]
    print('\n[FDR 显著结果]：')
    if len(sig) > 0:
        print(sig[['contrast', 'band', 'window', 'channel', 'mean_diff_dB',
                   't', 'p_fdr', 'cohen_dz']].to_string(index=False))
    else:
        print('  无 FDR 显著结果。')
    print(f'\n[Bonferroni 显著]: {ttest_df["sig_bonf"].sum()}')

    # 7. Sullivan 时频假说直接检验
    print('\n[step 6] Sullivan 时序假说（时频版本）直接检验...')
    sullivan_tf = test_sullivan_tf_hypothesis(window_df)

    # 8. 汇总
    summary = {
        'n_subjects_analyzed': len(kept),
        'subject_ids': kept,
        'freq_bands': {k: list(v) for k, v in FREQ_BANDS.items()},
        'freqs_used': FREQS.tolist(),
        'time_windows': {k: list(v) for k, v in TIME_WINDOWS.items()},
        'baseline_window': list(TF_BASELINE),
        'tfr_ttest_summary': {
            'n_tests': int(len(ttest_df)),
            'n_significant_fdr': int(ttest_df['sig_fdr'].sum()),
            'n_significant_bonferroni': int(ttest_df['sig_bonf'].sum()),
        },
        'sullivan_tf_test': sullivan_tf,
    }

    with open(RESULTS_DIR / 'tfr_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    print(f'\n[summary] {RESULTS_DIR / "tfr_summary.json"}')

    print('\n' + '=' * 60)
    print('A2 完成：时频分析')
    print('=' * 60)


if __name__ == '__main__':
    main()
