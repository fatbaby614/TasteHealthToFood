# -*- coding: utf-8 -*-
"""
EEG 预处理质量控制（QC）报告

生成：
1. 跨被试保留率分布图
2. 各 trial_type 保留数对比
3. 坏通道检测汇总
4. 示例被试 ERP（验证信号质量）
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import json

from config import FIGURES_DIR
from eeg_preprocessing import QC_DIR, EPOCHS_DIR, EEG_CHANNEL_NAMES, load_subject_epochs


def load_qc_summary():
    """加载 QC 汇总"""
    qc_path = QC_DIR / 'qc_summary.csv'
    if not qc_path.exists():
        return None
    return pd.read_csv(qc_path)


def plot_qc_overview(qc_df, save_path):
    """QC 总览：保留率、trial_type 分布、坏通道"""
    if 'n_stim_kept' in qc_df.columns:
        valid = qc_df[qc_df['n_stim_kept'].notna()].copy()
    else:
        valid = pd.DataFrame()
    if len(valid) == 0:
        print('[QC] 无有效数据')
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. stimulus-locked 保留率分布
    ax = axes[0, 0]
    rr = (valid['stim_reject_rate'].dropna() * 100
          if 'stim_reject_rate' in valid.columns else pd.Series(dtype=float))
    if len(rr) > 0:
        ax.hist(rr, bins=20, color='steelblue',
                edgecolor='white', alpha=0.8)
        ax.axvline(rr.mean(), color='red', linestyle='--',
                   label=f'mean={rr.mean():.1f}%')
        ax.legend()
    else:
        ax.text(0.5, 0.5, 'reject rate 不可用\n(缓存重建行无原始 trial 总数)',
                ha='center', va='center', transform=ax.transAxes)
    ax.set_xlabel('Reject rate (%)')
    ax.set_ylabel('Number of subjects')
    ax.set_title('Stimulus-locked epoch reject rate')
    ax.legend()

    # 2. 各 trial_type 保留数
    ax = axes[0, 1]
    types = ['health', 'taste', 'decision']
    colors = ['#3D7A8E', '#E07B39', '#5B9279']
    x = np.arange(len(valid))
    width = 0.25
    for i, (tt, color) in enumerate(zip(types, colors)):
        col = f'n_{tt}_kept'
        if col in valid.columns:
            ax.bar(x + i * width, valid[col], width, label=tt, color=color, alpha=0.8)
    ax.set_xlabel('Subject index')
    ax.set_ylabel('Number of epochs')
    ax.set_title('Epochs kept per trial_type')
    ax.legend()
    ax.set_xticks(x[::10])
    ax.set_xticklabels([str(i) for i in range(0, len(valid), 10)])

    # 3. response-locked 保留率
    ax = axes[1, 0]
    if 'resp_reject_rate' in valid.columns:
        resp_valid = valid[valid['resp_reject_rate'].notna()]
        if len(resp_valid) > 0:
            ax.hist(resp_valid['resp_reject_rate'] * 100, bins=20, color='#5B9279',
                    edgecolor='white', alpha=0.8)
            ax.axvline(resp_valid['resp_reject_rate'].mean() * 100, color='red',
                       linestyle='--',
                       label=f'mean={resp_valid["resp_reject_rate"].mean()*100:.1f}%')
            ax.set_xlabel('Reject rate (%)')
            ax.set_ylabel('Number of subjects')
            ax.set_title('Response-locked epoch reject rate')
            ax.legend()

    # 4. 坏通道统计
    ax = axes[1, 1]
    # 统计有多少被试用了官方坏通道、自动检测
    has_official = valid['official_bad_channels'].fillna('[]').apply(
        lambda x: len(eval(x)) > 0 if isinstance(x, str) else False
    ).sum() if 'official_bad_channels' in valid.columns else 0
    has_interp = valid['interpolated_channels'].fillna('[]').apply(
        lambda x: len(eval(x)) > 0 if isinstance(x, str) else False
    ).sum() if 'interpolated_channels' in valid.columns else 0
    clean = len(valid) - has_interp
    ax.bar(['No interpolation\n(clean)', 'Official bad\nonly', 'Official +\nauto-detected'],
           [clean, has_official, has_interp - has_official if has_interp > has_official else 0],
           color=['#5B9279', '#3D7A8E', '#E07B39'], alpha=0.85, edgecolor='black')
    ax.set_ylabel('Number of subjects')
    ax.set_title('Bad channel interpolation')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'[fig] saved: {save_path}')


def plot_example_erp(subject_id, save_path):
    """示例被试 ERP：验证 taste vs health 时间动力学差异

    在 Pz 和 Fz 通道上对比 taste vs health 判断的 ERP
    """
    try:
        X, labels = load_subject_epochs(subject_id, 'stim')
    except FileNotFoundError:
        print(f'[QC] {subject_id} epochs not found')
        return

    trial_types = labels['trial_type']
    sfreq = 512
    t = np.linspace(-0.2, 1.0, X.shape[2])

    # 关键通道
    ch_names = EEG_CHANNEL_NAMES
    key_channels = ['Pz', 'Fz', 'FCz', 'O1', 'P3', 'P4']
    ch_idx = {ch: ch_names.index(ch) for ch in key_channels if ch in ch_names}

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    colors = {'health': '#3D7A8E', 'taste': '#E07B39', 'decision': '#5B9279'}

    for ax, (ch, idx) in zip(axes.flat, ch_idx.items()):
        for tt in ['health', 'taste', 'decision']:
            mask = trial_types == tt
            if mask.sum() == 0:
                continue
            erp = X[mask, idx, :].mean(axis=0) * 1e6  # V → µV
            sem = X[mask, idx, :].std(axis=0) / np.sqrt(mask.sum()) * 1e6
            ax.plot(t, erp, label=f'{tt} (n={mask.sum()})', color=colors[tt], linewidth=1.5)
            ax.fill_between(t, erp - sem, erp + sem, alpha=0.2, color=colors[tt])
        ax.axvline(0, color='black', linestyle='--', alpha=0.5, linewidth=0.8)
        ax.axhline(0, color='black', linewidth=0.5)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude (µV)')
        ax.set_title(f'Channel {ch}')
        ax.legend(fontsize=8)

    plt.suptitle(f'{subject_id} — ERP by trial_type (stimulus-locked)', fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'[fig] saved: {save_path}')


def generate_qc_report():
    """生成完整 QC 报告"""
    print('=' * 60)
    print('EEG 预处理 QC 报告')
    print('=' * 60)

    qc_df = load_qc_summary()
    if qc_df is None:
        print('[QC] 未找到 qc_summary.csv，请先运行预处理')
        return

    # 总览统计
    if 'n_stim_kept' in qc_df.columns:
        valid = qc_df[qc_df['n_stim_kept'].notna()]
    else:
        valid = pd.DataFrame()
    print(f'\n处理被试: {len(qc_df)}')
    print(f'有效被试: {len(valid)}')
    if len(valid) == 0:
        return

    print(f'\n[stimulus-locked epochs]')
    print(f'  总 epochs: {int(valid["n_stim_kept"].sum())}')
    print(f'  平均保留: {valid["n_stim_kept"].mean():.1f} ± {valid["n_stim_kept"].std():.1f}')
    rr = (valid['stim_reject_rate'].dropna()
          if 'stim_reject_rate' in valid.columns else pd.Series(dtype=float))
    if len(rr) > 0:
        print(f'  平均 reject: {rr.mean()*100:.1f}% ± {rr.std()*100:.1f}%')
    else:
        print('  平均 reject: 不可用（缓存重建行无原始 trial 总数）')

    if 'resp_reject_rate' in valid.columns:
        resp_valid = valid[valid['resp_reject_rate'].notna()]
        if len(resp_valid) > 0:
            print(f'\n[response-locked epochs]')
            print(f'  总 epochs: {int(resp_valid["n_resp_kept"].sum())}')
            print(f'  平均保留: {resp_valid["n_resp_kept"].mean():.1f} ± {resp_valid["n_resp_kept"].std():.1f}')

    # trial_type 分布
    print(f'\n[trial_type 分布]')
    for tt in ['health', 'taste', 'decision']:
        col = f'n_{tt}_kept'
        if col in valid.columns:
            print(f'  {tt}: {valid[col].sum()} total, {valid[col].mean():.1f} ± {valid[col].std():.1f} per subject')

    # 坏通道
    if 'interpolated_channels' in valid.columns:
        all_bads = []
        for x in valid['interpolated_channels'].fillna('[]'):
            try:
                all_bads.extend(eval(x) if isinstance(x, str) else [])
            except Exception:
                pass
        if all_bads:
            from collections import Counter
            bad_counts = Counter(all_bads)
            print(f'\n[坏通道统计]')
            print(f'  被插值的被试: {len(valid[valid["interpolated_channels"].fillna("[]").apply(lambda x: len(eval(x)) > 0 if isinstance(x, str) else False)])}')
            print(f'  最常被插值的通道:')
            for ch, count in bad_counts.most_common(10):
                print(f'    {ch}: {count} 次')

    # 可视化
    plot_qc_overview(qc_df, FIGURES_DIR / 'fig7_qc_overview.png')

    # 示例 ERP（用前 3 个有效被试）
    example_subs = valid['subject_id'].head(3).tolist()
    for i, sid in enumerate(example_subs):
        try:
            plot_example_erp(sid, FIGURES_DIR / f'fig8_example_erp_{sid}.png')
        except Exception as e:
            print(f'[QC] {sid} ERP 绘制失败: {e}')

    print(f'\n[QC 报告完成] 图表保存至 {FIGURES_DIR}')


if __name__ == '__main__':
    generate_qc_report()
