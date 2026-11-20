# -*- coding: utf-8 -*-
"""
切入点 2 补充：时频滑动窗口解码
================================

验证：在全时间窗解码未显著的基础上，是否存在局部时间窗的可分性
（被全时间窗平均稀释的局部信号）

设计：
- 滑动窗口：100ms 窗，50ms 步长，0-1000ms 覆盖
- 19 个时间窗（0-100, 50-150, ..., 900-1000ms）
- 协方差矩阵在每个滑动窗内重新估计
- 分类器：TangentSpace(logeuclid) + StandardScaler + LogisticRegression
- LOSO 评估
- 主要对比：taste-health
- 输出：每个时间窗的群体准确率，找出峰值时间点
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import json
import logging
from scipy import stats
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.base import clone

# 添加 code 目录到 sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 配置 logging：同时输出到文件和 stdout（line-buffered）
_LOG_PATH = Path(__file__).parent.parent / 'outputs' / 'logs' / 'sliding_window_full.log'
_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
    handlers=[
        logging.FileHandler(_LOG_PATH, mode='w', encoding='utf-8'),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)
print = log.info  # 重定向 print 到 logging

from config import FIGURES_DIR, RESULTS_DIR, merge_results_csv
from eeg_preprocessing import (
    load_subject_epochs, EEG_CHANNEL_NAMES, QC_DIR,
    TMIN_STIM as TMIN, TMAX_STIM as TMAX,
)
from pyriemann.estimation import Covariances
from pyriemann.classification import TangentSpace

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

FS = 512
RANDOM_STATE = 42
MAX_TRIALS_PER_TYPE = 10  # 减小到 10 以加速 LOSO（训练集 94×20=1880 trials）

# 滑动窗口参数
WIN_MS = 100       # 窗长 100ms
STEP_MS = 50       # 步长 50ms
START_MS = 0       # 起始 0ms（覆盖后刺激期）
END_MS = 1000      # 终止 1000ms

# 主对比（核心：Sullivan 时序假说验证）
CONTRAST = ('taste', 'health')


def load_subject_list():
    qc_df = pd.read_csv(QC_DIR / 'qc_summary.csv')
    valid = qc_df[qc_df['n_stim_kept'].notna()]
    valid = valid[(valid['n_taste_kept'] >= 30)
                  & (valid['n_health_kept'] >= 30)
                  & (valid['n_decision_kept'] >= 30)]
    return valid['subject_id'].tolist()


def compute_sliding_windows():
    """生成滑动窗口列表（时间点为窗中心）"""
    t_axis = np.linspace(TMIN, TMAX, int((TMAX - TMIN) * FS) + 1)
    win_samples = int(WIN_MS * FS / 1000)
    step_samples = int(STEP_MS * FS / 1000)
    start_idx = int((START_MS / 1000 - TMIN) * FS)
    end_idx = int((END_MS / 1000 - TMIN) * FS)

    windows = []
    t_start = start_idx
    while t_start + win_samples <= end_idx + 1:
        t_end = t_start + win_samples
        t_center = (t_axis[t_start] + t_axis[t_end - 1]) / 2 * 1000  # ms
        windows.append({
            't_start_idx': t_start,
            't_end_idx': t_end,
            't_center_ms': t_center,
            'label': f'{int(t_axis[t_start] * 1000)}-{int(t_axis[t_end - 1] * 1000)}ms',
        })
        t_start += step_samples

    return windows, t_axis


# ============ 数据缓存 ============
_DATA_CACHE = None


def preload_data(subject_ids):
    """预加载所有被试的原始 epochs 到内存（不做协方差预计算，
    因为每个滑动窗需要重新切片）"""
    global _DATA_CACHE
    _DATA_CACHE = {}
    for i, sid in enumerate(subject_ids):
        try:
            X, labels = load_subject_epochs(sid, 'stim')
            _DATA_CACHE[sid] = (X, np.array(labels['trial_type']))
        except Exception as e:
            print(f'  [skip] {sid}: {e}')
            continue
        if (i + 1) % 20 == 0:
            print(f'  preload: {i+1}/{len(subject_ids)}')
    print(f'  预加载完成: {len(_DATA_CACHE)}/{len(subject_ids)} 被试')
    return _DATA_CACHE


def run_sliding_window_loso(subject_ids, contrast, windows):
    """滑动窗口 LOSO 解码

    返回
    ----
    results : list of dict, 每个被试×每个窗口一行
    """
    a, b = contrast
    cov_estimator = Covariances(estimator='lwf')
    clf_template = Pipeline([
        ('ts', TangentSpace(metric='logeuclid')),
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(max_iter=1000, C=1.0,
                                    solver='lbfgs', random_state=RANDOM_STATE))
    ])

    # 预先按被试收集 trial mask（避免重复计算）
    sub_data = {}
    for sid in subject_ids:
        if sid not in _DATA_CACHE:
            continue
        X, tt = _DATA_CACHE[sid]
        mask = np.array([(t == a or t == b) for t in tt])
        if mask.sum() < 10:
            continue
        y = np.array([1 if t == a else 0 for t in tt[mask]])
        if len(np.unique(y)) < 2:
            continue
        sub_data[sid] = (X[mask], y)

    print(f'  有效被试: {len(sub_data)}')

    # 对每个窗口做 LOSO
    all_results = []
    for w_idx, win in enumerate(windows):
        t_start, t_end = win['t_start_idx'], win['t_end_idx']
        win_label = win['label']
        t_center = win['t_center_ms']

        # 预计算每个被试在该窗口的协方差矩阵
        sub_covs = {}
        for sid, (X, y) in sub_data.items():
            X_win = X[:, :, t_start:t_end]
            covs = cov_estimator.transform(X_win)
            sub_covs[sid] = (covs, y)

        # LOSO
        accs = []
        for test_sid in sub_data.keys():
            covs_test, y_test = sub_covs[test_sid]

            # 训练集（其他被试，平衡抽样）
            cov_train_list, y_train_list = [], []
            rng = np.random.RandomState(RANDOM_STATE + w_idx)
            for train_sid in sub_data.keys():
                if train_sid == test_sid:
                    continue
                covs_t, y_t = sub_covs[train_sid]
                # 平衡抽样
                idx_a = np.where(y_t == 1)[0]
                idx_b = np.where(y_t == 0)[0]
                n_a = min(len(idx_a), MAX_TRIALS_PER_TYPE)
                n_b = min(len(idx_b), MAX_TRIALS_PER_TYPE)
                idx_a = rng.choice(idx_a, n_a, replace=False)
                idx_b = rng.choice(idx_b, n_b, replace=False)
                idx = np.concatenate([idx_a, idx_b])
                cov_train_list.append(covs_t[idx])
                y_train_list.append(y_t[idx])

            cov_train = np.concatenate(cov_train_list, axis=0)
            y_train = np.concatenate(y_train_list, axis=0)

            try:
                clf = clone(clf_template)
                clf.fit(cov_train, y_train)
                y_pred = clf.predict(covs_test)
                acc = balanced_accuracy_score(y_test, y_pred)
                accs.append(acc)
                all_results.append({
                    'subject_id': test_sid,
                    'window': win_label,
                    't_center_ms': t_center,
                    'contrast': f'{a}-{b}',
                    'balanced_acc': acc,
                    'n_test': len(y_test),
                })
            except Exception as e:
                continue

        if len(accs) > 0:
            mean_acc = np.mean(accs)
            std_acc = np.std(accs, ddof=1)
            t_stat, p_val = stats.ttest_1samp(accs, 0.5)
            d = (mean_acc - 0.5) / std_acc if std_acc > 0 else 0
            sig = '*' if p_val < 0.05 else ''
            print(f'  [{w_idx+1}/{len(windows)}] {win_label} (t={t_center:.0f}ms): '
                  f'mean={mean_acc:.3f}±{std_acc:.3f}, t={t_stat:.2f}, '
                  f'p={p_val:.4f} {sig}')

    return all_results


def plot_sliding_window_results(results_df, save_path):
    """滑动窗口解码结果可视化"""
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # 1. 群体准确率随时间窗变化
    ax = axes[0]
    grouped = results_df.groupby('t_center_ms')['balanced_acc']
    means = grouped.mean()
    sems = grouped.sem()
    t_centers = means.index

    ax.plot(t_centers, means.values, 'o-', color='steelblue', linewidth=2,
            markersize=8, label='Mean balanced acc')
    ax.fill_between(t_centers, means - sems, means + sems,
                   alpha=0.3, color='steelblue', label='±1 SEM')

    ax.axhline(0.5, color='black', linestyle='--', label='Chance (0.5)')

    # 标记 Sullivan 预测时间窗
    ax.axvspan(450, 650, alpha=0.15, color='red', label='LPP (Sullivan taste)')
    ax.axvspan(800, 1000, alpha=0.15, color='green', label='Late (Sullivan health)')

    # 标记显著点
    for t_c in t_centers:
        sub = results_df[results_df['t_center_ms'] == t_c]['balanced_acc']
        if len(sub) >= 10:
            t_stat, p_val = stats.ttest_1samp(sub, 0.5)
            if p_val < 0.05:
                ax.plot(t_c, means[t_c], 'r*', markersize=15, zorder=5)

    ax.set_xlabel('Time window center (ms)')
    ax.set_ylabel('Balanced accuracy')
    ax.set_title('Sliding-window decoding: tastiness vs healthiness (LOSO)')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)

    # 2. 单被试分布（每个时间窗）
    ax = axes[1]
    windows = sorted(results_df['t_center_ms'].unique())
    data_per_window = [results_df[results_df['t_center_ms'] == w]['balanced_acc'].values
                       for w in windows]
    bp = ax.boxplot(data_per_window, positions=windows, widths=40,
                    patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightsteelblue')
        patch.set_alpha(0.6)

    ax.axhline(0.5, color='black', linestyle='--', label='Chance')
    ax.set_xlabel('Time window center (ms)')
    ax.set_ylabel('Balanced accuracy')
    ax.set_title('Per-subject distribution across sliding windows')
    ax.legend()

    plt.suptitle('Sliding-window Riemannian decoding (100ms window, 50ms step)',
                fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'[fig] saved: {save_path}')


def plot_temporal_decoding_curve(results_df, save_path):
    """时间解码曲线（更精细的可视化）"""
    fig, ax = plt.subplots(figsize=(10, 5))

    grouped = results_df.groupby('t_center_ms')['balanced_acc']
    means = grouped.mean()
    sems = grouped.sem()
    t_centers = means.index

    # 平滑曲线
    from scipy.ndimage import gaussian_filter1d
    means_smooth = gaussian_filter1d(means.values, sigma=1.5)

    ax.plot(t_centers, means.values, 'o', color='steelblue',
            alpha=0.5, markersize=6, label='Raw mean')
    ax.plot(t_centers, means_smooth, '-', color='steelblue', linewidth=2.5,
            label='Smoothed mean')
    ax.fill_between(t_centers, means - sems, means + sems,
                   alpha=0.25, color='steelblue')

    ax.axhline(0.5, color='black', linestyle='--', alpha=0.7, label='Chance')

    # Sullivan 预测窗
    ax.axvspan(450, 650, alpha=0.1, color='red')
    ax.axvspan(800, 1000, alpha=0.1, color='green')
    ax.text(550, ax.get_ylim()[1] * 0.95, 'LPP\n(taste)', ha='center',
            fontsize=9, color='red')
    ax.text(900, ax.get_ylim()[1] * 0.95, 'Late\n(health)', ha='center',
            fontsize=9, color='green')

    # 显著点
    sig_times = []
    for t_c in t_centers:
        sub = results_df[results_df['t_center_ms'] == t_c]['balanced_acc']
        if len(sub) >= 10:
            t_stat, p_val = stats.ttest_1samp(sub, 0.5)
            if p_val < 0.05:
                sig_times.append(t_c)
    if sig_times:
        ax.plot(sig_times, [means[t] for t in sig_times], 'r*',
                markersize=15, zorder=5, label=f'Significant (p<0.05, n={len(sig_times)})')

    ax.set_xlabel('Time window center (ms post-stimulus)', fontsize=11)
    ax.set_ylabel('Balanced accuracy (LOSO)', fontsize=11)
    ax.set_title('Temporal decoding curve: tastiness vs healthiness', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'[fig] saved: {save_path}')


def main(n_subjects=None):
    print('=' * 60)
    print('补充实验：时频滑动窗口解码')
    print('=' * 60)

    subject_ids = load_subject_list()
    if n_subjects is not None:
        subject_ids = subject_ids[:n_subjects]
    print(f'\n有效被试: {len(subject_ids)}')

    # 计算滑动窗口
    windows, t_axis = compute_sliding_windows()
    print(f'\n滑动窗口: {len(windows)} 个')
    print(f'  窗长 {WIN_MS}ms, 步长 {STEP_MS}ms, '
          f'范围 {START_MS}-{END_MS}ms')
    print(f'  首窗: {windows[0]["label"]} (中心 {windows[0]["t_center_ms"]:.0f}ms)')
    print(f'  末窗: {windows[-1]["label"]} (中心 {windows[-1]["t_center_ms"]:.0f}ms)')

    # 预加载数据
    print('\n[step 0] 预加载数据到内存...')
    preload_data(subject_ids)

    # 滑动窗口 LOSO 解码
    print(f'\n[step 1] 滑动窗口 LOSO 解码 (contrast={CONTRAST[0]}-{CONTRAST[1]})...')
    results = run_sliding_window_loso(subject_ids, CONTRAST, windows)

    results_df = pd.DataFrame(results)
    slide_path = RESULTS_DIR / 'sliding_window_decoding_results.csv'
    results_df = merge_results_csv(results_df, slide_path,
                                   key_cols=['subject_id', 'window'])
    results_df.to_csv(slide_path, index=False)
    print(f'\n[saved] {slide_path}')

    # 群体统计
    print('\n[step 2] 群体统计...')
    # 阈值：有效被试数 >= max(10, n_subjects 的 80%)，避免小样本下 t 检验不稳定
    min_n = max(10, int(len(subject_ids) * 0.8))
    window_stats = []
    for win_label in results_df['window'].unique():
        sub = results_df[results_df['window'] == win_label]['balanced_acc']
        if len(sub) < min_n:
            continue
        t_c = results_df[results_df['window'] == win_label]['t_center_ms'].iloc[0]
        mean_acc = sub.mean()
        std_acc = sub.std(ddof=1)
        t_stat, p_val = stats.ttest_1samp(sub, 0.5)
        d = (mean_acc - 0.5) / std_acc if std_acc > 0 else 0

        # FDR 校正
        window_stats.append({
            'window': win_label,
            't_center_ms': t_c,
            'n': len(sub),
            'mean': mean_acc,
            'std': std_acc,
            't_stat': t_stat,
            'p_value': p_val,
            'cohen_d': d,
        })

    stats_df = pd.DataFrame(window_stats)
    if len(stats_df) == 0:
        print(f'[警告] 没有窗口满足最小样本数阈值 (n>={min_n})，跳过群体统计。')
        # 创建一个空的 stats.csv 占位
        pd.DataFrame(columns=['window', 't_center_ms', 'n', 'mean', 'std',
                              't_stat', 'p_value', 'p_fdr', 'sig_fdr',
                              'sig_uncorrected', 'cohen_d']).to_csv(
            RESULTS_DIR / 'sliding_window_stats.csv', index=False)
        # 跳过 FDR / 峰值 / 汇总
        summary = {
            'n_subjects': len(subject_ids),
            'contrast': f'{CONTRAST[0]}-{CONTRAST[1]}',
            'window_params': {
                'win_ms': WIN_MS, 'step_ms': STEP_MS,
                'start_ms': START_MS, 'end_ms': END_MS,
                'n_windows': len(windows),
            },
            'note': f'有效被试数不足（min_n={min_n}），未做群体统计',
            'windows': [],
        }
        with open(RESULTS_DIR / 'sliding_window_summary.json', 'w',
                  encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
        print(f'[saved] {RESULTS_DIR / "sliding_window_stats.csv"}')
        print(f'[summary] {RESULTS_DIR / "sliding_window_summary.json"}')
        # 仍然画图（基于单被试分布）
        print('\n[step 3] 可视化...')
        try:
            plot_sliding_window_results(
                results_df,
                FIGURES_DIR / 'fig21_sliding_window_decoding.png')
            plot_temporal_decoding_curve(
                results_df,
                FIGURES_DIR / 'fig22_temporal_decoding_curve.png')
        except Exception as e:
            print(f'[warn] 可视化失败: {e}')
        print('\n' + '=' * 60)
        print('补充实验结束（样本不足，未做统计）')
        print('=' * 60)
        return

    from mne.stats import fdr_correction
    _, p_fdr = fdr_correction(stats_df['p_value'].values)
    stats_df['p_fdr'] = p_fdr
    stats_df['sig_fdr'] = p_fdr < 0.05
    stats_df['sig_uncorrected'] = stats_df['p_value'] < 0.05

    stats_df.to_csv(RESULTS_DIR / 'sliding_window_stats.csv', index=False)
    print(f'[saved] {RESULTS_DIR / "sliding_window_stats.csv"}')

    # 打印结果
    print('\n群体统计:')
    print(stats_df[['window', 't_center_ms', 'mean', 'std', 't_stat',
                     'p_value', 'p_fdr', 'sig_uncorrected', 'sig_fdr']].to_string())

    # 找峰值
    peak_idx = stats_df['mean'].idxmax()
    peak = stats_df.loc[peak_idx]
    print(f'\n峰值时间窗: {peak["window"]} (t={peak["t_center_ms"]:.0f}ms)')
    print(f'  mean={peak["mean"]:.3f}, t={peak["t_stat"]:.2f}, '
          f'p={peak["p_value"]:.4f}, p_fdr={peak["p_fdr"]:.4f}')

    # 可视化
    print('\n[step 3] 可视化...')
    plot_sliding_window_results(
        results_df,
        FIGURES_DIR / 'fig21_sliding_window_decoding.png'
    )
    plot_temporal_decoding_curve(
        results_df,
        FIGURES_DIR / 'fig22_temporal_decoding_curve.png'
    )

    # 汇总
    n_sig_uncorrected = stats_df['sig_uncorrected'].sum()
    n_sig_fdr = stats_df['sig_fdr'].sum()
    summary = {
        'n_subjects': len(subject_ids),
        'contrast': f'{CONTRAST[0]}-{CONTRAST[1]}',
        'window_params': {
            'win_ms': WIN_MS,
            'step_ms': STEP_MS,
            'start_ms': START_MS,
            'end_ms': END_MS,
            'n_windows': len(windows),
        },
        'n_sig_uncorrected': int(n_sig_uncorrected),
        'n_sig_fdr': int(n_sig_fdr),
        'peak_window': {
            'label': peak['window'],
            't_center_ms': float(peak['t_center_ms']),
            'mean': float(peak['mean']),
            't_stat': float(peak['t_stat']),
            'p_value': float(peak['p_value']),
            'p_fdr': float(peak['p_fdr']),
        },
        'windows': stats_df.to_dict(orient='records'),
    }

    with open(RESULTS_DIR / 'sliding_window_summary.json', 'w',
              encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    print(f'\n[summary] {RESULTS_DIR / "sliding_window_summary.json"}')

    print('\n' + '=' * 60)
    print('补充实验完成：时频滑动窗口解码')
    print('=' * 60)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=None,
                        help='只用前 n 个被试（调试用）')
    args = parser.parse_args()
    main(n_subjects=args.n)
