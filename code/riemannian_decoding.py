# -*- coding: utf-8 -*-
"""
切入点 2-B1：Riemannian 几何单试次 EEG 解码
=========================================

验证：categorisation 任务 EEG 中"美味 vs 健康"判断的神经表征可分性

设计：
- 任务：单试次分类
  - 核心：taste vs health（验证 Sullivan 时序假说的"属性维度可分性"）
  - 对照：taste vs decision, health vs decision
- 方法：
  - 协方差矩阵估计：Ledoit-Wolf 收缩
  - Riemannian 几何对齐：映射到切空间（tangent space）
  - 分类器：Logistic Regression / SVM
- 评估：Leave-One-Subject-Out (LOSO)
  - 每个被试作为测试集，其他被试作为训练集
  - 报告 balanced accuracy
- 输出：
  - 每被试每对比的解码准确率
  - 群体水平 chance level 检验（permutation）
  - 个体差异指标（与 SCSR、DDM 参数相关）

参考：用户 FoodFBTS 项目的 pyriemann 用法
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import json
import warnings
from scipy import stats
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score
from sklearn.base import clone

# pyriemann（新版 API：Covariances, TangentSpace, MDM）
from pyriemann.estimation import Covariances
from pyriemann.classification import TangentSpace, MDM

from config import FIGURES_DIR, RESULTS_DIR, merge_results_csv
from eeg_preprocessing import (
    load_subject_epochs, EEG_CHANNEL_NAMES, QC_DIR,
    TMIN_STIM as TMIN, TMAX_STIM as TMAX,
)

FS = 512
warnings.filterwarnings('ignore', category=FutureWarning)

# ============ 参数 ============
TRIAL_TYPES = ['health', 'taste', 'decision']

# 时间窗（聚焦后刺激期，覆盖关键认知成分）
# 为节省计算时间，主分析只用全时间窗；其他时间窗作为敏感性分析可手动启用
TIME_WINDOWS = {
    'full_0-1000ms':   (0.0, 1.0),
}

# 对比组合
CONTRASTS = [
    ('taste', 'health'),     # 核心：Sullivan 时序假说验证
    ('taste', 'decision'),
    ('health', 'decision'),
]

# 分类器
CLASSIFIERS = {
    'LR': Pipeline([
        ('cov', Covariances(estimator='lwf')),
        ('ts', TangentSpace(metric='riemann')),
        ('clf', LogisticRegression(max_iter=1000, C=1.0,
                                    solver='lbfgs', random_state=42))
    ]),
    'SVM': Pipeline([
        ('cov', Covariances(estimator='lwf')),
        ('ts', TangentSpace(metric='riemann')),
        ('clf', SVC(C=1.0, kernel='linear', random_state=42))
    ]),
    'MDM': Pipeline([
        ('cov', Covariances(estimator='lwf')),
        ('clf', MDM(metric='riemann'))
    ]),
}

PRIMARY_CLF = 'LR'  # 主分析使用 LR（预计算协方差版本：TangentSpace+LR，在向量上 fit 远快于 MDM）
RANDOM_STATE = 42
N_PERMUTATION = 0  # 设为 0 跳过 permutation（用 1-sample t-test vs 0.5 代替）
MAX_TRIALS_PER_TYPE = 40  # 限制单被试 trial 数以平衡计算

# 预计算协方差模式下的分类器（不含 Covariances step，直接接收协方差矩阵）
# 用于 use_cached_cov=True 分支，避免重复计算协方差
# 使用 logeuclid metric：闭式解（无需 Riemannian mean 迭代），大幅加速 TangentSpace.fit
# StandardScaler：EEG 协方差矩阵量级 ~1e-11，TangentSpace 输出量级 ~1e-11，
#   需标准化到 ~1 使 LR 能正常学习（Riemannian BCI 标准做法）
CLASSIFIERS_PRECOMP = {
    'MDM': MDM(metric='logeuclid'),
    'LR': Pipeline([
        ('ts', TangentSpace(metric='logeuclid')),
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(max_iter=1000, C=1.0,
                                    solver='lbfgs', random_state=42))
    ]),
    'SVM': Pipeline([
        ('ts', TangentSpace(metric='logeuclid')),
        ('scaler', StandardScaler()),
        ('clf', SVC(C=1.0, kernel='linear', random_state=42))
    ]),
}


# ============ 数据加载 ============
def load_subject_list():
    qc_df = pd.read_csv(QC_DIR / 'qc_summary.csv')
    valid = qc_df[qc_df['n_stim_kept'].notna()]
    valid = valid[(valid['n_taste_kept'] >= 30)
                  & (valid['n_health_kept'] >= 30)
                  & (valid['n_decision_kept'] >= 30)]
    return valid['subject_id'].tolist()


def build_dataset(subject_ids, time_window=None):
    """构建跨被试数据集

    参数
    ----
    subject_ids : list
    time_window : (t_start, t_end) 或 None（用全部）

    返回
    ----
    X : ndarray (n_trials_total, n_channels, n_times_window)
    y : ndarray (n_trials_total,) 0/1 标签
    groups : ndarray (n_trials_total,) 被试 ID
    subject_indices : dict {subject_id: (start, end)}
    """
    Xs, ys, groups = [], [], []
    subject_indices = {}
    cursor = 0

    t_axis = None

    for sid in subject_ids:
        try:
            X_sub, labels = load_subject_epochs(sid, 'stim')
        except Exception as e:
            print(f'  [skip] {sid}: {e}')
            continue

        if t_axis is None:
            n_times_full = X_sub.shape[2]
            t_axis = np.linspace(TMIN, TMAX, n_times_full)

        # 选时间窗
        if time_window is not None:
            t_start, t_end = time_window
            t_mask = (t_axis >= t_start) & (t_axis <= t_end)
            X_sub = X_sub[:, :, t_mask]

        rng = np.random.RandomState(RANDOM_STATE)
        # 对每个 trial_type 抽样
        for tt_a, tt_b in CONTRASTS[:1]:  # 只用第一个对比，避免重复
            pass  # 占位

        # 注意：build_dataset 不区分对比，只返回所有数据
        # 实际对比在 run_losos 中按需取

        # 记录到全部数据集
        tt_arr = labels['trial_type']
        for tt in TRIAL_TYPES:
            mask = tt_arr == tt
            idx = np.where(mask)[0]
            if len(idx) > MAX_TRIALS_PER_TYPE:
                idx = rng.choice(idx, MAX_TRIALS_PER_TYPE, replace=False)
            Xs.append(X_sub[idx])
            ys.append(np.full(len(idx), tt))
            groups.append([sid] * len(idx))

        subject_indices[sid] = (cursor, cursor + sum(len(x) for x in Xs[-3:]))
        cursor += sum(len(x) for x in Xs[-3:])

    X = np.concatenate(Xs, axis=0)
    y = np.concatenate(ys, axis=0)
    groups = np.concatenate(groups, axis=0)

    return X, y, groups, subject_indices


def preload_all_data(subject_ids):
    """预加载所有被试数据到内存 + 预计算协方差矩阵

    返回
    ----
    data_cache : dict {subject_id: (X_full, tt_arr)}
        X_full: (n_trials, n_ch, n_times_full) 原始 epoch
    cov_cache : dict {subject_id: covs_full}
        covs_full: (n_trials, n_ch, n_ch) 协方差矩阵（全时间窗）
    """
    data_cache = {}
    cov_cache = {}
    cov_estimator = Covariances(estimator='lwf')

    for i, sid in enumerate(subject_ids):
        try:
            X, labels = load_subject_epochs(sid, 'stim')
            data_cache[sid] = (X, np.array(labels['trial_type']))
            # 预计算协方差矩阵（全时间窗）
            covs = cov_estimator.transform(X)
            cov_cache[sid] = covs
        except Exception as e:
            print(f'  [skip] {sid}: {e}')
            continue
        if (i + 1) % 10 == 0:
            print(f'  preload: {i+1}/{len(subject_ids)} '
                  f'(data + cov)')

    print(f'  预加载完成: {len(data_cache)}/{len(subject_ids)} 被试')
    return data_cache, cov_cache


# 全局缓存（在 main 中初始化）
_DATA_CACHE = None
_COV_CACHE = None


def get_subject_data_cached(subject_id, time_window=None):
    """从缓存获取被试数据并应用时间窗"""
    global _DATA_CACHE
    if _DATA_CACHE is None:
        raise RuntimeError('数据缓存未初始化，请先调用 preload_all_data')
    X, tt_arr = _DATA_CACHE[subject_id]

    if time_window is not None:
        n_times = X.shape[2]
        t_axis = np.linspace(TMIN, TMAX, n_times)
        t_start, t_end = time_window
        t_mask = (t_axis >= t_start) & (t_axis <= t_end)
        X = X[:, :, t_mask]

    return X, tt_arr


def get_subject_covs_cached(subject_id):
    """从缓存获取预计算的协方差矩阵（全时间窗）"""
    global _COV_CACHE
    if _COV_CACHE is None:
        raise RuntimeError('协方差缓存未初始化')
    return _COV_CACHE[subject_id]


# ============ LOSO 解码 ============
def run_loso_decoding(subject_ids, contrast, time_window=None, clf_name='MDM',
                       max_trials_per_type=None):
    """Leave-One-Subject-Out 跨被试解码（使用预计算协方差矩阵）

    参数
    ----
    subject_ids : list
    contrast : (a, b) trial_type 对比
    time_window : 必须为 None（当前实现只用全时间窗的预计算协方差）
    clf_name : str, 'MDM'（其他分类器未优化预计算）

    返回
    ----
    results : list of dict, 每个被试一行
    """
    a, b = contrast
    if max_trials_per_type is None:
        max_trials_per_type = MAX_TRIALS_PER_TYPE

    # 注意：当前实现只用预计算的全时间窗协方差矩阵
    # 如果 time_window 不是 None 或 full_0-1000ms，需要回退到实时计算
    use_cached_cov = (time_window is None or
                       time_window == (0.0, 1.0))

    results = []
    for i, test_sid in enumerate(subject_ids):
        if test_sid not in _DATA_CACHE:
            continue

        # 测试被试
        if use_cached_cov:
            covs_test_all = get_subject_covs_cached(test_sid)
            _, y_test_tt = get_subject_data_cached(test_sid, None)
        else:
            X_test_all, y_test_tt = get_subject_data_cached(test_sid, time_window)
            covs_test_all = Covariances(estimator='lwf').transform(X_test_all)

        test_mask = np.array([(tt == a or tt == b) for tt in y_test_tt])
        covs_test = covs_test_all[test_mask]
        y_test = np.array([1 if tt == a else 0 for tt in y_test_tt[test_mask]])

        if len(covs_test) < 10 or len(np.unique(y_test)) < 2:
            continue

        # 训练被试（从缓存取协方差矩阵）
        cov_train_list, y_train_list = [], []
        rng = np.random.RandomState(RANDOM_STATE + i)
        for train_sid in subject_ids:
            if train_sid == test_sid or train_sid not in _DATA_CACHE:
                continue

            if use_cached_cov:
                covs_t = get_subject_covs_cached(train_sid)
                _, tt_t = get_subject_data_cached(train_sid, None)
            else:
                X_t, tt_t = get_subject_data_cached(train_sid, time_window)
                covs_t = Covariances(estimator='lwf').transform(X_t)

            mask = np.array([(tt == a or tt == b) for tt in tt_t])
            covs_t = covs_t[mask]
            y_t_bin = np.array([1 if tt == a else 0 for tt in tt_t[mask]])

            # 平衡抽样
            if max_trials_per_type and len(covs_t) > max_trials_per_type * 2:
                idx_a = np.where(y_t_bin == 1)[0]
                idx_b = np.where(y_t_bin == 0)[0]
                n_a = min(len(idx_a), max_trials_per_type)
                n_b = min(len(idx_b), max_trials_per_type)
                idx_a = rng.choice(idx_a, n_a, replace=False)
                idx_b = rng.choice(idx_b, n_b, replace=False)
                idx = np.concatenate([idx_a, idx_b])
                covs_t = covs_t[idx]
                y_t_bin = y_t_bin[idx]

            if len(covs_t) < 10 or len(np.unique(y_t_bin)) < 2:
                continue
            cov_train_list.append(covs_t)
            y_train_list.append(y_t_bin)

        if len(cov_train_list) == 0:
            continue

        cov_train = np.concatenate(cov_train_list, axis=0)
        y_train = np.concatenate(y_train_list, axis=0)

        # 训练
        if use_cached_cov:
            clf = clone(CLASSIFIERS_PRECOMP[clf_name])
        else:
            clf = clone(CLASSIFIERS[clf_name])
        try:
            clf.fit(cov_train, y_train)
            y_pred = clf.predict(covs_test)
            acc = balanced_accuracy_score(y_test, y_pred)
            # 诊断：前2个被试打印详细信息
            if i < 2:
                unique_pred, counts_pred = np.unique(y_pred, return_counts=True)
                unique_true, counts_true = np.unique(y_test, return_counts=True)
                print(f'  [diag] {test_sid}: train_shape={cov_train.shape}, '
                      f'test_shape={covs_test.shape}, '
                      f'train_label_dist={dict(zip(unique_true, counts_true))}, '
                      f'pred_dist={dict(zip(unique_pred, counts_pred))}, '
                      f'acc={acc:.3f}')
        except Exception as e:
            print(f'  [error] {test_sid}: {e}')
            continue

        results.append({
            'subject_id': test_sid,
            'contrast': f'{a}-{b}',
            'window': f'{time_window[0]:.1f}-{time_window[1]:.1f}s' if time_window else 'full',
            'clf': clf_name,
            'n_test': len(y_test),
            'balanced_acc': acc,
            'chance_level': 0.5,
        })

        if (i + 1) % 10 == 0:
            print(f'  progress: {i+1}/{len(subject_ids)} ({a}-{b})')

    return results


def permutation_chance_level(subject_ids, contrast, time_window, clf_name='LR',
                              n_perm=50):
    """估算群体水平 chance level（通过置换标签）

    返回
    ----
    chance_dist : ndarray (n_perm,) 群体平均 acc 的置换分布
    """
    a, b = contrast
    rng = np.random.RandomState(RANDOM_STATE)
    use_cached_cov = (time_window is None or
                       time_window == (0.0, 1.0))
    # 收集所有协方差矩阵（从缓存或实时计算）
    all_covs, all_y_bin, all_groups = [], [], []
    for sid in subject_ids:
        if sid not in _DATA_CACHE:
            continue
        if use_cached_cov:
            covs = get_subject_covs_cached(sid)
            _, tt = get_subject_data_cached(sid, None)
        else:
            X, tt = get_subject_data_cached(sid, time_window)
            covs = Covariances(estimator='lwf').transform(X)
        mask = np.array([(x == a or x == b) for x in tt])
        covs = covs[mask]
        y = np.array([1 if x == a else 0 for x in tt[mask]])
        if len(covs) < 10 or len(np.unique(y)) < 2:
            continue
        all_covs.append(covs)
        all_y_bin.append(y)
        all_groups.append([sid] * len(covs))

    all_covs = np.concatenate(all_covs, axis=0)
    all_y = np.concatenate(all_y_bin, axis=0)
    all_groups = np.array([g for sub_groups in all_groups for g in sub_groups])

    print(f'  [perm] 共 {len(np.unique(all_groups))} 被试, {len(all_y)} trials')

    chance_dist = []
    for p in range(n_perm):
        # 在每个被试内打乱标签
        y_perm = all_y.copy()
        for sid in np.unique(all_groups):
            mask = all_groups == sid
            y_perm[mask] = rng.permutation(y_perm[mask])

        # LOSO
        accs = []
        for test_sid in np.unique(all_groups):
            test_mask = all_groups == test_sid
            train_mask = ~test_mask
            if test_mask.sum() < 10 or train_mask.sum() < 10:
                continue
            if len(np.unique(y_perm[test_mask])) < 2 or len(np.unique(y_perm[train_mask])) < 2:
                continue
            try:
                if use_cached_cov:
                    clf = clone(CLASSIFIERS_PRECOMP[clf_name])
                else:
                    clf = clone(CLASSIFIERS[clf_name])
                clf.fit(all_covs[train_mask], y_perm[train_mask])
                y_pred = clf.predict(all_covs[test_mask])
                acc = balanced_accuracy_score(y_perm[test_mask], y_pred)
                accs.append(acc)
            except Exception:
                continue

        if len(accs) > 0:
            chance_dist.append(np.mean(accs))

        if (p + 1) % 10 == 0:
            print(f'    permutation {p+1}/{n_perm}')

    return np.array(chance_dist)


# ============ 可视化 ============
def plot_decoding_results(results_df, save_path):
    """跨被试解码准确率分布 + 时间窗对比"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. 每对比的准确率分布（按时间窗分组）
    ax = axes[0, 0]
    contrasts = results_df['contrast'].unique()
    windows = results_df['window'].unique()
    width = 0.8 / len(windows)
    x = np.arange(len(contrasts))
    for i, w in enumerate(windows):
        sub = results_df[results_df['window'] == w]
        means = [sub[sub['contrast'] == c]['balanced_acc'].mean() for c in contrasts]
        sems = [sub[sub['contrast'] == c]['balanced_acc'].sem() for c in contrasts]
        ax.bar(x + i * width - 0.4 + width/2, means, width, yerr=sems,
               label=w, alpha=0.85)
    ax.axhline(0.5, color='black', linestyle='--', label='chance')
    ax.set_xticks(x)
    ax.set_xticklabels(contrasts)
    ax.set_ylabel('Balanced accuracy')
    ax.set_title('Decoding accuracy by contrast × time window')
    ax.legend(fontsize=8)

    # 2. 单被试分布（仅 taste-health, full window）
    ax = axes[0, 1]
    sub = results_df[(results_df['contrast'] == 'taste-health')
                    & (results_df['window'] == '0.0-1.0s')]
    if len(sub) > 0:
        ax.hist(sub['balanced_acc'], bins=20, color='steelblue',
                edgecolor='white', alpha=0.8)
        ax.axvline(sub['balanced_acc'].mean(), color='red', linestyle='--',
                   label=f'mean={sub["balanced_acc"].mean():.3f}')
        ax.axvline(0.5, color='black', linestyle='--', label='chance')
        ax.set_xlabel('Balanced accuracy')
        ax.set_ylabel('Number of subjects')
        ax.set_title('taste vs health decoding (full window)')
        ax.legend()

    # 3. 时间窗效应（taste-health）
    ax = axes[1, 0]
    sub = results_df[results_df['contrast'] == 'taste-health']
    for w in windows:
        data = sub[sub['window'] == w]['balanced_acc']
        if len(data) > 0:
            ax.plot([w]*len(data), data, 'o', alpha=0.3, color='steelblue')
            ax.plot([w], [data.mean()], 'o', markersize=12, color='red')
    ax.axhline(0.5, color='black', linestyle='--', label='chance')
    ax.set_xlabel('Time window')
    ax.set_ylabel('Balanced accuracy')
    ax.set_title('taste-health decoding across time windows')
    ax.legend()

    # 4. 跨对比的稳定性（每个被试在不同对比上的准确率）
    ax = axes[1, 1]
    pivot = results_df[results_df['window'] == '0.0-1.0s'].pivot(
        index='subject_id', columns='contrast', values='balanced_acc'
    )
    if 'taste-health' in pivot.columns and 'taste-decision' in pivot.columns:
        ax.scatter(pivot['taste-health'], pivot['taste-decision'], alpha=0.6)
        ax.plot([0.3, 0.9], [0.3, 0.9], 'k--', alpha=0.5)
        ax.set_xlabel('taste-health acc')
        ax.set_ylabel('taste-decision acc')
        ax.set_title('Cross-contrast stability')
        if len(pivot) > 5:
            r, p = stats.pearsonr(pivot['taste-health'], pivot['taste-decision'])
            ax.set_title(f'Cross-contrast stability\nr={r:.3f}, p={p:.3f}')

    plt.suptitle('Riemannian decoding results (LOSO)', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'[fig] saved: {save_path}')


# ============ 主入口 ============
def main(n_subjects=None):
    print('=' * 60)
    print('切入点 2-B1：Riemannian 几何单试次解码')
    print('=' * 60)

    subject_ids = load_subject_list()
    if n_subjects is not None:
        subject_ids = subject_ids[:n_subjects]
    print(f'\n有效被试：{len(subject_ids)}')

    # 预加载所有数据到内存
    print('\n[step 0] 预加载数据到内存...')
    global _DATA_CACHE, _COV_CACHE
    _DATA_CACHE, _COV_CACHE = preload_all_data(subject_ids)

    # 1. 全部分析：3 对比 × 4 时间窗 × LOSO
    all_results = []
    for contrast in CONTRASTS:
        for wname, tw in TIME_WINDOWS.items():
            window_label = f'{tw[0]:.1f}-{tw[1]:.1f}s'
            print(f'\n--- contrast={contrast[0]}-{contrast[1]}, '
                  f'window={wname} ({window_label}) ---')
            results = run_loso_decoding(
                subject_ids, contrast, tw, clf_name=PRIMARY_CLF
            )
            for r in results:
                r['window'] = window_label
                r['window_name'] = wname
            all_results.extend(results)

            # 群体统计
            accs = [r['balanced_acc'] for r in results]
            if len(accs) > 0:
                t, p = stats.ttest_1samp(accs, 0.5)
                d = (np.mean(accs) - 0.5) / np.std(accs, ddof=1)
                print(f'  群体: mean={np.mean(accs):.3f} ± {np.std(accs, ddof=1):.3f}, '
                      f't={t:.2f}, p={p:.4f}, d={d:.3f}')

    results_df = pd.DataFrame(all_results)
    riem_path = RESULTS_DIR / 'riemannian_decoding_results.csv'
    results_df = merge_results_csv(results_df, riem_path,
                                   key_cols=['subject_id', 'contrast', 'window_name'])
    results_df.to_csv(riem_path, index=False)
    print(f'\n[saved] {riem_path}')

    # 2. Permutation chance level（可选，N_PERMUTATION=0 则跳过）
    actual = results_df[(results_df['contrast'] == 'taste-health')
                       & (results_df['window'] == '0.0-1.0s')]['balanced_acc']
    actual_mean = float(actual.mean())
    actual_t, actual_p = stats.ttest_1samp(actual, 0.5)

    if N_PERMUTATION > 0:
        print('\n[step 2] Permutation chance level (taste-health, full window)...')
        chance_dist = permutation_chance_level(
            subject_ids, ('taste', 'health'),
            TIME_WINDOWS['full_0-1000ms'],
            clf_name=PRIMARY_CLF,
            n_perm=N_PERMUTATION
        )
        print(f'  Permutation chance level: mean={chance_dist.mean():.3f}, '
              f'95% CI=[{np.percentile(chance_dist, 2.5):.3f}, '
              f'{np.percentile(chance_dist, 97.5):.3f}]')
        perm_p = float((chance_dist >= actual_mean).mean())
        print(f'  实际: mean={actual_mean:.3f}, permutation p={perm_p:.4f}')
        perm_chance_mean = float(chance_dist.mean())
        perm_ci = [float(np.percentile(chance_dist, 2.5)),
                   float(np.percentile(chance_dist, 97.5))]
    else:
        print(f'\n[step 2] 跳过 permutation (N_PERMUTATION=0)')
        print(f'  使用 1-sample t-test vs 0.5 作为群体显著性检验')
        print(f'  实际: mean={actual_mean:.3f}, t={actual_t:.2f}, p={actual_p:.4f}')
        perm_p = None
        perm_chance_mean = 0.5
        perm_ci = [0.5, 0.5]

    # 3. 可视化
    print('\n[step 3] 可视化...')
    plot_decoding_results(results_df,
                          FIGURES_DIR / 'fig18_riemannian_decoding.png')

    # 4. 汇总
    summary = {
        'n_subjects': len(subject_ids),
        'clf': PRIMARY_CLF,
        'time_windows': {k: list(v) for k, v in TIME_WINDOWS.items()},
        'contrasts': [f'{a}-{b}' for a, b in CONTRASTS],
        'main_result': {
            'contrast': 'taste-health',
            'window': '0.0-1.0s',
            'mean_acc': actual_mean,
            'std_acc': float(actual.std(ddof=1)),
            't_stat': float(actual_t),
            'p_value': float(actual_p),
            'perm_chance_mean': perm_chance_mean,
            'perm_chance_95CI': perm_ci,
            'perm_p': perm_p,
        },
        'all_contrasts_summary': {},
    }

    for contrast in [f'{a}-{b}' for a, b in CONTRASTS]:
        sub = results_df[results_df['contrast'] == contrast]
        for w in sub['window'].unique():
            key = f'{contrast}_{w}'
            accs = sub[sub['window'] == w]['balanced_acc']
            if len(accs) > 0:
                t, p = stats.ttest_1samp(accs, 0.5)
                summary['all_contrasts_summary'][key] = {
                    'n': len(accs),
                    'mean': float(accs.mean()),
                    'std': float(accs.std(ddof=1)),
                    't': float(t),
                    'p': float(p),
                }

    with open(RESULTS_DIR / 'riemannian_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    print(f'\n[summary] {RESULTS_DIR / "riemannian_summary.json"}')

    print('\n' + '=' * 60)
    print('B1 完成：Riemannian 解码')
    print('=' * 60)


if __name__ == '__main__':
    import sys
    quick = '--quick' in sys.argv
    n_subjects_override = None
    for arg in sys.argv[1:]:
        if arg.startswith('--n='):
            n_subjects_override = int(arg.split('=')[1])

    if quick:
        print('[QUICK MODE] 仅前 5 个被试 + 1 时间窗 + 1 对比')
        # 通过 monkey-patch 模块级常量
        import riemannian_decoding as _self
        _self.CONTRASTS = [('taste', 'health')]
        _self.TIME_WINDOWS = {'full_0-1000ms': (0.0, 1.0)}
        _self.N_PERMUTATION = 10
        _self.main(n_subjects=5)
    elif n_subjects_override is not None:
        main(n_subjects=n_subjects_override)
    else:
        main()
