# -*- coding: utf-8 -*-
"""
svr_decoding.py
===============
基于 EEG 的连续评分回归（SVR）—— 替代官方 FoodEEG SVR 的 Python 实现。

科学目标：
    验证 EEG 神经活动能否**预测被试对每张食物图的连续评分**（0-100）
    包括：
        - rating_health      健康度评分
        - rating_taste       美味度评分
        - rating_willingnessToEat  愿意吃程度

    这是 Chae et al. (2025) bioRxiv 中官方 MVPA-SVR 的 Python 复现，
    同时也是 Bode et al. (2019) DDTBOX 的精神继承。

方法：
    - 协方差矩阵 (Ledoit-Wolf 收缩) → TangentSpace(logeuclid) → Ridge SVR
    - 关键优化：logm(X) 预计算缓存（每被试每窗口一次），
      每个 LOSO fold 只需重算 ddexpm（向量化批量矩阵乘）
    - LOSO 跨被试验证
    - 评估指标：每被试 Pearson r 和 MSE
    - 群体显著性：1-sample t-test of r vs 0

时间窗：分别对 [N1, P2, LPP, LATE, full] 几个时间窗做 SVR
"""

import sys
import os
import time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import warnings
from pathlib import Path
from scipy import stats
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

from pyriemann.estimation import Covariances

from config import FIGURES_DIR, RESULTS_DIR, PROJECT_ROOT, merge_results_csv

# ============ 常量内联（避免 import mne，其在本机 import 极慢） ============
EPOCHS_DIR = PROJECT_ROOT / 'data' / 'epochs'
QC_DIR = PROJECT_ROOT / 'outputs' / 'qc'
TMIN = -0.2   # stimulus-locked 起始 (s)
TMAX = 1.0    # stimulus-locked 结束 (s)

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

FS = 512
RANDOM_STATE = 42

# ============ 时间窗（基于 Sullivan 2015 + Kruse 2024 文献）============
TIME_WINDOWS = {
    'N1_100-200ms':     (0.100, 0.200),
    'P2_200-300ms':     (0.200, 0.300),
    'P3_300-450ms':     (0.300, 0.450),
    'LPP_400-700ms':    (0.400, 0.700),
    'LATE_700-1000ms':  (0.700, 1.000),
    'full_0-1000ms':    (0.000, 1.000),
}

# ============ 目标评分 ============
TARGETS = {
    'rating_health':           '健康度评分 (0-100)',
    'rating_taste':            '美味度评分 (0-100)',
    'rating_willingnessToEat': '愿意吃评分 (0-100)',
}

# ============ SVR pipeline ============
def make_pipeline():
    """StandardScaler + Ridge（特征 = TangentSpace(logeuclid)，见 tangent_space_logeuclid）

    特征已在 fold 内由 logm 协方差变换为切线空间向量，
    此处仅做标准化 + 岭回归。
    """
    return Pipeline([
        ('scaler', StandardScaler()),
        ('clf', Ridge(alpha=1.0, random_state=RANDOM_STATE)),
    ])


def batch_logm_sym(covs):
    """批量对称矩阵对数：S = V diag(log λ) V^T（eigh 向量化，远快于 scipy.linalg.logm）
    covs: (n, C, C) -> (n, C, C)
    """
    w, v = np.linalg.eigh(covs)
    logw = np.log(np.clip(w, 1e-12, None))
    return np.einsum('nij,nj,nkj->nik', v, logw, v)


def _first_divided_difference_exp(d):
    """exp 的一阶分差（与 pyriemann utils.base._first_divided_difference 一致）"""
    di = d[:, np.newaxis]
    dj = d[np.newaxis, :]
    close_ = np.isclose(di, dj, atol=1e-12, rtol=1e-12)
    safe_diff = np.where(close_, np.ones_like(di - dj), di - dj)
    return np.where(close_, np.exp(di), (np.exp(di) - np.exp(dj)) / safe_diff)


def tangent_space_logeuclid(logm_covs, logCref):
    """向量化 pyriemann TangentSpace(metric='logeuclid').transform 的等价实现

    pyriemann 的 log_map_logeuclid(X, Cref) 为：
        ddexpm(logm(X) - logm(Cref), logm(Cref))
    其中 ddexpm(A, B) = V @ (expfdd * (Vh @ A @ V)) @ Vh，
    {d, V} = eigh(B)，expfdd 为 exp 的一阶分差（只依赖 B）。
    因 logm(X) 可预计算、expfdd 只依赖 logCref，每 fold 仅需批量矩阵乘。

    参数
    ----
    logm_covs : (n, C, C) 预计算的协方差矩阵对数
    logCref   : (C, C) 参考点（训练集 log-euclidean 均值）的对数
                注：log-euclidean 均值 M = expm(mean(logm(X)))，
                故 logm(M) = mean(logm(X))，可直接用训练集 logm 的算术平均。
    返回
    ----
    T : (n, C*(C+1)/2)，upper 三角展平 + 离对角 sqrt(2) 权重（与 pyriemann upper() 一致）
    """
    d, V = np.linalg.eigh(logCref)
    Vh = V.T
    expfdd = _first_divided_difference_exp(d)
    X_log = logm_covs - logCref
    tmp = np.matmul(Vh, np.matmul(X_log, V))
    tmp = expfdd * tmp
    X_new = np.matmul(V, np.matmul(tmp, Vh))
    C = X_new.shape[1]
    iu = np.triu_indices(C)
    coeffs = (np.sqrt(2) * np.triu(np.ones((C, C)), 1) + np.eye(C))[iu]
    return X_new[:, iu[0], iu[1]] * coeffs


# ============ 数据加载 ============
def load_subject_epochs_with_ratings(subject_id):
    """加载附加了 ratings 的 stim epochs"""
    x_path = EPOCHS_DIR / f'{subject_id}_stim_epochs.npy'
    l_path = EPOCHS_DIR / f'{subject_id}_stim_labels_with_ratings.npz'
    if not x_path.exists() or not l_path.exists():
        raise FileNotFoundError(f'{subject_id} 缺少 epoch 或 ratings 标签')
    X = np.load(x_path)
    labels = dict(np.load(l_path, allow_pickle=True))
    return X, labels


def load_subject_list():
    """从 QC summary 加载有效被试（已被 EEG 排除的自动跳过）"""
    qc_df = pd.read_csv(QC_DIR / 'qc_summary.csv')
    valid = qc_df[qc_df['n_stim_kept'].notna()].copy()
    valid = valid[valid['n_stim_kept'] >= 30]
    return valid['subject_id'].tolist()


def get_time_slice(X, tw):
    """根据时间窗切片 epochs"""
    t_start, t_end = tw
    n_times = X.shape[2]
    t_axis = np.linspace(TMIN, TMAX, n_times)
    mask = (t_axis >= t_start) & (t_axis <= t_end)
    return X[:, :, mask]


# ============ 数据加载（模块级缓存，避免重复读盘） ============
_RAW_CACHE = {}


def load_raw_subject(subject_id):
    """加载某被试原始 epochs + labels，并缓存（只读盘一次）"""
    if subject_id not in _RAW_CACHE:
        X, labels = load_subject_epochs_with_ratings(subject_id)
        _RAW_CACHE[subject_id] = (X, labels)
    return _RAW_CACHE[subject_id]


def preload_full(subject_ids):
    """预加载所有被试完整 epochs + labels（不做 trial 过滤，供窗口共享）"""
    cache = {}
    for sid in subject_ids:
        try:
            X, labels = load_raw_subject(sid)
            cache[sid] = (X, labels)
        except Exception as e:
            print(f'  [skip preload] {sid}: {e}')
            continue
    return cache


def compute_logm_all(full_cache, time_window):
    """对一个时间窗：每被试全 trials 协方差 -> logm（一次计算，共享于 3 个 target）

    返回 {sid: (logm_covs (n_all, C, C), labels)}
    """
    cov_estimator = Covariances(estimator='lwf')
    out = {}
    for sid, (Xv, labels) in full_cache.items():
        Xw = get_time_slice(Xv, time_window)
        covs = cov_estimator.transform(Xw)
        out[sid] = (batch_logm_sym(covs), labels)
    return out


def build_target_views(logm_all, target_col, min_trials=30):
    """按 target 从全 trial logm 中选出有效行（与 y 对齐）

    返回 (logm_by_sid, y_by_sid)；不足 min_trials 的被试跳过。
    """
    logm_by_sid, y_by_sid = {}, {}
    for sid, (logm, labels) in logm_all.items():
        if target_col not in labels:
            continue
        y = labels[target_col]
        mask = ~np.isnan(y)
        if mask.sum() < min_trials:
            continue
        logm_by_sid[sid] = logm[mask]
        y_by_sid[sid] = y[mask]
    return logm_by_sid, y_by_sid


# ============ LOSO SVR ============
def run_loso_svr(subject_ids, target_col, time_window, max_trials=80,
                 logm_by_sid=None, y_by_sid=None):
    """Leave-One-Subject-Out 连续评分回归（TangentSpace(logeuclid)）

    参数
    ----
    logm_by_sid : {sid: logm 协方差 (n_trials, C, C)}（已按 target 过滤行）
    y_by_sid    : {sid: 连续评分 (n_trials,)}

    优化：logm(X) 已预计算；每个 fold 内
        参考点 logCref = 训练集 logm 算术平均（log-euclidean 均值）
        -> 向量化 ddexpm 切线空间变换 -> StandardScaler + Ridge
    返回: results (list of dict), 每被试一行
    """
    if logm_by_sid is None or y_by_sid is None:
        raise ValueError('需要预计算的 logm_by_sid / y_by_sid')
    if len(logm_by_sid) < 3:
        print(f'  [error] 有效被试 < 3')
        return []

    results = []
    n_done = 0
    for test_sid in subject_ids:
        if test_sid not in logm_by_sid:
            continue
        logm_test = logm_by_sid[test_sid]
        y_test = y_by_sid[test_sid]
        if len(y_test) < 30:
            continue

        # 训练集：其他被试（按被试抽样，与官方 LOSO 一致）
        logm_tr_list = []
        y_tr_list = []
        rng = np.random.RandomState(RANDOM_STATE + n_done)
        for train_sid in subject_ids:
            if train_sid == test_sid or train_sid not in logm_by_sid:
                continue
            logm_t = logm_by_sid[train_sid]
            y_t = y_by_sid[train_sid]
            # 平衡抽样（按 rating 中位数分高低，每被试最多 max_trials 个）
            if len(y_t) > max_trials:
                med = np.median(y_t)
                idx_high = np.where(y_t >= med)[0]
                idx_low = np.where(y_t < med)[0]
                n_each = max_trials // 2
                if len(idx_high) > n_each:
                    idx_high = rng.choice(idx_high, n_each, replace=False)
                if len(idx_low) > n_each:
                    idx_low = rng.choice(idx_low, n_each, replace=False)
                idx = np.concatenate([idx_high, idx_low])
                logm_t = logm_t[idx]
                y_t = y_t[idx]
            logm_tr_list.append(logm_t)
            y_tr_list.append(y_t)

        if not logm_tr_list:
            continue

        logm_tr = np.concatenate(logm_tr_list, axis=0)
        y_tr = np.concatenate(y_tr_list, axis=0)

        # TangentSpace(logeuclid)：参考点 = 训练集 log-euclidean 均值
        logCref = logm_tr.mean(axis=0)
        feats_tr = tangent_space_logeuclid(logm_tr, logCref)
        feats_te = tangent_space_logeuclid(logm_test, logCref)

        # SVR
        try:
            pipe = make_pipeline()
            pipe.fit(feats_tr, y_tr)
            y_pred = pipe.predict(feats_te)
            r, _ = stats.pearsonr(y_test, y_pred)
            if not np.isfinite(r):
                r = 0.0
            mse = mean_squared_error(y_test, y_pred)
            results.append({
                'subject_id': test_sid,
                'target': target_col,
                'window': f'{time_window[0]:.3f}-{time_window[1]:.3f}s',
                'n_test': len(y_test),
                'pearson_r': float(r),
                'mse': float(mse),
                'y_mean': float(y_test.mean()),
                'y_std': float(y_test.std()),
            })
        except Exception as e:
            print(f'  [error svr] {test_sid}: {e}')
            continue

        n_done += 1
        if n_done % 10 == 0:
            r_so_far = [r['pearson_r'] for r in results]
            print(f'    progress: {n_done}/{len(logm_by_sid)}, '
                  f'mean r={np.mean(r_so_far):.4f}')

    return results


# ============ 置换检验（chance level）============
def permutation_chance(subject_ids, target_col, n_perm=20,
                       logm_by_sid=None, y_by_sid=None):
    """置换检验：群体 chance level 分布（被试内打乱标签）

    训练集 = 全部被试（置换标签），参考点 = 全训练集 log-euclidean 均值；
    pipeline 只 fit 一次，逐个测试被试预测（数学等价于逐 fold fit，更快）。
    """
    if logm_by_sid is None or y_by_sid is None:
        raise ValueError('需要预计算的 logm_by_sid / y_by_sid')
    if len(logm_by_sid) < 3:
        return np.array([])

    # 预变换所有测试被试的候选（不依赖置换标签，可复用参考点变化前的 logm）
    rng = np.random.RandomState(RANDOM_STATE)
    chance_rs = []

    for p in range(n_perm):
        # 训练集 = 所有被试（特征已算好，只打乱标签）
        logm_tr_list = []
        y_tr_list = []
        for train_sid in subject_ids:
            if train_sid not in logm_by_sid:
                continue
            logm_t = logm_by_sid[train_sid]
            y_t = y_by_sid[train_sid]
            y_perm = rng.permutation(y_t)
            logm_tr_list.append(logm_t)
            y_tr_list.append(y_perm)
        logm_tr = np.concatenate(logm_tr_list, axis=0)
        y_tr = np.concatenate(y_tr_list, axis=0)

        logCref = logm_tr.mean(axis=0)
        feats_tr = tangent_space_logeuclid(logm_tr, logCref)

        # 变换各测试被试（一次），再统一 fit + predict
        test_items = []
        for test_sid in subject_ids:
            if test_sid not in logm_by_sid:
                continue
            y_test = y_by_sid[test_sid]
            if len(y_test) < 30:
                continue
            feats_te = tangent_space_logeuclid(logm_by_sid[test_sid], logCref)
            test_items.append((y_test, feats_te))
        if not test_items:
            continue

        pipe = make_pipeline()
        pipe.fit(feats_tr, y_tr)
        perms = []
        for y_test, feats_te in test_items:
            y_pred = pipe.predict(feats_te)
            r, _ = stats.pearsonr(y_test, y_pred)
            if np.isfinite(r):
                perms.append(r)
        if perms:
            chance_rs.append(np.mean(perms))
        if (p + 1) % 5 == 0:
            print(f'    permutation {p+1}/{n_perm}, '
                  f'current chance={chance_rs[-1] if chance_rs else float("nan"):.4f}')

    return np.array(chance_rs)


def permutation_chance_per_window(subject_ids, wname, tw, views_cache, n_perm=100):
    """对单一时间窗 × 全部 3 个 target 跑置换检验

    参数
    ----
    subject_ids : 有效被试 ID 列表
    wname       : 时间窗名（如 'LPP_400-700ms'）
    tw          : 时间窗 tuple (t_start, t_end)
    views_cache : {(wname, target): (logm_by_sid, y_by_sid)}，来自 main() 预计算
    n_perm      : 每格的置换次数

    返回
    ----
    list of dict: 每 target 一行 {target, actual_r, chance_mean, chance_std,
                                   chance_95ci, perm_p}
    """
    out = []
    for target_col in TARGETS.keys():
        if (wname, target_col) not in views_cache:
            continue
        logm_by_sid, y_by_sid = views_cache[(wname, target_col)]
        # 取实际 mean r（来自 LOSO 已写入的 results_df，但此处 main 已在外层算过；
        # 为解耦，函数内重算一次 LOSO 的 subject-mean r 作为 actual）
        rs_actual = []
        for sid in subject_ids:
            if sid not in logm_by_sid:
                continue
            y_test = y_by_sid[sid]
            if len(y_test) < 30:
                continue
            # 用 LOSO 一样的流程：除自己外的 logm 训练，测自己
            logm_tr_list, y_tr_list = [], []
            for tr_sid in subject_ids:
                if tr_sid == sid or tr_sid not in logm_by_sid:
                    continue
                logm_tr_list.append(logm_by_sid[tr_sid])
                y_tr_list.append(y_by_sid[tr_sid])
            if not logm_tr_list:
                continue
            logm_tr = np.concatenate(logm_tr_list, axis=0)
            y_tr = np.concatenate(y_tr_list, axis=0)
            logCref = logm_tr.mean(axis=0)
            feats_tr = tangent_space_logeuclid(logm_tr, logCref)
            feats_te = tangent_space_logeuclid(logm_by_sid[sid], logCref)
            try:
                pipe = make_pipeline()
                pipe.fit(feats_tr, y_tr)
                y_pred = pipe.predict(feats_te)
                r, _ = stats.pearsonr(y_test, y_pred)
                if np.isfinite(r):
                    rs_actual.append(r)
            except Exception:
                continue
        if not rs_actual:
            continue
        actual_mean = float(np.mean(rs_actual))

        chance_dist = permutation_chance(
            subject_ids, target_col, n_perm=n_perm,
            logm_by_sid=logm_by_sid, y_by_sid=y_by_sid,
        )
        if len(chance_dist) == 0:
            continue
        perm_p = float((chance_dist >= actual_mean).mean())
        out.append({
            'target': target_col,
            'window': wname,
            'window_tw': tw,
            'actual_r': actual_mean,
            'chance_mean': float(chance_dist.mean()),
            'chance_std': float(chance_dist.std(ddof=1)) if len(chance_dist) > 1 else 0.0,
            'chance_95ci': [float(np.percentile(chance_dist, 2.5)),
                            float(np.percentile(chance_dist, 97.5))],
            'perm_p': perm_p,
        })
        print(f'  [{wname}] {target_col}: actual r={actual_mean:.4f}, '
              f'chance={chance_dist.mean():.4f}±{chance_dist.std(ddof=1):.4f}, '
              f'perm p={perm_p:.4g}')
    return out


def apply_fdr_correction(pvals, alpha=0.05):
    """Benjamini-Hochberg FDR 校正

    参数
    ----
    pvals : 1-D array-like of p-values（长度 m）
    alpha : FDR 显著性水平（默认 0.05）

    返回
    ----
    dict: {
        'method': 'benjamini_hochberg',
        'alpha': alpha,
        'n_tests': m,
        'pvals': [...],
        'q_values': [...],   # 与 pvals 同序，q = min(p_(k) * m/k, 1)，并单调不减
        'rejected': [bool], # q < alpha
    }
    """
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    if m == 0:
        return {'method': 'benjamini_hochberg', 'alpha': alpha, 'n_tests': 0,
                'pvals': [], 'q_values': [], 'rejected': []}
    order = np.argsort(p)
    ranked = p[order]
    # BH q 值：q_(k) = p_(k) * m / k；并从尾部向上取累计最小（保单调不减）
    q_ranked = ranked * m / np.arange(1, m + 1)
    q_ranked = np.minimum.accumulate(q_ranked[::-1])[::-1]
    q_ranked = np.clip(q_ranked, 0.0, 1.0)
    # 还原原序
    q = np.empty(m)
    q[order] = q_ranked
    rejected = q < alpha
    return {
        'method': 'benjamini_hochberg',
        'alpha': float(alpha),
        'n_tests': int(m),
        'pvals': [float(x) for x in p],
        'q_values': [float(x) for x in q],
        'rejected': [bool(x) for x in rejected],
    }


# ============ 可视化 ============
def plot_svr_results(results_df, save_path):
    """每被试 r 分布 + 窗×目标热图"""
    targets = results_df['target'].unique()
    windows = results_df['window'].unique()
    n_rows = 2
    n_cols = max(2, len(targets))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 8))
    axes = np.atleast_2d(axes)

    # Row 1: 每目标 r 分布
    for c, target in enumerate(targets):
        ax = axes[0, c]
        sub = results_df[results_df['target'] == target]
        if len(sub) == 0:
            ax.set_visible(False)
            continue
        ax.hist(sub['pearson_r'], bins=15, color='steelblue',
                edgecolor='white', alpha=0.85)
        mean_r = sub['pearson_r'].mean()
        ax.axvline(mean_r, color='red', linestyle='--',
                   label=f'mean r={mean_r:.3f}')
        ax.axvline(0, color='black', linestyle=':', label='chance r=0')
        ax.set_xlabel('Pearson r')
        ax.set_ylabel('Subjects')
        ax.set_title(f'{target}\n({sub["window"].iloc[0]})', fontsize=10)
        ax.legend(fontsize=8)

    # Row 2: 窗 × 目标 热图
    pivot_r = results_df.pivot_table(index='window', columns='target',
                                      values='pearson_r', aggfunc='mean')
    if len(pivot_r) > 0:
        for c, target in enumerate(targets):
            ax = axes[1, c]
            if target not in pivot_r.columns:
                ax.set_visible(False)
                continue
            col_data = pivot_r[target]
            ax.barh(range(len(col_data)), col_data.values,
                    color='steelblue', alpha=0.85)
            ax.axvline(0, color='black', linestyle=':', linewidth=0.8)
            ax.set_yticks(range(len(col_data)))
            ax.set_yticklabels(col_data.index, fontsize=8)
            ax.set_xlabel('Mean r (LOSO)')
            ax.set_title(f'{target}: time window', fontsize=10)

    plt.suptitle('SVR decoding: predict continuous ratings from EEG', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'[fig] saved: {save_path}')


# ============ FDR 热图可视化 ============
def plot_fdr_heatmap(summary, save_path):
    """window × target FDR q 值热图（-log10(q)），matplotlib imshow，无 seaborn"""
    windows = list(TIME_WINDOWS.keys())
    targets = list(TARGETS.keys())
    n_rows = len(windows)
    n_cols = len(targets)

    if 'permutation_all' not in summary or not summary['permutation_all']:
        print('[fdr heatmap] permutation_all 缺失，跳过热图绘制')
        return

    q_matrix = np.full((n_rows, n_cols), np.nan, dtype=float)
    sig_matrix = np.zeros((n_rows, n_cols), dtype=bool)
    for r in summary['permutation_all']:
        try:
            wi = windows.index(r['window'])
            ti = targets.index(r['target'])
        except ValueError:
            continue
        qval = r.get('q_value')
        if qval is not None:
            q_matrix[wi, ti] = qval
        sig_matrix[wi, ti] = bool(r.get('fdr_significant', False))

    if np.all(np.isnan(q_matrix)):
        print('[fdr heatmap] 所有 q 值为 NaN，跳过热图绘制')
        return

    # -log10(q) 用于着色；NaN 灰显
    color_matrix = np.where(np.isnan(q_matrix), -np.inf, -np.log10(np.clip(q_matrix, 1e-10, 1.0)))
    vmax = np.nanmax(color_matrix) if not np.all(np.isinf(color_matrix)) else 3.0

    fig, ax = plt.subplots(figsize=(5, 4.5))
    cmap = plt.get_cmap('viridis_r').copy()
    cmap.set_bad(color='0.85')
    im = ax.imshow(color_matrix, aspect='auto', cmap=cmap,
                   vmin=0, vmax=min(float(vmax), 4.0))
    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label('-log10(q)', fontsize=10)

    # 网格线 + 显著性标注
    ax.set_xticks(range(n_cols))
    ax.set_yticks(range(n_rows))
    for i in range(n_rows + 1):
        ax.axhline(i - 0.5, color='white', linewidth=1.5)
    for j in range(n_cols + 1):
        ax.axvline(j - 0.5, color='white', linewidth=1.5)
    for i in range(n_rows):
        for j in range(n_cols):
            if np.isnan(q_matrix[i, j]):
                continue
            marker = '✓' if sig_matrix[i, j] else '✗'
            color = 'white' if color_matrix[i, j] > 1.5 else 'black'
            ax.text(j, i, marker, ha='center', va='center',
                    fontsize=14, color=color, fontweight='bold')

    ax.set_xticklabels(targets, fontsize=9, rotation=15, ha='right')
    ax.set_yticklabels(windows, fontsize=9)
    ax.set_xlabel('Target rating', fontsize=11)
    ax.set_ylabel('Time window', fontsize=11)
    alpha = summary.get('fdr', {}).get('alpha', 0.05)
    n_sig = int(sum(1 for r in summary['permutation_all'] if r.get('fdr_significant', False)))
    ax.set_title(f"FDR heatmap (BH, α={alpha})\n{n_sig}/{len(summary['permutation_all'])} significant",
                 fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'[fig] saved: {save_path}')


def main(n_subjects=None, quick_test=False, n_perm=20,
         perm_seeds=1, max_trials=80):
    print('=' * 60)
    print('SVR decoding: predict continuous food ratings from EEG')
    print('=' * 60)

    subject_ids = load_subject_list()
    if n_subjects:
        subject_ids = subject_ids[:n_subjects]
    if quick_test:
        subject_ids = subject_ids[:5]
    print(f'\n有效被试: {len(subject_ids)}')

    # 预加载全部被试完整数据（窗口与 target 共享）
    full_cache = preload_full(subject_ids)
    print(f'[preload] 完整数据: {len(full_cache)} 被试')

    all_results = []
    # 每窗口每 target 的 logm 视图缓存（供 LOSO + 置换检验复用）
    views_cache = {}

    for wname, tw in TIME_WINDOWS.items():
        print(f'\n=== window: {wname} ({tw[0]:.3f}-{tw[1]:.3f}s) ===')
        t0 = time.time()
        logm_all = compute_logm_all(full_cache, tw)
        print(f'  [feat] logm 计算完成 {len(logm_all)} 被试 ({time.time()-t0:.1f}s)')
        for target_col, target_label in TARGETS.items():
            logm_by_sid, y_by_sid = build_target_views(logm_all, target_col)
            views_cache[(wname, target_col)] = (logm_by_sid, y_by_sid)
            print(f'\n--- target: {target_col} ({target_label}), '
                  f'有效 {len(logm_by_sid)} 被试 ---')
            t0 = time.time()
            results = run_loso_svr(subject_ids, target_col, tw,
                                   max_trials=max_trials,
                                   logm_by_sid=logm_by_sid, y_by_sid=y_by_sid)
            print(f'  [loso] 完成 ({time.time()-t0:.1f}s)')
            for r in results:
                r['window_name'] = wname
                r['window_label'] = target_label
            all_results.extend(results)
            if results:
                rs = [r['pearson_r'] for r in results]
                t, p = stats.ttest_1samp(rs, 0)
                print(f'    群体: n={len(rs)}, mean r={np.mean(rs):.4f}±{np.std(rs, ddof=1):.4f}, '
                      f't={t:.2f}, p={p:.4g}')

    results_df = pd.DataFrame(all_results)
    svr_path = RESULTS_DIR / 'svr_decoding_results.csv'
    results_df = merge_results_csv(results_df, svr_path,
                                   key_cols=['subject_id', 'target', 'window_name'])
    results_df.to_csv(svr_path, index=False)
    print(f'\n[saved] {svr_path}')

    # 群体显著性汇总
    summary = {'per_window_per_target': []}
    for target_col in TARGETS.keys():
        for wname in TIME_WINDOWS.keys():
            sub = results_df[(results_df['target'] == target_col) &
                              (results_df['window_name'] == wname)]
            if len(sub) == 0:
                continue
            rs = sub['pearson_r'].values
            t, p = stats.ttest_1samp(rs, 0)
            summary['per_window_per_target'].append({
                'target': target_col,
                'window': wname,
                'n': len(rs),
                'mean_r': float(np.mean(rs)),
                'std_r': float(np.std(rs, ddof=1)),
                't': float(t),
                'p': float(p),
                'cohen_d': float(np.mean(rs) / np.std(rs, ddof=1)) if np.std(rs, ddof=1) > 0 else 0.0,
            })

    # 置换检验：全窗 × 全目标（LPP 仅作核心目标窗保留原打印行）
    print('\n=== Permutation chance level (核心目标在 LPP 400-700ms 窗) ===')
    perm_summary = {}
    for target_col in TARGETS.keys():
        sub = results_df[(results_df['target'] == target_col) &
                          (results_df['window_name'] == 'LPP_400-700ms')]
        if len(sub) == 0:
            continue
        actual_mean = sub['pearson_r'].mean()
        logm_by_sid, y_by_sid = views_cache[('LPP_400-700ms', target_col)]
        chance_dist = permutation_chance(
            subject_ids, target_col, n_perm=n_perm,
            logm_by_sid=logm_by_sid, y_by_sid=y_by_sid
        )
        if len(chance_dist) == 0:
            continue
        perm_p = float((chance_dist >= actual_mean).mean())
        perm_summary[target_col] = {
            'actual_mean_r': float(actual_mean),
            'chance_mean_r': float(chance_dist.mean()),
            'chance_95ci': [float(np.percentile(chance_dist, 2.5)),
                             float(np.percentile(chance_dist, 97.5))],
            'perm_p': perm_p,
        }
        print(f'  {target_col}: actual r={actual_mean:.4f}, '
              f'chance mean={chance_dist.mean():.4f}±{chance_dist.std(ddof=1):.4f}, '
              f'perm p={perm_p:.4g}')

    summary['permutation'] = perm_summary

    # 全窗 × 全目标 置换 + FDR 校正
    print('\n=== Permutation chance level (全窗 × 全目标, n_perm={}) ==='.format(n_perm))
    perm_all = []
    for wname, tw in TIME_WINDOWS.items():
        if (wname, next(iter(TARGETS))) not in views_cache:
            continue
        perms = permutation_chance_per_window(
            subject_ids, wname, tw, views_cache, n_perm=n_perm,
        )
        perm_all.extend(perms)

    # FDR 校正（对全部 perm_p 做 BH）
    if perm_all:
        pvals = [r['perm_p'] for r in perm_all]
        fdr = apply_fdr_correction(pvals, alpha=0.05)
        # 把 q 值与 rejected 拼回每条记录
        for r, q, rej in zip(perm_all, fdr['q_values'], fdr['rejected']):
            r['q_value'] = q
            r['fdr_significant'] = rej
        summary['permutation_all'] = perm_all
        summary['fdr'] = {
            'method': fdr['method'],
            'alpha': fdr['alpha'],
            'n_tests': fdr['n_tests'],
            'n_significant': int(sum(fdr['rejected'])),
            'results': [
                {
                    'target': r['target'],
                    'window': r['window'],
                    'perm_p': r['perm_p'],
                    'q_value': r['q_value'],
                    'significant': r['fdr_significant'],
                }
                for r in perm_all
            ],
        }
        n_sig = sum(1 for r in perm_all if r['fdr_significant'])
        print(f'\n[FDR] {n_sig}/{len(perm_all)} 个窗×目标组合在 BH-FDR q<0.05 下显著')
        for r in sorted(perm_all, key=lambda x: x['q_value']):
            sig = '✓' if r['fdr_significant'] else '✗'
            print(f"  {sig} {r['window']:18s} {r['target']:30s} "
                  f"r={r['actual_r']:+.4f}  perm_p={r['perm_p']:.4g}  "
                  f"q={r['q_value']:.4g}")

    # ===== 多 seed 置换（v2 口径，对齐 svr_roi_channels 审计方案）=====
    # 仅当 --perm-seeds>1 时激活；结果写独立文件，不覆盖 v1 产物
    # （svr_decoding_results.csv / svr_summary.json 的既有键保持不变）。
    if perm_seeds > 1:
        # 函数体在独立模块 svr_multiseed_perm.py（防 IDE 旧缓冲区回滚），
        # 依赖注入避免循环导入
        from svr_multiseed_perm import _perm_cell_multi_seed
        seed_pool = [42, 123, 456, 789, 2024][:perm_seeds]
        print(f'\n=== 多 seed 置换: {n_perm} 次 × {perm_seeds} seeds × '
              f'{len(views_cache)} 格（池级预算 max_trials={max_trials}）===')
        ms_all = []
        for wname in TIME_WINDOWS.keys():
            for target_col in TARGETS.keys():
                if (wname, target_col) not in views_cache:
                    continue
                logm_by_sid, y_by_sid = views_cache[(wname, target_col)]
                t0 = time.time()
                res = _perm_cell_multi_seed(
                    logm_by_sid, y_by_sid, subject_ids,
                    n_perm=n_perm, seeds=seed_pool, max_trials=max_trials,
                    tangent_space_logeuclid=tangent_space_logeuclid,
                    make_pipeline=make_pipeline)
                if res is None:
                    continue
                res['window'] = wname
                res['target'] = target_col
                ms_all.append(res)
                print(f"  {wname:18s} {target_col:30s} "
                      f"actual={res['actual_r_mean']:+.4f}±{res['actual_r_std']:.4f} "
                      f"perm_p={res['perm_p_mean']:.3f}±{res['perm_p_std']:.3f} "
                      f"[{res['perm_p_min']:.2f},{res['perm_p_max']:.2f}] "
                      f"({time.time()-t0:.0f}s)")
        if ms_all:
            # D1 决议：FDR 用跨 seed 平均 perm_p（与 ROI v2 一致）
            fdr_ms = apply_fdr_correction(
                [r['perm_p_mean'] for r in ms_all], alpha=0.05)
            for r, q, rej in zip(ms_all, fdr_ms['q_values'],
                                 fdr_ms['rejected']):
                r['q_value'] = float(q)
                r['fdr_significant'] = bool(rej)
            summary['permutation_multiseed'] = ms_all
            summary['fdr_multiseed'] = {
                'method': fdr_ms['method'],
                'alpha': fdr_ms['alpha'],
                'n_tests': fdr_ms['n_tests'],
                'n_significant': int(sum(fdr_ms['rejected'])),
                'perm_p_rule': 'perm_p_mean（跨 seed 平均，D1 决议 2026-09-04）',
                'permutation': 'pooled-label-shuffle（合并池打乱，与 ROI v2 一致）',
                'budget_semantics': f'池级：合并训练池整体平衡抽到 {max_trials}',
                'seeds': seed_pool,
                'n_perm_per_seed': n_perm,
            }
            ms_df = pd.DataFrame([{
                'window': r['window'], 'target': r['target'],
                'n_subjects': r['n_subjects'], 'n_trials': r['n_trials'],
                'actual_r_mean': r['actual_r_mean'],
                'actual_r_std': r['actual_r_std'],
                'converged_sd_le_0p006': bool(r['actual_r_std'] <= 0.006),
                'perm_p_mean': r['perm_p_mean'],
                'perm_p_std': r['perm_p_std'],
                'perm_p_min': r['perm_p_min'],
                'perm_p_max': r['perm_p_max'],
                'chance_mean': float(np.mean(
                    [d['chance_mean'] for d in r['per_seed']])),
                'q_value': r['q_value'],
                'fdr_significant': r['fdr_significant'],
            } for r in ms_all])
            ms_csv = RESULTS_DIR / 'svr_decoding_multiseed.csv'
            ms_df.to_csv(ms_csv, index=False)
            print(f'\n[saved] {ms_csv}')
            n_sig_ms = sum(r['fdr_significant'] for r in ms_all)
            print(f'[FDR-多seed] {n_sig_ms}/{len(ms_all)} 格 BH q<0.05 显著')
            ms_json = RESULTS_DIR / 'svr_decoding_multiseed.json'
            with open(ms_json, 'w', encoding='utf-8') as f:
                json.dump({'cells': ms_all, 'fdr': summary['fdr_multiseed']},
                          f, indent=2, ensure_ascii=False, default=str)
            print(f'[saved] {ms_json}')

    with open(RESULTS_DIR / 'svr_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    print(f'\n[summary] {RESULTS_DIR / "svr_summary.json"}')

    # 可视化
    print('\n[step 3] 可视化...')
    plot_svr_results(results_df, FIGURES_DIR / 'fig23_svr_decoding.png')
    plot_fdr_heatmap(summary, FIGURES_DIR / 'fig27_svr_fdr_heatmap.png')

    # 报告最强结果：按 FDR q 值排序，显著项标 ✓
    print('\n' + '=' * 60)
    print('最强 SVR 结果 (按 FDR q 值排序, BH-α=0.05):')
    print('=' * 60)

    # 构建 (per_window_per_target, perm_p, q_value, fdr_sig) 的索引
    perm_lookup = {}
    if 'permutation_all' in summary:
        for r in summary['permutation_all']:
            perm_lookup[(r['window'], r['target'])] = r
    q_lookup = {}
    if 'fdr' in summary:
        for fr in summary['fdr']['results']:
            q_lookup[(fr['window'], fr['target'])] = fr

    enriched = []
    for r in summary['per_window_per_target']:
        key = (r['window'], r['target'])
        pr = perm_lookup.get(key, {})
        fr = q_lookup.get(key, {})
        enriched.append({
            **r,
            'perm_p': pr.get('perm_p'),
            'q_value': fr.get('q_value'),
            'fdr_significant': fr.get('significant', False),
        })

    # 排序：FDR 显著的优先（按 q 值升序），其余按 mean_r 降序
    enriched.sort(key=lambda x: (
        not x['fdr_significant'],  # True (显著) 在前
        x['q_value'] if x['q_value'] is not None else 1.0,
        -x['mean_r'],
    ))
    for r in enriched[:10]:
        sig = '***' if r['p'] < 0.001 else ('**' if r['p'] < 0.01
              else ('*' if r['p'] < 0.05 else 'ns'))
        q_str = f"  q={r['q_value']:.3g}" if r['q_value'] is not None else "  q=NA"
        fdr_mark = ' ✓FDR' if r['fdr_significant'] else ''
        print(f"  {r['target']:30s} {r['window']:20s} "
              f"r={r['mean_r']:+.4f}  t={r['t']:+.2f}  p={r['p']:.4g} {sig}"
              f"{q_str}{fdr_mark}")


if __name__ == '__main__':
    import sys
    quick = '--quick' in sys.argv
    n = None
    n_perm = 100
    perm_seeds = 1     # 默认 1 = v1 行为（向后兼容，run_all.sh 不受影响）
    max_trials = 80    # 默认 80 = v1 行为；v2 多种子建议 200
    for arg in sys.argv[1:]:
        if arg.startswith('--n='):
            n = int(arg.split('=')[1])
        if arg.startswith('--perm='):
            n_perm = int(arg.split('=')[1])
        if arg.startswith('--perm-seeds='):
            perm_seeds = int(arg.split('=')[1])
        if arg.startswith('--max-trials='):
            max_trials = int(arg.split('=')[1])
    main(n_subjects=n, quick_test=quick, n_perm=n_perm,
         perm_seeds=perm_seeds, max_trials=max_trials)