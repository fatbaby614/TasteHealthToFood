# -*- coding: utf-8 -*-
"""
svr_baseline_compare.py
=======================
Riemannian vs conventional EEG 特征表示的连续评分解码对比实验。

在**同一 LOSO + permutation 协议**下对比 4 种特征（回答 CMPB 审稿人
"为什么采用 Riemannian tangent-space representation"）：
  1. riemannian    : Ledoit-Wolf 协方差 → log-Euclidean 切空间 → Ridge（论文主方法）
  2. euclidean_cov : Ledoit-Wolf 协方差 → logm 上三角展平（不投影切空间）→ Ridge
  3. raw_flat      : 时间窗原始样本降采样展平 → Ridge
  4. bandpower     : 频带功率（theta/alpha/beta/gamma）→ Ridge

协议与论文一致（见 paper/main.tex sec:stats）：
- LOSO 逐被试 Pearson r → 群体 mean r + 单样本 t 检验
- 置换检验在 subject 边界固定下打乱 trial 级评分（exchangeability null）
- 多随机种子取 mean±SD（默认 seeds 42,123,456）

用法：
  python code/svr_baseline_compare.py --quick                 # 冒烟：full 窗 × 3 targets × 10 perm
  python code/svr_baseline_compare.py --features=riemannian,euclidean_cov
  python code/svr_baseline_compare.py --seeds=42,123,456 --perm=100
输出：
  outputs/results/svr_baseline_results.csv
  outputs/results/svr_baseline_summary.json
"""

import sys
import os
import time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import json
import argparse
from scipy import stats
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

from pyriemann.estimation import Covariances

from config import RESULTS_DIR
from svr_decoding import (load_subject_list, preload_full, get_time_slice,
                          TIME_WINDOWS, TARGETS, FS, RANDOM_STATE,
                          tangent_space_logeuclid, batch_logm_sym,
                          apply_fdr_correction)

# ============ 特征名与时间窗（与论文一致）============
FEATURES = ['riemannian', 'euclidean_cov', 'raw_flat', 'bandpower']

# 频带定义（Hz）
BANDS = {'theta': (4, 8), 'alpha': (8, 13), 'beta': (13, 30), 'gamma': (30, 45)}

# raw_flat 降采样步长（512 Hz -> 128 Hz 等效）
RAW_DOWNSAMPLE = 4


def make_pipeline():
    """StandardScaler + Ridge（与 svr_decoding.make_pipeline 一致）"""
    return Pipeline([
        ('scaler', StandardScaler()),
        ('clf', Ridge(alpha=1.0, random_state=RANDOM_STATE)),
    ])


# ============ 特征提取 ============
def _cov_logm_for_sid(Xw):
    """Ledoit-Wolf 协方差 + logm（每被试每窗一次，供 riemannian / euclidean_cov 共享）"""
    cov_estimator = Covariances(estimator='lwf')
    covs = cov_estimator.transform(Xw)          # (n, C, C)
    logm = batch_logm_sym(covs)                 # (n, C, C)
    return covs, logm


def _upper_tri_vectorize(logm):
    """logm 协方差上三角展平（不含 sqrt2 加权，Ridge 对尺度鲁棒）"""
    C = logm.shape[1]
    iu = np.triu_indices(C)
    return logm[:, iu[0], iu[1]]                # (n, C*(C+1)/2)


def _raw_flat(Xw):
    """原始样本降采样展平"""
    return Xw[:, :, ::RAW_DOWNSAMPLE].reshape(Xw.shape[0], -1)


def _bandpower(Xw):
    """频带功率（FFT 幅度平方在各频带的均值），(n, n_bands*n_ch)"""
    n, C, T = Xw.shape
    Xf = np.fft.rfft(Xw, axis=-1)               # (n, C, T//2+1)
    freqs = np.fft.rfftfreq(T, 1.0 / FS)
    out = np.empty((n, len(BANDS) * C), dtype=np.float64)
    for b, (lo, hi) in enumerate(BANDS.values()):
        mask = (freqs >= lo) & (freqs < hi)
        if mask.sum() == 0:
            out[:, b * C:(b + 1) * C] = 0.0
            continue
        band_pow = np.abs(Xf[:, :, mask]) ** 2   # (n, C, n_freq_in_band)
        out[:, b * C:(b + 1) * C] = band_pow.mean(axis=-1)
    return out


def build_feat_cache(full_cache, time_window, feature):
    """对某时间窗、某特征：预计算每被试特征矩阵

    返回 {sid: (feats (n_all, d), logm (n_all, C, C) or None, labels)}
    logm 仅 riemannian 需要（fold 内 Cref）；euclidean_cov 也基于 logm 但不需要 Cref。
    """
    out = {}
    for sid, (Xv, labels) in full_cache.items():
        Xw = get_time_slice(Xv, time_window)
        n = Xw.shape[0]
        if feature == 'riemannian':
            _, logm = _cov_logm_for_sid(Xw)
            # 特征在 fold 内由 tangent_space_logeuclid 计算；此处只存 logm
            out[sid] = (None, logm, labels)
        elif feature == 'euclidean_cov':
            _, logm = _cov_logm_for_sid(Xw)
            out[sid] = (_upper_tri_vectorize(logm), None, labels)
        elif feature == 'raw_flat':
            out[sid] = (_raw_flat(Xw), None, labels)
        elif feature == 'bandpower':
            out[sid] = (_bandpower(Xw), None, labels)
        else:
            raise ValueError(f'未知特征: {feature}')
    return out


def _build_balanced_train(logm_or_feat_tr, y_tr, max_trials, rng):
    """按 rating 中位数高低平衡抽样（与 svr_decoding 一致）"""
    if len(y_tr) <= max_trials:
        return logm_or_feat_tr, y_tr
    med = np.median(y_tr)
    idx_high = np.where(y_tr >= med)[0]
    idx_low = np.where(y_tr < med)[0]
    n_each = max_trials // 2
    if len(idx_high) > n_each:
        idx_high = rng.choice(idx_high, n_each, replace=False)
    if len(idx_low) > n_each:
        idx_low = rng.choice(idx_low, n_each, replace=False)
    idx = np.concatenate([idx_high, idx_low])
    return logm_or_feat_tr[idx], y_tr[idx]


# ============ LOSO ============
def run_loso(subject_ids, feat_cache, feature, target_col, max_trials=80, seed=42):
    """Leave-One-Subject-Out：逐被试 Pearson r"""
    results = []
    for i, test_sid in enumerate(subject_ids):
        if test_sid not in feat_cache:
            continue
        feats_or_logm_test, logm_test, labels = feat_cache[test_sid]
        if target_col not in labels:
            continue
        y_test = labels[target_col]
        mask = ~np.isnan(y_test)
        if mask.sum() < 30:
            continue
        y_test = y_test[mask]
        feats_test = feats_or_logm_test[mask] if feats_or_logm_test is not None else logm_test[mask]

        # 训练集（其他被试）
        rng = np.random.RandomState(seed + i)
        tr_list, y_list = [], []
        for tr_sid in subject_ids:
            if tr_sid == test_sid or tr_sid not in feat_cache:
                continue
            t_feats, t_logm, t_labels = feat_cache[tr_sid]
            if target_col not in t_labels:
                continue
            t_y = t_labels[target_col]
            t_mask = ~np.isnan(t_y)
            t_y = t_y[t_mask]
            if t_y.sum() == 0 and len(t_y) == 0:
                continue
            t_feat = t_feats[t_mask] if t_feats is not None else t_logm[t_mask]
            t_feat, t_y = _build_balanced_train(t_feat, t_y, max_trials, rng)
            tr_list.append(t_feat)
            y_list.append(t_y)
        if not tr_list:
            continue

        # riemannian：fold 内 Cref -> 切空间
        if feature == 'riemannian':
            logm_tr = np.concatenate(tr_list, axis=0)
            y_tr = np.concatenate(y_list, axis=0)
            logCref = logm_tr.mean(axis=0)
            feats_tr = tangent_space_logeuclid(logm_tr, logCref)
            feats_te = tangent_space_logeuclid(feats_test, logCref)
        else:
            feats_tr = np.concatenate(tr_list, axis=0)
            y_tr = np.concatenate(y_list, axis=0)
            feats_te = feats_test

        try:
            pipe = make_pipeline()
            pipe.fit(feats_tr, y_tr)
            y_pred = pipe.predict(feats_te)
            r, _ = stats.pearsonr(y_test, y_pred)
            if not np.isfinite(r):
                r = 0.0
            results.append({'subject_id': test_sid, 'pearson_r': float(r),
                            'n_test': int(len(y_test))})
        except Exception as e:
            print(f'  [error loso] {test_sid}: {e}')
            continue
    return results


# ============ 置换检验（subject 边界固定、trial 级打乱）============
def permutation_generic(subject_ids, feat_cache, feature, target_col,
                        n_perm=100, seed=42, max_trials=80):
    """单种子置换：返回 chance mean-r 分布（n_perm,）"""
    # 预变换：riemannian 用全训练集 logm 均值作 Cref（置换只动 y，Cref 不变）
    rng = np.random.RandomState(seed)
    chance_rs = []

    for p in range(n_perm):
        tr_list, y_list = [], []
        for tr_sid in subject_ids:
            if tr_sid not in feat_cache:
                continue
            t_feats, t_logm, t_labels = feat_cache[tr_sid]
            if target_col not in t_labels:
                continue
            t_y = t_labels[target_col]
            t_mask = ~np.isnan(t_y)
            t_y = t_y[t_mask]
            if len(t_y) < 5:
                continue
            t_feat = t_feats[t_mask] if t_feats is not None else t_logm[t_mask]
            t_feat, t_y = _build_balanced_train(t_feat, t_y, max_trials, rng)
            y_perm = rng.permutation(t_y)
            tr_list.append(t_feat)
            y_list.append(y_perm)
        if not tr_list:
            return np.array([])

        if feature == 'riemannian':
            logm_tr = np.concatenate(tr_list, axis=0)
            logCref = logm_tr.mean(axis=0)
            feats_tr = tangent_space_logeuclid(logm_tr, logCref)
            # 测试被试特征
            test_items = []
            for te_sid in subject_ids:
                if te_sid not in feat_cache:
                    continue
                _, t_logm, t_labels = feat_cache[te_sid]
                if target_col not in t_labels:
                    continue
                t_y = t_labels[target_col]
                t_mask = ~np.isnan(t_y)
                t_y = t_y[t_mask]
                if len(t_y) < 30:
                    continue
                feats_te = tangent_space_logeuclid(t_logm[t_mask], logCref)
                test_items.append((t_y, feats_te))
        else:
            feats_tr = np.concatenate(tr_list, axis=0)
            test_items = []
            for te_sid in subject_ids:
                if te_sid not in feat_cache:
                    continue
                t_feats, _, t_labels = feat_cache[te_sid]
                if target_col not in t_labels:
                    continue
                t_y = t_labels[target_col]
                t_mask = ~np.isnan(t_y)
                t_y = t_y[t_mask]
                if len(t_y) < 30:
                    continue
                test_items.append((t_y, t_feats[t_mask]))

        if not test_items:
            return np.array([])

        pipe = make_pipeline()
        pipe.fit(feats_tr, y_list and np.concatenate(y_list, axis=0))
        perms = []
        for y_test, feats_te in test_items:
            y_pred = pipe.predict(feats_te)
            r, _ = stats.pearsonr(y_test, y_pred)
            if np.isfinite(r):
                perms.append(r)
        if perms:
            chance_rs.append(np.mean(perms))
        if (p + 1) % 10 == 0:
            print(f'      perm {p+1}/{n_perm} chance={np.mean(chance_rs):+.4f}')

    return np.array(chance_rs)


# ============ 单 cell 完整评估 ============
def evaluate_cell(subject_ids, feat_cache, feature, target_col, wname,
                  n_perm=100, seeds=(42,), max_trials=80):
    """对一个 特征×窗口×目标 cell：LOSO + 多种子置换"""
    t0 = time.time()
    rs = run_loso(subject_ids, feat_cache, feature, target_col,
                  max_trials=max_trials, seed=seeds[0])
    loso_time = time.time() - t0
    if not rs:
        return None
    r_arr = np.array([x['pearson_r'] for x in rs])
    mean_r = float(r_arr.mean())
    std_r = float(r_arr.std(ddof=1)) if len(r_arr) > 1 else 0.0
    t_stat, p_param = stats.ttest_1samp(r_arr, 0)
    cohen_d = float(mean_r / std_r) if std_r > 0 else 0.0

    # 多种子置换
    perm_p_list = []
    perm_t0 = time.time()
    for seed in seeds:
        dist = permutation_generic(subject_ids, feat_cache, feature, target_col,
                                   n_perm=n_perm, seed=seed, max_trials=max_trials)
        if len(dist) == 0:
            continue
        perm_p_list.append(float((dist >= mean_r).mean()))
    perm_time = time.time() - perm_t0

    if not perm_p_list:
        perm_mean, perm_sd, perm_range = None, None, None
    else:
        perm_mean = float(np.mean(perm_p_list))
        perm_sd = float(np.std(perm_p_list, ddof=1)) if len(perm_p_list) > 1 else 0.0
        perm_range = [float(min(perm_p_list)), float(max(perm_p_list))]

    return {
        'feature': feature,
        'window': wname,
        'target': target_col,
        'n_subjects': len(r_arr),
        'mean_r': mean_r,
        'std_r': std_r,
        't': float(t_stat),
        'p_parametric': float(p_param),
        'cohen_d': cohen_d,
        'perm_p_mean': perm_mean,
        'perm_p_sd': perm_sd,
        'perm_p_range': perm_range,
        'loso_time_s': round(loso_time, 1),
        'perm_time_s': round(perm_time, 1),
    }


def main(features, windows, targets, n_perm, seeds, n_subjects, quick):
    print('=' * 60)
    print('SVR baseline: Riemannian vs conventional EEG representations')
    print('=' * 60)
    print(f'特征: {features}')
    print(f'窗口: {windows}')
    print(f'目标: {targets}')
    print(f'置换: {n_perm}/seed, seeds={seeds}')

    subject_ids = load_subject_list()
    if n_subjects:
        subject_ids = subject_ids[:n_subjects]
    if quick:
        subject_ids = subject_ids[:8]
    print(f'有效被试: {len(subject_ids)}')

    full_cache = preload_full(subject_ids)
    print(f'[preload] {len(full_cache)} 被试')

    all_rows = []
    for wname in windows:
        tw = TIME_WINDOWS[wname]
        print(f'\n=== window: {wname} ({tw[0]:.3f}-{tw[1]:.3f}s) ===')
        # 各特征共享一次原始数据加载；每特征各自预计算
        for feature in features:
            print(f'\n--- feature: {feature} ---')
            feat_cache = build_feat_cache(full_cache, tw, feature)
            for target_col in targets:
                print(f'  target: {target_col}')
                row = evaluate_cell(subject_ids, feat_cache, feature,
                                    target_col, wname, n_perm=n_perm, seeds=seeds)
                if row is None:
                    print(f'    [skip] 无有效被试')
                    continue
                all_rows.append(row)
                print(f'    mean_r={row["mean_r"]:+.4f}±{row["std_r"]:.4f} '
                      f't={row["t"]:+.2f} p_param={row["p_parametric"]:.3g} '
                      f'perm_p={row["perm_p_mean"]}±{row["perm_p_sd"]} '
                      f'[{row["loso_time_s"]}+{row["perm_time_s"]}s]')

    df = pd.DataFrame(all_rows)
    csv_path = RESULTS_DIR / 'svr_baseline_results.csv'
    df.to_csv(csv_path, index=False)
    print(f'\n[saved] {csv_path}')

    # FDR 校正（对每组特征各自校正其窗×目标 perm_p）
    summary = {'results': df.to_dict(orient='records'), 'per_feature_fdr': {}}
    for feature in features:
        sub = df[df['feature'] == feature]
        if len(sub) == 0:
            continue
        pvals = sub['perm_p_mean'].fillna(1.0).values
        fdr = apply_fdr_correction(pvals, alpha=0.05)
        summary['per_feature_fdr'][feature] = {
            'n_tests': fdr['n_tests'],
            'n_significant': int(sum(fdr['rejected'])),
            'cells': [
                {'window': r['window'], 'target': r['target'],
                 'perm_p': r['perm_p_mean'], 'q': q, 'significant': rej}
                for r, q, rej in zip(sub.to_dict('records'), fdr['q_values'], fdr['rejected'])
            ],
        }
        print(f'\n[FDR] {feature}: {int(sum(fdr["rejected"]))}/{len(pvals)} cells significant')

    with open(RESULTS_DIR / 'svr_baseline_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    print(f'\n[summary] {RESULTS_DIR / "svr_baseline_summary.json"}')

    # 汇总打印：按窗口 × 目标，4 特征并排 mean_r
    print('\n' + '=' * 60)
    print('对比摘要 (mean r LOSO)')
    print('=' * 60)
    for wname in windows:
        for target_col in targets:
            line = f'{wname:20s} {target_col:26s}'
            for feature in features:
                sub = df[(df['window'] == wname) & (df['target'] == target_col)
                         & (df['feature'] == feature)]
                if len(sub):
                    r = sub.iloc[0]
                    line += f' | {feature[:6]}: {r["mean_r"]:+.3f}'
            print(line)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true', help='冒烟：full 窗 × 3 targets × 10 perm')
    parser.add_argument('--features', default=','.join(FEATURES))
    parser.add_argument('--windows', default=None)
    parser.add_argument('--targets', default=None)
    parser.add_argument('--perm', type=int, default=100)
    parser.add_argument('--seeds', default='42,123,456')
    parser.add_argument('--n', type=int, default=None)
    args = parser.parse_args()

    feats = [f.strip() for f in args.features.split(',') if f.strip()]
    wins = ([w.strip() for w in args.windows.split(',')] if args.windows
            else (['full_0-1000ms'] if args.quick else list(TIME_WINDOWS.keys())))
    tgts = ([t.strip() for t in args.targets.split(',')] if args.targets
            else list(TARGETS.keys()))
    seeds = [int(s) for s in args.seeds.split(',') if s.strip()]
    n_perm = 10 if args.quick else args.perm

    main(features=feats, windows=wins, targets=tgts, n_perm=n_perm,
         seeds=seeds, n_subjects=args.n, quick=args.quick)
