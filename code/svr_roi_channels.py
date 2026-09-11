# -*- coding: utf-8 -*-
"""
svr_roi_channels.py
===================
B1：SVR 通道 ROI 子集解码
=========================

目的
----
全通道 SVR 已证明 brain 可编码 continuous taste/health/willingnessToEat 评分。
现在问：**信号在哪类电极最强？** 把 64 通道分为若干 ROI 子集，每个子集做 LOSO SVR。

ROI 分组（BioSemi 64 标准 montage）
------------------------------------
- frontal       Fp1 Fp2 AFz AF3 AF4 AF7 AF8 Fz F1 F2 F3 F4 F5 F6 F7 F8 FCz FC1 FC2 FC3 FC4 FC5 FC6 FT7 FT8
- fronto-central Fz FCz FC1 FC2 FC3 FC4 FC5 FC6 (与上面重叠一部分)
- central       Fz FCz Cz C1 C2 C3 C4 C5 C6
- centro-parietal FCz Cz CPz CP1 CP2 CP3 CP4 CP5 CP6 P1 P2
- parietal      Pz P1 P2 P3 P4 P5 P6 P7 P8 POz PO3 PO4 PO5 PO6 PO7 PO8
- parieto-occipital POz PO3 PO4 PO5 PO6 PO7 PO8 Oz O1 O2
- occipital     Oz O1 O2 Iz
- temporal      FT7 FT8 T7 T8 TP7 TP8 TP9 TP10
- all (基线)    全部 64

为减少特征维数 & 节省时间，固定使用 full_0-1000ms 窗（全量最显著窗）
+ 3 个 target（taste / health / willingnessToEat）。

预计耗时（106 被试，全 9 ROI × 3 target × 1 窗）：
- 不含置换：约 3 小时
- 含 --perm=100（27 格置换）：约 12 小时
（每格 LOSO 单样本 t-test；置换复用 logm，不再重新切线空间变换）
"""
from __future__ import annotations
import sys
import os
import time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

# 复用 svr_decoding.py 已验证的最优 pipeline
from svr_decoding import (
    load_subject_list, load_raw_subject, preload_full,
    compute_logm_all, build_target_views,
    batch_logm_sym, tangent_space_logeuclid, make_pipeline,
    permutation_chance, apply_fdr_correction,
    TARGETS,
)
from config import FIGURES_DIR, RESULTS_DIR

# ============ BioSemi 64 真实通道顺序（从 eeg_preprocessing.py 直接复制）============
# 内联避免 import mne 慢（eeg_preprocessing.py import mne 需数分钟）
# 与 EEG_CHANNEL_NAMES 保持完全一致；若预处理顺序变动需同步更新
ALL_CHANNELS = [
    'Fp1', 'AF7', 'AF3', 'F1', 'F3', 'F5', 'F7', 'FT7', 'FC5', 'FC3', 'FC1',
    'C1', 'C3', 'C5', 'T7', 'TP7', 'CP5', 'CP3', 'CP1', 'P1', 'P3', 'P5', 'P7',
    'P9', 'PO7', 'PO3', 'O1', 'Iz', 'Oz', 'POz', 'Pz', 'CPz', 'Fpz', 'Fp2',
    'AF8', 'AF4', 'AFz', 'Fz', 'F2', 'F4', 'F6', 'F8', 'FT8', 'FC6', 'FC4',
    'FC2', 'FCz', 'Cz', 'C2', 'C4', 'C6', 'T8', 'TP8', 'CP6', 'CP4', 'CP2',
    'P2', 'P4', 'P6', 'P8', 'P10', 'PO8', 'PO4', 'O2'
]

# ============ ROI 分组（基于标准 10-20 命名）============
ROI_CHANNELS = {
    'frontal':         ['Fp1', 'Fp2', 'AFz', 'AF3', 'AF4', 'AF7', 'AF8',
                        'Fz', 'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8'],
    'fronto-central':  ['Fz', 'FCz', 'FC1', 'FC2', 'FC3', 'FC4', 'FC5', 'FC6'],
    'central':         ['Fz', 'FCz', 'Cz', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6'],
    'centro-parietal': ['FCz', 'Cz', 'CPz', 'CP1', 'CP2', 'CP3', 'CP4', 'CP5', 'CP6',
                        'P1', 'P2'],
    'parietal':        ['CPz', 'Pz', 'P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8',
                        'P9', 'P10', 'POz', 'PO3', 'PO4', 'PO7', 'PO8'],
    'occipital':       ['POz', 'PO3', 'PO4', 'PO7', 'PO8', 'Oz', 'O1', 'O2', 'Iz'],
    'temporal':        ['FT7', 'FT8', 'T7', 'T8', 'TP7', 'TP8'],
}

TIME_WINDOW = (0.0, 1.0)  # 全量已证明 full_0-1000ms 最强
WINDOW_NAME = 'full_0-1000ms'


def detect_channel_names(subject_ids):
    """直接返回 ALL_CHANNELS 内联列表（与 EEG_CHANNEL_NAMES 同步）
    epoch npy 不含通道名；这里用预处理脚本的固定通道顺序假设。
    """
    return list(ALL_CHANNELS)


def get_roi_indices(ch_names, roi_channels):
    """返回 roi_channels 在 ch_names 中的索引（保留顺序）"""
    name_to_idx = {n: i for i, n in enumerate(ch_names)}
    idx = []
    missing = []
    for n in roi_channels:
        if n in name_to_idx:
            idx.append(name_to_idx[n])
        else:
            missing.append(n)
    if missing:
        print(f'    [warn] {len(missing)} 个通道在数据中找不到: {missing[:5]}...')
    return idx, missing


def loso_svr_roi(logm_by_sid, y_by_sid, subject_ids, logm_all_keys=None,
                 max_trials=200, random_state=42):
    """LOSO SVR（同 svr_decoding.py，但要求 logm_by_sid 已经按 ROI 通道算好）

    输入 logm_by_sid: {sid: (n_trials, C, C)} —— 由调用方按 ROI 切片
    """
    results = []
    n_done = 0
    for test_sid in subject_ids:
        if test_sid not in logm_by_sid:
            continue
        logm_test = logm_by_sid[test_sid]
        y_test = y_by_sid[test_sid]
        if len(y_test) < 30:
            continue

        logm_tr_list, y_tr_list = [], []
        rng = np.random.RandomState(random_state + n_done)
        for train_sid in subject_ids:
            if train_sid == test_sid or train_sid not in logm_by_sid:
                continue
            logm_t = logm_by_sid[train_sid]
            y_t = y_by_sid[train_sid]
            if len(y_t) > max_trials:
                med = np.median(y_t)
                idx_h = np.where(y_t >= med)[0]
                idx_l = np.where(y_t < med)[0]
                n_each = max_trials // 2
                if len(idx_h) > n_each:
                    idx_h = rng.choice(idx_h, n_each, replace=False)
                if len(idx_l) > n_each:
                    idx_l = rng.choice(idx_l, n_each, replace=False)
                idx = np.concatenate([idx_h, idx_l])
                logm_t = logm_t[idx]
                y_t = y_t[idx]
            logm_tr_list.append(logm_t)
            y_tr_list.append(y_t)

        if not logm_tr_list:
            continue

        logm_tr = np.concatenate(logm_tr_list, axis=0)
        y_tr = np.concatenate(y_tr_list, axis=0)

        logCref = logm_tr.mean(axis=0)
        feats_tr = tangent_space_logeuclid(logm_tr, logCref)
        feats_te = tangent_space_logeuclid(logm_test, logCref)

        try:
            pipe = make_pipeline()
            pipe.fit(feats_tr, y_tr)
            y_pred = pipe.predict(feats_te)
            r, _ = stats.pearsonr(y_test, y_pred)
            if not np.isfinite(r):
                r = 0.0
            results.append({'subject_id': test_sid, 'pearson_r': float(r),
                            'n_test': len(y_test)})
        except Exception as e:
            print(f'    [error] {test_sid}: {e}')
            continue

        n_done += 1
        if n_done % 20 == 0:
            rs = [r['pearson_r'] for r in results]
            print(f'      progress: {n_done}, mean r={np.mean(rs):.4f}')

    return results


def _loso_mean_r(X, y, split_idx, max_trials=200, random_state=42):
    """内部 LOSO：给定 logm 矩阵 X / 目标 y / 被试边界 split_idx，
    返回 mean of per-subject Pearson r。"""
    rng = np.random.RandomState(random_state)
    rs = []
    for k in range(len(split_idx)):
        start_tr, end_tr = split_idx[k]
        # 训练：除 k 之外的所有被试
        tr_idx = np.concatenate([np.arange(0, start_tr), np.arange(end_tr, len(y))])
        if len(tr_idx) == 0:
            continue
        logm_tr = X[tr_idx]
        y_tr = y[tr_idx]
        if len(y_tr) > max_trials:
            med = np.median(y_tr)
            idx_h = np.where(y_tr >= med)[0]
            idx_l = np.where(y_tr < med)[0]
            n_each = max_trials // 2
            if len(idx_h) > n_each:
                idx_h = rng.choice(idx_h, n_each, replace=False)
            if len(idx_l) > n_each:
                idx_l = rng.choice(idx_l, n_each, replace=False)
            pick = np.concatenate([idx_h, idx_l])
            logm_tr = logm_tr[pick]
            y_tr = y_tr[pick]
        # 测试
        logm_te = X[start_tr:end_tr]
        y_te = y[start_tr:end_tr]
        if len(y_te) < 3:
            continue
        logCref = logm_tr.mean(axis=0)
        feats_tr = tangent_space_logeuclid(logm_tr, logCref)
        feats_te = tangent_space_logeuclid(logm_te, logCref)
        try:
            pipe = make_pipeline()
            pipe.fit(feats_tr, y_tr)
            yp = pipe.predict(feats_te)
            r, _ = stats.pearsonr(y_te, yp)
            if not np.isfinite(r):
                r = 0.0
            rs.append(float(r))
        except Exception:
            continue
    if not rs:
        return float('nan')
    return float(np.mean(rs))


def _permutation_roi(logm_by_sid_roi, y_by_sid, subject_ids,
                     n_perm, random_state, target_label, max_trials=200):
    """ROI 版本置换检验：LOSO actual r + 打乱 y 的 chance 分布"""
    rng = np.random.RandomState(random_state)
    logm_list, y_list = [], []
    sid_order = [s for s in subject_ids
                 if s in logm_by_sid_roi and len(y_by_sid.get(s, [])) >= 30]
    for s in sid_order:
        logm_list.append(logm_by_sid_roi[s])
        y_list.append(y_by_sid[s])
    if not logm_list:
        return {'actual_r': float('nan'), 'chance_mean': float('nan'),
                'chance_95ci': [float('nan'), float('nan')], 'perm_p': float('nan')}
    X = np.concatenate(logm_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    cumsum = np.cumsum([0] + [len(m) for m in logm_list])
    split_idx = np.array(list(zip(cumsum[:-1], cumsum[1:])))
    actual_r = _loso_mean_r(X, y, split_idx, max_trials=max_trials,
                            random_state=random_state)
    if not np.isfinite(actual_r):
        return {'actual_r': float(actual_r), 'chance_mean': float('nan'),
                'chance_95ci': [float('nan'), float('nan')], 'perm_p': float('nan')}
    chance_r = np.empty(n_perm)
    for pi in range(n_perm):
        y_p = rng.permutation(y)
        chance_r[pi] = _loso_mean_r(X, y_p, split_idx, max_trials=max_trials,
                                    random_state=random_state + pi + 1)
    perm_p = float((chance_r >= actual_r).mean())
    return {
        'actual_r': float(actual_r),
        'chance_mean': float(np.mean(chance_r)),
        'chance_95ci': [float(np.percentile(chance_r, 2.5)),
                        float(np.percentile(chance_r, 97.5))],
        'perm_p': perm_p,
    }


def _permutation_roi_multi_seed(logm_by_sid_roi, y_by_sid, subject_ids,
                                n_perm, seeds, target_label, max_trials=200):
    """多 seed 置换：返回每个 seed 的 perm_p 序列与聚合统计。

    解决单 seed 偶然问题：80 次 max_trials 不足以收敛 C/gamma，
    不同 seed 给出波动的 actual_r，导致单一 seed 的 perm_p 不稳定。
    多 seed 平均后给出更稳健的判断。
    """
    logm_list, y_list = [], []
    sid_order = [s for s in subject_ids
                 if s in logm_by_sid_roi and len(y_by_sid.get(s, [])) >= 30]
    for s in sid_order:
        logm_list.append(logm_by_sid_roi[s])
        y_list.append(y_by_sid[s])
    if not logm_list:
        return {'actual_rs': [], 'perm_ps': [],
                'perm_p_mean': float('nan'), 'perm_p_std': float('nan'),
                'perm_p_median': float('nan'),
                'actual_r_mean': float('nan'), 'actual_r_std': float('nan')}

    per_seed_results = []
    for seed in seeds:
        res = _permutation_roi(
            logm_by_sid_roi, y_by_sid, subject_ids,
            n_perm=n_perm, random_state=seed,
            target_label=target_label, max_trials=max_trials,
        )
        per_seed_results.append(res)

    actual_rs = [r['actual_r'] for r in per_seed_results if np.isfinite(r['actual_r'])]
    perm_ps = [r['perm_p'] for r in per_seed_results if np.isfinite(r['perm_p'])]

    return {
        'actual_rs': [float(x) for x in actual_rs],
        'perm_ps': [float(x) for x in perm_ps],
        'perm_p_mean': float(np.mean(perm_ps)) if perm_ps else float('nan'),
        'perm_p_std': float(np.std(perm_ps, ddof=1)) if len(perm_ps) > 1 else 0.0,
        'perm_p_median': float(np.median(perm_ps)) if perm_ps else float('nan'),
        'actual_r_mean': float(np.mean(actual_rs)) if actual_rs else float('nan'),
        'actual_r_std': float(np.std(actual_rs, ddof=1)) if len(actual_rs) > 1 else 0.0,
        'per_seed': per_seed_results,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=None, help='限制被试数（冒烟用）')
    parser.add_argument('--quick', action='store_true', help='5 被试冒烟')
    parser.add_argument('--target', type=str, default=None,
                        help='只跑单个 target: health / taste / willing')
    parser.add_argument('--perm', type=int, default=0,
                        help='置换检验次数，0=不跑（默认），100=正式')
    parser.add_argument('--perm-seeds', type=int, default=1,
                        help='置换检验使用的 seed 数（多 seed 平均更稳健，建议 3-5）')
    parser.add_argument('--max-trials', type=int, default=200,
                        help='_loso_mean_r 内部 SVR 随机搜索次数（默认 200）')
    parser.add_argument('--fdr-alpha', type=float, default=0.05)
    args, _ = parser.parse_known_args()

    # 强制 stdout 行缓冲，解决 python -u 被忽略或 tee 缓冲问题
    import sys
    try:
        sys.stdout.reconfigure(line_buffering=True, write_through=True)
    except (AttributeError, OSError):
        pass

    print('=' * 60)
    print('B1: SVR channel-ROI decoding')
    print('=' * 60)

    # target 选择
    if args.target:
        tgt_map = {'health': 'rating_health',
                   'taste': 'rating_taste',
                   'willing': 'rating_willingnessToEat'}
        target_col_sel = tgt_map.get(args.target.lower())
        if target_col_sel is None:
            raise SystemExit(f'[error] --target 取值 {args.target!r} 无效，可选 {list(tgt_map.keys())}')
        targets_use = {target_col_sel: TARGETS[target_col_sel]}
    else:
        targets_use = TARGETS
    print(f'  targets: {list(targets_use.keys())}')
    print(f'  --perm={args.perm}  --perm-seeds={args.perm_seeds}  --max-trials={args.max_trials}')

    subject_ids = load_subject_list()
    if args.quick:
        subject_ids = subject_ids[:5]
    elif args.n:
        subject_ids = subject_ids[:args.n]
    print(f'\n有效被试: {len(subject_ids)}')

    ch_names = detect_channel_names(subject_ids)
    print(f'  通道顺序: {ch_names[:5]} ... (n={len(ch_names)})')

    print('\n[feat] 预计算 full_0-1000ms logm...')
    full_cache = preload_full(subject_ids)
    t0 = time.time()
    logm_all = compute_logm_all(full_cache, TIME_WINDOW)
    print(f'  完成 ({time.time()-t0:.1f}s), {len(logm_all)} 被试')

    # ===== LOSO 主循环 =====
    rows = []
    summary_raw = {}
    for roi_name, roi_chs in ROI_CHANNELS.items():
        roi_idx, missing = get_roi_indices(ch_names, roi_chs)
        if len(roi_idx) < 4:
            print(f'  [skip] {roi_name}: 只找到 {len(roi_idx)} 个通道')
            continue
        print(f'\n=== ROI: {roi_name} ({len(roi_idx)} channels) ===')
        for target_col, target_label in targets_use.items():
            if target_col not in TARGETS:
                continue
            logm_by_sid_full, y_by_sid = build_target_views(logm_all, target_col)
            logm_by_sid_roi = {
                sid: logm[np.ix_(np.arange(len(logm)), roi_idx, roi_idx)]
                for sid, logm in logm_by_sid_full.items()
            }
            t0 = time.time()
            results = loso_svr_roi(logm_by_sid_roi, y_by_sid, subject_ids)
            if not results:
                continue
            rs = [r['pearson_r'] for r in results]
            t_stat, p_val = stats.ttest_1samp(rs, 0)
            mean_r = float(np.mean(rs))
            std_r = float(np.std(rs, ddof=1))
            d_val = mean_r / std_r if std_r > 0 else 0.0
            print(f'  {target_col:30s} n={len(rs)} r={mean_r:+.4f}±{std_r:.4f} '
                  f't={t_stat:+.2f} p={p_val:.4g} d={d_val:+.2f}  ({time.time()-t0:.1f}s)')
            for r in results:
                rows.append({
                    'roi': roi_name,
                    'n_channels': len(roi_idx),
                    'target': target_col,
                    'subject_id': r['subject_id'],
                    'pearson_r': r['pearson_r'],
                    'n_test': r['n_test'],
                    'mean_r': mean_r,
                    'std_r': std_r,
                    't': t_stat,
                    'p': p_val,
                    'cohen_d': d_val,
                })
            summary_raw[(roi_name, target_col)] = {
                'roi': roi_name,
                'target': target_col,
                'n': len(rs),
                'mean_r': mean_r,
                'std_r': std_r,
                't': float(t_stat),
                'p': float(p_val),
                'cohen_d': d_val,
            }

    df = pd.DataFrame(rows)
    out_csv = RESULTS_DIR / 'svr_roi_channels_results.csv'
    df.to_csv(out_csv, index=False)
    print(f'\n[saved] {out_csv}')

    # ===== 置换检验（可选，多 seed）=====
    perm_all = []
    if args.perm > 0:
        n_seeds = max(1, args.perm_seeds)
        seed_pool = [42, 123, 456, 789, 2024][:n_seeds]
        print(f'\n=== 置换检验: {args.perm} 次 × {n_seeds} seeds × '
              f'{len(ROI_CHANNELS) * len(targets_use)} 格 (max_trials={args.max_trials}) ===')
        for roi_name, roi_chs in ROI_CHANNELS.items():
            roi_idx, missing = get_roi_indices(ch_names, roi_chs)
            if len(roi_idx) < 4:
                continue
            for target_col, target_label in targets_use.items():
                if target_col not in TARGETS:
                    continue
                logm_by_sid_full, y_by_sid = build_target_views(logm_all, target_col)
                logm_by_sid_roi = {
                    sid: logm[np.ix_(np.arange(len(logm)), roi_idx, roi_idx)]
                    for sid, logm in logm_by_sid_full.items()
                }
                t0 = time.time()
                if n_seeds == 1:
                    perm_res = _permutation_roi(
                        logm_by_sid_roi, y_by_sid, subject_ids,
                        args.perm, random_state=42,
                        target_label=target_label, max_trials=args.max_trials)
                    perm_res['roi'] = roi_name
                    perm_res['target'] = target_col
                    print(f'  {roi_name:18s} {target_col:30s} '
                          f'actual={perm_res["actual_r"]:+.4f} '
                          f'chance={perm_res["chance_mean"]:+.4f} '
                          f'perm_p={perm_res["perm_p"]:.3f} '
                          f'({time.time()-t0:.1f}s)')
                else:
                    multi = _permutation_roi_multi_seed(
                        logm_by_sid_roi, y_by_sid, subject_ids,
                        args.perm, seeds=seed_pool,
                        target_label=target_label, max_trials=args.max_trials)
                    perm_res = {
                        'roi': roi_name,
                        'target': target_col,
                        'actual_r': float(multi['actual_r_mean']),
                        'actual_r_std': float(multi['actual_r_std']),
                        'chance_mean': float(np.nanmean(
                            [r['chance_mean'] for r in multi['per_seed']])),
                        'perm_p': float(multi['perm_p_mean']),
                        'perm_p_std': float(multi['perm_p_std']),
                        'perm_p_median': float(multi['perm_p_median']),
                        'actual_rs': multi['actual_rs'],
                        'perm_ps': multi['perm_ps'],
                        'n_seeds': n_seeds,
                        'max_trials': args.max_trials,
                    }
                    # 95% CI 取所有 seed 合并的 chance
                    all_chance = []
                    for r in multi['per_seed']:
                        all_chance.append(r['chance_mean'])
                    perm_res['chance_95ci'] = [
                        float(np.nanpercentile(all_chance, 2.5)),
                        float(np.nanpercentile(all_chance, 97.5)),
                    ] if all_chance else [float('nan'), float('nan')]
                    print(f'  {roi_name:18s} {target_col:30s} '
                          f'actual={perm_res["actual_r"]:+.4f}±{perm_res["actual_r_std"]:.4f} '
                          f'chance={perm_res["chance_mean"]:+.4f} '
                          f'perm_p={perm_res["perm_p"]:.3f}±{perm_res["perm_p_std"]:.3f} '
                          f'(seeds={n_seeds}, {time.time()-t0:.1f}s)')
                perm_all.append(perm_res)
    else:
        print('\n[skip] --perm=0，不做置换检验')

    # ===== FDR 校正（仅当跑了置换）=====
    fdr_result = None
    if perm_all:
        # 多 seed 时用 perm_p_mean（跨 seed 平均）做 FDR；单 seed 用 perm_p
        perm_p_for_fdr = [
            p.get('perm_p_mean', p['perm_p']) for p in perm_all
        ]
        fdr_result = apply_fdr_correction(perm_p_for_fdr, alpha=args.fdr_alpha)
        for p, q, rej, pp in zip(perm_all, fdr_result['q_values'],
                                 fdr_result['rejected'], perm_p_for_fdr):
            p['q_value'] = float(q)
            p['fdr_significant'] = bool(rej)
            p['perm_p_for_fdr'] = float(pp)
        n_sig = sum(1 for p in perm_all if p['fdr_significant'])
        print(f'\n=== BH-FDR (alpha={args.fdr_alpha}, {len(perm_all)} tests): {n_sig} 显著 ===')
        for p in perm_all:
            tag = 'SIG' if p['fdr_significant'] else 'ns'
            pstr = f'{p["perm_p"]:.3f}'
            if 'perm_p_std' in p:
                pstr += f'±{p["perm_p_std"]:.3f}'
            print(f'  {p["roi"]:18s} {p["target"]:30s} perm_p={pstr} '
                  f'q={p["q_value"]:.3f}  {tag}')

    # ===== 汇总写入 summary_raw =====
    for k, v in summary_raw.items():
        perm_match = [p for p in perm_all
                      if p['roi'] == k[0] and p['target'] == k[1]]
        if perm_match:
            v['perm_p'] = float(perm_match[0].get('perm_p_for_fdr',
                                                  perm_match[0]['perm_p']))
            v['q_value'] = float(perm_match[0].get('q_value', float('nan')))
            v['fdr_significant'] = bool(perm_match[0].get('fdr_significant', False))
            v['chance_mean'] = float(perm_match[0]['chance_mean'])
            v['actual_r_perm'] = float(perm_match[0]['actual_r'])
            if 'actual_r_std' in perm_match[0]:
                v['actual_r_std'] = float(perm_match[0]['actual_r_std'])
            if 'perm_p_std' in perm_match[0]:
                v['perm_p_std'] = float(perm_match[0]['perm_p_std'])

    summary_df = pd.DataFrame(list(summary_raw.values()))
    summary_df = summary_df.sort_values(['target', 'mean_r'], ascending=[True, False])
    summary_csv = RESULTS_DIR / 'svr_roi_channels_summary.csv'
    summary_df.to_csv(summary_csv, index=False)
    print(f'\n[saved] {summary_csv}')
    print('\n=== ROI 汇总（按 mean r 排序）===')
    for _, row in summary_df.iterrows():
        sig = '***' if row['p'] < 0.001 else ('**' if row['p'] < 0.01
              else ('*' if row['p'] < 0.05 else 'ns'))
        extra = ''
        if 'q_value' in row.index and not pd.isna(row['q_value']):
            extra += f'  q={row["q_value"]:.3f}' + (' ✓FDR' if bool(row.get('fdr_significant')) else '')
        print(f"  {row['roi']:18s} {row['target']:30s} "
              f"r={row['mean_r']:+.4f}  t={row['t']:+.2f}  p={row['p']:.4g} {sig}{extra}")

    # ===== 可视化：3 target × 7 ROI 热图 + FDR 叠加 =====
    pivot = summary_df.pivot(index='roi', columns='target', values='mean_r')
    fig, ax = plt.subplots(figsize=(9, 5.5))
    vmin = -abs(pivot.values).max()
    vmax = abs(pivot.values).max()
    im = ax.imshow(pivot.values, aspect='auto', cmap='RdBu_r', vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=20, ha='right', fontsize=9)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=9)
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            ax.text(j, i, f'{val:+.3f}', ha='center', va='center',
                    fontsize=8, color='white' if abs(val) > 0.02 else 'black')
    # FDR 显著格打 ✓，ns 打 ✗
    if perm_all:
        for pi, pr in enumerate(perm_all):
            roi = pr['roi']
            tgt = pr['target']
            if tgt not in pivot.columns or roi not in pivot.index:
                continue
            j = list(pivot.columns).index(tgt)
            i = list(pivot.index).index(roi)
            color = '#31a354' if pr['fdr_significant'] else '#e74c3c'
            ax.text(j + 0.35, i + 0.3, '✓' if pr['fdr_significant'] else '✗',
                    ha='left', va='top', fontsize=7, color=color, fontweight='bold')
    title = (f'SVR LOSO decoding ({WINDOW_NAME}): mean Pearson r by ROI\n'
             f'BioSemi 64-channel subselection')
    if perm_all:
        n_sig = sum(1 for p in perm_all if p['fdr_significant'])
        title += f'\nBH-FDR α={args.fdr_alpha}: {n_sig}/{len(perm_all)} significant '
        title += '(✓sig / ✗ns)'
    ax.set_title(title, fontsize=11)
    plt.colorbar(im, ax=ax, label='mean r')
    plt.tight_layout()
    out_png = FIGURES_DIR / 'fig24_svr_roi_channels.png'
    plt.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'\n[fig] {out_png}')

    # ===== JSON 汇总 =====
    summary_json = {
        'n_subjects': len(subject_ids),
        'n_rois': len(ROI_CHANNELS),
        'n_targets': len(targets_use),
        'window': WINDOW_NAME,
        'time_window_s': list(TIME_WINDOW),
        'results': [dict(r) for r in list(summary_raw.values())],
    }
    if perm_all:
        summary_json['permutation'] = perm_all
        summary_json['fdr'] = {
            'method': 'benjamini_hochberg',
            'alpha': float(args.fdr_alpha),
            'n_tests': len(perm_all),
            'n_significant': sum(1 for p in perm_all if p['fdr_significant']),
            'results': perm_all,
        }
    out_json = RESULTS_DIR / 'svr_roi_channels_summary.json'
    import json
    with open(out_json, 'w') as f:
        json.dump(summary_json, f, indent=2)
    print(f'[saved] {out_json}')


if __name__ == '__main__':
    main()