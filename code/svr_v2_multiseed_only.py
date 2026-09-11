# -*- coding: utf-8 -*-
"""
svr_v2_multiseed_only.py
========================
仅跑 18-cell 全头 SVR 族的 v2 多 seed 置换阶段。

背景：
    2026-09-07 的完整 svr_decoding.py 回跑在 v1 全窗置换阶段被外部终止
    （exit -1，无 traceback），但 LOSO 群体结果已落盘 svr_decoding_results.csv。
    本脚本复用 svr_decoding 的数据加载 / logm 视图 / FDR，以及
    svr_multiseed_perm 的 _perm_cell_multi_seed，跳过耗时的 LOSO 与 v1 置换
    （v1 数值有 backup_v1_18cell_20260907 兜底），直接产出与主脚本
    完全同 schema 的 svr_decoding_multiseed.csv / .json。

v2 配置（与被终止的完整运行一致）：
    - 5 个种子 [42, 123, 456, 789, 2024]
    - 池级 trial budget 200（跨被试合并池按 y 中位平衡抽样）
    - 每 seed 100 次标签打乱置换（合并池整体打乱）
    - FDR 用跨 seed 平均 perm_p 做 BH
"""

import sys
import os
import time
import json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

from config import RESULTS_DIR
from svr_decoding import (
    load_subject_list, preload_full, compute_logm_all, build_target_views,
    apply_fdr_correction, tangent_space_logeuclid, make_pipeline,
    TIME_WINDOWS, TARGETS,
)
from svr_multiseed_perm import _perm_cell_multi_seed


def main(n_perm=100, n_seeds=5, max_trials=200, windows=None):
    print('=' * 60)
    print('SVR v2 multi-seed permutation ONLY (18-cell family)')
    print(f'  seeds={n_seeds}, n_perm/seed={n_perm}, pool budget={max_trials}')
    print('=' * 60)

    subject_ids = load_subject_list()
    print(f'\n有效被试: {len(subject_ids)}')

    t0 = time.time()
    full_cache = preload_full(subject_ids)
    print(f'[preload] 完整数据: {len(full_cache)} 被试 ({time.time()-t0:.1f}s)')

    seed_pool = [42, 123, 456, 789, 2024][:n_seeds]
    win_items = [(w, TIME_WINDOWS[w]) for w in windows] if windows \
        else list(TIME_WINDOWS.items())
    print(f'\n=== 多 seed 置换: {n_perm} 次 × {n_seeds} seeds × '
          f'{len(win_items) * len(TARGETS)} 格（池级预算 max_trials={max_trials}）===')

    ms_all = []
    for wname, tw in win_items:
        t0w = time.time()
        logm_all = compute_logm_all(full_cache, tw)
        for target_col in TARGETS.keys():
            logm_by_sid, y_by_sid = build_target_views(logm_all, target_col)
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
            print(f"  {wname:18s} {target_col:26s} "
                  f"actual={res['actual_r_mean']:+.4f}±{res['actual_r_std']:.4f} "
                  f"perm_p={res['perm_p_mean']:.3f}±{res['perm_p_std']:.3f} "
                  f"[{res['perm_p_min']:.2f},{res['perm_p_max']:.2f}] "
                  f"({time.time()-t0:.0f}s)")
        print(f'  [window {wname} done, {time.time()-t0w:.0f}s]')

    # FDR：跨 seed 平均 perm_p 做 BH（与主脚本 / ROI v2 一致，D1 决议）
    fdr_ms = apply_fdr_correction(
        [r['perm_p_mean'] for r in ms_all], alpha=0.05)
    for r, q, rej in zip(ms_all, fdr_ms['q_values'], fdr_ms['rejected']):
        r['q_value'] = float(q)
        r['fdr_significant'] = bool(rej)

    n_sig = sum(r['fdr_significant'] for r in ms_all)
    print(f'\n=== FDR （多 seed, BH α=0.05, 输入=跨 seed 平均 perm_p） ===')
    for r in sorted(ms_all, key=lambda x: x['q_value']):
        flag = ' *** FDR' if r['fdr_significant'] else ''
        print(f"  {r['window']:18s} {r['target']:26s} "
              f"perm_p(mean)={r['perm_p_mean']:.4g} q={r['q_value']:.4g}{flag}")
    print(f'  -> {n_sig}/{len(ms_all)} FDR-significant')

    # CSV（列与主脚本 L842-856 完全一致）
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
    tag = ('_' + '_'.join(w.split('_')[0] for w in windows)) if windows else ''
    ms_csv = RESULTS_DIR / f'svr_decoding_multiseed{tag}.csv'
    ms_df.to_csv(ms_csv, index=False)
    print(f'\n[saved] {ms_csv}')

    fdr_meta = {
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
    ms_json = RESULTS_DIR / f'svr_decoding_multiseed{tag}.json'
    with open(ms_json, 'w', encoding='utf-8') as f:
        json.dump({'cells': ms_all, 'fdr': fdr_meta},
                  f, indent=2, ensure_ascii=False, default=str)
    print(f'[saved] {ms_json}')


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--perm', type=int, default=100)
    p.add_argument('--seeds', type=int, default=5)
    p.add_argument('--max-trials', type=int, default=200)
    p.add_argument('--windows', type=str, default=None,
                   help='逗号分隔的窗口名子集；缺省跑全部 6 窗')
    args = p.parse_args()
    wins = args.windows.split(',') if args.windows else None
    main(n_perm=args.perm, n_seeds=args.seeds,
         max_trials=args.max_trials, windows=wins)
