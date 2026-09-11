"""B2 诊断（修改计划 2026-09-04）：观测统计量 r 随训练预算语义与规模的变化。

解释论文 §3.5 的数值断层（Table 3/4 的 r̄≈0.050 vs 多 seed 置换观测统计量
r≈0.015–0.016）：
- per_subject 语义：run_loso_svr —— 每个训练被试各自平衡抽到 max_trials 条
  （budget=80/200 时总训练量 ≈ 被试数×budget），Table 3/4 的 r̄/t/d 来源；
- pool 语义：_loso_mean_r_pool —— 合并训练池整体平衡抽到 max_trials 条
  （总训练量恒等于 budget），v2 多 seed 置换的观测统计量来源。

锚点验证：occipital×taste/willing 在两种语义、budget=200 下的 r 应分别
复现 ≈0.050 与 ≈0.015–0.016。

用法：
    python code/svr_budget_sensitivity.py             # 全量 106 被试
    python code/svr_budget_sensitivity.py --quick     # 10 被试冒烟

产出：outputs/results/budget_sensitivity.csv（+ 控制台摘要）
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from svr_decoding import (
    load_subject_list, preload_full, compute_logm_all, build_target_views,
    run_loso_svr, TIME_WINDOWS, TARGETS,
    tangent_space_logeuclid, make_pipeline,
)
from svr_multiseed_perm import _loso_mean_r_pool
from svr_roi_channels import ROI_CHANNELS, get_roi_indices, detect_channel_names
from config import RESULTS_DIR

INF = 10 ** 9  # 以大数表示"无预算上限"

SEEDS = [42, 123, 456, 789, 2024]
PER_SUBJECT_BUDGETS = [80, 120, 160, 200, INF]
POOL_BUDGETS = [80, 200, 400, 800, 1600, INF]
ANCHOR_BUDGET = 200
ANCHOR_ROI = 'occipital'
ANCHOR_TARGETS = ['rating_taste', 'rating_willingnessToEat']


def label_budget(b):
    return 'inf' if b >= INF else str(b)


def main(quick=False):
    t_start = time.time()
    subject_ids = load_subject_list()
    if quick:
        subject_ids = subject_ids[:10]
    print(f'B2 预算敏感性诊断: {len(subject_ids)} 被试')

    wname = next(k for k in TIME_WINDOWS if k.startswith('full'))
    tw = TIME_WINDOWS[wname]
    print(f'窗口: {wname}')

    full_cache = preload_full(subject_ids)
    logm_all = compute_logm_all(full_cache, tw)
    ch_names = detect_channel_names(subject_ids)
    roi_idx, missing = get_roi_indices(ch_names, ROI_CHANNELS[ANCHOR_ROI])
    print(f'锚点 ROI {ANCHOR_ROI}: {len(roi_idx)} 通道'
          + (f'（缺失 {missing}）' if missing else ''))

    rows = []

    def add(semantics, roi, target, budget, r_mean, r_std, n_subj, n_seeds):
        rows.append({
            'semantics': semantics,          # per_subject | pool
            'roi': roi,                      # full_montage | occipital
            'target': target,
            'window': wname,
            'budget': label_budget(budget),
            'r_mean': float(r_mean),
            'r_std': float(r_std) if r_std is not None else np.nan,
            'n_subjects': int(n_subj),
            'n_seeds': int(n_seeds),
        })
        print(f'  [{semantics:11s}] {roi:12s} {target:28s} '
              f'budget={label_budget(budget):>4s} '
              f'r={r_mean:+.4f}' + (f'±{r_std:.4f}' if r_std is not None else ''))

    for target_col in TARGETS.keys():
        logm_by_sid, y_by_sid = build_target_views(logm_all, target_col)
        n_subj = len(logm_by_sid)
        if n_subj == 0:
            continue

        # ---- per_subject 语义（run_loso_svr，v1 固定内部 rng，单次实现）----
        for budget in PER_SUBJECT_BUDGETS:
            results = run_loso_svr(subject_ids, target_col, tw,
                                   max_trials=budget,
                                   logm_by_sid=logm_by_sid, y_by_sid=y_by_sid)
            if not results:
                continue
            rs = [r['pearson_r'] for r in results]
            add('per_subject', 'full_montage', target_col, budget,
                np.mean(rs), None, len(rs), 1)

        # ---- pool 语义（_loso_mean_r_pool，多 seed 取 mean±SD）----
        sid_order = [s for s in subject_ids
                     if s in logm_by_sid and len(y_by_sid.get(s, [])) >= 30]
        if sid_order:
            X_pool = np.concatenate([logm_by_sid[s] for s in sid_order], axis=0)
            y_pool = np.concatenate([y_by_sid[s] for s in sid_order], axis=0)
            cumsum = np.cumsum([0] + [len(logm_by_sid[s]) for s in sid_order])
            split_idx = np.array(list(zip(cumsum[:-1], cumsum[1:])))
            for budget in POOL_BUDGETS:
                rs = [_loso_mean_r_pool(X_pool, y_pool, split_idx,
                                        max_trials=budget, random_state=seed,
                                        tangent_space_logeuclid=tangent_space_logeuclid,
                                        make_pipeline=make_pipeline)
                      for seed in SEEDS]
                rs = [r for r in rs if np.isfinite(r)]
                if rs:
                    add('pool', 'full_montage', target_col, budget,
                        np.mean(rs), np.std(rs, ddof=1), len(sid_order), len(rs))
            del X_pool, y_pool

    # ---- 锚点：occipital × taste/willing × budget=200，两种语义 ----
    for target_col in ANCHOR_TARGETS:
        logm_by_sid, y_by_sid = build_target_views(logm_all, target_col)
        if not logm_by_sid:
            continue
        logm_roi = {sid: logm[np.ix_(np.arange(len(logm)), roi_idx, roi_idx)]
                    for sid, logm in logm_by_sid.items()}

        results = run_loso_svr(subject_ids, target_col, tw,
                               max_trials=ANCHOR_BUDGET,
                               logm_by_sid=logm_roi, y_by_sid=y_by_sid)
        if results:
            rs = [r['pearson_r'] for r in results]
            add('per_subject', ANCHOR_ROI, target_col, ANCHOR_BUDGET,
                np.mean(rs), None, len(rs), 1)

        sid_order = [s for s in subject_ids
                     if s in logm_roi and len(y_by_sid.get(s, [])) >= 30]
        if sid_order:
            X_pool = np.concatenate([logm_roi[s] for s in sid_order], axis=0)
            y_pool = np.concatenate([y_by_sid[s] for s in sid_order], axis=0)
            cumsum = np.cumsum([0] + [len(logm_roi[s]) for s in sid_order])
            split_idx = np.array(list(zip(cumsum[:-1], cumsum[1:])))
            rs = [_loso_mean_r_pool(X_pool, y_pool, split_idx,
                                    max_trials=ANCHOR_BUDGET,
                                    random_state=seed,
                                    tangent_space_logeuclid=tangent_space_logeuclid,
                                    make_pipeline=make_pipeline)
                  for seed in SEEDS]
            rs = [r for r in rs if np.isfinite(r)]
            if rs:
                add('pool', ANCHOR_ROI, target_col, ANCHOR_BUDGET,
                    np.mean(rs), np.std(rs, ddof=1), len(sid_order), len(rs))
            del X_pool, y_pool

    df = pd.DataFrame(rows)
    out = RESULTS_DIR / 'budget_sensitivity.csv'
    df.to_csv(out, index=False)
    print(f'\n[saved] {out}  ({len(df)} 行, 总耗时 {time.time()-t_start:.0f}s)')

    # 摘要：两条语义的 r 随 budget 的变化（full_montage × taste 为例）
    print('\n=== 摘要（full_montage × rating_taste）===')
    for sem in ['per_subject', 'pool']:
        sub = df[(df['semantics'] == sem) &
                 (df['target'] == 'rating_taste') &
                 (df['roi'] == 'full_montage')]
        line = '  ' + sem + ': ' + '  '.join(
            f'{r.budget}→{r.r_mean:+.4f}' for r in sub.itertuples())
        print(line)


if __name__ == '__main__':
    main(quick='--quick' in sys.argv)
