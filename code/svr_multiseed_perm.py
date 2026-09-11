"""svr_decoding 的多 seed 置换函数（v2 口径，对齐 svr_roi_channels 审计方案）。

独立成模块的原因：svr_decoding.py 在本机受 IDE 旧缓冲区间歇性自动保存
覆盖（2026-09-04 多次复现），大段插入易被回滚；独立新文件不受影响。

被两处引用：
- svr_decoding.py main() 的 --perm-seeds>1 分支（18 格回归族 v2 重跑）
- svr_budget_sensitivity.py（B2 预算敏感性诊断）
"""
import numpy as np
from scipy import stats


def _loso_mean_r_pool(X_pool, y_pool, split_idx, max_trials=200, random_state=42,
                      tangent_space_logeuclid=None, make_pipeline=None):
    """池级预算 LOSO：合并训练池整体按 max_trials 做中位平衡抽样。

    语义与 svr_roi_channels._loso_mean_r 完全一致（v2 审计口径），注意与
    svr_decoding.run_loso_svr 的区别：后者按【每被试】max_trials 抽样，
    此处按【合并池】整体抽样 —— 论文 §3.5 的跨 seed 观测统计量即来自本口径。

    tangent_space_logeuclid / make_pipeline 由调用方注入
    （svr_decoding 模块内同名函数，避免循环导入）。
    """
    rng = np.random.RandomState(random_state)
    rs = []
    for k in range(len(split_idx)):
        start_te, end_te = split_idx[k]
        tr_idx = np.concatenate([np.arange(0, start_te),
                                 np.arange(end_te, len(y_pool))])
        if len(tr_idx) == 0:
            continue
        logm_tr = X_pool[tr_idx]
        y_tr = y_pool[tr_idx]
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
        logm_te = X_pool[start_te:end_te]
        y_te = y_pool[start_te:end_te]
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
    return float(np.mean(rs)) if rs else float('nan')


def _perm_cell_multi_seed(logm_by_sid, y_by_sid, subject_ids,
                          n_perm, seeds, max_trials=200,
                          tangent_space_logeuclid=None, make_pipeline=None):
    """单一 (window, target) 格的多 seed 置换（对齐 svr_roi_channels 口径）。

    - actual 与 chance 均用池级预算 LOSO（_loso_mean_r_pool）；
    - 标签在【合并 trial 池】内整体打乱（跨被试重排；被试边界仅约束
      EEG-trial 归属，与论文 §2.6/§3.5 描述及 ROI v2 一致）；
    - 每 seed 独立给 perm_p，聚合 mean±SD/min/max（D1 决议：FDR 用 mean）。
    """
    sid_order = [s for s in subject_ids
                 if s in logm_by_sid and len(y_by_sid.get(s, [])) >= 30]
    if not sid_order:
        return None
    X_pool = np.concatenate([logm_by_sid[s] for s in sid_order], axis=0)
    y_pool = np.concatenate([y_by_sid[s] for s in sid_order], axis=0)
    cumsum = np.cumsum([0] + [len(logm_by_sid[s]) for s in sid_order])
    split_idx = np.array(list(zip(cumsum[:-1], cumsum[1:])))
    per_seed = []
    for seed in seeds:
        actual_r = _loso_mean_r_pool(
            X_pool, y_pool, split_idx, max_trials=max_trials,
            random_state=seed,
            tangent_space_logeuclid=tangent_space_logeuclid,
            make_pipeline=make_pipeline)
        rng = np.random.RandomState(seed)
        chance = np.empty(n_perm)
        for pi in range(n_perm):
            y_p = rng.permutation(y_pool)
            chance[pi] = _loso_mean_r_pool(
                X_pool, y_p, split_idx, max_trials=max_trials,
                random_state=seed + pi + 1,
                tangent_space_logeuclid=tangent_space_logeuclid,
                make_pipeline=make_pipeline)
        per_seed.append({
            'seed': int(seed),
            'actual_r': float(actual_r),
            'chance_mean': float(np.mean(chance)),
            'chance_95ci': [float(np.percentile(chance, 2.5)),
                            float(np.percentile(chance, 97.5))],
            'perm_p': float((chance >= actual_r).mean()),
        })
    actual_rs = [d['actual_r'] for d in per_seed if np.isfinite(d['actual_r'])]
    perm_ps = [d['perm_p'] for d in per_seed if np.isfinite(d['perm_p'])]
    return {
        'n_subjects': len(sid_order),
        'n_trials': int(len(y_pool)),
        'actual_rs': [float(x) for x in actual_rs],
        'actual_r_mean': float(np.mean(actual_rs)) if actual_rs else float('nan'),
        'actual_r_std': float(np.std(actual_rs, ddof=1)) if len(actual_rs) > 1 else 0.0,
        'perm_ps': [float(x) for x in perm_ps],
        'perm_p_min': float(np.min(perm_ps)) if perm_ps else float('nan'),
        'perm_p_max': float(np.max(perm_ps)) if perm_ps else float('nan'),
        'perm_p_mean': float(np.mean(perm_ps)) if perm_ps else float('nan'),
        'perm_p_std': float(np.std(perm_ps, ddof=1)) if len(perm_ps) > 1 else 0.0,
        'per_seed': per_seed,
        'max_trials_pool': int(max_trials),
        'permutation': 'pooled-label-shuffle',
    }
