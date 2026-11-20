# -*- coding: utf-8 -*-
"""
audit_seed_inference.py
=======================
单试次 EEG 解码的随机种子推断审计。
（Cell Reports Methods 转投核心新分析；Array 大修沿用并扩展，
见 paper/Array/ARRAY_major_revision_plan.md §1–§4。）

解决的问题
----------
原稿用"跨 seed 平均 perm_p"(mean-p) 作为推断量。mean-p 既不是合法的 p 值，
也无法给出校准过的第一类错误率；单 seed 推断则因平衡抽样随机性而高度波动。
本脚本把推断量替换为**种子聚合检验统计量**

        T_obs = (1/S) * sum_s T_s

并以**共享置换**构造其零分布（同一批标签置换施加于所有 seed，使
T^(b) = (1/S) * sum_s T_s^(b) 成为聚合统计量的合法零实现），

        p = (1 + #{ T^(b) >= T_obs }) / (B + 1)

并在**真零**数据上给出四类规则（单 seed / min-p / mean-p / 种子聚合）
在 alpha=0.05 下的第一类错误率。

统计量口径与论文一致：池级预算 LOSO（_loso_mean_r，max_trials=200）、
合并 trial 池打乱标签、Pearson r 的被试均值。全部计算按 seed / 置换并行。

Array 大修新增维度
------------------
--unit     pooled（原口径）| within（被试段内置换，§1.3 置换单位敏感性）
--semantics  pool（配置 A，总预算 200/fold）| per_subject（配置 B，200/被试）
--max-trials 预算数值；inf = 配置 C（全部训练 trial，§2 预算匹配）
同参数重复运行会自动跳过已完成 cell（断点续跑，--out 相同即可）。

子分析
------
cell   观测数据：(a) 种子聚合统计量与规则对照；(b) 种子数敏感性 S'∈{1,2,3,5,10,20}。
calib  真零复制（标签置换）下的第一类错误率校准（代表单元）。

用法
----
    python code/audit_seed_inference.py --selftest
    python code/audit_seed_inference.py --stage cell  --perm 100 --seeds 10 --n_jobs 20
    python code/audit_seed_inference.py --stage calib --reps 200 --perm 100 --seeds 10 \\
        --calib-cell occipital|rating_taste --n_jobs 20
    # Array 大修示例（Linux 工作站，推荐经 run_major_revision_audits.sh 调度）：
    python code/audit_seed_inference.py --stage cell --perm 5000 --seeds 20 \\
        --cells occipital|rating_taste,occipital|rating_willingnessToEat
    python code/audit_seed_inference.py --stage cell --perm 2000 --seeds 5 --unit within \\
        --cells occipital|rating_taste --out outputs/results/seed_inference_audit_within.json
    python code/audit_seed_inference.py --stage cell --perm 2000 --seeds 5 \\
        --semantics per_subject --out outputs/results/budget_audit_per_subject.json
"""
from __future__ import annotations
import sys
import os

# 进程内 BLAS 单线程，避免 joblib 多进程时线程超订
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
           'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '1')

import time
import json
import argparse
from pathlib import Path
from itertools import combinations

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from svr_decoding import (
    load_subject_list, preload_full, compute_logm_all, build_target_views,
    TIME_WINDOWS,
)
from svr_roi_channels import (
    ROI_CHANNELS, get_roi_indices, detect_channel_names, _loso_mean_r,
)
from config import RESULTS_DIR

# seed 池：前 5 个与论文/原稿一致（5-seed 结果直接对齐原稿）；
# 扩展至 20 个以支撑 Array 大修 §4.2 的 S={3,5,10,20} 敏感性曲线。
SEED_POOL = [42, 123, 456, 789, 2024, 7, 99, 555, 31415, 2718,
             5, 11, 17, 23, 29, 31, 37, 41, 43, 47]
PAPER_SEEDS = SEED_POOL[:5]

WINDOW_NAME = 'full_0-1000ms'
TIME_WINDOW = TIME_WINDOWS[WINDOW_NAME]

# 审计单元：(ROI 名, target)；'all' 表示全通道
CELLS = [
    ('occipital', 'rating_taste'),
    ('occipital', 'rating_willingnessToEat'),
    ('all',       'rating_taste'),
    ('all',       'rating_health'),
    ('all',       'rating_willingnessToEat'),
]

# Array 大修 §1.4：正式 p 值必须在**预定义检验家族**内做 BH 校正。
# 原稿的 ROI 家族 = 7 个 ROI × 3 个 target = 21 个 cell
# （见 outputs/results/svr_roi_channels_summary.json：n_tests=21，
#   ROI = frontal / fronto-central / central / centro-parietal /
#         parietal / occipital / temporal）。
# 本轮补跑该家族，用种子聚合统计量的置换 p 值重算 BH。
TARGETS = ['rating_health', 'rating_taste', 'rating_willingnessToEat']
ROI_FAMILY_CELLS = [(roi, t) for roi in ROI_CHANNELS for t in TARGETS]
SCALP_CELLS = [('all', t) for t in TARGETS]
# 全部合法审计单元（ROI 家族 21 + 全脑 3）
REGISTRY_CELLS = ROI_FAMILY_CELLS + SCALP_CELLS

# 原稿跨 seed 平均口径下 full-window 的报告值，用于流水线一致性核对
PAPER_ACTUAL_R_MEAN = {
    'all|rating_taste': 0.008774,
    'all|rating_health': 0.004756,
    'all|rating_willingnessToEat': 0.006303,
}

MAX_TRIALS = 200
ALPHA = 0.05
CALIB_SEED_COUNTS = [1, 5, 10]

# Array 大修 §2：配置 C（全部训练 trial）用大数表示无预算上限
INF_BUDGET = 10 ** 9
# Array 大修 §4.2：seed 数敏感性网格
SEED_COUNT_GRID = (1, 2, 3, 5, 10, 20)

# 全脑 64 通道单次解码约为枕区的 ~27 倍，故全脑单元只用前 5 个论文 seed，
# 以控制总耗时；枕区单元用满 20 个 seed 以支撑 seed 数敏感性曲线（§4.2）。
MAX_SEEDS_WHOLESCALP = 5

OUT_JSON = RESULTS_DIR / 'seed_inference_audit.json'


def _dump(result, out_path):
    """原子式写出当前 result（含部分结果），便于中断后保留已完成单元。"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + '.tmp')
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    os.replace(tmp, out_path)


# ============================================================
# 并行原语
# ============================================================
def _one_run(X, y, split_idx, max_trials, random_state, semantics='pool'):
    """单次 _loso_mean_r（模块级，便于 joblib 序列化）。"""
    return _loso_mean_r(X, y, split_idx, max_trials=max_trials,
                        random_state=int(random_state), semantics=semantics)


def make_permuted_ys(y, split_idx, n_perm, unit='pooled', rng=None):
    """生成 B 个标签置换（同一批施加于所有 seed，构成共享零分布）。

    unit（Array 大修 §1.3 置换单位敏感性）：
      'pooled' —— 整个 trial 池打乱（原口径，与旧实现逐位一致）；
      'within' —— 每个被试段内独立打乱（保持被试内标签边际不变）。
    """
    if rng is None:
        rng = np.random.RandomState(20240901)
    if unit == 'pooled':
        return [rng.permutation(y) for _ in range(n_perm)]
    out = []
    for _ in range(n_perm):
        py = np.array(y, copy=True)
        for s0, e0 in split_idx:
            py[s0:e0] = rng.permutation(y[s0:e0])
        out.append(py)
    return out


def _norm_budget(v):
    """max_trials 归一：'inf'/float('inf')/>=INF_BUDGET → 'inf'，否则 int。"""
    if isinstance(v, str):
        try:
            v = float(v)
        except ValueError:
            return 'inf' if v.strip().lower() in ('inf', 'infinity') else v
    return 'inf' if float(v) >= INF_BUDGET else int(v)


def _parallel(n_jobs):
    from joblib import Parallel, delayed
    if n_jobs is None or n_jobs <= 0:
        n_jobs = max(1, (os.cpu_count() or 4) - 4)
    return Parallel(n_jobs=n_jobs, backend='loky'), delayed


def _run_perm_chunk(X, split_idx, max_trials, seed, b_start, perm_ys_chunk,
                    semantics='pool'):
    """(worker) 对一个 seed 的一批共享置换求统计量。

    同一任务内复用同一份 X（joblib 对大数组做 memmap），
    按 chunk 摊薄数据搬运开销 —— 全脑 64 通道 X≈0.75GB，逐置换派发会被
    memmap 反复搬运拖垮；分批后开销可忽略。
    """
    return np.asarray([
        _one_run(X, py, split_idx, max_trials, int(seed) + (b_start + j) + 1,
                 semantics)
        for j, py in enumerate(perm_ys_chunk)], float)


def per_seed_shared_perm(X, y, split_idx, seeds, n_perm, max_trials,
                         perm_rng, n_jobs=1, chunk=None,
                         unit='pooled', semantics='pool'):
    """给定 (X,y)：每 seed 的 actual_r，以及共享置换下的统计量矩阵。

    共享置换：B 个标签置换只抽一次并施加到所有 seed，使
    T^(b) = mean_s T_s^(b) 成为聚合统计量的合法零实现。
    unit 控置换单位（pooled/within，§1.3），semantics 控预算语义（§2）。

    置换任务按 chunk 分批（默认使任务数≈n_jobs），以摊薄大数组传输开销。
    返回 (actual_rs (S,), stat (S,B))。
    """
    par, delayed = _parallel(n_jobs)
    S = len(seeds)
    actual = np.asarray(par(
        delayed(_one_run)(X, y, split_idx, max_trials, int(s), semantics)
        for s in seeds), float)

    perm_ys = make_permuted_ys(y, split_idx, n_perm, unit=unit, rng=perm_rng)
    if chunk is None:
        chunk = max(1, int(np.ceil(S * n_perm / max(1, n_jobs))))
    tasks = []
    for si, s in enumerate(seeds):
        for b0 in range(0, n_perm, chunk):
            tasks.append((si, int(s), b0, perm_ys[b0:b0 + chunk]))
    vals = par(
        delayed(_run_perm_chunk)(X, split_idx, max_trials, s, b0, sub,
                                 semantics)
        for (si, s, b0, sub) in tasks)
    stat = np.empty((S, n_perm), float)
    for (si, s, b0, sub), v in zip(tasks, vals):
        stat[si, b0:b0 + len(sub)] = v
    return actual, stat


# ============================================================
# 统计量 / p 值
# ============================================================
def pval_correct(null_vals, obs):
    """标准置换 p：(1 + #{null >= obs}) / (B + 1)。"""
    return float((1 + np.sum(null_vals >= obs)) / (len(null_vals) + 1))


def pval_correct_two(null_vals, obs):
    """标准置换 p（双侧）：(1 + #{ |null| >= |obs| }) / (B + 1)。

    Array 大修 §1.2 给出的双侧公式。方向性假设必须在分析前固定，
    因此两种口径**同时**计算并写入结果，正式 p 由 --sided 指定，
    事后不可更改（避免"看完结果再选方向"）。
    """
    return float((1 + np.sum(np.abs(null_vals) >= abs(obs))) / (len(null_vals) + 1))


def pval_legacy(null_vals, obs):
    """原稿口径：fraction(null >= obs)（保留以便与旧稿数值对照）。"""
    return float(np.mean(null_vals >= obs))


def per_seed_pvals(actual, stat, convention='correct', sided='one'):
    """每 seed 各自零分布下的置换 p。

    convention: 'correct' = 加一修正；'legacy' = 原稿 fraction 口径（仅单侧）
    sided:     'one' = 单侧上尾；'two' = 双侧 |null| >= |obs|
    """
    if convention == 'correct':
        fn = pval_correct_two if sided == 'two' else pval_correct
    else:
        fn = pval_legacy     # legacy 口径本身即单侧，双侧无对应物
    S, B = stat.shape
    return np.array([fn(stat[s, :], actual[s]) for s in range(S)])


def eval_rules(actual, stat, seed_idx, sided='one'):
    """给定选中的 seed 行，计算聚合统计量、聚合 p 与三类对照规则。

    sided: 聚合统计量置换 p 的方向口径（'one' 单侧上尾 / 'two' 双侧）。
    两种口径的 p 同时计算并返回；正式 p 由 sided 指定并写入 meta，
    避免事后按结果选择方向。
    """
    idx = list(seed_idx)
    a = actual[idx]
    T_obs = float(np.mean(a))
    T_null = stat[idx, :].mean(axis=0)
    p_one = pval_correct(T_null, T_obs)
    p_two = pval_correct_two(T_null, T_obs)
    p_agg = p_two if sided == 'two' else p_one
    ps_c = per_seed_pvals(actual, stat, 'correct', sided)[idx]
    ps_l = per_seed_pvals(actual, stat, 'legacy')[idx]
    return {
        'T_obs': T_obs,
        'T_null_mean': float(np.mean(T_null)),
        'T_null_std': float(np.std(T_null, ddof=1)) if len(T_null) > 1 else 0.0,
        'T_null_95ci': [float(np.percentile(T_null, 2.5)),
                        float(np.percentile(T_null, 97.5))],
        'p_agg': p_agg,
        'p_agg_one_sided': p_one,
        'p_agg_two_sided': p_two,
        'sided_primary': sided,
        'p_single': float(ps_c[0]),
        'p_min': float(np.min(ps_c)),
        'p_mean': float(np.mean(ps_c)),
        'p_median': float(np.median(ps_c)),
        'per_seed_p': [float(x) for x in ps_c],
        'per_seed_p_one_sided': [float(x) for x in
                                  per_seed_pvals(actual, stat, 'correct', 'one')[idx]],
        'per_seed_p_two_sided': [float(x) for x in
                                 per_seed_pvals(actual, stat, 'correct', 'two')[idx]],
        'per_seed_p_legacy': [float(x) for x in ps_l],
        'p_mean_legacy': float(np.mean(ps_l)),
    }


# ============================================================
# 池化
# ============================================================
def pool_cell(logm_all, target_col, roi_name, subject_ids, ch_names):
    """返回该 cell 的 (X, y, split_idx, sid_order)。X 为 logm 协方差 (N,C,C)。"""
    logm_by_sid_full, y_by_sid = build_target_views(logm_all, target_col)
    if roi_name == 'all':
        X_by = logm_by_sid_full
    else:
        roi_idx, _ = get_roi_indices(ch_names, ROI_CHANNELS[roi_name])
        X_by = {sid: logm[np.ix_(np.arange(len(logm)), roi_idx, roi_idx)]
                for sid, logm in logm_by_sid_full.items()}
    sid_order = [s for s in subject_ids
                 if s in X_by and len(y_by_sid.get(s, [])) >= 30]
    X = np.concatenate([X_by[s] for s in sid_order], axis=0)
    y = np.concatenate([y_by_sid[s] for s in sid_order], axis=0)
    cumsum = np.cumsum([0] + [len(X_by[s]) for s in sid_order])
    split_idx = np.array(list(zip(cumsum[:-1], cumsum[1:])))
    return X, y, split_idx, sid_order


# ============================================================
# 子分析 1：观测数据的聚合统计量、规则对照与种子数敏感性
# ============================================================
def _cell_sufficient(rec, unit, semantics, max_trials_n, n_perm, n_seeds,
                     sided='one'):
    """断点续跑：已有 cell 结果与本次参数一致且充分则跳过（不重算）。

    sided 必须一致：旧结果缺该字段（单侧口径）时，若本次要求双侧则重算。
    注意：agg 中同时保存了 p_agg_one_sided / p_agg_two_sided，
    因此同一份零分布即可支持两种方向口径；但 by_seed_count 内的
    p_mean / frac_p_le_0.05 是按 sided 口径统计的，故方向改变需重算。
    """
    if not rec:
        return False
    if rec.get('unit', 'pooled') != unit or \
            rec.get('semantics', 'pool') != semantics:
        return False
    if _norm_budget(rec.get('max_trials', MAX_TRIALS)) != max_trials_n:
        return False
    if rec.get('sided', 'one') != sided:
        return False
    # 旧版代码产出的 cell 缺双口径 p（p_agg_one_sided / p_agg_two_sided），
    # 视为不充分 → 强制重算，保证所有上报 cell 都带精确双侧参考值
    if 'p_agg_two_sided' not in (rec.get('agg') or {}) or \
            'p_agg_one_sided' not in (rec.get('agg') or {}):
        return False
    if rec.get('n_perm', 0) < n_perm:
        return False
    seeds_have = rec.get('seeds', [])
    if len(seeds_have) < n_seeds:
        return False
    want = {str(s) for s in SEED_COUNT_GRID if s <= len(seeds_have)}
    return want.issubset(set(rec.get('by_seed_count', {}).keys()))


def run_cell(logm_all, subject_ids, ch_names, seeds, n_perm, n_jobs, cells=None,
             max_trials=MAX_TRIALS, unit='pooled', semantics='pool',
             existing=None, sided='one', max_seeds_scalp=None):
    out = []
    max_trials_n = _norm_budget(max_trials)
    for roi_name, target_col in (cells or CELLS):
        key = f'{roi_name}|{target_col}'
        # 全脑单元的 seed 上限可配（--max-seeds-scalp）：全脑单次解码约为
        # 枕区的 27 倍，per_subject 预算下更达 78 倍（实测），故默认仍限制 5。
        scalp_cap = (MAX_SEEDS_WHOLESCALP if max_seeds_scalp is None
                     else max_seeds_scalp)
        seeds_cell = seeds if roi_name != 'all' else seeds[:scalp_cap]
        if _cell_sufficient((existing or {}).get(key), unit, semantics,
                            max_trials_n, n_perm, len(seeds_cell), sided):
            print(f'  [skip] {key} 已有充分结果（断点续跑）')
            continue
        S = len(seeds_cell)
        X, y, split_idx, sid_order = pool_cell(
            logm_all, target_col, roi_name, subject_ids, ch_names)
        # pooled 与旧实现同种子逐位一致；within 用独立种子区分零分布
        perm_seed = 20240901 if unit == 'pooled' else 20240902
        perm_rng = np.random.RandomState(perm_seed)
        t0 = time.time()
        actual, stat = per_seed_shared_perm(
            X, y, split_idx, seeds_cell, n_perm, max_trials, perm_rng, n_jobs,
            unit=unit, semantics=semantics)
        full = eval_rules(actual, stat, range(S), sided)

        per_S = {}
        for Ssize in [s for s in SEED_COUNT_GRID if s <= S]:
            combos = list(combinations(range(S), Ssize))
            if len(combos) > 300:
                # S=20 时组合数巨大：固定种子随机抽 300 个，避免字典序偏差
                _rng = np.random.RandomState(20240904)
                _sel = sorted(_rng.choice(len(combos), 300, replace=False))
                combos = [combos[i] for i in _sel]
            # 每个组合只调用一次 eval_rules（此前 T_obs 与 p_agg 各调一次，
            # 重复计算了整份零分布，S=20 时开销翻倍）
            _ev = [eval_rules(actual, stat, c, sided) for c in combos]
            Tob = np.array([e['T_obs'] for e in _ev])
            pp = np.array([e['p_agg'] for e in _ev])
            per_S[str(Ssize)] = {
                'n_combinations': len(combos),
                'T_obs_mean': float(Tob.mean()),
                'T_obs_std': float(Tob.std(ddof=1)) if len(Tob) > 1 else 0.0,
                'T_obs_p2.5': float(np.percentile(Tob, 2.5)),
                'T_obs_p97.5': float(np.percentile(Tob, 97.5)),
                'p_mean': float(pp.mean()),
                'p_std': float(pp.std(ddof=1)) if len(pp) > 1 else 0.0,
                'p_min': float(pp.min()),
                'p_max': float(pp.max()),
                'frac_p_le_0.05': float(np.mean(pp <= ALPHA)),
            }

        key = f'{roi_name}|{target_col}'
        rec = {
            'roi': roi_name, 'target': target_col,
            'unit': unit, 'semantics': semantics,
            'max_trials': max_trials_n, 'perm_seed': perm_seed,
            'sided': sided,
            'n_subjects': len(sid_order), 'n_trials': int(len(y)),
            'n_channels': int(X.shape[1]),
            'seeds': [int(s) for s in seeds_cell], 'n_perm': int(n_perm),
            'actual_rs': [float(v) for v in actual],
            'actual_r_mean': float(np.mean(actual)),
            'actual_r_std': float(np.std(actual, ddof=1)) if S > 1 else 0.0,
            'actual_r_mean_first5': float(np.mean(actual[:min(5, S)])),
            'paper_actual_r_mean': PAPER_ACTUAL_R_MEAN.get(key),
            'agg': full,
            'by_seed_count': per_S,
            'runtime_s': round(time.time() - t0, 2),
        }
        out.append(rec)
        print(f"  [cell] {roi_name:10s} {target_col:26s} "
              f"r_mean={rec['actual_r_mean']:+.5f} "
              f"T_obs={full['T_obs']:+.5f} p_agg={full['p_agg']:.3f} | "
              f"p_single={full['p_single']:.3f} p_min={full['p_min']:.3f} "
              f"p_mean={full['p_mean']:.3f} ({rec['runtime_s']}s)")
        yield list(out)


# ============================================================
# 子分析 2：真零复制下的第一类错误率校准
# ------------------------------------------------------------
# 真零：把合并 trial 池的 rating 标签整体置换一次，得到 y_r。
# 在 H0 下，每一个"标签置换"给出的 seed 统计向量
#     B_j = ( T_1(y_j), ..., T_S(y_j) ),  y_j = pi_j(y), pi_j 均匀随机
# 都是同一分布 (T_1(pi(y)), ..., T_S(pi(y))) 的独立实现，且与"观测"
# 统计向量交换（观测只是其中一个均匀置换）。因此可先构建一个"块池"
# {B_j}，再对块做重采样来蒙特卡洛模拟每个复制内的置换检验
# （观测块 1 个 + 零块 B 个），从而在不逐复制重复 Sx(B+1) 次解码的
# 前提下得到高重复数下的经验第一类错误率。该重采样与逐复制置换在
# H0 下同分布，属于合法的蒙特卡洛等价实现。
# ============================================================
def _block_chunk(X, split_idx, max_trials, seeds, perm_ys_chunk, j0,
                 semantics='pool'):
    """(worker) 对一批标签置换，各 seed 求一次统计量。"""
    S = len(seeds)
    out = np.empty((len(perm_ys_chunk), S), float)
    for i, py in enumerate(perm_ys_chunk):
        j = j0 + i
        for si, s in enumerate(seeds):
            out[i, si] = _one_run(X, py, split_idx, max_trials,
                                  int(s) + j + 1, semantics)
    return out


def build_blocks(X, y, split_idx, seeds, n_blocks, max_trials, rng, n_jobs,
                 chunk=None, unit='pooled', semantics='pool'):
    """生成 n_blocks 个"标签置换块"：每块给出各 seed 的统计量 (n_blocks, S)。

    unit/semantics 与 cell 阶段同口径（§1.3/§2）；pooled 路径与旧实现逐位一致。
    """
    par, delayed = _parallel(n_jobs)
    perms = make_permuted_ys(y, split_idx, n_blocks, unit=unit, rng=rng)
    if chunk is None:
        chunk = max(1, int(np.ceil(n_blocks / max(1, n_jobs))))
    tasks = [(j0, perms[j0:j0 + chunk]) for j0 in range(0, n_blocks, chunk)]
    vals = par(delayed(_block_chunk)(X, split_idx, max_trials, seeds, sub,
                                     j0, semantics)
               for (j0, sub) in tasks)
    return np.concatenate(vals, axis=0)


def simulate_calibration(blocks, n_perm, n_rep, alpha, seed_counts, rng):
    """对块池做蒙特卡洛置换检验，估计各规则/各 seed 数的经验第一类错误率。

    同时统计单侧与双侧两种方向口径的 FPR（§1.2 要求方向事先固定，
    但校准表须同时给出两种口径，才能证明所选口径确实校准良好）。
    """
    P, S = blocks.shape
    Ss = [k for k in seed_counts if k <= S]
    rules = ['single', 'min', 'mean', 'agg',
             'single_two', 'agg_two']
    acc = {k: {r: 0 for r in rules} for k in Ss}
    sum_p = {k: {r: 0.0 for r in rules} for k in Ss}
    rows = []
    for _ in range(n_rep):
        idx = rng.choice(P, size=n_perm + 1, replace=False)
        obs = blocks[idx[0]]
        nul = blocks[idx[1:]]
        rec = {}
        for k in Ss:
            obsk = obs[:k]
            nullk = nul[:, :k]
            ps = np.array([pval_correct(nullk[:, s], obsk[s])
                           for s in range(k)])
            ps2 = np.array([pval_correct_two(nullk[:, s], obsk[s])
                            for s in range(k)])
            p_agg = pval_correct(nullk.mean(axis=1), float(obsk.mean()))
            p_agg2 = pval_correct_two(nullk.mean(axis=1), float(obsk.mean()))
            rec[str(k)] = {'T_obs': float(obsk.mean()),
                           'p_single': float(ps[0]),
                           'p_min': float(ps.min()),
                           'p_mean': float(ps.mean()),
                           'p_agg': float(p_agg),
                           'p_agg_two_sided': float(p_agg2)}
            acc[k]['single'] += int(ps[0] <= alpha)
            acc[k]['min'] += int(ps.min() <= alpha)
            acc[k]['mean'] += int(ps.mean() <= alpha)
            acc[k]['agg'] += int(p_agg <= alpha)
            acc[k]['single_two'] += int(ps2[0] <= alpha)
            acc[k]['agg_two'] += int(p_agg2 <= alpha)
            sum_p[k]['single'] += ps[0]
            sum_p[k]['min'] += ps.min()
            sum_p[k]['mean'] += ps.mean()
            sum_p[k]['agg'] += p_agg
            sum_p[k]['single_two'] += ps2[0]
            sum_p[k]['agg_two'] += p_agg2
        rows.append(rec)
    return acc, sum_p, rows, Ss


def run_calib(logm_all, subject_ids, ch_names, seeds, n_perm, n_rep,
              calib_cell, n_jobs, n_blocks,
              max_trials=MAX_TRIALS, unit='pooled', semantics='pool'):
    roi_name, target_col = calib_cell
    X, y, split_idx, sid_order = pool_cell(
        logm_all, target_col, roi_name, subject_ids, ch_names)
    Ss = [k for k in CALIB_SEED_COUNTS if k <= len(seeds)]
    rng = np.random.RandomState(20240903)
    t0 = time.time()
    print(f'  [calib] 构建块池: n_blocks={n_blocks} '
          f'(每块 {len(seeds)} 次解码, 共 {n_blocks * len(seeds)}) ...')
    blocks = build_blocks(X, y, split_idx, seeds, n_blocks, max_trials, rng,
                          n_jobs, unit=unit, semantics=semantics)
    t_build = time.time() - t0
    print(f'  [calib] 块池完成 {blocks.shape} ({t_build:.0f}s)'
          f', 模拟 n_rep={n_rep} n_perm={n_perm} ...')
    acc, sum_p, rows, Ss = simulate_calibration(
        blocks, n_perm, n_rep, ALPHA, CALIB_SEED_COUNTS, rng)
    yield _calib_snapshot(roi_name, target_col, sid_order, seeds, n_perm,
                          n_rep, Ss, acc, sum_p, rows, time.time() - t0,
                          n_blocks, t_build, unit, semantics, max_trials)


def _calib_snapshot(roi_name, target_col, sid_order, seeds, n_perm, n_rep,
                    Ss, acc, sum_p, rows, elapsed, n_blocks, t_build,
                    unit='pooled', semantics='pool', max_trials=MAX_TRIALS):
    """组装第一类错误率快照。"""
    by_S = {}
    for k in Ss:
        fpr = {rule: acc[k][rule] / n_rep for rule in acc[k]}
        se = {rule: float(np.sqrt(fpr[rule] * (1 - fpr[rule]) / n_rep))
              for rule in fpr}
        meanp = {rule: sum_p[k][rule] / n_rep for rule in sum_p[k]}
        by_S[str(k)] = {'n_seeds': k, 'fpr': fpr, 'fpr_se': se,
                        'mean_p': meanp}
    return {
        'cell': {'roi': roi_name, 'target': target_col},
        'unit': unit, 'semantics': semantics,
        'max_trials': _norm_budget(max_trials),
        'n_subjects': len(sid_order), 'n_rep': n_rep, 'n_perm': n_perm,
        'n_done': n_rep,
        'seeds': [int(s) for s in seeds], 'alpha': ALPHA,
        'nominal': ALPHA, 'by_seed_count': by_S,
        'n_blocks': n_blocks, 'build_runtime_s': round(t_build, 2),
        'design': 'block-resampling Monte-Carlo equivalent of '
                  'per-rep permutation under H0',
        'runtime_s': round(elapsed, 2), 'per_rep': rows,
    }


def _calib_sufficient(rec, unit, semantics, max_trials_n, n_perm, n_rep,
                      n_blocks):
    """断点续跑：已有校准结果与本次参数一致且充分则跳过。"""
    if not rec:
        return False
    if rec.get('unit', 'pooled') != unit or \
            rec.get('semantics', 'pool') != semantics:
        return False
    if _norm_budget(rec.get('max_trials', MAX_TRIALS)) != max_trials_n:
        return False
    # 旧版代码产出的校准缺 single_two/agg_two 双侧规则（§3.1 校准表须
    # 同时给出两种方向口径），视为不充分 → 强制重算
    for _sd in (rec.get('by_seed_count') or {}).values():
        if 'agg_two' not in (_sd.get('fpr') or {}) or \
                'agg_two' not in (_sd.get('mean_p') or {}):
            return False
    return (rec.get('n_perm', 0) >= n_perm and rec.get('n_rep', 0) >= n_rep
            and rec.get('n_blocks', 0) >= n_blocks)


# ============================================================
# 主入口
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', default='all',
                    choices=['cell', 'calib', 'all'])
    ap.add_argument('--seeds', type=int, default=10,
                    help='使用 SEED_POOL 前 N 个 seed（前 5 个=论文 seed）')
    ap.add_argument('--perm', type=int, default=100)
    ap.add_argument('--reps', type=int, default=200)
    ap.add_argument('--blocks', type=int, default=500,
                    help='校准块池大小（每块 S 次解码）')
    ap.add_argument('--calib-cell', type=str, default='occipital|rating_taste')
    ap.add_argument('--out', type=str, default='', help='输出 JSON 路径（默认内置）')
    ap.add_argument('--n_jobs', type=int, default=0, help='0=自动(cpu-4)')
    ap.add_argument('--nsub', type=int, default=0, help='>0 限制被试数（自检）')
    ap.add_argument('--cells', type=str, default='',
                    help="逗号分隔 roi|target 过滤；空=内置默认。"
                         "支持家族别名：roi_family（21 个 ROI×target，"
                         "§1.4 正式检验家族）、scalp（全脑 3 个 target）；"
                         "别名可与显式 cell 混用")
    ap.add_argument('--sided', choices=['one', 'two'], default='one',
                    help="正式 p 值的方向口径（§1.2 要求事先固定）："
                         "one=单侧上尾（正向单侧假设），two=双侧 |null|>=|obs|。"
                         "两种口径的 p 都会写入结果，此参数只决定正式 p")
    ap.add_argument('--window', choices=list(TIME_WINDOWS), default=WINDOW_NAME,
                    help='分析时间窗（§1.4 全脑 18-cell 家族 = 6 窗 × 3 target）。'
                         '不同窗口请配独立 --out 文件，混写会被拒绝')
    ap.add_argument('--max-seeds-scalp', type=int, default=0,
                    help="全脑单元的 seed 数上限（0=内置默认 5）。"
                         "全脑 per_subject 预算下单次解码约 347s，"
                         "提高上限会显著放大总耗时")
    ap.add_argument('--unit', choices=['pooled', 'within'], default='pooled',
                    help='置换单位（§1.3）：整池 pooled / 被试段内 within')
    ap.add_argument('--semantics', choices=['pool', 'per_subject'],
                    default='pool',
                    help='预算语义（§2）：总预算 pool / 每被试预算 per_subject')
    ap.add_argument('--max-trials', type=str, default=str(MAX_TRIALS),
                    help="训练预算数值；'inf'=配置C（全部训练 trial）")
    ap.add_argument('--selftest', action='store_true')
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(line_buffering=True, write_through=True)
    except (AttributeError, OSError):
        pass

    n_jobs = args.n_jobs if args.n_jobs > 0 else max(
        1, (os.cpu_count() or 4) - 4)
    seeds = SEED_POOL[:args.seeds]
    subject_ids = load_subject_list()
    if args.selftest:
        subject_ids = subject_ids[:5]
        args.seeds = 3
        seeds = SEED_POOL[:3]
        args.perm = 5
        args.reps = 2
        args.blocks = 30
        n_jobs = 1
        print('[selftest] 5 被试 / 3 seed / perm=5 / reps=2 / n_jobs=1')
    if args.nsub > 0:
        subject_ids = subject_ids[:args.nsub]

    mt_raw = str(args.max_trials).strip().lower()
    max_trials = INF_BUDGET if mt_raw in ('inf', 'infinity') else int(mt_raw)
    max_trials_n = _norm_budget(max_trials)

    window_name = args.window
    time_window = TIME_WINDOWS[window_name]

    print('=' * 72)
    print('随机种子推断审计 (audit_seed_inference)')
    print(f'  stage={args.stage}  n_seeds={len(seeds)}  perm={args.perm} '
          f'reps={args.reps}  n_jobs={n_jobs}')
    print(f'  unit={args.unit}  semantics={args.semantics}  '
          f'max_trials={max_trials_n}')
    print(f'  被试={len(subject_ids)}  窗口={window_name}')
    print('=' * 72)

    t0 = time.time()
    full_cache = preload_full(subject_ids)
    logm_all = compute_logm_all(full_cache, time_window)
    ch_names = detect_channel_names(subject_ids)
    print(f'[feat] logm 预计算完成: {len(logm_all)} 被试, {time.time()-t0:.1f}s\n')

    result = {
        'meta': {
            'window': window_name,
            'seed_pool': [int(s) for s in SEED_POOL],
            'seeds_used': [int(s) for s in seeds],
            'n_perm': args.perm, 'n_rep': args.reps,
            'max_trials': max_trials_n, 'unit': args.unit,
            'semantics': args.semantics,
            'n_subjects': len(subject_ids), 'alpha': ALPHA, 'n_jobs': n_jobs,
            'permutation': f'{args.unit}-label-shuffle (shared across seeds)',
            'statistic': ('mean over seeds of LOSO mean Pearson r '
                          f'({args.semantics}-budget, '
                          f'max_trials={max_trials_n})'),
            'sided_primary': args.sided,
            'pconvention_primary': ('p=(1+#{null>=obs})/(B+1) (单侧上尾)'
                                    if args.sided == 'one' else
                                    'p=(1+#{abs(null)>=abs(obs)})/(B+1) (双侧)'),
            'pconvention_also_recorded': (
                'p_agg_one_sided 与 p_agg_two_sided 同时写入；'
                '正式 p 由 --sided 事先固定，事后不得更改'),
            'pconvention_legacy': 'fraction(null>=obs) (原稿口径)',
        },
        'cells': {f'{c[0]}|{c[1]}': {'roi': c[0], 'target': c[1]}
                  for c in CELLS},
    }

    out_path = Path(args.out) if args.out else OUT_JSON
    if args.selftest and not args.out:
        # selftest 只写独立草稿文件，避免污染真实审计结果
        out_path = RESULTS_DIR / 'seed_inference_audit_selftest.json'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        try:
            with open(out_path, encoding='utf-8') as f:
                _prev = json.load(f)
            _prev_win = (_prev.get('meta') or {}).get('window', WINDOW_NAME)
            if _prev_win != window_name:
                raise SystemExit(
                    f'[error] {out_path} 已有结果的 window='
                    f'{_prev_win!r} 与本次 --window {window_name!r} 不符；'
                    '不同窗口必须用独立 --out 文件，禁止混写')
            for _k in ('cell', 'calib'):
                if _k in _prev:
                    result[_k] = _prev[_k]
        except SystemExit:
            raise
        except Exception:
            pass

    # cell 记录统一为 dict（按 roi|target 键合并，修复旧版整体覆盖 bug）
    cell_recs = result.get('cell', {})
    if isinstance(cell_recs, list):
        cell_recs = {f"{r['roi']}|{r['target']}": r for r in cell_recs}
    result['cell'] = cell_recs
    _dump(result, out_path)

    if args.stage in ('cell', 'all'):
        print('--- 子分析 1: 观测数据聚合统计量 / 规则对照 / 种子数敏感性 ---')
        cells = None
        if args.cells.strip():
            want = [c.strip() for c in args.cells.split(',') if c.strip()]
            # 家族别名展开（§1.4 正式检验家族 = 21 个 ROI×target）
            expanded = []
            for w in want:
                if w == 'roi_family':
                    expanded += [f'{a}|{b}' for a, b in ROI_FAMILY_CELLS]
                elif w == 'scalp':
                    expanded += [f'{a}|{b}' for a, b in SCALP_CELLS]
                else:
                    expanded.append(w)
            # 在完整注册表（ROI 家族 21 + 全脑 3）上过滤，顺序按注册表
            cells = [c for c in REGISTRY_CELLS
                     if f'{c[0]}|{c[1]}' in set(expanded)]
            missing = [w for w in expanded
                       if w not in {f'{a}|{b}' for a, b in REGISTRY_CELLS}]
            if missing:
                raise SystemExit(f'[error] 未知 cell: {missing}；'
                                 f'合法名称见 ROI_FAMILY_CELLS / SCALP_CELLS')
            if not cells:
                raise SystemExit('[error] --cells 过滤后为空')
            print(f'  [filter] {len(cells)} 个 cell: '
                  f'{[f"{a}|{b}" for a, b in cells]}')
        for part in run_cell(
                logm_all, subject_ids, ch_names, seeds, args.perm, n_jobs,
                cells, max_trials=max_trials, unit=args.unit,
                semantics=args.semantics, existing=cell_recs,
                sided=args.sided,
                max_seeds_scalp=(args.max_seeds_scalp
                                 if args.max_seeds_scalp > 0 else None)):
            for rec in part:
                cell_recs[f"{rec['roi']}|{rec['target']}"] = rec
            result['cell'] = cell_recs
            _dump(result, out_path)
        print()

    if args.stage in ('calib', 'all'):
        print('--- 子分析 2: 真零复制第一类错误率校准 ---')
        if _calib_sufficient(result.get('calib'), args.unit, args.semantics,
                             max_trials_n, args.perm, args.reps, args.blocks):
            print('  [skip] 校准已有充分结果（断点续跑）')
        else:
            cell = tuple(args.calib_cell.split('|'))
            if args.blocks < args.perm + 1:
                args.blocks = args.perm + 1
                print(f'  [auto] n_blocks 调整为 perm+1={args.blocks} '
                      '（每复制需无放回抽 n_perm+1 块）')
            for part in run_calib(
                    logm_all, subject_ids, ch_names, seeds, args.perm,
                    args.reps, cell, n_jobs, args.blocks,
                    max_trials=max_trials, unit=args.unit,
                    semantics=args.semantics):
                result['calib'] = part
                _dump(result, out_path)
        print()

    result['meta']['total_runtime_s'] = round(time.time() - t0, 1)
    _dump(result, out_path)
    print(f'[saved] {out_path}')
    print(f'[total] {result["meta"]["total_runtime_s"]}s')


if __name__ == '__main__':
    main()
