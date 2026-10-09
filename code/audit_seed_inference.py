# -*- coding: utf-8 -*-
"""
audit_seed_inference.py
=======================
Cell Reports Methods 转投核心新分析：单试次 EEG 解码的随机种子推断审计。

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

子分析
------
cell   观测数据：(a) 种子聚合统计量与规则对照；(b) 种子数敏感性 S'∈{1,2,3,5,10}。
calib  真零复制（合并标签池置换）下的第一类错误率校准（代表单元）。

用法
----
    python code/audit_seed_inference.py --selftest
    python code/audit_seed_inference.py --stage cell  --perm 100 --seeds 10 --n_jobs 20
    python code/audit_seed_inference.py --stage calib --reps 200 --perm 100 --seeds 10 \
        --calib-cell occipital|rating_taste --n_jobs 20
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

# 10 个 seed：前 5 个与论文/原稿一致，便于 5-seed 结果直接对齐原稿
SEED_POOL = [42, 123, 456, 789, 2024, 7, 99, 555, 31415, 2718]
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

# 原稿跨 seed 平均口径下 full-window 的报告值，用于流水线一致性核对
PAPER_ACTUAL_R_MEAN = {
    'all|rating_taste': 0.008774,
    'all|rating_health': 0.004756,
    'all|rating_willingnessToEat': 0.006303,
}

MAX_TRIALS = 200
ALPHA = 0.05
CALIB_SEED_COUNTS = [1, 5, 10]

# 全脑 64 通道单次解码约为枕区的 ~27 倍，故全脑单元只用前 5 个论文 seed，
# 以控制总耗时；枕区单元用满 10 个 seed 以支撑 seed 数敏感性曲线。
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
def _one_run(X, y, split_idx, max_trials, random_state):
    """单次 _loso_mean_r（模块级，便于 joblib 序列化）。"""
    return _loso_mean_r(X, y, split_idx, max_trials=max_trials,
                        random_state=int(random_state))


def _parallel(n_jobs):
    from joblib import Parallel, delayed
    if n_jobs is None or n_jobs <= 0:
        n_jobs = max(1, (os.cpu_count() or 4) - 4)
    return Parallel(n_jobs=n_jobs, backend='loky'), delayed


def _run_perm_chunk(X, split_idx, max_trials, seed, b_start, perm_ys_chunk):
    """(worker) 对一个 seed 的一批共享置换求统计量。

    同一任务内复用同一份 X（joblib 对大数组做 memmap），
    按 chunk 摊薄数据搬运开销 —— 全脑 64 通道 X≈0.75GB，逐置换派发会被
    memmap 反复搬运拖垮；分批后开销可忽略。
    """
    return np.asarray([
        _one_run(X, py, split_idx, max_trials, int(seed) + (b_start + j) + 1)
        for j, py in enumerate(perm_ys_chunk)], float)


def per_seed_shared_perm(X, y, split_idx, seeds, n_perm, max_trials,
                         perm_rng, n_jobs=1, chunk=None):
    """给定 (X,y)：每 seed 的 actual_r，以及共享置换下的统计量矩阵。

    共享置换：B 个标签置换只抽一次并施加到所有 seed，使
    T^(b) = mean_s T_s^(b) 成为聚合统计量的合法零实现。

    置换任务按 chunk 分批（默认使任务数≈n_jobs），以摊薄大数组传输开销。
    返回 (actual_rs (S,), stat (S,B))。
    """
    par, delayed = _parallel(n_jobs)
    S = len(seeds)
    actual = np.asarray(par(
        delayed(_one_run)(X, y, split_idx, max_trials, int(s))
        for s in seeds), float)

    perm_ys = [perm_rng.permutation(y) for _ in range(n_perm)]
    if chunk is None:
        chunk = max(1, int(np.ceil(S * n_perm / max(1, n_jobs))))
    tasks = []
    for si, s in enumerate(seeds):
        for b0 in range(0, n_perm, chunk):
            tasks.append((si, int(s), b0, perm_ys[b0:b0 + chunk]))
    vals = par(
        delayed(_run_perm_chunk)(X, split_idx, max_trials, s, b0, sub)
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


def pval_legacy(null_vals, obs):
    """原稿口径：fraction(null >= obs)（保留以便与旧稿数值对照）。"""
    return float(np.mean(null_vals >= obs))


def per_seed_pvals(actual, stat, convention='correct'):
    """每 seed 各自零分布下的置换 p。"""
    fn = pval_correct if convention == 'correct' else pval_legacy
    S, B = stat.shape
    return np.array([fn(stat[s, :], actual[s]) for s in range(S)])


def eval_rules(actual, stat, seed_idx):
    """给定选中的 seed 行，计算聚合统计量、聚合 p 与三类对照规则。"""
    idx = list(seed_idx)
    a = actual[idx]
    T_obs = float(np.mean(a))
    T_null = stat[idx, :].mean(axis=0)
    p_agg = pval_correct(T_null, T_obs)
    ps_c = per_seed_pvals(actual, stat, 'correct')[idx]
    ps_l = per_seed_pvals(actual, stat, 'legacy')[idx]
    return {
        'T_obs': T_obs,
        'T_null_mean': float(np.mean(T_null)),
        'T_null_std': float(np.std(T_null, ddof=1)) if len(T_null) > 1 else 0.0,
        'T_null_95ci': [float(np.percentile(T_null, 2.5)),
                        float(np.percentile(T_null, 97.5))],
        'p_agg': p_agg,
        'p_single': float(ps_c[0]),
        'p_min': float(np.min(ps_c)),
        'p_mean': float(np.mean(ps_c)),
        'p_median': float(np.median(ps_c)),
        'per_seed_p': [float(x) for x in ps_c],
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
def run_cell(logm_all, subject_ids, ch_names, seeds, n_perm, n_jobs, cells=None):
    out = []
    for roi_name, target_col in (cells or CELLS):
        seeds_cell = seeds if roi_name != 'all' else seeds[:MAX_SEEDS_WHOLESCALP]
        S = len(seeds_cell)
        X, y, split_idx, sid_order = pool_cell(
            logm_all, target_col, roi_name, subject_ids, ch_names)
        perm_rng = np.random.RandomState(20240901)
        t0 = time.time()
        actual, stat = per_seed_shared_perm(
            X, y, split_idx, seeds_cell, n_perm, MAX_TRIALS, perm_rng, n_jobs)
        full = eval_rules(actual, stat, range(S))

        per_S = {}
        for Ssize in [1, 2, 3, 5, 10]:
            if Ssize > S:
                continue
            combos = list(combinations(range(S), Ssize))
            if len(combos) > 300:
                combos = combos[:300]
            Tob = np.array([eval_rules(actual, stat, c)['T_obs']
                            for c in combos])
            pp = np.array([eval_rules(actual, stat, c)['p_agg']
                           for c in combos])
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
def _block_chunk(X, split_idx, max_trials, seeds, perm_ys_chunk, j0):
    """(worker) 对一批标签置换，各 seed 求一次统计量。"""
    S = len(seeds)
    out = np.empty((len(perm_ys_chunk), S), float)
    for i, py in enumerate(perm_ys_chunk):
        j = j0 + i
        for si, s in enumerate(seeds):
            out[i, si] = _one_run(X, py, split_idx, max_trials, int(s) + j + 1)
    return out


def build_blocks(X, y, split_idx, seeds, n_blocks, max_trials, rng, n_jobs,
                 chunk=None):
    """生成 n_blocks 个"标签置换块"：每块给出各 seed 的统计量 (n_blocks, S)。"""
    par, delayed = _parallel(n_jobs)
    perms = [rng.permutation(y) for _ in range(n_blocks)]
    if chunk is None:
        chunk = max(1, int(np.ceil(n_blocks / max(1, n_jobs))))
    tasks = [(j0, perms[j0:j0 + chunk]) for j0 in range(0, n_blocks, chunk)]
    vals = par(delayed(_block_chunk)(X, split_idx, max_trials, seeds, sub, j0)
               for (j0, sub) in tasks)
    return np.concatenate(vals, axis=0)


def simulate_calibration(blocks, n_perm, n_rep, alpha, seed_counts, rng):
    """对块池做蒙特卡洛置换检验，估计各规则/各 seed 数的经验第一类错误率。"""
    P, S = blocks.shape
    Ss = [k for k in seed_counts if k <= S]
    acc = {k: {'single': 0, 'min': 0, 'mean': 0, 'agg': 0} for k in Ss}
    sum_p = {k: {'single': 0.0, 'min': 0.0, 'mean': 0.0, 'agg': 0.0}
             for k in Ss}
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
            p_agg = pval_correct(nullk.mean(axis=1), float(obsk.mean()))
            rec[str(k)] = {'T_obs': float(obsk.mean()),
                           'p_single': float(ps[0]),
                           'p_min': float(ps.min()),
                           'p_mean': float(ps.mean()),
                           'p_agg': float(p_agg)}
            acc[k]['single'] += int(ps[0] <= alpha)
            acc[k]['min'] += int(ps.min() <= alpha)
            acc[k]['mean'] += int(ps.mean() <= alpha)
            acc[k]['agg'] += int(p_agg <= alpha)
            sum_p[k]['single'] += ps[0]
            sum_p[k]['min'] += ps.min()
            sum_p[k]['mean'] += ps.mean()
            sum_p[k]['agg'] += p_agg
        rows.append(rec)
    return acc, sum_p, rows, Ss


def run_calib(logm_all, subject_ids, ch_names, seeds, n_perm, n_rep,
              calib_cell, n_jobs, n_blocks):
    roi_name, target_col = calib_cell
    X, y, split_idx, sid_order = pool_cell(
        logm_all, target_col, roi_name, subject_ids, ch_names)
    Ss = [k for k in CALIB_SEED_COUNTS if k <= len(seeds)]
    rng = np.random.RandomState(20240903)
    t0 = time.time()
    print(f'  [calib] 构建块池: n_blocks={n_blocks} '
          f'(每块 {len(seeds)} 次解码, 共 {n_blocks * len(seeds)}) ...')
    blocks = build_blocks(X, y, split_idx, seeds, n_blocks, MAX_TRIALS, rng,
                          n_jobs)
    t_build = time.time() - t0
    print(f'  [calib] 块池完成 {blocks.shape} ({t_build:.0f}s)'
          f', 模拟 n_rep={n_rep} n_perm={n_perm} ...')
    acc, sum_p, rows, Ss = simulate_calibration(
        blocks, n_perm, n_rep, ALPHA, CALIB_SEED_COUNTS, rng)
    yield _calib_snapshot(roi_name, target_col, sid_order, seeds, n_perm,
                          n_rep, Ss, acc, sum_p, rows, time.time() - t0,
                          n_blocks, t_build)


def _calib_snapshot(roi_name, target_col, sid_order, seeds, n_perm, n_rep,
                    Ss, acc, sum_p, rows, elapsed, n_blocks, t_build):
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
        'n_subjects': len(sid_order), 'n_rep': n_rep, 'n_perm': n_perm,
        'n_done': n_rep,
        'seeds': [int(s) for s in seeds], 'alpha': ALPHA,
        'nominal': ALPHA, 'by_seed_count': by_S,
        'n_blocks': n_blocks, 'build_runtime_s': round(t_build, 2),
        'design': 'block-resampling Monte-Carlo equivalent of '
                  'per-rep permutation under H0',
        'runtime_s': round(elapsed, 2), 'per_rep': rows,
    }


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
                    help='逗号分隔 roi|target 过滤；空=全部')
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

    print('=' * 72)
    print('随机种子推断审计 (audit_seed_inference)')
    print(f'  stage={args.stage}  seeds={seeds}  perm={args.perm} '
          f'reps={args.reps}  n_jobs={n_jobs}')
    print(f'  被试={len(subject_ids)}  窗口={WINDOW_NAME}')
    print('=' * 72)

    t0 = time.time()
    full_cache = preload_full(subject_ids)
    logm_all = compute_logm_all(full_cache, TIME_WINDOW)
    ch_names = detect_channel_names(subject_ids)
    print(f'[feat] logm 预计算完成: {len(logm_all)} 被试, {time.time()-t0:.1f}s\n')

    result = {
        'meta': {
            'window': WINDOW_NAME,
            'seed_pool': [int(s) for s in SEED_POOL],
            'seeds_used': [int(s) for s in seeds],
            'n_perm': args.perm, 'n_rep': args.reps, 'max_trials': MAX_TRIALS,
            'n_subjects': len(subject_ids), 'alpha': ALPHA, 'n_jobs': n_jobs,
            'permutation': 'pooled-label-shuffle (shared across seeds)',
            'statistic': 'mean over seeds of pooled-budget LOSO mean Pearson r',
            'pconvention_primary': 'p=(1+#{null>=obs})/(B+1)',
            'pconvention_legacy': 'fraction(null>=obs) (原稿口径)',
        },
        'cells': {f'{c[0]}|{c[1]}': {'roi': c[0], 'target': c[1]}
                  for c in CELLS},
    }

    out_path = Path(args.out) if args.out else OUT_JSON
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        try:
            with open(out_path, encoding='utf-8') as f:
                _prev = json.load(f)
            for _k in ('cell', 'calib'):
                if _k in _prev:
                    result[_k] = _prev[_k]
        except Exception:
            pass
    _dump(result, out_path)

    if args.stage in ('cell', 'all'):
        print('--- 子分析 1: 观测数据聚合统计量 / 规则对照 / 种子数敏感性 ---')
        cells = None
        if args.cells.strip():
            want = [c.strip() for c in args.cells.split(',') if c.strip()]
            cells = [c for c in CELLS if f'{c[0]}|{c[1]}' in want]
            print(f'  [filter] cells={[f"{a}|{b}" for a, b in cells]}')
        for part in run_cell(
                logm_all, subject_ids, ch_names, seeds, args.perm, n_jobs,
                cells):
            result['cell'] = part
            _dump(result, out_path)
        print()

    if args.stage in ('calib', 'all'):
        print('--- 子分析 2: 真零复制第一类错误率校准 ---')
        cell = tuple(args.calib_cell.split('|'))
        for part in run_calib(
                logm_all, subject_ids, ch_names, seeds, args.perm, args.reps,
                cell, n_jobs, args.blocks):
            result['calib'] = part
            _dump(result, out_path)
        print()

    result['meta']['total_runtime_s'] = round(time.time() - t0, 1)
    _dump(result, out_path)
    print(f'[saved] {out_path}')
    print(f'[total] {result["meta"]["total_runtime_s"]}s')


if __name__ == '__main__':
    main()
