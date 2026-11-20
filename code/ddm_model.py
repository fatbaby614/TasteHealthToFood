# -*- coding: utf-8 -*-
"""
DDM (Drift Diffusion Model) 拟合：taste / health 对食物决策动力学的影响

用 hssm (PyMC) 实现，目标：
1. 提取每个被试的个体 DDM 参数：v (drift rate), a (threshold), t (non-decision time)
2. 拟合回归 DDM：drift rate 受 taste_diff + health_diff 调制
   → 量化 tastiness 与 healthiness 对证据积累速率的贡献
3. 高/低 SCSR 组对比：taste/health 漂移权重是否有差异

科学问题（对应切入点 1 + 2）：
- taste 是否比 health 更快驱动决策（drift rate 权重更大）？
  → 验证 Sullivan (2015) "tastiness 先于 healthiness" 的属性加工假说
- 个体差异：低 SCSR 被试是否 taste 漂移权重 >> health？
- DDM 参数可作为后续 EEG 神经相关分析的目标变量
"""

import os
os.environ['PYTHONWARNINGS'] = 'ignore'

# ---- 兼容性垫片：Python 3.12+ 移除了 pkgutil.ImpImporter，而老版本 setuptools 的
# pkg_resources 在模块级引用它（register_finder(pkgutil.ImpImporter, ...)），
# 会在拟合期懒加载时触发 AttributeError。必须在 import hssm/pymc 之前注入占位类。
# 治本方案：pip install -U setuptools wheel
import pkgutil

if not hasattr(pkgutil, 'ImpImporter'):
    class _ImpImporterCompat:
        pass
    pkgutil.ImpImporter = _ImpImporterCompat
    print('[ddm] 已注入 pkgutil.ImpImporter 兼容垫片（Python 3.12+ + 老 setuptools；建议 pip install -U setuptools）')

import numpy as np
import pandas as pd
import hssm
import pymc as pm
import arviz as az
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import json
import traceback
import warnings
warnings.filterwarnings('ignore')

# ---- 兼容性补丁：hssm 0.3.0 与 arviz 1.2 / xarray 新版不兼容 ----
# 1. _clean_posterior_group 用旧式列表索引 idata["posterior"][list]，新版 xarray 不支持。
# 2. log_likelihood 调用 idata.add_groups()，arviz 1.2 已废弃该方法。
# 将两者都改为 no-op：跳过后处理，不影响参数提取（v/a/t/z 仍在 posterior 中）。
def _noop_clean_posterior_group(self, idata):
    return None

def _noop_log_likelihood(self, idata=None, data=None, inplace=True, keep_likelihood_params=False):
    return None

hssm.HSSM._clean_posterior_group = _noop_clean_posterior_group
hssm.HSSM.log_likelihood = _noop_log_likelihood

from config import RESULTS_DIR, FIGURES_DIR, OUTPUT_DIR, merge_results_csv
from load_behavior import load_all_pairedchoice, compute_self_control_success


# ============ 数据准备 ============

def prepare_ddm_data():
    """准备 hssm 格式的数据

    hssm 要求列：
    - rt: 反应时（秒）
    - response: 0/1 二元选择（1 = chose stim1, 0 = chose stim2）
    - 可选协变量：taste_diff, health_diff

    返回
    ----
    df_all : 全体试次 DataFrame
    by_subject : dict, {subject_id: sub_df}
    """
    df = load_all_pairedchoice()

    # 过滤无效 RT（NaN 或 <=0 或过短 <100ms）
    df = df.dropna(subset=['respRT']).copy()
    df = df[df['respRT'] > 0.1].copy()  # <100ms 视为按键错误
    df = df[df['respRT'] < 10].copy()    # >10s 视为不注意

    # 转换为 hssm 需要的格式
    df['rt'] = df['respRT'].astype(float)
    # hssm DDM 默认 choices=[-1, 1]：1=upper boundary (chose stim1), -1=lower (chose stim2)
    df['response'] = df['chose_stim1'].map({1: 1, 0: -1}).astype(int)

    # 标准化协变量（中心化，便于先验设置）
    df['taste_diff_c'] = df['taste_diff'] - df['taste_diff'].mean()
    df['health_diff_c'] = df['health_diff'] - df['health_diff'].mean()

    print(f'[DDM] 准备数据: {len(df)} trials, {df.subject.nunique()} subjects')
    print(f'[DDM] RT 描述: mean={df.rt.mean():.3f}s, median={df.rt.median():.3f}s, std={df.rt.std():.3f}s')
    print(f'[DDM] accuracy (P(chose stim1)): {df["response"].mean():.3f} (response=2 means chose stim1)')

    by_subject = {s: g.copy() for s, g in df.groupby('subject')}
    return df, by_subject


# ============ 模型 1: 简单 DDM（每被试独立拟合）============

def fit_simple_ddm_per_subject(by_subject, n_subjects=None, quick_test=False):
    """对每个被试独立拟合简单 DDM，提取 v/a/t 个体参数

    参数
    ----
    by_subject : dict
    n_subjects : int, 拟合前 N 个被试（None=全部）
    quick_test : bool, True 时只跑 1 个被试、少量采样用于验证流程
    """
    if quick_test:
        n_subjects = 1

    subjects = list(by_subject.keys())[:n_subjects] if n_subjects else list(by_subject.keys())

    results = []
    sampler_kwargs = dict(
        cores=1,
        chains=2,
        draws=500,
        tune=500,
        target_accept=0.9,
        progressbar=False,
        # 跳过 log_likelihood 计算（hssm 0.3.0 与 arviz 1.2 不兼容：add_groups 已废弃）
        idata_kwargs={"log_likelihood": False},
    )

    for i, subj in enumerate(subjects):
        sub_df = by_subject[subj]
        if len(sub_df) < 50:
            print(f'[{i+1}/{len(subjects)}] {subj}: 试次过少 ({len(sub_df)})，跳过')
            continue
        try:
            print(f'[{i+1}/{len(subjects)}] {subj}: 拟合中... ({len(sub_df)} trials)')
            model = hssm.HSSM(
                data=sub_df[['rt', 'response']],
                model='ddm',
            )
            idata = model.sample(**sampler_kwargs)

            # 提取后验均值
            post = idata.posterior
            v_hat = float(post['v'].mean().values)
            a_hat = float(post['a'].mean().values)
            t_hat = float(post['t'].mean().values)

            # 收敛诊断
            rhat_v = float(az.rhat(post['v']).values)
            rhat_a = float(az.rhat(post['a']).values)
            rhat_t = float(az.rhat(post['t']).values)

            results.append({
                'subject': subj,
                'v': v_hat, 'a': a_hat, 't': t_hat,
                'rhat_v': rhat_v, 'rhat_a': rhat_a, 'rhat_t': rhat_t,
                'n_trials': len(sub_df),
                'converged': all(r < 1.1 for r in [rhat_v, rhat_a, rhat_t]),
            })
            print(f'    v={v_hat:.3f}, a={a_hat:.3f}, t={t_hat:.3f}, '
                  f'Rhat=[{rhat_v:.2f},{rhat_a:.2f},{rhat_t:.2f}]')

        except Exception as e:
            print(f'    拟合失败: {type(e).__name__}: {e}')
            print(traceback.format_exc())
            results.append({'subject': subj, 'error': f'{type(e).__name__}: {e}'})

    return pd.DataFrame(results)


# ============ 模型 2: 回归 DDM（drift rate 受 taste/health 调制）============

def fit_regression_ddm_per_subject(by_subject, n_subjects=None, quick_test=False):
    """对每个被试拟合回归 DDM：v = v_base + v_taste * taste_diff + v_health * health_diff

    关键参数：
    - v_base: 基线漂移率（被试的整体偏好倾向）
    - v_taste: taste_diff 对 drift rate 的回归系数（>0 = taste 驱动证据积累）
    - v_health: health_diff 对 drift rate 的回归系数
    - a, t: 与简单 DDM 相同

    hssm 支持通过 formula 参数指定 v 的回归
    """
    if quick_test:
        n_subjects = 1

    subjects = list(by_subject.keys())[:n_subjects] if n_subjects else list(by_subject.keys())

    results = []
    sampler_kwargs = dict(
        cores=1,
        chains=2,
        draws=500,
        tune=500,
        target_accept=0.9,
        progressbar=False,
        # 跳过 log_likelihood 计算（hssm 0.3.0 与 arviz 1.2 不兼容：add_groups 已废弃）
        idata_kwargs={"log_likelihood": False},
    )

    for i, subj in enumerate(subjects):
        sub_df = by_subject[subj]
        if len(sub_df) < 50:
            continue
        try:
            print(f'[reg-{i+1}/{len(subjects)}] {subj}: 拟合回归 DDM... ({len(sub_df)} trials)')
            model = hssm.HSSM(
                data=sub_df[['rt', 'response', 'taste_diff_c', 'health_diff_c']],
                model='ddm',
                include=[
                    dict(name='v',
                         formula='v ~ 1 + taste_diff_c + health_diff_c'),
                ],
            )
            idata = model.sample(**sampler_kwargs)

            # 提取回归系数
            post = idata.posterior
            # hssm 回归 DDM 系数命名：v_Intercept, v_taste_diff_c, v_health_diff_c
            v_base = float(post['v_Intercept'].mean().values)
            v_taste = float(post['v_taste_diff_c'].mean().values)
            v_health = float(post['v_health_diff_c'].mean().values)
            a_hat = float(post['a'].mean().values)
            t_hat = float(post['t'].mean().values)

            rhats = {
                'v_base': float(az.rhat(post['v_Intercept']).values),
                'v_taste': float(az.rhat(post['v_taste_diff_c']).values),
                'v_health': float(az.rhat(post['v_health_diff_c']).values),
                'a': float(az.rhat(post['a']).values),
                't': float(az.rhat(post['t']).values),
            }

            results.append({
                'subject': subj,
                'v_base': v_base, 'v_taste': v_taste, 'v_health': v_health,
                'a': a_hat, 't': t_hat,
                'n_trials': len(sub_df),
                'rhat_max': max(rhats.values()),
                'converged': all(r < 1.1 for r in rhats.values()),
                'rhats': rhats,
            })
            print(f'    v_base={v_base:.3f}, v_taste={v_taste:.4f}, v_health={v_health:.4f}, '
                  f'a={a_hat:.3f}, t={t_hat:.3f}, Rhat_max={max(rhats.values()):.2f}')

        except Exception as e:
            print(f'    拟合失败: {type(e).__name__}: {e}')
            print(traceback.format_exc())
            results.append({'subject': subj, 'error': f'{type(e).__name__}: {e}'})

    return pd.DataFrame(results)


# ============ 分析与可视化 ============

def summarize_ddm_results(simple_df, reg_df, scsr_df):
    """汇总 DDM 结果：组水平描述 + SCSR 相关 + 高低组对比"""
    summary = {}

    # 简单 DDM 描述
    if 'converged' in simple_df.columns:
        valid_simple = simple_df[simple_df['converged'] == True]
    else:
        valid_simple = simple_df
    if 'v' not in valid_simple.columns or len(valid_simple) == 0:
        summary['simple_ddm'] = {'n_converged': 0, 'note': 'no successful fits'}
    else:
        summary['simple_ddm'] = {
            'n_converged': int(len(valid_simple)),
            'v_mean': float(valid_simple['v'].mean()),
            'v_std': float(valid_simple['v'].std()),
            'a_mean': float(valid_simple['a'].mean()),
            'a_std': float(valid_simple['a'].std()),
            't_mean': float(valid_simple['t'].mean()),
            't_std': float(valid_simple['t'].std()),
        }

    # 回归 DDM 描述
    if 'converged' in reg_df.columns:
        valid_reg = reg_df[reg_df['converged'] == True]
    else:
        valid_reg = reg_df
    if 'v_taste' not in valid_reg.columns or len(valid_reg) == 0:
        summary['reg_ddm'] = {'n_converged': 0, 'note': 'no successful fits'}
        return summary

    summary['reg_ddm'] = {
        'n_converged': int(len(valid_reg)),
        'v_base_mean': float(valid_reg['v_base'].mean()),
        'v_taste_mean': float(valid_reg['v_taste'].mean()),
        'v_taste_std': float(valid_reg['v_taste'].std()),
        'v_health_mean': float(valid_reg['v_health'].mean()),
        'v_health_std': float(valid_reg['v_health'].std()),
    }

    # 关键统计：taste vs health 漂移权重
    from scipy import stats as sp_stats
    t_test = sp_stats.ttest_1samp(valid_reg['v_taste'], 0)
    h_test = sp_stats.ttest_1samp(valid_reg['v_health'], 0)
    taste_health_diff = valid_reg['v_taste'] - valid_reg['v_health']
    diff_test = sp_stats.ttest_1samp(taste_health_diff, 0)

    summary['reg_ddm']['taste_t'] = float(t_test.statistic)
    summary['reg_ddm']['taste_p'] = float(t_test.pvalue)
    summary['reg_ddm']['health_t'] = float(h_test.statistic)
    summary['reg_ddm']['health_p'] = float(h_test.pvalue)
    summary['reg_ddm']['taste_minus_health_t'] = float(diff_test.statistic)
    summary['reg_ddm']['taste_minus_health_p'] = float(diff_test.pvalue)
    summary['reg_ddm']['taste_minus_health_mean'] = float(taste_health_diff.mean())

    # SCSR 相关分析
    merged = valid_reg.merge(scsr_df, on='subject', how='inner')
    if len(merged) > 5:
        corrs = {}
        for param in ['v_taste', 'v_health', 'v_base', 'a', 't']:
            r, p = sp_stats.pearsonr(merged[param], merged['scsr'])
            corrs[param] = {'r': float(r), 'p': float(p)}
        summary['scsr_corr'] = corrs

        # 高/低 SCSR 组对比
        median_scsr = merged['scsr'].median()
        high = merged[merged['scsr'] >= median_scsr]
        low = merged[merged['scsr'] < median_scsr]
        group_cmp = {}
        for param in ['v_taste', 'v_health', 'v_base']:
            t_stat, p_val = sp_stats.ttest_ind(low[param], high[param])
            group_cmp[param] = {
                'low_mean': float(low[param].mean()),
                'high_mean': float(high[param].mean()),
                't': float(t_stat), 'p': float(p_val),
                'n_low': int(len(low)), 'n_high': int(len(high)),
            }
        summary['group_comparison'] = group_cmp

    return summary


def plot_ddm_results(simple_df, reg_df, summary, figdir):
    """DDM 结果可视化"""
    # 图1: 个体参数分布（v, a, t）
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, param, label in zip(axes, ['v', 'a', 't'], ['Drift rate v', 'Threshold a', 'Non-decision t (s)']):
        valid = simple_df[simple_df[param].notna()] if param in simple_df.columns else pd.DataFrame()
        if len(valid):
            ax.hist(valid[param].values, bins=20, color='steelblue',
                    edgecolor='white', alpha=0.8)
            ax.axvline(valid[param].mean(), color='red', linestyle='--',
                       label=f'mean={valid[param].mean():.3f}')
            ax.legend()
        ax.set_xlabel(label)
        ax.set_ylabel('Number of subjects')
        ax.set_title(f'{label} distribution')
    plt.tight_layout()
    plt.savefig(figdir / 'fig4_ddm_params_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'[fig] saved: {figdir / "fig4_ddm_params_distribution.png"}')

    # 图2: 回归 DDM - taste vs health 漂移权重
    # reg_df 全部拟合失败时只有 subject/error 列，此时跳过图 2/3
    if 'v_taste' not in reg_df.columns or not reg_df['v_taste'].notna().any():
        print('[fig] 回归 DDM 无成功拟合结果，跳过图 2/3')
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax = axes[0]
    valid = reg_df[reg_df['v_taste'].notna()]
    ax.hist(valid['v_taste'].values, bins=20, color='#E07B39', alpha=0.7, label='v_taste')
    ax.hist(valid['v_health'].values, bins=20, color='#3D7A8E', alpha=0.7, label='v_health')
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_xlabel('Drift rate regression coefficient')
    ax.set_ylabel('Number of subjects')
    ax.set_title('taste vs health drift weights\n(regression DDM)')
    ax.legend()

    ax = axes[1]
    x = np.arange(len(valid))
    ax.scatter(valid['v_taste'], valid['v_health'], alpha=0.6, edgecolor='black')
    ax.axhline(0, color='black', linewidth=0.5)
    ax.axvline(0, color='black', linewidth=0.5)
    lim = max(abs(valid['v_taste']).max(), abs(valid['v_health']).max()) * 1.1
    ax.plot([-lim, lim], [-lim, lim], 'r--', alpha=0.5, label='y=x (equal weights)')
    ax.set_xlabel('v_taste (taste drift weight)')
    ax.set_ylabel('v_health (health drift weight)')
    ax.set_title('Individual taste vs health drift weights')
    ax.legend()
    ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig(figdir / 'fig5_taste_vs_health_drift.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'[fig] saved: {figdir / "fig5_taste_vs_health_drift.png"}')

    # 图3: SCSR 与 DDM 参数相关
    if 'scsr_corr' in summary:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
        for ax, param in zip(axes, ['v_taste', 'v_health', 'v_base']):
            corrs = summary['scsr_corr'].get(param, {})
            r, p = corrs.get('r', 0), corrs.get('p', 1)
            valid_pair = reg_df.merge(
                compute_self_control_success(load_all_pairedchoice()),
                on='subject', how='inner'
            )
            ax.scatter(valid_pair['scsr'], valid_pair[param], alpha=0.6, edgecolor='black')
            ax.set_xlabel('SCSR')
            ax.set_ylabel(param)
            ax.set_title(f'{param} vs SCSR\nr={r:.3f}, p={p:.3f}')
            ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(figdir / 'fig6_scsr_ddm_corr.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f'[fig] saved: {figdir / "fig6_scsr_ddm_corr.png"}')


# ============ 主入口 ============

def run_ddm_analysis(n_subjects=None, quick_test=False):
    """主入口

    参数
    ----
    n_subjects : int, 拟合前 N 个被试（None=全部 115）
    quick_test : bool, True 时只跑 1 个被试验证流程
    """
    print('=' * 60)
    print('FoodEEG DDM 分析 (hssm)')
    print('切入点：taste / health 对决策动力学的影响')
    print('=' * 60)

    # 数据准备
    df_all, by_subject = prepare_ddm_data()

    if quick_test:
        print('\n[!] 快速测试模式：仅拟合 1 个被试')
        n_subjects = 1

    # 模型 1: 简单 DDM
    print('\n' + '=' * 60)
    print('模型 1: 简单 DDM (每被试独立拟合 v, a, t)')
    print('=' * 60)
    simple_df = fit_simple_ddm_per_subject(by_subject, n_subjects=n_subjects, quick_test=quick_test)
    simple_path = RESULTS_DIR / 'ddm_simple_params.csv'
    simple_df = merge_results_csv(simple_df, simple_path, key_cols=['subject'])
    simple_df.to_csv(simple_path, index=False)
    print(f'\n[结果] 简单 DDM 参数保存: {simple_path}')

    # 模型 2: 回归 DDM
    print('\n' + '=' * 60)
    print('模型 2: 回归 DDM (v ~ taste_diff + health_diff)')
    print('=' * 60)
    reg_df = fit_regression_ddm_per_subject(by_subject, n_subjects=n_subjects, quick_test=quick_test)
    reg_path = RESULTS_DIR / 'ddm_regression_params.csv'
    reg_df = merge_results_csv(reg_df, reg_path, key_cols=['subject'])
    reg_df.to_csv(reg_path, index=False)
    print(f'\n[结果] 回归 DDM 参数保存: {reg_path}')

    # 汇总分析
    print('\n' + '=' * 60)
    print('汇总分析')
    print('=' * 60)
    df_for_scsr = load_all_pairedchoice()
    scsr_df = compute_self_control_success(df_for_scsr)
    summary = summarize_ddm_results(simple_df, reg_df, scsr_df)

    # 打印关键结果
    print('\n[简单 DDM 参数描述]')
    sd = summary['simple_ddm']
    if sd.get('n_converged', 0) == 0:
        print(f'  收敛被试: 0 ({sd.get("note", "无成功拟合")})')
    else:
        print(f'  收敛被试: {sd["n_converged"]}')
        print(f'  v: {sd["v_mean"]:.3f} ± {sd["v_std"]:.3f}')
        print(f'  a: {sd["a_mean"]:.3f} ± {sd["a_std"]:.3f}')
        print(f'  t: {sd["t_mean"]:.3f} ± {sd["t_std"]:.3f}s')

    print('\n[回归 DDM - 关键统计]')
    rd = summary['reg_ddm']
    if rd.get('n_converged', 0) == 0:
        print(f'  收敛被试: 0 ({rd.get("note", "无成功拟合")})')
    else:
        print(f'  收敛被试: {rd["n_converged"]}')
        print(f'  v_taste:  {rd["v_taste_mean"]:.4f} ± {rd["v_taste_std"]:.4f}, '
              f't={rd["taste_t"]:.2f}, p={rd["taste_p"]:.2e}')
        print(f'  v_health: {rd["v_health_mean"]:.4f} ± {rd["v_health_std"]:.4f}, '
              f't={rd["health_t"]:.2f}, p={rd["health_p"]:.2e}')
        print(f'  taste - health: mean={rd["taste_minus_health_mean"]:.4f}, '
              f't={rd["taste_minus_health_t"]:.2f}, p={rd["taste_minus_health_p"]:.2e}')

    if 'scsr_corr' in summary:
        print('\n[SCSR 相关]')
        for param, c in summary['scsr_corr'].items():
            print(f'  {param}: r={c["r"]:.3f}, p={c["p"]:.3f}')

    if 'group_comparison' in summary:
        print('\n[高/低 SCSR 组对比]')
        for param, g in summary['group_comparison'].items():
            print(f'  {param}: low={g["low_mean"]:.4f}, high={g["high_mean"]:.4f}, '
                  f't={g["t"]:.2f}, p={g["p"]:.4f}')

    # 保存摘要
    summary_path = RESULTS_DIR / 'ddm_summary.json'
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f'\n[结果] 摘要保存: {summary_path}')

    # 两个模型全部拟合失败 → fail-fast：打印根因诊断并以非零码退出，
    # 不再进入绘图（否则会以误导性的 KeyError 崩溃，掩盖真实原因）
    if sd.get('n_converged', 0) == 0 and rd.get('n_converged', 0) == 0:
        errs = []
        for err_df in (simple_df, reg_df):
            if 'error' in err_df.columns:
                errs += err_df['error'].dropna().tolist()
        first_err = errs[0] if errs else 'unknown'
        print('\n' + '=' * 60)
        print('[错误] 所有 DDM 拟合均失败，终止分析')
        print(f'  首个错误: {first_err}')
        if 'ImpImporter' in first_err:
            print('  诊断: pkgutil.ImpImporter 已在 Python 3.12+ 移除，当前环境的老版本')
            print('        setuptools/pkg_resources 仍在引用它。')
            print('  修复: 激活环境后执行  pip install -U setuptools wheel')
            print('  若加垫片后仍失败，元凶可能是其他老包，见上方各被试拟合失败的完整 traceback。')
            print('  建议使用此前复现论文结果时的专用 conda 环境，而非 (base)。')
        else:
            print('  完整 traceback 见上方各被试"拟合失败"输出。')
        print('=' * 60)
        raise SystemExit(1)

    # 可视化
    plot_ddm_results(simple_df, reg_df, summary, FIGURES_DIR)

    return summary


if __name__ == '__main__':
    import sys
    quick = '--quick' in sys.argv
    n = None
    for arg in sys.argv[1:]:
        if arg.startswith('--n='):
            n = int(arg.split('=')[1])
    summary = run_ddm_analysis(n_subjects=n, quick_test=quick)
    print('\n' + '=' * 60)
    print('DDM 分析完成')
    print('=' * 60)
