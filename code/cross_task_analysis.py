# -*- coding: utf-8 -*-
"""
切入点 2-B2：跨任务关联分析（EEG 解码 ↔ 行为/DDM 参数）
======================================================

检验"神经表征可分性 ↔ 行为决策倾向"的对应关系：
1. Riemannian 解码准确率（taste-health / taste-decision / health-decision）
2. 行为指标：SCSR（自控成功率）、DDM 参数（v_taste, v_health, a, t）
3. 神经指标（A3）：ERP/时频个体差异指标

核心假设：
- H1: taste-health 解码准确率 ↑ → SCSR ↑（神经可分性高的被试更善自控）
- H2: taste-health 解码准确率 ↑ → v_taste / v_health ↑（属性维度敏感度高）
- H3: 跨对比稳定性（taste-health acc 与 taste-decision acc 的相关）
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import json
from scipy import stats

from config import FIGURES_DIR, RESULTS_DIR


# ============ 数据加载 ============
def load_decoding_results():
    """读取 B1 的解码结果"""
    path = RESULTS_DIR / 'riemannian_decoding_results.csv'
    if not path.exists():
        print(f'[warn] {path} 不存在，先运行 riemannian_decoding.py')
        return None
    df = pd.read_csv(path)
    return df


def load_ddm_params():
    path = RESULTS_DIR / 'ddm_regression_params.csv'
    if not path.exists():
        print(f'[warn] {path} 不存在')
        return None
    df = pd.read_csv(path)
    # 列名 'subject' → 'subject_id'
    if 'subject' in df.columns:
        df = df.rename(columns={'subject': 'subject_id'})
    return df


def load_scsr():
    """从 load_behavior 计算个体 SCSR"""
    from load_behavior import load_all_pairedchoice, compute_self_control_success
    df = load_all_pairedchoice()
    scsr_df = compute_self_control_success(df)
    if 'subject' in scsr_df.columns:
        scsr_df = scsr_df.rename(columns={'subject': 'subject_id'})
    return scsr_df


def load_individual_metrics():
    """读取 A3 的神经个体差异指标"""
    path = RESULTS_DIR / 'individual_metrics.csv'
    if not path.exists():
        print(f'[warn] {path} 不存在，先运行 sullivan_integration.py')
        return None
    df = pd.read_csv(path)
    return df


def load_baseline_summary():
    """读取 baseline 行为汇总（含 pooled 系数等）"""
    path = RESULTS_DIR / 'baseline_summary.json'
    if not path.exists():
        return None
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


# ============ 个体解码准确率提取 ============
def extract_decoding_metrics(decoding_df):
    """从解码结果提取每被试的准确率指标

    返回 DataFrame，每行一个被试
    """
    if decoding_df is None or len(decoding_df) == 0:
        return None

    # pivot：每被试一行，列为各对比的准确率
    pivot = decoding_df.pivot_table(
        index='subject_id',
        columns='contrast',
        values='balanced_acc',
        aggfunc='mean'
    ).reset_index()

    # 重命名列
    for col in pivot.columns:
        if col != 'subject_id':
            pivot = pivot.rename(columns={col: f'decoding_{col}_acc'})

    # 跨对比稳定性：taste-health 与 taste-decision 的差异
    if 'decoding_taste-health_acc' in pivot.columns:
        if 'decoding_taste-decision_acc' in pivot.columns:
            pivot['decoding_taste_health_vs_decision_diff'] = (
                pivot['decoding_taste-health_acc'] - pivot['decoding_taste-decision_acc']
            )

    return pivot


# ============ 相关分析 ============
def correlate_decoding_with_behavior(decoding_metrics, ddm_df, scsr_df,
                                      neural_metrics=None):
    """解码准确率与行为/DDM/神经指标的相关分析

    返回
    ----
    corr_df : 相关系数表
    merged_df : 合并后的完整数据（用于后续分析）
    """
    merged = decoding_metrics.copy()

    # 合并 DDM
    if ddm_df is not None:
        ddm_cols = ['subject_id']
        for c in ['v_taste', 'v_health', 'a', 't']:
            if c in ddm_df.columns:
                ddm_cols.append(c)
        merged = merged.merge(ddm_df[ddm_cols], on='subject_id', how='left')

    # 合并 SCSR
    if scsr_df is not None:
        if 'scsr' in scsr_df.columns:
            merged = merged.merge(
                scsr_df[['subject_id', 'scsr']], on='subject_id', how='left'
            )

    # 合并 A3 神经指标
    if neural_metrics is not None:
        if 'subject_id' in neural_metrics.columns:
            # 只取数值列，避免重复
            neu_cols = ['subject_id']
            for c in neural_metrics.columns:
                if c not in merged.columns and c != 'subject_idx':
                    neu_cols.append(c)
            merged = merged.merge(
                neural_metrics[neu_cols], on='subject_id', how='left'
            )

    # 解码指标列
    decoding_cols = [c for c in merged.columns if c.startswith('decoding_')]
    # 行为/DDM 指标列
    behavior_cols = [c for c in ['v_taste', 'v_health', 'a', 't', 'scsr']
                     if c in merged.columns]
    # 神经指标列（A3）
    neural_cols = [c for c in merged.columns
                   if c not in decoding_cols + behavior_cols + ['subject_id']
                   and c != 'subject_idx'
                   and pd.api.types.is_numeric_dtype(merged[c])]

    # 相关分析
    results = []
    for dec_col in decoding_cols:
        for beh_col in behavior_cols + neural_cols:
            valid = merged[[dec_col, beh_col]].dropna()
            if len(valid) < 10:
                continue
            r, p = stats.pearsonr(valid[dec_col], valid[beh_col])
            # 95% CI via Fisher z
            z = np.arctanh(r) if abs(r) < 1 else 0
            se = 1.0 / np.sqrt(len(valid) - 3)
            ci_low = np.tanh(z - 1.96 * se)
            ci_high = np.tanh(z + 1.96 * se)
            results.append({
                'decoding_metric': dec_col,
                'target_metric': beh_col,
                'n': len(valid),
                'r': r,
                'r_ci_low': ci_low,
                'r_ci_high': ci_high,
                'p': p,
            })

    corr_df = pd.DataFrame(results)

    # FDR 校正
    if len(corr_df) > 0:
        from mne.stats import fdr_correction
        _, p_fdr = fdr_correction(corr_df['p'].values)
        corr_df['p_fdr'] = p_fdr
        corr_df['sig_fdr'] = p_fdr < 0.05

    return corr_df, merged


# ============ 可视化 ============
def plot_cross_task_correlations(corr_df, merged_df, save_path):
    """跨任务关联可视化"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. 解码准确率分布（3个对比）
    ax = axes[0, 0]
    decoding_cols = [c for c in merged_df.columns if c.startswith('decoding_')
                     and c.endswith('_acc')]
    data = [merged_df[c].dropna() for c in decoding_cols]
    if data:
        bp = ax.boxplot(data, labels=[c.replace('decoding_', '').replace('_acc', '')
                                       for c in decoding_cols])
        ax.axhline(0.5, color='red', linestyle='--', label='chance')
        ax.set_ylabel('Balanced accuracy')
        ax.set_title('Decoding accuracy distribution')
        ax.legend()

    # 2. taste-health 解码 vs SCSR
    ax = axes[0, 1]
    if 'decoding_taste-health_acc' in merged_df.columns and 'scsr' in merged_df.columns:
        valid = merged_df[['decoding_taste-health_acc', 'scsr']].dropna()
        if len(valid) > 5:
            ax.scatter(valid['decoding_taste-health_acc'], valid['scsr'],
                      alpha=0.6, edgecolor='white')
            r, p = stats.pearsonr(valid['decoding_taste-health_acc'],
                                   valid['scsr'])
            ax.set_xlabel('taste-health decoding acc')
            ax.set_ylabel('SCSR')
            sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else ''))
            ax.set_title(f'Decoding vs SCSR\nr={r:.3f}, p={p:.4f} {sig}')

    # 3. taste-health 解码 vs DDM v_taste
    ax = axes[1, 0]
    if 'decoding_taste-health_acc' in merged_df.columns and 'v_taste' in merged_df.columns:
        valid = merged_df[['decoding_taste-health_acc', 'v_taste']].dropna()
        if len(valid) > 5:
            ax.scatter(valid['decoding_taste-health_acc'], valid['v_taste'],
                      alpha=0.6, edgecolor='white', color='orange')
            r, p = stats.pearsonr(valid['decoding_taste-health_acc'],
                                   valid['v_taste'])
            ax.set_xlabel('taste-health decoding acc')
            ax.set_ylabel('DDM v_taste')
            sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else ''))
            ax.set_title(f'Decoding vs DDM v_taste\nr={r:.3f}, p={p:.4f} {sig}')

    # 4. 跨对比稳定性
    ax = axes[1, 1]
    if 'decoding_taste-health_acc' in merged_df.columns and 'decoding_taste-decision_acc' in merged_df.columns:
        valid = merged_df[['decoding_taste-health_acc', 'decoding_taste-decision_acc']].dropna()
        if len(valid) > 5:
            ax.scatter(valid['decoding_taste-health_acc'],
                      valid['decoding_taste-decision_acc'],
                      alpha=0.6, edgecolor='white', color='green')
            r, p = stats.pearsonr(valid['decoding_taste-health_acc'],
                                   valid['decoding_taste-decision_acc'])
            ax.set_xlabel('taste-health acc')
            ax.set_ylabel('taste-decision acc')
            ax.set_title(f'Cross-contrast stability\nr={r:.3f}, p={p:.4f}')
            ax.plot([0.3, 0.9], [0.3, 0.9], 'k--', alpha=0.3)

    plt.suptitle('B2: Cross-task associations (EEG decoding vs behavior/DDM)', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'[fig] saved: {save_path}')


def plot_corr_heatmap(corr_df, save_path):
    """相关热图"""
    if len(corr_df) == 0:
        return
    pivot = corr_df.pivot_table(
        index='decoding_metric',
        columns='target_metric',
        values='r'
    )
    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(pivot.values, cmap='RdBu_r', vmin=-0.4, vmax=0.4, aspect='auto')
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha='right', fontsize=8)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=8)
    plt.colorbar(im, ax=ax, label='Pearson r')
    ax.set_title('B2: Decoding vs behavioral/neural metrics (Pearson r)')

    # 标注显著
    p_pivot = corr_df.pivot_table(
        index='decoding_metric',
        columns='target_metric',
        values='p_fdr'
    )
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            if not np.isnan(p_pivot.values[i, j]) and p_pivot.values[i, j] < 0.05:
                ax.text(j, i, '*', ha='center', va='center', fontsize=12,
                       color='black', fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'[fig] saved: {save_path}')


# ============ 主入口 ============
def main():
    print('=' * 60)
    print('切入点 2-B2：跨任务关联分析')
    print('=' * 60)

    # 1. 加载数据
    print('\n[step 1] 加载数据...')
    decoding_df = load_decoding_results()
    ddm_df = load_ddm_params()
    scsr_df = load_scsr()
    neural_df = load_individual_metrics()

    if decoding_df is None:
        print('[error] 解码结果不存在，退出')
        return

    print(f'  解码结果: {len(decoding_df)} 行, '
          f'{decoding_df["subject_id"].nunique()} 被试')
    if ddm_df is not None:
        print(f'  DDM 参数: {len(ddm_df)} 被试')
    if scsr_df is not None:
        print(f'  SCSR: {len(scsr_df)} 被试')
    if neural_df is not None:
        print(f'  A3 神经指标: {len(neural_df)} 被试, '
              f'{neural_df.shape[1]} 列')

    # 2. 提取个体解码指标
    print('\n[step 2] 提取个体解码指标...')
    decoding_metrics = extract_decoding_metrics(decoding_df)
    print(f'  解码指标: {len(decoding_metrics)} 被试, '
          f'{decoding_metrics.shape[1]} 列')
    print(f'  列: {decoding_metrics.columns.tolist()}')

    # 3. 相关分析
    print('\n[step 3] 相关分析...')
    corr_df, merged_df = correlate_decoding_with_behavior(
        decoding_metrics, ddm_df, scsr_df, neural_df
    )

    if len(corr_df) == 0:
        print('[warn] 无相关结果')
        return

    # 保存
    corr_df.to_csv(RESULTS_DIR / 'cross_task_correlations.csv', index=False)
    print(f'\n[saved] {RESULTS_DIR / "cross_task_correlations.csv"}')
    print(f'  共 {len(corr_df)} 个相关')

    # 显著结果
    sig = corr_df[corr_df['sig_fdr'] == True]
    print(f'\n  FDR 显著: {len(sig)} 个')
    if len(sig) > 0:
        print(sig[['decoding_metric', 'target_metric', 'r', 'p', 'p_fdr']].to_string())

    # 4. 可视化
    print('\n[step 4] 可视化...')
    plot_cross_task_correlations(
        corr_df, merged_df,
        FIGURES_DIR / 'fig19_cross_task_correlations.png'
    )
    plot_corr_heatmap(
        corr_df,
        FIGURES_DIR / 'fig20_cross_task_heatmap.png'
    )

    # 5. 汇总
    summary = {
        'n_subjects': len(decoding_metrics),
        'n_correlations': len(corr_df),
        'n_fdr_significant': int(len(sig)),
        'decoding_metrics': [c for c in decoding_metrics.columns
                             if c.startswith('decoding_')],
        'main_findings': {},
    }

    # 主结果：taste-health 解码 vs 各行为指标
    main_dec = 'decoding_taste-health_acc'
    if main_dec in corr_df['decoding_metric'].values:
        sub = corr_df[corr_df['decoding_metric'] == main_dec]
        for _, row in sub.iterrows():
            summary['main_findings'][row['target_metric']] = {
                'r': float(row['r']),
                'p': float(row['p']),
                'p_fdr': float(row['p_fdr']),
                'sig': bool(row['sig_fdr']),
            }

    with open(RESULTS_DIR / 'cross_task_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    print(f'\n[summary] {RESULTS_DIR / "cross_task_summary.json"}')

    print('\n' + '=' * 60)
    print('B2 完成：跨任务关联分析')
    print('=' * 60)


if __name__ == '__main__':
    main()
