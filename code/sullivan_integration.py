# -*- coding: utf-8 -*-
"""
切入点 1-A3：Sullivan 时序假说综合验证 + 个体差异提取
======================================================

整合 A1（ERP）与 A2（时频）结果：
1. 综合检验 Sullivan (2015) 时序假说
   - ERP 层面：taste ~500ms（LPP），health ~1000ms（LATE）
   - 时频层面：taste 中期 theta 增强，health 晚期 alpha/beta 调制
2. 提取个体差异指标：
   - taste_early_advantage_erp
   - health_late_advantage_erp
   - taste_mid_theta_advantage
   - health_late_alpha_advantage
3. 与行为 SCSR、DDM 参数做相关分析
   - 检验"神经时序优势 ↔ 行为决策倾向"的对应关系
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import json
from scipy import stats

from config import FIGURES_DIR, RESULTS_DIR, DATA_DIR


# ============ 数据加载 ============
def load_erp_window_means():
    path = RESULTS_DIR / 'erp_window_means.csv'
    if not path.exists():
        print(f'[warn] {path} 不存在，先运行 erp_analysis.py')
        return None
    df = pd.read_csv(path)
    df['subject_idx'] = df['subject_idx'].astype(int)
    return df


def load_tfr_window_means():
    path = RESULTS_DIR / 'tfr_band_window_means.csv'
    if not path.exists():
        print(f'[warn] {path} 不存在，先运行 timefreq_analysis.py')
        return None
    df = pd.read_csv(path)
    return df


def load_ddm_params():
    path = RESULTS_DIR / 'ddm_regression_params.csv'
    if not path.exists():
        print(f'[warn] {path} 不存在')
        return None
    return pd.read_csv(path)


def load_behavior_summary():
    path = RESULTS_DIR / 'baseline_summary.json'
    if not path.exists():
        return None
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_scsr_per_subject():
    """从 load_behavior 重新计算个体 SCSR"""
    from load_behavior import load_all_pairedchoice, compute_self_control_success
    df = load_all_pairedchoice()
    scsr_df = compute_self_control_success(df)
    scsr_df = scsr_df.rename(columns={'subject': 'subject_id'})
    return scsr_df


# ============ 个体差异指标提取 ============
def extract_individual_metrics(erp_df, tfr_df):
    """提取每个被试的时序加工优势指标

    返回 DataFrame，每行一个被试
    """
    metrics = pd.DataFrame()

    # ERP 指标
    if erp_df is not None:
        metrics['subject_idx'] = erp_df['subject_idx']

        # taste 早期加工优势 = taste_LPP_Pz - health_LPP_Pz
        metrics['taste_early_advantage_erp_Pz'] = (
            erp_df['taste_LPP_450-650ms_Pz'] - erp_df['health_LPP_450-650ms_Pz']
        )
        # taste 在 LPP 的绝对幅值
        metrics['taste_LPP_Pz'] = erp_df['taste_LPP_450-650ms_Pz']
        metrics['health_LPP_Pz'] = erp_df['health_LPP_450-650ms_Pz']

        # health 晚期加工优势 = health_LATE_Pz - taste_LATE_Pz
        metrics['health_late_advantage_erp_Pz'] = (
            erp_df['health_LATE_800-1000ms_Pz'] - erp_df['taste_LATE_800-1000ms_Pz']
        )
        metrics['taste_LATE_Pz'] = erp_df['taste_LATE_800-1000ms_Pz']
        metrics['health_LATE_Pz'] = erp_df['health_LATE_800-1000ms_Pz']

        # 早期 vs 晚期交互（Pz）
        metrics['taste_early_minus_late_Pz'] = (
            erp_df['taste_LPP_450-650ms_Pz'] - erp_df['taste_LATE_800-1000ms_Pz']
        )
        metrics['health_early_minus_late_Pz'] = (
            erp_df['health_LPP_450-650ms_Pz'] - erp_df['health_LATE_800-1000ms_Pz']
        )

    # 时频指标（dB）
    if tfr_df is not None:
        # 通过 subject_id 对齐
        if 'subject_id' in tfr_df.columns and 'subject_idx' in metrics.columns:
            # erp_df 的 subject_idx 是 0-based 内部序号
            # tfr_df 的 subject_idx 也是 0-based 内部序号
            tfr_df = tfr_df.copy()
            if 'subject_idx' not in tfr_df.columns:
                tfr_df['subject_idx'] = range(len(tfr_df))

        # taste 中期 theta 优势（Fz）
        if 'taste_theta_mid_350-600ms_Fz' in tfr_df.columns:
            metrics['taste_mid_theta_Fz'] = tfr_df['taste_theta_mid_350-600ms_Fz'].values
            metrics['health_mid_theta_Fz'] = tfr_df['health_theta_mid_350-600ms_Fz'].values
            metrics['taste_mid_theta_advantage_Fz'] = (
                metrics['taste_mid_theta_Fz'] - metrics['health_mid_theta_Fz']
            )

        # health 晚期 alpha 优势（Fz）
        if 'health_alpha_late_700-1000ms_Fz' in tfr_df.columns:
            metrics['health_late_alpha_Fz'] = tfr_df['health_alpha_late_700-1000ms_Fz'].values
            metrics['taste_late_alpha_Fz'] = tfr_df['taste_alpha_late_700-1000ms_Fz'].values
            metrics['health_late_alpha_advantage_Fz'] = (
                metrics['health_late_alpha_Fz'] - metrics['taste_late_alpha_Fz']
            )

        # health 晚期 beta 优势（Fz）
        if 'health_beta_late_700-1000ms_Fz' in tfr_df.columns:
            metrics['health_late_beta_Fz'] = tfr_df['health_beta_late_700-1000ms_Fz'].values
            metrics['taste_late_beta_Fz'] = tfr_df['taste_beta_late_700-1000ms_Fz'].values
            metrics['health_late_beta_advantage_Fz'] = (
                metrics['health_late_beta_Fz'] - metrics['taste_late_beta_Fz']
            )

        # taste 中期 beta 优势（Fz）
        if 'taste_beta_mid_350-600ms_Fz' in tfr_df.columns:
            metrics['taste_mid_beta_Fz'] = tfr_df['taste_beta_mid_350-600ms_Fz'].values
            metrics['health_mid_beta_Fz'] = tfr_df['health_beta_mid_350-600ms_Fz'].values
            metrics['taste_mid_beta_advantage_Fz'] = (
                metrics['taste_mid_beta_Fz'] - metrics['health_mid_beta_Fz']
            )

    return metrics


# ============ 与行为/DDM 关联 ============
def correlate_with_behavior(metrics_df, ddm_df, scsr_df):
    """将神经指标与行为指标做相关分析

    参数
    ----
    metrics_df : 个体神经指标 DataFrame（每行一个被试，含 subject_idx）
    ddm_df : DDM 参数 DataFrame
    scsr_df : 含 subject_id 和 scsr 列的 DataFrame

    返回
    ----
    corr_df : 相关系数表
    """
    # 准备 DDM 数据：按 subject_idx 对齐
    # ddm_regression_params.csv 应该有 subject_id, v_taste, v_health, a, t 等列
    print('DDM df columns:', ddm_df.columns.tolist() if ddm_df is not None else 'None')
    print('SCSR df columns:', scsr_df.columns.tolist() if scsr_df is not None else 'None')

    # 合并：神经指标使用 subject_idx（0-based），DDM 使用 subject_id
    # 我们需要建立映射：subject_idx → subject_id
    # 从 erp_summary.json 中读取被试列表顺序
    erp_summary_path = RESULTS_DIR / 'erp_summary.json'
    subject_ids = None
    if erp_summary_path.exists():
        with open(erp_summary_path, 'r', encoding='utf-8') as f:
            erp_summary = json.load(f)
        subject_ids = erp_summary.get('subject_ids')

    if subject_ids is None:
        print('[warn] 无法获取 subject_ids 顺序')
        return None

    # 添加 subject_id 列
    metrics_df = metrics_df.copy()
    metrics_df['subject_id'] = [subject_ids[i] for i in metrics_df['subject_idx']]

    # 合并 DDM
    merged = metrics_df.copy()
    if ddm_df is not None:
        # DDM df 列名 'subject'，重命名为 'subject_id' 以统一
        ddm_renamed = ddm_df.rename(columns={'subject': 'subject_id'})
        # 取 v_taste, v_health 列
        ddm_cols = ['subject_id']
        for c in ['v_taste', 'v_health', 'a', 't']:
            if c in ddm_renamed.columns:
                ddm_cols.append(c)
        merged = merged.merge(ddm_renamed[ddm_cols], on='subject_id', how='left')

    # 合并 SCSR
    if scsr_df is not None:
        if 'scsr' not in scsr_df.columns:
            # 尝试其他名字
            for c in scsr_df.columns:
                if 'scsr' in c.lower():
                    scsr_df = scsr_df.rename(columns={c: 'scsr'})
                    break
        if 'subject_id' in scsr_df.columns and 'scsr' in scsr_df.columns:
            merged = merged.merge(scsr_df[['subject_id', 'scsr']], on='subject_id', how='left')

    # 数值列
    numeric_cols = merged.select_dtypes(include=[np.number]).columns.tolist()
    # 排除 subject_idx
    numeric_cols = [c for c in numeric_cols if c != 'subject_idx']

    # 计算相关矩阵
    corr = merged[numeric_cols].corr(method='pearson')

    # 提取与 v_taste, v_health, scsr 的相关
    target_cols = [c for c in ['v_taste', 'v_health', 'scsr'] if c in corr.columns]
    neural_cols = [c for c in numeric_cols if c not in target_cols]

    rows = []
    for nc in neural_cols:
        for tc in target_cols:
            valid = merged[[nc, tc]].dropna()
            if len(valid) < 10:
                continue
            r, p = stats.pearsonr(valid[nc], valid[tc])
            # 95% CI via Fisher z
            z = np.arctanh(r)
            se = 1 / np.sqrt(len(valid) - 3)
            ci_low = np.tanh(z - 1.96 * se)
            ci_high = np.tanh(z + 1.96 * se)
            rows.append({
                'neural_metric': nc,
                'behavior_metric': tc,
                'n': len(valid),
                'r': r,
                'r_ci_low': ci_low,
                'r_ci_high': ci_high,
                'p': p,
            })

    corr_df = pd.DataFrame(rows)
    # FDR 校正
    from mne.stats import fdr_correction
    if len(corr_df) > 0:
        _, p_fdr = fdr_correction(corr_df['p'].values)
        corr_df['p_fdr'] = p_fdr
        corr_df['sig_fdr'] = p_fdr < 0.05
    return corr_df, merged


# ============ 可视化 ============
def plot_correlation_heatmap(merged_df, save_path):
    """神经 vs 行为指标的相关矩阵热图"""
    target_cols = [c for c in ['v_taste', 'v_health', 'scsr']
                   if c in merged_df.columns]
    neural_cols = [c for c in merged_df.columns
                   if c not in ['subject_idx', 'subject_id', 'v_taste',
                                'v_health', 'scsr', 'a', 't']
                   and pd.api.types.is_numeric_dtype(merged_df[c])]

    if not target_cols or not neural_cols:
        print('[warn] 无可绘制的相关')
        return

    corr = merged_df[neural_cols + target_cols].corr()
    # 截取相关矩阵
    corr_subset = corr.loc[neural_cols, target_cols]

    fig, ax = plt.subplots(figsize=(max(6, len(target_cols) * 1.5),
                                     max(8, len(neural_cols) * 0.4)))
    im = ax.imshow(corr_subset.values, aspect='auto', cmap='RdBu_r',
                   vmin=-0.5, vmax=0.5)
    ax.set_xticks(range(len(target_cols)))
    ax.set_xticklabels(target_cols, rotation=45, ha='right')
    ax.set_yticks(range(len(neural_cols)))
    ax.set_yticklabels(neural_cols, fontsize=9)
    # 标注数值
    for i in range(len(neural_cols)):
        for j in range(len(target_cols)):
            v = corr_subset.values[i, j]
            color = 'white' if abs(v) > 0.3 else 'black'
            ax.text(j, i, f'{v:.2f}', ha='center', va='center',
                    color=color, fontsize=8)
    plt.colorbar(im, ax=ax, label='Pearson r')
    ax.set_title('Neural × Behavior correlations', fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'[fig] saved: {save_path}')


def plot_sullivan_summary(merged_df, save_path):
    """Sullivan 假说核心散点：神经时序优势 vs 行为倾向"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. taste 早期优势 ERP vs v_taste
    ax = axes[0, 0]
    if 'taste_early_advantage_erp_Pz' in merged_df.columns and 'v_taste' in merged_df.columns:
        valid = merged_df[['taste_early_advantage_erp_Pz', 'v_taste']].dropna()
        ax.scatter(valid['taste_early_advantage_erp_Pz'], valid['v_taste'],
                   alpha=0.6, color='#E07B39')
        if len(valid) > 5:
            r, p = stats.pearsonr(valid['taste_early_advantage_erp_Pz'],
                                   valid['v_taste'])
            ax.set_title(f'taste ERP advantage vs v_taste\nr={r:.3f}, p={p:.3f}')
        # 拟合线
        if len(valid) > 5:
            z = np.polyfit(valid['taste_early_advantage_erp_Pz'],
                           valid['v_taste'], 1)
            xs = np.linspace(valid['taste_early_advantage_erp_Pz'].min(),
                             valid['taste_early_advantage_erp_Pz'].max(), 50)
            ax.plot(xs, np.polyval(z, xs), 'k--', alpha=0.5)
        ax.set_xlabel('taste_early_advantage_erp_Pz (µV)')
        ax.set_ylabel('v_taste (DDM)')
        ax.axhline(0, color='black', linewidth=0.5)
        ax.axvline(0, color='black', linewidth=0.5)
    else:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center')
        ax.set_title('taste ERP advantage vs v_taste')

    # 2. health 晚期优势 ERP vs v_health
    ax = axes[0, 1]
    if 'health_late_advantage_erp_Pz' in merged_df.columns and 'v_health' in merged_df.columns:
        valid = merged_df[['health_late_advantage_erp_Pz', 'v_health']].dropna()
        ax.scatter(valid['health_late_advantage_erp_Pz'], valid['v_health'],
                   alpha=0.6, color='#3D7A8E')
        if len(valid) > 5:
            r, p = stats.pearsonr(valid['health_late_advantage_erp_Pz'],
                                   valid['v_health'])
            ax.set_title(f'health ERP advantage vs v_health\nr={r:.3f}, p={p:.3f}')
            z = np.polyfit(valid['health_late_advantage_erp_Pz'],
                           valid['v_health'], 1)
            xs = np.linspace(valid['health_late_advantage_erp_Pz'].min(),
                             valid['health_late_advantage_erp_Pz'].max(), 50)
            ax.plot(xs, np.polyval(z, xs), 'k--', alpha=0.5)
        ax.set_xlabel('health_late_advantage_erp_Pz (µV)')
        ax.set_ylabel('v_health (DDM)')
        ax.axhline(0, color='black', linewidth=0.5)
        ax.axvline(0, color='black', linewidth=0.5)
    else:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center')
        ax.set_title('health ERP advantage vs v_health')

    # 3. taste theta 优势 vs SCSR
    ax = axes[1, 0]
    if 'taste_mid_theta_advantage_Fz' in merged_df.columns and 'scsr' in merged_df.columns:
        valid = merged_df[['taste_mid_theta_advantage_Fz', 'scsr']].dropna()
        ax.scatter(valid['taste_mid_theta_advantage_Fz'], valid['scsr'],
                   alpha=0.6, color='#E07B39')
        if len(valid) > 5:
            r, p = stats.pearsonr(valid['taste_mid_theta_advantage_Fz'],
                                   valid['scsr'])
            ax.set_title(f'taste mid theta advantage vs SCSR\nr={r:.3f}, p={p:.3f}')
            z = np.polyfit(valid['taste_mid_theta_advantage_Fz'],
                           valid['scsr'], 1)
            xs = np.linspace(valid['taste_mid_theta_advantage_Fz'].min(),
                             valid['taste_mid_theta_advantage_Fz'].max(), 50)
            ax.plot(xs, np.polyval(z, xs), 'k--', alpha=0.5)
        ax.set_xlabel('taste_mid_theta_advantage_Fz (dB)')
        ax.set_ylabel('SCSR')
        ax.axhline(0.5, color='black', linewidth=0.5, linestyle='--')
    else:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center')
        ax.set_title('taste theta advantage vs SCSR')

    # 4. health alpha 优势 vs SCSR
    ax = axes[1, 1]
    if 'health_late_alpha_advantage_Fz' in merged_df.columns and 'scsr' in merged_df.columns:
        valid = merged_df[['health_late_alpha_advantage_Fz', 'scsr']].dropna()
        ax.scatter(valid['health_late_alpha_advantage_Fz'], valid['scsr'],
                   alpha=0.6, color='#3D7A8E')
        if len(valid) > 5:
            r, p = stats.pearsonr(valid['health_late_alpha_advantage_Fz'],
                                   valid['scsr'])
            ax.set_title(f'health late alpha advantage vs SCSR\nr={r:.3f}, p={p:.3f}')
            z = np.polyfit(valid['health_late_alpha_advantage_Fz'],
                           valid['scsr'], 1)
            xs = np.linspace(valid['health_late_alpha_advantage_Fz'].min(),
                             valid['health_late_alpha_advantage_Fz'].max(), 50)
            ax.plot(xs, np.polyval(z, xs), 'k--', alpha=0.5)
        ax.set_xlabel('health_late_alpha_advantage_Fz (dB)')
        ax.set_ylabel('SCSR')
        ax.axhline(0.5, color='black', linewidth=0.5, linestyle='--')
    else:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center')
        ax.set_title('health alpha advantage vs SCSR')

    plt.suptitle('Sullivan hypothesis: neural timing advantage vs behavior',
                 fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'[fig] saved: {save_path}')


# ============ 主入口 ============
def main():
    print('=' * 60)
    print('切入点 1-A3：Sullivan 假说综合验证 + 个体差异')
    print('=' * 60)

    # 1. 加载数据
    print('\n[step 1] 加载数据...')
    erp_df = load_erp_window_means()
    tfr_df = load_tfr_window_means()
    ddm_df = load_ddm_params()
    scsr_df = load_scsr_per_subject()

    print(f'  ERP 窗口均值: {erp_df.shape if erp_df is not None else None}')
    print(f'  TFR 窗口均值: {tfr_df.shape if tfr_df is not None else None}')
    print(f'  DDM 参数: {ddm_df.shape if ddm_df is not None else None}')
    print(f'  SCSR: {scsr_df.shape if scsr_df is not None else None}')

    if erp_df is None:
        print('[error] 缺少 ERP 数据，退出')
        return

    # 2. 提取个体指标
    print('\n[step 2] 提取个体差异指标...')
    metrics = extract_individual_metrics(erp_df, tfr_df)
    print(f'  指标数: {len(metrics.columns) - 1}（不含 subject_idx）')
    print(f'  被试数: {len(metrics)}')

    erp_summary_path = RESULTS_DIR / 'erp_summary.json'
    if erp_summary_path.exists():
        with open(erp_summary_path, 'r', encoding='utf-8') as f:
            subject_ids = json.load(f).get('subject_ids')
        if subject_ids is not None:
            metrics['subject_id'] = [subject_ids[i] for i in metrics['subject_idx']]
        else:
            print('[warn] erp_summary.json 无 subject_ids，individual_metrics.csv 将不含 subject_id')
    else:
        print('[warn] erp_summary.json 不存在，individual_metrics.csv 将不含 subject_id')

    metrics.to_csv(RESULTS_DIR / 'individual_metrics.csv', index=False)
    print(f'  保存: {RESULTS_DIR / "individual_metrics.csv"}')

    # 3. 描述性统计
    print('\n[step 3] 描述性统计：')
    desc = metrics.drop(columns=['subject_idx']).describe()
    print(desc.to_string())

    # 4. 与行为/DDM 相关
    print('\n[step 4] 与行为/DDM 参数相关分析...')
    if ddm_df is None and scsr_df is None:
        print('[warn] 无行为/DDM 数据，跳过相关分析')
        corr_df = None
        merged = metrics
    else:
        corr_df, merged = correlate_with_behavior(metrics, ddm_df, scsr_df)
        corr_df.to_csv(RESULTS_DIR / 'neural_behavior_correlations.csv', index=False)
        print(f'\n  保存相关表至 {RESULTS_DIR / "neural_behavior_correlations.csv"}')

        print('\n[FDR 显著相关]：')
        if 'sig_fdr' in corr_df.columns:
            sig = corr_df[corr_df['sig_fdr']]
            if len(sig) > 0:
                print(sig[['neural_metric', 'behavior_metric', 'n', 'r',
                           'r_ci_low', 'r_ci_high', 'p', 'p_fdr']].to_string(index=False))
            else:
                print('  无 FDR 显著相关')

    # 5. 可视化
    print('\n[step 5] 可视化...')
    plot_correlation_heatmap(merged, FIGURES_DIR / 'fig16_corr_heatmap.png')
    plot_sullivan_summary(merged, FIGURES_DIR / 'fig17_sullivan_summary.png')

    # 6. 综合判定 Sullivan 假说
    print('\n[step 6] Sullivan 假说综合判定...')

    # 加载 A1 和 A2 的 summary
    erp_summary = None
    tfr_summary = None
    erp_summary_path = RESULTS_DIR / 'erp_summary.json'
    tfr_summary_path = RESULTS_DIR / 'tfr_summary.json'
    if erp_summary_path.exists():
        with open(erp_summary_path, 'r', encoding='utf-8') as f:
            erp_summary = json.load(f)
    if tfr_summary_path.exists():
        with open(tfr_summary_path, 'r', encoding='utf-8') as f:
            tfr_summary = json.load(f)

    sullivan_verdict = {
        'hypothesis': 'Sullivan (2015): tastiness ~500ms, healthiness ~1000ms',
        'erp_findings': {
            'roi_ttest_n_sig_fdr': erp_summary.get('roi_ttest_summary', {}).get('n_significant_fdr') if erp_summary else None,
            'sullivan_direct_test': erp_summary.get('sullivan_direct_test') if erp_summary else None,
            'cluster_n_sig': {k: v.get('n_clusters_significant') for k, v in
                              erp_summary.get('cluster_results', {}).items()} if erp_summary else None,
        },
        'tfr_findings': {
            'tfr_ttest_n_sig_fdr': tfr_summary.get('tfr_ttest_summary', {}).get('n_significant_fdr') if tfr_summary else None,
            'sullivan_tf_test': tfr_summary.get('sullivan_tf_test') if tfr_summary else None,
        } if tfr_summary else None,
        'individual_differences': {
            'n_significant_correlations_fdr': int(corr_df['sig_fdr'].sum()) if corr_df is not None and 'sig_fdr' in corr_df.columns else None,
        },
    }

    # 整体判定
    erp_sig = erp_summary.get('sullivan_direct_test', {}).get('test_3_taste_gt_health_at_LPP_Pz', {}).get('p', 1.0) if erp_summary else 1.0
    erp_sig = float(erp_sig) if erp_sig is not None else 1.0
    erp_interact = erp_summary.get('sullivan_direct_test', {}).get('test_5_interaction_taste_advantage_early_vs_late_Pz', {}).get('p', 1.0) if erp_summary else 1.0
    erp_interact = float(erp_interact) if erp_interact is not None else 1.0

    if erp_sig < 0.05 and erp_interact < 0.05:
        verdict = 'STRONGLY SUPPORTED'
    elif erp_sig < 0.05 or erp_interact < 0.05:
        verdict = 'PARTIALLY SUPPORTED'
    else:
        verdict = 'NOT SUPPORTED at group level'

    sullivan_verdict['overall_verdict'] = verdict
    print(f'\n  整体判定: {verdict}')
    print(f'  ERP test_3 (taste>health@LPP_Pz) p = {erp_sig:.4f}')
    print(f'  ERP test_5 (interaction) p = {erp_interact:.4f}')

    # 保存
    final_summary = {
        'n_subjects': len(metrics),
        'individual_metrics_columns': metrics.columns.tolist(),
        'sullivan_verdict': sullivan_verdict,
        'erp_summary_ref': str(erp_summary_path),
        'tfr_summary_ref': str(tfr_summary_path),
    }

    with open(RESULTS_DIR / 'cutpoint1_summary.json', 'w', encoding='utf-8') as f:
        json.dump(final_summary, f, indent=2, ensure_ascii=False, default=str)
    print(f'\n[summary] {RESULTS_DIR / "cutpoint1_summary.json"}')

    print('\n' + '=' * 60)
    print('A3 完成：Sullivan 假说综合验证')
    print('=' * 60)


if __name__ == '__main__':
    main()
