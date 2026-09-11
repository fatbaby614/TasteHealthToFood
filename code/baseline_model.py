# -*- coding: utf-8 -*-
"""
Baseline logistic 模型拟合

科学问题：
1. taste 和 health 评分差异能否预测被试的食物选择？(Hare 2009 框架)
2. 在 conflict 试次中（美味-健康方向相反），自控成功率（SCSR）整体如何？
   (Sullivan 2015 框架：tastiness 加工快于 healthiness → 倾向选美味)
3. 个体差异：高/低 SCSR 被试的 taste/health 权重是否有差异？

模型：
- M1: chose_stim1 ~ taste_diff （单变量 baseline）
- M2: chose_stim1 ~ health_diff （单变量 baseline）
- M3: chose_stim1 ~ taste_diff + health_diff （联合模型，对比 M1/M2）
- M4 (核心): chose_stim1 ~ taste_diff + health_diff + (1 + taste_diff + health_diff | subject)
   混合效应 logistic 回归，捕捉个体差异
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from config import RESULTS_DIR, FIGURES_DIR, ALPHA
from load_behavior import load_all_pairedchoice, compute_self_control_success


def fit_pooled_logistic(df):
    """在所有试次上拟合 pooled logistic 回归模型 (M1, M2, M3)"""
    results = {}

    # M1: 单变量 taste
    print('\n' + '=' * 60)
    print('M1: chose_stim1 ~ taste_diff')
    print('=' * 60)
    m1 = smf.logit('chose_stim1 ~ taste_diff', data=df).fit(disp=0)
    print(m1.summary().tables[1])
    results['M1_taste'] = {
        'coef': m1.params['taste_diff'],
        'p': m1.pvalues['taste_diff'],
        'pseudo_r2': m1.prsquared,
        'aic': m1.aic,
        'model': m1,
    }

    # M2: 单变量 health
    print('\n' + '=' * 60)
    print('M2: chose_stim1 ~ health_diff')
    print('=' * 60)
    m2 = smf.logit('chose_stim1 ~ health_diff', data=df).fit(disp=0)
    print(m2.summary().tables[1])
    results['M2_health'] = {
        'coef': m2.params['health_diff'],
        'p': m2.pvalues['health_diff'],
        'pseudo_r2': m2.prsquared,
        'aic': m2.aic,
        'model': m2,
    }

    # M3: 联合模型
    print('\n' + '=' * 60)
    print('M3: chose_stim1 ~ taste_diff + health_diff')
    print('=' * 60)
    m3 = smf.logit('chose_stim1 ~ taste_diff + health_diff', data=df).fit(disp=0)
    print(m3.summary().tables[1])
    results['M3_combined'] = {
        'coef_taste': m3.params['taste_diff'],
        'coef_health': m3.params['health_diff'],
        'p_taste': m3.pvalues['taste_diff'],
        'p_health': m3.pvalues['health_diff'],
        'pseudo_r2': m3.prsquared,
        'aic': m3.aic,
        'model': m3,
    }

    # 模型对比
    print('\n' + '=' * 60)
    print('模型对比')
    print('=' * 60)
    print(f"M1 (taste only):   pseudo R² = {results['M1_taste']['pseudo_r2']:.4f}, AIC = {results['M1_taste']['aic']:.1f}")
    print(f"M2 (health only):  pseudo R² = {results['M2_health']['pseudo_r2']:.4f}, AIC = {results['M2_health']['aic']:.1f}")
    print(f"M3 (combined):     pseudo R² = {results['M3_combined']['pseudo_r2']:.4f}, AIC = {results['M3_combined']['aic']:.1f}")

    # 似然比检验：M3 vs M1, M3 vs M2
    lr_m1 = 2 * (m3.llf - m1.llf)
    lr_m2 = 2 * (m3.llf - m2.llf)
    p_m1 = stats.chi2.sf(lr_m1, df=1)
    p_m2 = stats.chi2.sf(lr_m2, df=1)
    print(f'\n似然比检验 (M3 vs M1，加 health 的贡献): χ² = {lr_m1:.2f}, p = {p_m1:.4e}')
    print(f'似然比检验 (M3 vs M2，加 taste 的贡献):  χ² = {lr_m2:.2f}, p = {p_m2:.4e}')

    return results


def fit_mixed_logistic(df):
    """拟合混合效应 logistic 回归 (M4) - 捕捉个体差异

    用 Binomial BayesMixed GLM 替代 lme4，因为 statsmodels 的 BinomialBayesMixedGLM 较稳定
    """
    print('\n' + '=' * 60)
    print('M4: 混合效应 logistic - chose_stim1 ~ taste + health + (1+...|subject)')
    print('=' * 60)
    try:
        md = smf.mixedlm(
            'chose_stim1 ~ taste_diff + health_diff',
            data=df,
            groups=df['subject'],
            re_formula='1 + taste_diff + health_diff'
        )
        mdf = md.fit(method='lbfgs', maxiter=200)
        print(mdf.summary().tables[0])
        print(mdf.summary().tables[1])

        # 提取固定效应
        fixed_taste = mdf.fe_params.get('taste_diff', np.nan)
        fixed_health = mdf.fe_params.get('health_diff', np.nan)
        p_taste = mdf.pvalues.get('taste_diff', np.nan)
        p_health = mdf.pvalues.get('health_diff', np.nan)

        # 提取随机效应（个体权重）
        random_effects = mdf.random_effects

        return {
            'fixed_taste': fixed_taste,
            'fixed_health': fixed_health,
            'p_taste': p_taste,
            'p_health': p_health,
            'random_effects': random_effects,
            'model': mdf,
        }
    except Exception as e:
        print(f'混合效应模型拟合失败: {e}')
        return None


def test_scsr_distribution(df):
    """检验自控成功率 (SCSR) 分布

    在 conflict 试次中，被试选更健康食物的比例 SCSR：
    - 若 SCSR < 0.5，说明被试更倾向选美味（自控失败）
    - Sullivan 2015 假说：tastiness 加工快，导致非自控倾向
    """
    print('\n' + '=' * 60)
    print('SCSR 检验（conflict 试次中选更健康食物的比例）')
    print('=' * 60)

    scsr_df = compute_self_control_success(df)
    scsr_values = scsr_df['scsr'].values

    # 描述统计
    print(f'被试数: {len(scsr_values)}')
    print(f'均值: {scsr_values.mean():.4f}')
    print(f'中位数: {np.median(scsr_values):.4f}')
    print(f'标准差: {scsr_values.std():.4f}')
    print(f'范围: [{scsr_values.min():.4f}, {scsr_values.max():.4f}]')

    # 单样本 t 检验 vs 0.5
    t_stat, p_val = stats.ttest_1samp(scsr_values, 0.5)
    cohen_d = (scsr_values.mean() - 0.5) / scsr_values.std()

    print(f'\n单样本 t 检验 vs 0.5:')
    print(f'  t({len(scsr_values)-1}) = {t_stat:.3f}, p = {p_val:.4e}')
    print(f'  Cohen\'s d = {cohen_d:.3f}')
    print(f'  方向: {"SCSR < 0.5 (倾向选美味，自控失败倾向)" if t_stat < 0 else "SCSR > 0.5 (倾向选健康)"}')

    # Wilcoxon 符号秩检验（非参数）
    w_stat, w_p = stats.wilcoxon(scsr_values - 0.5)
    print(f'\nWilcoxon 符号秩检验:')
    print(f'  W = {w_stat:.3f}, p = {w_p:.4e}')

    # 按中位数分组：高 SCSR vs 低 SCSR
    median_scsr = np.median(scsr_values)
    scsr_df['scsr_group'] = np.where(scsr_df['scsr'] >= median_scsr, 'high_scsr', 'low_scsr')

    return {
        'scsr_df': scsr_df,
        'mean': scsr_values.mean(),
        'median': np.median(scsr_values),
        'std': scsr_values.std(),
        't_stat': t_stat,
        'p_value': p_val,
        'cohen_d': cohen_d,
        'median_split': median_scsr,
    }


def compare_high_low_scsr(df, scsr_results):
    """比较高/低 SCSR 被试在 taste/health 权重上的差异"""
    print('\n' + '=' * 60)
    print('高/低 SCSR 被试的 taste/health 权重对比')
    print('=' * 60)

    scsr_df = scsr_results['scsr_df']
    median_scsr = scsr_results['median_split']

    high_subs = scsr_df[scsr_df['scsr'] >= median_scsr]['subject'].values
    low_subs = scsr_df[scsr_df['scsr'] < median_scsr]['subject'].values

    print(f'高 SCSR 组: {len(high_subs)} 人 (SCSR >= {median_scsr:.3f})')
    print(f'低 SCSR 组: {len(low_subs)} 人 (SCSR < {median_scsr:.3f})')

    # 各组分别拟合 logistic
    high_df = df[df['subject'].isin(high_subs)]
    low_df = df[df['subject'].isin(low_subs)]

    group_results = {}
    for name, sub_df in [('high_scsr', high_df), ('low_scsr', low_df)]:
        m = smf.logit('chose_stim1 ~ taste_diff + health_diff', data=sub_df).fit(disp=0)
        group_results[name] = {
            'coef_taste': m.params['taste_diff'],
            'coef_health': m.params['health_diff'],
            'p_taste': m.pvalues['taste_diff'],
            'p_health': m.pvalues['health_diff'],
            'pseudo_r2': m.prsquared,
        }
        print(f'\n{name} (n={len(sub_df.subject.unique())}):')
        print(f'  taste 系数: {m.params["taste_diff"]:.4f} (p={m.pvalues["taste_diff"]:.4e})')
        print(f'  health 系数: {m.params["health_diff"]:.4f} (p={m.pvalues["health_diff"]:.4e})')
        print(f'  pseudo R²: {m.prsquared:.4f}')

    # taste/health 权重比
    high_ratio = group_results['high_scsr']['coef_health'] / group_results['high_scsr']['coef_taste']
    low_ratio = group_results['low_scsr']['coef_health'] / group_results['low_scsr']['coef_taste']
    print(f'\n权重比 (health/taste):')
    print(f'  高 SCSR: {high_ratio:.4f}')
    print(f'  低 SCSR: {low_ratio:.4f}')

    return group_results


def plot_choice_curves(df, results_path):
    """绘制选择曲线（taste/health 对选择概率的影响）"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 左图: P(chose_stim1) vs taste_diff，按 health_diff 分层
    ax = axes[0]
    bins = np.linspace(-100, 100, 21)
    for h_label, h_range, color in [('<0 (stim1 less healthy)', (-100, -10), 'red'),
                                     ('~0 (neutral)', (-10, 10), 'gray'),
                                     ('>0 (stim1 more healthy)', (10, 100), 'green')]:
        mask = (df['health_diff'] >= h_range[0]) & (df['health_diff'] < h_range[1])
        sub = df[mask].copy()
        if len(sub) == 0:
            continue
        sub['taste_bin'] = pd.cut(sub['taste_diff'], bins=bins)
        agg = sub.groupby('taste_bin')['chose_stim1'].agg(['mean', 'count']).reset_index()
        agg['taste_mid'] = [(v.left + v.right) / 2 for v in agg['taste_bin']]
        valid = agg['count'] >= 10
        ax.plot(agg.loc[valid, 'taste_mid'], agg.loc[valid, 'mean'],
                'o-', label=h_label, color=color, markersize=5)
    ax.set_xlabel('taste_diff (stim1_taste - stim2_taste)')
    ax.set_ylabel('P(choose stim1)')
    ax.set_title('Taste effect on choice\n(stratified by health_diff)')
    ax.legend(loc='upper left', fontsize=9)
    ax.axhline(0.5, color='black', linestyle='--', alpha=0.5)
    ax.axvline(0, color='black', linestyle='--', alpha=0.5)
    ax.set_ylim(-0.05, 1.05)

    # 右图: P(chose_stim1) vs health_diff，按 taste_diff 分层
    ax = axes[1]
    for t_label, t_range, color in [('<0 (stim1 less tasty)', (-100, -10), 'red'),
                                     ('~0 (neutral)', (-10, 10), 'gray'),
                                     ('>0 (stim1 more tasty)', (10, 100), 'green')]:
        mask = (df['taste_diff'] >= t_range[0]) & (df['taste_diff'] < t_range[1])
        sub = df[mask].copy()
        if len(sub) == 0:
            continue
        sub['health_bin'] = pd.cut(sub['health_diff'], bins=bins)
        agg = sub.groupby('health_bin')['chose_stim1'].agg(['mean', 'count']).reset_index()
        agg['health_mid'] = [(v.left + v.right) / 2 for v in agg['health_bin']]
        valid = agg['count'] >= 10
        ax.plot(agg.loc[valid, 'health_mid'], agg.loc[valid, 'mean'],
                'o-', label=t_label, color=color, markersize=5)
    ax.set_xlabel('health_diff (stim1_health - stim2_health)')
    ax.set_ylabel('P(choose stim1)')
    ax.set_title('Health effect on choice\n(stratified by taste_diff)')
    ax.legend(loc='upper left', fontsize=9)
    ax.axhline(0.5, color='black', linestyle='--', alpha=0.5)
    ax.axvline(0, color='black', linestyle='--', alpha=0.5)
    ax.set_ylim(-0.05, 1.05)

    plt.tight_layout()
    plt.savefig(results_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'[fig] saved: {results_path}')


def plot_scsr_distribution(scsr_results, results_path):
    """绘制 SCSR 分布直方图"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 左图: SCSR 直方图
    ax = axes[0]
    scsr_values = scsr_results['scsr_df']['scsr'].values
    ax.hist(scsr_values, bins=20, color='steelblue', edgecolor='white', alpha=0.8)
    ax.axvline(0.5, color='red', linestyle='--', label='0.5 (random level)', linewidth=2)
    ax.axvline(scsr_results['mean'], color='orange', linestyle='-',
               label=f'mean = {scsr_results["mean"]:.3f}', linewidth=2)
    ax.axvline(scsr_results['median'], color='green', linestyle='-',
               label=f'median = {scsr_results["median"]:.3f}', linewidth=2)
    ax.set_xlabel('Self-Control Success Ratio (SCSR)')
    ax.set_ylabel('Number of subjects')
    ax.set_title(f'SCSR distribution (n={len(scsr_values)})\n'
                 f't({len(scsr_values)-1})={scsr_results["t_stat"]:.2f}, '
                 f'p={scsr_results["p_value"]:.2e}, d={scsr_results["cohen_d"]:.2f}')
    ax.legend(loc='upper right')

    # 右图: 美味-健康选择比例（conflict 试次）
    ax = axes[1]
    conflict_df = scsr_results['scsr_df']
    chose_healthy = conflict_df['scsr'].values
    chose_tasty = 1 - chose_healthy
    mean_healthy = chose_healthy.mean()
    mean_tasty = chose_tasty.mean()

    bars = ax.bar(['Choose tastier\n(System 1)', 'Choose healthier\n(System 2)'],
                 [mean_tasty, mean_healthy],
                 color=['#E07B39', '#3D7A8E'], alpha=0.85, edgecolor='black')
    ax.set_ylabel('Choice proportion (conflict trials)')
    ax.set_title('Choice tendency in conflict trials')
    ax.set_ylim(0, 1)
    for bar, val in zip(bars, [mean_tasty, mean_healthy]):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.02,
                f'{val:.3f}', ha='center', fontsize=12, fontweight='bold')
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(results_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'[fig] saved: {results_path}')


def plot_weight_comparison(group_results, results_path):
    """绘制高/低 SCSR 组的 taste/health 权重对比"""
    fig, ax = plt.subplots(figsize=(8, 5))

    groups = ['low_scsr', 'high_scsr']
    taste_coefs = [group_results[g]['coef_taste'] for g in groups]
    health_coefs = [group_results[g]['coef_health'] for g in groups]

    x = np.arange(len(groups))
    width = 0.35
    bars1 = ax.bar(x - width / 2, taste_coefs, width, label='taste coef',
                   color='#E07B39', alpha=0.85, edgecolor='black')
    bars2 = ax.bar(x + width / 2, health_coefs, width, label='health coef',
                   color='#3D7A8E', alpha=0.85, edgecolor='black')

    ax.set_ylabel('Logistic regression coefficient')
    ax.set_title('taste / health weights: low vs high SCSR groups')
    ax.set_xticks(x)
    ax.set_xticklabels(['Low SCSR group\n(tasty-led)', 'High SCSR group\n(healthy-led)'])
    ax.legend()
    ax.axhline(0, color='black', linewidth=0.8)

    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.001 * (1 if h >= 0 else -1),
                    f'{h:.4f}', ha='center', fontsize=9,
                    va='bottom' if h >= 0 else 'top')

    plt.tight_layout()
    plt.savefig(results_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'[fig] saved: {results_path}')


def run_baseline_analysis():
    """主入口：运行 baseline 行为分析"""
    print('=' * 60)
    print('FoodEEG 行为数据 Baseline 分析')
    print('切入点：taste / health 对食物决策的影响')
    print('=' * 60)

    # 1. 加载数据
    df = load_all_pairedchoice()
    print(f'\n总试次数: {len(df)}')

    # 2. Pooled logistic 模型
    pooled_results = fit_pooled_logistic(df)

    # 3. SCSR 分布检验
    scsr_results = test_scsr_distribution(df)

    # 4. 高/低 SCSR 组对比
    group_results = compare_high_low_scsr(df, scsr_results)

    # 5. 可视化
    plot_choice_curves(df, FIGURES_DIR / 'fig1_choice_curves.png')
    plot_scsr_distribution(scsr_results, FIGURES_DIR / 'fig2_scsr_distribution.png')
    plot_weight_comparison(group_results, FIGURES_DIR / 'fig3_weight_comparison.png')

    # 6. 保存结果摘要
    summary = {
        'pooled': {
            'M1_taste': {
                'coef': pooled_results['M1_taste']['coef'],
                'p': float(pooled_results['M1_taste']['p']),
                'pseudo_r2': float(pooled_results['M1_taste']['pseudo_r2']),
            },
            'M2_health': {
                'coef': pooled_results['M2_health']['coef'],
                'p': float(pooled_results['M2_health']['p']),
                'pseudo_r2': float(pooled_results['M2_health']['pseudo_r2']),
            },
            'M3_combined': {
                'coef_taste': float(pooled_results['M3_combined']['coef_taste']),
                'coef_health': float(pooled_results['M3_combined']['coef_health']),
                'p_taste': float(pooled_results['M3_combined']['p_taste']),
                'p_health': float(pooled_results['M3_combined']['p_health']),
                'pseudo_r2': float(pooled_results['M3_combined']['pseudo_r2']),
            },
        },
        'scsr': {
            'n': int(len(scsr_results['scsr_df'])),
            'mean': float(scsr_results['mean']),
            'median': float(scsr_results['median']),
            'std': float(scsr_results['std']),
            't_stat': float(scsr_results['t_stat']),
            'p_value': float(scsr_results['p_value']),
            'cohen_d': float(scsr_results['cohen_d']),
        },
        'groups': {
            g: {k: float(v) for k, v in res.items()}
            for g, res in group_results.items()
        },
    }

    import json
    summary_path = RESULTS_DIR / 'baseline_summary.json'
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f'\n[结果] 摘要保存: {summary_path}')

    return summary


if __name__ == '__main__':
    summary = run_baseline_analysis()
    print('\n' + '=' * 60)
    print('分析完成')
    print('=' * 60)
