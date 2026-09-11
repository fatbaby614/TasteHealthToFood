# -*- coding: utf-8 -*-
"""
行为数据加载与清洗模块

paired choice 任务结构（events.tsv）:
- blockNo, trialNoPerBlock
- stim1_path, stim1_healthRating, stim1_tasteRating, stim1_willingToEatRating
- stim2_path, stim2_healthRating, stim2_tasteRating, stim2_willingToEatRating
- stimChosen_path, respKey (m=stim1, z=stim2), respRT

baseline 模型构造的试次变量:
- chose_stim1: 0/1 二元选择结果
- taste_diff: stim1_taste - stim2_taste （正值=stim1更美味）
- health_diff: stim1_health - stim2_health （正值=stim1更健康）
- wte_diff: stim1_willingToEat - stim2_willingToEat
- |taste_diff|, |health_diff|: 冲突强度
- conflict_type: 一致 / taste-led / health-led / 冲突
"""

import pandas as pd
import numpy as np
from pathlib import Path
from config import DATA_DIR, VALID_SUBJECTS, RATING_DIMENSIONS


def load_subject_pairedchoice(subject_id):
    """加载单个被试的 paired choice 行为数据

    参数
    ----
    subject_id : str, 如 'sub-007'

    返回
    ----
    df : pandas.DataFrame，含原始列 + 构造的试次变量
    """
    f = DATA_DIR / subject_id / 'beh' / f'{subject_id}_task-pairedchoice_beh.tsv'
    df = pd.read_csv(f, sep='\t')

    # 选择结果：stimChosen_path 与 stim1_path 比较
    df['chose_stim1'] = (df['stimChosen_path'] == df['stim1_path']).astype(int)

    # 评分差异变量（正值=stim1 占优）
    df['taste_diff'] = df['stim1_tasteRating'] - df['stim2_tasteRating']
    df['health_diff'] = df['stim1_healthRating'] - df['stim2_healthRating']
    df['wte_diff'] = df['stim1_willingToEatRating'] - df['stim2_willingToEatRating']

    # 冲突强度（绝对值）
    df['taste_diff_abs'] = df['taste_diff'].abs()
    df['health_diff_abs'] = df['health_diff'].abs()

    # 试次类型分类
    # - 一致 (congruent): 两个维度差异方向相同
    # - taste-led: 被试选了更美味但更不健康的
    # - health-led: 被试选了更健康但更不美味的
    # - 冲突 (conflict): 两个维度差异方向相反，且都较大
    def classify_trial(row):
        t, h = row['taste_diff'], row['health_diff']
        # 阈值：评分差 >= 10 视为"有差异"
        thresh = 10
        t_dom = abs(t) >= thresh
        h_dom = abs(h) >= thresh
        if not t_dom and not h_dom:
            return 'low_diff'
        if t * h > 0:
            return 'congruent'  # 两维度方向一致
        if t * h < 0:
            return 'conflict'   # 两维度方向相反
        return 'neutral'

    df['trial_type'] = df.apply(classify_trial, axis=1)

    # 选择的"维度驱动"：被试选了更美味还是更健康？
    # 注：taste_diff>0 表示 stim1 更美味
    # 若 chose_stim1=1 且 taste_diff>0 → 被试选了更美味
    # 若 chose_stim1=0 且 taste_diff<0 → 被试也选了更美味（stim2 更美味）
    df['chose_tasty'] = ((df['chose_stim1'] == 1) & (df['taste_diff'] > 0)) | \
                        ((df['chose_stim1'] == 0) & (df['taste_diff'] < 0))
    df['chose_tasty'] = df['chose_tasty'].astype(int)

    df['chose_healthy'] = ((df['chose_stim1'] == 1) & (df['health_diff'] > 0)) | \
                         ((df['chose_stim1'] == 0) & (df['health_diff'] < 0))
    df['chose_healthy'] = df['chose_healthy'].astype(int)

    df['subject'] = subject_id
    return df


def load_all_pairedchoice():
    """加载所有有效被试的 paired choice 数据，合并为单个 DataFrame"""
    all_dfs = []
    n_loaded = 0
    for s in VALID_SUBJECTS:
        try:
            df = load_subject_pairedchoice(s)
            all_dfs.append(df)
            n_loaded += 1
        except Exception as e:
            print(f'[load] 警告: {s} 加载失败: {e}')
    combined = pd.concat(all_dfs, ignore_index=True)
    print(f'[load] 成功加载 {n_loaded} 名被试，共 {len(combined)} trials')
    return combined


def compute_self_control_success(df):
    """计算每个被试的自控成功率 (SCSR)

    定义：在 conflict 试次中（美味与健康方向相反），
    被试选择更健康食物的比例。

    Sullivan 2015 / Hare 2009 框架：
    - 高 SCSR = 自控成功者（更倾向选择健康）
    - 低 SCSR = 非自控者（更倾向选择美味）
    """
    conflict_df = df[df['trial_type'] == 'conflict'].copy()
    if len(conflict_df) == 0:
        return pd.DataFrame(columns=['subject', 'scsr', 'n_conflict'])

    scsr = conflict_df.groupby('subject')['chose_healthy'].agg(['mean', 'count']).reset_index()
    scsr.columns = ['subject', 'scsr', 'n_conflict']
    return scsr


if __name__ == '__main__':
    # 快速自检
    df = load_all_pairedchoice()
    print('\n[检查] 数据基本信息:')
    print(df.groupby('subject').size().describe())
    print(f'\n[检查] 选择 stim1 的总比例: {df.chose_stim1.mean():.3f}')
    print(f'[检查] 选择更美味的比例: {df.chose_tasty.mean():.3f}')
    print(f'[检查] 选择更健康的比例: {df.chose_healthy.mean():.3f}')
    print(f'\n[检查] 试次类型分布:')
    print(df.trial_type.value_counts())

    scsr = compute_self_control_success(df)
    print(f'\n[检查] 自控成功率 (SCSR) 描述统计:')
    print(scsr.scsr.describe())
