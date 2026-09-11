# -*- coding: utf-8 -*-
"""
attach_ratings.py
=================
为已经预处理好的 stim_epochs 附加每张食物图的连续评分标签。

数据流：
  1. stim_epochs['food_image_code'] = ['foodImage.85', 'foodImage.63', ...]
     → 提取数字 stimCode (85, 63)
  2. 读取 E:\\datasets\\foodeeg-ds007012\\code\\ratings\\in-person\\FOODEEG_pXXX_ratings.csv
     → 含 stimCode(1-120), rating_healthiness(0-100), rating_tastiness(0-100),
       rating_willingnessToEat(0-100)
  3. 按 stimCode 关联，得到每 trial 的 3 个评分
  4. 保存为新的 *_stim_labels_with_ratings.npz（不动原始 labels.npz）
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import json
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

from config import VALID_SUBJECTS, DATA_DIR
from eeg_preprocessing import EPOCHS_DIR


# ratings 文件夹（数据集根目录下的 code/ratings/in-person，跨平台）
RATINGS_DIR = DATA_DIR / 'code' / 'ratings' / 'in-person'


def load_subject_ratings(subject_id):
    """加载某被试的连续评分

    返回 dict {stimCode(int 1-120): (health, taste, wte)}
    """
    # 提取数字编号: 'sub-007' -> 7
    num = int(subject_id.split('-')[1])
    f = RATINGS_DIR / f'FOODEEG_p{num:03d}_ratings.csv'
    if not f.exists():
        return None
    df = pd.read_csv(f)
    # 列: stimCode, stimPath, rating_healthiness, rating_tastiness, rating_willingnessToEat
    out = {}
    for _, row in df.iterrows():
        code = int(row['stimCode'])
        out[code] = (
            float(row['rating_healthiness']),
            float(row['rating_tastiness']),
            float(row['rating_willingnessToEat']),
        )
    return out


def attach_ratings_to_subject(subject_id, force=False):
    """为单个被试的 stim_labels 附加连续评分

    返回：n_attached, n_total, n_no_match
    """
    l_path = EPOCHS_DIR / f'{subject_id}_stim_labels.npz'
    out_path = EPOCHS_DIR / f'{subject_id}_stim_labels_with_ratings.npz'

    if out_path.exists() and not force:
        print(f'[skip] {subject_id}: 已存在 {out_path.name}')
        return None

    if not l_path.exists():
        print(f'[skip] {subject_id}: {l_path.name} 不存在')
        return None

    labels = dict(np.load(l_path, allow_pickle=True))
    food_codes = labels['food_image_code']

    ratings = load_subject_ratings(subject_id)
    if ratings is None:
        print(f'[skip] {subject_id}: ratings 文件缺失')
        return None

    n_total = len(food_codes)
    n_no_match = 0
    health = np.full(n_total, np.nan, dtype=np.float32)
    taste = np.full(n_total, np.nan, dtype=np.float32)
    wte = np.full(n_total, np.nan, dtype=np.float32)

    for i, code in enumerate(food_codes):
        # code 形如 'foodImage.85'
        try:
            num = int(code.split('.')[1])
        except Exception:
            n_no_match += 1
            continue
        if num in ratings:
            h, t, w = ratings[num]
            health[i] = h
            taste[i] = t
            wte[i] = w
        else:
            n_no_match += 1

    n_attached = int((~np.isnan(health)).sum())

    # 合并保存
    new_labels = {k: v for k, v in labels.items()}
    new_labels['rating_health'] = health
    new_labels['rating_taste'] = taste
    new_labels['rating_willingnessToEat'] = wte
    np.savez(out_path, **new_labels)

    return n_attached, n_total, n_no_match


def main():
    print('=' * 60)
    print('attach_ratings: 为 stim epochs 附加连续评分')
    print('=' * 60)

    summary = []
    for sid in VALID_SUBJECTS:
        result = attach_ratings_to_subject(sid)
        if result is None:
            continue
        n_att, n_tot, n_miss = result
        summary.append({
            'subject_id': sid,
            'n_total': n_tot,
            'n_attached': n_att,
            'n_no_match': n_miss,
        })

    df = pd.DataFrame(summary)
    df.to_csv(EPOCHS_DIR / 'ratings_attachment_summary.csv', index=False)

    print(f'\n[完成] 处理 {len(summary)} 名被试')
    if len(df):
        print(f'  平均附加率: {(df.n_attached / df.n_total).mean()*100:.1f}%')
        print(f'  总 trial 数: {df.n_total.sum()}')
        print(f'  关联成功: {df.n_attached.sum()}')
        print(f'  未匹配: {df.n_no_match.sum()}')
    print(f'\n[汇总] {EPOCHS_DIR / "ratings_attachment_summary.csv"}')
    return df


if __name__ == '__main__':
    main()