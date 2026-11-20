# -*- coding: utf-8 -*-
"""
TasteHealthToFood 项目配置
课题：美味与健康对食物决策的影响
数据集：FoodEEG (ds007012)

切入点 1：用 EEG 时间频率动力学验证 Sullivan (2015) 属性加工时序假说
切入点 2：Riemannian 几何单试次解码美味-健康决策权衡
"""

from pathlib import Path

# ============ 路径 ============
# 原始数据集路径（跨平台自动检测，也可用环境变量 DATA_DIR 覆盖）
import os

_DATA_DIR_ENV = os.environ.get('DATA_DIR', '')
_WIN_DATA_DIR = Path(r'E:\datasets\foodeeg-ds007012')
_LINUX_DATA_DIR = Path('/mnt/data1/home/tanhuang/datasets/foodeeg-ds007012')

if _DATA_DIR_ENV:
    DATA_DIR = Path(_DATA_DIR_ENV)
elif _WIN_DATA_DIR.exists():
    DATA_DIR = _WIN_DATA_DIR
else:
    DATA_DIR = _LINUX_DATA_DIR

# 项目根目录（config.py 位于 code/，需要上一级）
PROJECT_ROOT = Path(__file__).parent.parent.resolve()

# 输出目录（可用环境变量 OUTPUT_DIR 覆盖，便于测试隔离）
_OUTPUT_DIR_ENV = os.environ.get('OUTPUT_DIR', '')
if _OUTPUT_DIR_ENV:
    OUTPUT_DIR = Path(_OUTPUT_DIR_ENV)
else:
    OUTPUT_DIR = PROJECT_ROOT / 'outputs'
RESULTS_DIR = OUTPUT_DIR / 'results'
FIGURES_DIR = OUTPUT_DIR / 'figures'

# 创建输出目录
for d in [OUTPUT_DIR, RESULTS_DIR, FIGURES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ============ 被试 ============
# 所有可用被试 ID（sub-006 ~ sub-122，共 117 人）
def get_all_subjects():
    """返回所有有 paired choice 数据的被试 ID"""
    subs = []
    for sub_dir in sorted(DATA_DIR.iterdir()):
        if not sub_dir.is_dir() or not sub_dir.name.startswith('sub-'):
            continue
        beh_file = sub_dir / 'beh' / f'{sub_dir.name}_task-pairedchoice_beh.tsv'
        if beh_file.exists():
            subs.append(sub_dir.name)
    return subs

ALL_SUBJECTS = get_all_subjects()

# 行为分析排除的被试（仅排除行为数据本身不可靠的）
# 注：EEG 噪声的被试（如 sub-010, sub-039, sub-050）其行为数据仍可用
# 仅排除 paired choice 与连续评分高度不一致的 sub-027, sub-096
EXCLUDED_BEHAVIOR = ['sub-027', 'sub-096']

def get_valid_subjects():
    """行为分析用的有效被试"""
    return [s for s in ALL_SUBJECTS if s not in EXCLUDED_BEHAVIOR]

VALID_SUBJECTS = get_valid_subjects()

# ============ 任务参数 ============
# paired choice 任务：每对食物选择更愿意吃的
# 评分为 0-100 连续量表
PAIRED_CHOICE_TRIALS = 300  # 标准 trial 数

# 评分维度
RATING_DIMENSIONS = ['healthRating', 'tasteRating', 'willingToEatRating']

# ============ 统计参数 ============
ALPHA = 0.05  # 显著性水平

# ============ 输出 ============
print(f"[config] 数据集路径: {DATA_DIR}")
print(f"[config] 项目输出目录: {OUTPUT_DIR}")
print(f"[config] 总被试数: {len(ALL_SUBJECTS)}")
print(f"[config] 行为分析有效被试数: {len(VALID_SUBJECTS)} (排除 {len(EXCLUDED_BEHAVIOR)} 人)")


# ============ 结果文件合并写入 ============
def merge_results_csv(new_df, path, key_cols):
    """逐被试结果 CSV 的合并写入：部分被试运行（如 --quick）不覆盖全量结果。

    规则（按被试数判断本次是否为部分运行）：
    - 已有文件的被试数 > 本次运行 → 部分运行 → 旧行优先（keep='first'），
      避免少量被试的临时结果覆盖全量有效行；
    - 否则（全量重跑或首次生成）→ 新行优先（keep='last'）。
    合并键 key_cols[0] 须为被试 ID 列（'subject' 或 'subject_id'）。
    """
    import pandas as pd

    if path.exists():
        try:
            existing = pd.read_csv(path)
            if (all(c in existing.columns for c in key_cols)
                    and all(c in new_df.columns for c in key_cols)
                    and key_cols[0] in existing.columns
                    and key_cols[0] in new_df.columns):
                n_old = existing[key_cols[0]].nunique()
                n_new = new_df[key_cols[0]].nunique()
                keep = 'first' if n_old > n_new else 'last'
                if n_old > n_new:
                    print(f'[merge] 已有 {n_old} 被试 > 本次 {n_new} 被试，'
                          f'保留已有结果，仅补充新被试: {path.name}')
                combined = pd.concat([existing, new_df], ignore_index=True)
                return combined.drop_duplicates(subset=key_cols, keep=keep)
        except Exception as e:
            print(f'[warn] 合并已有结果失败，将直接覆盖 {path.name}: {e}')
    return new_df
