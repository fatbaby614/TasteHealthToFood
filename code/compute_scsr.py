# -*- coding: utf-8 -*-
"""
compute_scsr.py
================
一次性计算所有被试的 SCSR 并保存为 csv（供后续脚本使用）。
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from load_behavior import load_all_pairedchoice, compute_self_control_success
from config import RESULTS_DIR
import pandas as pd

df = load_all_pairedchoice()
scsr = compute_self_control_success(df)
scsr['subject'] = scsr['subject'].astype(str).apply(
    lambda x: f'sub-{int(x):03d}' if not str(x).startswith('sub-') else x)
scsr.to_csv(RESULTS_DIR / 'scsr_per_subject.csv', index=False)
print(f'\n[saved] {RESULTS_DIR / "scsr_per_subject.csv"}')
print(scsr.describe())
print(f'\n有效 SCSR 被试: {len(scsr)}')
print(f'group sizes: high={(scsr.scsr >= scsr.scsr.median()).sum()}, '
      f'low={(scsr.scsr < scsr.scsr.median()).sum()}')