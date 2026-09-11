# -*- coding: utf-8 -*-
"""
check_channel_order.py
======================
诊断脚本：验证 ROI 通道覆盖率。

直接从 eeg_preprocessing.py 的 EEG_CHANNEL_NAMES 字符串复制（避免 import mne 慢）。
"""
import numpy as np

# 直接从 eeg_preprocessing.py:55-63 复制的真实通道顺序（不变）
EEG_CHANNEL_NAMES = [
    'Fp1', 'AF7', 'AF3', 'F1', 'F3', 'F5', 'F7', 'FT7', 'FC5', 'FC3', 'FC1',
    'C1', 'C3', 'C5', 'T7', 'TP7', 'CP5', 'CP3', 'CP1', 'P1', 'P3', 'P5', 'P7',
    'P9', 'PO7', 'PO3', 'O1', 'Iz', 'Oz', 'POz', 'Pz', 'CPz', 'Fpz', 'Fp2',
    'AF8', 'AF4', 'AFz', 'Fz', 'F2', 'F4', 'F6', 'F8', 'FT8', 'FC6', 'FC4',
    'FC2', 'FCz', 'Cz', 'C2', 'C4', 'C6', 'T8', 'TP8', 'CP6', 'CP4', 'CP2',
    'P2', 'P4', 'P6', 'P8', 'P10', 'PO8', 'PO4', 'O2'
]

print('=' * 60)
print('实际 epoch 通道顺序')
print('=' * 60)
print(f'总通道数: {len(EEG_CHANNEL_NAMES)}')
for i, ch in enumerate(EEG_CHANNEL_NAMES):
    print(f'  [{i:2d}] {ch}')

# ROI 分组（标准 10-20 系统命名）
ROI_CHANNELS = {
    'frontal':         ['Fp1', 'Fp2', 'Fpz', 'AFz', 'AF3', 'AF4', 'AF7', 'AF8',
                        'Fz', 'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8'],
    'fronto-central':  ['Fz', 'FCz', 'FC1', 'FC2', 'FC3', 'FC4', 'FC5', 'FC6'],
    'central':         ['Fz', 'FCz', 'Cz', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6'],
    'centro-parietal': ['FCz', 'Cz', 'CPz', 'CP1', 'CP2', 'CP3', 'CP4', 'CP5', 'CP6',
                        'P1', 'P2'],
    'parietal':        ['CPz', 'Pz', 'P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8',
                        'P9', 'P10', 'POz', 'PO3', 'PO4', 'PO7', 'PO8'],
    'occipital':       ['POz', 'PO3', 'PO4', 'PO7', 'PO8', 'Oz', 'O1', 'O2', 'Iz'],
    'temporal':        ['FT7', 'FT8', 'T7', 'T8', 'TP7', 'TP8'],
}

print('\n' + '=' * 60)
print('ROI 通道覆盖率')
print('=' * 60)
name_to_idx = {n: i for i, n in enumerate(EEG_CHANNEL_NAMES)}
print(f'\n{"ROI":<18} {"找到":>6} {"缺失":>6} {"缺失通道"}')
for roi_name, roi_chs in ROI_CHANNELS.items():
    idx = [name_to_idx[n] for n in roi_chs if n in name_to_idx]
    missing = [n for n in roi_chs if n not in name_to_idx]
    print(f'{roi_name:<18} {len(idx):>6} {len(missing):>6} {missing}')

print('\n数据中存在的通道全部名称:')
print(EEG_CHANNEL_NAMES)