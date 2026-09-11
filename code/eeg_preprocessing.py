# -*- coding: utf-8 -*-
"""
EEG 预处理 pipeline（基于 categorisation 任务）

课题支撑：
- 切入点 1：对比 taste vs health 判断的 ERP/时频时间动力学
  → 验证 Sullivan (2015) "tastiness 先于 healthiness 加工" 假说
- 切入点 2：Riemannian 几何单试次解码 + 跨任务迁移预测 paired choice 行为

设计要点：
1. 沿用 FoodFBTS 预处理参数（已验证可靠）：
   - 64 通道 Biosemi 系统，平均参考
   - 0.5-45 Hz 带通滤波
   - epoch: -0.2 ~ 1.0 s, baseline: -0.2 ~ 0 s
   - reject: ±100 µV
2. 扩展试次标签（新课题需要）：
   - 基础: trial_type (health/taste/decision), resp_choice (Yes/No), choice_rt
   - 关联: food_image_code（与 paired choice 评分连接）
   - 新增: response_locked epoch（用于分析决策阶段神经活动）
3. 输出兼容 FoodFBTS 格式（.npy + .npz），便于复用下游 FBTS/Riemannian 代码

注：paired choice 任务无 EEG 数据，只有行为。
EEG 分析完全基于 categorisation 任务（同一张食物图在 3 个判断维度各呈现一次）。
"""

import numpy as np
import pandas as pd
import mne
import os
import platform
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

mne.set_log_level('WARNING')

# ============ 配置 ============

if platform.system() == 'Windows':
    _default_dataset = Path(r'E:\datasets\foodeeg-ds007012')
else:
    _default_dataset = Path('/mnt/data1/home/tanhuang/datasets/foodeeg-ds007012')

DATA_ROOT = Path(os.environ.get('FOODEEG_DATASET_DIR', str(_default_dataset)))

# 输出目录（与 FoodFBTS 分开，便于区分）
# eeg_preprocessing.py 位于 code/，项目根目录需要上一级
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
EPOCHS_DIR = PROJECT_ROOT / 'data' / 'epochs'
QC_DIR = PROJECT_ROOT / 'outputs' / 'qc'
for d in [EPOCHS_DIR, QC_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# 64 通道 Biosemi 顺序
EEG_CHANNEL_NAMES = [
    'Fp1', 'AF7', 'AF3', 'F1', 'F3', 'F5', 'F7', 'FT7', 'FC5', 'FC3', 'FC1',
    'C1', 'C3', 'C5', 'T7', 'TP7', 'CP5', 'CP3', 'CP1', 'P1', 'P3', 'P5', 'P7',
    'P9', 'PO7', 'PO3', 'O1', 'Iz', 'Oz', 'POz', 'Pz', 'CPz', 'Fpz', 'Fp2',
    'AF8', 'AF4', 'AFz', 'Fz', 'F2', 'F4', 'F6', 'F8', 'FT8', 'FC6', 'FC4',
    'FC2', 'FCz', 'Cz', 'C2', 'C4', 'C6', 'T8', 'TP8', 'CP6', 'CP4', 'CP2',
    'P2', 'P4', 'P6', 'P8', 'P10', 'PO8', 'PO4', 'O2'
]

# EEG 排除被试（沿用 FoodFBTS 验证过的清单）
# 这些被试 EEG 质量问题（covariance non-PD 等），但行为数据仍可用
EEG_EXCLUDED_SUBJECTS = {
    'sub-010', 'sub-011', 'sub-012',   # 上次预处理失败
    'sub-027', 'sub-039', 'sub-050',    # 行为不可靠 / EEG 噪声
    'sub-096', 'sub-100',               # covariance non-PD across all tasks
}

# 官方坏通道清单（提取自 FOODEEG_bad_channels.m）
# 仅包含 64 通道 EEG（忽略 EXG 外部通道）
# 这些通道在滤波后、epoching 前用 interpolate_bads() 插值
OFFICIAL_BAD_CHANNELS = {
    'sub-007': ['F5', 'F7'],
    'sub-008': ['T7', 'F5', 'F7', 'F6', 'F8'],
    'sub-009': ['T7'],
    'sub-019': ['F8', 'AF8'],
    'sub-026': ['C4', 'FT8'],
    'sub-027': ['FC2'],
    'sub-033': ['FT7'],
    'sub-042': ['FT8', 'F7'],
    'sub-052': ['POz'],
    'sub-071': ['POz', 'P2'],
    'sub-080': ['AFz'],
    'sub-083': ['FC6'],
    'sub-086': ['Fp2', 'AF8'],
    'sub-101': ['FT8', 'CP4'],
    'sub-113': ['F7'],
}


def get_eeg_valid_subjects():
    """返回 EEG 分析可用的被试 ID"""
    subs = []
    for d in sorted(DATA_ROOT.glob('sub-*')):
        sid = d.name
        if sid in EEG_EXCLUDED_SUBJECTS:
            continue
        bdf = d / 'eeg' / f'{sid}_task-categorisation_eeg.bdf'
        if bdf.exists():
            subs.append(sid)
    return subs


# ============ 预处理参数 ============

# Stimulus-locked epoch（食物图呈现）
TMIN_STIM = -0.2
TMAX_STIM = 1.0
BASELINE = (-0.2, 0)

# Response-locked epoch（按键决策）
TMIN_RESP = -0.8
TMAX_RESP = 0.2
BASELINE_RESP = (-0.8, -0.6)  # 决策前 600-800ms 作为基线

LFREQ = 0.5
HFREQ = 45
REJECT_THRESHOLD = 100e-6  # ±100 µV


# ============ 预处理函数 ============

def load_and_filter_raw(subject_id):
    """读取 BDF 并滤波

    步骤：
    1. 读取 BDF，pick 64 EEG 通道
    2. 标记官方坏通道（FOODEEG_bad_channels.m）
    3. 自动检测额外坏通道（基于无 reject epoch 的 peak-to-peak 统计）
    4. 插值所有坏通道
    5. 平均参考 + 带通滤波
    """
    bdf_path = DATA_ROOT / subject_id / 'eeg' / f'{subject_id}_task-categorisation_eeg.bdf'
    raw = mne.io.read_raw_bdf(bdf_path, preload=True, verbose='ERROR')
    raw.pick_channels(EEG_CHANNEL_NAMES)

    # 设置标准 Biosemi64 montage（BDF 不含 digitization 信息）
    montage = mne.channels.make_standard_montage('biosemi64')
    raw.set_montage(montage, verbose='ERROR')

    # 1. 标记官方坏通道
    bads = list(OFFICIAL_BAD_CHANNELS.get(subject_id, []))
    # 只保留实际存在的通道
    bads = [ch for ch in bads if ch in raw.ch_names]
    raw.info['bads'] = bads

    # 2. 自动检测：先做一次临时 epoch 统计 peak-to-peak
    # 用 1s 窗口扫描整个 raw，找持续高幅值通道
    try:
        sfreq = raw.info['sfreq']
        # 创建扫描事件（每 2s 一个窗口）
        n_windows = max(1, int(raw.n_times / sfreq / 2) - 1)
        scan_events = np.array([
            [int(i * 2 * sfreq), 0, 1] for i in range(n_windows)
        ])
        if len(scan_events) > 10:
            scan_epochs = mne.Epochs(
                raw.copy().filter(LFREQ, HFREQ, fir_design='firwin', verbose='ERROR'),
                scan_events, event_id=1,
                tmin=0, tmax=1.0, baseline=None,
                preload=True, verbose='ERROR', reject=None,
            )
            scan_data = scan_epochs.get_data()
            # 每个通道的 peak-to-peak 中位数（跨所有扫描窗口）
            ch_ptp_median = np.median(np.ptp(scan_data, axis=2), axis=0)
            # 全局中位数 + 3 倍 MAD 作为阈值
            global_median = np.median(ch_ptp_median)
            mad = np.median(np.abs(ch_ptp_median - global_median))
            threshold = global_median + 3 * mad * 1.4826
            # 找出额外坏通道（不在官方清单中的）
            for ch_idx, ch_name in enumerate(raw.ch_names):
                if ch_name in bads:
                    continue
                if ch_ptp_median[ch_idx] > threshold:
                    bads.append(ch_name)
            raw.info['bads'] = bads
            del scan_epochs, scan_data
    except Exception:
        pass  # 自动检测失败时不影响主流程

    # 3. 插值坏通道
    if bads:
        raw.interpolate_bads(reset_bads=True, verbose='ERROR')

    # 4. 平均参考 + 滤波
    raw.set_eeg_reference('average', projection=False, verbose='ERROR')
    raw.filter(LFREQ, HFREQ, fir_design='firwin', verbose='ERROR')
    return raw, bads


def build_stim_events(events_df, sfreq):
    """构造食物图呈现事件（stimulus-locked）

    返回 mne 格式 events 数组 + 对应的 trial 元数据 DataFrame
    """
    food = events_df[events_df['event_type'] == 'foodImageDisplay'].copy().reset_index(drop=True)
    samples = (food['onset'] * sfreq).astype(int).values
    # event_id 编码 trial_type: health=1, taste=2, decision=3
    type_map = {'health': 1, 'taste': 2, 'decision': 3}
    ids = food['trial_type'].map(type_map).fillna(0).astype(int).values
    mne_events = np.column_stack([samples, np.zeros(len(samples), dtype=int), ids])
    event_id = {'health': 1, 'taste': 2, 'decision': 3}
    return mne_events, event_id, food


def build_response_events(events_df, sfreq):
    """构造按键响应事件（response-locked）

    用于分析决策阶段（按键前后）的神经活动
    """
    resp = events_df[events_df['event_type'] == 'response'].copy().reset_index(drop=True)
    # 过滤过慢响应（>3s）和过快响应（<150ms，按键错误）
    resp = resp[(resp['choice_rt'] > 0.15) & (resp['choice_rt'] < 3.0)].reset_index(drop=True)
    samples = (resp['onset'] * sfreq).astype(int).values
    type_map = {'health': 1, 'taste': 2, 'decision': 3}
    ids = resp['trial_type'].map(type_map).fillna(0).astype(int).values
    mne_events = np.column_stack([samples, np.zeros(len(samples), dtype=int), ids])
    event_id = {'health': 1, 'taste': 2, 'decision': 3}
    return mne_events, event_id, resp


def extract_epoch_labels(trial_df):
    """从 trial 元数据提取标签字典

    标签包括：
    - trial_type: health/taste/decision（任务判断维度）
    - resp_choice: Yes/No
    - choice_rt: 反应时（秒）
    - food_image_code: 食物图 ID（用于关联 paired choice 评分）
    - block, trial_total: 位置信息
    - resp_too_slow: 是否超时
    """
    return {
        'trial_type': trial_df['trial_type'].values,
        'resp_choice': trial_df['resp_choice'].values,
        'choice_rt': trial_df['choice_rt'].values.astype(np.float32),
        'food_image_code': trial_df['food_image_code'].values,
        'block': trial_df['block'].values.astype(int),
        'trial_total': trial_df['trial_total'].values.astype(int),
        'resp_too_slow': trial_df['resp_too_slow'].values.astype(int),
        'subject_id': np.full(len(trial_df), trial_df.name if hasattr(trial_df, 'name') else ''),
    }


def preprocess_subject(subject_id, save_response_locked=True):
    """单被试 EEG 预处理

    返回
    ----
    stim_X : ndarray, shape (n_trials, n_channels, n_times_stim)
    stim_labels : dict
    resp_X : ndarray or None (response-locked)
    resp_labels : dict or None
    qc_info : dict, 质量控制信息
    """
    # 读取并滤波
    raw, bads = load_and_filter_raw(subject_id)
    sfreq = raw.info['sfreq']

    # 读取事件
    events_path = DATA_ROOT / subject_id / 'eeg' / f'{subject_id}_task-categorisation_events.tsv'
    events_df = pd.read_csv(events_path, sep='\t')

    qc_info = {
        'subject_id': subject_id,
        'sfreq': float(sfreq),
        'n_channels': len(EEG_CHANNEL_NAMES),
        'duration_s': float(raw.n_times / sfreq),
        'official_bad_channels': OFFICIAL_BAD_CHANNELS.get(subject_id, []),
        'interpolated_channels': bads,
    }

    # ===== Stimulus-locked epochs =====
    mne_events, event_id, stim_df = build_stim_events(events_df, sfreq)
    stim_df.name = subject_id  # 用于 extract_epoch_labels

    stim_epochs = mne.Epochs(
        raw, mne_events, event_id=event_id,
        tmin=TMIN_STIM, tmax=TMAX_STIM,
        baseline=BASELINE,
        preload=True, verbose='ERROR',
        reject=dict(eeg=REJECT_THRESHOLD),
        on_missing='warn',
    )

    if len(stim_epochs) == 0:
        return None, None, None, None, qc_info

    stim_X = stim_epochs.get_data(copy=False).astype(np.float32)
    kept_idx = stim_epochs.selection
    stim_df_kept = stim_df.iloc[kept_idx].reset_index(drop=True)
    stim_df_kept.name = subject_id
    stim_labels = extract_epoch_labels(stim_df_kept)
    stim_labels['subject_id'] = np.full(len(stim_df_kept), subject_id)

    qc_info['n_stim_total'] = int(len(stim_df))
    qc_info['n_stim_kept'] = int(len(stim_X))
    qc_info['stim_reject_rate'] = float(1 - len(stim_X) / len(stim_df))
    # 每种 trial_type 的保留数
    for tt in ['health', 'taste', 'decision']:
        qc_info[f'n_{tt}_kept'] = int((stim_labels['trial_type'] == tt).sum())

    # ===== Response-locked epochs =====
    resp_X, resp_labels = None, None
    if save_response_locked:
        mne_resp, resp_id, resp_df = build_response_events(events_df, sfreq)
        resp_df.name = subject_id
        if len(mne_resp) > 0:
            try:
                resp_epochs = mne.Epochs(
                    raw, mne_resp, event_id=resp_id,
                    tmin=TMIN_RESP, tmax=TMAX_RESP,
                    baseline=BASELINE_RESP,
                    preload=True, verbose='ERROR',
                    reject=dict(eeg=REJECT_THRESHOLD),
                    on_missing='warn',
                )
                if len(resp_epochs) > 0:
                    resp_X = resp_epochs.get_data(copy=False).astype(np.float32)
                    resp_idx = resp_epochs.selection
                    resp_df_kept = resp_df.iloc[resp_idx].reset_index(drop=True)
                    resp_df_kept.name = subject_id
                    resp_labels = extract_epoch_labels(resp_df_kept)
                    resp_labels['subject_id'] = np.full(len(resp_df_kept), subject_id)
                    qc_info['n_resp_total'] = int(len(resp_df))
                    qc_info['n_resp_kept'] = int(len(resp_X))
                    qc_info['resp_reject_rate'] = float(1 - len(resp_X) / len(resp_df))
            except Exception as e:
                qc_info['resp_error'] = str(e)

    # 释放内存
    del raw

    return stim_X, stim_labels, resp_X, resp_labels, qc_info


def save_subject(subject_id, stim_X, stim_labels, resp_X, resp_labels, qc_info):
    """保存单被试结果"""
    # Stimulus-locked
    np.save(EPOCHS_DIR / f'{subject_id}_stim_epochs.npy', stim_X)
    np.savez(EPOCHS_DIR / f'{subject_id}_stim_labels.npz', **stim_labels)

    # Response-locked
    if resp_X is not None:
        np.save(EPOCHS_DIR / f'{subject_id}_resp_epochs.npy', resp_X)
        np.savez(EPOCHS_DIR / f'{subject_id}_resp_labels.npz', **resp_labels)

    # QC info
    with open(QC_DIR / f'{subject_id}_qc.json', 'w', encoding='utf-8') as f:
        json.dump(qc_info, f, indent=2, ensure_ascii=False)


def reconstruct_qc_from_cache(subject_id):
    """缓存命中但缺少 {sid}_qc.json 时，从已保存的 epochs/labels 重建 QC 指标

    拒绝率与 trial 总数无法从保留 epochs 反推，留空（NaN）；
    下游有效性过滤只依赖 n_stim_kept 与各 trial_type 计数。
    """
    info = {'subject_id': subject_id, 'cached': True,
            'status': 'cached_no_qc_json'}
    try:
        X = np.load(EPOCHS_DIR / f'{subject_id}_stim_epochs.npy', mmap_mode='r')
        labels = np.load(EPOCHS_DIR / f'{subject_id}_stim_labels.npz',
                         allow_pickle=True)
        info['n_channels'] = int(X.shape[1])
        info['n_stim_kept'] = int(X.shape[0])
        tt = labels['trial_type']
        for tt_name in ['health', 'taste', 'decision']:
            info[f'n_{tt_name}_kept'] = int((tt == tt_name).sum())
    except Exception as e:
        info['status'] = 'cache_load_error'
        info['error'] = str(e)
        return info
    if (EPOCHS_DIR / f'{subject_id}_resp_epochs.npy').exists():
        try:
            Xr = np.load(EPOCHS_DIR / f'{subject_id}_resp_epochs.npy',
                         mmap_mode='r')
            info['n_resp_kept'] = int(Xr.shape[0])
        except Exception:
            pass
    return info


def merge_qc_summaries(existing, new_qc):
    """合并新旧 QC 汇总：非空字段更多的行优先保留

    防止缓存占位行（仅 subject_id/cached）覆盖历史指标行；
    error / no_epochs 状态行强制覆盖，保证重跑失败能及时反映。
    """
    qc_df = pd.concat([existing, new_qc], ignore_index=True)
    qc_df['__info'] = qc_df.drop(columns=['subject_id']).notna().sum(axis=1)
    if 'status' in qc_df.columns:
        hard = qc_df['status'].isin(['error', 'no_epochs'])
        qc_df.loc[hard, '__info'] = qc_df['__info'].max() + 1
    qc_df = (qc_df.sort_values('__info', kind='mergesort')
                  .drop_duplicates(subset='subject_id', keep='last')
                  .drop(columns='__info'))
    return qc_df


def preprocess_all_subjects(force_reprocess=False, n_subjects=None, save_response_locked=True):
    """批量预处理所有被试

    参数
    ----
    force_reprocess : bool, True 时强制重新处理（覆盖缓存）
    n_subjects : int, 只处理前 N 个（None=全部）
    save_response_locked : bool, 是否同时保存 response-locked epochs
    """
    subject_ids = get_eeg_valid_subjects()
    if n_subjects:
        subject_ids = subject_ids[:n_subjects]
    print(f'[EEG] 找到 {len(subject_ids)} 个有效被试（排除 {len(EEG_EXCLUDED_SUBJECTS)} 个）')
    print(f'[EEG] 参数: stim epoch=[{TMIN_STIM}, {TMAX_STIM}]s, '
          f'filter=[{LFREQ}, {HFREQ}] Hz, reject=±{REJECT_THRESHOLD*1e6:.0f} µV')
    print(f'[EEG] response_locked: {save_response_locked}')

    all_qc = []
    n_success = 0
    n_skip = 0
    n_fail = 0

    for i, sid in enumerate(subject_ids):
        stim_cache = EPOCHS_DIR / f'{sid}_stim_epochs.npy'
        if stim_cache.exists() and not force_reprocess:
            # 加载 QC 信息；缺失时从缓存 epochs/labels 重建关键指标
            qc_path = QC_DIR / f'{sid}_qc.json'
            if qc_path.exists():
                with open(qc_path, 'r', encoding='utf-8') as f:
                    all_qc.append(json.load(f))
            else:
                all_qc.append(reconstruct_qc_from_cache(sid))
            n_skip += 1
            continue

        try:
            print(f'[{i+1}/{len(subject_ids)}] {sid}: 处理中...')
            stim_X, stim_labels, resp_X, resp_labels, qc_info = preprocess_subject(
                sid, save_response_locked=save_response_locked
            )
            if stim_X is None or len(stim_X) == 0:
                print(f'  WARNING: {sid} 无有效 epoch')
                qc_info['status'] = 'no_epochs'
                all_qc.append(qc_info)
                n_fail += 1
                continue
            save_subject(sid, stim_X, stim_labels, resp_X, resp_labels, qc_info)
            all_qc.append(qc_info)
            n_success += 1
            print(f'  {sid}: stim {qc_info["n_stim_kept"]}/{qc_info["n_stim_total"]} '
                  f'(reject {qc_info["stim_reject_rate"]*100:.1f}%)'
                  + (f', resp {qc_info.get("n_resp_kept", 0)}/{qc_info.get("n_resp_total", 0)}'
                     if save_response_locked else ''))
        except Exception as e:
            print(f'  ERROR: {sid} 失败: {e}')
            all_qc.append({'subject_id': sid, 'status': 'error', 'error': str(e)})
            n_fail += 1

    # 保存 QC 汇总
    # 合并已存在的 qc_summary.csv：--quick / 部分处理时只更新本次涉及的被试行，
    # 避免覆盖全量 QC 汇总（下游 svr/timefreq/erp/解码脚本依赖它做被试过滤）
    qc_path = QC_DIR / 'qc_summary.csv'
    new_qc = pd.DataFrame(all_qc)
    if qc_path.exists() and 'subject_id' in new_qc.columns:
        try:
            existing = pd.read_csv(qc_path)
            if 'subject_id' in existing.columns and len(existing) > 0:
                qc_df = merge_qc_summaries(existing, new_qc)
            else:
                qc_df = new_qc
        except Exception:
            qc_df = new_qc
    else:
        qc_df = new_qc
    qc_df.to_csv(qc_path, index=False)

    # 打印汇总
    print('\n' + '=' * 60)
    print('预处理完成')
    print('=' * 60)
    print(f'成功: {n_success}, 跳过(缓存): {n_skip}, 失败: {n_fail}')
    if 'n_stim_kept' in qc_df.columns:
        valid = qc_df[qc_df['n_stim_kept'].notna()]
        print(f'总 stimulus epochs: {int(valid["n_stim_kept"].sum())}')
        if 'stim_reject_rate' in valid.columns:
            rr = valid['stim_reject_rate'].dropna()
            if len(rr) > 0:
                print(f'平均保留率: {(1 - rr.mean())*100:.1f}%')
                print(f'平均 reject 率: {rr.mean()*100:.1f}%')
            else:
                print('平均 reject 率: 不可用（缓存重建行无原始 trial 总数，'
                      '如需完整指标请删除 outputs/qc 后全量重跑或 --force-reprocess）')
        else:
            print('平均 reject 率: 不可用（QC JSON 缺失，无法从缓存重建该列；'
                  '如需完整指标请删除 outputs/qc 后全量重跑或 --force-reprocess）')
    print(f'QC 报告: {QC_DIR / "qc_summary.csv"}')

    return qc_df


def load_subject_epochs(subject_id, epoch_type='stim'):
    """加载已预处理的单被试 epoch

    参数
    ----
    subject_id : str
    epoch_type : 'stim' 或 'resp'

    返回
    ----
    X : ndarray, shape (n_trials, n_channels, n_times)
    labels : dict
    """
    abbr = 'stim' if epoch_type == 'stim' else 'resp'
    x_path = EPOCHS_DIR / f'{subject_id}_{abbr}_epochs.npy'
    l_path = EPOCHS_DIR / f'{subject_id}_{abbr}_labels.npz'
    if not x_path.exists():
        raise FileNotFoundError(f'{subject_id} {epoch_type} epochs not found: {x_path}')
    X = np.load(x_path)
    labels = dict(np.load(l_path, allow_pickle=True))
    return X, labels


if __name__ == '__main__':
    import sys
    quick = '--quick' in sys.argv
    force = '--force' in sys.argv
    n = None
    for arg in sys.argv[1:]:
        if arg.startswith('--n='):
            n = int(arg.split('=')[1])

    if quick:
        print('快速测试模式：仅处理 2 个被试')
        n = 2

    qc_df = preprocess_all_subjects(n_subjects=n, force_reprocess=force, save_response_locked=True)
