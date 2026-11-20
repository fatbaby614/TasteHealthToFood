"""
verify_temporal_perm_smoke.py
冒烟版本：5 被试，20 次置换 × 2 random_state
"""
import sys
import numpy as np
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from svr_roi_channels import (
    _loso_mean_r, _permutation_roi,
    load_subject_list, ROI_CHANNELS, TIME_WINDOW, TARGETS,
    detect_channel_names, get_roi_indices
)
from svr_decoding import preload_full, compute_logm_all, build_target_views


def main():
    print('=' * 60)
    print('冒烟验证（5 被试，20 次置换 × 2 random_state）')
    print('=' * 60)

    subject_ids = load_subject_list()[:5]
    print(f'被试数: {len(subject_ids)}')

    ch_names = detect_channel_names(subject_ids)
    roi = 'temporal'
    roi_chs = ROI_CHANNELS[roi]
    roi_idx, missing = get_roi_indices(ch_names, roi_chs)
    print(f'ROI {roi}: 找到 {len(roi_idx)} 个通道')

    target_col = 'rating_willingnessToEat'
    print(f'Target: {target_col}')

    t0 = time.time()
    print('\n[feat] preload_full...')
    full_cache = preload_full(subject_ids)
    logm_all = compute_logm_all(full_cache, TIME_WINDOW)
    print(f'  完成 ({time.time()-t0:.1f}s), {len(logm_all)} 被试')

    logm_by_sid_full, y_by_sid = build_target_views(logm_all, target_col)
    logm_by_sid_roi = {
        sid: logm[np.ix_(np.arange(len(logm)), roi_idx, roi_idx)]
        for sid, logm in logm_by_sid_full.items()
    }

    sid_order = [s for s in subject_ids
                 if s in logm_by_sid_roi and len(y_by_sid.get(s, [])) >= 30]
    logm_list = [logm_by_sid_roi[s] for s in sid_order]
    y_list = [y_by_sid[s] for s in sid_order]
    X = np.concatenate(logm_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    cumsum = np.cumsum([0] + [len(m) for m in logm_list])
    split_idx = np.array(list(zip(cumsum[:-1], cumsum[1:])))
    print(f'X.shape = {X.shape}, y.shape = {y.shape}')

    print('\n[Test 1] actual_r 稳定性 × 3 seed')
    for seed in [42, 123, 456]:
        rs = _loso_mean_r(X, y, split_idx, max_trials=20, random_state=seed)
        print(f'  seed={seed}: actual_r={rs:+.5f}')

    print('\n[Test 2] 20 次置换 × 2 seed')
    for seed in [42, 123]:
        t0 = time.time()
        res = _permutation_roi(
            logm_by_sid_roi, y_by_sid, subject_ids,
            n_perm=20, random_state=seed, target_label=target_col
        )
        print(f'  seed={seed}: actual={res["actual_r"]:+.4f} '
              f'chance_mean={res["chance_mean"]:+.4f} '
              f'perm_p={res["perm_p"]:.4f} '
              f'({time.time()-t0:.1f}s)')

    print('\n✓ 冒烟通过 — 可运行全量版')


if __name__ == '__main__':
    main()