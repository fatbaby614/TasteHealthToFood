"""
verify_temporal_perm.py
验证 temporal×willingnessToEat 的置换检验稳定性（max_trials=200 多 seed）
"""
import sys
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from svr_roi_channels import (
    _loso_mean_r, _permutation_roi, _permutation_roi_multi_seed,
    load_subject_list, ROI_CHANNELS, TIME_WINDOW, TARGETS,
    detect_channel_names, get_roi_indices
)
from svr_decoding import preload_full, compute_logm_all, build_target_views


def main():
    print('=' * 60)
    print('验证 temporal × willingnessToEat（max_trials=200, 多 seed）')
    print('=' * 60)

    subject_ids = load_subject_list()[:106]
    print(f'被试数: {len(subject_ids)}')

    ch_names = detect_channel_names(subject_ids)
    roi = 'temporal'
    roi_idx, _ = get_roi_indices(ch_names, ROI_CHANNELS[roi])
    print(f'ROI {roi}: {len(roi_idx)} 通道')

    target_col = 'rating_willingnessToEat'

    t0 = time.time()
    print('\n[feat] preload_full + logm...')
    full_cache = preload_full(subject_ids)
    logm_all = compute_logm_all(full_cache, TIME_WINDOW)
    print(f'  ({time.time()-t0:.1f}s), {len(logm_all)} 被试')

    logm_by_sid_full, y_by_sid = build_target_views(logm_all, target_col)
    logm_by_sid_roi = {
        sid: logm[np.ix_(np.arange(len(logm)), roi_idx, roi_idx)]
        for sid, logm in logm_by_sid_full.items()
    }

    # === Test 1: actual_r 稳定性（max_trials=200）===
    print('\n=== Test 1: actual_r 稳定性（max_trials=200, 5 seed）===')
    actual_rs = []
    for seed in [42, 123, 456, 789, 2024]:
        # 在 _permutation_roi 内部，X/y/split_idx 都会被重建
        res = _permutation_roi(
            logm_by_sid_roi, y_by_sid, subject_ids,
            n_perm=10, random_state=seed, target_label=target_col,
            max_trials=200,
        )
        actual_rs.append(res['actual_r'])
        print(f'  seed={seed}: actual_r={res["actual_r"]:+.5f}')
    print(f'  mean={np.mean(actual_rs):+.5f}, std={np.std(actual_rs):.5f}')
    if np.std(actual_rs) > 0.005:
        print('  ⚠ 仍然不稳定')
    else:
        print('  ✓ max_trials=200 已收敛')

    # === Test 2: 多 seed × 100 次置换 ===
    print('\n=== Test 2: 多 seed × 100 次置换（max_trials=200）===')
    seeds = [42, 123, 456, 789, 2024]
    t0 = time.time()
    multi = _permutation_roi_multi_seed(
        logm_by_sid_roi, y_by_sid, subject_ids,
        n_perm=100, seeds=seeds,
        target_label=target_col, max_trials=200,
    )
    print(f'  ({time.time()-t0:.1f}s)')
    print(f'  actual_rs = {[f"{x:+.4f}" for x in multi["actual_rs"]]}')
    print(f'  perm_ps   = {[f"{p:.4f}" for p in multi["perm_ps"]]}')
    print(f'  actual_r: {multi["actual_r_mean"]:+.5f} ± {multi["actual_r_std"]:.5f}')
    print(f'  perm_p  : {multi["perm_p_mean"]:.4f} ± {multi["perm_p_std"]:.4f} '
          f'(median={multi["perm_p_median"]:.4f})')
    if multi['perm_p_mean'] < 0.05:
        print('  ✓ 多 seed 平均后显著')
    else:
        print('  ✗ 多 seed 平均后不显著（确认假阳性）')

    # === Test 3: 跨所有 ROI × target 跑多 seed 完整重做 ===
    print('\n=== Test 3: 完整 7 ROI × 3 target × 5 seeds × 100 perm ===')
    print('  （约需 7×3×5×100 = 10500 次 LOSO 拟合，预计 2-3 小时）')
    print('  此项独立脚本 run_svr_roi_channels_v2.py 已用新参数调用')


if __name__ == '__main__':
    main()