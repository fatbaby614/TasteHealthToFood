# -*- coding: utf-8 -*-
"""合并 svr_v2_multiseed_only.py 的 6 个分窗结果，重算全族 FDR，写最终文件。"""

import sys
import os
import json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd

from config import RESULTS_DIR
from svr_decoding import apply_fdr_correction, TIME_WINDOWS, TARGETS

WINDOW_TAGS = {w: w.split('_')[0] for w in TIME_WINDOWS}


def main():
    frames = []
    cells = []
    win_fdr = None
    for wname, tag in WINDOW_TAGS.items():
        csv_p = RESULTS_DIR / f'svr_decoding_multiseed_{tag}.csv'
        json_p = RESULTS_DIR / f'svr_decoding_multiseed_{tag}.json'
        if not csv_p.exists() or not json_p.exists():
            missing = csv_p.name if not csv_p.exists() else json_p.name
            print(f'[missing] {missing} —— 等待该窗完成后再合并')
            return
        frames.append(pd.read_csv(csv_p))
        with open(json_p, encoding='utf-8') as f:
            j = json.load(f)
            cells.extend(j['cells'])
            win_fdr = j['fdr']  # 各窗 fdr 元信息一致，取最后读到的即可

    assert len(cells) == 18, f'expected 18 cells, got {len(cells)}'

    # 全族 FDR：跨 seed 平均 perm_p（D1 决议）
    fdr = apply_fdr_correction([c['perm_p_mean'] for c in cells], alpha=0.05)
    for c, q, rej in zip(cells, fdr['q_values'], fdr['rejected']):
        c['q_value'] = float(q)
        c['fdr_significant'] = bool(rej)

    df = pd.concat(frames, ignore_index=True)
    df['q_value'] = [c['q_value'] for c in cells]
    df['fdr_significant'] = [c['fdr_significant'] for c in cells]
    df = df.sort_values(['window', 'target']).reset_index(drop=True)
    df.to_csv(RESULTS_DIR / 'svr_decoding_multiseed.csv', index=False)

    # 运行参数从分窗 json 继承（与 run_all.sh --svr-perm/--svr-seeds 传入一致）
    fdr_meta = {
        'method': fdr['method'], 'alpha': fdr['alpha'],
        'n_tests': fdr['n_tests'],
        'n_significant': int(sum(fdr['rejected'])),
        'perm_p_rule': win_fdr['perm_p_rule'],
        'permutation': win_fdr['permutation'],
        'budget_semantics': win_fdr['budget_semantics'],
        'seeds': win_fdr['seeds'],
        'n_perm_per_seed': win_fdr['n_perm_per_seed'],
    }
    with open(RESULTS_DIR / 'svr_decoding_multiseed.json', 'w',
              encoding='utf-8') as f:
        json.dump({'cells': cells, 'fdr': fdr_meta}, f, indent=2,
                  ensure_ascii=False, default=str)

    n_sig = fdr_meta['n_significant']
    print(f'\n=== 18-cell v2 FDR 结果: {n_sig}/18 显著 ===')
    order = list(TIME_WINDOWS.keys())
    for c in sorted(cells, key=lambda x: x['q_value']):
        flag = ' *** FDR' if c['fdr_significant'] else ''
        print(f"  {c['window']:18s} {c['target']:26s} "
              f"actual={c['actual_r_mean']:+.4f}±{c['actual_r_std']:.4f} "
              f"perm_p={c['perm_p_mean']:.3f}±{c['perm_p_std']:.3f} "
              f"[{c['perm_p_min']:.2f},{c['perm_p_max']:.2f}] "
              f"q={c['q_value']:.4g}{flag}")
    print('\n[saved] svr_decoding_multiseed.csv / .json')


if __name__ == '__main__':
    main()
