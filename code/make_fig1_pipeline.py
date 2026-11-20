# -*- coding: utf-8 -*-
"""重绘论文 Fig.1 计算流程图（v2 叙事版）。

输出尺寸按 cas-dc 双栏 \textwidth=6.84in 设计，全部文字 >=7.2pt，
显示后有效字号不低于标称值（tight bbox 只会缩小图片宽度）。
"""
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.colors as mcolors  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import (FancyArrowPatch, FancyBboxPatch,  # noqa: E402
                                Rectangle)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
# 保护：本脚本是 matplotlib 备选版本，论文正文实际使用的是 GPT-Image 生成的
# fig1_computational_workflow.png。输出改用 _script 后缀，避免误跑覆盖论文用图。
OUT_PATH = (PROJECT_ROOT / 'paper' / 'figures'
            / 'fig1_computational_workflow_script.png')

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'axes.unicode_minus': False,
})

STAGES = [
    {
        'header': 'Data &\npreprocessing',
        'color': '#3D7A8E',
        'bullets': [
            'FoodEEG (ds007012)',
            '64 channels @ 512 Hz',
            'Bad-channel interp.',
            'Average reference',
            '0.5–45 Hz band-pass',
            'Epochs −0.2–1.0 s',
            'Baseline −0.2–0 s',
            '±100 µV rejection',
            '108 participants',
            '22,915 epochs',
        ],
    },
    {
        'header': 'Single-trial\nfeatures',
        'color': '#E07B39',
        'bullets': [
            'Broadband epochs,',
            'window-restricted',
            'LW covariances',
            'Ref = train-fold mean',
            'log-Euclidean',
            'tangent vectors',
        ],
    },
    {
        'header': 'Decoding models\n(LOSO)',
        'color': '#5B9279',
        'bullets': [
            'Classification:',
            'logistic regression',
            'on tangent features',
            '3 contrasts',
            'Regression:',
            'ridge (α = 1)',
            'on rating scales',
        ],
    },
    {
        'header': 'Statistical\nvalidation',
        'color': '#7A6FA0',
        'bullets': [
            'Per-subject acc / r',
            'one-sample t-tests',
            'vs 0.5 / vs 0',
            'Label permutation',
            'within training pool',
            'Multi-seed:',
            '5 seeds × 100 perms',
            'Budget check: 80/200',
        ],
    },
    {
        'header': 'Inference\n& audit',
        'color': '#B05A6B',
        'bullets': [
            'BH-FDR per family',
            'Verdict labels:',
            '· significant',
            '· descriptive only',
            '· retracted',
            'Audit trail:',
            'seeds · budgets · q',
        ],
    },
]

FOOTER_LINES = [
    ('Group-level & behavioural context', 8.0, 'bold', '#1a1a1a'),
    ('ERP window amplitudes (N1–LPP) · Morlet TFR (θ / α / β) · '
     'Sullivan neural–behaviour tests', 7.2, 'normal', '#333333'),
    ('Logistic choice model · hierarchical DDM (evidence accumulation) · '
     'SCSR high/low groups', 7.2, 'normal', '#333333'),
]

BOX_W = 18.4
BOX_TOP = 86.0
BOX_BOT = 30.0
HEAD_H = 7.0
X_STEP = 20.15


def tint(c, f=0.88):
    r, g, b, _ = mcolors.to_rgba(c)
    return (r + (1 - r) * f, g + (1 - g) * f, b + (1 - b) * f)


def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(6.84, 4.5))
    ax = fig.add_axes([0.012, 0.012, 0.976, 0.976])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')

    ax.text(50, 95, 'Reproducible computational workflow for '
            'EEG decoding and validation',
            ha='center', va='center', fontsize=10, fontweight='bold',
            color='#1a1a1a')

    for i, stage in enumerate(STAGES):
        x0 = 0.5 + i * X_STEP
        color = stage['color']
        ax.add_patch(FancyBboxPatch((x0, BOX_BOT), BOX_W, BOX_TOP - BOX_BOT,
                                    boxstyle='square,pad=0',
                                    facecolor=tint(color),
                                    edgecolor=color, linewidth=1.1,
                                    zorder=1))
        ax.add_patch(Rectangle((x0, BOX_TOP - HEAD_H), BOX_W, HEAD_H,
                               facecolor=color, edgecolor='none',
                               zorder=2))
        ax.text(x0 + BOX_W / 2, BOX_TOP - HEAD_H / 2, stage['header'],
                ha='center', va='center', fontsize=8, fontweight='bold',
                color='white', linespacing=1.2, zorder=3)
        y = BOX_TOP - HEAD_H - 2.8
        for line in stage['bullets']:
            ax.text(x0 + 1.1, y, line, ha='left', va='top', fontsize=7.2,
                    color='#1a1a1a', zorder=3)
            y -= 4.3
        if i < len(STAGES) - 1:
            arr_x0 = x0 + BOX_W + 0.3
            arr_x1 = x0 + X_STEP - 0.3
            ax.add_patch(FancyArrowPatch((arr_x0, (BOX_TOP + BOX_BOT) / 2),
                                         (arr_x1, (BOX_TOP + BOX_BOT) / 2),
                                         arrowstyle='-|>',
                                         mutation_scale=11,
                                         color='#555555', linewidth=1.1,
                                         zorder=4))

    ax.add_patch(FancyBboxPatch((0.5, 6.0), 99.0, 18.0,
                                boxstyle='square,pad=0',
                                facecolor='#F2F2F2', edgecolor='#888888',
                                linewidth=0.9, zorder=1))
    footer_ys = [19.8, 14.6, 9.2]
    for (txt, fs, fw, fc), fy in zip(FOOTER_LINES, footer_ys):
        ax.text(50, fy, txt, ha='center', va='center', fontsize=fs,
                fontweight=fw, color=fc, zorder=2)

    fig.savefig(OUT_PATH, dpi=300, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)
    print(f'saved: {OUT_PATH}')


if __name__ == '__main__':
    main()
