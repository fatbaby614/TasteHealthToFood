"""生成 Array 投稿用 Graphical Abstract。

设计约束（Array Guide for authors）：
- 最长边横向，≥1328 px 宽，13 cm 宽度下可读
- 输出 TIFF/PDF/PNG

实现说明：全部使用**数据坐标下的等比矩形**，并在轴上关闭
aspect equal 的自动缩放，用显式 `mutation_scale` 控制圆角，
避免 FancyBboxPatch 在归一化非等比坐标下失真。
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import numpy as np

OUT = Path(__file__).resolve().parents[1] / "paper" / "Array"

INK = "#1B2A41"
MUTED = "#5B6B7C"
BLUE = "#2E6FA7"
TEAL = "#2F8F83"
AMBER = "#D98A28"
RED = "#C0392B"
GREEN = "#2E7D4F"
PAPER = "#FFFFFF"
SOFT = "#F4F7FA"
LINE = "#D3DCE6"
DARK = "#0F1E33"

FS_TITLE = 14.0
FS_HEAD = 9.0
FS_BODY = 7.2
FS_NOTE = 6.4
FS_MICRO = 5.6

#画布：13.5 x 5.6 in @300 dpi -> 4050 x 1680 px（远超 1328 px 下限）
W_IN, H_IN = 13.5, 5.6


def rounded(ax, x, y, w, h, fc, ec=LINE, lw=0.9, pad=0.10, z=2):
    """圆角矩形：x/y/w/h 为数据单位，pad 为圆角半径（数据单位）。"""
    p = FancyBboxPatch((x + pad, y + pad), w - 2 * pad, h - 2 * pad,
                       boxstyle=f"round,pad={pad},rounding_size={pad}",
                       facecolor=fc, edgecolor=ec, linewidth=lw, zorder=z)
    ax.add_patch(p)
    return p


def arrow(ax, x1, y1, x2, y2, color=MUTED, lw=1.3, ms=7, z=6):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=ms, color=color, linewidth=lw,
                                 zorder=z, shrinkA=0, shrinkB=0))


def head(ax, x, y, color, num, label, dx=0.30):
    ax.add_patch(Circle((x, y), 0.155, facecolor=color, edgecolor="none", zorder=7))
    ax.text(x, y, num, ha="center", va="center", color="white",
            fontsize=FS_HEAD, fontweight="bold", zorder=8)
    ax.text(x + dx, y, label, ha="left", va="center", color=color,
            fontsize=FS_HEAD + 0.6, fontweight="bold", zorder=8)


def chip(ax, x, y, w, h, title, sub, ec, fc=PAPER, ts=FS_BODY, ss=FS_MICRO):
    rounded(ax, x, y, w, h, fc, ec=ec, lw=0.8, pad=0.07)
    ax.text(x + w / 2, y + h * 0.64, title, ha="center", va="center",
            fontsize=ts, color=INK, fontweight="bold", zorder=7)
    ax.text(x + w / 2, y + h * 0.27, sub, ha="center", va="center",
            fontsize=ss, color=MUTED, zorder=7, linespacing=1.3)


def build():
    fig = plt.figure(figsize=(W_IN, H_IN), dpi=300, facecolor=PAPER)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100 * H_IN / W_IN)
    ax.set_aspect("equal")
    ax.axis("off")

    TOP = 100 * H_IN / W_IN          # ≈ 41.5
    # ---- 标题 ----
    ax.text(50, TOP - 2.2, "A multi-seed audit retracts the whole-scalp claim "
            "and retains two occipital effects",
            ha="center", va="center", fontsize=FS_TITLE, color=INK, fontweight="bold")
    ax.plot([24, 76], [TOP - 4.3, TOP - 4.3], color=LINE, lw=1.0, zorder=1)

    # ---- 面板几何 ----
    PAD = 0.06
    top = TOP - 5.6
    bot = 6.6
    ph = top - bot
    w1, w2, w3, w4 = 21.0, 23.5, 24.5, 24.5
    gap = 1.9
    x1 = PAD
    x2 = x1 + w1 + gap
    x3 = x2 + w2 + gap
    x4 = x3 + w3 + gap

    # ============ 1. 数据 ============
    rounded(ax, x1, bot, w1, ph, SOFT, ec=BLUE, lw=1.2, pad=0.12)
    head(ax, x1 + 2.0, top - 1.7, BLUE, "1", "Public dataset")
    chip(ax, x1 + 1.5, top - 7.4, w1 - 3.0, 4.6,
         "64-channel EEG", "512 Hz, BioSemi", BLUE)
    # 电极头形
    cx, cy = x1 + w1 / 2, top - 12.6
    for rr, fc in ((2.55, PAPER), (1.55, BLUE)):
        ax.add_patch(Circle((cx, cy), rr, facecolor=fc,
                            edgecolor=BLUE if fc == PAPER else "none",
                            lw=0.9, zorder=5))
    th = np.linspace(np.pi * 0.15, np.pi * 1.85, 9)
    ax.plot(cx + 2.55 * np.cos(th), cy + 2.55 * np.sin(th), "o", ms=2.0,
            color=BLUE, zorder=6)
    ax.plot([cx - 1.05, cx + 1.05], [cy + 2.15, cy + 2.15], color=BLUE,
            lw=1.8, zorder=6, solid_capstyle="round")
    ax.text(cx, cy - 3.15, "n = 117", ha="center", va="center",
            fontsize=FS_BODY + 0.6, color=INK, fontweight="bold", zorder=6)
    chip(ax, x1 + 1.5, bot + 2.2, w1 - 3.0, 4.4,
         "Trial-wise ratings", "taste / health / willingness", BLUE)

    # ============ 2. 流水线 ============
    rounded(ax, x2, bot, w2, ph, SOFT, ec=TEAL, lw=1.2, pad=0.12)
    head(ax, x2 + 2.0, top - 1.7, TEAL, "2", "Analysis pipeline")
    steps = [("Behavioral modeling", "hierarchical drift-diffusion"),
             ("Covariance matrices", "trial-wise, SPD"),
             ("Riemannian tangent space", "log-map, vectorized"),
             ("Decoding", "classification + regression")]
    sh = 4.2
    sgap = 0.75
    sy = top - 5.4
    for i, (t1, t2) in enumerate(steps):
        yy = sy - sh - i * (sh + sgap)
        chip(ax, x2 + 1.4, yy - sh, w2 - 2.8, sh, t1, t2, TEAL)
        if i < len(steps) - 1:
            arrow(ax, x2 + w2 / 2, yy - sh - 0.10, x2 + w2 / 2, yy - sh - sgap + 0.10,
                  color=TEAL, lw=1.0, ms=6)

    # ============ 3. 多种子审计 ============
    rounded(ax, x3, bot, w3, ph, "#FDF6EC", ec=AMBER, lw=1.2, pad=0.12)
    head(ax, x3 + 2.0, top - 1.7, AMBER, "3", "Multi-seed audit")
    for i in range(5):
        ax.add_patch(Circle((x3 + 2.6 + i * 1.15, top - 5.6), 0.42,
                            facecolor=AMBER, edgecolor="none", zorder=6,
                            alpha=0.32 + 0.14 * i))
    ax.text(x3 + 9.2, top - 5.6, "20 seeds x 5000 permutations", ha="left",
            va="center", fontsize=FS_MICRO, color=MUTED, zorder=6)
    # 置换分布（x 显式映射到面板内）
    rng = np.random.default_rng(11)
    null = rng.normal(0, 1, 1200)
    dx0, dy0, dw, dh = x3 + 2.2, top - 16.4, w3 - 4.4, 8.6
    rounded(ax, dx0, dy0, dw, dh, PAPER, ec=LINE, lw=0.8, pad=0.06)
    zx0, zx1 = -3.0, 3.0                      # 零分布取值范围
    def zx(z):                                  # z 值 ->面板 x
        return dx0 + (z - zx0) / (zx1 - zx0) * (dw - 1.2) + 0.6
    cnt, edges = np.histogram(null, bins=24, range=(zx0, zx1))
    base = dy0 + 0.55
    plot_h = (dy0 + dh - 1.5) - base          # 正的绘图区高度
    for c, e0, e1 in zip(cnt, edges[:-1], edges[1:]):
        if c == 0:
            continue
        ax.add_patch(plt.Rectangle((zx(e0), base), max(zx(e1) - zx(e0), 0.05),
                                   plot_h * c / cnt.max(),
                                   facecolor="#BCC9D7", edgecolor="none", zorder=5))
    ax.plot([zx(-3), zx(3)], [base, base], color=LINE, lw=0.8, zorder=5)
    # 观测统计量（正对照效应量所在位置，示意）
    zobs = 1.55
    ax.plot([zx(zobs), zx(zobs)], [base, dy0 + dh - 1.0],
            color=RED, lw=1.7, zorder=7, solid_capstyle="round")
    ax.text(zx(zobs), dy0 + dh - 1.0, "observed", ha="center", va="top",
            fontsize=FS_MICRO, color=RED, fontweight="bold", zorder=7)
    ax.text(dx0 + 0.7, dy0 + dh - 1.0, "permutation null", ha="left", va="top",
            fontsize=FS_MICRO, color=MUTED, zorder=7)
    ax.text(dx0 + dw / 2, dy0 - 0.7,
            "the null reruns the pipeline on permuted labels",
            ha="center", va="top", fontsize=FS_MICRO, color=MUTED,
            style="italic", zorder=7)
    checks = [("BH-FDR correction", GREEN),
              ("Cross-seed convergence", GREEN),
              ("Budget fixed in advance", GREEN)]
    cy0 = bot + 1.4
    for i, (lab, cc) in enumerate(checks):
        yy = cy0 + (2 - i) * 3.05
        rounded(ax, x3 + 1.4, yy, w3 - 2.8, 2.55, PAPER, ec=LINE, lw=0.7, pad=0.05)
        ax.plot([x3 + 2.5], [yy + 1.28], marker="o", ms=3.0, color=cc, zorder=7)
        ax.text(x3 + 3.5, yy + 1.28, lab, ha="left", va="center",
                fontsize=FS_BODY, color=INK, zorder=7)
        ax.text(x3 + w3 - 2.0, yy + 1.28, "pass", ha="right", va="center",
                fontsize=FS_MICRO, color=cc, fontweight="bold", zorder=7)

    # ============ 4. 结论 ============
    rounded(ax, x4, bot, w4, ph, SOFT, ec=RED, lw=1.2, pad=0.12)
    head(ax, x4 + 2.0, top - 1.7, RED, "4", "What the audit found")
    finds = [(GREEN, "o", "Positive control holds", "taste-advantage LPP   dz = 0.82   p = 3.3e-12"),
             (RED, "x", "Whole-scalp claim retracted", "18 window x target cells: 0/18 pass screening"),
             (GREEN, "o", "Occipital ROI retained", "taste + willingness, 2/21, both conventions"),
             (RED, "!", "False positive caught", "1 seed p = 0.000 -> seed-mean p = 0.344 +/- 0.287")]
    for i, (cc, icon, t1, t2) in enumerate(finds):
        yy = top - 5.4 - i * 5.0
        rounded(ax, x4 + 1.4, yy - 4.3, w4 - 2.8, 4.3, PAPER, ec=cc, lw=1.0, pad=0.07)
        ax.add_patch(Circle((x4 + 3.0, yy - 1.35), 0.62, facecolor=cc,
                            edgecolor="none", zorder=7))
        ax.text(x4 + 3.0, yy - 1.35, icon, ha="center", va="center",
                fontsize=FS_MICRO + 0.6, color="white", fontweight="bold", zorder=8)
        ax.text(x4 + 4.3, yy - 1.35, t1, ha="left", va="center",
                fontsize=FS_BODY, color=INK, fontweight="bold", zorder=7)
        ax.text(x4 + 4.3, yy - 3.15, t2, ha="left", va="center",
                fontsize=FS_MICRO, color=MUTED, zorder=7)

    # ---- 面板箭头 ----
    for (xa, wa, xb) in ((x1, w1, x2), (x2, w2, x3), (x3, w3, x4)):
        arrow(ax, xa + wa + 0.12, (top + bot) / 2, xb - 0.12, (top + bot) / 2,
              color=INK, lw=1.5, ms=9)

    # ---- 底栏 ----
    rounded(ax, PAD, 0.4, 100 - 2 * PAD, 4.7, DARK, ec=DARK, lw=0, pad=0.10)
    ax.text(PAD + 2.0, 2.75, "Reproducible by construction", ha="left", va="center",
            fontsize=FS_BODY + 0.4, color="#9FC4E8", fontweight="bold", zorder=7)
    ax.text(PAD + 23.0, 2.75,
            "open code + replication scripts   |   fixed seeds, budgets and "
            "software versions   |   every result traceable to its seed",
            ha="left", va="center", fontsize=FS_MICRO, color="#E6EDF5", zorder=7)
    return fig


def main():
    fig = build()
    OUT.mkdir(parents=True, exist_ok=True)
    # 不用 bbox_inches="tight"：会按内容裁边导致底栏被切；固定边距更可控
    # TIFF 用 300 dpi（8100x3360 仍远超 1328 px 下限），避免 600 dpi 产生 >100 MB 的大文件
    for ext, dpi in (("tiff", 300), ("pdf", 600), ("png", 300)):
        p = OUT / f"Graphical_Abstract.{ext}"
        fig.savefig(p, dpi=dpi, facecolor="#FFFFFF")
        print(f"[ga] saved: {p.name} ({p.stat().st_size / 1e6:.1f} MB)")
    plt.close(fig)


if __name__ == "__main__":
    main()
