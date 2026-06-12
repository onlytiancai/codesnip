"""
ch05_mlp: MLP 拓扑 + 两条直线组合
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from _fonts import new_figure, save, CHOSEN_FONT

OUT_DIR = "../assets/images"
ACCENT = "#10b981"
ACCENT2 = "#0ea5e9"
WARN = "#f59e0b"
DANGER = "#ef4444"
MUTED = "#64748b"
TEXT = "#1e293b"


def mlp_topology(lang="zh"):
    is_en = lang == "en"
    fig, ax = plt.subplots(figsize=(8, 5), dpi=120)
    ax.set_xlim(0, 10); ax.set_ylim(0, 7)
    ax.axis("off")

    ax.text(5, 6.5, "2 → 4 → 1 MLP 拓扑", ha="center", fontsize=16, fontweight="bold", color=TEXT)

    # 3 层节点
    layers = [
        ([(1.5, 4.5), (1.5, 3.0)], ["x₁", "x₂"], ACCENT2, "输入层" if not is_en else "Input"),
        ([(5, 5.5), (5, 4.2), (5, 2.9), (5, 1.6)], ["h₁", "h₂", "h₃", "h₄"], ACCENT, "隐藏层" if not is_en else "Hidden"),
        ([(8.5, 3.5)], ["ŷ"], DANGER, "输出层" if not is_en else "Output"),
    ]
    for li, (positions, labels, color, layer_name) in enumerate(layers):
        for (px, py), lbl in zip(positions, labels):
            c = plt.Circle((px, py), 0.4, fc=color, ec=color, alpha=0.4, lw=2.5)
            ax.add_patch(c)
            ax.text(px, py, lbl, ha="center", va="center", fontsize=11, fontweight="bold", color="white")
        ax.text(positions[0][0], 0.4, f"{layer_name} ({len(positions)})",
                ha="center", fontsize=10, color=MUTED)

    # 全连接边
    for a_pos, _ in zip(layers[0][0], layers[0][1]):
        for b_pos, _ in zip(layers[1][0], layers[1][1]):
            ax.plot([a_pos[0]+0.4, b_pos[0]-0.4], [a_pos[1], b_pos[1]], color=MUTED, lw=0.5, alpha=0.4)
    for a_pos, _ in zip(layers[1][0], layers[1][1]):
        for b_pos, _ in zip(layers[2][0], layers[2][1]):
            ax.plot([a_pos[0]+0.4, b_pos[0]-0.4], [a_pos[1], b_pos[1]], color=MUTED, lw=0.5, alpha=0.4)

    save(fig, f"{OUT_DIR}/ch05_mlp_topology_{lang}.png")


def two_lines_compose():
    fig, ax = new_figure(6, 5, 120, "两条直线组合 = 弯线（XOR 区域）")

    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    Y = np.array([0, 1, 1, 0])

    xx, yy = np.meshgrid(np.linspace(-0.5, 1.5, 200), np.linspace(-0.5, 1.5, 200))

    # 第 1 个隐藏神经元：检测 (0,1) 区
    Z1 = -xx + 2*yy - 0.5
    # 第 2 个隐藏神经元：检测 (1,0) 区
    Z2 = 2*xx - yy - 0.5
    # 组合：Z1 OR Z2
    Z_combine = np.maximum(0, Z1) + np.maximum(0, Z2)

    ax.contourf(xx, yy, Z_combine, levels=[-0.1, 0.5, 5], colors=[ACCENT2 + "22", ACCENT + "55"], alpha=0.7)
    ax.contour(xx, yy, Z1, levels=[0], colors=[WARN], linewidths=2)
    ax.contour(xx, yy, Z2, levels=[0], colors=[ACCENT2], linewidths=2)

    # 4 个点
    colors = [DANGER if y == 1 else ACCENT2 for y in Y]
    ax.scatter(X[:, 0], X[:, 1], c=colors, s=300, edgecolors="white", linewidths=2, zorder=5)
    for (x, y), lbl in zip(X, Y):
        ax.annotate(f"({x},{y})→{lbl}", (x, y), xytext=(10, 10), textcoords="offset points",
                    fontsize=10, color=TEXT)

    ax.set_xlim(-0.5, 1.5); ax.set_ylim(-0.5, 1.5)
    ax.set_xlabel("x₁"); ax.set_ylabel("x₂")
    ax.text(1.1, 1.35, "黄线 OR 蓝线 = 弯线", fontsize=10, color=ACCENT, fontweight="bold")
    save(fig, f"{OUT_DIR}/ch05_two_lines_compose.png")


if __name__ == "__main__":
    print(f"== ch05_mlp（使用字体：{CHOSEN_FONT}）==")
    mlp_topology("zh")
    mlp_topology("en")
    two_lines_compose()
    print(f"完成 3 张 MLP 配图")
