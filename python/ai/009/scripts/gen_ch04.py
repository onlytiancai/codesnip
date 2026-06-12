"""
ch04_xor: XOR 难题（双语）
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# 让 OUT_DIR = "../assets/images" 解析到 009/assets/images/
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib.pyplot as plt
from _fonts import new_figure, save, CHOSEN_FONT

LANG = sys.argv[1] if len(sys.argv) > 1 else "zh"
IS_EN = LANG == "en"

OUT_DIR = "../assets/images"
ACCENT = "#10b981"
ACCENT2 = "#0ea5e9"
WARN = "#f59e0b"
DANGER = "#ef4444"
MUTED = "#64748b"
TEXT = "#1e293b"


def xor_scatter():
    title = "XOR Problem: one line cannot separate 4 points" if IS_EN else "XOR 难题：一条直线分不开 4 个点"
    fig, ax = new_figure(6, 5, 120, title)

    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    Y = np.array([0, 1, 1, 0])

    # 3 条尝试失败的直线
    if IS_EN:
        attempts = [
            (0.5, 0.5, -0.3, DANGER, "Try 1"),
            (0.7, -0.7, 0.0, WARN, "Try 2"),
            (-0.5, 0.5, 0.3, ACCENT2, "Try 3"),
        ]
    else:
        attempts = [
            (0.5, 0.5, -0.3, DANGER, "尝试 1"),
            (0.7, -0.7, 0.0, WARN, "尝试 2"),
            (-0.5, 0.5, 0.3, ACCENT2, "尝试 3"),
        ]
    xx, yy = np.meshgrid(np.linspace(-0.5, 1.5, 100), np.linspace(-0.5, 1.5, 100))
    for w1, w2, b, c, _ in attempts:
        Z = w1 * xx + w2 * yy + b
        ax.contour(xx, yy, Z, levels=[0], colors=[c], linewidths=1.8, alpha=0.7, linestyles="--")

    # 4 个点
    colors = [DANGER if y == 1 else ACCENT2 for y in Y]
    ax.scatter(X[:, 0], X[:, 1], c=colors, s=300, edgecolors="white", linewidths=2, zorder=5)
    for (x, y), lbl in zip(X, Y):
        ax.annotate(f"({x},{y})→{lbl}", (x, y), xytext=(10, 10), textcoords="offset points",
                    fontsize=10, color=TEXT)

    ax.set_xlim(-0.5, 1.5); ax.set_ylim(-0.5, 1.5)
    ax.set_xlabel("x₁"); ax.set_ylabel("x₂")
    save(fig, f"{OUT_DIR}/ch04_xor_scatter_{LANG}.png")


if __name__ == "__main__":
    print(f"== ch04_xor [{LANG}]（使用字体：{CHOSEN_FONT}）==")
    xor_scatter()
    print(f"完成 1 张 XOR 配图（{LANG}）")
