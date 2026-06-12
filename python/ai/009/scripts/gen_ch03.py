"""
ch03_perceptron: 感知机
- perceptron_boundary: AND 4 点 + 训练直线
- and_or_scatter: AND/OR 可分对比
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib.pyplot as plt
from _fonts import new_figure, save, CHOSEN_FONT

OUT_DIR = "../assets/images"
ACCENT = "#10b981"
ACCENT2 = "#0ea5e9"
WARN = "#f59e0b"
DANGER = "#ef4444"
MUTED = "#64748b"
TEXT = "#1e293b"


def perceptron_boundary():
    fig, ax = new_figure(6, 5, 120, "感知机学到的分界线（AND）")

    # AND 真值表的 4 个点
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    Y = np.array([0, 0, 0, 1])

    # 学到的分界线（w = [0.5, 0.7], b = -0.6，约 0.5x + 0.7y - 0.6 = 0）
    w = np.array([0.5, 0.7]); b = -0.6

    # 网格背景色
    xx, yy = np.meshgrid(np.linspace(-0.5, 1.5, 100), np.linspace(-0.5, 1.5, 100))
    Z = w[0] * xx + w[1] * yy + b
    ax.contourf(xx, yy, Z, levels=[-10, 0, 10], colors=[ACCENT2 + "22", ACCENT + "33"], alpha=0.6)
    ax.contour(xx, yy, Z, levels=[0], colors=[DANGER], linewidths=2.5)

    # 4 个点
    colors = [DANGER if y == 1 else ACCENT2 for y in Y]
    ax.scatter(X[:, 0], X[:, 1], c=colors, s=200, edgecolors="white", linewidths=2, zorder=5)
    for (x, y), lbl in zip(X, ["(0,0)→0", "(0,1)→0", "(1,0)→0", "(1,1)→1"]):
        ax.annotate(lbl, (x, y), xytext=(10, 10), textcoords="offset points",
                    fontsize=10, color=TEXT)

    ax.set_xlim(-0.5, 1.5); ax.set_ylim(-0.5, 1.5)
    ax.set_xlabel("x₁", fontsize=12, color=MUTED)
    ax.set_ylabel("x₂", fontsize=12, color=MUTED)
    ax.text(1.1, 1.35, r"分界线：$0.5 x_1 + 0.7 x_2 - 0.6 = 0$", fontsize=10, color=DANGER, fontweight="bold")

    save(fig, f"{OUT_DIR}/ch03_perceptron_boundary.png")


def and_or_scatter():
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=120)
    for ax, (name, Y) in zip(axes, [("AND", [0, 0, 0, 1]), ("OR", [0, 1, 1, 1])]):
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        ax.scatter(X[:, 0], X[:, 1], c=[DANGER if y == 1 else ACCENT2 for y in Y],
                   s=300, edgecolors="white", linewidths=2)
        for (x, y), lbl in zip(X, Y):
            ax.annotate(f"({x},{y})→{lbl}", (x, y), xytext=(8, 8), textcoords="offset points",
                        fontsize=10, color=TEXT)
        # 决策边界（感知机可以学到）
        ax.axhline(0.5, color=DANGER, lw=2, linestyle="--", alpha=0.7)
        ax.set_xlim(-0.5, 1.5); ax.set_ylim(-0.5, 1.5)
        ax.set_title(f"{name}：一条直线分得开", fontsize=12, fontweight="bold", color=TEXT)
        ax.set_xlabel("x₁"); ax.set_ylabel("x₂")
        ax.grid(True, alpha=0.3)

    fig.suptitle("AND / OR 都是线性可分的", fontsize=13, fontweight="bold", color=TEXT, y=1.02)
    save(fig, f"{OUT_DIR}/ch03_and_or_scatter.png")


if __name__ == "__main__":
    print(f"== ch03_perceptron（使用字体：{CHOSEN_FONT}）==")
    perceptron_boundary()
    and_or_scatter()
    print(f"完成 2 张感知机配图")
