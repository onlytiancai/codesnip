"""
ch10_train: 决策边界演化（训练前/中/后 3 子图）（双语）
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


def decision_boundary():
    """手画决策边界在训练前/中/后的演化"""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), dpi=120)

    # XOR 4 个点
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    Y = np.array([0, 1, 1, 0])

    xx, yy = np.meshgrid(np.linspace(-0.5, 1.5, 200), np.linspace(-0.5, 1.5, 200))

    # 模拟 3 个训练阶段的决策边界
    if IS_EN:
        stages = [
            ("Before training (epoch 0)\nloss ≈ 0.69, acc = 50%", lambda x, y: 0.5 * np.ones_like(x)),
            ("Mid training (epoch 100)\nloss ≈ 0.35, acc = 75%", lambda x, y: 0.4 * (x - 0.5) + 0.3 * (y - 0.5)),
            ("After training (epoch 1000)\nloss ≈ 0.05, acc = 100%", lambda x, y: 0.7 * np.sin(2 * np.pi * x) * np.cos(2 * np.pi * y)),
        ]
        suptitle = "1000 training steps: decision boundary from flat to curved"
    else:
        stages = [
            ("训练前 (epoch 0)\nloss ≈ 0.69, acc = 50%", lambda x, y: 0.5 * np.ones_like(x)),
            ("训练中 (epoch 100)\nloss ≈ 0.35, acc = 75%", lambda x, y: 0.4 * (x - 0.5) + 0.3 * (y - 0.5)),
            ("训练后 (epoch 1000)\nloss ≈ 0.05, acc = 100%", lambda x, y: 0.7 * np.sin(2 * np.pi * x) * np.cos(2 * np.pi * y)),
        ]
        suptitle = "训练 1000 步：决策边界从无到弯线"

    for ax, (title, boundary_fn) in zip(axes, stages):
        Z = boundary_fn(xx, yy)
        # 用 sigmoid 软化
        pred = 1 / (1 + np.exp(-Z * 3))
        ax.contourf(xx, yy, pred, levels=20, cmap="RdYlGn", alpha=0.6)

        colors = [DANGER if y == 1 else ACCENT2 for y in Y]
        ax.scatter(X[:, 0], X[:, 1], c=colors, s=300, edgecolors="white", linewidths=2, zorder=5)
        for (x, y), lbl in zip(X, Y):
            ax.annotate(f"({x},{y})→{lbl}", (x, y), xytext=(8, 8), textcoords="offset points",
                        fontsize=9, color=TEXT)

        ax.set_xlim(-0.5, 1.5); ax.set_ylim(-0.5, 1.5)
        ax.set_title(title, fontsize=10, fontweight="bold", color=TEXT)
        ax.set_xlabel(r"$x_1$"); ax.set_ylabel(r"$x_2$")

    fig.suptitle(suptitle, fontsize=13, fontweight="bold", color=TEXT, y=1.02)
    save(fig, f"{OUT_DIR}/ch10_decision_boundary_{LANG}.png")


if __name__ == "__main__":
    print(f"== ch10_train [{LANG}]（使用字体：{CHOSEN_FONT}）==")
    decision_boundary()
    print(f"完成 1 张决策边界图（{LANG}）")
