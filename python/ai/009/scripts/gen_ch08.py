"""
ch08_backprop: 反向传播计算图（双语）
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# 让 OUT_DIR = "../assets/images" 解析到 009/assets/images/
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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


def chain_rule_backprop():
    fig, ax = plt.subplots(figsize=(14, 5), dpi=120)
    ax.set_xlim(0, 14); ax.set_ylim(0, 6)
    ax.axis("off")

    title = "Backprop: from L back to W₁ (chain rule, 4 steps)" if IS_EN else "反向传播：从 L 一路反推到 W₁（4 步链式法则）"
    ax.text(7, 5.5, title, ha="center", fontsize=14, fontweight="bold", color=TEXT)

    # 节点（同 ch06）
    nodes = [
        (1, 3, "X", ACCENT2),
        (3, 3, "W₁", ACCENT),
        (5, 3, "z₁", WARN),
        (7, 3, "σ", ACCENT2),
        (9, 3, "a₁", WARN),
        (11, 3, "W₂", ACCENT),
        (12.5, 3, "z₂", WARN),
        (13.5, 3, "L", DANGER),
    ]
    for x, y, lbl, c in nodes:
        if lbl in ["σ"]:
            ax.add_patch(mpatches.FancyBboxPatch((x-0.4, y-0.4), 0.8, 0.8,
                boxstyle="round,pad=0.05", fc=c, ec=c, alpha=0.2, lw=2))
        else:
            ax.add_patch(plt.Circle((x, y), 0.4, fc=c, ec=c, alpha=0.3, lw=2.5))
        ax.text(x, y, lbl, ha="center", va="center", fontsize=11, fontweight="bold", color="white")

    # 前向箭头（黑色实线，向右）
    for i in range(len(nodes) - 1):
        if nodes[i][2] in ["σ"] or nodes[i+1][2] in ["σ"]:
            continue
        x1, x2 = nodes[i][0] + 0.4, nodes[i+1][0] - 0.4
        ax.annotate("", xy=(x2, 3), xytext=(x1, 3),
                    arrowprops=dict(arrowstyle="->", lw=1.2, color=MUTED, alpha=0.5))

    # 反向箭头（红色虚线，向左）
    for i in range(len(nodes) - 1, 0, -1):
        if nodes[i-1][2] in ["σ"] or nodes[i][2] in ["σ"]:
            continue
        x1, x2 = nodes[i][0] - 0.4, nodes[i-1][0] + 0.4
        ax.annotate("", xy=(x2, 2.2), xytext=(x1, 2.2),
                    arrowprops=dict(arrowstyle="->", lw=1.5, color=DANGER, linestyle="--"))
        # 公式标注
        if i == 1:  # W₁ 上的梯度
            ax.text((x1+x2)/2, 1.5, r"$\frac{\partial L}{\partial W_1}$",
                    ha="center", fontsize=11, color=DANGER, fontweight="bold")

    # 关键公式
    formula = r"$\frac{\partial L}{\partial W_1} = \frac{\partial L}{\partial z_2} \cdot \frac{\partial z_2}{\partial a_1} \cdot \frac{\partial a_1}{\partial z_1} \cdot \frac{\partial z_1}{\partial W_1}$"
    suffix = "  (chain rule: 4 partial derivatives multiplied)" if IS_EN else "  (链式法则：4 个偏导相乘)"
    ax.text(7, 0.3, formula + suffix, ha="center", fontsize=12, color=TEXT, style="italic")

    save(fig, f"{OUT_DIR}/ch08_chain_rule_{LANG}.png")


if __name__ == "__main__":
    print(f"== ch08_backprop [{LANG}]（使用字体：{CHOSEN_FONT}）==")
    chain_rule_backprop()
    print(f"完成 1 张反向传播图（{LANG}）")
