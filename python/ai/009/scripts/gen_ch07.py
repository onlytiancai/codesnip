"""
ch07_loss: BCE 损失曲线
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


def bce_curve():
    fig, ax = new_figure(7, 5, 120, r"二元交叉熵 BCE: $L = -[y \log \hat{y} + (1-y)\log(1-\hat{y})]$")

    y_pred = np.linspace(0.001, 0.999, 200)
    bce_y1 = -np.log(y_pred)            # y = 1
    bce_y0 = -np.log(1 - y_pred)         # y = 0

    ax.plot(y_pred, bce_y1, color=ACCENT2, lw=2.5, label=r"$y=1$ 时的损失：$-\log(\hat{y})$")
    ax.plot(y_pred, bce_y0, color=DANGER, lw=2.5, label=r"$y=0$ 时的损失：$-\log(1-\hat{y})$")
    ax.fill_between(y_pred, 0, bce_y1, color=ACCENT2, alpha=0.1)
    ax.fill_between(y_pred, 0, bce_y0, color=DANGER, alpha=0.1)

    # 标注"对的预测损失小，错的大"
    ax.plot([0.99], [-np.log(0.99)], "o", color=ACCENT, markersize=10)
    ax.annotate("预测 0.99 真值 1：损失 ≈ 0.01", xy=(0.99, -np.log(0.99)),
                xytext=(0.5, 4), fontsize=10, color=ACCENT,
                arrowprops=dict(arrowstyle="->", color=ACCENT, lw=1.5))
    ax.plot([0.01], [-np.log(0.01)], "o", color=DANGER, markersize=10)
    ax.annotate("预测 0.01 真值 1：损失 ≈ 4.6", xy=(0.01, -np.log(0.01)),
                xytext=(0.4, 5.5), fontsize=10, color=DANGER,
                arrowprops=dict(arrowstyle="->", color=DANGER, lw=1.5))

    ax.set_xlabel(r"预测值 $\hat{y}$", fontsize=11, color=MUTED)
    ax.set_ylabel("损失 L", fontsize=11, color=MUTED)
    ax.set_xlim(0, 1); ax.set_ylim(0, 6)
    ax.legend(loc="upper center", fontsize=10)
    save(fig, f"{OUT_DIR}/ch07_bce_curve.png")


if __name__ == "__main__":
    print(f"== ch07_loss（使用字体：{CHOSEN_FONT}）==")
    bce_curve()
    print(f"完成 1 张 BCE 损失曲线")
