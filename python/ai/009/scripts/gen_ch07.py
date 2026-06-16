"""
ch07_loss: BCE 损失曲线（双语）
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


def bce_curve():
    title = r"Binary Cross-Entropy BCE: $L = -[y \log \hat{y} + (1-y)\log(1-\hat{y})]$" if IS_EN else r"二元交叉熵 BCE: $L = -[y \log \hat{y} + (1-y)\log(1-\hat{y})]$"
    fig, ax = new_figure(7, 5, 120, title)

    y_pred = np.linspace(0.001, 0.999, 200)
    bce_y1 = -np.log(y_pred)            # y = 1
    bce_y0 = -np.log(1 - y_pred)         # y = 0

    if IS_EN:
        label1 = r"Loss when $y=1$: $-\log(\hat{y})$"
        label2 = r"Loss when $y=0$: $-\log(1-\hat{y})$"
        note1 = "Predict 0.99, true 1: loss ≈ 0.01"
        note2 = "Predict 0.01, true 1: loss ≈ 4.6"
        xlabel = r"Prediction $\hat{y}$"
        ylabel = "Loss L"
    else:
        label1 = r"$y=1$ 时的损失：$-\log(\hat{y})$"
        label2 = r"$y=0$ 时的损失：$-\log(1-\hat{y})$"
        note1 = "预测 0.99 真值 1：损失 ≈ 0.01"
        note2 = "预测 0.01 真值 1：损失 ≈ 4.6"
        xlabel = r"预测值 $\hat{y}$"
        ylabel = "损失 L"

    ax.plot(y_pred, bce_y1, color=ACCENT2, lw=2.5, label=label1)
    ax.plot(y_pred, bce_y0, color=DANGER, lw=2.5, label=label2)
    ax.fill_between(y_pred, 0, bce_y1, color=ACCENT2, alpha=0.1)
    ax.fill_between(y_pred, 0, bce_y0, color=DANGER, alpha=0.1)

    # 标注"对的预测损失小，错的大"
    ax.plot([0.99], [-np.log(0.99)], "o", color=ACCENT, markersize=10)
    ax.annotate(note1, xy=(0.99, -np.log(0.99)),
                xytext=(0.5, 4), fontsize=10, color=ACCENT,
                arrowprops=dict(arrowstyle="->", color=ACCENT, lw=1.5))
    ax.plot([0.01], [-np.log(0.01)], "o", color=DANGER, markersize=10)
    ax.annotate(note2, xy=(0.01, -np.log(0.01)),
                xytext=(0.4, 5.5), fontsize=10, color=DANGER,
                arrowprops=dict(arrowstyle="->", color=DANGER, lw=1.5))

    ax.set_xlabel(xlabel, fontsize=11, color=MUTED)
    ax.set_ylabel(ylabel, fontsize=11, color=MUTED)
    ax.set_xlim(0, 1); ax.set_ylim(0, 6)
    ax.legend(loc="upper center", fontsize=10)
    save(fig, f"{OUT_DIR}/ch07_bce_curve_{LANG}.png")


if __name__ == "__main__":
    print(f"== ch07_loss [{LANG}]（使用字体：{CHOSEN_FONT}）==")
    bce_curve()
    print(f"完成 1 张 BCE 损失曲线（{LANG}）")
