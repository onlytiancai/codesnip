"""
ch01_intro: 神经元卡通
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

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


def neuron_cartoon(lang="zh"):
    is_en = lang == "en"
    fig, ax = plt.subplots(figsize=(9, 5), dpi=120)
    ax.set_xlim(0, 12); ax.set_ylim(0, 7)
    ax.axis("off")

    # 标题
    ax.text(6, 6.5, "一个神经元" if not is_en else "One Neuron",
            ha="center", fontsize=18, fontweight="bold", color=TEXT)

    # 输入（左 3 个圆球）
    for i, lbl in enumerate(["x₁", "x₂", "x₃"]):
        y = 4.5 - i * 1.5
        c = plt.Circle((1.5, y), 0.5, color=ACCENT2, alpha=0.7, ec=ACCENT2, lw=2)
        ax.add_patch(c)
        ax.text(1.5, y, lbl, ha="center", va="center", fontsize=14, fontweight="bold", color="white")

    # 中央圆球（神经元主体）
    body = plt.Circle((6, 3), 1.2, color=ACCENT, alpha=0.3, ec=ACCENT, lw=3)
    ax.add_patch(body)
    # 内部"细胞核"
    nucleus = plt.Circle((6, 3), 0.4, color=ACCENT, alpha=0.8, ec=ACCENT, lw=1.5)
    ax.add_patch(nucleus)
    ax.text(6, 3, "σ", ha="center", va="center", fontsize=20, fontweight="bold", color="white")

    # 触突（连线）
    for i in range(3):
        y = 4.5 - i * 1.5
        ax.annotate("", xy=(4.8, 3), xytext=(2.0, y),
                    arrowprops=dict(arrowstyle="->", lw=1.5, color=MUTED, alpha=0.6))
        # 权重标签
        ax.text((2.0 + 4.8) / 2, (y + 3) / 2 - 0.1, f"w{i+1}", fontsize=10, color=ACCENT,
                fontweight="bold", ha="center", bbox=dict(boxstyle="round,pad=0.2",
                fc="white", ec=ACCENT, lw=0.5))

    # 偏置
    ax.annotate("", xy=(6, 4.2), xytext=(6, 5.3),
                arrowprops=dict(arrowstyle="->", lw=1.5, color=WARN))
    ax.text(6.4, 4.9, "b" if not is_en else "bias", fontsize=11, color=WARN, fontweight="bold")

    # 输出
    ax.annotate("", xy=(10.5, 3), xytext=(7.2, 3),
                arrowprops=dict(arrowstyle="->", lw=2, color=DANGER))
    out_circle = plt.Circle((11, 3), 0.5, color=DANGER, alpha=0.7, ec=DANGER, lw=2)
    ax.add_patch(out_circle)
    ax.text(11, 3, "y", ha="center", va="center", fontsize=14, fontweight="bold", color="white")
    ax.text(11, 2, "输出" if not is_en else "output", ha="center", fontsize=10, color=MUTED)

    # 标签
    ax.text(1.5, 1, "输入 (Input)" if not is_en else "Inputs", ha="center", fontsize=10, color=MUTED)
    ax.text(6, 1, "神经元 (Neuron)" if not is_en else "Neuron", ha="center", fontsize=10, color=MUTED)
    ax.text(6, 0.3, r"$y = \sigma(w_1 x_1 + w_2 x_2 + w_3 x_3 + b)$",
            ha="center", fontsize=12, color=TEXT, style="italic")

    save(fig, f"{OUT_DIR}/ch01_neuron_cartoon_{lang}.png")


if __name__ == "__main__":
    print(f"== ch01_intro（使用字体：{CHOSEN_FONT}）==")
    neuron_cartoon("zh")
    neuron_cartoon("en")
    print(f"完成 2 张神经元配图")
