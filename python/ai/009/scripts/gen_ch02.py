"""
ch02_neuron: 神经元公式对照图（双语）
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# 让 OUT_DIR = "../assets/images" 解析到 009/assets/images/
os.chdir(os.path.dirname(os.path.abspath(__file__)))

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


def neuron_function():
    fig, ax = plt.subplots(figsize=(10, 4.5), dpi=120)
    ax.set_xlim(0, 14); ax.set_ylim(0, 6)
    ax.axis("off")

    title = "Neuron = Function" if IS_EN else "神经元 = 函数"
    ax.text(7, 5.5, title, ha="center", fontsize=18, fontweight="bold", color=TEXT)

    # 5 步流水线
    if IS_EN:
        steps = [
            ("x", "Input", ACCENT2, 1.5),
            ("w·x + b", "Linear", ACCENT, 4.3),
            ("z", "Sum", ACCENT, 7),
            ("σ(z)", "Activate", WARN, 9.7),
            ("y", "Output", DANGER, 12.3),
        ]
    else:
        steps = [
            ("x", "输入", ACCENT2, 1.5),
            ("w·x + b", "线性变换", ACCENT, 4.3),
            ("z", "求和结果", ACCENT, 7),
            ("σ(z)", "激活", WARN, 9.7),
            ("y", "输出", DANGER, 12.3),
        ]
    for i, (lbl, sub, c, x) in enumerate(steps):
        box = mpatches.FancyBboxPatch((x-1.0, 1.5), 2.0, 2.0, boxstyle="round,pad=0.1",
                                       fc=c, ec=c, alpha=0.15, lw=2)
        ax.add_patch(box)
        ax.text(x, 2.8, lbl, ha="center", va="center", fontsize=15, fontweight="bold", color=c)
        ax.text(x, 1.9, sub, ha="center", fontsize=10, color=MUTED)
        # 箭头
        if i < 4:
            nx = steps[i+1][3]
            ax.annotate("", xy=(nx-1.1, 2.5), xytext=(x+1.0, 2.5),
                        arrowprops=dict(arrowstyle="->", lw=1.8, color=TEXT))

    # 底部公式
    if IS_EN:
        caption = r"Overall: $y = \sigma(w \cdot x + b)$ — a function of 4 numbers $(x, w, b, \sigma)$"
    else:
        caption = r"整体看：$y = \sigma(w \cdot x + b)$ —— 一个『装』了 4 个数 $(x, w, b, \sigma)$ 的函数"
    ax.text(7, 0.5, caption, ha="center", fontsize=12, color=TEXT, style="italic")

    save(fig, f"{OUT_DIR}/ch02_neuron_function_{LANG}.png")


if __name__ == "__main__":
    print(f"== ch02_neuron [{LANG}]（使用字体：{CHOSEN_FONT}）==")
    neuron_function()
    print(f"完成 1 张神经元公式图（{LANG}）")
