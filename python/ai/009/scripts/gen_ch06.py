"""
ch06_forward: 前向计算图（双语）
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


def compute_graph():
    fig, ax = plt.subplots(figsize=(14, 4), dpi=120)
    ax.set_xlim(0, 14); ax.set_ylim(0, 5)
    ax.axis("off")

    if IS_EN:
        title = "Forward Pass: $X \\to z_1 \\to a_1 \\to z_2 \\to a_2 = \\hat{y}$"
    else:
        title = "前向传播 (Forward Pass): $X \\to z_1 \\to a_1 \\to z_2 \\to a_2 = \\hat{y}$"
    ax.text(7, 4.5, title, ha="center", fontsize=14, fontweight="bold", color=TEXT)

    # 节点
    if IS_EN:
        nodes = [
            (1, 2, "X", ACCENT2, "Input (2,)"),
            (3, 2, "W₁, b₁", MUTED, "Params"),
            (5, 2, "z₁", ACCENT, "Linear (4,)"),
            (7, 2, "σ", WARN, "Activate"),
            (9, 2, "a₁", ACCENT, "(4,)"),
            (11, 2, "z₂", ACCENT, "(1,)"),
            (13, 2, "ŷ", DANGER, "Predict"),
        ]
    else:
        nodes = [
            (1, 2, "X", ACCENT2, "输入 (2,)"),
            (3, 2, "W₁, b₁", MUTED, "参数"),
            (5, 2, "z₁", ACCENT, "线性 (4,)"),
            (7, 2, "σ", WARN, "激活"),
            (9, 2, "a₁", ACCENT, "(4,)"),
            (11, 2, "z₂", ACCENT, "(1,)"),
            (13, 2, "ŷ", DANGER, "预测"),
        ]
    for x, y, lbl, c, sub in nodes:
        size = 0.6 if lbl not in ["W₁, b₁", "σ"] else 0.45
        if lbl == "σ":
            ax.add_patch(mpatches.FancyBboxPatch((x-0.55, y-0.45), 1.1, 0.9,
                boxstyle="round,pad=0.05", fc=c, ec=c, alpha=0.18, lw=2))
            ax.text(x, y, lbl, ha="center", va="center", fontsize=14, fontweight="bold", color=c)
        else:
            circle = plt.Circle((x, y), size, fc=c, ec=c, alpha=0.3, lw=2.5)
            ax.add_patch(circle)
            ax.text(x, y, lbl, ha="center", va="center", fontsize=12, fontweight="bold", color="white")
        ax.text(x, y - 1.0, sub, ha="center", fontsize=9, color=MUTED)

    # 箭头
    for i in range(len(nodes) - 1):
        if nodes[i][2] == "W₁, b₁" or nodes[i+1][2] == "W₁, b₁":
            # 跳过 W₁, b₁（连到 W₁, b₁ 的特殊样式）
            continue
        x1, x2 = nodes[i][0] + 0.6, nodes[i+1][0] - 0.6
        if nodes[i+1][2] == "σ":
            x2 = nodes[i+1][0] - 0.55
        ax.annotate("", xy=(x2, 2), xytext=(x1, 2),
                    arrowprops=dict(arrowstyle="->", lw=1.8, color=TEXT))

    # 标注 W₁, b₁ 从 W₁, b₁ 指向 z₁
    w_x = nodes[1][0]
    z_x = nodes[2][0]
    ax.annotate("", xy=(z_x - 0.6, 2), xytext=(w_x + 0.5, 2.5),
                arrowprops=dict(arrowstyle="->", lw=1.5, color=ACCENT2, linestyle="--"))
    ax.annotate("", xy=(nodes[5][0] - 0.6, 2), xytext=(w_x + 0.5, 1.5),
                arrowprops=dict(arrowstyle="->", lw=1.5, color=ACCENT2, linestyle="--"))

    legend = "Solid = data flow; dashed = params" if IS_EN else "实线 = 数据流；虚线 = 参数作用"
    ax.text(7, 0.2, legend, ha="center", fontsize=10, color=MUTED, style="italic")

    save(fig, f"{OUT_DIR}/ch06_compute_graph_{LANG}.png")


if __name__ == "__main__":
    print(f"== ch06_forward [{LANG}]（使用字体：{CHOSEN_FONT}）==")
    compute_graph()
    print(f"完成 1 张前向计算图（{LANG}）")
