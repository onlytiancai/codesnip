"""
ch00_math: 6 件兵器配图（双语）
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# 让 OUT_DIR = "../assets/images" 解析到 009/assets/images/（与原版行为一致）
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from _fonts import new_figure, save, CHOSEN_FONT

LANG = sys.argv[1] if len(sys.argv) > 1 else "zh"
IS_EN = LANG == "en"

OUT_DIR = "../assets/images"

# 主题色（与 CSS 变量保持一致）
ACCENT = "#10b981"
ACCENT2 = "#0ea5e9"
WARN = "#f59e0b"
DANGER = "#ef4444"
MUTED = "#64748b"
TEXT = "#1e293b"
BG = "#fafbff"


def function_machine():
    """0.1 函数机器：输入 → f → 输出"""
    fig, ax = plt.subplots(figsize=(8, 3.5), dpi=120)
    ax.set_xlim(0, 10); ax.set_ylim(0, 4)
    ax.axis("off")

    # 输入圆
    input_circle = plt.Circle((1.5, 2), 0.7, color=ACCENT2, alpha=0.25, ec=ACCENT2, lw=2)
    ax.add_patch(input_circle)
    ax.text(1.5, 2, "x", ha="center", va="center", fontsize=24, fontweight="bold", color=ACCENT2)
    ax.text(1.5, 0.7, "Input" if IS_EN else "输入", ha="center", fontsize=11, color=MUTED)

    # 机器方框
    box = mpatches.FancyBboxPatch((3.5, 1), 3, 2, boxstyle="round,pad=0.1",
                                   fc=ACCENT, ec=ACCENT, alpha=0.2, lw=2.5)
    ax.add_patch(box)
    ax.text(5, 2, r"$f(x)$", ha="center", va="center", fontsize=22, fontweight="bold", color=ACCENT)
    ax.text(5, 0.7, "Function" if IS_EN else "函数（机器）", ha="center", fontsize=11, color=MUTED)

    # 输出圆
    out_circle = plt.Circle((8.5, 2), 0.7, color=WARN, alpha=0.25, ec=WARN, lw=2)
    ax.add_patch(out_circle)
    ax.text(8.5, 2, "y", ha="center", va="center", fontsize=24, fontweight="bold", color=WARN)
    ax.text(8.5, 0.7, "Output" if IS_EN else "输出", ha="center", fontsize=11, color=MUTED)

    # 箭头
    ax.annotate("", xy=(3.4, 2), xytext=(2.3, 2), arrowprops=dict(arrowstyle="->", lw=2, color=TEXT))
    ax.annotate("", xy=(7.7, 2), xytext=(6.6, 2), arrowprops=dict(arrowstyle="->", lw=2, color=TEXT))

    # 例子
    example = r"Example: $f(x) = 2x + 1$, input $3 \to$ output $7$" if IS_EN else r"例子：$f(x) = 2x + 1$，输入 3 → 输出 7"
    ax.text(5, 3.4, example, ha="center", fontsize=12, color=TEXT, style="italic")

    save(fig, f"{OUT_DIR}/ch00_function_machine_{LANG}.png")


def coordinate_grid():
    """0.2 坐标系：横轴/纵轴 + 4 个标点"""
    title = "Coordinate System (Cartesian)" if IS_EN else "坐标系（笛卡尔）"
    fig, ax = new_figure(7, 6, 120, title)
    ax.set_xlim(-5, 5); ax.set_ylim(-5, 5)

    points = [(3, 4, "A"), (-2, 3, "B"), (3, -2, "C"), (-3, -3, "D")]
    colors = [ACCENT, ACCENT2, WARN, DANGER]
    for (x, y, name), c in zip(points, colors):
        ax.plot(x, y, "o", color=c, markersize=12, markeredgecolor="white", markeredgewidth=2)
        ax.annotate(f"  {name}({x}, {y})", (x, y), fontsize=12, color=TEXT, fontweight="bold")

    ax.axhline(0, color=TEXT, lw=1, alpha=0.5)
    ax.axvline(0, color=TEXT, lw=1, alpha=0.5)
    ax.set_xlabel("x (horizontal)" if IS_EN else "x (横轴)", fontsize=11, color=MUTED)
    ax.set_ylabel("y (vertical)" if IS_EN else "y (纵轴)", fontsize=11, color=MUTED)
    ax.set_xticks(range(-5, 6))
    ax.set_yticks(range(-5, 6))
    save(fig, f"{OUT_DIR}/ch00_coordinate_grid_{LANG}.png")


def slope_tangent():
    """0.3 斜率：曲线 + 切线"""
    if IS_EN:
        title = r"Slope: tangent of $f(x) = x^2$ at $x=1.5$"
    else:
        title = r"斜率：$f(x) = x^2$ 在 $x=1.5$ 处的切线"
    fig, ax = new_figure(7, 5, 120, title)
    x = np.linspace(-3, 3, 200)
    ax.plot(x, x**2, color=ACCENT, lw=2.5, label=r"$f(x) = x^2$")

    x0 = 1.5
    slope = 2 * x0
    y0 = x0 ** 2
    x_tan = np.linspace(-0.5, 3, 50)
    y_tan = slope * (x_tan - x0) + y0
    tan_label = f"Tangent (slope={slope:.1f})" if IS_EN else f"切线 (斜率={slope:.1f})"
    ax.plot(x_tan, y_tan, "--", color=DANGER, lw=2, label=tan_label)
    ax.plot(x0, y0, "o", color=DANGER, markersize=12, markeredgecolor="white", markeredgewidth=2)
    if IS_EN:
        note = f"Tangent point ({x0}, {y0:.1f})\nTangent slope = {slope:.1f}"
    else:
        note = f"切点 ({x0}, {y0:.1f})\n切线斜率 = {slope:.1f}"
    ax.annotate(note, xy=(x0, y0),
                xytext=(2.2, 6), fontsize=11, color=DANGER, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=DANGER, lw=1.5))
    ax.legend(loc="upper left", fontsize=10)
    ax.set_xlabel("x", color=MUTED); ax.set_ylabel("f(x)", color=MUTED)
    save(fig, f"{OUT_DIR}/ch00_slope_tangent_{LANG}.png")


def vector_personal():
    """0.4 向量：4 维个人档案"""
    fig, ax = plt.subplots(figsize=(6, 4), dpi=120)
    ax.axis("off")
    ax.set_xlim(0, 10); ax.set_ylim(0, 6)

    # 方括号
    ax.plot([1, 1], [0.5, 5.5], color=TEXT, lw=2.5)
    ax.plot([9, 9], [0.5, 5.5], color=TEXT, lw=2.5)
    ax.plot([1, 2.5], [0.5, 0.5], color=TEXT, lw=2.5)
    ax.plot([1, 2.5], [5.5, 5.5], color=TEXT, lw=2.5)
    ax.plot([7.5, 9], [0.5, 0.5], color=TEXT, lw=2.5)
    ax.plot([7.5, 9], [5.5, 5.5], color=TEXT, lw=2.5)

    # 内容
    if IS_EN:
        items = [("Height (cm)", "165", ACCENT2),
                 ("Weight (kg)", "50", ACCENT),
                 ("Age", "12", WARN),
                 ("Math score", "92", DANGER)]
    else:
        items = [("身高 (cm)", "165", ACCENT2),
                 ("体重 (kg)", "50", ACCENT),
                 ("年龄", "12", WARN),
                 ("数学成绩", "92", DANGER)]
    for i, (lbl, val, c) in enumerate(items):
        y = 4.6 - i * 1.3
        ax.text(2.8, y, lbl, fontsize=12, color=MUTED, va="center")
        ax.text(7.2, y, val, fontsize=14, color=c, fontweight="bold", ha="right", va="center")
        if i < 3:
            ax.plot([1, 9], [y - 0.6, y - 0.6], color=MUTED, lw=0.5, alpha=0.3)

    title = "Xiaoming's 4-D Vector" if IS_EN else "小明的 4 维向量"
    ax.text(5, 5.8, title, ha="center", fontsize=14, fontweight="bold", color=TEXT)
    save(fig, f"{OUT_DIR}/ch00_vector_personal_{LANG}.png")


def matrix_multiplication():
    """0.5 矩阵乘法：2×2 × 2×1 = 2×1"""
    fig, ax = plt.subplots(figsize=(8, 4), dpi=120)
    ax.axis("off")
    ax.set_xlim(0, 14); ax.set_ylim(0, 6)

    # A (2x2)
    ax.add_patch(mpatches.Rectangle((1, 2), 3, 3, fc=ACCENT, ec=ACCENT, alpha=0.15, lw=2))
    ax.text(2.5, 5.4, "A (2×2)", ha="center", fontsize=12, fontweight="bold", color=ACCENT)
    A = [[1, 2], [3, 4]]
    for i in range(2):
        for j in range(2):
            ax.text(1.7 + j * 1.4, 4.2 - i * 1.2, str(A[i][j]), ha="center", va="center",
                    fontsize=16, fontweight="bold", color=TEXT)

    # ×
    ax.text(4.7, 3.5, "×", ha="center", va="center", fontsize=28, fontweight="bold", color=MUTED)

    # B (2x1)
    ax.add_patch(mpatches.Rectangle((5.5, 2.7), 1.5, 1.5, fc=ACCENT2, ec=ACCENT2, alpha=0.15, lw=2))
    ax.text(6.25, 4.6, "B (2×1)", ha="center", fontsize=12, fontweight="bold", color=ACCENT2)
    B = [[5], [6]]
    for i in range(2):
        ax.text(6.25, 3.9 - i * 1.2, str(B[i][0]), ha="center", va="center",
                fontsize=16, fontweight="bold", color=TEXT)

    # = ?
    ax.text(7.8, 3.5, "=", ha="center", va="center", fontsize=24, fontweight="bold", color=MUTED)

    # 结果 (2x1)
    ax.add_patch(mpatches.Rectangle((8.5, 2.7), 1.5, 1.5, fc=WARN, ec=WARN, alpha=0.15, lw=2))
    ax.text(9.25, 4.6, "C (2×1)", ha="center", fontsize=12, fontweight="bold", color=WARN)
    C = [[A[0][0]*B[0][0]+A[0][1]*B[1][0]], [A[1][0]*B[0][0]+A[1][1]*B[1][0]]]
    ax.text(9.25, 3.9, str(C[0][0]), ha="center", va="center", fontsize=16, fontweight="bold", color=TEXT)
    ax.text(9.25, 2.7, str(C[1][0]), ha="center", va="center", fontsize=16, fontweight="bold", color=TEXT)

    # 详细计算
    ax.text(11.5, 3.9, r"$c_{1} = 1 \times 5 + 2 \times 6$", fontsize=11, color=TEXT, va="center")
    ax.text(11.5, 2.7, r"$c_{2} = 3 \times 5 + 4 \times 6$", fontsize=11, color=TEXT, va="center")

    shape_note = "Shape: (2×2) × (2×1) = (2×1)\nleft cols = right rows → match" if IS_EN else "形状：(2×2) × (2×1) = (2×1)\n左列数 = 右行数 → 匹配"
    ax.text(7, 0.5, shape_note, ha="center",
            fontsize=10, color=MUTED, style="italic")

    save(fig, f"{OUT_DIR}/ch00_matrix_multiplication_{LANG}.png")


def chain_rule():
    """0.6 链式法则：嵌套函数的信号反向流"""
    fig, ax = plt.subplots(figsize=(9, 3), dpi=120)
    ax.axis("off")
    ax.set_xlim(0, 14); ax.set_ylim(0, 4)

    # 三个方框 + 箭头
    boxes = [(2, "x", ACCENT2), (5.5, "g", ACCENT), (9, "f", WARN), (12, "L", DANGER)]
    for x, lbl, c in boxes:
        ax.add_patch(mpatches.FancyBboxPatch((x-0.7, 1.3), 1.4, 1.4, boxstyle="round,pad=0.05",
                                              fc=c, ec=c, alpha=0.2, lw=2))
        ax.text(x, 2, lbl, ha="center", va="center", fontsize=18, fontweight="bold", color=c)

    # 正向箭头
    for i in range(3):
        x1 = boxes[i][0] + 0.7
        x2 = boxes[i+1][0] - 0.7
        ax.annotate("", xy=(x2, 2), xytext=(x1, 2),
                    arrowprops=dict(arrowstyle="->", lw=2, color=TEXT))

    # 反向箭头（红色虚线）
    for i in range(3):
        x1 = boxes[i+1][0] - 0.7
        x2 = boxes[i][0] + 0.7
        ax.annotate("", xy=(x2, 1.0), xytext=(x1, 1.0),
                    arrowprops=dict(arrowstyle="->", lw=1.5, color=DANGER, linestyle="--"))

    # 标签
    fwd = "Forward" if IS_EN else "前向（Forward）"
    bwd = "Backward" if IS_EN else "反向（Backward）"
    ax.text(7, 3.5, fwd, ha="center", fontsize=11, color=ACCENT, fontweight="bold")
    ax.text(7, 0.5, bwd, ha="center", fontsize=11, color=DANGER, fontweight="bold")
    ax.text(11, 0.0, r"$\frac{\partial L}{\partial x} = \frac{\partial L}{\partial f} \cdot \frac{\partial f}{\partial g} \cdot \frac{\partial g}{\partial x}$",
            ha="center", fontsize=10, color=TEXT, style="italic")

    save(fig, f"{OUT_DIR}/ch00_chain_rule_{LANG}.png")


if __name__ == "__main__":
    print(f"== ch00_math [{LANG}]（使用字体：{CHOSEN_FONT}）==")
    function_machine()
    coordinate_grid()
    slope_tangent()
    vector_personal()
    matrix_multiplication()
    chain_rule()
    print(f"完成 6 张数学配图（{LANG}）")
