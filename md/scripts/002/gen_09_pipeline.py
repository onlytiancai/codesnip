"""
图 9: 整体 Self-Attention 流程框图
用 matplotlib.patches 自绘，零外部依赖
用于 §4.4 整体管道
"""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from _style import save, COLOR_Q, COLOR_K, COLOR_V, COLOR_HIGHLIGHT, COLOR_NEUTRAL


def box(ax, x, y, w, h, text, color="#4C78A8", text_color="white", fs=11):
    patch = FancyBboxPatch((x, y), w, h,
                            boxstyle="round,pad=0.05,rounding_size=0.12",
                            facecolor=color, edgecolor="white", lw=2)
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, color=text_color, fontweight="bold")


def arrow(ax, x1, y1, x2, y2, color="#888"):
    a = FancyArrowPatch((x1, y1), (x2, y2),
                         arrowstyle="->,head_length=8,head_width=5",
                         color=color, lw=1.6,
                         connectionstyle="arc3,rad=0.0")
    ax.add_patch(a)


def main():
    fig, ax = plt.subplots(figsize=(12, 7.5))
    ax.set_xlim(0, 12)
    ax.set_ylim(1.8, 8)
    ax.axis("off")

    # ====== 输入层 ======
    box(ax, 0.3, 6.5, 1.6, 0.8, r"$\mathbf{X}$" + "\n(seq "+r"$\times$"+" d)", "#666", "white", 10)
    # 三路投影
    box(ax, 0.0, 4.8, 1.2, 0.7, r"$\mathbf{W}_Q$", COLOR_Q, "white", 11)
    box(ax, 0.0, 3.8, 1.2, 0.7, r"$\mathbf{W}_K$", COLOR_K, "white", 11)
    box(ax, 0.0, 2.8, 1.2, 0.7, r"$\mathbf{W}_V$", COLOR_V, "white", 11)

    arrow(ax, 1.1, 6.5, 0.6, 5.5)
    arrow(ax, 1.1, 6.5, 0.6, 4.5)
    arrow(ax, 1.1, 6.5, 0.6, 3.5)

    # Q/K/V
    box(ax, 1.6, 4.8, 1.0, 0.7, r"$\mathbf{Q}$" + "\n(seq "+r"$\times$"+" d)", COLOR_Q, "white", 10)
    box(ax, 1.6, 3.8, 1.0, 0.7, r"$\mathbf{K}$" + "\n(seq "+r"$\times$"+" d)", COLOR_K, "white", 10)  
    box(ax, 1.6, 2.8, 1.0, 0.7, r"$\mathbf{V}$" + "\n(seq "+r"$\times$"+" d)", COLOR_V, "white", 10)


    # ====== Q·K^T ======
    box(ax, 3.2, 4.0, 1.5, 0.9, r"$\mathbf{Q}\mathbf{K}^\top$" + "\n(seq "+r"$\times$"+" seq)", "#555", "white", 10)
    arrow(ax, 2.6, 5.15, 3.2, 4.7, COLOR_Q)
    arrow(ax, 2.6, 4.15, 3.2, 4.3, COLOR_K)

    # ====== Scale (示意) ======
    box(ax, 5.2, 4.0, 1.3, 0.9, r"$\frac{1}{\sqrt{d_k}}$", "#999", "white", 14)
    arrow(ax, 4.7, 4.45, 5.2, 4.45)

    # ====== Softmax ======
    box(ax, 7.0, 4.0, 1.5, 0.9, r"$\text{softmax}$"+"\n(逐行)", COLOR_HIGHLIGHT, "white", 10)
    arrow(ax, 6.5, 4.45, 7.0, 4.45, "#555")

    # ====== × V ======
    box(ax, 9.0, 4.0, 1.4, 0.9, r"$\times$ $\mathbf{V}$"+"\n(seq "+r"$\times$"+" d)", "#555", "white", 10)
    arrow(ax, 8.5, 4.45, 9.0, 4.45, "#555")
    # V 接到这里
    arrow(ax, 2.6, 3.15, 9.0, 4.0, COLOR_V)

    # ====== Output ======
    box(ax, 10.7, 3.8, 1.1, 1.3, r"$\mathbf{Y}$" + "\n(seq "+r"$\times$"+" d)", "#333", "white", 10)
    arrow(ax, 10.4, 4.45, 10.7, 4.45)

    # 顶部标题 + 公式
    ax.text(6, 7.6, r"Self-Attention 完整流程",
            ha="center", fontsize=15, fontweight="bold", color="#222")
    ax.text(6, 7.1, r"$\text{Attention}(\mathbf{Q},\mathbf{K},\mathbf{V}) = \text{softmax}\!\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}}\right)\mathbf{V}$",
            ha="center", fontsize=13, color="#444")
    ax.text(3, 6.65, r"seq = 序列长度，d = embedding 维度，$d_k$ = $\mathbf{Q}\mathbf{K}^\top$ 向量维度", ha="left", fontsize=10, color="#888")
    ax.text(3, 6.4, "Q：我当前需要什么信息？K：我提供什么特征供别人匹配？V：如果匹配成功，我真正传递什么内容？", ha="left", fontsize=10, color="#888")

    save(fig, "09_qkv_pipeline.png")


if __name__ == "__main__":
    main()
