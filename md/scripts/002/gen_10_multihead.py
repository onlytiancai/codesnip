"""
图 10: Multi-Head Attention 简化示意
用于 §7.3 进阶
"""
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
from _style import save, COLOR_Q, COLOR_K, COLOR_V, COLOR_HIGHLIGHT, COLOR_NEUTRAL


def main():
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 5)
    ax.axis("off")

    # 输入
    ax.text(0.5, 2.5, "X", ha="center", va="center",
            fontsize=18, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#666", edgecolor="white"),
            color="white")
    ax.text(0.5, 1.6, "(seq × d)", ha="center", fontsize=9, color="#666")

    # 8 个 head
    head_colors = ["#E45756", "#F58518", "#EECA3B", "#54A24B",
                   "#4C78A8", "#72B7B2", "#B279A2", "#FF9DA6"]
    for i in range(8):
        y = 4.5 - i * 0.5
        # 投影
        for j, c in enumerate([COLOR_Q, COLOR_K, COLOR_V]):
            ax.add_patch(Rectangle((2.0 + j * 0.35, y - 0.18), 0.32, 0.36,
                                    facecolor=c, edgecolor="white", lw=0.5))
        # head
        ax.add_patch(Rectangle((3.3, y - 0.2), 1.6, 0.4,
                                facecolor=head_colors[i],
                                edgecolor="white", lw=1.2))
        ax.text(4.1, y, f"head {i+1}", ha="center", va="center",
                fontsize=9, color="white", fontweight="bold")
        # 箭头从 X 过来
        arrow = FancyArrowPatch((1.0, 2.5), (2.0, y),
                                 arrowstyle="->,head_length=6,head_width=4",
                                 color="#888", lw=0.8)
        ax.add_patch(arrow)

    # Concat
    box_x = 5.4
    ax.add_patch(Rectangle((box_x, 0.4), 1.4, 4.0,
                            facecolor="#FFE0B2", edgecolor=COLOR_HIGHLIGHT, lw=2))
    ax.text(box_x + 0.7, 2.4, "concat", ha="center", va="center",
            fontsize=12, fontweight="bold", color=COLOR_HIGHLIGHT)
    ax.text(box_x + 0.7, 1.7, "seq × 8d", ha="center", fontsize=10, color="#555")

    # 箭头到 concat
    for i in range(8):
        y = 4.5 - i * 0.5
        ax.add_patch(FancyArrowPatch((4.9, y), (box_x, y),
                                      arrowstyle="->,head_length=5,head_width=3",
                                      color="#888", lw=0.6))

    # W_O
    ax.add_patch(Rectangle((7.2, 2.0), 1.0, 1.0,
                            facecolor="#333", edgecolor="white", lw=1.5))
    ax.text(7.7, 2.5, "W_O", ha="center", va="center",
            fontsize=14, fontweight="bold", color="white")
    ax.add_patch(FancyArrowPatch((6.8, 2.5), (7.2, 2.5),
                                  arrowstyle="->,head_length=8,head_width=5",
                                  color="#555", lw=1.5))

    # 输出
    ax.text(9.5, 2.5, "Output", ha="center", va="center",
            fontsize=14, fontweight="bold", color="white",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#333", edgecolor="white"))
    ax.text(9.5, 1.6, "(seq × d)", ha="center", fontsize=10, color="#666")
    ax.add_patch(FancyArrowPatch((8.2, 2.5), (8.9, 2.5),
                                  arrowstyle="->,head_length=8,head_width=5",
                                  color="#555", lw=1.5))

    # 标题
    ax.text(5.5, 4.85, r"\textbf{Multi-Head Attention}：8 个 head 并行关注不同子空间",
            ha="center", fontsize=13, fontweight="bold", color="#222")

    # 底部说明
    ax.text(5.5, 0.1,
            r"每个 head 学不同的找法 (语义指代 / 句法结构 / 位置关系 ...)",
            ha="center", fontsize=10, color="#666", style="italic")

    save(fig, "10_multihead.png")


if __name__ == "__main__":
    main()
