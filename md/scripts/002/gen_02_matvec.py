"""
图 2: 矩阵 × 向量 —— 把矩阵的每一行看作"询问器"
用于 §1.2 矩阵乘向量 + §3 W_K · embedding 的可视化
"""
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from _style import save, COLOR_K, COLOR_HIGHLIGHT, COLOR_NEUTRAL


def main():
    fig, ax = plt.subplots(figsize=(8, 5.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")

    # 矩阵 W_K (4x4) - 4 个询问器（行）
    W_K = [
        [1, 0, 0, 0],   # dim0: 名词
        [0, 1, 0, 0],   # dim1: 有生命
        [0, 0, 1, 0],   # dim2: 能累
        [0, 0, 0, 0],   # dim3
    ]
    row_labels = ["dim0=名词", "dim1=有生命", "dim2=能累", "dim3=占位"]

    # 左：W_K 矩阵
    ax.text(0.5, 5.5, r"$W_K$ (4$\times$4 矩阵)", fontsize=13, fontweight="bold")
    cell_w, cell_h = 0.55, 0.55
    for r in range(4):
        for c in range(4):
            v = W_K[r][c]
            color = COLOR_K if v == 1 else "#EEEEEE"
            ax.add_patch(Rectangle((1 + c * cell_w, 4.0 - r * cell_h),
                                    cell_w, cell_h,
                                    facecolor=color, edgecolor="white", lw=1.2))
            ax.text(1 + c * cell_w + cell_w / 2,
                    4.0 - r * cell_h + cell_h / 2,
                    f"{v:.0f}", ha="center", va="center",
                    fontsize=11, color="white" if v == 1 else "#999")
        ax.text(0.5, 4.0 - r * cell_h + cell_h / 2,
                row_labels[r], ha="right", va="center", fontsize=10, color="#444")

    # 中：embedding("animal") = [1,1,1,0]
    ax.text(4.8, 5.5, r"$\text{embedding}(animal)$", fontsize=13, fontweight="bold")
    emb = [1, 1, 1, 0]
    for i, v in enumerate(emb):
        y = 4.0 - i * cell_h
        color = COLOR_HIGHLIGHT if v == 1 else "#EEEEEE"
        ax.add_patch(Rectangle((4.8, y), cell_w, cell_h,
                                facecolor=color, edgecolor="white", lw=1.2))
        ax.text(4.8 + cell_w / 2, y + cell_h / 2, f"{v}",
                ha="center", va="center", fontsize=11,
                color="white" if v == 1 else "#999")

    # 乘号
    ax.text(4.0, 2.7, r"$\times$", fontsize=22, ha="center", va="center", fontweight="bold")

    # 等号
    ax.text(6.7, 2.7, "=", fontsize=22, ha="center", va="center", fontweight="bold")

    # 右：K(animal) = [1,1,1,0]
    ax.text(7.2, 5.5, r"$K(\text{animal})$", fontsize=13, fontweight="bold", color=COLOR_K)
    for i, v in enumerate(emb):
        y = 4.0 - i * cell_h
        color = COLOR_K
        ax.add_patch(Rectangle((7.2, y), cell_w, cell_h,
                                facecolor=color, edgecolor="white", lw=1.2))
        ax.text(7.2 + cell_w / 2, y + cell_h / 2, f"{v}",
                ha="center", va="center", fontsize=11, color="white", fontweight="bold")

    # 中间高亮：W_K 第 2 行 × embedding dim1 = 1 (核心示范)
    ax.annotate(
        "",
        xy=(4.8 + cell_w * 0.5, 4.0 - 1 * cell_h + cell_h / 2),
        xytext=(1 + 1 * cell_w + cell_w / 2, 4.0 - 1 * cell_h + cell_h / 2),
        arrowprops=dict(arrowstyle="->", color=COLOR_HIGHLIGHT, lw=2.2, alpha=0.8),
    )
    ax.text(3.0, 3.1, r"第 2 行 $\cdot$ 第 1 个数 = $1 \times 1 = 1$",
            ha="center", fontsize=10, color=COLOR_HIGHLIGHT, fontweight="bold")

    # 底部说明
    ax.text(5, 0.6,
            r"矩阵的行 = 4 个询问器; 每行点积 embedding 的对应维度, 得到 $K$ 的一维",
            ha="center", fontsize=11, color="#444",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#FFF8E1", edgecolor="#DDD"))

    save(fig, "02_matvec.png")


if __name__ == "__main__":
    main()
