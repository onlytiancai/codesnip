"""
图 1: 向量作为 2D 平面上的箭头
用于 §1.1 标量与向量
"""
import matplotlib.pyplot as plt
from _style import save, COLOR_Q, COLOR_K, COLOR_V, COLOR_NEUTRAL


def main():
    fig, ax = plt.subplots(figsize=(6, 6))

    # 三个示例向量
    vectors = [
        (r"$v_1 = (3, 1)$", 3, 1, COLOR_Q),
        (r"$v_2 = (1, 3)$", 1, 3, COLOR_K),
        (r"$v_3 = (-2, 2)$", -2, 2, COLOR_V),
    ]
    for label, x, y, c in vectors:
        ax.annotate(
            "", xy=(x, y), xytext=(0, 0),
            arrowprops=dict(arrowstyle="->", color=c, lw=2.5),
        )
        ax.text(x * 1.08, y * 1.08, label, color=c, fontsize=13, fontweight="bold")

    # 坐标轴
    ax.axhline(0, color="#888", lw=0.8)
    ax.axvline(0, color="#888", lw=0.8)
    ax.set_xlim(-3.5, 4.2)
    ax.set_ylim(-0.5, 4.2)
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.set_title(r"向量 = 从原点出发的有向箭头")
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")
    ax.text(
        0.02, 0.98,
        r"$n$ 维向量同样有方向 + 长度" + "\n" + r"只是 $n > 2$ 时画不出来",
        transform=ax.transAxes, va="top", ha="left",
        fontsize=11, color="#555",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#FFF8E1", edgecolor="#DDD"),
    )

    save(fig, "01_vector_geometry.png")


if __name__ == "__main__":
    main()
