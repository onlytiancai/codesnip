"""
图 6: Q(it) · K(t) 7 个分数条形图
用于 §4.1 步骤 1：算 Q·K
"""
import matplotlib.pyplot as plt
from _style import save, COLOR_HIGHLIGHT, COLOR_NEUTRAL


def main():
    tokens = ["The", "animal", "didn't", "cross", "the", "street", "it"]
    # 真实跑出来的分数（来自 code/attention_demo.py 当前版本）
    scores = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    fig, ax = plt.subplots(figsize=(7, 4))
    colors = [COLOR_HIGHLIGHT if i == 1 else COLOR_NEUTRAL for i in range(7)]
    bars = ax.bar(tokens, scores, color=colors, edgecolor="white", lw=1.5)

    for bar, s in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width() / 2, s + 0.04, f"{s:.1f}",
                ha="center", fontsize=12,
                color=COLOR_HIGHLIGHT if s > 0 else "#888",
                fontweight="bold" if s > 0 else "normal")

    ax.set_ylim(0, 1.25)
    ax.set_ylabel(r"$\text{score} = Q(it) \cdot K(t)$")
    ax.set_title(r"步骤 1：对所有 token 算 $Q(it) \cdot K(t)$")
    ax.grid(True, axis="y", alpha=0.3)

    ax.annotate(
        r"$Q(it)$ 只想找\n「有生命」",
        xy=(1, 1.0), xytext=(3.2, 1.05),
        fontsize=11, color=COLOR_HIGHLIGHT, fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=COLOR_HIGHLIGHT, lw=1.5),
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFF8E1",
                  edgecolor=COLOR_HIGHLIGHT),
    )
    save(fig, "06_qk_scores.png")


if __name__ == "__main__":
    main()
