"""
图 7: softmax 后的 7 个注意力权重
用于 §4.2 步骤 2：softmax → 权重
"""
import math
import matplotlib.pyplot as plt
from _style import save, COLOR_HIGHLIGHT, COLOR_NEUTRAL


def softmax(xs):
    m = max(xs)
    exps = [math.exp(x - m) for x in xs]
    s = sum(exps)
    return [e / s for e in exps]


def main():
    tokens = ["The", "animal", "didn't", "cross", "the", "street", "it"]
    scores = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    weights = softmax(scores)
    # = [0.1147, 0.3118, 0.1147, 0.1147, 0.1147, 0.1147, 0.1147]

    fig, ax = plt.subplots(figsize=(7, 4))
    colors = [COLOR_HIGHLIGHT if i == 1 else COLOR_NEUTRAL for i in range(7)]
    bars = ax.bar(tokens, weights, color=colors, edgecolor="white", lw=1.5)

    for bar, w in zip(bars, weights):
        ax.text(bar.get_x() + bar.get_width() / 2, w + 0.008, f"{w:.3f}",
                ha="center", fontsize=11,
                color=COLOR_HIGHLIGHT if w > 0.2 else "#888",
                fontweight="bold" if w > 0.2 else "normal")

    # 标注 baseline 1/7
    ax.axhline(1 / 7, color="#888", ls="--", lw=0.8)
    ax.text(6.1, 1 / 7 + 0.008, r"$1/7 \approx 0.143$\n（平均）",
            ha="right", fontsize=9, color="#666")

    ax.set_ylim(0, 0.4)
    ax.set_ylabel(r"$\alpha$ (注意力权重)")
    ax.set_title(r"步骤 2：softmax $\to$ 加起来等于 1 的概率")
    ax.grid(True, axis="y", alpha=0.3)
    save(fig, "07_attn_weights.png")


if __name__ == "__main__":
    main()
