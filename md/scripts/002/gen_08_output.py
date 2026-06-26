"""
图 8: output[0] = 0.43 的瀑布图分解
用于 §5 / §6 output 解读
"""
import math
import matplotlib.pyplot as plt
import numpy as np
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
    # W_V 只保留 dim0 → V(t)[0] = embedding(t)[0] = 1 仅对 animal/street
    V0 = [1, 1, 0, 0, 0, 1, 0]   # dim0 of V
    contributions = [w * v for w, v in zip(weights, V0)]
    # 实际值: animal 0.3118, street 0.1147, 其余 0
    total = sum(contributions)  # ≈ 0.4265

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    # 堆叠：animal 在底，street 叠在上面
    bottom = 0
    color_map = [COLOR_NEUTRAL] * 7
    color_map[1] = COLOR_HIGHLIGHT
    color_map[5] = "#4C78A8"

    for i, (tok, c) in enumerate(zip(tokens, contributions)):
        if c < 1e-6:
            continue
        ax.bar(i, c, bottom=bottom, color=color_map[i], edgecolor="white", lw=1.5)
        ax.text(i, bottom + c / 2, f"{c:.3f}",
                ha="center", va="center", color="white",
                fontsize=11, fontweight="bold")
        bottom += c

    # 顶部 total
    ax.axhline(total, color="#222", ls="--", lw=1)
    ax.text(6.4, total + 0.01, rf"$\text{{output}}[0] = {total:.3f}$",
            ha="right", fontsize=12, fontweight="bold", color="#222")

    # 标注贡献者
    ax.annotate(r"\textbf{animal} 贡献 0.312", xy=(1, contributions[1] / 2),
                xytext=(2.4, contributions[1] - 0.05),
                fontsize=10, color=COLOR_HIGHLIGHT, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=COLOR_HIGHLIGHT, lw=1.2))
    ax.annotate(r"\textbf{street} 贡献 0.115", xy=(5, contributions[1] + contributions[5] / 2),
                xytext=(3.5, contributions[1] + contributions[5] + 0.05),
                fontsize=10, color="#4C78A8", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#4C78A8", lw=1.2))

    ax.set_xticks(range(7))
    ax.set_xticklabels(tokens, fontsize=11)
    ax.set_ylabel(r"对 $\text{output}[0]$ 的贡献")
    ax.set_title(r"$\text{output}[0] = 0.43$ 是怎么从 7 个 token 借来的")
    ax.set_ylim(0, 0.55)
    ax.grid(True, axis="y", alpha=0.3)
    save(fig, "08_output_decomposition.png")


if __name__ == "__main__":
    main()
