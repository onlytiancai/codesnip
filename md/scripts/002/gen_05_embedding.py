"""
图 5: 7 token × 4 维 embedding 热图
用于 §2 Embedding：从 token 到向量
"""
import matplotlib.pyplot as plt
import numpy as np
from _style import save, COLOR_HIGHLIGHT, COLOR_NEUTRAL


def main():
    tokens = ["The", "animal", "didn't", "cross", "the", "street", "it"]
    dim_labels = ["dim0\n名词", "dim1\n有生命", "dim2\n能累", "dim3\n代词"]
    # 取自 code/attention_demo.py
    embeddings = np.array([
        [0, 0, 0, 0],   # The
        [1, 1, 1, 0],   # animal
        [0, 0, 0, 0],   # didn't
        [0, 0, 0, 0],   # cross
        [0, 0, 0, 0],   # the
        [1, 0, 0, 0],   # street
        [0, 0, 0, 1],   # it
    ], dtype=float)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    im = ax.imshow(embeddings, cmap="YlGnBu", vmin=0, vmax=1, aspect="auto")

    ax.set_xticks(range(4))
    ax.set_xticklabels(dim_labels, fontsize=11)
    ax.set_yticks(range(len(tokens)))
    ax.set_yticklabels(tokens, fontsize=12)
    ax.set_title(r"Embedding：每个 token $\to$ 一个 4 维 0/1 向量")

    # 标注每个格子
    for r in range(len(tokens)):
        for c in range(4):
            v = embeddings[r, c]
            color = "white" if v > 0.5 else "#444"
            ax.text(c, r, f"{int(v)}", ha="center", va="center",
                    fontsize=13, fontweight="bold", color=color)

    # 在右边加一列说明
    ax.text(4.3, 6, "animal", fontsize=11, color="#222", fontweight="bold",
            va="center")
    ax.text(4.3, 1, "it", fontsize=11, color=COLOR_HIGHLIGHT, fontweight="bold",
            va="center")
    ax.text(5.4, 6, "名词+有生命\n+能累", fontsize=10, color="#444", va="center")
    ax.text(5.4, 1, "代词", fontsize=10, color=COLOR_HIGHLIGHT, va="center")

    ax.set_ylim(len(tokens) - 0.5, -0.5)
    save(fig, "05_embedding_heatmap.png")


if __name__ == "__main__":
    main()
