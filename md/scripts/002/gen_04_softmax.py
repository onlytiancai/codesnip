"""
图 4: softmax —— 从实数（logits）到概率分布
用于 §1.4 softmax
双子图：
  左：softmax 函数曲线 f(x_i) = e^{x_i} / Σ e^{x_j}（n=7 demo 场景）
  右：把 demo 里 7 个 Q·K 分数 [1, 0, 0, 0, 0, 0, 0] 送进 softmax 得到权重
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
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 4.6))

    # ====== 左图：softmax 曲线（把某个 xᵢ 推高，看其它维度被压扁） ======
    base = np.zeros(7)
    xs = np.linspace(-3, 3, 200)
    # 取 5 条不同 logits 下的 softmax 曲线
    for highlight in [0.5, 1.0, 1.5, 2.0, 2.5]:
        logits = base.copy()
        logits[2] = highlight  # 把第 3 个 logit 推高
        weights = np.array([math.exp(x) for x in (logits - logits.max())])
        weights = weights / weights.sum()
        axL.plot(xs, weights[2] * np.exp(-(xs - highlight)),
                 label=f"logit=2 维={highlight}")
    # 简化：实际画 7 个 token 的权重随 "animal 维 logit" 变化
    animal_axis = np.linspace(-2, 3, 100)
    weights_animal = []
    for a in animal_axis:
        logits = np.zeros(7)
        logits[1] = a
        w = np.exp(logits - logits.max())
        w = w / w.sum()
        weights_animal.append(w[1])
    axL.plot(animal_axis, weights_animal, color=COLOR_HIGHLIGHT, lw=2.5)
    axL.fill_between(animal_axis, 0, weights_animal, alpha=0.18, color=COLOR_HIGHLIGHT)
    axL.set_xlabel(r"\text{animal} 那一个 token 的 logit 值")
    axL.set_ylabel(r"softmax 后它分到的权重")
    axL.set_title(r"一个维度拉高 $\to$ 它的权重指数级增长")
    axL.grid(True, alpha=0.3)
    axL.axhline(1 / 7, color="#888", ls="--", lw=0.8)
    axL.text(2.3, 1 / 7 + 0.01, r"$1/7 = $ 平均权重", color="#666", fontsize=10)

    # ====== 右图：demo 场景的 logits → weights ======
    tokens = ["The", "animal", "didn't", "cross", "the", "street", "it"]
    scores = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    weights = softmax(scores)

    x = range(len(tokens))
    bars = axR.bar(x, scores, color=COLOR_NEUTRAL, edgecolor="white")
    bars[1].set_color(COLOR_HIGHLIGHT)
    axR.set_xticks(list(x))
    axR.set_xticklabels(tokens, rotation=20, ha="right")
    axR.set_ylim(0, 1.2)
    axR.set_ylabel(r"$Q \cdot K$ 分数 (logit)")
    axR.set_title("Step 1：算 Q·K（logits）")
    axR.grid(True, axis="y", alpha=0.3)
    for i, s in enumerate(scores):
        axR.text(i, s + 0.04, f"{s:.1f}", ha="center", fontsize=10, color="#555")

    # 在右图下方再画一个 weights 子图
    axR2 = axR.twinx()
    axR2.plot(x, weights, "o-", color=COLOR_HIGHLIGHT, lw=2, markersize=8)
    axR2.set_ylim(0, 0.4)
    axR2.set_ylabel(r"softmax 权重", color=COLOR_HIGHLIGHT)
    axR2.tick_params(axis="y", labelcolor=COLOR_HIGHLIGHT)
    for i, w in enumerate(weights):
        axR2.text(i, w + 0.012, f"{w:.3f}", ha="center",
                  fontsize=10, color=COLOR_HIGHLIGHT, fontweight="bold")

    fig.suptitle(r"softmax：把 $K$ 个实数变成 $K$ 个加起来等于 1 的概率", fontsize=13, fontweight="bold")
    save(fig, "04_softmax.png")


if __name__ == "__main__":
    main()
