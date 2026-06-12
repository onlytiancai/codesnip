"""
ch09_gradient: 损失曲线 + 梯度下降（双语）
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# 让 OUT_DIR = "../assets/images" 解析到 009/assets/images/
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib.pyplot as plt
from _fonts import new_figure, save, CHOSEN_FONT

LANG = sys.argv[1] if len(sys.argv) > 1 else "zh"
IS_EN = LANG == "en"

OUT_DIR = "../assets/images"
ACCENT = "#10b981"
ACCENT2 = "#0ea5e9"
WARN = "#f59e0b"
DANGER = "#ef4444"
MUTED = "#64748b"
TEXT = "#1e293b"


def loss_curve():
    """训练 200 步的 loss 下降 + accuracy 上升（双子图）"""
    np.random.seed(42)

    # 模拟一次训练过程
    N = 200
    t = np.arange(1, N + 1)
    # 指数衰减 + 噪声
    loss = 0.7 * np.exp(-t / 50) + 0.02 * np.exp(-t / 200) + 0.005 * np.random.rand(N)
    acc = 1 - 0.8 * np.exp(-t / 40) + 0.04 * np.random.rand(N)
    acc = np.clip(acc, 0, 1)

    if IS_EN:
        sub_titles = ["Loss decreasing", "Accuracy increasing"]
        xlabel = "Epoch (training steps)"
        suptitle = "200 training steps: loss ↓, acc ↑"
    else:
        sub_titles = ["损失下降曲线", "正确率上升曲线"]
        xlabel = "Epoch (训练步数)"
        suptitle = "训练 200 步：loss ↓, acc ↑"

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), dpi=120)
    for ax, data, color, ylabel, title in zip(
        axes, [loss, acc], [DANGER, ACCENT2], ["Loss", "Accuracy"], sub_titles
    ):
        ax.plot(t, data, color=color, lw=2)
        ax.fill_between(t, 0, data, color=color, alpha=0.15)
        ax.set_xlabel(xlabel, fontsize=11, color=MUTED)
        ax.set_ylabel(ylabel, fontsize=11, color=color, fontweight="bold")
        ax.set_title(title, fontsize=12, fontweight="bold", color=TEXT)
        ax.set_xlim(0, N)
        ax.grid(True, alpha=0.3)

    fig.suptitle(suptitle, fontsize=14, fontweight="bold", color=TEXT, y=1.02)
    save(fig, f"{OUT_DIR}/ch09_loss_curve_{LANG}.png")


def gradient_descent():
    """损失曲面 + 下降路径"""
    fig = plt.figure(figsize=(8, 5), dpi=120)
    ax = fig.add_subplot(111, projection="3d")

    # 简单的二次损失曲面 L = w1² + w2²
    w1 = np.linspace(-3, 3, 50)
    w2 = np.linspace(-3, 3, 50)
    W1, W2 = np.meshgrid(w1, w2)
    L = W1**2 + W2**2

    surf = ax.plot_surface(W1, W2, L, cmap="RdYlGn_r", alpha=0.6, edgecolor="none")

    # 梯度下降路径（沿负梯度方向）
    path_w1, path_w2 = [2.5], [2.5]
    lr = 0.15
    for _ in range(20):
        dw1 = -2 * path_w1[-1]
        dw2 = -2 * path_w2[-1]
        path_w1.append(path_w1[-1] + lr * dw1)
        path_w2.append(path_w2[-1] + lr * dw2)
    path_w1 = np.array(path_w1)
    path_w2 = np.array(path_w2)
    path_L = path_w1**2 + path_w2**2

    if IS_EN:
        path_label = "Gradient descent path"
        opt_label = "Optimum"
        title = "Gradient descent: down the negative gradient"
    else:
        path_label = "梯度下降路径"
        opt_label = "最优点"
        title = "梯度下降：沿负梯度方向下山"

    ax.plot(path_w1, path_w2, path_L + 0.1, "o-", color=ACCENT, lw=2, markersize=4, label=path_label)
    ax.scatter([0], [0], [0], color="red", s=200, marker="*", label=opt_label)

    ax.set_xlabel("w₁", fontsize=10, color=MUTED)
    ax.set_ylabel("w₂", fontsize=10, color=MUTED)
    ax.set_zlabel("Loss", fontsize=10, color=MUTED)
    ax.set_title(title, fontsize=12, fontweight="bold", color=TEXT)
    ax.view_init(elev=30, azim=45)
    ax.legend(loc="upper right", fontsize=9)
    save(fig, f"{OUT_DIR}/ch09_gradient_descent_{LANG}.png")


if __name__ == "__main__":
    print(f"== ch09_gradient [{LANG}]（使用字体：{CHOSEN_FONT}）==")
    loss_curve()
    gradient_descent()
    print(f"完成 2 张梯度下降配图（{LANG}）")
