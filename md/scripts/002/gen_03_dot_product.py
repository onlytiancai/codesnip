"""
图 3: 点积 = |a||b|cosθ，几何意义
用于 §1.3 点积的几何意义 + Q·K(animal) vs Q·K(street) 的对比
"""
import matplotlib.pyplot as plt
import numpy as np
from _style import save, COLOR_Q, COLOR_K, COLOR_NEUTRAL


def main():
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.8))

    # ====== 左图：点积的几何意义 ======
    a = np.array([2.5, 0.5])
    b = np.array([2.2, 1.6])
    cos_t = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    theta = np.degrees(np.arccos(np.clip(cos_t, -1, 1)))

    axL.set_xlim(-0.3, 4.0)
    axL.set_ylim(-0.3, 2.4)
    axL.set_aspect("equal")
    axL.axhline(0, color="#888", lw=0.6)
    axL.axvline(0, color="#888", lw=0.6)
    axL.grid(True, alpha=0.3)
    axL.set_title(r"点积 $= |a||b|\cos\theta$")

    axL.annotate("", xy=a, xytext=(0, 0),
                 arrowprops=dict(arrowstyle="->", color=COLOR_Q, lw=2.5))
    axL.annotate("", xy=b, xytext=(0, 0),
                 arrowprops=dict(arrowstyle="->", color=COLOR_K, lw=2.5))
    axL.text(a[0] * 1.05, a[1] * 1.05, "a", color=COLOR_Q, fontsize=14, fontweight="bold")
    axL.text(b[0] * 1.05, b[1] * 1.05, "b", color=COLOR_K, fontsize=14, fontweight="bold")

    # 夹角弧
    arc = np.linspace(0, np.radians(theta), 40)
    r = 0.55
    axL.plot(r * np.cos(arc), r * np.sin(arc), color="#888", lw=1.5)
    axL.text(r * 1.5, r * 0.5, rf"$\theta={theta:.1f}^\circ$", fontsize=11, color="#555")

    # 公式
    axL.text(0.05, 0.95,
             r"$\theta$ 越小 $\to \cos\theta$ 越接近 1" + "\n" + r"$\to a \cdot b$ 越大 $\to$ 越像",
             transform=axL.transAxes, va="top", fontsize=11, color="#444",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#FFF8E1", edgecolor="#DDD"))

    axL.set_xlabel(r"$x_1$")
    axL.set_ylabel(r"$x_2$")

    # ====== 右图：Q(it) 与 K(animal) / K(street) 的对比 ======
    # 真实值：Q(it) = [0,1,0,0]，K(animal) = [1,1,1,0]，K(street) = [1,0,0,0]
    # 取最后 2 维画 2D：Q=(0,0), K(animal)=(1,1), K(street)=(1,0)
    Q = np.array([0.0, 0.0])
    K_animal = np.array([1.0, 1.0])   # 与 Q 夹角 45°
    K_street = np.array([1.0, 0.0])   # 与 Q 夹角 90°

    axR.set_xlim(-0.3, 2.0)
    axR.set_ylim(-0.5, 1.6)
    axR.set_aspect("equal")
    axR.axhline(0, color="#888", lw=0.6)
    axR.axvline(0, color="#888", lw=0.6)
    axR.grid(True, alpha=0.3)
    axR.set_title("Q(it) · K(t)：animal 比 street 大")

    # 三条向量
    for vec, c, lbl in [(Q, COLOR_NEUTRAL, r"$Q(it)=(0,0)$"),
                        (K_animal, "#54A24B", r"$K(\text{animal})=(1,1)$"),
                        (K_street, COLOR_NEUTRAL, r"$K(\text{street})=(1,0)$")]:
        axR.annotate("", xy=vec, xytext=(0, 0),
                     arrowprops=dict(arrowstyle="->", color=c, lw=2.5))
        if lbl:
            axR.text(vec[0] * 1.1, vec[1] * 1.15, lbl,
                     color=c, fontsize=11, fontweight="bold")

    # 点积结果
    d_a = np.dot(Q, K_animal)
    d_s = np.dot(Q, K_street)
    axR.text(0.05, 0.95,
             r"$Q \cdot K(animal) = $" + f"{d_a:.1f}" + "\n" +
             r"$Q \cdot K(street) = $" + f"{d_s:.1f}" + "\n\n" +
             "animal 胜出!",
             transform=axR.transAxes, va="top", fontsize=11, color="#222",
             bbox=dict(boxstyle="round,pad=0.4",
                       facecolor="#E8F5E9", edgecolor="#54A24B", lw=1.5))

    axR.set_xlabel(r"$\text{dim}_1$ (K 投影后第一维示意)")
    axR.set_ylabel(r"$\text{dim}_2$")

    fig.suptitle(r"点积越大 = 两向量方向越一致 = 越像", fontsize=13, fontweight="bold")
    save(fig, "03_dot_product_angle.png")


if __name__ == "__main__":
    main()
