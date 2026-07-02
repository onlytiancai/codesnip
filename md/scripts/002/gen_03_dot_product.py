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
    axL.text(a[0] + 0.1, a[1] + 0.1, r"$a=(2.5,0.5)$", color=COLOR_Q, fontsize=12, fontweight="bold")
    axL.text(b[0] + 0.1, b[1] + 0.1, r"$b=(2.2,1.6)$", color=COLOR_K, fontsize=12, fontweight="bold")

    # 夹角弧（从向量 a 的方向画到向量 b 的方向）
    angle_a = np.arctan2(a[1], a[0])
    angle_b = np.arctan2(b[1], b[0])
    arc = np.linspace(angle_a, angle_b, 40)
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

    # ====== 右图：Q 与 K(animal) / K(street) 的 dot product 对比 ======
    # 真实值：Q=[0,1,0,0]，K(animal)=[1,1,1,0]，K(street)=[1,0,0,0]
    # 取 dim_1 × dim_2 画 2D：Q=(0.5, 0.9), K(animal)=(0.7, 0.85), K(street)=(0.9, 0.3)
    Q = np.array([0.5, 0.9])
    K_animal = np.array([0.7, 0.85])  # Q·K ≈ 1.04，与 Q 夹角约 23°
    K_street = np.array([0.9, 0.3])   # Q·K ≈ 0.72，与 Q 夹角约 57°

    axR.set_xlim(-0.1, 1.2)
    axR.set_ylim(-0.1, 1.2)
    axR.set_aspect("equal")
    axR.axhline(0, color="#888", lw=0.6)
    axR.axvline(0, color="#888", lw=0.6)
    axR.grid(True, alpha=0.3)
    axR.set_title("Q · K：animal 投影更长 → 点积更大")

    # Q 向量（查询方向，蓝色）
    axR.annotate("", xy=Q, xytext=(0, 0),
                 arrowprops=dict(arrowstyle="->", color=COLOR_K, lw=2.5))
    axR.text(Q[0] - 0.45, Q[1] - 0.12, r"$Q=(0.5,0.9)$",
             color=COLOR_K, fontsize=12, fontweight="bold")

    # Q 方向上的单位向量（虚线，作为投影参照线）
    q_dir = Q / np.linalg.norm(Q)
    axR.plot([0, q_dir[0] * 1.4], [0, q_dir[1] * 1.4],
             color=COLOR_K, lw=1.0, ls="--", alpha=0.5)

    # K(animal)：绿色，投影最长
    proj_a = np.dot(K_animal, q_dir) * q_dir
    axR.annotate("", xy=K_animal, xytext=(0, 0),
                 arrowprops=dict(arrowstyle="->", color="#54A24B", lw=2.5))
    axR.text(K_animal[0] - 0.01, K_animal[1] - 0.10, r"$K_{animal}=(0.7,0.85)$",
             color="#54A24B", fontsize=12, fontweight="bold")
    # 投影线（红色虚线，从 K_animal 垂直到 Q 方向）
    axR.plot([K_animal[0], proj_a[0]], [K_animal[1], proj_a[1]],
             color="#E53935", lw=1.5, ls="--")
    # 垂直投影参考线
    #axR.plot([K_animal[0], K_animal[0]], [K_animal[1], 0],
    #         color="#E53935", lw=1.0, ls=":", alpha=0.6)
    axR.text(proj_a[0] + 0.05, proj_a[1] + 0.01,
             #rf"$|Q| cos \theta_a = {np.dot(K_animal, Q):.2f}$", 
             r"$|K_{animal}| \cos\theta_a \approx 1.01$",
             color="#E53935", fontsize=9)

    # K(street)：灰色，投影为 0 → dot product = 0
    proj_s = np.dot(K_street, q_dir) * q_dir
    axR.annotate("", xy=K_street, xytext=(0, 0),
                 arrowprops=dict(arrowstyle="->", color=COLOR_NEUTRAL, lw=2.5))
    axR.text(K_street[0] - 0.15, K_street[1] - 0.1, r"$K_{street}=(0.9,0.3)$",
             color=COLOR_NEUTRAL, fontsize=12, fontweight="bold")
    # 投影线（垂直到 Q 方向）
    axR.plot([K_street[0], proj_s[0]], [K_street[1], proj_s[1]],
             color="#E53935", lw=1.5, ls="--")
    #axR.text(proj_s[0] - 0.28, proj_s[1] + 0.04,
    #         f"$|Q|\cos\theta_s={np.dot(K_street, Q):.2f}$", color="#E53935", fontsize=9)

    # dot product 结果标注
    d_a = np.dot(Q, K_animal)
    d_s = np.dot(Q, K_street)
    theta_a = np.degrees(np.arccos(np.clip(d_a / (np.linalg.norm(Q) * np.linalg.norm(K_animal)), -1, 1)))
    theta_s = np.degrees(np.arccos(np.clip(d_s / (np.linalg.norm(Q) * np.linalg.norm(K_street)), -1, 1)))
    axR.text(0.05, 0.96,
             r"$Q\!\cdot\!K(animal) = " + f"{d_a:.2f}" + r"$" + "\n" +
             r"$Q\!\cdot\!K(street) = " + f"{d_s:.2f}" + r"$" + "\n" +
             r"$\theta_{animal} \approx "+f"{theta_a:.0f}$°" + "\n" +
             r"$\theta_{street} \approx "+f"{theta_s:.0f}$°",
             transform=axR.transAxes, va="top", fontsize=10.5, color="#222",
             bbox=dict(boxstyle="round,pad=0.5",
                       facecolor="#E8F5E9", edgecolor="#54A24B", lw=1.5))

    axR.set_xlabel(r"$\text{dim}_1$")
    axR.set_ylabel(r"$\text{dim}_2$")

    fig.suptitle(r"点积越大 = 两向量方向越一致 = 越像", fontsize=13, fontweight="bold")
    save(fig, "03_dot_product_angle.png")


if __name__ == "__main__":
    main()
