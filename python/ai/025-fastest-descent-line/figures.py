"""
配图脚本：最速降线与摆线
=========================

运行：
    python figures.py 1   # 仅生成第 1 张图
    python figures.py all # 生成全部 6 张图

生成的图片保存在当前目录，命名 figure1.png ~ figure6.png。
"""

import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")  # 无 GUI 后端，保证脚本环境也能跑
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# ---- 中文字体回退（CLAUDE.md 要求） ----
matplotlib.rcParams["font.sans-serif"] = [
    "PingFang SC",
    "Hiragino Sans GB",
    "Heiti TC",
    "DejaVu Sans",
]
matplotlib.rcParams["axes.unicode_minus"] = False
# 数学公式用 Computer Modern 字体
matplotlib.rcParams["mathtext.fontset"] = "cm"

# ---- 物理参数（无量纲，g=1） ----
A = 1.0          # 摆线方程中的常数 a，对应"轮子半径"
G = 1.0          # 重力加速度（无量纲）
X_END = np.pi    # 终点 x 坐标
Y_END = 2.0      # 终点 y 坐标（向下为正，所以这里 y=2 表示下降 2）

# numpy 2.x 兼容：trapz 被改名为 trapezoid
_trapz = getattr(np, "trapezoid", None) or np.trapz


# ============================================================
# 候选曲线：从 (0, 0) 到 (π, 2) 的三条路径
# 约定：y 向下为正（最速降线的经典设置，与正文推导一致）。
# ============================================================
def line_xy(n=400):
    """直线：y = (2/π) x"""
    x = np.linspace(0, X_END, n)
    y = (Y_END / X_END) * x
    return x, y


def parabola_xy(n=400):
    """抛物线：y = (2/π²) x² （在 x=π 处 y=2，过 (0,0)）"""
    x = np.linspace(0, X_END, n)
    y = (Y_END / X_END ** 2) * x ** 2
    return x, y


def cycloid_xy(n=1000):
    """摆线：a=1, θ ∈ [0, π]，对应起点 (0, 0) → 终点 (π, 2)"""
    theta = np.linspace(0, np.pi, n)
    x = A * (theta - np.sin(theta))
    y = A * (1 - np.cos(theta))
    return x, y


# ============================================================
# 总时间：T = ∫ ds / v = ∫ √(1+y'²) / √(2gy) dx
# y 是下落深度（向下为正），速度 = √(2gy)。
# ============================================================
def travel_time(x, y):
    """数值积分求总时间。物理 g = 1。"""
    drop = np.maximum(y, 0.0)
    integrand = np.sqrt(1 + np.gradient(y, x) ** 2) / np.sqrt(2 * G * drop)
    integrand[0] = integrand[1]
    return _trapz(integrand, x)


# ============================================================
# Figure 1：三条候选路径对比
# 数据用 y 向下为正（与正文一致），但翻转 y 轴让 A 显示在左上、B 在右下。
# ============================================================
def figure1():
    fig, ax = plt.subplots(figsize=(9, 5))
    x_l, y_l = line_xy()
    x_p, y_p = parabola_xy()
    x_c, y_c = cycloid_xy()

    ax.plot(x_l, y_l, "b--", lw=2, label="直线（直觉答案）")
    ax.plot(x_p, y_p, "g-.", lw=2, label="抛物线")
    ax.plot(x_c, y_c, "r-", lw=2.5, label="摆线（真正的最优解）")

    A_pt = (0.0, 0.0)
    B_pt = (X_END, Y_END)

    ax.scatter([A_pt[0], B_pt[0]], [A_pt[1], B_pt[1]],
               color="black", zorder=5, s=50)
    # 注：在 invert_yaxis 后，"y 偏移正"在视觉上是"向下"
    ax.annotate("起点 A（高处）", A_pt, xytext=(-15, -15),
                textcoords="offset points", fontsize=11,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                          edgecolor="gray", alpha=0.9))
    ax.annotate("终点 B（低处）", B_pt, xytext=(10, 12),
                textcoords="offset points", fontsize=11,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                          edgecolor="gray", alpha=0.9))

    # 摆线起点切线垂直（沿 +y，即向"低处"）
    ax.annotate(
        "切线垂直\n（先自由落体加速）",
        xy=(0.10, 0.20), xytext=(1.1, 1.1),
        arrowprops=dict(arrowstyle="->", color="red", lw=1),
        color="red", fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  edgecolor="red", alpha=0.9),
    )

    ax.set_xlabel("水平距离 $x$", fontsize=12)
    ax.set_ylabel("下落深度 $y$（向下为正，值越大越低）", fontsize=12)
    ax.set_title("三条候选路径：从 A 到 B 谁最快？", fontsize=14)
    ax.set_xlim(-0.3, X_END + 0.4)
    ax.set_ylim(-0.6, Y_END + 0.4)
    # 关键：翻转 y 轴，让 y=0 显示在图的顶部（高处），y=2 在底部（低处）
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left", fontsize=11)
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig("figure1.png", dpi=120)
    plt.close(fig)
    print("已生成 figure1.png")


# ============================================================
# Figure 2：轮子滚动画摆线（多帧动画的静态版本）
# ============================================================
def figure2():
    fig, ax = plt.subplots(figsize=(10, 5))

    # 几个不同角度的轮子
    angles = np.linspace(0, 2 * np.pi, 9)  # 9 帧
    theta_fine = np.linspace(0, 2 * np.pi, 400)
    x_full = A * (theta_fine - np.sin(theta_fine))
    y_full = A * (1 - np.cos(theta_fine))

    # 先画完整的摆线（浅灰背景）
    ax.plot(x_full, y_full, color="lightgray", lw=2, zorder=1)

    # 画每帧的圆和轮缘点
    for th in angles:
        cx = A * th            # 圆心水平位置
        cy = A                 # 圆心高度 = 半径
        # 圆
        circle = Circle((cx, cy), A, fill=False, ec="steelblue",
                        lw=1.2, alpha=0.5, zorder=2)
        ax.add_patch(circle)
        # 圆心
        ax.plot(cx, cy, "k.", ms=4, zorder=3)
        # 轮缘点
        px = A * (th - np.sin(th))
        py = A * (1 - np.cos(th))
        ax.plot(px, py, "ro", ms=6, zorder=4)
        # 连线：圆心到轮缘点
        ax.plot([cx, px], [cy, py], "r--", lw=0.8, alpha=0.5, zorder=2)

    # 画地面线
    ax.axhline(0, color="brown", lw=2, label="地面")

    ax.set_xlabel("水平位置 $x$", fontsize=12)
    ax.set_ylabel("高度 $y$", fontsize=12)
    ax.set_title("摆线的几何构造：轮子滚动，轮缘点描出轨迹", fontsize=14)
    ax.set_xlim(-0.5, 2 * np.pi * A + 0.5)
    ax.set_ylim(-0.3, 2 * A + 0.3)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=10)
    fig.tight_layout()
    fig.savefig("figure2.png", dpi=120)
    plt.close(fig)
    print("已生成 figure2.png")


# ============================================================
# Figure 3：多条候选曲线的总时间（柱状图）
# ============================================================
def figure3():
    fig, ax = plt.subplots(figsize=(9, 5))

    candidates = {
        "直线\n$y=\\frac{2}{\\pi}x$": line_xy(),
        "抛物线\n$y=\\frac{2}{\\pi^2}x^2$": parabola_xy(),
        "摆线（最优解）": cycloid_xy(),
    }

    names = list(candidates.keys())
    times = [travel_time(*candidates[k]) for k in names]

    # 理论最优 = 摆线时间，作为参考
    colors = ["steelblue", "seagreen", "crimson"]
    bars = ax.bar(names, times, color=colors, edgecolor="black")

    # 在柱顶标数值
    for bar, t in zip(bars, times):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"$T = {t:.4f}$",
            ha="center", va="bottom", fontsize=11,
        )

    # 摆线优势百分比
    ratio = (times[0] - times[2]) / times[0] * 100
    ax.text(
        0.5, 0.92,
        f"摆线比直线快 {ratio:.2f}%",
        transform=ax.transAxes,
        ha="center", fontsize=12,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow",
                  edgecolor="gray"),
    )

    ax.set_ylabel("总时间 $T$（无量纲）", fontsize=12)
    ax.set_title("三条路径的总滑行时间对比", fontsize=14)
    ax.set_ylim(0, max(times) * 1.2)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig("figure3.png", dpi=120)
    plt.close(fig)
    print("已生成 figure3.png")


# ============================================================
# Figure 4：摆线一拱的几何标注
# ============================================================
def figure4():
    fig, ax = plt.subplots(figsize=(10, 5))

    theta = np.linspace(0, 2 * np.pi, 500)
    x = A * (theta - np.sin(theta))
    y = A * (1 - np.cos(theta))

    ax.plot(x, y, "b-", lw=2.5, label="摆线一拱")

    # 标注关键点
    key_thetas = [
        (0,        "起点\n$\\theta=0$\n$(0, 0)$",         "left",   "bottom"),
        (np.pi / 2, "$\\theta=\\pi/2$",                  "left",   "bottom"),
        (np.pi,    "拱顶\n$\\theta=\\pi$\n$(\\pi, 2)$",   "center", "top"),
        (3 * np.pi / 2, "$\\theta=3\\pi/2$",             "right",  "top"),
        (2 * np.pi, "终点\n$\\theta=2\\pi$\n$(2\\pi, 0)$", "right", "top"),
    ]
    for th, label, ha, va in key_thetas:
        xt = A * (th - np.sin(th))
        yt = A * (1 - np.cos(th))
        ax.plot(xt, yt, "ro", ms=7)
        # 拱顶标注：放到曲线内部下方，远离标题和"半径"标注
        if th == np.pi:
            ax.annotate(
                label, (xt, yt), xytext=(-60, -45),
                textcoords="offset points",
                ha="center", va="top", fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow",
                          edgecolor="gray", alpha=0.9),
                arrowprops=dict(arrowstyle="-", color="gray", lw=0.5),
            )
        else:
            ax.annotate(
                label, (xt, yt), xytext=(15 if ha == "left" else -15,
                                         15 if va == "bottom" else -15),
                textcoords="offset points",
                ha=ha, va=va, fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow",
                          edgecolor="gray", alpha=0.9),
            )

    # 半径示意
    ax.annotate(
        "", xy=(np.pi / 2 - 1, 2), xytext=(np.pi / 2, 1),
        arrowprops=dict(arrowstyle="<->", color="purple", lw=1.5),
    )
    ax.text(np.pi / 2 - 0.7, 1.5, "半径 $a=1$", color="purple", fontsize=10)

    # 拱顶切线（水平）
    ax.plot([np.pi - 0.4, np.pi + 0.4], [2, 2], "g-", lw=2,
            label="拱顶切线（水平）")

    ax.set_xlabel("$x$", fontsize=12)
    ax.set_ylabel("$y$", fontsize=12)
    ax.set_title("摆线一拱的关键几何", fontsize=14)
    ax.set_xlim(-0.5, 2 * np.pi + 0.5)
    ax.set_ylim(-0.5, 2.5)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=10)
    fig.tight_layout()
    fig.savefig("figure4.png", dpi=120)
    plt.close(fig)
    print("已生成 figure4.png")


# ============================================================
# Figure 5：弧长可视化（堆叠弧长元素 ds）
# ============================================================
def figure5():
    fig, ax = plt.subplots(figsize=(11, 5))

    theta = np.linspace(0, 2 * np.pi, 500)
    x = A * (theta - np.sin(theta))
    y = A * (1 - np.cos(theta))

    # 主曲线
    ax.plot(x, y, "b-", lw=2.5, label="摆线一拱")

    # 弧长元素 ds = 2 sin(θ/2) dθ（在 a=1 下）
    # 在若干个 θ 上画切线方向的短线段
    sample_theta = np.linspace(0.05, 2 * np.pi - 0.05, 12)
    for th in sample_theta:
        xt = A * (th - np.sin(th))
        yt = A * (1 - np.cos(th))
        # 切向量（导数）
        dx = A * (1 - np.cos(th))
        dy = A * np.sin(th)
        norm = np.sqrt(dx ** 2 + dy ** 2)
        # 画一段小切线
        length = 0.4
        ax.plot(
            [xt, xt + length * dx / norm],
            [yt, yt + length * dy / norm],
            "r-", lw=2,
        )

    # 标注 ds = 2 sin(θ/2) dθ（放在曲线下方不重叠处）
    ax.text(
        0.97, 0.35,
        r"$ds = 2a\sin(\theta/2)\, d\theta$" + "\n" + r"$s = \int_0^{2\pi} ds = 8a$",
        transform=ax.transAxes, fontsize=12, ha="right",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow",
                  edgecolor="gray"),
    )

    ax.set_xlabel("$x$", fontsize=12)
    ax.set_ylabel("$y$", fontsize=12)
    ax.set_title("弧长元素 $ds$ 沿曲线累积 → 总弧长 $8a$",
                 fontsize=14)
    ax.set_xlim(-0.5, 2 * np.pi + 0.5)
    ax.set_ylim(-0.5, 2.5)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=10)
    fig.tight_layout()
    fig.savefig("figure5.png", dpi=120)
    plt.close(fig)
    print("已生成 figure5.png")


# ============================================================
# Figure 6：面积可视化（堆叠面积元素 y dx）
# ============================================================
def figure6():
    fig, ax = plt.subplots(figsize=(11, 5))

    theta = np.linspace(0, 2 * np.pi, 500)
    x = A * (theta - np.sin(theta))
    y = A * (1 - np.cos(theta))

    # 主曲线
    ax.plot(x, y, "b-", lw=2.5, label="摆线一拱")

    # 用半透明色填充曲线下方
    ax.fill_between(x, 0, y, color="lightblue", alpha=0.4,
                    label=r"$\int y\,dx = 3\pi a^2$")

    # 在若干 x 处画竖直窄条 y·dx
    sample_theta = np.linspace(0.1, 2 * np.pi - 0.1, 16)
    for i, th in enumerate(sample_theta):
        xt = A * (th - np.sin(th))
        yt = A * (1 - np.cos(th))
        width = 0.15
        rect = plt.Rectangle(
            (xt - width / 2, 0), width, yt,
            facecolor="coral", edgecolor="darkred", alpha=0.4,
        )
        ax.add_patch(rect)

    # 标注（放在曲线下方不重叠处）
    ax.text(
        0.97, 0.35,
        r"$A = \int_0^{2\pi a} y\,dx = 3\pi a^2$" + "\n" +
        r"（约为轮子面积 $\pi a^2$ 的 3 倍）",
        transform=ax.transAxes, fontsize=12, ha="right",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow",
                  edgecolor="gray"),
    )

    ax.set_xlabel("$x$", fontsize=12)
    ax.set_ylabel("$y$", fontsize=12)
    ax.set_title("面积元素 $y\\,dx$ 沿 x 累积 → 总面积 $3\\pi a^2$",
                 fontsize=14)
    ax.set_xlim(-0.5, 2 * np.pi + 0.5)
    ax.set_ylim(-0.5, 2.5)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=10)
    fig.tight_layout()
    fig.savefig("figure6.png", dpi=120)
    plt.close(fig)
    print("已生成 figure6.png")


# ============================================================
# 数值验证：弧长和面积
# ============================================================
def verify_numerics():
    """用数值积分验证 8a 和 3πa²。"""
    theta = np.linspace(0, 2 * np.pi, 20000)
    x = A * (theta - np.sin(theta))
    y = A * (1 - np.cos(theta))

    # 弧长 = ∫ sqrt(x'² + y'²) dθ
    dx = np.gradient(x, theta)
    dy = np.gradient(y, theta)
    arc_length = _trapz(np.sqrt(dx ** 2 + dy ** 2), theta)

    # 面积 = ∫ y dx
    area = _trapz(y, x)

    print("=" * 50)
    print(f"数值弧长  = {arc_length:.6f}    理论值 8a = {8 * A}")
    print(f"数值面积  = {area:.6f}    理论值 3πa² = {3 * np.pi * A ** 2:.6f}")
    print("=" * 50)


# ============================================================
# 主入口
# ============================================================
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法：python figures.py [1|2|3|4|5|6|all|verify]")
        sys.exit(0)

    arg = sys.argv[1]

    if arg == "1":
        figure1()
    elif arg == "2":
        figure2()
    elif arg == "3":
        figure3()
    elif arg == "4":
        figure4()
    elif arg == "5":
        figure5()
    elif arg == "6":
        figure6()
    elif arg == "verify":
        verify_numerics()
    elif arg == "all":
        figure1()
        figure2()
        figure3()
        figure4()
        figure5()
        figure6()
        verify_numerics()
    else:
        print(f"未知参数：{arg}")
        sys.exit(1)
