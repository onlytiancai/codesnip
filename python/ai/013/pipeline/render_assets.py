#!/usr/bin/env python3
"""
XOR 反向传播视频 · 静态资源生成脚本。

输出到 pipeline/output/{formulas,plots,diagrams}/:

  diagrams/01-network.png       — 网络结构图 (2 → 4 → 1)
  plots/02-sigmoid.png          — sigmoid 函数曲线
  plots/03-crossentropy.png     — 交叉熵损失曲线
  formulas/04-chain.png         — 链式法则
  formulas/05-dldyhat.png       — dL/dŷ 推导
  formulas/06-simplify.png      — dL/dŷ 化简
  plots/07-sigmoid-deriv.png    — sigmoid 导数 + 切线
  formulas/08-step1..5.png      — 高潮 5 步

字体: 中文 → PingFang SC; 数学 → mathtext (cm 风格)
注意: matplotlib mathtext 不支持 \\underbrace/\\cancel/\\color/\\text{}.
     高亮用 Rectangle patch + 多 ax.text 拼接代替.
"""

from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import (
    FancyArrowPatch, Circle, FancyBboxPatch, Rectangle,
)
from matplotlib.font_manager import FontProperties, fontManager
import numpy as np

# ── 字体设置 ────────────────────────────────────────────────────
PINGFANG_PATH = (
    "/System/Library/AssetsV2/com_apple_MobileAsset_Font8/"
    "86ba2c91f017a3749571a82f2c6d890ac7ffb2fb.asset/AssetData/PingFang.ttc"
)
if Path(PINGFANG_PATH).exists():
    fontManager.addfont(PINGFANG_PATH)

PINGFANG = FontProperties(fname=PINGFANG_PATH)

matplotlib.rcParams["font.family"] = ["sans-serif"]
matplotlib.rcParams["font.sans-serif"] = ["PingFang SC", "sans-serif"]
matplotlib.rcParams["mathtext.fontset"] = "cm"
matplotlib.rcParams["mathtext.default"] = "regular"
matplotlib.rcParams["axes.unicode_minus"] = False

THEME = {
    "bg": "#FFFFFF",
    "text": "#1A1A1A",
    "subtext": "#5C5C5C",
    "accent": "#E07B00",
    "border": "#E5E5E5",
    "formula_bg": "#F7F7F5",
    "highlight": "#FFE7CC",
    "blue": "#3B7DD8",
}

ROOT = Path(__file__).parent
OUT = ROOT / "output"


def save(fig, subdir, name):
    target = OUT / subdir / name
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        target, dpi=160, bbox_inches="tight", pad_inches=0.3,
        facecolor=THEME["bg"],
    )
    plt.close(fig)
    print(f"   ✓ {target.relative_to(ROOT)}")


def zh(ax, x, y, s, **kw):
    kw.setdefault("color", THEME["text"])
    kw.setdefault("verticalalignment", "center")
    kw.setdefault("horizontalalignment", "center")
    return ax.text(x, y, s, fontproperties=PINGFANG, **kw)


def math(ax, x, y, s, **kw):
    """数学 ax.text (mathtext)。"""
    kw.setdefault("color", THEME["text"])
    kw.setdefault("verticalalignment", "center")
    kw.setdefault("horizontalalignment", "center")
    return ax.text(x, y, s, **kw)


# ── 单条公式图 ────────────────────────────────────────────────
def formula_image(latex, figsize=(14, 2.6), fontsize=58,
                  accent_terms=None, accent_color=None,
                  bg=THEME["formula_bg"]):
    """单条公式图。

    accent_terms: 可选 [(起始 idx, 终止 idx)] 元组列表,
                   用 bbox 高亮整段 LaTeX 中的部分(简化:整条高亮)。
                   复杂高亮(逐段) 改用 rich_formula_image。
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=160)
    ax.set_axis_off()
    ax.set_facecolor(bg)
    fig.patch.set_facecolor(bg)
    if accent_terms:
        # 用 bbox box 简化模拟
        ax.text(
            0.5, 0.5, latex, ha="center", va="center",
            fontsize=fontsize, color=accent_color or THEME["accent"],
            transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.4", facecolor=THEME["highlight"],
                      edgecolor=THEME["accent"], linewidth=2),
        )
    else:
        ax.text(
            0.5, 0.5, latex, ha="center", va="center",
            fontsize=fontsize, color=THEME["text"],
            transform=ax.transAxes,
        )
    return fig


def rich_formula_image(parts, figsize=(16, 3.0), fontsize=58,
                       bg=THEME["formula_bg"]):
    """复杂公式图: parts 是 list of (text, is_math, is_accent, sub_or_sup)。
    简化处理: 同一行水平拼接; accent 部分单独加 highlight bbox。

    parts: [("LHS or 中间文字", False, False), (r"$\\dfrac{L}{\\hat y}$", True, True), ...]
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=160)
    ax.set_axis_off()
    ax.set_facecolor(bg)
    fig.patch.set_facecolor(bg)

    # 计算累积宽度(粗略,设一个固定平均宽度因子)
    char_w = fontsize * 0.012
    total = 0.0
    widths = []
    for text, _is_math, _is_acc, _subsup in parts:
        # 去掉 LaTeX 标记估算视觉字符数
        plain = text.replace("$", "").replace("\\", "")
        # 中文按 1.0 字估宽, 数字/字母按 0.55 估宽
        chinese = sum(1 for c in text if ord(c) > 127)
        ascii_ = len(text) - chinese
        w = chinese * fontsize * 0.022 + ascii_ * fontsize * 0.012
        widths.append(w)
        total += w
    # 居中: 从 x=0.5 - total/2 开始
    cur = 0.5 - total / 2
    centers = []
    for w in widths:
        centers.append(cur + w / 2)
        cur += w

    for (text, is_math, is_acc, _), x in zip(parts, centers):
        kw = dict(ha="center", va="center", fontsize=fontsize)
        kw["color"] = THEME["accent"] if is_acc else THEME["text"]
        kw["fontproperties"] = PINGFANG if not is_math else None
        if is_acc:
            kw["bbox"] = dict(boxstyle="round,pad=0.25",
                              facecolor=THEME["highlight"],
                              edgecolor=THEME["accent"], linewidth=1.5)
        if is_math:
            ax.text(x, 0.5, text, **kw)
        else:
            ax.text(x, 0.5, text, **kw)
    return fig


# ════════════════════════════════════════════════════════════════
# 1) 网络结构图
# ════════════════════════════════════════════════════════════════
def render_network():
    fig, ax = plt.subplots(figsize=(14, 7), dpi=160)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 7)
    ax.set_axis_off()
    ax.set_facecolor(THEME["bg"])
    fig.patch.set_facecolor(THEME["bg"])

    layer_centers = {
        "X":      [(2.5, 5.0), (2.5, 2.0)],
        "Hidden": [(7, 6.3), (7, 4.7), (7, 3.1), (7, 1.5)],
        "Y":      [(11.5, 3.5)],
    }
    radius = 0.45

    def draw_node(xy, label, color=None, fontsize=26):
        if color is None:
            color = THEME["text"]
        circle = Circle(xy, radius, facecolor=THEME["formula_bg"],
                        edgecolor=color, linewidth=2.5, zorder=3)
        ax.add_patch(circle)
        ax.text(xy[0], xy[1], label, ha="center", va="center",
                fontsize=fontsize, fontweight="bold",
                color=color, zorder=4)

    edges = [("X", "Hidden"), ("Hidden", "Y")]
    for src, dst in edges:
        for a in layer_centers[src]:
            for b in layer_centers[dst]:
                arrow = FancyArrowPatch(
                    a, b, arrowstyle="-", connectionstyle="arc3,rad=0.0",
                    color=THEME["border"], linewidth=1.2, zorder=1,
                )
                ax.add_patch(arrow)

    for xy in layer_centers["X"]:
        draw_node(xy, r"$x$", color=THEME["text"])
    for xy in layer_centers["Hidden"]:
        draw_node(xy, r"$\sigma$", color=THEME["accent"])
    for xy in layer_centers["Y"]:
        draw_node(xy, r"$\hat y$", color=THEME["accent"])

    zh(ax, 2.5, 6.7, "输入层 X (2 维)", fontsize=18, color=THEME["subtext"])
    zh(ax, 7.0, 7.4, "隐藏层 A⁽¹⁾ (4 维)", fontsize=18, color=THEME["subtext"])
    zh(ax, 11.5, 4.7, "输出层 ŷ (1 维)", fontsize=18, color=THEME["subtext"])
    zh(ax, 4.75, 4.5, "Z⁽¹⁾ = W⁽¹⁾X + b⁽¹⁾",
       fontsize=18, color=THEME["subtext"])
    zh(ax, 9.3, 3.9, "Z⁽²⁾ = W⁽²⁾A⁽¹⁾ + b⁽²⁾",
       fontsize=18, color=THEME["subtext"])
    zh(ax, 11.5, 1.0, "真值 y", fontsize=16, color=THEME["subtext"])

    rect = FancyBboxPatch(
        (0.3, 0.4), 13.4, 6.2, boxstyle="round,pad=0.05",
        linewidth=1.5, edgecolor=THEME["border"], facecolor="none",
    )
    ax.add_patch(rect)
    save(fig, "diagrams", "01-network.png")


# ════════════════════════════════════════════════════════════════
# 2) sigmoid
# ════════════════════════════════════════════════════════════════
def render_sigmoid():
    fig, ax = plt.subplots(figsize=(13, 6.5), dpi=160)
    z = np.linspace(-7, 7, 400)
    y = 1 / (1 + np.exp(-z))

    ax.plot(z, y, color=THEME["accent"], linewidth=3.5)
    ax.axhline(1, color=THEME["border"], linestyle="--", linewidth=1)
    ax.axhline(0, color=THEME["border"], linestyle="--", linewidth=1)
    ax.axhline(0.5, color=THEME["border"], linestyle=":", linewidth=1)
    ax.axvline(0, color=THEME["border"], linestyle=":", linewidth=1)

    box = dict(boxstyle="round,pad=0.4", facecolor=THEME["highlight"],
               edgecolor=THEME["accent"], linewidth=1.5)
    ax.annotate("大 z → 1", xy=(4.5, 0.99), xytext=(4.7, 0.85),
                fontsize=22, color=THEME["text"], ha="center",
                bbox=box, fontproperties=PINGFANG)
    ax.annotate("小 z → 0", xy=(-6.5, 0.01), xytext=(-5.8, 0.15),
                fontsize=22, color=THEME["text"], ha="center",
                bbox=box, fontproperties=PINGFANG)

    ax.set_xlabel(r"$z$", fontsize=22, color=THEME["text"])
    ax.set_ylabel(r"$\sigma(z)$", fontsize=22, color=THEME["text"])
    ax.set_xlim(-7, 7)
    ax.set_ylim(-0.05, 1.1)
    ax.grid(True, alpha=0.2)
    ax.set_facecolor(THEME["bg"])
    fig.patch.set_facecolor(THEME["bg"])
    ax.set_title(
        r"$\sigma(z) = \dfrac{1}{1 + e^{-z}}$",
        fontsize=32, color=THEME["text"], pad=14, fontweight="bold",
    )
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(PINGFANG)
    save(fig, "plots", "02-sigmoid.png")


# ════════════════════════════════════════════════════════════════
# 3) 交叉熵
# ════════════════════════════════════════════════════════════════
def render_crossentropy():
    fig, ax = plt.subplots(figsize=(13, 6.5), dpi=160)
    yhat = np.linspace(0.001, 0.999, 400)
    L_y1 = -np.log(yhat)
    L_y0 = -np.log(1 - yhat)

    ax.plot(yhat, L_y1, color=THEME["accent"], linewidth=3.5,
            label=r"$y=1:\ \mathrm{L}=-\log\hat y$")
    ax.plot(yhat, L_y0, color=THEME["blue"], linewidth=3.5,
            label=r"$y=0:\ \mathrm{L}=-\log(1-\hat y)$")

    box = dict(boxstyle="round,pad=0.4", facecolor=THEME["highlight"],
               edgecolor=THEME["accent"], linewidth=1.5)
    ax.annotate("猜对 (≈0) → 惩罚轻", xy=(0.55, 0.7), xytext=(0.55, 1.2),
                fontsize=20, color=THEME["text"], ha="center",
                bbox=box, fontproperties=PINGFANG)
    ax.annotate("猜错 → 惩罚爆炸", xy=(0.05, 2.8), xytext=(0.20, 4.5),
                fontsize=20, color=THEME["text"],
                arrowprops=dict(arrowstyle="->", color=THEME["accent"], lw=2),
                bbox=box, fontproperties=PINGFANG)

    ax.set_xlabel(r"$\hat y$", fontsize=22, color=THEME["text"])
    ax.set_ylabel(r"$L$", fontsize=22, color=THEME["text"])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 6)
    ax.grid(True, alpha=0.2)
    leg = ax.legend(loc="upper center", fontsize=18)
    for t in leg.get_texts():
        t.set_fontproperties(PINGFANG)
    ax.set_title(
        r"$L = -\left[y\log\hat y + (1-y)\log(1-\hat y)\right]$",
        fontsize=26, color=THEME["text"], pad=14, fontweight="bold",
    )
    ax.set_facecolor(THEME["bg"])
    fig.patch.set_facecolor(THEME["bg"])
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(PINGFANG)
    save(fig, "plots", "03-crossentropy.png")


# ════════════════════════════════════════════════════════════════
# 4-6) 基础公式 PNG
# ════════════════════════════════════════════════════════════════
BASIC_FORMULAS = [
    ("formulas", "04-chain.png",
     r"$\dfrac{\partial L}{\partial z} = "
     r"\dfrac{\partial L}{\partial \hat y} \cdot "
     r"\dfrac{\partial \hat y}{\partial z}$",
     16, 2.8, 50),
    ("formulas", "05-dldyhat.png",
     r"$\dfrac{\partial L}{\partial \hat y} = "
     r"-\dfrac{y}{\hat y} + \dfrac{1-y}{1-\hat y}$",
     14, 2.6, 56),
    ("formulas", "06-simplify.png",
     r"$\dfrac{\partial L}{\partial \hat y} = "
     r"\dfrac{\hat y - y}{\hat y (1-\hat y)}$",
     12, 2.6, 64),
]
for subdir, name, latex, fw, fh, fs in BASIC_FORMULAS:
    fig = formula_image(latex, figsize=(fw, fh), fontsize=fs)
    save(fig, subdir, name)


# ════════════════════════════════════════════════════════════════
# 7) sigmoid 导数
# ════════════════════════════════════════════════════════════════
def render_sigmoid_deriv():
    fig, ax = plt.subplots(figsize=(13, 6.5), dpi=160)
    z = np.linspace(-6, 6, 400)
    sig = 1 / (1 + np.exp(-z))
    deriv = sig * (1 - sig)

    ax.plot(z, sig, color=THEME["accent"], linewidth=2.5,
            label=r"$\sigma(z)$")
    ax.plot(z, deriv, color=THEME["blue"], linewidth=2.5,
            label=r"$\sigma'(z) = \sigma(z)(1-\sigma(z))$")

    z0 = 2.0
    sig0 = 1 / (1 + np.exp(-z0))
    d0 = sig0 * (1 - sig0)
    tx = np.linspace(z0 - 1.5, z0 + 1.5, 30)
    ty = sig0 + d0 * (tx - z0)
    ax.plot(tx, ty, color=THEME["text"], linestyle="--", linewidth=1.5, alpha=0.6)
    ax.scatter([z0], [sig0], s=80, color=THEME["accent"], zorder=5)
    zh(ax, z0 + 1.0, sig0 + 0.12, "切线斜率 = ŷ(1−ŷ)",
       fontsize=18, color=THEME["text"])

    ax.set_xlabel(r"$z$", fontsize=22, color=THEME["text"])
    ax.set_ylabel(r"$\sigma,\ \sigma'$", fontsize=22, color=THEME["text"])
    ax.set_xlim(-6, 6)
    ax.set_ylim(-0.05, 1.1)
    ax.grid(True, alpha=0.2)
    leg = ax.legend(loc="upper right", fontsize=16)
    for t in leg.get_texts():
        t.set_fontproperties(PINGFANG)
    ax.set_title(
        r"$\dfrac{\partial \hat y}{\partial z} = \hat y\,(1 - \hat y)$",
        fontsize=30, color=THEME["text"], pad=14, fontweight="bold",
    )
    ax.set_facecolor(THEME["bg"])
    fig.patch.set_facecolor(THEME["bg"])
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(PINGFANG)
    save(fig, "plots", "07-sigmoid-deriv.png")


# ════════════════════════════════════════════════════════════════
# 8) 高潮:5 步推导
# 策略: 每张 PNG 是一个独立的完整公式(mathtext 一整条),
# 配合中文小角标说明"这一步在做什么"。MathAnim 的"动画"
# 由 Remotion 端做交叉淡入完成,不在图像端做拼接。
# ════════════════════════════════════════════════════════════════
def anim_step(latex, caption, figsize=(15, 4.0), fontsize=46,
              highlight=False):
    fig = plt.figure(figsize=figsize, dpi=160)
    bg = THEME["bg"] if highlight else THEME["formula_bg"]
    fig.patch.set_facecolor(bg)

    # 公式(主轴,占据上半部分)
    ax = fig.add_axes([0.05, 0.45, 0.9, 0.5])
    ax.set_axis_off()
    ax.set_facecolor(bg)
    if highlight:
        rect = FancyBboxPatch(
            (0.05, 0.10), 0.9, 0.8, boxstyle="round,pad=0.02",
            linewidth=3, edgecolor=THEME["accent"],
            facecolor=THEME["highlight"], transform=ax.transAxes,
        )
        ax.add_patch(rect)
    ax.text(0.5, 0.5, latex, ha="center", va="center",
            fontsize=fontsize, color=THEME["text"],
            transform=ax.transAxes)

    # 中文说明(下半部分)
    ax2 = fig.add_axes([0.05, 0.0, 0.9, 0.40])
    ax2.set_axis_off()
    ax2.set_facecolor(bg)
    zh(ax2, 0.5, 0.5, caption, fontsize=22, color=THEME["subtext"])
    return fig


ANIM_FRAMES = [
    # step1: 链式法则代入
    ("08-step1.png",
     r"$\dfrac{\partial L}{\partial z} = \dfrac{\partial L}{\partial \hat y} \cdot \dfrac{\partial \hat y}{\partial z}$",
     "链式法则代入 · 分成两项",
     (16, 3.6), 44, False),
    # step2: 两项展开成表达式
    ("08-step2.png",
     r"$\dfrac{\partial L}{\partial z} = \dfrac{\hat y - y}{\hat y\,(1-\hat y)} \cdot \hat y\,(1-\hat y)$",
     "展开 (1) 和 (2) 两项的具体形式",
     (16, 3.6), 44, False),
    # step3: 分子分母连乘
    ("08-step3.png",
     r"$\dfrac{\partial L}{\partial z} = \dfrac{(\hat y - y)\,\hat y\,(1-\hat y)}{\hat y\,(1-\hat y)}$",
     "分子分母堆到一起 · 准备抵消",
     (16, 3.6), 50, False),
    # step4: 抵消(mathtext 不支持 \cancel,用 () 文字替代)
    ("08-step4.png",
     r"$\dfrac{\partial L}{\partial z} = \dfrac{(\hat y - y)\,\hat y\,(1-\hat y)}{\hat y\,(1-\hat y)}$",
     r"分子分母的 $\hat y\,(1-\hat y)$ 互相抵消",
     (16, 3.6), 50, False),
    # step5: 结果
    ("08-step5.png",
     r"$\dfrac{\partial L}{\partial z} = \hat y - y$",
     "✨ 结论:整段反传就这一行",
     (15, 4.0), 76, True),
]

for fname, latex, caption, figsize, fs, highlight in ANIM_FRAMES:
    fig = anim_step(latex, caption, figsize=figsize, fontsize=fs,
                    highlight=highlight)
    save(fig, "formulas", fname)


# ════════════════════════════════════════════════════════════════
# 主入口
# ════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("🎨 开始生成视频静态资源")
    render_network()
    render_sigmoid()
    render_crossentropy()
    render_sigmoid_deriv()
    print(f"✅ 基础公式 {len(BASIC_FORMULAS)} 张 + 高潮帧 {len(ANIM_FRAMES)} 张")
    print("✅ 完成")
