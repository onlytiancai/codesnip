"""
matplotlib 中文字体配置（复用 000.md 的速查）
所有配图生成脚本都先 import 本模块

执行 python 脚本先 pyenv activate qlib 激活环境
"""
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 按优先级尝试可用 CJK 字体
CJK_CANDIDATES = [
    "PingFang SC", "Heiti SC", "Hiragino Sans GB",
    "Songti SC", "STSong", "Arial Unicode MS",
    "Microsoft YaHei", "SimHei", "WenQuanYi Zen Hei",
    "Noto Sans CJK SC", "Source Han Sans SC",  # Linux 常见
]

def configure_cjk():
    """配置 matplotlib 使用第一个可用的 CJK 字体"""
    available = {f.name for f in fm.fontManager.ttflist}
    chosen = None
    for cand in CJK_CANDIDATES:
        if cand in available:
            chosen = cand
            break
    if chosen:
        matplotlib.rcParams["font.sans-serif"] = [chosen, "DejaVu Sans"]
    else:
        # 最后兜底：用系统中任意一个 sans-serif 字体（可能显示成方块）
        matplotlib.rcParams["font.sans-serif"] = ["DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False
    return chosen


def new_figure(width=8, height=5, dpi=120, title=None):
    """快速建图工具"""
    fig, ax = plt.subplots(figsize=(width, height), dpi=dpi)
    if title:
        ax.set_title(title)
    ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)
    return fig, ax


def save(fig, path, transparent=False):
    """保存图片，自动创建目录"""
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=fig.dpi, bbox_inches="tight",
                transparent=transparent, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  ✓ {os.path.basename(path)}")


# 模块 import 时自动配置
CHOSEN_FONT = configure_cjk()

# LaTeX 风格数学公式（Computer Modern 字体集，最接近 LaTeX 默认）
# 比 matplotlib 默认的 DejaVu Sans 更好看，且不依赖 usetex（无需 texlive）
matplotlib.rcParams["mathtext.fontset"] = "cm"
