"""
9 张配图共用的样式常量与保存函数。
不生成图片，只被其他 gen_*.py 脚本 import。
"""

from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# ---------- CJK 字体配置 ----------
CJK_CANDIDATES = [
    "PingFang SC", "Heiti SC", "Hiragino Sans GB",
    "Songti SC", "STSong", "Arial Unicode MS",
    "Microsoft YaHei", "SimHei", "WenQuanYi Zen Hei",
    "Noto Sans CJK SC", "Source Han Sans SC",
]

available = {f.name for f in fm.fontManager.ttflist}
CHOSEN_FONT = None
for cand in CJK_CANDIDATES:
    if cand in available:
        CHOSEN_FONT = cand
        break
if CHOSEN_FONT:
    matplotlib.rcParams["font.sans-serif"] = [CHOSEN_FONT, "DejaVu Sans"]
else:
    matplotlib.rcParams["font.sans-serif"] = ["DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

# ---------- 数学公式渲染 (Computer Modern 字体，最接近 LaTeX) ----------
matplotlib.rcParams["mathtext.fontset"] = "cm"

# ---------- 统一字号 ----------
matplotlib.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "figure.dpi": 110,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
})

# ---------- 配色 ----------
COLOR_Q = "#E45756"   # 暖红 - Query
COLOR_K = "#4C78A8"   # 蓝 - Key
COLOR_V = "#54A24B"   # 绿 - Value
COLOR_HIGHLIGHT = "#F58518"  # 橙 - 高亮
COLOR_NEUTRAL = "#BAB0AC"    # 灰
COLOR_BG = "#F2F2F2"

# ---------- 路径 ----------
HERE = Path(__file__).resolve().parent
OUT_DIR = HERE.parent.parent / "images" / "002"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def save(fig, name):
    """保存 PNG 到 images/002/，并打印相对路径。"""
    out = OUT_DIR / name
    fig.savefig(out)
    print(f"  ✓ {out.relative_to(OUT_DIR.parent.parent)}")
    plt.close(fig)