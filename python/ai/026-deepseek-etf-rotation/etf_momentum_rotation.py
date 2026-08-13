#!/usr/bin/env python3
"""ETF 动量轮动策略回测。

策略规则（月度调仓，21 个交易日）：
1. 每期计算 6 只股票 ETF 近 N 日动量（收盘价涨幅）；
2. 绝对动量过滤：动量 > 0 才可入选；
3. 相对动量排序：取前 2 名，各 50% 仓位；
4. 未填满的仓位槽由国债 ETF 补足（组合始终满仓）；
5. 信号在 T 日收盘后计算，仓位自 T+1 日起生效（无未来函数）。

运行：
    /Users/huhao/.pyenv/versions/3.11.9/bin/python3.11 etf_momentum_rotation.py [--window 20] [--save-csv]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.ticker import PercentFormatter

# ---------------------------------------------------------------- 1. rcParams
# 亮色图表规范（dataviz）：浅色表面 + 次级/柔和墨色 + 细网格
SURFACE = "#fcfcfb"
PRIMARY_INK = "#0b0b0b"
SECONDARY_INK = "#52514e"
MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"

# 中文按 PingFang SC → Hiragino Sans GB → Heiti TC 顺序回退；
# 先过滤出本机实际可用的字体，避免 matplotlib 对缺失字体刷屏警告
_FONT_PREFS = ["PingFang SC", "Hiragino Sans GB", "Heiti TC"]
_available = {f.name for f in font_manager.fontManager.ttflist}
_FONT_FAMILY = [f for f in _FONT_PREFS if f in _available] + ["sans-serif"]

plt.rcParams.update({
    "font.family": _FONT_FAMILY,
    "axes.unicode_minus": False,          # 否则负号显示为方块
    "mathtext.fontset": "cm",             # 数学公式用 CM 字体
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.grid": True, "grid.color": GRIDLINE, "grid.linewidth": 0.6,
    "axes.edgecolor": BASELINE, "axes.linewidth": 0.8,
    "axes.labelcolor": SECONDARY_INK, "axes.titlecolor": PRIMARY_INK,
    "text.color": PRIMARY_INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "font.size": 10, "figure.dpi": 150,
    "legend.frameon": False,
})

# ---------------------------------------------------------------- 2. 配置常量
HERE = Path(__file__).resolve().parent
KLINES_DIR = HERE / "klines"
OUTPUT_DIR = HERE / "output"

STOCK_ETFS = ["510050.SH", "510300.SH", "510500.SH", "159845.SZ", "159915.SZ", "588000.SH"]
TRESURY = "511010.SH"          # 国债 ETF：只做避险补位，不参与动量排名

ETF_NAMES = {
    "510050.SH": "上证50", "510300.SH": "沪深300", "510500.SH": "中证500",
    "159845.SZ": "中证1000", "159915.SZ": "创业板", "588000.SH": "科创50",
    "511010.SH": "国债",
}

WINDOW = 20                    # 动量窗口（交易日）
TOP_N = 2                      # 持有动量前 2 名
SLOT_WEIGHT = 0.5              # 每只 50% 仓位
REBALANCE_DAYS = 21            # 月度调仓（交易日计数）
COMMISSION_PER_SIDE = 1e-4     # 佣金单边万1（ETF 无印花税）
RISK_FREE_ANNUAL = 0.02        # 无风险利率，默认年化 2%
TRADING_DAYS = 252             # 年化用 252 个交易日（行业惯例，非本数据实际 242）

# —— v1 优化参数（引擎默认关 = 原版行为；main 默认按消融+网格择优）
# 消融结论：广度择时显著改善收益与回撤（网格单调：≤3@0.4 → 年化 15.82%/回撤
# -19.14%/夏普 0.66）；混合动量 20/60 为负优化（年化 7.64%），默认弃用。
MOM_BLEND = None               # 多窗口动量混合（None = 单窗口）
USE_SCALE = True               # 广度择时 scale 叠加
BREADTH_DAYS = 60              # 广度 = 站上 N 日均线的宽基数量
BREADTH_TRIGGER = 3            # 广度 ≤ 3（即 3+ 只跌破 MA60）触发防御
SCALE_DEFENSIVE = 0.4          # 防御时权益仓位比例

# ETF 固定色（dataviz 校验通过的 categorical 色板，跨图一致——颜色跟实体走）
ETF_COLORS = {
    "510050.SH": "#2a78d6", "510300.SH": "#eb6834", "510500.SH": "#1baf7a",
    "159845.SZ": "#eda100", "159915.SZ": "#e87ba4", "588000.SH": "#4a3aa7",
    "511010.SH": "#008300",
}
# 敏感性柱状图：单色蓝序数梯度（窗口小→大），色板 step 250/300/400/500/600
RAMP_ORDINAL = ["#86b6ef", "#6da7ec", "#3987e5", "#256abf", "#184f95"]
SCAN_WINDOWS = [5, 10, 20, 30, 60]

# ---------------------------------------------------------------- 3. 数据加载
def load_closes(klines_dir: Path) -> pd.DataFrame:
    """读取全部 CSV 的收盘价，按日期对齐成一张表（columns = ETF 代码）。"""
    closes = {}
    for f in sorted(klines_dir.glob("*.csv")):
        df = pd.read_csv(f, parse_dates=["date"], index_col="date")
        closes[f.stem] = df["close"]
    out = pd.DataFrame(closes).sort_index()   # 按索引自动对齐（inner join 语义）
    assert out.isna().to_numpy().sum() == 0, "日期未对齐或数据缺失"
    assert len(out) == 726, f"行数 {len(out)} 不符预期 726"
    return out

# ---------------------------------------------------------------- 4. 信号
def compute_target_weights(mom: pd.DataFrame, top_n: int, slot_weight: float) -> pd.DataFrame:
    """绝对动量过滤 + 相对动量排序，产出目标权重（仅股票池竞争槽位）。

    动量 ≤ 0 → NaN → rank 得 NaN → (NaN <= top_n) 为 False → 0。
    因此 0 只入选→全 0（国债 100%）；1 只入选→0.5；2 只入选→各 0.5，无需 if 分支。
    """
    eligible = mom.where(mom > 0)
    rank = eligible.rank(axis=1, ascending=False, method="first")   # method="first" 保确定性
    return (rank <= top_n).astype(float) * slot_weight

def rebalance_weights(target: pd.DataFrame, all_dates: pd.DatetimeIndex,
                      rebalance_days: int) -> pd.DataFrame:
    """信号 T 日收盘定权重，T+1 日起生效。

    顺序必须是 reindex → ffill → shift(1)：
    - ffill 后信号日当天拿到新值；再 shift(1) 把它推迟一天生效，信号日当天仍持旧仓。
    - 首行 shift 成 NaN → fillna(0) → 热身期（首个信号日前）全仓国债，属预期行为。
    """
    signal_dates = all_dates[::rebalance_days]
    held = target.loc[signal_dates]
    return held.reindex(all_dates).ffill().shift(1).fillna(0.0)

def breadth_scale(closes: pd.DataFrame, *, breadth_days: int = BREADTH_DAYS,
                  breadth_trigger: int = BREADTH_TRIGGER,
                  scale_defensive: float = SCALE_DEFENSIVE) -> pd.Series:
    """广度择时 scale：站上 MA 的宽基数量 ≤ 阈值时收缩权益仓位。

    T 日收盘可算 → shift(1) 从 T+1 起生效（无未来函数）。热身期 MA 为 NaN，
    比较得 False → scale 1.0，无害（此时本就全仓国债）。
    """
    ma = closes[STOCK_ETFS].rolling(breadth_days).mean()
    breadth = (closes[STOCK_ETFS] > ma).sum(axis=1)
    sc = pd.Series(np.where(breadth <= breadth_trigger, scale_defensive, 1.0),
                   index=closes.index)
    return sc.shift(1).fillna(1.0)

# ---------------------------------------------------------------- 5. 回测引擎
def backtest(closes: pd.DataFrame, window: int, *,
             rebalance_days: int = REBALANCE_DAYS,
             top_n: int = TOP_N,
             slot_weight: float = SLOT_WEIGHT,
             commission: float = COMMISSION_PER_SIDE,
             use_scale: bool = False,
             mom_blend: tuple[int, ...] | None = None,
             breadth_days: int = BREADTH_DAYS,
             breadth_trigger: int = BREADTH_TRIGGER,
             scale_defensive: float = SCALE_DEFENSIVE) -> tuple[pd.Series, pd.DataFrame, pd.Series]:
    """回测主流程，返回 (净值, 每日持仓权重, 每日换手 Σ|Δw|)。

    引擎参数默认 = 原版行为（use_scale=False, mom_blend=None）；
    优化开关由调用方（main 的消融结果）决定，保证 v2 等 import 方不受影响。
    """
    if mom_blend:   # 多窗口动量混合：缺失窗口按 0 贡献（不改变符号，只轻度压低幅度）
        mom = None
        for nb in mom_blend:
            pc = closes[STOCK_ETFS].pct_change(nb, fill_method=None)  # 显式 fill_method=None，pandas 3.0 兼容
            mom = pc if mom is None else mom.add(pc, fill_value=0.0)
        mom = mom / len(mom_blend)
    else:
        mom = closes[STOCK_ETFS].pct_change(window, fill_method=None)
    target = compute_target_weights(mom, top_n, slot_weight)
    port = rebalance_weights(target, closes.index, rebalance_days)
    port[TRESURY] = 1.0 - port[STOCK_ETFS].sum(axis=1)   # 国债补足，整列赋值 CoW 安全
    if use_scale:   # 广度择时：权益仓位 × scale，国债补足
        sc = breadth_scale(closes, breadth_days=breadth_days,
                           breadth_trigger=breadth_trigger, scale_defensive=scale_defensive)
        port[STOCK_ETFS] = port[STOCK_ETFS].mul(sc, axis=0)
        port[TRESURY] = 1.0 - port[STOCK_ETFS].sum(axis=1)

    daily_ret = closes.pct_change(fill_method=None)
    gross_ret = (port * daily_ret).sum(axis=1)
    turnover = port.diff().abs().sum(axis=1).fillna(0.0)
    # 成本 = Σ|Δw| × 单边佣金率。Σ|Δw| 已含买卖两腿（卖 0.5 + 买 0.5 = 1.0），
    # 等于 2×单边成交额×费率，故不再乘 2。成本只在权重生效日扣一次。
    cost = turnover * commission
    net_ret = (gross_ret - cost).fillna(0.0)
    nav = (1.0 + net_ret).cumprod()

    # 结构自检
    assert np.allclose(port.sum(axis=1), 1.0, atol=1e-9), "权重行和必须恒为 1"
    assert np.isclose(nav.iloc[0], 1.0)
    assert ((cost > 0) == (turnover > 0)).all(), "成本只应出现在换仓日"
    assert port[STOCK_ETFS].iloc[:rebalance_days + 1].to_numpy().sum() == 0.0, \
        "热身期（首个信号生效前）应全仓国债"
    return nav, port, turnover

# ---------------------------------------------------------------- 6. 指标与基准
def compute_metrics(nav: pd.Series, rf_annual: float) -> dict:
    """从净值序列计算绩效指标。年化按 252 个交易日。"""
    n = len(nav)
    total = nav.iloc[-1] / nav.iloc[0] - 1
    years = (n - 1) / TRADING_DAYS
    ann_ret = (nav.iloc[-1] / nav.iloc[0]) ** (1 / years) - 1
    r = nav.pct_change(fill_method=None).dropna()
    ann_vol = r.std() * np.sqrt(TRADING_DAYS)
    rf_daily = (1 + rf_annual) ** (1 / TRADING_DAYS) - 1   # 复利折算，不用 rf/252
    sharpe = (r.mean() - rf_daily) / r.std() * np.sqrt(TRADING_DAYS)
    dd = nav / nav.cummax() - 1
    max_dd = dd.min()
    calmar = ann_ret / abs(max_dd)
    return {"总收益": total, "年化收益": ann_ret, "年化波动": ann_vol,
            "最大回撤": max_dd, "夏普": sharpe, "卡玛": calmar}

def benchmark_navs(closes: pd.DataFrame) -> dict[str, pd.Series]:
    """对比基准（均不计成本，策略已扣费，对比偏保守）。

    等权买入持有用 (close/close0).mean(axis=1) —— 权重随涨跌漂移，不每日再平衡。
    """
    stocks = closes[STOCK_ETFS]
    eq = (stocks / stocks.iloc[0]).mean(axis=1)
    hs300 = closes["510300.SH"] / closes["510300.SH"].iloc[0]
    bond = closes[TRESURY] / closes[TRESURY].iloc[0]
    return {"等权持有": eq, "沪深300": hs300, "国债ETF": bond}

def trade_log(port: pd.DataFrame, rebalance_days: int) -> list[tuple[pd.Timestamp, list[tuple[str, float]]]]:
    """调仓记录：每次生效权重与上一期不同则记一条（生效日 = 信号日次日）。"""
    logs: list[tuple[pd.Timestamp, list[tuple[str, float]]]] = []
    prev: tuple[tuple[str, float], ...] | None = None
    for p in port.index[::rebalance_days]:
        i = port.index.get_loc(p)
        w = port.iloc[min(i + 1, len(port) - 1)]
        held = tuple((c, w[c]) for c in port.columns if w[c] > 1e-9)
        if held != prev:
            logs.append((port.index[min(i + 1, len(port) - 1)], list(held)))
        prev = held
    return logs

# ---------------------------------------------------------------- 7. 敏感性
def sensitivity_scan(closes: pd.DataFrame, **engine_kw) -> list[dict]:
    """扫动量窗口 N ∈ {5,10,20,30,60}（单窗口，可叠加 scale 等引擎参数），返回每组的指标。"""
    rows = []
    for n in SCAN_WINDOWS:
        nav, port, turnover = backtest(closes, n, **engine_kw)
        m = compute_metrics(nav, RISK_FREE_ANNUAL)
        years = (len(nav) - 1) / TRADING_DAYS
        m["N"] = n
        m["年换手率"] = turnover.sum() / 2 / years   # 单边口径
        m["调仓次数"] = int((turnover > 1e-12).sum())
        rows.append(m)
    return rows

# ---------------------------------------------------------------- 8. 绘图
def plot_main(closes: pd.DataFrame, nav: pd.Series, port: pd.DataFrame,
              bench: dict[str, pd.Series], window: int, outpath: Path,
              scale: pd.Series | None = None) -> None:
    dates = closes.index
    fig, axes = plt.subplots(
        3, 1, figsize=(12, 10), sharex=True,
        gridspec_kw={"height_ratios": [3, 2, 2.5]})
    ax1, ax2, ax3 = axes

    # ---- 面板 1：净值对比（对数刻度）
    ax1.plot(dates, nav, color="#2a78d6", lw=2.5, label="轮动策略")
    ax1.plot(dates, bench["等权持有"], color="#eb6834", lw=1.5, label="等权持有")
    ax1.plot(dates, bench["国债ETF"], color="#008300", lw=1.5, label="国债ETF")
    ax1.plot(dates, bench["沪深300"], color="#1baf7a", lw=1.5, label="沪深300")
    ax1.set_yscale("log")
    ax1.set_title(f"ETF 动量轮动 vs 基准（动量窗口 N={window}，月度调仓，持有前 2）")
    ax1.set_ylabel("净值（对数刻度）")
    # 4 条线：图例 + 末端直接标注（颜色不单独承载身份）；图例横排置于面板上方，不遮数据
    ax1.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncols=4, fontsize=9)
    right_pad = dates[-1] + pd.Timedelta(days=int((dates[-1] - dates[0]).days * 0.20))
    ax1.set_xlim(dates[0], right_pad)
    ax1.margins(y=0.08)
    series = [(nav, "轮动策略"), (bench["等权持有"], "等权持有"),
              (bench["沪深300"], "沪深300"), (bench["国债ETF"], "国债ETF")]
    # 在 log 空间贪心推开标签，保证相邻标注间距 ≥ min_gap（对数单位）
    logv = np.log([s.iloc[-1] for s, _ in series])
    order = np.argsort(logv)
    for idx in range(1, len(order)):
        lo, hi = order[idx - 1], order[idx]
        if logv[hi] - logv[lo] < 0.07:
            logv[hi] = logv[lo] + 0.07
    for (s, name), y in zip(series, np.exp(logv)):
        ax1.text(1.02, y, f"{name}  {s.iloc[-1]:.2f}",
                 transform=ax1.get_yaxis_transform(), ha="left", va="center",
                 fontsize=9, color=SECONDARY_INK)

    # ---- 面板 2：回撤
    dd_strat = nav / nav.cummax() - 1
    dd_hs = bench["沪深300"] / bench["沪深300"].cummax() - 1
    ax2.fill_between(dates, 0, dd_strat, color="#2a78d6", alpha=0.30, lw=0)
    ax2.plot(dates, dd_strat, color="#2a78d6", lw=1.5, label="轮动策略")
    ax2.plot(dates, dd_hs, color="#1baf7a", lw=1.2, ls="--", label="沪深300")
    ax2.set_ylabel("回撤")
    ax2.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax2.legend(loc="lower right", fontsize=9)
    ax2.set_ylim(top=0.0)

    # ---- 面板 3：持仓历史（堆叠面积）
    cols = STOCK_ETFS + [TRESURY]
    x = mdates.date2num(dates.to_pydatetime())
    ax3.stackplot(x, port[cols].to_numpy().T, colors=[ETF_COLORS[c] for c in cols],
                  linewidth=0.5, edgecolor=SURFACE, labels=[ETF_NAMES[c] for c in cols])
    if scale is not None:
        # 白色衬底线 + 深色虚线：保证在任何色块上都清晰可辨
        ax3.plot(x, scale.to_numpy(), color=SURFACE, lw=3.4, zorder=3)
        ax3.plot(x, scale.to_numpy(), color=PRIMARY_INK, ls="--", lw=1.6, zorder=4,
                 label="scale（权益仓位）")
    ax3.set_ylabel("持仓权重")
    ax3.set_yticks([0, 0.5, 1.0])
    ax3.set_ylim(0, 1)
    ax3.axhline(0.5, color=GRIDLINE, ls="--", lw=0.8)
    ax3.legend(ncols=4, loc="upper center", bbox_to_anchor=(0.5, -0.18), fontsize=9)

    ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)

def plot_sensitivity(rows: list[dict], default_n: int, outpath: Path) -> None:
    """动量窗口敏感性：年化收益与夏普双柱状图，序数蓝梯度 + 默认值标注。"""
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    ns = [r["N"] for r in rows]
    for ax, key, title in ((axes[0], "年化收益", "年化收益"),
                           (axes[1], "夏普", "夏普比率")):
        vals = [r[key] for r in rows]
        bars = ax.bar(range(len(ns)), vals, width=0.62,
                      color=RAMP_ORDINAL[:len(ns)])
        for i, (b, v) in enumerate(zip(bars, vals)):
            txt = f"{v:.2f}" if key == "夏普" else f"{v:.1%}"
            is_def = ns[i] == default_n
            ax.text(b.get_x() + b.get_width() / 2, v, txt,
                    ha="center", va="bottom", fontsize=8.5,
                    color=PRIMARY_INK if is_def else SECONDARY_INK,
                    fontweight="bold" if is_def else "normal")
        if default_n in ns:
            i = ns.index(default_n)
            bars[i].set_edgecolor("#0d366b")
            bars[i].set_linewidth(2)
            # "默认"标注放在数值标签上方，用 offset points 分开，避免挤压
            ax.annotate("默认", (i, vals[i]), xytext=(0, 16),
                        textcoords="offset points", ha="center", va="bottom",
                        fontsize=8.5, color=PRIMARY_INK, fontweight="bold")
        ax.margins(y=0.16)   # 给柱顶标签留出头空间
        ax.set_title(title, fontsize=11)
        ax.set_xticks(range(len(ns)))
        ax.set_xticklabels([f"N={n}" for n in ns])
        if key == "年化收益":
            ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.suptitle(f"参数敏感性（动量窗口 N）", fontsize=12, y=1.04)
    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)

# ---------------------------------------------------------------- 9. main
def _pad_cjk(s: str, width: int) -> str:
    """按显示宽度右对齐：CJK 字符按 2 列计。"""
    disp = sum(2 if ord(ch) > 0x2E7F else 1 for ch in s)
    return s + " " * max(0, width - disp)

def print_comparison(strat: dict, extra: dict, bench_m: dict[str, dict]) -> None:
    rows = [("总收益", "总收益"), ("年化收益", "年化收益"), ("年化波动", "年化波动"),
            ("最大回撤", "最大回撤"), ("夏普", "夏普"), ("卡玛", "卡玛")]
    names = ["轮动策略", "等权持有", "沪深300", "国债ETF"]
    print(f"\n===== 绩效对比（2023-08-14 ~ 2026-08-12，策略已扣佣金万1，基准不计成本）=====")
    print(_pad_cjk("指标", 8) + "  " + "  ".join(_pad_cjk(h, 8) for h in names))
    for label, key in rows:
        vals = [strat[key]] + [bench_m[n][key] for n in names[1:]]
        cells = [f"{v:>8.2%}" if key != "夏普" else f"{v:>8.2f}" for v in vals]
        print(_pad_cjk(label, 8) + "  " + "  ".join(cells))
    print(f"\n策略附加：年度单边换手率 {extra['年换手率']:.0%} ｜ 调仓次数 {extra['调仓次数']} 次"
          f" ｜ 国债避险天数占比 {extra['国债占比']:.0%}")

def print_sensitivity(rows: list[dict], default_n: int) -> None:
    headers = ["年化收益", "年化波动", "最大回撤", "夏普", "卡玛", "年换手率", "调仓次数"]
    widths = [8, 8, 8, 6, 6, 8, 8]
    print(f"\n===== 参数敏感性（动量窗口 N）=====")
    print("N=      " + "  ".join(_pad_cjk(h, w) for h, w in zip(headers, widths)))
    for r in rows:
        vals = [f"{r['年化收益']:>8.2%}", f"{r['年化波动']:>8.2%}", f"{r['最大回撤']:>8.2%}",
                f"{r['夏普']:>6.2f}", f"{r['卡玛']:>6.2f}", f"{r['年换手率']:>8.0%}", f"{r['调仓次数']:>8d}"]
        mark = "*" if r["N"] == default_n else " "
        print(f"N={r['N']:<4}{mark} " + "  ".join(vals))
    print("（* 为默认参数）")

def main() -> None:
    ap = argparse.ArgumentParser(description="ETF 动量轮动回测")
    ap.add_argument("--window", type=int, default=WINDOW, help="动量窗口（交易日），默认 20")
    ap.add_argument("--no-scale", action="store_true", help="关闭广度择时 scale")
    ap.add_argument("--no-blend", action="store_true", help="关闭多窗口动量混合")
    ap.add_argument("--save-csv", action="store_true", help="保存净值 CSV 到 output/")
    ap.add_argument("--outdir", type=Path, default=OUTPUT_DIR, help="输出目录")
    args = ap.parse_args()

    use_scale = USE_SCALE and not args.no_scale
    mom_blend = None if args.no_blend else MOM_BLEND

    closes = load_closes(KLINES_DIR)
    print(f"数据：7 只 ETF，{len(closes)} 个交易日（{closes.index[0]:%Y-%m-%d} ~ "
          f"{closes.index[-1]:%Y-%m-%d}）")
    print(f"策略：动量窗口 N={args.window}"
          + (f" 混合{mom_blend}" if mom_blend else "")
          + (f" ｜ 广度择时（≤{BREADTH_TRIGGER}只站上MA{BREADTH_DAYS}→权益{SCALE_DEFENSIVE:.0%}）"
             if use_scale else "")
          + f" ｜ 每 {REBALANCE_DAYS} 个交易日调仓 ｜ 动量>0 入选前 {TOP_N} 各 50%"
          + f" ｜ 国债补足避险 ｜ 佣金单边万1")

    # 消融 A/B：引擎参数对比（同成本口径）
    print(f"\n===== v1 优化消融（引擎参数对比，同成本万1）=====")
    combos = [
        ("原版 N=20", dict(window=20)),
        ("文档值≤2@0.4", dict(window=20, use_scale=True, breadth_trigger=2, scale_defensive=0.4)),
        ("+混合动量20/60", dict(window=20, mom_blend=(20, 60))),
        ("默认≤3@0.4", dict(window=20, use_scale=True, breadth_trigger=3, scale_defensive=0.4)),
        ("更激进≤4@0.4", dict(window=20, use_scale=True, breadth_trigger=4, scale_defensive=0.4)),
    ]
    print(_pad_cjk("变体", 16) + _pad_cjk("年化收益", 9) + _pad_cjk("最大回撤", 9)
          + _pad_cjk("夏普", 7) + _pad_cjk("卡玛", 8) + _pad_cjk("换手/年", 9)
          + _pad_cjk("调仓", 5))
    for name, kw in combos:
        nav_a, _, to_a = backtest(closes, **kw)
        m_a = compute_metrics(nav_a, RISK_FREE_ANNUAL)
        years_a = (len(nav_a) - 1) / TRADING_DAYS
        to_rate = to_a.sum() / 2 / years_a
        print(f"{_pad_cjk(name, 16)}{m_a['年化收益']:>9.2%}  {m_a['最大回撤']:>9.2%}"
              f"  {m_a['夏普']:>7.2f}  {m_a['卡玛']:>8.2%}  {to_rate:>9.0%}"
              f"  {int((to_a > 1e-12).sum()):>5d}")

    nav, port, turnover = backtest(closes, args.window,
                                   use_scale=use_scale, mom_blend=mom_blend)
    strat = compute_metrics(nav, RISK_FREE_ANNUAL)
    years = (len(nav) - 1) / TRADING_DAYS
    extra = {"年换手率": turnover.sum() / 2 / years,
             "调仓次数": int((turnover > 1e-12).sum()),
             "国债占比": float((port[TRESURY] > 0.99).mean())}
    bench = benchmark_navs(closes)
    bench_m = {name: compute_metrics(s, RISK_FREE_ANNUAL) for name, s in bench.items()}
    print_comparison(strat, extra, bench_m)

    print(f"\n===== 调仓记录（生效日）=====")
    for d, held in trade_log(port, REBALANCE_DAYS):
        parts = [f"{ETF_NAMES[c]} {w:.0%}" for c, w in held]
        print(f"{d:%Y-%m-%d}  {' + '.join(parts)}")

    scan = sensitivity_scan(closes, use_scale=use_scale)
    print_sensitivity(scan, args.window)

    args.outdir.mkdir(exist_ok=True)
    p1 = args.outdir / f"backtest_nav_N{args.window}.png"
    p2 = args.outdir / "sensitivity.png"
    sc = breadth_scale(closes) if use_scale else None
    plot_main(closes, nav, port, bench, args.window, p1, scale=sc)
    plot_sensitivity(scan, args.window, p2)
    print(f"\n图表已保存：{p1}  {p2}")

    if args.save_csv:
        out = pd.DataFrame({"轮动策略": nav, **{n: s for n, s in bench.items()}})
        pcsv = args.outdir / f"nav_N{args.window}.csv"
        out.to_csv(pcsv, index_label="date")
        print(f"净值已保存：{pcsv}")

if __name__ == "__main__":
    main()
