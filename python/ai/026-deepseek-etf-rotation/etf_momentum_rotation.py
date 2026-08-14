#!/usr/bin/env python3
"""ETF 动量轮动策略回测。

策略规则（月度调仓，21 个交易日）：
1. 每期计算 6 只股票 ETF 近 N 日动量（收盘价涨幅）；
2. 绝对动量过滤：动量 > 0 才可入选；
3. 相对动量排序：取前 2 名，各 50% 仓位；
4. 未填满的仓位槽由国债 ETF 补足（组合始终满仓）；
5. 信号在 T 日收盘后计算，仓位自 T+1 日起生效（无未来函数）。

运行：
    /Users/huhao/.pyenv/versions/3.11.9/bin/python3.11 etf_momentum_rotation.py [--window 10] [--save-csv]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
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

# 动量窗口：10。N 的选择对样本敏感（3 年样本最优 20、10 年样本最优 10），
# N=10 在两个样本均排前二（3年 13.94% / 10年 9.09%），取更稳健的短窗口。
WINDOW = 10                    # 动量窗口（交易日）
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
# 滞回带宽 2：10 年网格显示 h=2 换手从 2156% 降到 799%（141 次调仓）且收益
# 微升、夏普最优（≤3@0.4 h=2：9.49% / -23.09% / 0.48），防阈值附近反复横跳。
BREADTH_HYST = 2               # 广度滞回带宽（≤trigger−h 入防御、≥trigger+h 恢复）
# 指数触发（真指数 000300.SH）：60 日年化波动>25% 或收盘<MA120 → 防御。
# 20 年回测：回撤 -41.3%→-26.5%（2015 股灾的快信号短板），收益不变，夏普
# 0.50→0.58；9 窗滚出对照 4 胜 3 负 2 平、超额总和打平 → 纯风险端改善。
USE_INDEX_TRIGGERS = True      # 指数快/慢触发叠加广度择时
INDEX_VOL_TRIGGER = 0.25       # 60 日年化波动触发线
INDEX_MA_DAYS = 120            # 指数收盘跌破该日均线 → 防御

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
    """读取全部 CSV 的收盘价，union 索引对齐成一张表（columns = ETF 代码）。

    NaN 语义 = 未上市或停牌日（159845 上市 2021-03、588000 上市 2020-11、
    159915 停牌 2021-02-08）。下游按此语义处理：动量 NaN → 自动排除出轮动池；
    日收益 fillna(0)（未上市/停牌无收益）；广度统计不计未上市 ETF。
    """
    closes = {}
    for f in sorted(klines_dir.glob("*.csv")):
        df = pd.read_csv(f, parse_dates=["date"], index_col="date")
        closes[f.stem] = df["close"]
    out = pd.DataFrame(closes).sort_index()   # union 索引，缺日 NaN
    assert out.index.is_monotonic_increasing
    assert len(out) > 2000, f"行数 {len(out)} 异常（预期 ≥2426）"
    nan_share = out.isna().to_numpy().sum() / out.size
    # 20 年数据（2006 起）中 588000/159845 等上市晚，NaN 占比 ~33%；留余量放宽到 42%
    assert nan_share < 0.42, f"NaN 占比 {nan_share:.0%} 异常（预期仅未上市/停牌缺口）"
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
                  scale_defensive: float = SCALE_DEFENSIVE,
                  normalize: bool = True,
                  hyst: int = 0,
                  use_index_triggers: bool = False,
                  index_vol_ann: float = 0.25,
                  index_ma_days: int = 120,
                  index_col: str = "000300.SH") -> pd.Series:
    """广度择时 scale：站上 MA 的宽基数量 ≤ 阈值时收缩权益仓位。

    - T 日收盘可算 → shift(1) 从 T+1 起生效（无未来函数）。
    - 未上市 ETF 的 close/MA 为 NaN，比较得 False → 既不算"站上"，也不计入分母。
    - normalize=True（默认）：阈值随已上市数量归一为"站上数量 ≤ 一半"
      （全池 6 只时 = 3，与调参值一致；早期 4~5 只时 = 2）。
    - hyst：滞回带宽。广度 ≤ trigger−hyst 进入防御、≥ trigger+hyst 恢复满仓，
      之间保持原状态（防阈值附近反复横跳）；hyst=0 即原行为。
      状态机按日循环（状态依赖历史，无法纯向量化；2426 行无性能问题）。
    """
    ma = closes[STOCK_ETFS].rolling(breadth_days).mean()
    breadth = (closes[STOCK_ETFS] > ma).sum(axis=1)   # NaN → False，未上市不计
    if normalize:
        n_valid = ma.notna().sum(axis=1)              # 上市且满窗口的 ETF 数
        trigger = (n_valid // 2).clip(lower=2, upper=breadth_trigger)
    else:
        trigger = pd.Series(breadth_trigger, index=closes.index)
    low = trigger - hyst
    high = trigger + hyst
    # 指数触发（可选，用真指数）：60 日年化波动 > 阈值 或 收盘 < MA_N → 防御。
    # 快信号（波动率）弥补广度对急跌反应慢的短板（2015 股灾的教训），无滞回。
    if use_index_triggers and index_col in closes.columns:
        idx_ret = closes[index_col].pct_change(fill_method=None)
        vol_ann = idx_ret.rolling(60).std() * np.sqrt(TRADING_DAYS)
        ma_idx = closes[index_col].rolling(index_ma_days).mean()
        idx_def = ((vol_ann > index_vol_ann) | (closes[index_col] < ma_idx)).to_numpy()
    else:
        idx_def = np.zeros(len(breadth), dtype=bool)
    states = np.empty(len(breadth), dtype=bool)
    defensive = False
    for i, b in enumerate(breadth.to_numpy()):
        if not np.isnan(b):
            if defensive:
                if b >= high.iloc[i]:
                    defensive = False
            elif b <= low.iloc[i]:
                defensive = True
        states[i] = defensive or idx_def[i]
    sc = np.where(states, scale_defensive, 1.0)
    return pd.Series(sc, index=closes.index).shift(1).fillna(1.0)

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
             scale_defensive: float = SCALE_DEFENSIVE,
             normalize_trigger: bool = True,
             breadth_hyst: int = BREADTH_HYST,
             limit_guard: bool = False,
             use_index_triggers: bool = False) -> tuple[pd.Series, pd.DataFrame, pd.Series]:
    """回测主流程，返回 (净值, 每日持仓权重, 每日换手 Σ|Δw|)。

    引擎参数默认 = 原版行为（use_scale=False, mom_blend=None）；
    优化开关由调用方（main 的消融结果）决定，保证 v2 等 import 方不受影响。
    未上市 ETF（NaN）自动排除出轮动池（动量 NaN → 权重 0），停牌日收益按 0 计。
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
                           breadth_trigger=breadth_trigger,
                           scale_defensive=scale_defensive,
                           normalize=normalize_trigger,
                           hyst=breadth_hyst,
                           use_index_triggers=use_index_triggers)
        port[STOCK_ETFS] = port[STOCK_ETFS].mul(sc, axis=0)
        port[TRESURY] = 1.0 - port[STOCK_ETFS].sum(axis=1)

    if limit_guard:   # 跌停卖不出/涨停买不进 → 换仓顺延（2015 式流动性危机的保守模拟）
        port = limit_guard_patch(port, closes)

    # 未上市/停牌日收益按 0 计（停牌无收益；未上市 ETF 权重为 0，不参与组合）
    daily_ret = closes.pct_change(fill_method=None).fillna(0.0)
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

def limit_guard_patch(port: pd.DataFrame, closes: pd.DataFrame) -> pd.DataFrame:
    """跌停/涨停不可成交约束：换仓日若卖出方一字跌停（或买入方涨停），
    当日成交不了 → 换仓顺延（前向迭代级联，连续封板自动继续推迟）。

    阈值：主板 10%；创业板 159915 自 2020-08-24 起 20%；科创50 上市即 20%。
    近似：收盘触及涨跌停（留 0.5% 容差）视为当日无法成交。
    """
    ret = closes[STOCK_ETFS].pct_change(fill_method=None)
    limit = pd.DataFrame(0.10, index=closes.index, columns=STOCK_ETFS)
    limit.loc[closes.index >= "2020-08-24", "159915.SZ"] = 0.20
    limit["588000.SH"] = 0.20
    blocked_sell = ret <= -(limit - 0.005)   # 跌停 → 卖不出
    blocked_buy = ret >= +(limit - 0.005)    # 涨停 → 买不进
    port = port.copy()
    for i in range(1, len(port)):
        dw = port.iloc[i] - port.iloc[i - 1]
        if (dw.abs() < 1e-9).all():
            continue
        t = port.index[i]
        # 涨跌停约束只作用于股票 ETF 腿（国债腿不受限）
        sell_mask = (dw < -1e-9) & dw.index.isin(STOCK_ETFS)
        buy_mask = (dw > 1e-9) & dw.index.isin(STOCK_ETFS)
        sell_blocked = bool(blocked_sell.loc[t, dw.index[sell_mask]].any()) if sell_mask.any() else False
        buy_blocked = bool(blocked_buy.loc[t, dw.index[buy_mask]].any()) if buy_mask.any() else False
        if sell_blocked or buy_blocked:
            port.iloc[i] = port.iloc[i - 1]   # 顺延：当日保持旧仓位
    return port

EPISODES = [          # 极端行情复盘区间（A 股主要危机）
    ("2008 大熊市", "2007-10-16", "2008-11-04"),
    ("2015 股灾", "2015-06-12", "2016-01-28"),
    ("2016 熔断", "2016-01-04", "2016-01-28"),
    ("2018 熊市", "2018-01-24", "2019-01-03"),
    ("2021-24 熊市", "2021-02-18", "2024-02-05"),
]

def print_episodes(closes: pd.DataFrame, nav: pd.Series, port: pd.DataFrame,
                   bench: dict[str, pd.Series]) -> None:
    """极端行情复盘：各段策略 vs 基准段内累计收益 + 防御状态统计。
    沪深300 基准用 000300 指数（2006 年即有，ETF 2012 年才上市）。"""
    idx300 = closes["000300.SH"]
    nav300 = idx300 / idx300.dropna().iloc[0]
    print(f"\n===== 极端行情复盘（段内累计收益；300 基准 = 000300 指数）=====")
    print(_pad_cjk("区间", 14) + _pad_cjk("轮动策略", 10) + _pad_cjk("等权持有", 10)
          + _pad_cjk("000300", 10) + _pad_cjk("平均权益仓", 10) + _pad_cjk("最低权益仓", 10))
    for name, a, b in EPISODES:
        mask = (closes.index >= a) & (closes.index <= b)
        idx = closes.index[mask]
        if len(idx) == 0:
            continue
        def seg_ret(s: pd.Series) -> float:
            seg = s.loc[idx].dropna()
            return float(seg.iloc[-1] / seg.iloc[0] - 1) if len(seg) else np.nan
        def cell(v: float) -> str:
            return "—".rjust(10) if pd.isna(v) else f"{v:>10.2%}"
        equity = 1.0 - port[TRESURY].loc[idx]
        print(f"{_pad_cjk(f'{name}', 14)}{cell(seg_ret(nav))}{cell(seg_ret(bench['等权持有']))}"
              f"{cell(seg_ret(nav300))}{equity.mean():>10.0%}{equity.min():>10.0%}")
    print("（注：2008 段等权实际仅上证50一只；2015 段含千股跌停流动性危机）")

# ---------------------------------------------------------------- 6. 指标与基准
def compute_metrics(nav: pd.Series, rf_annual: float) -> dict:
    """从净值序列计算绩效指标。年化按 252 个交易日。

    首尾可能为 NaN（基准在 union 索引上的未上市期）→ 按有效区间计算；
    年化年限取有效区间的长度（策略无 NaN，与全区间等价）。
    """
    v = nav.dropna()
    if len(v) == 0:   # 全 NaN（如切片期内该标的未上市）→ 指标记 NaN
        return {"总收益": np.nan, "年化收益": np.nan, "年化波动": np.nan,
                "最大回撤": np.nan, "夏普": np.nan, "卡玛": np.nan}
    n = len(v)
    total = v.iloc[-1] / v.iloc[0] - 1
    years = (n - 1) / TRADING_DAYS
    ann_ret = (v.iloc[-1] / v.iloc[0]) ** (1 / years) - 1
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

    等权买入持有用 (close/首个有效收盘价).mean(axis=1) —— 权重随涨跌漂移，
    不每日再平衡；未上市 ETF 为 NaN 自动跳过，即"已上市 ETF 动态等权"。
    """
    stocks = closes[STOCK_ETFS]
    # 首个有效收盘价（勿用 iloc[0]，可能为 NaN）；整列全 NaN（切片期未上市）→ 基价 NaN，均值时跳过
    def _first_valid(s: pd.Series) -> float:
        return float(s.dropna().iloc[0]) if s.notna().any() else np.nan

    base = stocks.apply(_first_valid)
    eq = (stocks / base).mean(axis=1)
    hs300 = closes["510300.SH"] / _first_valid(closes["510300.SH"])
    bench = {"等权持有": eq, "沪深300": hs300}
    if "000300.SH" in closes.columns:   # 真指数（2006 年起，比 ETF 长 6 年）
        bench["000300指数"] = closes["000300.SH"] / _first_valid(closes["000300.SH"])
    bond = closes[TRESURY] / _first_valid(closes[TRESURY])
    bench["国债ETF"] = bond
    return bench

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

def walk_forward(closes: pd.DataFrame, *, min_train: int = 504, step: int = 504,
                 metric: str = "夏普",
                 grid_n: tuple = (5, 10, 20, 30),
                 grid_trig: tuple = (2, 3, 4),
                 grid_sd: tuple = (0.4, 0.6),
                 grid_hyst: tuple = (0, 1, 2)) -> tuple[pd.DataFrame, list]:
    """滚出验证：每 step 个交易日，仅用此前数据网格选参（按 train 段夏普），
    在前瞻窗口上评估；同时评估固定默认参数作对照。

    关键点：测试段回测在 data[:test_end] 上跑再切片（保留 2016 起的完整热身），
    不用截断数据重跑。末窗口可能不完整。
    """
    rows = []
    picks = []
    test_starts = list(range(min_train, len(closes), step))
    for t0 in test_starts:
        test_end = min(t0 + step, len(closes))
        train = closes.iloc[:t0]
        full = closes.iloc[:test_end]
        best = None
        for n in grid_n:
            for trig in grid_trig:
                for sd in grid_sd:
                    for hyst in grid_hyst:
                        nav, _, _ = backtest(train, n, use_scale=True,
                                             breadth_trigger=trig,
                                             scale_defensive=sd,
                                             breadth_hyst=hyst)
                        score = compute_metrics(nav, RISK_FREE_ANNUAL)[metric]
                        if best is None or score > best[0]:
                            best = (score, n, trig, sd, hyst)
        _, n, trig, sd, hyst = best
        picks.append((n, trig, sd, hyst))
        # 选定参数在 full 上回测（含完整热身），切片测试段；默认参数同法对照
        nav_wf, _, _ = backtest(full, n, use_scale=True, breadth_trigger=trig,
                                scale_defensive=sd, breadth_hyst=hyst)
        nav_def, _, _ = backtest(full, WINDOW, use_scale=True,
                                 breadth_trigger=BREADTH_TRIGGER,
                                 scale_defensive=SCALE_DEFENSIVE,
                                 breadth_hyst=BREADTH_HYST,
                                 use_index_triggers=True)
        bench = benchmark_navs(closes.iloc[:test_end])
        m_wf = compute_metrics(nav_wf.iloc[t0:test_end], RISK_FREE_ANNUAL)
        m_def = compute_metrics(nav_def.iloc[t0:test_end], RISK_FREE_ANNUAL)
        m_eq = compute_metrics(bench["等权持有"].iloc[t0:test_end], RISK_FREE_ANNUAL)
        m_hs = compute_metrics(bench["沪深300"].iloc[t0:test_end], RISK_FREE_ANNUAL)
        rows.append({
            "起点": closes.index[t0], "结束": closes.index[test_end - 1],
            "N": n, "触发": trig, "仓位": sd, "滞回": hyst,
            "wf年化": m_wf["年化收益"], "wf回撤": m_wf["最大回撤"],
            "默认年化": m_def["年化收益"], "默认回撤": m_def["最大回撤"],
            "等权年化": m_eq["年化收益"], "300年化": m_hs["年化收益"],
        })
    return pd.DataFrame(rows), picks


def print_walk_forward(wf: pd.DataFrame, picks: list) -> None:
    from collections import Counter
    print(f"\n===== 滚出验证（每 2 年用此前数据按夏普网格选参，前瞻评估；"
          f"网格 N∈{{5,10,20,30}} × 触发∈{{2,3,4}} × 仓位∈{{0.4,0.6}} × 滞回∈{{0,1,2}}）=====")
    print(_pad_cjk("测试区间", 22) + _pad_cjk("选中参数", 14) + _pad_cjk("wf年化", 8)
          + _pad_cjk("wf回撤", 8) + _pad_cjk("默认年化", 9) + _pad_cjk("默认回撤", 9)
          + _pad_cjk("等权年化", 9) + _pad_cjk("300年化", 8))
    def cell(v: float, w: int = 8) -> str:
        return "—".rjust(w) if pd.isna(v) else f"{v:>{w}.2%}"

    for _, r in wf.iterrows():
        params = f"N={r['N']}≤{r['触发']}@{r['仓位']:.1f}h{r['滞回']}"
        print(f"{r['起点']:%Y-%m}~{r['结束']:%Y-%m}   {_pad_cjk(params, 14)}"
              f"{cell(r['wf年化'])}  {cell(r['wf回撤'])}  {cell(r['默认年化'], 9)}"
              f"  {cell(r['默认回撤'], 9)}  {cell(r['等权年化'], 9)}  {cell(r['300年化'])}")
    wf_ex = (wf["wf年化"] - wf["等权年化"]).dropna()   # 早期窗口基准未上市 → 剔除
    def_ex = (wf["默认年化"] - wf["等权年化"]).dropna()
    print(f"\n滚出选参：超额 vs 等权 均值 {wf_ex.mean():+.2%} ｜ 胜率 {(wf_ex > 0).mean():.0%}"
          f"（{int((wf_ex > 0).sum())}/{len(wf_ex)} 窗）")
    print(f"固定默认：超额 vs 等权 均值 {def_ex.mean():+.2%} ｜ 胜率 {(def_ex > 0).mean():.0%}"
          f"（{int((def_ex > 0).sum())}/{len(def_ex)} 窗）")
    for label, idx in (("N", 0), ("触发", 1), ("仓位", 2), ("滞回", 3)):
        c = Counter(p[idx] for p in picks)
        dist = "  ".join(f"{k}:{v}" for k, v in sorted(c.items()))
        print(f"选中参数分布 {label} = {dist}")


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

    ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=12))
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

def monthly_returns(rets: dict[str, pd.Series]) -> dict[str, pd.DataFrame]:
    """日收益 → {名称: 年份×月份 月度复利收益矩阵}（热力图与 CSV 共用）。"""
    out = {}
    for name, s in rets.items():
        m = (1.0 + s.fillna(0.0)).resample("ME").prod() - 1.0   # 月内复利
        df = pd.DataFrame({"year": m.index.year, "month": m.index.month, "ret": m.to_numpy()})
        out[name] = df.pivot(index="year", columns="month", values="ret")
    return out


def plot_monthly_heatmap(rets: dict[str, pd.Series], outpath: Path) -> None:
    """月度收益热力图：年份 × 月份。

    红涨绿跌（中国金融惯例）；每格标注数值作次级编码（色盲读者靠正负号读数）。
    颜色深浅 = 收益幅度，零点为中性浅灰（diverging 双色 + 中性中点）。
    首末月（2016-08、2026-08）不完整，标题注明。
    """
    names = list(rets.keys())
    mats = [monthly_returns(rets)[n] for n in names]
    vmax = max(np.nanmax(np.abs(m.to_numpy())) for m in mats)
    vmax = max(vmax, 0.005)
    cmap = LinearSegmentedColormap.from_list(
        "cn_ret", ["#0d7a0d", "#a8d4a8", "#f4f3f0", "#f0bcbc", "#c62828"])
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    cmap.set_bad("#ffffff")

    fig, axes = plt.subplots(1, len(mats), figsize=(6.4 * len(mats), 4.8), sharey=True)
    if len(mats) == 1:
        axes = [axes]
    for ax, mat, name in zip(axes, mats, names):
        im = ax.imshow(mat.to_numpy(), cmap=cmap, norm=norm, aspect="auto")
        ax.set_title(f"{name} 月度收益", fontsize=11)
        ax.set_xticks(range(12))
        ax.set_xticklabels([f"{i}月" for i in range(1, 13)], fontsize=8)
        ax.set_yticks(range(len(mat.index)))
        ax.set_yticklabels([str(y) for y in mat.index], fontsize=8)
        ax.set_xticks(np.arange(-0.5, 12, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(mat.index), 1), minor=True)
        ax.grid(which="minor", color="#ffffff", linewidth=0.8)
        ax.tick_params(which="minor", length=0)
        for iy, yi in enumerate(mat.index):
            for ix in range(12):
                xj = ix + 1
                v = mat.loc[yi, xj] if xj in mat.columns else np.nan
                if pd.isna(v):
                    continue
                txt_col = "#ffffff" if abs(v) > 0.55 * vmax else "#33322f"
                ax.text(ix, iy, f"{v * 100:+.1f}", ha="center", va="center",
                        fontsize=7.5, color=txt_col)
    fig.colorbar(im, ax=axes, fraction=0.03, pad=0.02,
                 format=PercentFormatter(1.0, decimals=0), label="月度收益")
    fig.suptitle("月度收益热力图（红涨绿跌；首末月不完整）", fontsize=12, y=1.0)
    # 不用 tight_layout（与 colorbar/suptitle 不兼容告警）；bbox_inches="tight" 已足够
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- 9. main
def _pad_cjk(s: str, width: int) -> str:
    """按显示宽度右对齐：CJK 字符按 2 列计。"""
    disp = sum(2 if ord(ch) > 0x2E7F else 1 for ch in s)
    return s + " " * max(0, width - disp)

def print_comparison(strat: dict, extra: dict, bench_m: dict[str, dict],
                     date_range: str) -> None:
    rows = [("总收益", "总收益"), ("年化收益", "年化收益"), ("年化波动", "年化波动"),
            ("最大回撤", "最大回撤"), ("夏普", "夏普"), ("卡玛", "卡玛")]
    names = ["轮动策略"] + list(bench_m.keys())
    print(f"\n===== 绩效对比（{date_range}，策略已扣佣金万1，基准不计成本）=====")
    print(_pad_cjk("指标", 10) + "  " + "  ".join(_pad_cjk(h, 10) for h in names))
    for label, key in rows:
        vals = [strat[key]] + [bench_m[n][key] for n in names[1:]]
        cells = [f"{v:>10.2%}" if key != "夏普" else f"{v:>10.2f}" for v in vals]
        print(_pad_cjk(label, 10) + "  " + "  ".join(cells))
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
    ap.add_argument("--window", type=int, default=WINDOW, help="动量窗口（交易日），默认 10")
    ap.add_argument("--no-scale", action="store_true", help="关闭广度择时 scale")
    ap.add_argument("--no-index-triggers", action="store_true", help="关闭指数触发（波动/均线）")
    ap.add_argument("--blend", action="store_true",
                    help="启用多窗口动量混合 20/60（消融显示为负优化，默认关闭）")
    ap.add_argument("--walk-forward", action="store_true",
                    help="运行滚出验证（每 2 年网格选参+前瞻评估，约 1 分钟）")
    ap.add_argument("--limit-guard", action="store_true",
                    help="启用跌停/涨停不可成交约束（换仓顺延）")
    ap.add_argument("--save-csv", action="store_true", help="保存净值 CSV 到 output/")
    ap.add_argument("--outdir", type=Path, default=OUTPUT_DIR, help="输出目录")
    args = ap.parse_args()

    use_scale = USE_SCALE and not args.no_scale
    use_idx = USE_INDEX_TRIGGERS and not args.no_index_triggers
    mom_blend = (20, 60) if args.blend else MOM_BLEND   # MOM_BLEND 默认 None（消融证伪混合）

    closes = load_closes(KLINES_DIR)
    print(f"数据：7 只 ETF，{len(closes)} 个交易日（{closes.index[0]:%Y-%m-%d} ~ "
          f"{closes.index[-1]:%Y-%m-%d}）；起始时间不对齐（159845 上市 2021-03、"
          f"588000 上市 2020-11），未上市期计 NaN：动态股票池自动排除、"
          f"广度触发按已上市数量归一（≤一半）")
    print(f"策略：动量窗口 N={args.window}"
          + (f" 混合{mom_blend}" if mom_blend else "")
          + (f" ｜ 广度择时（站上MA{BREADTH_DAYS}数量≤一半→权益{SCALE_DEFENSIVE:.0%}，"
             f"滞回±{BREADTH_HYST}）" if use_scale else "")
          + (f" ｜ 指数触发（60日波动>{INDEX_VOL_TRIGGER:.0%}或破MA{INDEX_MA_DAYS}）"
             if use_idx else "")
          + f" ｜ 每 {REBALANCE_DAYS} 个交易日调仓 ｜ 动量>0 入选前 {TOP_N} 各 50%"
          + f" ｜ 国债补足避险 ｜ 佣金单边万1")

    # 消融 A/B：引擎参数对比（同成本口径）
    print(f"\n===== v1 优化消融（引擎参数对比，同成本万1）=====")
    combos = [
        (f"原版 N={args.window}", dict(window=args.window)),
        ("文档值≤2@0.4原始", dict(window=args.window, use_scale=True, breadth_trigger=2,
                                scale_defensive=0.4, normalize_trigger=False,
                                breadth_hyst=0)),
        ("广度默认(无指数)", dict(window=args.window, use_scale=True)),
        ("+混合动量20/60", dict(window=args.window, mom_blend=(20, 60))),
        ("+scale+混合", dict(window=args.window, use_scale=True, mom_blend=(20, 60))),
        ("默认(广度+指数)", dict(window=args.window, use_scale=True, use_index_triggers=True)),
        ("默认+跌停约束", dict(window=args.window, use_scale=True, use_index_triggers=True,
                             limit_guard=True)),
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
                                   use_scale=use_scale, mom_blend=mom_blend,
                                   limit_guard=args.limit_guard,
                                   use_index_triggers=use_idx)
    strat = compute_metrics(nav, RISK_FREE_ANNUAL)
    years = (len(nav) - 1) / TRADING_DAYS
    extra = {"年换手率": turnover.sum() / 2 / years,
             "调仓次数": int((turnover > 1e-12).sum()),
             "国债占比": float((port[TRESURY] > 0.99).mean())}
    bench = benchmark_navs(closes)
    bench_m = {name: compute_metrics(s, RISK_FREE_ANNUAL) for name, s in bench.items()}
    print_comparison(strat, extra, bench_m,
                     f"{closes.index[0]:%Y-%m-%d} ~ {closes.index[-1]:%Y-%m-%d}")
    spans = {n: (s.dropna().index[0], s.dropna().index[-1]) for n, s in bench.items()}
    print("基准有效区间：" + " ｜ ".join(
        f"{n} {a:%Y-%m-%d}~{b:%Y-%m-%d}" for n, (a, b) in spans.items()))
    print("（注：各基准按自身有效区间年化；等权 2006-2011 年实际仅上证50一只，"
          "沪深300/国债ETF 上市较晚）")

    trades = trade_log(port, REBALANCE_DAYS)
    print(f"\n===== 调仓记录（生效日，共 {len(trades)} 次，显示最近 60 次）=====")
    for d, held in trades[-60:]:
        parts = [f"{ETF_NAMES[c]} {w:.0%}" for c, w in held]
        print(f"{d:%Y-%m-%d}  {' + '.join(parts)}")

    print_episodes(closes, nav, port, bench)

    scan = sensitivity_scan(closes, use_scale=use_scale)
    print_sensitivity(scan, args.window)

    if args.walk_forward:
        wf_df, picks = walk_forward(closes)
        print_walk_forward(wf_df, picks)

    args.outdir.mkdir(exist_ok=True)
    p1 = args.outdir / f"backtest_nav_N{args.window}.png"
    p2 = args.outdir / "sensitivity.png"
    p3 = args.outdir / "monthly_heatmap.png"
    sc = breadth_scale(closes) if use_scale else None
    plot_main(closes, nav, port, bench, args.window, p1, scale=sc)
    plot_sensitivity(scan, args.window, p2)
    ret_dict = {"轮动策略": nav.pct_change(fill_method=None).fillna(0.0),
                "等权持有": bench["等权持有"].pct_change(fill_method=None).fillna(0.0)}
    plot_monthly_heatmap(ret_dict, p3)
    print(f"\n图表已保存：{p1}  {p2}  {p3}")

    if args.save_csv:
        out = pd.DataFrame({"轮动策略": nav, **{n: s for n, s in bench.items()}})
        pcsv = args.outdir / f"nav_N{args.window}.csv"
        out.to_csv(pcsv, index_label="date")
        # 月度收益长表：year, month, 轮动策略, 等权持有（与热力图同源）
        mr = monthly_returns(ret_dict)
        mlong = pd.concat([mr["轮动策略"].stack().rename("轮动策略"),
                           mr["等权持有"].stack().rename("等权持有")], axis=1)
        mlong = mlong.dropna(how="all")   # 去掉无数据的月份（首末不完整月之外的空白）
        mlong.index.names = ["year", "month"]
        pcsv_m = args.outdir / "monthly_returns.csv"
        mlong.to_csv(pcsv_m)
        print(f"净值已保存：{pcsv}  {pcsv_m}")

if __name__ == "__main__":
    main()
