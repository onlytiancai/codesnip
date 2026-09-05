"""
ETF 轮动策略 v4 — 四策略并行回测

策略：
  S1  纯横截面动量（top-3 等权，MA60 个股过滤，沪深300 MA200 趋势开关）
  S2  风险预算（top-3，权重 ∝ 1/vol_20d，组合 vol 上限触发减仓）
  S3  中证全指宽度择时 + RSRS（bull/bear/neutral 三态，日频带 5% 偏离阈值）
  S4  横截面反转 + 趋势过滤（bottom-3，超跌反弹，5 日反转 MA60 趋势 + MA20 斜率）

回测引擎：无 lookahead（T 日 close 信号 → T+1 日 open 成交）
         双边万 1.5 佣金 + 单边 0.1% 冲击 = 总 0.13% / 单边换手
         初始资金 100 万

数据：klines/*.csv（11 个 ETF 日线）+ breadth.csv（中证全指宽度）
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager, rcParams

warnings.filterwarnings("ignore", category=FutureWarning)

# ============================================================================
# 0. 字体与路径
# ============================================================================

def setup_chinese_font() -> str | None:
    """matplotlib 中文回退字体，按 PingFang SC → Hiragino Sans GB → Heiti TC 顺序。"""
    candidates = [
        "PingFang SC", "Hiragino Sans GB", "Heiti TC",
        "STHeiti", "Microsoft YaHei", "SimHei",
        "Arial Unicode MS", "DejaVu Sans",
    ]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for c in candidates:
        if c in available:
            rcParams["font.sans-serif"] = [c, "DejaVu Sans"]
            rcParams["axes.unicode_minus"] = False
            return c
    return None


CN_FONT = setup_chinese_font()

PROJECT_ROOT = Path(__file__).resolve().parent
KLINES_DIR = PROJECT_ROOT / "klines"
BREADTH_PATH = PROJECT_ROOT / "breadth.csv"
OUTPUT_DIR = PROJECT_ROOT / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# ============================================================================
# 1. CONFIG — ETF 列表与名称、策略参数
# ============================================================================

ETF_LIST = [
    "000300.SH", "510050.SH", "159915.SZ", "510170.SH", "510300.SH",
    "510500.SH", "511010.SH", "518880.SH", "510880.SH", "588000.SH",
    "159845.SZ",
]

ETF_NAME = {
    "000300.SH": "沪深300",
    "510050.SH": "上证50",
    "159915.SZ": "创业板",
    "510170.SH": "商品",
    "510300.SH": "沪深300",
    "510500.SH": "中证500",
    "511010.SH": "国债",
    "518880.SH": "黄金",
    "510880.SH": "红利",
    "588000.SH": "科创50",
    "159845.SZ": "中证1000",
}

ETF_CLASS = {
    "000300.SH": "index",
    "510050.SH": "equity",
    "159915.SZ": "equity",
    "510170.SH": "commodity",
    "510300.SH": "equity",
    "510500.SH": "equity",
    "511010.SH": "bond",
    "518880.SH": "commodity",
    "510880.SH": "equity",
    "588000.SH": "equity",
    "159845.SZ": "equity",
}

INIT_CASH = 1_000_000
COMMISSION_RATE = 0.00015   # 单边
IMPACT_RATE = 0.001          # 单边
COST_RATE = COMMISSION_RATE * 2 + IMPACT_RATE  # 总 0.0013

# --- S1：纯横截面动量 ---
S1_PARAMS = dict(
    pool=ETF_LIST,
    momentum_lookback=20,
    ma_filter=60,
    top_k=3,
    rebalance_every=5,
    index_code="000300.SH",
    index_ma_filter=200,
    stop_per_position=-0.08,
    stop_portfolio_dd=-0.15,
    stop_portfolio_days=5,
)

# --- S2：风险预算（波动率倒数加权）---
S2_PARAMS = dict(
    pool=["510050.SH", "159915.SZ", "510300.SH", "510500.SH", "588000.SH", "159845.SZ"],
    momentum_lookback=60,
    vol_window=20,
    vol_cap=0.40,
    top_k=3,
    rebalance_every=10,
    portfolio_vol_cap=0.18,
    portfolio_scale=0.5,
    stop_trailing=-0.10,
)

# --- S3：宽度择时 + RSRS ---
S3_PARAMS = dict(
    pool=ETF_LIST,
    rsrs_window=18,
    rsrs_z_window=600,
    rsrs_z_threshold=0.7,
    breadth_window=10,
    breadth_bull=0.55,
    breadth_bear=0.45,
    breadth_panic=0.20,
    bond_code="511010.SH",   # 熊市转国债 ETF
    equity_pool=[
        "000300.SH", "510050.SH", "159915.SZ", "510170.SH", "510300.SH",
        "510500.SH", "510880.SH", "588000.SH", "159845.SZ",
    ],
    top_k=2,
    rebalance_threshold=0.05,
    stop_per_position=-0.08,
)

# --- S4：横截面反转 + 趋势过滤 ---
S4_PARAMS = dict(
    pool=["510050.SH", "159915.SZ", "510170.SH", "510300.SH", "510500.SH", "588000.SH", "159845.SZ"],
    reversion_lookback=5,
    ma_filter=60,
    ma_slope_window=20,
    ma_slope_diff=5,
    top_k=3,
    rebalance_every=5,
    index_code="000300.SH",
    index_vol_window=20,
    index_vol_cap=0.45,  # 放宽到 45%（避免熊市一刀切）
    max_single_weight=0.40,
    stop_per_position=-0.05,
    time_stop_days=10,
)

# ============================================================================
# 2. DATA — 加载与清洗
# ============================================================================

def load_klines(klines_dir: Path = KLINES_DIR) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    返回 (open_df, high_df, low_df, close_df, close_filled_df)
    close_filled_df 用于回测引擎的市值估算（ffill 处理停牌/末日期不对齐）。
    """
    opens, highs, lows, closes = [], [], [], []
    for code in ETF_LIST:
        path = klines_dir / f"{code}.csv"
        df = pd.read_csv(path, parse_dates=["date"], index_col="date")
        opens.append(df["open"].rename(code))
        highs.append(df["high"].rename(code))
        lows.append(df["low"].rename(code))
        closes.append(df["close"].rename(code))

    open_df = pd.concat(opens, axis=1).sort_index()
    high_df = pd.concat(highs, axis=1).sort_index()
    low_df = pd.concat(lows, axis=1).sort_index()
    close_df = pd.concat(closes, axis=1).sort_index()

    all_nan = open_df.isna().all(axis=1)
    if all_nan.any():
        first_valid = open_df.dropna(how="all").index[0]
        last_valid = open_df.dropna(how="all").index[-1]
        open_df = open_df.loc[first_valid:last_valid]
        high_df = high_df.loc[first_valid:last_valid]
        low_df = low_df.loc[first_valid:last_valid]
        close_df = close_df.loc[first_valid:last_valid]

    # ffill 用于市值估算（保留末日/停牌日的最近 close）。仅引擎使用。
    close_filled_df = close_df.ffill()

    return open_df, high_df, low_df, close_df, close_filled_df


def load_breadth(path: Path = BREADTH_PATH) -> pd.DataFrame | None:
    """加载中证全指每日 up/down/flat/total。文件不存在或格式异常返回 None。"""
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path, parse_dates=["date"], index_col="date", encoding="utf-8-sig")
        df = df.sort_index()
        return df
    except Exception:
        return None


def apply_guards(close_df: pd.DataFrame) -> pd.DataFrame:
    """
    数据质量修复：
      - 510880 在 2008-10-16 ~ 2009-01-13 出现 41 个负 close，全部 NaN 化
      - 2016-01-01 之前 510880 整段 NaN（与 v3 同处理，避开前复权污染期）
    """
    df = close_df.copy()
    if "510880.SH" in df.columns:
        polluted_start = pd.Timestamp("2008-10-16")
        polluted_end = pd.Timestamp("2009-01-13")
        mask = (df.index >= polluted_start) & (df.index <= polluted_end)
        df.loc[mask, "510880.SH"] = np.nan
        # 2016-01-01 之前整段 NaN
        df.loc[df.index < pd.Timestamp("2016-01-01"), "510880.SH"] = np.nan
    return df


# ============================================================================
# 3. SIGNALS — 通用技术信号
# ============================================================================

def momentum(close: pd.DataFrame, lookback: int) -> pd.DataFrame:
    """N 日动量 close_t / close_{t-lookback} - 1"""
    return close / close.shift(lookback) - 1


def rolling_ann_vol(close: pd.DataFrame, window: int) -> pd.DataFrame:
    """N 日实现年化波动率（基于日对数收益）。"""
    log_ret = np.log(close / close.shift(1))
    return log_ret.rolling(window).std() * np.sqrt(252)


def ma(series: pd.DataFrame, window: int) -> pd.DataFrame:
    return series.rolling(window).mean()


def ma_slope_positive(close: pd.DataFrame, ma_window: int, diff: int) -> pd.DataFrame:
    """MA[N] 当前值 > MA[N] 前 diff 日值（趋势向上）。"""
    m = ma(close, ma_window)
    return m > m.shift(diff)


def rsrs_slope(high_df: pd.DataFrame, low_df: pd.DataFrame, window: int) -> pd.DataFrame:
    """每个 ETF 用过去 window 日的 high 对 low 做 OLS 回归，输出斜率序列。"""
    cov = high_df.rolling(window).cov(low_df)
    var_low = low_df.rolling(window).var()
    return cov.div(var_low)


def rsrs_z(high_df: pd.DataFrame, low_df: pd.DataFrame,
           slope_window: int = 18, z_window: int = 600) -> pd.DataFrame:
    """RSRS 斜率 z-score：(slope - MA600) / STD600"""
    slope = rsrs_slope(high_df, low_df, slope_window)
    z = (slope - slope.rolling(z_window).mean()) / slope.rolling(z_window).std()
    return z


def breadth_pct_series(breadth_df: pd.DataFrame) -> pd.Series:
    """breadth_pct_d = up_d / (up_d + down_d)。"""
    denom = breadth_df["up"] + breadth_df["down"]
    pct = breadth_df["up"] / denom.replace(0, np.nan)
    return pct.reindex(pct.index).rename("breadth_pct")


def top_k_equal_weight(scores: pd.DataFrame, valid_mask: pd.DataFrame, k: int) -> pd.DataFrame:
    """
    每天在 valid_mask=True 的 ETF 中，按 scores 取 top-K，等权分配。
    无候选则全 0。
    """
    masked = scores.where(valid_mask, -np.inf)
    ranks = masked.rank(axis=1, method="first", ascending=False)
    selected = (ranks <= k) & (masked > -np.inf)
    n_selected = selected.sum(axis=1)
    out = selected.astype(float).div(n_selected.replace(0, np.nan), axis=0).fillna(0)
    return out


def rebalance_every_n(weights_daily: pd.DataFrame, every_n: int) -> pd.DataFrame:
    """权重每 every_n 个交易日更新一次；其余日期 forward-fill 上次值。"""
    if every_n <= 1:
        return weights_daily
    # 把非调仓日置 NaN，再用 ffill 传播
    rebal_mask = pd.Series(False, index=weights_daily.index)
    rebal_mask.iloc[::every_n] = True
    out = weights_daily.where(rebal_mask, np.nan)
    return out.ffill().fillna(0)


# ============================================================================
# 4. STRATEGIES — 输出 target_weights[t]，T 日 close 信号 → T+1 日 open 成交
# ============================================================================

def strategy_S1(close_df: pd.DataFrame, high_df: pd.DataFrame, low_df: pd.DataFrame,
                idx_close: pd.Series, breadth_df: pd.DataFrame | None,
                params: dict) -> pd.DataFrame:
    """纯横截面动量：top-3 等权 + MA60 个股趋势过滤 + 沪深300 MA200 趋势开关。"""
    p = params
    dates = close_df.index
    etfs = [c for c in p["pool"] if c in close_df.columns]
    cols = etfs + ["CASH"]

    mom = momentum(close_df[etfs], p["momentum_lookback"])
    ma60 = ma(close_df[etfs], p["ma_filter"])

    # 个股过滤：close > MA60（个股上升趋势）
    above_ma60 = (close_df[etfs] > ma60).fillna(False)
    # 数据充分性过滤
    history_ok = close_df[etfs].notna().rolling(p["ma_filter"]).sum() >= p["ma_filter"]

    # 趋势开关：沪深300 close > MA200（保护熊市）
    idx_ma200 = idx_close.rolling(p["index_ma_filter"]).mean()
    risk_on = (idx_close > idx_ma200).reindex(dates).fillna(False)

    valid = above_ma60 & history_ok
    # 整体熊市时所有候选失效
    valid.loc[~risk_on.values, :] = False

    daily_weights = top_k_equal_weight(mom.fillna(-np.inf), valid, p["top_k"])
    weights = rebalance_every_n(daily_weights, p["rebalance_every"])
    weights["CASH"] = (1 - weights[etfs].sum(axis=1)).clip(lower=0)

    return weights[cols]


def strategy_S2(close_df: pd.DataFrame, high_df: pd.DataFrame, low_df: pd.DataFrame,
                idx_close: pd.Series, breadth_df: pd.DataFrame | None,
                params: dict) -> pd.DataFrame:
    """风险预算：top-3 等权初选，权重按 1/vol 归一；组合 vol 上限触发减仓。"""
    p = params
    dates = close_df.index
    etfs = [c for c in p["pool"] if c in close_df.columns]
    cols = etfs + ["CASH"]

    mom = momentum(close_df[etfs], p["momentum_lookback"])
    vol = rolling_ann_vol(close_df[etfs], p["vol_window"])
    history_ok = close_df[etfs].notna().rolling(p["momentum_lookback"]).sum() >= p["momentum_lookback"]

    vol_ok = (vol < p["vol_cap"]) | vol.isna()  # NaN 不参与（IPO 早期）
    valid = history_ok & vol_ok

    # 候选：动量正向（避免做多最弱的）
    candidate_scores = mom.where(valid & (mom > 0), -np.inf)
    daily_top = top_k_equal_weight(candidate_scores, valid, p["top_k"])

    # 把等权改为 1/vol 归一
    inv_vol = (1 / vol).where(valid, np.nan)
    inv_vol_topk = inv_vol.where(daily_top > 0, np.nan)
    w_sum = inv_vol_topk.sum(axis=1)
    weighted = inv_vol_topk.div(w_sum.replace(0, np.nan), axis=0).fillna(0)
    weighted = weighted.where(daily_top > 0, 0)

    # 组合 vol 触发减仓
    port_ret = (close_df[etfs].pct_change() * weighted.shift(1)).sum(axis=1)
    port_vol = port_ret.rolling(p["vol_window"]).std() * np.sqrt(252)
    risk_scale = pd.Series(1.0, index=dates)
    risk_scale[port_vol > p["portfolio_vol_cap"]] = p["portfolio_scale"]
    weighted_scaled = weighted.mul(risk_scale, axis=0)
    weighted_scaled["CASH"] = (1 - weighted_scaled[etfs].sum(axis=1)).clip(lower=0)

    # 周期调仓
    weights = rebalance_every_n(weighted_scaled, p["rebalance_every"])
    return weights[cols]


def strategy_S3(close_df: pd.DataFrame, high_df: pd.DataFrame, low_df: pd.DataFrame,
                idx_close: pd.Series, breadth_df: pd.DataFrame | None,
                params: dict) -> pd.DataFrame:
    """中证全指宽度择时 + RSRS。每日输出 target，引擎按 5% 偏离阈值过滤换手。

    状态机：bull → 股票池 top-K by RSRS z；bear → 全 CASH（国债 ETF 自己也有熊市，直接 cash 更稳）；
    neutral → 维持上一信号。
    """
    p = params
    dates = close_df.index
    all_etfs = [c for c in p["pool"] if c in close_df.columns]
    equity_pool = [c for c in p["equity_pool"] if c in all_etfs]
    cols = all_etfs + ["CASH"]

    if breadth_df is None:
        breadth_start = dates[-1] + pd.Timedelta(days=1)
    else:
        breadth_start = breadth_df.index[0]

    # regime: 1=bull, 0=neutral, -1=bear
    regime = pd.Series(0, index=dates, dtype=int)

    for t in dates:
        if t < breadth_start:
            regime.loc[t] = 0
            continue
        hist_breadth = breadth_df.loc[breadth_df.index <= t].tail(p["breadth_window"] + 5)
        if len(hist_breadth) < p["breadth_window"]:
            regime.loc[t] = 0
            continue
        up = hist_breadth["up"].tail(p["breadth_window"]).sum()
        dn = hist_breadth["down"].tail(p["breadth_window"]).sum()
        if up + dn == 0:
            regime.loc[t] = 0
            continue
        pct = up / (up + dn)
        if pct > p["breadth_bull"]:
            regime.loc[t] = 1
        elif pct < p["breadth_bear"]:
            regime.loc[t] = -1
        else:
            prev_idx = regime.index.get_loc(t) - 1
            if prev_idx >= 0:
                regime.loc[t] = regime.iloc[prev_idx]
            else:
                regime.loc[t] = 0

    panic = pd.Series(False, index=dates)
    if breadth_df is not None:
        for t in dates:
            if t not in breadth_df.index:
                continue
            row = breadth_df.loc[t]
            up, dn = row["up"], row["down"]
            if up + dn > 0 and up / (up + dn) < p["breadth_panic"]:
                panic.loc[t] = True

    # RSRS z
    z = rsrs_z(high_df[all_etfs], low_df[all_etfs],
               p["rsrs_window"], p["rsrs_z_window"])
    z_threshold = p["rsrs_z_threshold"]
    z_ok = z > z_threshold

    history_ok = close_df[all_etfs].notna().rolling(p["rsrs_window"]).sum() >= p["rsrs_window"]

    out = pd.DataFrame(0.0, index=dates, columns=cols)

    for t in dates:
        if bool(panic.loc[t]):
            out.loc[t, "CASH"] = 1
            continue

        rg = int(regime.loc[t])
        if rg == -1:
            # bear：直接全 cash（国债/黄金也都有回撤，不如 cash 稳）
            out.loc[t, "CASH"] = 1
        elif rg == 1:
            # bull：股票池中 z>thresh 的，按 z 排序取 top-K
            valid_e = z_ok.loc[t, equity_pool] & history_ok.loc[t, equity_pool]
            if valid_e.sum() == 0:
                out.loc[t, "CASH"] = 1
            else:
                zvals = z.loc[t, equity_pool].where(valid_e, -np.inf)
                ranks = zvals.rank(method="first", ascending=False)
                sel = ranks <= p["top_k"]
                selected = sel[sel].index.tolist()
                n = len(selected)
                for c in selected:
                    out.loc[t, c] = 1.0 / n
                out.loc[t, "CASH"] = 0
        else:
            # neutral：维持上一信号
            out.loc[t, "CASH"] = 1

    return out[cols]


def strategy_S4(close_df: pd.DataFrame, high_df: pd.DataFrame, low_df: pd.DataFrame,
                idx_close: pd.Series, breadth_df: pd.DataFrame | None,
                params: dict) -> pd.DataFrame:
    """横截面反转：bottom-3（5 日负收益），MA60 趋势 + MA20 斜率过滤。"""
    p = params
    dates = close_df.index
    etfs = [c for c in p["pool"] if c in close_df.columns]
    cols = etfs + ["CASH"]

    ret_5 = close_df[etfs].pct_change(p["reversion_lookback"])
    ma60 = ma(close_df[etfs], p["ma_filter"])
    above_ma60 = (close_df[etfs] > ma60).fillna(False)
    slope_pos = ma_slope_positive(close_df[etfs], p["ma_slope_window"], p["ma_slope_diff"])

    history_ok = close_df[etfs].notna().rolling(p["ma_filter"]).sum() >= p["ma_filter"]

    # 整体趋势：沪深300 > MA60
    idx_ma60 = idx_close.rolling(p["ma_filter"]).mean()
    risk_on = (idx_close > idx_ma60).reindex(dates).fillna(False)

    # 危机过滤：沪深300 20 日年化 vol > 35%
    idx_vol = rolling_ann_vol(idx_close.to_frame("x"), p["index_vol_window"])["x"]
    vol_calm = (idx_vol <= p["index_vol_cap"]).reindex(dates).fillna(False)

    # 双重过滤：个股 above_ma60 & slope_pos & history_ok；整体 risk_on & vol_calm
    valid = above_ma60 & slope_pos & history_ok
    valid.loc[~risk_on.values, :] = False
    valid.loc[~vol_calm.values, :] = False

    # 反转：取 5 日收益最低（最负）的 top-K，等权
    # 对有效 ETF，按 ret_5 升序取前 K 个
    masked = ret_5.where(valid, np.inf)
    daily = pd.DataFrame(0.0, index=dates, columns=etfs)

    for t in dates:
        s = masked.loc[t]
        s_valid = s[s < np.inf].sort_values()
        if len(s_valid) == 0:
            continue
        # 仅取负收益的（避免 "smallest positive = weakest" 误选）
        s_neg = s_valid[s_valid < 0]
        chosen = s_neg.head(p["top_k"])
        if len(chosen) == 0:
            continue
        n = len(chosen)
        for c in chosen.index:
            daily.loc[t, c] = 1.0 / n

    # 单票权重上限
    daily = daily.clip(upper=p["max_single_weight"])
    daily["CASH"] = (1 - daily[etfs].sum(axis=1)).clip(lower=0)

    weights = rebalance_every_n(daily, p["rebalance_every"])
    return weights[cols]


# ============================================================================
# 5. ENGINE — 回测引擎（含止损、偏离阈值、成本）
# ============================================================================

def backtest(open_df: pd.DataFrame, close_df: pd.DataFrame, close_filled_df: pd.DataFrame,
             raw_target: pd.DataFrame, stop_rules: dict,
             rebalance_threshold: float = 0.0,
             cost_rate: float = COST_RATE, init_cash: float = INIT_CASH):
    """
    raw_target[t] = strategy 在 close[t] 时计算的目标权重
    引擎在 open[t+1] 成交

    close_filled_df: ffill 后的 close，用于停牌/末日估值（信号仍用原 close_df）

    stop_rules 支持的键：
      - stop_per_position: float, 从 entry_price 起的固定止损（如 -0.08）
      - stop_trailing: float, 从 entry 后最高 close 起的跟踪止损（如 -0.10）
      - stop_time_days: int, 持仓超过 N 日强制平仓
      - stop_portfolio_dd: float, 组合回撤阈值（如 -0.15）
      - stop_portfolio_days: int, 触发后保持 cash 的天数

    rebalance_threshold: 仅当 |target - held| > 此值时才真换手（S3 用）
    """
    dates = close_df.index
    etfs = list(close_df.columns)
    cash = init_cash
    shares = pd.Series(0.0, index=etfs)
    entry_price = pd.Series(np.nan, index=etfs)
    entry_peak = pd.Series(np.nan, index=etfs)
    entry_date = pd.Series(pd.NaT, index=etfs)
    peak_nav = init_cash
    cash_lock_until = None  # type: pd.Timestamp | None

    nav = pd.Series(np.nan, index=dates, dtype=float)
    weights_held = pd.DataFrame(0.0, index=dates, columns=etfs + ["CASH"])
    turnover = pd.Series(0.0, index=dates)
    cost_paid = pd.Series(0.0, index=dates)

    for i, t in enumerate(dates):
        # ---- 1. 更新 entry_peak（基于今日 close）----
        for code in etfs:
            if shares[code] > 0 and not pd.isna(close_df.loc[t, code]):
                p = close_df.loc[t, code]
                if pd.isna(entry_peak[code]) or p > entry_peak[code]:
                    entry_peak[code] = p

        # ---- 2. 标记市值（用 ffill 后的 close，处理停牌/末日）----
        close_t = close_df.loc[t]
        value_t = close_filled_df.loc[t]
        pos_val = float(np.nansum(shares.values * value_t.fillna(0).values))
        nav[t] = cash + pos_val
        # 只在"非现金锁定"期间更新 peak_nav
        if cash_lock_until is None or t > cash_lock_until:
            if peak_nav < nav[t]:
                peak_nav = float(nav[t])

        # 锁定到期则清掉，并立即把 peak_nav 重置为当前 nav
        if cash_lock_until is not None and t > cash_lock_until:
            cash_lock_until = None
            peak_nav = float(nav[t])

        if nav[t] > 0:
            weights_held.loc[t, etfs] = (shares.values * value_t.fillna(0).values) / nav[t]
            weights_held.loc[t, "CASH"] = cash / nav[t]

        # ---- 3. 读取 raw target ----
        if t in raw_target.index:
            target = raw_target.loc[t].reindex(etfs + ["CASH"]).fillna(0).copy()
        else:
            target = pd.Series(0.0, index=etfs + ["CASH"])
            target["CASH"] = 1.0

        # ---- 4. 应用止损 ----
        # 4a. 组合回撤止损：仅在未锁时触发；触发后 N 天全 cash
        if stop_rules.get("stop_portfolio_dd") is not None and peak_nav > 0 \
                and (cash_lock_until is None or t > cash_lock_until):
            dd_th = stop_rules["stop_portfolio_dd"]
            lock_days = stop_rules.get("stop_portfolio_days", 0)
            cur_dd = nav[t] / peak_nav - 1
            if cur_dd <= dd_th:
                cash_lock_until = dates[min(i + lock_days, len(dates) - 1)]
        if cash_lock_until is not None and t <= cash_lock_until:
            target[etfs] = 0
            target["CASH"] = 1

        # 4b. 单票固定止损（from entry）
        if stop_rules.get("stop_per_position") is not None:
            stop_pct = stop_rules["stop_per_position"]
            for code in etfs:
                if shares[code] > 0 and not pd.isna(entry_price[code]) \
                        and not pd.isna(close_t[code]):
                    if close_t[code] / entry_price[code] - 1 <= stop_pct:
                        target[code] = 0
        # 4c. 单票跟踪止损（from entry_peak）
        if stop_rules.get("stop_trailing") is not None:
            stop_pct = stop_rules["stop_trailing"]
            for code in etfs:
                if shares[code] > 0 and not pd.isna(entry_peak[code]) \
                        and not pd.isna(close_t[code]):
                    if close_t[code] / entry_peak[code] - 1 <= stop_pct:
                        target[code] = 0
        # 4d. 时间止损
        if stop_rules.get("stop_time_days") is not None:
            max_days = stop_rules["stop_time_days"]
            for code in etfs:
                if shares[code] > 0 and not pd.isna(entry_date[code]):
                    days_held = i - dates.get_loc(entry_date[code])
                    if days_held >= max_days:
                        target[code] = 0

        # 归一 cash
        target_sum = sum(target[c] for c in etfs)
        if target_sum > 1:
            scale = 1.0 / target_sum
            for c in etfs:
                target[c] *= scale
            target_sum = 1.0
        target["CASH"] = max(0.0, 1.0 - target_sum)

        # ---- 5. 偏离阈值（S3）：仅当 |target - held| > thr 才换 ----
        if rebalance_threshold > 0 and i > 0:
            held_w = weights_held.loc[t]  # 今日收盘时实际权重
            for code in etfs:
                if abs(target[code] - held_w[code]) <= rebalance_threshold:
                    target[code] = held_w[code]
            target["CASH"] = max(0.0, 1.0 - sum(target[c] for c in etfs))

        # ---- 6. T+1 open 成交 ----
        if i + 1 < len(dates):
            t_next = dates[i + 1]
            open_next = open_df.loc[t_next]

            new_shares = shares.copy()  # 默认保持当前仓位
            trade_value = 0.0

            for code in etfs:
                w = float(target[code])
                px = open_next[code] if code in open_next.index else np.nan
                if pd.isna(px) or px <= 0:
                    # 次日不交易，保持当前仓位
                    continue
                target_val = w * nav[t]
                desired_shares = target_val / px
                diff = desired_shares - shares[code]
                if pd.isna(diff):
                    continue
                if diff != 0:
                    trade_value += abs(diff) * px
                new_shares[code] = desired_shares

            # 更新 cash（先卖后买，cost 最后扣）
            for code in etfs:
                diff = new_shares[code] - shares[code]
                if pd.isna(diff) or diff == 0:
                    continue
                px = open_next[code]
                if pd.isna(px) or px <= 0:
                    new_shares[code] = shares[code]  # 无法成交
                    continue
                if diff > 0:
                    cash -= diff * px
                else:
                    cash += (-diff) * px

            cost = trade_value * cost_rate
            cash -= cost
            cost_paid[t] = cost
            turnover[t] = trade_value / 2  # 单边换手

            # 更新 entry 信息
            was_holding = shares > 0
            for code in etfs:
                if new_shares[code] > 0:
                    if not was_holding[code]:
                        entry_price[code] = open_next[code]
                        entry_peak[code] = open_next[code]
                        entry_date[code] = t_next
                    # 否则保留原 entry
                else:
                    entry_price[code] = np.nan
                    entry_peak[code] = np.nan
                    entry_date[code] = pd.NaT

            shares = new_shares

    return nav, weights_held, turnover, cost_paid


# ============================================================================
# 6. METRICS — 评估指标
# ============================================================================

def compute_metrics(nav: pd.Series, weights: pd.DataFrame,
                    turnover: pd.Series, cost_paid: pd.Series) -> dict:
    """单个策略的指标。nav 第一天等于初始资金。"""
    nav = nav.dropna()
    if len(nav) < 2:
        return {}

    daily_ret = nav.pct_change().dropna()
    n_days = len(daily_ret)
    total_ret = nav.iloc[-1] / nav.iloc[0] - 1
    ann_ret = (1 + total_ret) ** (252 / n_days) - 1
    ann_vol = daily_ret.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0

    nav_running_max = nav.cummax()
    drawdown = nav / nav_running_max - 1
    max_dd = drawdown.min()
    calmar = ann_ret / abs(max_dd) if max_dd < 0 else 0

    win_rate = (daily_ret > 0).mean()
    avg_nav = nav.mean()
    # 单边换手 / 平均 NAV = 每次调仓换手比例
    if (turnover > 0).any() and avg_nav > 0:
        avg_turnover_pct = turnover[turnover > 0].mean() / avg_nav
    else:
        avg_turnover_pct = 0.0
    # 持仓比例（非 CASH 的时间占比）
    if "CASH" in weights.columns:
        invested_pct = 1 - weights["CASH"].mean()
    else:
        invested_pct = 0.0
    total_cost = cost_paid.sum()
    n_trades = (turnover > 0).sum()

    by_year = daily_ret.groupby(daily_ret.index.year)
    best_year = by_year.apply(lambda x: (1 + x).prod() - 1).max()
    worst_year = by_year.apply(lambda x: (1 + x).prod() - 1).min()

    return dict(
        start=nav.index[0].strftime("%Y-%m-%d"),
        end=nav.index[-1].strftime("%Y-%m-%d"),
        n_days=int(n_days),
        total_ret=float(total_ret),
        ann_ret=float(ann_ret),
        ann_vol=float(ann_vol),
        sharpe=float(sharpe),
        max_dd=float(max_dd),
        calmar=float(calmar),
        win_rate=float(win_rate),
        avg_turnover_pct=float(avg_turnover_pct),
        invested_pct=float(invested_pct),
        total_cost=float(total_cost),
        n_trades=int(n_trades),
        best_year=float(best_year),
        worst_year=float(worst_year),
    )


def benchmark_backtest(idx_open: pd.Series, idx_close: pd.Series,
                       cost_rate: float = COST_RATE,
                       init_cash: float = INIT_CASH) -> pd.Series:
    """沪深300 买入持有（同成本模型：只在建仓时扣成本）。"""
    nav = pd.Series(np.nan, index=idx_close.index)
    nav.iloc[0] = init_cash
    shares = init_cash / idx_open.iloc[0]
    cost = init_cash * cost_rate
    cash = init_cash - cost - shares * idx_open.iloc[0]
    nav.iloc[0] = cash + shares * idx_close.iloc[0]
    for i in range(1, len(idx_close)):
        nav.iloc[i] = cash + shares * idx_close.iloc[i]
    return nav


# ============================================================================
# 7. PLOTS
# ============================================================================

COLORS = {
    "S1": "#d62728", "S2": "#1f77b4", "S3": "#2ca02c", "S4": "#ff7f0e",
    "Bench": "#7f7f7f",
}


def plot_nav(navs: dict, bench_nav: pd.Series, title: str, path: Path):
    plt.figure(figsize=(11, 6))
    for k, nav in navs.items():
        plt.plot(nav.index, nav.values / 1e6, label=k, color=COLORS.get(k, None),
                 linewidth=1.6)
    plt.plot(bench_nav.index, bench_nav.values / 1e6, label="000300 基准",
             color=COLORS["Bench"], linewidth=1.4, linestyle="--")
    plt.title(title, fontsize=13)
    plt.ylabel("净值 (百万 CNY)")
    plt.xlabel("日期")
    plt.legend(loc="upper left", framealpha=0.85)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=110)
    plt.close()


def plot_drawdown(navs: dict, path: Path):
    plt.figure(figsize=(11, 6))
    for k, nav in navs.items():
        dd = nav / nav.cummax() - 1
        plt.plot(dd.index, dd.values * 100, label=k, color=COLORS.get(k, None),
                 linewidth=1.4)
    plt.title("回撤对比", fontsize=13)
    plt.ylabel("回撤 (%)")
    plt.xlabel("日期")
    plt.legend(loc="lower left", framealpha=0.85)
    plt.grid(True, alpha=0.3)
    plt.axhline(0, color="black", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(path, dpi=110)
    plt.close()


def plot_holdings(weights: pd.DataFrame, title: str, path: Path):
    """持仓权重热力图（按季度末采样）。"""
    # 季度末采样
    sampled = weights.resample("QE").last()
    fig, ax = plt.subplots(figsize=(13, 6))
    # 只画股票+债券+商品 ETF，不画 CASH（最后一行）
    etf_cols = [c for c in weights.columns if c != "CASH"]
    mat = sampled[etf_cols].T  # 行=ETF, 列=季度
    im = ax.imshow(mat.values, aspect="auto", cmap="YlGnBu", vmin=0, vmax=max(0.4, mat.values.max()))
    ax.set_yticks(range(len(etf_cols)))
    ax.set_yticklabels([f"{c} {ETF_NAME.get(c, '')}" for c in etf_cols], fontsize=9)
    # x 轴：每 4 个季度一个 label
    n_q = len(mat.columns)
    step = max(1, n_q // 12)
    xticks = list(range(0, n_q, step))
    xlabels = [f"{mat.columns[i].year}Q{(mat.columns[i].month - 1) // 3 + 1}" for i in xticks]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, rotation=45, ha="right", fontsize=8)
    ax.set_title(title, fontsize=12)
    cbar = plt.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label("权重", fontsize=9)
    plt.tight_layout()
    plt.savefig(path, dpi=110)
    plt.close()


def plot_monthly(nav: pd.Series, title: str, path: Path):
    """月度收益热力图：行=年份，列=月份。"""
    daily_ret = nav.pct_change().dropna()
    monthly = (1 + daily_ret).resample("ME").prod() - 1
    monthly_table = pd.DataFrame({
        "year": monthly.index.year,
        "month": monthly.index.month,
        "ret": monthly.values,
    })
    pivot = monthly_table.pivot(index="year", columns="month", values="ret").sort_index()
    # 填充缺月
    for m in range(1, 13):
        if m not in pivot.columns:
            pivot[m] = np.nan
    pivot = pivot[sorted(pivot.columns)]

    fig, ax = plt.subplots(figsize=(11, max(3, 0.4 * len(pivot))))
    vmax = max(abs(pivot.values.min()), abs(pivot.values.max()), 0.05)
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn",
                   vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(12))
    ax.set_xticklabels(["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12"])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([str(y) for y in pivot.index])
    ax.set_title(title, fontsize=12)
    # 标注
    for i in range(len(pivot.index)):
        for j in range(12):
            v = pivot.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v*100:+.1f}", ha="center", va="center",
                        fontsize=7, color="black" if abs(v) < vmax * 0.6 else "white")
    cbar = plt.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label("月度收益", fontsize=9)
    plt.tight_layout()
    plt.savefig(path, dpi=110)
    plt.close()


# ============================================================================
# 8. CLI / main
# ============================================================================

STRATEGY_REGISTRY = {
    "S1": ("纯横截面动量", strategy_S1, S1_PARAMS,
           dict(stop_per_position=S1_PARAMS["stop_per_position"],
                stop_portfolio_dd=S1_PARAMS["stop_portfolio_dd"],
                stop_portfolio_days=S1_PARAMS["stop_portfolio_days"]),
           0.0),
    "S2": ("风险预算(波动率倒数)", strategy_S2, S2_PARAMS,
           dict(stop_trailing=S2_PARAMS["stop_trailing"]),
           0.0),
    "S3": ("宽度择时+RSRS", strategy_S3, S3_PARAMS,
           dict(stop_per_position=S3_PARAMS["stop_per_position"]),
           S3_PARAMS["rebalance_threshold"]),
    "S4": ("横截面反转+趋势过滤", strategy_S4, S4_PARAMS,
           dict(stop_per_position=S4_PARAMS["stop_per_position"],
                stop_time_days=S4_PARAMS["time_stop_days"]),
           0.0),
}


def run_strategy(name: str, open_df, close_df, close_filled_df, high_df, low_df,
                 idx_close, breadth_df, output_dir: Path):
    short_name, fn, params, stop_rules, rebal_thr = STRATEGY_REGISTRY[name]
    print(f"\n=== 运行 {name} {short_name} ===")

    target = fn(close_df, high_df, low_df, idx_close, breadth_df, params)
    # 第一天默认全 cash（无 prior 决策）
    target.iloc[0] = 0
    target.iloc[0, target.columns.get_loc("CASH")] = 1.0

    nav, weights, turnover, cost = backtest(
        open_df, close_df, close_filled_df, target,
        stop_rules=stop_rules,
        rebalance_threshold=rebal_thr,
    )

    # 输出 nav csv（含日收益、累计成本）
    nav_df = pd.DataFrame({
        "nav": nav,
        "ret": nav.pct_change(),
        "drawdown": nav / nav.cummax() - 1,
        "cost_cumsum": cost.cumsum().reindex(nav.index).ffill(),
    })
    weights_out = weights.copy()
    out_combined = pd.concat([nav_df, weights_out], axis=1)
    out_combined.to_csv(output_dir / f"nav_{name}.csv", encoding="utf-8-sig")

    # 单独保存 weights
    weights_out.to_csv(output_dir / f"weights_{name}.csv", encoding="utf-8-sig")

    # 指标
    metrics = compute_metrics(nav, weights, turnover, cost)
    metrics["name"] = name
    metrics["desc"] = short_name

    # 图表
    plot_holdings(weights, f"{name} {short_name} — 持仓权重（季度采样）",
                  output_dir / f"plot_holdings_{name}.png")
    plot_monthly(nav.dropna(), f"{name} {short_name} — 月度收益",
                 output_dir / f"plot_monthly_{name}.png")

    print(f"  年化={metrics['ann_ret']*100:.2f}%  Sharpe={metrics['sharpe']:.2f}  "
          f"最大回撤={metrics['max_dd']*100:.2f}%  换手={metrics['avg_turnover_pct']*100:.2f}%/次")
    return metrics


def main():
    parser = argparse.ArgumentParser(description="ETF 轮动 v4 — 四策略并行回测")
    parser.add_argument("--strategies", nargs="*", default=None,
                        help="要跑的策略，如 S1 S3；不指定则跑全部")
    parser.add_argument("--all", action="store_true", help="跑全部 4 个策略")
    parser.add_argument("--no-breadth", action="store_true",
                        help="忽略 breadth.csv（影响 S3）")
    parser.add_argument("--start", default=None, help="回测起始日期 YYYY-MM-DD")
    parser.add_argument("--cost", type=float, default=COST_RATE,
                        help="单边换手总成本，默认 0.0013")
    args = parser.parse_args()

    if args.all or not args.strategies:
        strategies = list(STRATEGY_REGISTRY.keys())
    else:
        strategies = args.strategies
        for s in strategies:
            if s not in STRATEGY_REGISTRY:
                raise SystemExit(f"未知策略 {s}；可选：{list(STRATEGY_REGISTRY.keys())}")

    print(f"字体: {CN_FONT or '未找到中文字体（可能中文显示异常）'}")
    print(f"输出目录: {OUTPUT_DIR}")

    # ---- 加载数据 ----
    open_df, high_df, low_df, close_df, close_filled_df = load_klines()
    close_df = apply_guards(close_df)
    close_filled_df = apply_guards(close_filled_df)

    breadth_df = None if args.no_breadth else load_breadth()
    if breadth_df is None and not args.no_breadth:
        print("[警告] breadth.csv 未加载或格式异常，S3 将 fallback 到 沪深300 MA200")
    else:
        print(f"宽度数据: {len(breadth_df)} 天 ({breadth_df.index[0].date()} → {breadth_df.index[-1].date()})")

    print(f"K 线对齐后: {len(close_df)} 天 ({close_df.index[0].date()} → {close_df.index[-1].date()})")

    if args.start:
        start = pd.Timestamp(args.start)
        open_df = open_df.loc[start:]
        high_df = high_df.loc[start:]
        low_df = low_df.loc[start:]
        close_df = close_df.loc[start:]

    idx_close = close_df["000300.SH"]

    # ---- 基准 ----
    bench_nav = benchmark_backtest(open_df["000300.SH"], idx_close, cost_rate=args.cost)
    bench_metrics = compute_metrics(bench_nav, pd.DataFrame(),
                                    pd.Series(0.0, index=bench_nav.index),
                                    pd.Series(0.0, index=bench_nav.index))
    bench_metrics["name"] = "Bench"
    bench_metrics["desc"] = "沪深300 买入持有"

    # ---- 跑策略 ----
    all_metrics = [bench_metrics]
    navs_for_plot = {"000300": bench_nav}

    for sname in strategies:
        m = run_strategy(sname, open_df, close_df, close_filled_df,
                         high_df, low_df, idx_close, breadth_df, OUTPUT_DIR)
        all_metrics.append(m)
        navs_for_plot[sname] = pd.read_csv(OUTPUT_DIR / f"nav_{sname}.csv",
                                           parse_dates=[0], index_col=0)["nav"]

    # ---- 汇总 ----
    metrics_df = pd.DataFrame(all_metrics)
    cols_order = ["name", "desc", "start", "end", "n_days",
                  "ann_ret", "ann_vol", "sharpe", "max_dd", "calmar",
                  "win_rate", "invested_pct", "avg_turnover_pct", "n_trades",
                  "best_year", "worst_year", "total_cost"]
    metrics_df = metrics_df[cols_order]
    # 百分比格式化
    for c in ["ann_ret", "ann_vol", "max_dd", "win_rate",
              "invested_pct", "avg_turnover_pct",
              "best_year", "worst_year"]:
        metrics_df[c] = metrics_df[c].map(lambda x: f"{x*100:.2f}%")
    metrics_df["sharpe"] = metrics_df["sharpe"].map(lambda x: f"{x:.3f}")
    metrics_df["calmar"] = metrics_df["calmar"].map(lambda x: f"{x:.3f}")
    metrics_df["total_cost"] = metrics_df["total_cost"].map(lambda x: f"{x:.0f}")
    metrics_df.to_csv(OUTPUT_DIR / "metrics.csv", index=False, encoding="utf-8-sig")

    # ---- 综合图表 ----
    plot_nav(navs_for_plot, bench_nav, "ETF 轮动 v4 — NAV 对比", OUTPUT_DIR / "plot_nav.png")
    plot_drawdown({k: v for k, v in navs_for_plot.items() if k != "000300"},
                  OUTPUT_DIR / "plot_drawdown.png")

    print("\n=== 完成 ===")
    print(f"\nmetrics.csv 摘要：")
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()
