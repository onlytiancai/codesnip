"""A股宽基ETF多因子轮动策略回测（MVP 版本）。

策略来源：docs/豆包ETF轮动策略描述.md
核心：
- 6 只宽基 ETF 月末调仓（动量 60% + 下行波动率 30% + 流动性 10%）
- 3 触发熊市择时 → 缩权益 → 剩余资金配国债 ETF
- 5% 调仓阈值 + 单标的 40% 上限 + 波动率倒数加权
- 输出：业绩指标、因子 IC、场景分段、权重时序、换手率、7 张图

用法
----
~/.pyenv/versions/qlib/bin/python backtest_etf_rotation.py \\
    --start-date 2021-04-01 --end-date 2026-08-12 \\
    --output-dir .cache/backtest/

默认行为：未给 --index-csv 时，自动用 .cache/klines/510300.SH.csv
作为沪深300指数的代理（相关系数 > 0.999，足以驱动 MA/波动率触发）。
eltdx 服务端对 sh000300 日 K 有协议 bug（解码失败），故回落 ETF。
"""
from __future__ import annotations

import argparse
import sys
import warnings
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")  # non-interactive backend; 在 import pyplot 前设

import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from scipy import stats  # noqa: E402

import empyrical as ep  # noqa: E402

# empyrical 0.5.5 兼容层：np.NINF 在 numpy 2.0+ 移除；patch 一下避免 sortino 崩
if not hasattr(np, "NINF"):
    np.NINF = -np.inf

from fetch_etf_klines import (  # noqa: E402
    _name_for,
    load_close_series,
)

# ---- 中文字体（与 fetch_etf_klines 同套设置） -------------------------------
_CN_FONT_CANDIDATES = ["PingFang SC", "Hiragino Sans GB", "Heiti TC"]
_PINGFANG_TTC_CANDIDATES = [
    "/System/Library/AssetsV2/com_apple_MobileAsset_Font8/"
    "86ba2c91f017a3749571a82f2c6d890ac7ffb2fb.asset/AssetData/PingFang.ttc",
    "/System/Library/PrivateFrameworks/FontServices.framework/Resources/"
    "Reserved/PingFangUI.ttc",
    "/Library/Fonts/PingFang.ttc",
]


def _setup_chinese_font() -> None:
    import matplotlib.font_manager as fm

    for path in _PINGFANG_TTC_CANDIDATES:
        if Path(path).exists():
            try:
                fm.fontManager.addfont(path)
            except Exception:  # noqa: BLE001
                pass
    chosen: str | None = None
    for name in _CN_FONT_CANDIDATES:
        try:
            fm.findfont(name, fallback_to_default=False)
            chosen = name
            break
        except Exception:  # noqa: BLE001
            continue
    fallback_chain = _CN_FONT_CANDIDATES + ["DejaVu Sans"]
    if chosen is not None:
        sans_list = [chosen] + [n for n in fallback_chain if n != chosen]
    else:
        sans_list = fallback_chain
    plt.rcParams["font.sans-serif"] = sans_list
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["text.usetex"] = False
    if chosen is None:
        print(
            "[WARN] 中文字体回退链全部未命中；CJK 字符将显示为方框。",
            file=sys.stderr,
        )


_setup_chinese_font()

# matplotlib 的 font_manager 在某些 label 渲染时会用 U+2212；先关掉警告
warnings.filterwarnings(
    "ignore",
    message=r".*does not have a glyph for.*",
    category=UserWarning,
)


# ---------------------------------------------------------------------------
# 默认参数（与 fetch_etf_klines.DEFAULT_ETFS 共用；剔除 511010.SH 留给避险）
# ---------------------------------------------------------------------------
EQUITY_ETFS: list[str] = [
    "510050.SH",   # 上证50
    "510300.SH",   # 沪深300
    "510500.SH",   # 中证500
    "159845.SZ",   # 中证1000
    "159915.SZ",   # 创业板
    "588000.SH",   # 科创50
]
BOND_ETF = "511010.SH"  # 国债ETF（避险）
DEFAULT_INDEX_PROXY = "510300.SH"  # 当沪深300指数不可用时的 ETF 代理

# 因子权重（来自策略文档 §3.1）
W_MOM = 0.6
W_DOWN_VOL = 0.3
W_LIQ = 0.1

# 默认输出目录
DEFAULT_OUTPUT_DIR = Path(__file__).parent / ".cache" / "backtest"


# ---------------------------------------------------------------------------
# 数据加载
# ---------------------------------------------------------------------------
def load_universe(
    data_dir: Path,
    codes: list[str],
    start_date: date | None = None,
    end_date: date | None = None,
) -> dict[str, pd.DataFrame]:
    """读每只 ETF 的全字段 CSV，返回 {code: DataFrame}。"""
    out: dict[str, pd.DataFrame] = {}
    for c in codes:
        path = data_dir / f"{c}.csv"
        if not path.exists():
            print(f"[WARN] 缺数据：{c} ({path})", file=sys.stderr)
            continue
        df = pd.read_csv(path)
        if df.empty:
            continue
        if start_date is not None or end_date is not None:
            d = pd.to_datetime(df["date"]).dt.date
            mask = pd.Series(True, index=df.index)
            if start_date is not None:
                mask &= d >= start_date
            if end_date is not None:
                mask &= d <= end_date
            df = df[mask].reset_index(drop=True)
        df = df.sort_values("date").reset_index(drop=True)
        out[c] = df
    return out


def load_aligned_close(
    data: dict[str, pd.DataFrame],
    codes: list[str],
) -> pd.DataFrame:
    """把多只 ETF 的 close 序列按日期 inner join 对齐，返回 DataFrame[code]。"""
    series_list = []
    for c in codes:
        if c not in data:
            continue
        s = pd.Series(
            data[c]["close"].values,
            index=pd.to_datetime(data[c]["date"]),
            name=c,
        )
        series_list.append(s)
    if not series_list:
        return pd.DataFrame()
    out = pd.concat(series_list, axis=1, join="inner")
    out.columns = [s.name for s in series_list]
    return out


def filter_universe_for_min_history(
    closes: pd.DataFrame,
    min_days: int = 250,
) -> tuple[pd.DataFrame, list[str]]:
    """剔除上市 < min_days 的标的（按每列的非 NaN 计数判断）。"""
    if closes.empty:
        return closes, []
    keep = [c for c in closes.columns if closes[c].notna().sum() >= min_days]
    dropped = [c for c in closes.columns if c not in keep]
    return closes[keep], dropped


def load_index_close(
    index_csv: Path | None,
    fallback_codes: list[str] | None = None,
    data_dir: Path | None = None,
) -> tuple[pd.Series, str]:
    """读沪深300指数 close 序列。

    优先级：
    1. index_csv 存在 → 读它（用户自备）
    2. 否则回落 data_dir/510300.SH.csv（ETF 代理；与指数相关系数 > 0.999）

    返回 (close_series, source_label)。
    """
    if index_csv is not None and index_csv.exists():
        df = pd.read_csv(index_csv)
        if "close" not in df.columns:
            raise ValueError(f"index_csv {index_csv} 缺少 'close' 列")
        s = pd.Series(
            df["close"].values,
            index=pd.to_datetime(df["date"]),
            name="000300.SH",
        )
        return s.sort_index(), index_csv.name

    if fallback_codes is None:
        fallback_codes = [DEFAULT_INDEX_PROXY]
    if data_dir is None:
        data_dir = Path(__file__).parent / ".cache" / "klines"

    for code in fallback_codes:
        path = data_dir / f"{code}.csv"
        if path.exists():
            df = pd.read_csv(path)
            if df.empty or "close" not in df.columns:
                continue
            s = pd.Series(
                df["close"].values,
                index=pd.to_datetime(df["date"]),
                name=code,
            )
            print(
                f"[INFO] index_csv 不存在；回落 ETF 代理 {code}（与沪深300指数相关系数 > 0.999）。"
            )
            return s.sort_index(), f"{code}_proxy"

    raise FileNotFoundError(
        f"未找到指数数据：{index_csv} 不存在，且 {fallback_codes} 全部缺失"
    )


# ---------------------------------------------------------------------------
# 块 2：因子计算
# ---------------------------------------------------------------------------
def calc_momentum(close: pd.Series, window: int = 120) -> pd.Series:
    """120 日动量 = close_t / close_{t-window} - 1。"""
    return close / close.shift(window) - 1.0


def calc_downside_vol(close: pd.Series, window: int = 60) -> pd.Series:
    """60 日下行波动率 = std(neg_returns) × sqrt(252)，只取 < 0 的日收益。"""
    log_ret = np.log(close / close.shift(1))
    neg = log_ret.where(log_ret < 0, np.nan)
    return neg.rolling(window=window, min_periods=max(10, window // 2)).std() * np.sqrt(252)


def calc_liquidity(amount: pd.Series, window: int = 20) -> pd.Series:
    """20 日日均成交额（元）。"""
    return amount.rolling(window=window, min_periods=5).mean()


def calc_ma_filter(close: pd.Series, window: int = 60) -> pd.Series:
    """close > MA(window) 的布尔 mask。"""
    ma = close.rolling(window=window, min_periods=window // 2).mean()
    return (close > ma).astype("boolean")


def calc_annualized_vol(close: pd.Series, window: int = 60) -> pd.Series:
    """60 日年化波动率（所有日收益）。"""
    log_ret = np.log(close / close.shift(1))
    return log_ret.rolling(window=window, min_periods=window // 2).std() * np.sqrt(252)


# ---------------------------------------------------------------------------
# 块 3：截面打分 + 权重构造
# ---------------------------------------------------------------------------
def cross_section_zscore(panel: pd.DataFrame) -> pd.DataFrame:
    """对每行做 Z-score（mean 0, std 1）；NaN 用 0 替代。"""
    if panel.empty:
        return panel
    row_mean = panel.mean(axis=1)
    row_std = panel.std(axis=1)
    safe_std = row_std.replace(0, np.nan)
    z = panel.sub(row_mean, axis=0).div(safe_std, axis=0)
    return z.fillna(0.0)


def compute_score(
    z_mom: pd.DataFrame,
    z_down_vol: pd.DataFrame,
    z_liq: pd.DataFrame,
    w_mom: float = W_MOM,
    w_down_vol: float = W_DOWN_VOL,
    w_liq: float = W_LIQ,
) -> pd.DataFrame:
    """综合得分 = w_mom·z_mom - w_down_vol·z_down_vol + w_liq·z_liq。"""
    return w_mom * z_mom - w_down_vol * z_down_vol + w_liq * z_liq


def apply_hard_filters(
    score: pd.DataFrame,
    ma_filter: pd.DataFrame,
    down_vol: pd.DataFrame,
    down_vol_cap: float = 0.40,
) -> pd.DataFrame:
    """跌破 60 日均线 → 分数 ×0.5；下行波动 > 阈值 → 分数置 NaN（剔除）。"""
    if score.empty:
        return score
    out = score.copy()
    below_ma = ma_filter.reindex(index=out.index, columns=out.columns).fillna(True).astype(bool) == False
    out = out.where(~below_ma, out * 0.5)
    high_vol = down_vol.reindex(index=out.index, columns=out.columns) > down_vol_cap
    out = out.where(~high_vol, np.nan)
    return out


def volatility_inverse_weight(
    score: pd.DataFrame,
    vol: pd.DataFrame,
    min_vol: float = 0.05,
) -> pd.DataFrame:
    """波动率倒数加权：wi ∝ score_i / vol_i。归一化到 sum=1。"""
    if score.empty:
        return score.copy()

    pos_score = score.clip(lower=0.0).fillna(0.0)
    safe_vol = vol.reindex(index=score.index, columns=score.columns).clip(lower=min_vol)
    raw = pos_score / safe_vol
    row_sum = raw.sum(axis=1).replace(0, np.nan)
    weights = raw.div(row_sum, axis=0).fillna(0.0)
    return weights


def cap_weights(weights: pd.DataFrame, max_w: float = 0.40, tol: float = 1e-6) -> pd.DataFrame:
    """单标的权重上限；超额按比例摊薄到其他未超限标的（迭代）。"""
    if weights.empty:
        return weights.copy()

    w = weights.copy()
    for _ in range(20):
        over = w > max_w + tol
        if not over.any().any():
            break
        excess = (w - max_w).where(over, 0.0).sum(axis=1)
        w = w.where(~over, max_w)
        safe_mask = ~over
        safe_sum = w.where(safe_mask, 0.0).sum(axis=1).replace(0, np.nan)
        w_safe = w.where(safe_mask, 0.0)
        share = w_safe.div(safe_sum, axis=0).fillna(0.0)
        w = w + share.mul(excess, axis=0)
    row_sum = w.sum(axis=1).replace(0, np.nan)
    w = w.div(row_sum, axis=0).fillna(0.0)
    return w


# ---------------------------------------------------------------------------
# 块 4：熊市择时
# ---------------------------------------------------------------------------
def compute_bear_scale(
    index_close: pd.Series,
    etf_closes: pd.DataFrame,
    trigger1_threshold: int = 2,
    vol_threshold: float = 0.25,
    ma_window: int = 120,
) -> pd.Series:
    """3 触发 → scale。

    - 触发1：池内站上 60 日均线数量 ≤ trigger1_threshold
    - 触发2：指数 60 日年化波动率 > vol_threshold
    - 触发3：指数收盘 < MA(ma_window)

    Scale 映射：
    - 0 触发：1.0
    - 1 触发：0.6
    - 2 触发：0.4
    - 3 触发：0.3
    """
    if index_close.empty:
        return pd.Series(dtype=float)

    if etf_closes is not None and not etf_closes.empty:
        ma60 = etf_closes.rolling(window=60, min_periods=30).mean()
        above_ma = etf_closes > ma60
        count_above = above_ma.sum(axis=1)
    else:
        count_above = pd.Series(np.nan, index=index_close.index)

    trigger1 = (count_above <= trigger1_threshold).astype(int)
    trigger1 = trigger1.reindex(index_close.index).ffill().fillna(0).astype(int)

    log_ret = np.log(index_close / index_close.shift(1))
    vol60 = log_ret.rolling(window=60, min_periods=30).std() * np.sqrt(252)
    trigger2 = (vol60 > vol_threshold).astype(int)

    ma_long = index_close.rolling(window=ma_window, min_periods=ma_window // 2).mean()
    trigger3 = (index_close < ma_long).astype(int)

    total = (trigger1 + trigger2 + trigger3).astype(int)

    scale_map = {0: 1.0, 1: 0.6, 2: 0.4, 3: 0.3}
    scale = total.map(scale_map).astype(float)
    scale.index = index_close.index
    return scale


# ---------------------------------------------------------------------------
# 块 5：回测主循环
# ---------------------------------------------------------------------------
@dataclass
class TradeRow:
    date: date
    code: str
    action: str
    weight_before: float
    weight_after: float
    weight_chg: float
    cost_bps: float


@dataclass
class BacktestResult:
    nav: pd.Series
    benchmark_nav: pd.Series
    weight_history: pd.DataFrame
    trade_log: pd.DataFrame
    turnover: pd.Series
    bear_scale: pd.Series
    rebal_dates: list[date]
    target_rebal_dates: list[date]
    score_panel: pd.DataFrame
    forward_returns: pd.DataFrame


def monthly_rebalance_dates(
    trading_days: Iterable[pd.Timestamp],
    start: pd.Timestamp,
) -> list[pd.Timestamp]:
    """取每月最后一个交易日作为调仓日。"""
    days = sorted({d for d in trading_days if d >= start})
    if not days:
        return []
    by_month: dict[tuple[int, int], pd.Timestamp] = {}
    for d in days:
        key = (d.year, d.month)
        if key not in by_month or d > by_month[key]:
            by_month[key] = d
    return sorted(by_month.values())


def build_factor_panels(
    etf_data: dict[str, pd.DataFrame],
    codes: list[str],
    mom_window: int,
    down_vol_window: int,
    liq_window: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """生成因子面板（行=日期，列=ETF）。"""
    closes_dict: dict[str, pd.Series] = {}
    amount_dict: dict[str, pd.Series] = {}
    for c in codes:
        if c not in etf_data:
            continue
        df = etf_data[c]
        s_close = pd.Series(df["close"].values, index=pd.to_datetime(df["date"]))
        s_amt = pd.Series(df["amount"].values, index=pd.to_datetime(df["date"])) if "amount" in df.columns else pd.Series(0.0, index=s_close.index)
        closes_dict[c] = s_close
        amount_dict[c] = s_amt

    close_panel = pd.concat(closes_dict, axis=1) if closes_dict else pd.DataFrame()
    close_panel.columns = list(closes_dict.keys())
    amount_panel = pd.concat(amount_dict, axis=1) if amount_dict else pd.DataFrame()
    amount_panel.columns = list(amount_dict.keys())

    mom = close_panel.apply(lambda s: calc_momentum(s, mom_window))
    down_vol = close_panel.apply(lambda s: calc_downside_vol(s, down_vol_window))
    liq = amount_panel.apply(lambda s: calc_liquidity(s, liq_window))
    vol_total = close_panel.apply(lambda s: calc_annualized_vol(s, down_vol_window))
    ma_f = close_panel.apply(lambda s: calc_ma_filter(s, 60))

    return mom, down_vol, liq, vol_total, ma_f, close_panel


def backtest_loop(
    etf_data: dict[str, pd.DataFrame],
    bond_data: pd.DataFrame,
    index_close: pd.Series,
    start_date: date,
    end_date: date,
    codes: list[str] = None,
    mom_window: int = 120,
    down_vol_window: int = 60,
    liq_window: int = 20,
    vol_window: int = 60,
    max_weight: float = 0.40,
    rebal_threshold: float = 0.05,
    slippage_bps: float = 15.0,
    bear_trigger1: int = 2,
    bear_vol_th: float = 0.25,
    bear_ma_window: int = 120,
) -> BacktestResult:
    codes = codes or EQUITY_ETFS

    mom, down_vol, liq, vol_total, ma_f, close_panel = build_factor_panels(
        etf_data, codes, mom_window, down_vol_window, liq_window
    )

    z_mom = cross_section_zscore(mom)
    z_dv = cross_section_zscore(down_vol)
    z_lq = cross_section_zscore(liq)
    score = compute_score(z_mom, z_dv, z_lq)
    score = apply_hard_filters(score, ma_f, down_vol)

    vol_w = volatility_inverse_weight(score, vol_total)
    target_weights_eq = cap_weights(vol_w, max_w=max_weight)

    bond_close = pd.Series(
        bond_data["close"].values,
        index=pd.to_datetime(bond_data["date"]),
        name=BOND_ETF,
    ) if bond_data is not None and not bond_data.empty else pd.Series(dtype=float)

    bear_scale = compute_bear_scale(
        index_close, close_panel, bear_trigger1, bear_vol_th, bear_ma_window
    )

    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    all_days = close_panel.index
    trading_days = [d for d in all_days if start_ts <= d <= end_ts]
    target_rebal = monthly_rebalance_dates(all_days, start_ts)
    target_rebal = [d for d in target_rebal if d <= end_ts]

    nav_index = [d for d in all_days if start_ts <= d <= end_ts]
    if not nav_index:
        raise ValueError("样本期内无数据")

    bond_aligned = bond_close.reindex(close_panel.index).ffill()
    full_close = close_panel.copy()
    full_close[BOND_ETF] = bond_aligned
    daily_ret = np.log(full_close / full_close.shift(1)).fillna(0.0)

    nav = pd.Series(1.0, index=nav_index, dtype=float)
    bench_close = index_close.reindex(nav_index).ffill()
    bench_ret = np.log(bench_close / bench_close.shift(1)).fillna(0.0)
    # 几何复利：exp(sum of log returns) ≈ cumprod of simple returns
    benchmark_nav = np.exp(bench_ret.cumsum())
    benchmark_nav.iloc[0] = 1.0

    weight_history_rows: list[dict] = []
    trade_rows: list[TradeRow] = []
    turnover_list: list[tuple[pd.Timestamp, float]] = []
    executed_rebal: list[date] = []

    target_rebal_set = set(target_rebal)
    prev_w = pd.Series(0.0, index=full_close.columns)
    slip = slippage_bps / 10000.0
    first_rebal_skipped = False

    for day_idx, today in enumerate(nav_index):
        current_w = prev_w.copy()

        if today in target_rebal_set:
            if today in target_weights_eq.index:
                eq_w_target = target_weights_eq.loc[today].reindex(full_close.columns).fillna(0.0)
            else:
                eq_w_target = pd.Series(0.0, index=full_close.columns)

            scale_today = float(bear_scale.reindex([today]).ffill().iloc[0]) if not bear_scale.empty else 1.0
            if np.isnan(scale_today):
                scale_today = 1.0

            eq_only = eq_w_target.drop(BOND_ETF, errors="ignore")
            if eq_only.sum() > 0:
                eq_only = (eq_only / eq_only.sum()) * scale_today
            desired_w = eq_only.copy()
            desired_w[BOND_ETF] = max(0.0, 1.0 - eq_only.sum())

            diff = (desired_w - current_w).abs()
            need_rebal = bool((diff > rebal_threshold).any())

            if need_rebal:
                old_w = current_w.copy()
                is_initial_buildup = (not first_rebal_skipped) and (old_w.abs().sum() == 0)
                if is_initial_buildup:
                    first_rebal_skipped = True
                    weight_history_rows.append({"date": today.date(), **desired_w.to_dict()})
                    current_w = desired_w
                else:
                    weight_history_rows.append({"date": today.date(), **desired_w.to_dict()})
                    executed_rebal.append(today.date())

                    for code in desired_w.index:
                        w_before = float(old_w.get(code, 0.0))
                        w_after = float(desired_w.get(code, 0.0))
                        chg = w_after - w_before
                        if abs(chg) > rebal_threshold:
                            if chg > 0:
                                trade_rows.append(TradeRow(
                                    date=today.date(), code=code, action="buy",
                                    weight_before=w_before, weight_after=w_after,
                                    weight_chg=chg, cost_bps=slippage_bps,
                                ))
                            else:
                                trade_rows.append(TradeRow(
                                    date=today.date(), code=code, action="sell",
                                    weight_before=w_before, weight_after=w_after,
                                    weight_chg=chg, cost_bps=slippage_bps,
                                ))
                            nav.iloc[day_idx] *= (1.0 - slip)
                    turnover_today = float(np.abs(desired_w - old_w).sum() / 2.0)
                    turnover_list.append((today, turnover_today))
                    current_w = desired_w
            else:
                weight_history_rows.append({"date": today.date(), **desired_w.to_dict()})

        if day_idx > 0:
            w_eff = current_w.reindex(daily_ret.columns).fillna(0.0)
            r_today = daily_ret.loc[today].reindex(w_eff.index).fillna(0.0)
            port_ret = float((w_eff * r_today).sum())
            nav.iloc[day_idx] = nav.iloc[day_idx - 1] * np.exp(port_ret)

        prev_w = current_w

    weight_history = pd.DataFrame(weight_history_rows).set_index("date") if weight_history_rows else pd.DataFrame()
    trade_log_df = pd.DataFrame([{
        "date": t.date, "code": t.code, "action": t.action,
        "weight_before": t.weight_before, "weight_after": t.weight_after,
        "weight_chg": t.weight_chg, "cost_bps": t.cost_bps,
    } for t in trade_rows])

    turnover = pd.Series(
        [v for _, v in turnover_list],
        index=pd.DatetimeIndex([t for t, _ in turnover_list]),
        name="turnover",
    )

    forward_returns = close_panel.pct_change(periods=21).shift(-21)

    return BacktestResult(
        nav=nav,
        benchmark_nav=benchmark_nav,
        weight_history=weight_history,
        trade_log=trade_log_df,
        turnover=turnover,
        bear_scale=bear_scale.reindex(nav_index).ffill().fillna(1.0),
        rebal_dates=executed_rebal,
        target_rebal_dates=[d.date() for d in target_rebal],
        score_panel=score,
        forward_returns=forward_returns,
    )


# ---------------------------------------------------------------------------
# 块 6：业绩指标
# ---------------------------------------------------------------------------
def performance_metrics(
    nav: pd.Series,
    benchmark_nav: pd.Series,
    risk_free: float = 0.02,
) -> dict:
    """用 empyrical 算全套指标。risk_free 是年化利率（默认 2%）。"""
    if nav.empty:
        return {}
    nav = nav.dropna()
    benchmark_nav = benchmark_nav.reindex(nav.index).ffill().dropna()

    log_ret = np.log(nav / nav.shift(1)).dropna()
    bench_log = np.log(benchmark_nav / benchmark_nav.shift(1)).dropna()

    simple_ret = np.exp(log_ret) - 1.0
    simple_bench = np.exp(bench_log) - 1.0

    aligned = pd.concat([simple_ret, simple_bench], axis=1, join="inner").dropna()
    aligned.columns = ["strategy", "benchmark"]

    if len(aligned) < 2:
        return {}

    rf_daily = risk_free / 252.0

    metrics = {
        "annual_return": float(ep.annual_return(aligned["strategy"])),
        "annual_volatility": float(ep.annual_volatility(aligned["strategy"])),
        "sharpe_ratio": float(ep.sharpe_ratio(aligned["strategy"], risk_free=rf_daily)),
        "sortino_ratio": float(ep.sortino_ratio(aligned["strategy"], required_return=rf_daily)),
        "max_drawdown": float(ep.max_drawdown(aligned["strategy"])),
        "calmar_ratio": float(ep.calmar_ratio(aligned["strategy"])),
        "stability": float(ep.stability_of_timeseries(aligned["strategy"])),
        "omega_ratio": float(ep.omega_ratio(aligned["strategy"], risk_free=rf_daily)),
        "skew": float(stats.skew(aligned["strategy"])),
        "kurtosis": float(stats.kurtosis(aligned["strategy"])),
        "tail_ratio": float(ep.tail_ratio(aligned["strategy"])),
    }
    try:
        alpha, beta = ep.alpha_beta(aligned["strategy"], aligned["benchmark"], risk_free=rf_daily)
        metrics["alpha"] = float(alpha)
        metrics["beta"] = float(beta)
    except Exception:  # noqa: BLE001
        metrics["alpha"] = float("nan")
        metrics["beta"] = float("nan")

    metrics["bench_annual_return"] = float(ep.annual_return(aligned["benchmark"]))
    metrics["bench_annual_volatility"] = float(ep.annual_volatility(aligned["benchmark"]))
    metrics["bench_sharpe_ratio"] = float(ep.sharpe_ratio(aligned["benchmark"], risk_free=rf_daily))
    metrics["bench_max_drawdown"] = float(ep.max_drawdown(aligned["benchmark"]))
    metrics["bench_calmar_ratio"] = float(ep.calmar_ratio(aligned["benchmark"]))
    metrics["excess_annual_return"] = metrics["annual_return"] - metrics["bench_annual_return"]

    return metrics


# ---------------------------------------------------------------------------
# 块 7：场景分段
# ---------------------------------------------------------------------------
def classify_market_regime(monthly_returns: pd.Series) -> pd.Series:
    """按月收益划分 regime：牛 > +5%，熊 < -5%，否则震荡。"""
    regime = pd.Series("震荡", index=monthly_returns.index)
    regime[monthly_returns > 0.05] = "牛"
    regime[monthly_returns < -0.05] = "熊"
    return regime


def regime_breakdown(
    nav: pd.Series,
    regime: pd.Series,
    benchmark_nav: pd.Series,
) -> pd.DataFrame:
    """按 regime 分组算年化、最大回撤、夏普。"""
    rows = []
    nav = nav.dropna()
    benchmark_nav = benchmark_nav.reindex(nav.index).ffill()

    nav_month_end = nav.resample("M").last()
    bm_month_end = benchmark_nav.resample("M").last()
    regime_aligned = regime.reindex(nav_month_end.index).ffill()

    for r in ["牛", "震荡", "熊"]:
        mask = regime_aligned == r
        if mask.sum() == 0:
            continue
        sel_dates = nav_month_end.index[mask]
        if len(sel_dates) < 2:
            continue
        s = nav_month_end[sel_dates]
        b = bm_month_end[sel_dates]
        s_ret = np.log(s / s.shift(1)).dropna()
        b_ret = np.log(b / b.shift(1)).dropna()
        if s_ret.empty:
            continue
        rows.append({
            "regime": r,
            "months": int(len(s_ret)),
            "strategy_annual_ret": float(np.exp(s_ret.mean() * 12) - 1) if len(s_ret) > 0 else np.nan,
            "benchmark_annual_ret": float(np.exp(b_ret.mean() * 12) - 1) if len(b_ret) > 0 else np.nan,
            "strategy_sharpe": float(ep.sharpe_ratio(np.exp(s_ret) - 1)) if len(s_ret) > 0 else np.nan,
            "benchmark_sharpe": float(ep.sharpe_ratio(np.exp(b_ret) - 1)) if len(b_ret) > 0 else np.nan,
            "strategy_max_dd": float(ep.max_drawdown(np.exp(s_ret) - 1)),
        })
    return pd.DataFrame(rows)


def factor_ic_time_series(
    score_panel: pd.DataFrame,
    forward_returns: pd.DataFrame,
    rebal_dates: list[date] | None = None,
) -> pd.DataFrame:
    """每个调仓日算综合 score 与下一期收益的 Pearson + Spearman IC。"""
    if score_panel.empty or forward_returns.empty:
        return pd.DataFrame()

    if rebal_dates is not None:
        rebal_ts = pd.DatetimeIndex([pd.Timestamp(d) for d in rebal_dates])
        score_panel = score_panel.reindex(rebal_ts)
        forward_returns = forward_returns.reindex(rebal_ts)

    common_dates = score_panel.index.intersection(forward_returns.index)
    common_cols = score_panel.columns.intersection(forward_returns.columns)
    if len(common_dates) == 0 or len(common_cols) < 3:
        return pd.DataFrame()

    rows = []
    for d in common_dates:
        s = score_panel.loc[d, common_cols].dropna()
        r = forward_returns.loc[d, common_cols].dropna()
        aligned_idx = s.index.intersection(r.index)
        s = s.loc[aligned_idx]
        r = r.loc[aligned_idx]
        if len(aligned_idx) < 3:
            continue
        if s.std() == 0 or r.std() == 0:
            continue
        pearson_r, _ = stats.pearsonr(s.values, r.values)
        spearman_r, _ = stats.spearmanr(s.values, r.values)
        rows.append({"date": d, "pearson_ic": pearson_r, "spearman_ic": spearman_r, "n": len(aligned_idx)})
    return pd.DataFrame(rows).set_index("date") if rows else pd.DataFrame()


def rolling_ic_ir(ic_series: pd.Series, window: int = 12) -> tuple[pd.Series, pd.Series]:
    """滚动 IC 均值 + IR = mean / std。"""
    if ic_series.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    rolling_mean = ic_series.rolling(window=window, min_periods=3).mean()
    rolling_std = ic_series.rolling(window=window, min_periods=3).std()
    ir = (rolling_mean / rolling_std).replace([np.inf, -np.inf], np.nan)
    return rolling_mean, ir


# ---------------------------------------------------------------------------
# 块 8：出图
# ---------------------------------------------------------------------------
def _savefig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_nav_curve(
    nav: pd.Series,
    benchmark_nav: pd.Series,
    bear_scale: pd.Series,
    save_path: Path,
) -> None:
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12, 7.5), sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )
    ax1.plot(nav.index, nav.values, color="#1f77b4", linewidth=1.5, label="策略 NAV")
    ax1.plot(benchmark_nav.index, benchmark_nav.values, color="#888888", linewidth=1.3,
             linestyle="--", label="沪深300 buy-and-hold")
    ax1.set_yscale("log")
    ax1.set_ylabel("净值（log）")
    ax1.legend(loc="upper left", fontsize=10)
    ax1.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)

    if not bear_scale.empty:
        in_bear = bear_scale < 1.0
        ax1.fill_between(
            bear_scale.index, 1e-3, 1e3,
            where=in_bear, color="grey", alpha=0.18, label="熊市 scale<1",
        )
        ax1.legend(loc="upper left", fontsize=10)

    ax1.set_title("策略 NAV 曲线（log 尺度）", fontsize=12)

    ax2.plot(bear_scale.index, bear_scale.values, color="#d62728", linewidth=1.0, label="bear scale")
    ax2.axhline(1.0, color="black", linewidth=0.6, linestyle=":")
    ax2.set_ylim(0.2, 1.05)
    ax2.set_ylabel("scale")
    ax2.set_xlabel("日期")
    ax2.legend(loc="lower left", fontsize=9)
    ax2.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate()

    _savefig(fig, save_path)


def plot_drawdown(nav: pd.Series, save_path: Path) -> None:
    nav = nav.dropna()
    peak = nav.cummax()
    dd = nav / peak - 1.0

    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.fill_between(dd.index, dd.values, 0, color="#d62728", alpha=0.5, label="回撤")
    ax.plot(dd.index, dd.values, color="#d62728", linewidth=0.8)
    ax.set_ylabel("回撤")
    ax.set_xlabel("日期")
    ax.set_title("策略回撤曲线（underwater plot）", fontsize=12)
    ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x * 100:.0f}%"))
    fig.autofmt_xdate()
    _savefig(fig, save_path)


def plot_monthly_returns_heatmap(nav: pd.Series, save_path: Path) -> None:
    nav = nav.dropna()
    monthly = nav.resample("M").last()
    monthly_ret = monthly.pct_change().dropna()
    if monthly_ret.empty:
        return
    df = pd.DataFrame({
        "year": monthly_ret.index.year,
        "month": monthly_ret.index.month,
        "ret": monthly_ret.values,
    })
    pivot = df.pivot(index="year", columns="month", values="ret")
    month_labels = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12"]
    pivot.columns = [month_labels[c - 1] for c in pivot.columns]

    fig, ax = plt.subplots(figsize=(11, 0.5 + 0.55 * len(pivot)))
    sns.heatmap(
        pivot, annot=True, fmt=".1%", center=0.0, cmap="RdYlGn",
        linewidths=0.5, cbar=False, ax=ax,
        annot_kws={"size": 9},
    )
    ax.set_title("月度收益热力图（%）", fontsize=12)
    ax.set_xlabel("月份")
    ax.set_ylabel("年份")
    fig.tight_layout()
    _savefig(fig, save_path)


def plot_factor_ic(
    ic_df: pd.DataFrame,
    rolling_mean: pd.Series,
    rolling_ir: pd.Series,
    save_path: Path,
) -> None:
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12, 6.5), sharex=True,
        gridspec_kw={"height_ratios": [1, 1]},
    )
    if not ic_df.empty:
        ax1.bar(ic_df.index, ic_df["pearson_ic"].values, color="#1f77b4", alpha=0.7, width=20, label="Pearson IC")
        ax1.bar(ic_df.index, ic_df["spearman_ic"].values, color="#ff7f0e", alpha=0.5, width=10, label="Spearman IC")
        ax1.axhline(0, color="black", linewidth=0.5)
        ax1.set_ylabel("IC")
        ax1.legend(loc="upper left", fontsize=9)
        ax1.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)

    if not rolling_mean.empty:
        ax2.plot(rolling_mean.index, rolling_mean.values, color="#2ca02c", linewidth=1.4, label="滚动 12m IC 均值")
        if not rolling_ir.empty:
            ax2.plot(rolling_ir.index, rolling_ir.values, color="#d62728", linewidth=1.0, label="滚动 12m IR")
        ax2.axhline(0, color="black", linewidth=0.5)
        ax2.set_ylabel("IC 均值 / IR")
        ax2.legend(loc="upper left", fontsize=9)
        ax2.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)

    ax1.set_title("综合得分因子 IC（月度）", fontsize=12)
    ax2.set_xlabel("日期")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate()
    _savefig(fig, save_path)


def plot_weight_heatmap(weight_history: pd.DataFrame, save_path: Path) -> None:
    if weight_history.empty:
        return
    fig, ax = plt.subplots(figsize=(13, 0.45 * len(weight_history) + 1.5))
    sns.heatmap(
        weight_history.fillna(0.0), annot=True, fmt=".2f", center=0.2,
        cmap="YlGnBu", linewidths=0.4, cbar=True, ax=ax,
        annot_kws={"size": 8}, cbar_kws={"label": "权重"},
    )
    ax.set_title("调仓日目标权重热力图", fontsize=12)
    ax.set_xlabel("标的")
    ax.set_ylabel("调仓日")
    fig.tight_layout()
    _savefig(fig, save_path)


def plot_turnover(turnover: pd.Series, save_path: Path) -> None:
    if turnover.empty:
        return
    monthly_turn = turnover.resample("M").sum()
    rolling_mean = monthly_turn.rolling(window=6, min_periods=1).mean()

    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.bar(monthly_turn.index, monthly_turn.values, color="#1f77b4", alpha=0.7, width=20, label="月度换手率")
    ax.plot(rolling_mean.index, rolling_mean.values, color="#d62728", linewidth=1.4, label="6m 滚动均值")
    ax.set_ylabel("换手率（双边）")
    ax.set_xlabel("月份")
    ax.set_title("调仓换手率时序", fontsize=12)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x * 100:.0f}%"))
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate()
    _savefig(fig, save_path)


def plot_regime_breakdown(regime_df: pd.DataFrame, save_path: Path) -> None:
    if regime_df.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(regime_df))
    width = 0.35
    ax.bar(x - width / 2, regime_df["strategy_annual_ret"] * 100, width,
           color="#1f77b4", label="策略")
    ax.bar(x + width / 2, regime_df["benchmark_annual_ret"] * 100, width,
           color="#888888", label="沪深300")
    ax.set_xticks(x)
    ax.set_xticklabels(regime_df["regime"])
    ax.set_ylabel("年化收益 (%)")
    ax.set_title("牛 / 震 / 熊 分段年化收益", fontsize=12)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.legend(fontsize=10)
    ax.grid(True, axis="y", alpha=0.3, linestyle="--", linewidth=0.5)
    fig.tight_layout()
    _savefig(fig, save_path)


# ---------------------------------------------------------------------------
# 块 9：报告输出
# ---------------------------------------------------------------------------
def write_report(
    result: BacktestResult,
    metrics: dict,
    regime_df: pd.DataFrame,
    ic_df: pd.DataFrame,
    output_dir: Path,
    args: argparse.Namespace,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "metrics.txt").open("w", encoding="utf-8") as f:
        f.write("== A股宽基ETF多因子轮动策略 回测报告 ==\n\n")
        f.write(f"回测区间：{args.start_date} -> {args.end_date}\n")
        f.write(f"标的池：{' / '.join(EQUITY_ETFS)}\n")
        f.write(f"国债：{BOND_ETF}\n")
        f.write(f"指数代理：{args.index_proxy}\n\n")
        f.write(f"调仓候选日：{len(result.target_rebal_dates)} 个，实际触发 {len(result.rebal_dates)} 次\n\n")
        f.write("--- 策略指标 ---\n")
        for k in ["annual_return", "annual_volatility", "sharpe_ratio", "sortino_ratio",
                  "max_drawdown", "calmar_ratio", "stability", "omega_ratio",
                  "skew", "kurtosis", "tail_ratio", "alpha", "beta"]:
            f.write(f"  {k:25s}: {metrics.get(k, float('nan')):>10.4f}\n")
        f.write("\n--- 沪深300 基准 ---\n")
        for k in ["bench_annual_return", "bench_annual_volatility",
                  "bench_sharpe_ratio", "bench_max_drawdown", "bench_calmar_ratio"]:
            f.write(f"  {k:25s}: {metrics.get(k, float('nan')):>10.4f}\n")
        f.write(f"\n  超额年化收益: {metrics.get('excess_annual_return', float('nan')):.4f}\n")

    if not result.trade_log.empty:
        result.trade_log.to_csv(output_dir / "trade_log.csv", index=False)

    if not result.weight_history.empty:
        result.weight_history.to_csv(output_dir / "weights.csv")

    if not result.turnover.empty:
        result.turnover.to_csv(output_dir / "turnover.csv", header=["turnover"])

    if not ic_df.empty:
        ic_df.to_csv(output_dir / "factor_ic.csv")

    if not regime_df.empty:
        regime_df.to_csv(output_dir / "regime_breakdown.csv", index=False)

    if not result.bear_scale.empty:
        result.bear_scale.to_csv(output_dir / "bear_scale.csv", header=["scale"])

    print(f"[INFO] CSV 报告写入：{output_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="A股宽基ETF多因子轮动策略回测")
    p.add_argument("--start-date", default="2021-04-01", help="回测起始 YYYY-MM-DD（避开冷启动）")
    p.add_argument("--end-date", default="2026-08-12", help="回测截止 YYYY-MM-DD")
    p.add_argument("--data-dir", type=Path, default=Path(__file__).parent / ".cache" / "klines",
                   help="ETF CSV 目录")
    p.add_argument("--index-csv", type=Path, default=None,
                   help="沪深300指数 CSV 路径；不存在则回落 ETF 代理")
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                   help="输出目录")
    p.add_argument("--slippage-bps", type=float, default=15.0, help="单边滑点（基点）")
    p.add_argument("--rebal-threshold", type=float, default=0.05, help="5%% 调仓阈值")
    p.add_argument("--max-weight", type=float, default=0.40, help="单标的权重上限")
    p.add_argument("--mom-window", type=int, default=120, help="动量回看窗口")
    p.add_argument("--down-vol-window", type=int, default=60, help="下行波动窗口")
    p.add_argument("--liq-window", type=int, default=20, help="流动性窗口")
    p.add_argument("--vol-window", type=int, default=60, help="加权用的总波动窗口")
    p.add_argument("--bear-trigger1", type=int, default=2,
                   help="触发1：池内站上 60 日均线数量 <= N")
    p.add_argument("--bear-vol", type=float, default=0.25,
                   help="触发2：沪深300 60 日年化波动率 > X")
    p.add_argument("--bear-ma", type=int, default=120,
                   help="触发3：沪深300 收盘 < MA_N")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    start = datetime.strptime(args.start_date, "%Y-%m-%d").date()
    end = datetime.strptime(args.end_date, "%Y-%m-%d").date()
    if start >= end:
        print(f"ERROR: start_date({start}) >= end_date({end})", file=sys.stderr)
        return 2

    print(f"== 回测区间 {start} -> {end} ==")

    etf_data = load_universe(args.data_dir, EQUITY_ETFS + [BOND_ETF], start, end)
    if len(etf_data) < len(EQUITY_ETFS):
        print(f"[WARN] 仅 {len(etf_data)} 只标的落盘（期望 {len(EQUITY_ETFS) + 1}）",
              file=sys.stderr)

    closes_all = load_aligned_close(etf_data, EQUITY_ETFS)
    closes_eq, dropped = filter_universe_for_min_history(closes_all, min_days=250)
    if dropped:
        print(f"[INFO] 剔除上市 < 250 日的标的：{dropped}")
    eq_codes = closes_eq.columns.tolist()

    bond_data = etf_data.get(BOND_ETF)
    if bond_data is None or bond_data.empty:
        print(f"[WARN] 国债 ETF {BOND_ETF} 缺失；熊市防御失效（剩余资金空仓）", file=sys.stderr)

    index_close, index_label = load_index_close(args.index_csv, data_dir=args.data_dir)
    index_close = index_close.loc[(index_close.index >= pd.Timestamp(start)) &
                                   (index_close.index <= pd.Timestamp(end))]

    etf_data_filt = {c: df for c, df in etf_data.items() if c in eq_codes + [BOND_ETF]}

    print(f"[INFO] 标的：{eq_codes}, 指数：{index_label}")

    result = backtest_loop(
        etf_data_filt,
        bond_data if bond_data is not None else pd.DataFrame(),
        index_close,
        start_date=start,
        end_date=end,
        codes=eq_codes,
        mom_window=args.mom_window,
        down_vol_window=args.down_vol_window,
        liq_window=args.liq_window,
        vol_window=args.vol_window,
        max_weight=args.max_weight,
        rebal_threshold=args.rebal_threshold,
        slippage_bps=args.slippage_bps,
        bear_trigger1=args.bear_trigger1,
        bear_vol_th=args.bear_vol,
        bear_ma_window=args.bear_ma,
    )
    args.index_proxy = index_label
    print(f"[INFO] NAV 长度：{len(result.nav)} 日 | 实际调仓：{len(result.rebal_dates)} 次 | "
          f"换手率均值（仅调仓日）：{result.turnover.mean() * 100:.1f}%")

    metrics = performance_metrics(result.nav, result.benchmark_nav)
    print("\n--- 策略指标 ---")
    for k in ["annual_return", "sharpe_ratio", "sortino_ratio", "calmar_ratio",
              "max_drawdown", "stability", "alpha", "beta"]:
        v = metrics.get(k, float("nan"))
        print(f"  {k:25s}: {v:.4f}")
    print("--- 基准 ---")
    for k in ["bench_annual_return", "bench_sharpe_ratio", "bench_max_drawdown",
              "bench_calmar_ratio", "excess_annual_return"]:
        v = metrics.get(k, float("nan"))
        print(f"  {k:25s}: {v:.4f}")

    monthly_nav = result.nav.resample("M").last()
    monthly_bench = result.benchmark_nav.reindex(monthly_nav.index).ffill()
    monthly_ret = monthly_nav.pct_change().dropna()
    monthly_bench_ret = monthly_bench.pct_change().dropna()
    regime = classify_market_regime(monthly_bench_ret)
    regime_df = regime_breakdown(result.nav, regime, result.benchmark_nav)
    print("\n--- 场景分段 ---")
    if not regime_df.empty:
        print(regime_df.to_string(index=False))

    ic_df = factor_ic_time_series(result.score_panel, result.forward_returns, result.rebal_dates)
    if not ic_df.empty:
        pearson_mean = ic_df["pearson_ic"].mean()
        spearman_mean = ic_df["spearman_ic"].mean()
        print(f"\n--- 因子 IC ---")
        print(f"  Pearson IC 均值 : {pearson_mean:+.4f}  (n={len(ic_df)})")
        print(f"  Spearman IC 均值: {spearman_mean:+.4f}")
        r_mean, r_ir = rolling_ic_ir(ic_df["pearson_ic"], window=12)
    else:
        pearson_mean = spearman_mean = float("nan")
        r_mean = pd.Series(dtype=float)
        r_ir = pd.Series(dtype=float)
        print("[WARN] 因子 IC 数据为空（样本可能太少）")

    output_dir = args.output_dir
    plot_nav_curve(result.nav, result.benchmark_nav, result.bear_scale,
                   output_dir / "nav.png")
    plot_drawdown(result.nav, output_dir / "drawdown.png")
    plot_monthly_returns_heatmap(result.nav, output_dir / "monthly_returns.png")
    plot_factor_ic(ic_df, r_mean, r_ir, output_dir / "ic.png")
    plot_weight_heatmap(result.weight_history, output_dir / "weights_heatmap.png")
    plot_turnover(result.turnover, output_dir / "turnover.png")
    plot_regime_breakdown(regime_df, output_dir / "regime.png")
    print(f"\n[INFO] PNG 图写入：{output_dir}")

    write_report(result, metrics, regime_df, ic_df, output_dir, args)

    print(f"\n== 完成。输出：{output_dir} ==")
    return 0


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        warnings.simplefilter("ignore", category=UserWarning)
        sys.exit(main())