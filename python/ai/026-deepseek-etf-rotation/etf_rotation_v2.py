#!/usr/bin/env python3
"""ETF 多因子轮动策略 v2（借鉴豆包文档 + 批判性验证）。

策略规格（按文档，参数全部可配置，默认取文档值）：
1. 因子（月末截面使用）：120 日动量（60%）、60 日下行波动率（负向 30%）、
   20 日流动性（10%），截面 z-score 加权打分；
2. 硬过滤：收盘价跌破 60 日均线剔除；下行波动 z 超阈值强制扣分；
3. 权重：score → softmax → 波动率倒数修正 → 单只上限 40% → 归一化；
4. 全局择时 scale：①站上 MA60 的宽基数量≤2；②510300 的 60 日年化波动>25%；
   ③510300 收盘 < MA120。按触发数分级 {0:1.0, 1:0.6, 2:0.45, 3:0.3}，权益仓位
   × scale，剩余国债补足；
5. 调仓：自然月月末 + 权重偏离>5% 才调（对比漂移后实际权重，路径依赖）；
6. 成本：佣金万1 + 滑点 0.1%/边（v1 用同口径合并费率对比，不改 v1 代码）。

批判性立场与数据局限：
- 文档因子权重/阈值均为无证据数字，本实现全部参数化，用 IC/分场景/滚动校验
  与 A/B 变体审判其宣称（"不牺牲收益降回撤"、"防御逻辑生效"等）；
- 数据无复权价：ETF 分红约 1%/年，动量被低估，方向一致（本数据 726 日中
  各 ETF 仅 1-2 次除息，影响有限）；
- 无 000300.SH 指数数据，用 510300 ETF 代理（跟踪误差极小）；
- 文档自注"部分内容可能由 AI 生成"，本质是未验证的讨论总结。

运行：
    /Users/huhao/.pyenv/versions/3.11.9/bin/python3.11 etf_rotation_v2.py [--save-csv]
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field, replace
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter

from etf_momentum_rotation import (
    BASELINE, GRIDLINE, KLINES_DIR, MUTED, OUTPUT_DIR, PRIMARY_INK,
    RISK_FREE_ANNUAL, SECONDARY_INK, STOCK_ETFS, SURFACE, TRADING_DAYS,
    TRESURY, ETF_COLORS, ETF_NAMES, _pad_cjk, backtest, benchmark_navs,
    compute_metrics, load_closes,
)

# ---------------------------------------------------------------- 配置常量
# 因子权重（文档 60/30/10，无证据的拍脑袋数字，IC 校验审判）
W_MOM, W_DOWNVOL, W_LIQ = 0.6, 0.3, 0.1
# 硬过滤
MA_FILTER_DAYS = 60
MIN_SCORABLE = 2                 # 可评分 ETF < 2 只 → 全仓国债（保守，与硬过滤哲学一致）
DOWNVOL_Z_CUT = 1.5              # 下行波动 z 超此值强制扣分（文档"超阈值扣分"的具体化）
DOWNVOL_PENALTY = 1.0            # 扣分幅度（与单个因子贡献同量级）
# 权重
SOFTMAX_TEMP = 1.0
MAX_WEIGHT = 0.4                 # 单只上限 40%（文档数字）
USE_VOL_WEIGHT = True            # 波动率倒数修正开关（A/B 用）
# 全局择时 scale（文档 0.3~0.6 的具体化；0.3 档在本样本从未命中）
SCALE_MAP: dict[int, float] = {0: 1.0, 1: 0.6, 2: 0.45, 3: 0.3}
VOL_TRIGGER_ANN = 0.25           # 沪深300 60 日年化波动触发线
BREADTH_TRIGGER = 2              # 站上 MA60 的宽基数量 ≤ 2
# 调仓与成本
REBAL_THRESHOLD = 0.05           # max|Δw| ≤ 5% 跳过调仓
COMMISSION = 1e-4
SLIPPAGE_PER_SIDE = 1e-3
COST_RATE = COMMISSION + SLIPPAGE_PER_SIDE   # 1.1e-3，v1/v2 同口径
# 校验
FWD_DAYS = 21
IC_MIN_PAIRS = 3
ROLL_WIN, ROLL_STEP = 252, 21

HS300_PROXY = "510300.SH"        # 沪深300 ETF 代理 000300 指数

V2_PURPLE = "#4a3aa7"            # v2 策略线色（dataviz slot 7，与 v1 蓝线同图已过校验）


@dataclass
class V2Config:
    """v2 全部可调参数，默认取文档值。"""
    w_mom: float = W_MOM
    w_downvol: float = W_DOWNVOL
    w_liq: float = W_LIQ
    ma_filter_days: int = MA_FILTER_DAYS
    min_scorable: int = MIN_SCORABLE
    downvol_z_cut: float = DOWNVOL_Z_CUT
    downvol_penalty: float = DOWNVOL_PENALTY
    softmax_temp: float = SOFTMAX_TEMP
    max_weight: float = MAX_WEIGHT
    use_vol_weight: bool = USE_VOL_WEIGHT
    scale_map: dict = field(default_factory=lambda: dict(SCALE_MAP))
    vol_trigger: float = VOL_TRIGGER_ANN
    breadth_trigger: int = BREADTH_TRIGGER
    rebal_threshold: float = REBAL_THRESHOLD
    cost_rate: float = COST_RATE


# ---------------------------------------------------------------- 数据与因子
def load_amounts(klines_dir: Path) -> pd.DataFrame:
    """读取全部 CSV 的成交额（f_liq 需要），镜像 v1 的 load_closes。

    union 索引；NaN = 未上市/停牌，下游按语义处理（因子 NaN → 不可评分）。
    """
    amounts = {}
    for f in sorted(klines_dir.glob("*.csv")):
        df = pd.read_csv(f, parse_dates=["date"], index_col="date")
        amounts[f.stem] = df["amount"]
    out = pd.DataFrame(amounts).sort_index()
    assert out.index.is_monotonic_increasing
    assert len(out) > 2000, f"行数 {len(out)} 异常（预期 ~2426）"
    nan_share = out.isna().to_numpy().sum() / out.size
    assert nan_share < 0.35, f"NaN 占比 {nan_share:.0%} 异常（预期仅未上市/停牌缺口）"
    return out


@dataclass
class Ctx:
    """日频预计算结果，引擎与 IC 共用。"""
    ret: pd.DataFrame
    f: dict[str, pd.DataFrame]           # mom / downvol / liq
    ma60: pd.DataFrame
    vol60: pd.DataFrame
    ma60_300: pd.Series
    ma120_300: pd.Series
    vol_ann300: pd.Series
    breadth: pd.Series


def precompute(closes: pd.DataFrame, amounts: pd.DataFrame) -> Ctx:
    ret = closes.pct_change(fill_method=None)
    f = {
        "mom": closes[STOCK_ETFS].pct_change(120, fill_method=None),
        # 下行波动率（Sortino 式半方差平方根）：惩罚"下跌频率+幅度"两个维度；
        # std(r[r<0]) 会丢弃下跌频次信息（5 天小跌与 25 天阴跌同值），弃用
        "downvol": (ret[STOCK_ETFS].clip(upper=0.0) ** 2).rolling(60).mean().pow(0.5),
        "liq": amounts[STOCK_ETFS].rolling(20).mean(),
    }
    ma60 = closes[STOCK_ETFS].rolling(60).mean()
    vol60 = ret[STOCK_ETFS].rolling(60).std()
    ma60_300 = closes[HS300_PROXY].rolling(60).mean()
    ma120_300 = closes[HS300_PROXY].rolling(120).mean()
    vol_ann300 = ret[HS300_PROXY].rolling(60).std() * np.sqrt(TRADING_DAYS)
    breadth = (closes[STOCK_ETFS] > ma60).sum(axis=1)
    return Ctx(ret, f, ma60, vol60, ma60_300, ma120_300, vol_ann300, breadth)


def month_end_dates(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """每月最后一个【实际交易日】。resample('ME') 的标签≠交易日（14/37 错位），
    必须取 .last().values 而非标签。"""
    return pd.DatetimeIndex(idx.to_series().resample("ME").last().values)


# ---------------------------------------------------------------- 权重管线
def softmax(score: pd.Series, temp: float) -> pd.Series:
    e = np.exp((score - score.max()) / temp)   # 减 max 数值稳定
    return e / e.sum()


def cap_weights(w: pd.Series, cap: float, max_iter: int = 20) -> pd.Series:
    """迭代 cap-and-redistribute：超限部分按当前权重比例再分配，与列序无关。"""
    w = w.copy()
    for _ in range(max_iter):
        excess = float((w - cap).clip(lower=0.0).sum())
        if excess <= 1e-12:
            break
        over = w > cap
        w.loc[over] = cap
        rest = w.loc[~over]
        if rest.sum() > 0:
            w.loc[~over] = rest + excess * rest / rest.sum()
    return w


def factor_cross_section(t: pd.Timestamp, closes: pd.DataFrame, ctx: Ctx,
                         cfg: V2Config) -> tuple[pd.Series | None, pd.Series]:
    """月末截面：硬过滤 + z-score + 加权打分。返回 (score over eligible, eligible)。"""
    fac = pd.DataFrame({k: v.loc[t] for k, v in ctx.f.items()}, index=STOCK_ETFS)
    eligible = fac.notna().all(axis=1) & (closes[STOCK_ETFS].loc[t] > ctx.ma60.loc[t])
    if eligible.sum() < cfg.min_scorable:
        return None, eligible          # 可评分 <2 → 全仓国债（保守，防单标的 z 退化）
    fe = fac[eligible]
    z = (fe - fe.mean()) / fe.std().replace(0.0, np.nan).fillna(1.0)   # std=0 → 中立 0
    score = cfg.w_mom * z["mom"] - cfg.w_downvol * z["downvol"] + cfg.w_liq * z["liq"]
    # 硬过滤②：下行波动 z 超阈值强制扣分
    score = score.where(z["downvol"] <= cfg.downvol_z_cut, score - cfg.downvol_penalty)
    return score, eligible


def compute_target_row(t: pd.Timestamp, closes: pd.DataFrame, ctx: Ctx,
                       cfg: V2Config) -> tuple[pd.Series, int]:
    """月末目标权重（含国债补足）+ 触发数。NaN 比较为 False → 热身期不触发，无害。"""
    n_trig = (int(ctx.breadth.loc[t] <= cfg.breadth_trigger)
              + int(ctx.vol_ann300.loc[t] > cfg.vol_trigger)
              + int(closes[HS300_PROXY].loc[t] < ctx.ma120_300.loc[t]))
    scale = cfg.scale_map[n_trig]
    cols = STOCK_ETFS + [TRESURY]
    out = pd.Series(0.0, index=cols)
    score, eligible = factor_cross_section(t, closes, ctx, cfg)
    if score is None:
        out[TRESURY] = 1.0
        return out, n_trig
    w = softmax(score, cfg.softmax_temp)
    if cfg.use_vol_weight:
        w = w * (1.0 / ctx.vol60.loc[t, w.index].clip(lower=1e-4))
        w = w / w.sum()
    w = cap_weights(w, cfg.max_weight)
    w = w * scale
    out.loc[w.index] = w.to_numpy()
    out[TRESURY] = 1.0 - float(w.sum())
    assert abs(out.sum() - 1.0) < 1e-9
    return out, n_trig


# ---------------------------------------------------------------- 事件循环引擎
def backtest_v2(closes: pd.DataFrame, amounts: pd.DataFrame,
                cfg: V2Config) -> dict:
    """事件循环回测：月末信号 T 日收盘定权重，T+1 生效；权重偏离>5% 或
    scale 变化才调仓（对比对象是漂移后的实际权重，路径依赖）。"""
    ctx = precompute(closes, amounts)
    idx = closes.index
    cols = STOCK_ETFS + [TRESURY]
    month_ends = set(month_end_dates(idx))
    ret = ctx.ret.fillna(0.0)

    w = pd.Series(0.0, index=cols)
    w[TRESURY] = 1.0
    held_scale = 1.0
    pending: tuple[pd.Series, float, float] | None = None   # (target, cost, scale)
    prev_scale: float | None = None      # 上一【生效 target】的 scale

    held_rows, net_rets, costs, scales = [], [], [], []
    eff_flags = []
    rebal_signals, skipped = [], 0
    month_log: list[tuple[pd.Timestamp, int, bool]] = []
    total_turnover = 0.0

    for i, t in enumerate(idx):
        r_t = ret.loc[t]
        if pending is not None:          # 生效日：昨日信号落地
            target, cost, held_scale = pending
            w = target.copy()
            pending = None
            eff_flags.append(True)
        else:
            cost = 0.0
            eff_flags.append(False)
        gross = float((w * r_t).sum())
        net = gross - cost
        held_rows.append(w.copy())
        net_rets.append(net)
        costs.append(cost)
        scales.append(held_scale)

        # 收盘漂移 → 次日实际权重（买入持有的精确分数更新）
        denom = 1.0 + gross
        w = w * (1.0 + r_t) / denom
        assert abs(w.sum() - 1.0) < 1e-9

        if t in month_ends:              # 月末收盘后算信号（对比对象 = 已漂移权重）
            target, n_trig = compute_target_row(t, closes, ctx, cfg)
            scale_now = cfg.scale_map[n_trig]
            forced = (prev_scale is not None) and (scale_now != prev_scale)
            delta = float((target - w).abs().max())
            if delta > cfg.rebal_threshold or forced:
                cost_p = float((target - w).abs().sum()) * cfg.cost_rate
                if cost_p > 1e-12:   # 零成本（如全仓国债时 scale 变化）不产生实际交易
                    pending = (target, cost_p, scale_now)
                    prev_scale = scale_now
                    rebal_signals.append(t)
                    total_turnover += float((target - w).abs().sum())
                    month_log.append((t, n_trig, True))
                else:
                    skipped += 1
                    month_log.append((t, n_trig, False))
            else:
                skipped += 1
                month_log.append((t, n_trig, False))
                # 跳过 → w 继续漂移，绝不覆盖

    net_ret_s = pd.Series(net_rets, index=idx)
    nav = (1.0 + net_ret_s).cumprod()
    held = pd.DataFrame(held_rows, index=idx)
    costs_s = pd.Series(costs, index=idx)
    scales_s = pd.Series(scales, index=idx)
    assert np.allclose(nav.iloc[0], 1.0)
    assert ((costs_s > 0).to_numpy() <= np.array(eff_flags, dtype=bool)).all(), \
        "成本只应出现在调仓生效日"
    assert np.isclose(costs_s.sum(), total_turnover * cfg.cost_rate)
    return {"nav": nav, "held": held, "costs": costs_s, "scales": scales_s,
            "net_rets": net_ret_s, "rebal_signals": rebal_signals, "skipped": skipped,
            "month_log": month_log, "total_turnover": total_turnover}


# ---------------------------------------------------------------- 校验三件套
def factor_ic(closes: pd.DataFrame, ctx: Ctx, cfg: V2Config,
              fwd_days: int = FWD_DAYS, min_pairs: int = IC_MIN_PAIRS) -> pd.DataFrame:
    """月末因子 vs 未来 21 日收益的截面 Spearman IC（rank 后 Pearson，免 scipy）。"""
    fwd = closes[STOCK_ETFS].shift(-fwd_days) / closes[STOCK_ETFS] - 1
    me = month_end_dates(closes.index)
    rows = {}
    for t in me:
        fx = fwd.loc[t]
        ic_row = {}
        for k in ("mom", "downvol", "liq"):
            fac_v = ctx.f[k].loc[t]
            valid = fac_v.notna() & fx.notna()
            if valid.sum() >= min_pairs:
                ic_row[k] = fac_v[valid].rank().corr(fx[valid].rank())
            else:
                ic_row[k] = np.nan
        score, eligible = factor_cross_section(t, closes, ctx, cfg)
        if score is not None:
            valid = fx.loc[score.index].notna()
            if valid.sum() >= min_pairs:
                ic_row["score"] = score.rank().corr(fx.loc[score.index].rank())
            else:
                ic_row["score"] = np.nan
        else:
            ic_row["score"] = np.nan
        rows[t] = ic_row
    return pd.DataFrame(rows).T


def classify_regime(c300: pd.Series) -> pd.Series:
    """日级牛熊震荡分类（510300 代理 000300）。热身期 NaN 不计入统计。"""
    ma120 = c300.rolling(120).mean()
    ret60 = c300.pct_change(60, fill_method=None)
    r = pd.Series("震荡", index=c300.index, dtype=object)
    r[(c300 > ma120) & (ret60 > 0.03)] = "牛市"
    r[(c300 < ma120) & (ret60 < -0.03)] = "熊市"
    r[ma120.isna() | ret60.isna()] = np.nan
    return r


def scenario_stats(rets: dict[str, pd.Series], regime: pd.Series) -> pd.DataFrame:
    """每场景×每系列：天数 / 段内累计收益 / 段内回撤（段内 cumprod 自建基线，
    用日收益而非净值比——净值比会混入场景外交易日的收益）。"""
    rows = []
    for name in ("牛市", "熊市", "震荡"):
        mask = (regime == name).to_numpy()
        days = int(mask.sum())
        for sname, r in rets.items():
            seg = r[mask]
            if len(seg) == 0:
                cum = dd = np.nan
            else:
                nav_seg = (1.0 + seg).cumprod()
                cum = float(nav_seg.iloc[-1] - 1)
                dd = float((nav_seg / nav_seg.cummax() - 1).min())
            rows.append((name, sname, days, cum, dd))
    return pd.DataFrame(rows, columns=["场景", "系列", "天数", "累计收益", "段内回撤"])


def rolling_eval(net_v2: pd.Series, net_eq: pd.Series,
                 window: int = ROLL_WIN, step: int = ROLL_STEP) -> pd.DataFrame:
    """单次全历史回测后切段（保留路径依赖状态；窗口 0 含热身期，报告注明）。"""
    rows = []
    for s in range(0, len(net_v2) - window + 1, step):
        seg_v2 = net_v2.iloc[s:s + window]
        seg_eq = net_eq.iloc[s:s + window]
        ann_v2 = float((1.0 + seg_v2).prod() ** (TRADING_DAYS / window) - 1)
        ann_eq = float((1.0 + seg_eq).prod() ** (TRADING_DAYS / window) - 1)
        rows.append({"起点": net_v2.index[s], "策略": ann_v2, "等权": ann_eq,
                     "超额": ann_v2 - ann_eq})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 绘图
def plot_v2_comparison(closes: pd.DataFrame, res: dict, nav1: pd.Series,
                       bench: dict[str, pd.Series], outpath: Path) -> None:
    dates = closes.index
    nav2 = res["nav"]
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True,
                             gridspec_kw={"height_ratios": [3, 2, 2.5]})
    ax1, ax2, ax3 = axes

    # 面板 1：净值对比（log）
    ax1.plot(dates, nav2, color=V2_PURPLE, lw=2.5, label="v2 多因子")
    ax1.plot(dates, nav1, color="#2a78d6", lw=1.8, label="v1 动量")
    ax1.plot(dates, bench["等权持有"], color="#eb6834", lw=1.5, label="等权持有")
    ax1.plot(dates, bench["沪深300"], color="#1baf7a", lw=1.5, label="沪深300")
    ax1.plot(dates, bench["国债ETF"], color="#008300", lw=1.5, label="国债ETF")
    ax1.set_yscale("log")
    ax1.set_title("v2 多因子 vs v1 动量 vs 基准（同成本口径，月末调仓）")
    ax1.set_ylabel("净值（对数刻度）")
    ax1.legend(loc="upper left", fontsize=9)
    right_pad = dates[-1] + pd.Timedelta(days=int((dates[-1] - dates[0]).days * 0.14))
    ax1.set_xlim(dates[0], right_pad)
    ax1.margins(y=0.08)
    series = [(nav2, "v2多因子"), (nav1, "v1动量"), (bench["等权持有"], "等权持有"),
              (bench["沪深300"], "沪深300"), (bench["国债ETF"], "国债ETF")]
    logv = np.log([s.iloc[-1] for s, _ in series])
    order = np.argsort(logv)
    for k in range(1, len(order)):
        lo, hi = order[k - 1], order[k]
        if logv[hi] - logv[lo] < 0.07:
            logv[hi] = logv[lo] + 0.07
    for (s, name), y in zip(series, np.exp(logv)):
        ax1.text(1.01, y, f"{name}  {s.iloc[-1]:.2f}",
                 transform=ax1.get_yaxis_transform(), ha="left", va="center",
                 fontsize=9, color=SECONDARY_INK)

    # 面板 2：回撤
    dd2 = nav2 / nav2.cummax() - 1
    dd1 = nav1 / nav1.cummax() - 1
    dd_hs = bench["沪深300"] / bench["沪深300"].cummax() - 1
    ax2.fill_between(dates, 0, dd2, color=V2_PURPLE, alpha=0.30, lw=0)
    ax2.plot(dates, dd2, color=V2_PURPLE, lw=1.5, label="v2 多因子")
    ax2.plot(dates, dd1, color="#2a78d6", lw=1.5, label="v1 动量")
    ax2.plot(dates, dd_hs, color=MUTED, lw=1.2, ls="--", label="沪深300")
    ax2.set_ylabel("回撤")
    ax2.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax2.legend(loc="lower right", fontsize=9)
    ax2.set_ylim(top=0.0)

    # 面板 3：v2 持仓堆叠（梯度权重）+ scale 参考线（堆顶即权益仓位和）
    cols = STOCK_ETFS + [TRESURY]
    x = mdates.date2num(dates.to_pydatetime())
    ax3.stackplot(x, res["held"][cols].to_numpy().T, colors=[ETF_COLORS[c] for c in cols],
                  linewidth=0.5, edgecolor=SURFACE, labels=[ETF_NAMES[c] for c in cols])
    ax3.plot(x, res["scales"].to_numpy(), color=SURFACE, lw=3.4, zorder=3)
    ax3.plot(x, res["scales"].to_numpy(), color=PRIMARY_INK, ls="--", lw=1.6, zorder=4,
             label="scale（权益仓位）")
    ax3.set_ylabel("持仓权重")
    ax3.set_yticks([0, 0.5, 1.0])
    ax3.set_ylim(0, 1)
    ax3.legend(ncols=4, loc="upper center", bbox_to_anchor=(0.5, -0.18), fontsize=9)

    ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=12))
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


def plot_rolling(roll: pd.DataFrame, outpath: Path) -> None:
    """滚动 1 年超额收益柱状图。正=PRIMARY_INK、负=MUTED（不用红绿，
    红绿语义留给 status/Delta），零基线。"""
    win_rate = float((roll["超额"] > 0).mean())
    fig, ax = plt.subplots(figsize=(12, 3.6))
    x = np.arange(len(roll))
    colors = [PRIMARY_INK if v >= 0 else MUTED for v in roll["超额"]]
    bars = ax.bar(x, roll["超额"], width=0.62, color=colors)
    for b, v in zip(bars, roll["超额"]):
        ax.text(b.get_x() + b.get_width() / 2, v,
                f"{v:+.1%}", ha="center",
                va="bottom" if v >= 0 else "top",
                fontsize=8.5, color=SECONDARY_INK)
    ax.axhline(0, color=BASELINE, lw=1.0, zorder=2)
    ax.set_title(f"滚动 1 年超额收益（v2 − 等权持有，窗口 252 日 / 步长 21 日）｜"
                 f"胜率 {win_rate:.0%} · 平均超额 {roll['超额'].mean():+.1%}")
    ax.set_ylabel("年化超额")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_xticks(x)
    ax.set_xticklabels([d.strftime("%y-%m") for d in roll["起点"]],
                       rotation=45, ha="right", fontsize=7.5)
    ax.margins(y=0.14)
    fig.subplots_adjust(bottom=0.20)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- 终端输出
def print_metrics_table(metrics: dict[str, dict]) -> None:
    rows = [("总收益", "总收益"), ("年化收益", "年化收益"), ("年化波动", "年化波动"),
            ("最大回撤", "最大回撤"), ("夏普", "夏普"), ("卡玛", "卡玛")]
    names = list(metrics.keys())
    print(_pad_cjk("指标", 8) + "  " + "  ".join(_pad_cjk(h, 8) for h in names))
    for label, key in rows:
        cells = [f"{m[key]:>8.2%}" if key != "夏普" else f"{m[key]:>8.2f}"
                 for m in metrics.values()]
        print(_pad_cjk(label, 8) + "  " + "  ".join(cells))


def print_v2_extras(res: dict, years: float) -> None:
    n_me = len(res["month_log"])
    applied = sum(1 for _, _, a in res["month_log"] if a)
    from collections import Counter
    trig_dist = Counter(n for _, n, _ in res["month_log"])
    day_scale = res["scales"].value_counts(normalize=True).sort_index()
    print(f"\n===== v2 附加 =====")
    print(f"月末信号 {n_me} 次：实际调仓 {applied} 次 ｜ 跳过 {res['skipped']} 次"
          f" ｜ 年度单边换手率 {res['total_turnover'] / 2 / years:.0%}"
          f" ｜ 总成本 {res['costs'].sum():.2%}")
    print(f"月末触发分布（n触发: 次数）= "
          + "  ".join(f"{k}:{v}" for k, v in sorted(trig_dist.items())))
    print(f"按日 scale 分布 = "
          + "  ".join(f"{k:.2f}:{v:.0%}" for k, v in day_scale.items())
          + f" ｜ 平均国债权重 {res['held'][TRESURY].mean():.0%}")


def print_ic(ic: pd.DataFrame) -> None:
    print(f"\n===== 因子 IC（月末截面 vs 未来 21 日收益，Spearman）=====")
    print(_pad_cjk("因子", 8) + _pad_cjk("均值IC", 9) + _pad_cjk("标准差", 9)
          + _pad_cjk("ICIR", 8) + _pad_cjk("正占比", 9) + _pad_cjk("观测数", 8))
    for col in ("mom", "downvol", "liq", "score"):
        s = ic[col].dropna()
        if len(s) == 0:
            continue
        mean = s.mean()
        std = s.std()
        print(_pad_cjk(col, 8)
              + f"{mean:>9.3f}  {std:>9.3f}  {mean / std:>8.2f}"
              + f"  {(s > 0).mean():>9.0%}  {len(s):>8d}")
    print("⚠ 仅约 29 个非重叠观测，ICIR 统计力弱，只作方向性参考；"
          "downvol 为负向因子（IC 为负才贡献正收益）")


def print_scenarios(st: pd.DataFrame) -> None:
    print(f"\n===== 分场景（日级分类，510300 代理沪深300："
          f"牛=破MA120且60日>+3%，熊=反之，余为震荡）=====")
    for name in ("牛市", "熊市", "震荡"):
        sub = st[st["场景"] == name]
        days = int(sub["天数"].iloc[0])
        print(f"\n{name}（{days} 天）:")
        for _, row in sub.iterrows():
            print(f"  {_pad_cjk(row['系列'], 8)}  累计 {row['累计收益']:+8.2%}"
                  f"  ｜ 段内最大回撤 {row['段内回撤']:8.2%}")


def print_rolling(roll: pd.DataFrame) -> None:
    win_rate = float((roll["超额"] > 0).mean())
    print(f"\n===== 滚动 1 年回测（252 日窗口 / 21 日起点，{len(roll)} 窗；"
          f"单次回测切段，窗口 0 含热身期）=====")
    print(f"策略年化：mean {roll['策略'].mean():+.2%} ｜ median {roll['策略'].median():+.2%}"
          f" ｜ min {roll['策略'].min():+.2%} ｜ max {roll['策略'].max():+.2%}")
    print(f"等权年化：mean {roll['等权'].mean():+.2%}"
          f" ｜ 超额：mean {roll['超额'].mean():+.2%} ｜ median {roll['超额'].median():+.2%}"
          f" ｜ 胜率 {win_rate:.0%}")


# ---------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description="ETF 多因子轮动 v2 回测（豆包文档借鉴+批判性验证）")
    ap.add_argument("--save-csv", action="store_true", help="保存净值/权重/IC CSV 到 output/")
    ap.add_argument("--outdir", type=Path, default=OUTPUT_DIR, help="输出目录")
    args = ap.parse_args()

    closes = load_closes(KLINES_DIR)
    amounts = load_amounts(KLINES_DIR)
    ctx = precompute(closes, amounts)
    cfg = V2Config()
    print(f"数据：7 只 ETF，{len(closes)} 个交易日（{closes.index[0]:%Y-%m-%d} ~ "
          f"{closes.index[-1]:%Y-%m-%d}）；无复权价、510300 代理 000300（数据局限）")
    print(f"v2：120日动量60% + 60日下行波动30%(负向) + 20日流动性10% ｜ 硬过滤(破MA60剔除/"
          f"下行波动z>{cfg.downvol_z_cut}扣分) ｜ softmax→逆波动→上限{cfg.max_weight:.0%} ｜ "
          f"择时scale {cfg.scale_map} ｜ 月末调仓偏离>{cfg.rebal_threshold:.0%}才调 ｜ "
          f"成本佣金万1+滑点0.1%/边")

    res = backtest_v2(closes, amounts, cfg)
    m2 = compute_metrics(res["nav"], RISK_FREE_ANNUAL)
    years = (len(res["nav"]) - 1) / TRADING_DAYS

    # v1 同成本口径基线（成本线性，传合并费率即可，不改 v1）；另跑 v1 原口径参考
    nav1_sc, _, _ = backtest(closes, 20, commission=COST_RATE)
    nav1_orig, _, _ = backtest(closes, 20)
    bench = benchmark_navs(closes)
    bench_m = {name: compute_metrics(s, RISK_FREE_ANNUAL) for name, s in bench.items()}

    print(f"\n===== 绩效对比（同成本口径 {COST_RATE:.1%}/边，策略已扣费基准不计）=====")
    print_metrics_table({"v1同成本": compute_metrics(nav1_sc, RISK_FREE_ANNUAL),
                         "v2多因子": m2, **bench_m})
    m1o = compute_metrics(nav1_orig, RISK_FREE_ANNUAL)
    print(f"（参考：v1 原口径仅佣金万1 年化 {m1o['年化收益']:.2%} / "
          f"回撤 {m1o['最大回撤']:.2%} / 夏普 {m1o['夏普']:.2f}）")

    print_v2_extras(res, years)

    # A/B 变体：检验文档宣称
    print(f"\n===== A/B 变体（同引擎换参数，检验文档宣称）=====")
    variants = [
        ("默认", cfg),
        ("no_scale", replace(cfg, scale_map={0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0})),
        ("no_volw", replace(cfg, use_vol_weight=False)),
        ("no_downvol", replace(cfg, w_downvol=0.0, w_mom=0.9)),
    ]
    print(_pad_cjk("变体", 12) + _pad_cjk("年化收益", 10) + _pad_cjk("最大回撤", 10)
          + _pad_cjk("夏普", 7) + _pad_cjk("换手/年", 9) + _pad_cjk("调仓", 5))
    for name, c in variants:
        r = backtest_v2(closes, amounts, c)
        m = compute_metrics(r["nav"], RISK_FREE_ANNUAL)
        to_rate = r["total_turnover"] / 2 / years
        print(f"{_pad_cjk(name, 12)}{m['年化收益']:>10.2%}  {m['最大回撤']:>10.2%}"
              f"  {m['夏普']:>7.2f}  {to_rate:>9.0%}  {len(r['rebal_signals']):>5d}")

    # 校验三件套
    ic = factor_ic(closes, ctx, cfg)
    print_ic(ic)

    regime = classify_regime(closes[HS300_PROXY])
    rets = {"v2多因子": res["net_rets"],
            "v1同成本": nav1_sc.pct_change(fill_method=None).fillna(0.0),
            "等权持有": bench["等权持有"].pct_change(fill_method=None).fillna(0.0),
            "沪深300": bench["沪深300"].pct_change(fill_method=None).fillna(0.0)}
    print_scenarios(scenario_stats(rets, regime))

    roll = rolling_eval(res["net_rets"],
                        bench["等权持有"].pct_change(fill_method=None).fillna(0.0))
    print_rolling(roll)

    # 图表与 CSV
    args.outdir.mkdir(exist_ok=True)
    p1 = args.outdir / "v2_comparison.png"
    p2 = args.outdir / "v2_rolling.png"
    plot_v2_comparison(closes, res, nav1_sc, bench, p1)
    plot_rolling(roll, p2)
    print(f"\n图表已保存：{p1}  {p2}")

    if args.save_csv:
        out = pd.DataFrame({"v2多因子": res["nav"], "v1动量": nav1_sc,
                            **{n: s for n, s in bench.items()}, **res["held"],
                            "scale": res["scales"]})
        pcsv = args.outdir / "v2_nav.csv"
        out.to_csv(pcsv, index_label="date")
        ic.to_csv(args.outdir / "v2_ic.csv", index_label="date")
        print(f"数据已保存：{pcsv}  {args.outdir / 'v2_ic.csv'}")


if __name__ == "__main__":
    main()
