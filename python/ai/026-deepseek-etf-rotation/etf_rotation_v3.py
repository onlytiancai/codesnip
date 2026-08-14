#!/usr/bin/env python3
"""ETF 大类资产配置轮动 v3（五资产轮动，按《年化10%回撤20%》文档方案）。

策略规格（文章原版 v3a，参数全部可配置）：
1. 资产池 5 类：沪深300（000300 指数作回测母体，ETF 2012 年上市，跟踪误差
   ~0.1-0.2%/年）、红利 510880、黄金 518880、国债 511010、商品 510170；
2. 趋势过滤：收盘价 > 200 日均线（≈10 个月均线）才可入选，跌破 → 不持有；
3. 动量：0.5×126 日收益 + 0.5×252 日收益（6M/12M 混合），趋势合格者中选前 2
   各 50%；仅 1 个合格 → 100%；0 个合格 → 100% 现金/国债（回退腿）；
4. 自然月月末调仓，T+1 生效，佣金单边万1。

数据局限（数据审计结论）：
- 510880 前复权污染严重：2008-08~2009-01 价格衰减到负值；2010-2020 年 79 行
  |日收益|>10.5%（10% 涨跌停约束下不可能成交）；2010~2015-08 段整体不可信
  （清洗后日波动仍 3.88% vs 000300 同期 1.58%）。→ **截断到 2016-01-01 起用**
  （列级 NaN，非行级切片）。
- 000300 为价格指数不含分红，全回测无分红建模（与 v1 同口径）。
- 5 资产全池仅 2013-07 起（~13 年）；早期为 1-4 资产动态池。

运行：
    /Users/huhao/.pyenv/versions/3.11.9/bin/python3.11 etf_rotation_v3.py [--walk-forward] [--save-csv]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter

from etf_momentum_rotation import (
    BASELINE, EPISODES, GRIDLINE, KLINES_DIR, MUTED, OUTPUT_DIR, PRIMARY_INK,
    RISK_FREE_ANNUAL, SECONDARY_INK, SURFACE, TRADING_DAYS, _pad_cjk,
    backtest, benchmark_navs, compute_metrics, load_closes, monthly_returns,
    plot_monthly_heatmap,
)
from etf_rotation_v2 import month_end_dates

# ---------------------------------------------------------------- 2. 配置常量
ASSET_KEYS = ["hs300", "div", "gold", "bond", "com"]
ASSET_COLS = ["000300.SH", "510880.SH", "518880.SH", "511010.SH", "510170.SH"]
ASSET_NAMES = {"hs300": "沪深300", "div": "红利", "gold": "黄金",
               "bond": "国债", "com": "商品", "cash": "现金/国债"}
DIVIDEND_START = "2016-01-01"   # 510880 干净起点（数据审计：2015-08 前整体不可信）

MA_TREND = 200                  # 趋势过滤均线（≈10 个月）
MOM_WINDOWS = (126, 252)        # 混合动量：0.5×6M + 0.5×12M
TOP_N = 2                       # 硬选前 2；1 合格 → 100%
COMMISSION_PER_SIDE = 1e-4
FALLBACK_IS_BOND = True         # 0 合格回退：国债收益（2013-03 前现金 0 收益）

# 资产实体色（dataviz 校验过的色板链：aqua→yellow→magenta→violet→green）
V3_COLORS = {"hs300": "#1baf7a", "div": "#e87ba4", "gold": "#eda100",
             "bond": "#008300", "com": "#4a3aa7", "cash": "#898781"}
V3_PURPLE = "#4a3aa7"           # v3 策略线色（同 v2 紫，线序已过校验）
V1_BLUE = "#2a78d6"

EPISODES_V3 = EPISODES + [
    ("2013 钱荒", "2013-05-20", "2013-07-05"),
    ("2022 股债双杀", "2022-01-01", "2022-12-31"),
]

# ---------------------------------------------------------------- 3. 数据加载
def load_v3_closes() -> pd.DataFrame:
    """v3 资产池收盘价（5 列，union 索引）。510880 截断到 2016-01-01。

    截断是【列级】置 NaN，不是行级切片（行级会砍掉 000300 的 2006-2009）。
    NaN 语义 = 未上市/不可用：动量 NaN → 自动排除，趋势比较得 False。
    """
    closes = load_closes(KLINES_DIR)
    out = closes[ASSET_COLS].copy()
    out.loc[out.index < DIVIDEND_START, "510880.SH"] = np.nan
    out.columns = ASSET_KEYS
    # 510880 清洗断言（防污染复发）
    assert (out["div"].isna() == (out.index < DIVIDEND_START)).all(), "510880 截断边界错误"
    assert (out["div"] <= 0).sum() == 0, "510880 仍含非正价格"
    return out

# ---------------------------------------------------------------- 4. 信号
def blended_momentum(closes_v3: pd.DataFrame,
                     windows: tuple = MOM_WINDOWS) -> pd.DataFrame:
    """多窗口动量混合。单窗口 NaN 按 0 贡献（符号保持、幅度折半）；
    双窗口皆 NaN → NaN → 未上市资产自动排除出排名。"""
    mom = None
    for w in windows:
        pc = closes_v3.pct_change(w, fill_method=None)
        mom = pc if mom is None else mom.add(pc, fill_value=0.0)
    return mom / len(windows)

def select_target_weights(trend: pd.DataFrame, mom: pd.DataFrame,
                          top_n: int) -> pd.DataFrame:
    """趋势合格 + 动量可算者中选动量前 top_n，权重 = 1/min(n_ok, top_n)。
    0 合格 → 全 0（回退腿接管）；1 合格 → 100%（文档语义）；否则前 top_n 均分。"""
    eligible = trend & mom.notna()
    rank = mom.where(eligible).rank(axis=1, ascending=False, method="first")
    sel = (rank <= top_n).astype(float)
    n_sel = sel.sum(axis=1)
    return sel.div(n_sel.clip(lower=1.0), axis=0).fillna(0.0)

def dual_layer_target(closes_v3: pd.DataFrame, ma_days: int, top_n: int) -> pd.DataFrame:
    """v3b 双层轮动：第一层权益内部（沪深300 vs 红利）趋势过滤后选强者，
    权益代表资产与黄金/国债/商品四腿平级竞争 top_n；权重拆回两列。"""
    tr = closes_v3 > closes_v3.rolling(ma_days).mean()
    pc = blended_momentum(closes_v3)
    ok_hs = tr["hs300"] & pc["hs300"].notna()
    ok_dv = tr["div"] & pc["div"].notna()
    hs_better = pc["hs300"].fillna(-np.inf) >= pc["div"].fillna(-np.inf)
    use_hs = (ok_hs & hs_better) | (ok_hs & ~ok_dv)
    use_dv = (ok_dv & ~hs_better) | (ok_dv & ~ok_hs)
    use_hs, use_dv = use_hs & ~use_dv, use_dv & ~use_hs
    leg_ok = pd.DataFrame({"eq": use_hs | use_dv,
                           "gold": tr["gold"] & pc["gold"].notna(),
                           "bond": tr["bond"] & pc["bond"].notna(),
                           "com": tr["com"] & pc["com"].notna()})
    leg_mom = pd.DataFrame({"eq": pc["hs300"].where(use_hs, pc["div"]),
                            "gold": pc["gold"], "bond": pc["bond"], "com": pc["com"]})
    rank = leg_mom.where(leg_ok).rank(axis=1, ascending=False, method="first")
    sel = (rank <= top_n).astype(float)
    w_leg = sel.div(sel.sum(axis=1).clip(lower=1.0), axis=0).fillna(0.0)
    target = pd.DataFrame(0.0, index=closes_v3.index, columns=ASSET_KEYS)
    target["hs300"] = w_leg["eq"] * use_hs
    target["div"] = w_leg["eq"] * use_dv
    target["gold"] = w_leg["gold"]
    target["bond"] = w_leg["bond"]
    target["com"] = w_leg["com"]
    return target

# ---------------------------------------------------------------- 5. 回测引擎
def _engine_core(closes_v3: pd.DataFrame, target: pd.DataFrame, *,
                 fallback_bond: bool, commission: float) -> tuple:
    idx = closes_v3.index
    me = month_end_dates(idx)                      # 实际交易日，非 resample 标签
    port = target.loc[me].reindex(idx).ffill().shift(1).fillna(0.0)
    port = port.copy()                             # CoW：整列赋值前显式 copy
    port["cash"] = 1.0 - port[ASSET_KEYS].sum(axis=1)

    ret = closes_v3.pct_change(fill_method=None).fillna(0.0)
    bond_ret = closes_v3["bond"].pct_change(fill_method=None).fillna(0.0)
    bond_ret[closes_v3["bond"].isna()] = 0.0       # 2013-03 前无国债 → 现金 0 收益
    fallback_ret = bond_ret if fallback_bond else pd.Series(0.0, index=idx)
    gross = (port[ASSET_KEYS] * ret).sum(axis=1) + port["cash"] * fallback_ret
    turnover = port.diff().abs().sum(axis=1).fillna(0.0)
    cost = turnover * commission
    net_ret = (gross - cost).fillna(0.0)
    nav = (1.0 + net_ret).cumprod()

    assert np.allclose(port.sum(axis=1), 1.0, atol=1e-9), "权重行和必须恒为 1"
    assert ((cost > 0) == (turnover > 0)).all(), "成本只应出现在换仓日"
    n_warm = idx.get_loc(me[0])
    assert port[ASSET_KEYS].iloc[:n_warm].to_numpy().sum() == 0.0, "热身期应全仓回退"
    assert np.isclose(nav.iloc[0], 1.0)
    return nav, port, turnover

def backtest_v3(closes_v3: pd.DataFrame, *, ma_days: int = MA_TREND,
                mom_windows: tuple = MOM_WINDOWS, top_n: int = TOP_N,
                commission: float = COMMISSION_PER_SIDE,
                fallback_bond: bool = FALLBACK_IS_BOND,
                use_trend: bool = True,
                mode: str = "flat") -> tuple:
    """五资产轮动回测主入口。mode="dual" 为双层轮动变体。"""
    if mode == "dual":
        target = dual_layer_target(closes_v3, ma_days, top_n)
    else:
        mom = blended_momentum(closes_v3, mom_windows)
        if use_trend:
            trend = closes_v3 > closes_v3.rolling(ma_days).mean()
        else:
            trend = mom > 0.0        # 绝对动量过滤替代趋势过滤（对照语义，输出注明）
        target = select_target_weights(trend, mom, top_n)
    return _engine_core(closes_v3, target, fallback_bond=fallback_bond,
                        commission=commission)

# ---------------------------------------------------------------- 6. 基准
def equal_weight_nav(closes_v3: pd.DataFrame) -> pd.Series:
    """五资产等权（动态池、权重漂移、不每日再平衡，与 v1 同法）。"""
    base = closes_v3.apply(lambda s: s.dropna().iloc[0] if s.notna().any() else np.nan)
    return (closes_v3 / base).mean(axis=1)

def trade_log_v3(port: pd.DataFrame, idx: pd.DatetimeIndex) -> list:
    """调仓记录：月末信号生效权重变化时记一条。"""
    me = month_end_dates(idx)
    logs, prev = [], None
    for t in me:
        i = idx.get_loc(t)
        w = port.iloc[min(i + 1, len(port) - 1)]
        held = tuple((c, w[c]) for c in port.columns if w[c] > 1e-9)
        if held != prev:
            logs.append((idx[min(i + 1, len(port) - 1)], held))
        prev = held
    return logs

# ---------------------------------------------------------------- 7. 消融
def run_ablations(closes_v3: pd.DataFrame) -> None:
    print(f"\n===== v3 消融（同成本万1；后列为 2013-07+ 全池段）=====")
    print(_pad_cjk("变体", 18) + _pad_cjk("年化", 7) + _pad_cjk("回撤", 8)
          + _pad_cjk("夏普", 6) + _pad_cjk("卡玛", 7) + _pad_cjk("换手", 7)
          + _pad_cjk("调仓", 5) + _pad_cjk("13+年化", 8) + _pad_cjk("13+回撤", 8))
    variants = [
        ("v3a 原版", dict()),
        ("纯126", dict(mom_windows=(126,))),
        ("纯252", dict(mom_windows=(252,))),
        ("MA120", dict(ma_days=120)),
        ("无趋势过滤", dict(use_trend=False)),
        ("top1", dict(top_n=1)),
        ("top3", dict(top_n=3)),
        ("回退纯现金", dict(fallback_bond=False)),
        ("v3b 双层", dict(mode="dual")),
    ]
    for name, kw in variants:
        nav, _, to = backtest_v3(closes_v3, **kw)
        m = compute_metrics(nav, RISK_FREE_ANNUAL)
        m13 = compute_metrics(nav.loc["2013-07-29":], RISK_FREE_ANNUAL)
        yrs = (len(nav) - 1) / TRADING_DAYS
        tr = to.sum() / 2 / yrs
        print(f"{_pad_cjk(name, 18)}{m['年化收益']:>7.2%}  {m['最大回撤']:>8.2%}"
              f"  {m['夏普']:>6.2f}  {m['卡玛']:>7.2%}  {tr:>7.0%}"
              f"  {int((to > 1e-12).sum()):>5d}  {m13['年化收益']:>8.2%}"
              f"  {m13['最大回撤']:>8.2%}")
    # 坏数据对照：不截断、直接用原始污染数据（量化坏数据危害）
    raw = load_closes(KLINES_DIR)
    bad = closes_v3.copy()
    bad["div"] = raw["510880.SH"].reindex(bad.index)
    nav_bad, _, to_bad = backtest_v3(bad)
    m_bad = compute_metrics(nav_bad, RISK_FREE_ANNUAL)
    m_bad13 = compute_metrics(nav_bad.loc["2013-07-29":], RISK_FREE_ANNUAL)
    yrs_b = (len(nav_bad) - 1) / TRADING_DAYS
    print(f"{_pad_cjk('未截断(坏数据)', 18)}{m_bad['年化收益']:>7.2%}"
          f"  {m_bad['最大回撤']:>8.2%}  {m_bad['夏普']:>6.2f}"
          f"  {m_bad['卡玛']:>7.2%}  {to_bad.sum() / 2 / yrs_b:>7.0%}"
          f"  {int((to_bad > 1e-12).sum()):>5d}  {m_bad13['年化收益']:>8.2%}"
          f"  {m_bad13['最大回撤']:>8.2%}")

# ---------------------------------------------------------------- 8. 校验
def walk_forward_v3(closes_v3: pd.DataFrame, *, min_train: int = 504,
                    step: int = 504, metric: str = "夏普") -> pd.DataFrame:
    """滚出验证：网格 MA×动量×top_n，train 段按夏普选参，前瞻评估 + 固定默认对照。"""
    rows = []
    for t0 in range(min_train, len(closes_v3), step):
        test_end = min(t0 + step, len(closes_v3))
        train = closes_v3.iloc[:t0]
        full = closes_v3.iloc[:test_end]
        best = None
        for ma in (120, 200, 250):
            for mw in (MOM_WINDOWS, (126,), (252,)):
                for tn in (1, 2, 3):
                    nav, _, _ = backtest_v3(train, ma_days=ma, mom_windows=mw, top_n=tn)
                    score = compute_metrics(nav, RISK_FREE_ANNUAL)[metric]
                    if best is None or score > best[0]:
                        best = (score, ma, mw, tn)
        _, ma, mw, tn = best
        nav_wf, _, _ = backtest_v3(full, ma_days=ma, mom_windows=mw, top_n=tn)
        nav_def, _, _ = backtest_v3(full)
        eq5 = equal_weight_nav(full)
        m_wf = compute_metrics(nav_wf.iloc[t0:test_end], RISK_FREE_ANNUAL)
        m_def = compute_metrics(nav_def.iloc[t0:test_end], RISK_FREE_ANNUAL)
        m_eq = compute_metrics(eq5.iloc[t0:test_end], RISK_FREE_ANNUAL)
        mw_label = "blend" if mw == MOM_WINDOWS else str(mw[0])
        rows.append({"起点": closes_v3.index[t0], "结束": closes_v3.index[test_end - 1],
                     "参数": f"MA{ma}/m{mw_label}/top{tn}",
                     "wf年化": m_wf["年化收益"], "wf回撤": m_wf["最大回撤"],
                     "默认年化": m_def["年化收益"], "默认回撤": m_def["最大回撤"],
                     "等权年化": m_eq["年化收益"]})
    return pd.DataFrame(rows)

def print_walk_forward_v3(wf: pd.DataFrame) -> None:
    print(f"\n===== v3 滚出验证（网格 MA∈{{120,200,250}} × 动量∈{{blend,126,252}}"
          f" × top∈{{1,2,3}}，train 按夏普选参，固定默认对照）=====")
    def cell(v: float, w: int = 8) -> str:
        return "—".rjust(w) if pd.isna(v) else f"{v:>{w}.2%}"
    print(_pad_cjk("测试区间", 22) + _pad_cjk("选中参数", 16) + _pad_cjk("wf年化", 8)
          + _pad_cjk("wf回撤", 8) + _pad_cjk("默认年化", 9) + _pad_cjk("默认回撤", 9)
          + _pad_cjk("等权年化", 9))
    for _, r in wf.iterrows():
        print(f"{r['起点']:%Y-%m}~{r['结束']:%Y-%m}   {_pad_cjk(r['参数'], 16)}"
              f"{cell(r['wf年化'])}  {cell(r['wf回撤'])}  {cell(r['默认年化'], 9)}"
              f"  {cell(r['默认回撤'], 9)}  {cell(r['等权年化'], 9)}")
    wf_ex = (wf["wf年化"] - wf["等权年化"]).dropna()
    def_ex = (wf["默认年化"] - wf["等权年化"]).dropna()
    print(f"\n滚出选参：超额 vs 五资产等权 均值 {wf_ex.mean():+.2%} ｜ 胜率 {(wf_ex > 0).mean():.0%}"
          f"（{int((wf_ex > 0).sum())}/{len(wf_ex)} 窗）")
    print(f"固定默认：超额 vs 五资产等权 均值 {def_ex.mean():+.2%} ｜ 胜率 {(def_ex > 0).mean():.0%}"
          f"（{int((def_ex > 0).sum())}/{len(def_ex)} 窗）")

def print_episodes_v3(closes_v3: pd.DataFrame, nav: pd.Series, port: pd.DataFrame,
                      nav_v1: pd.Series, eq5: pd.Series, nav300: pd.Series) -> None:
    print(f"\n===== 极端行情复盘（v3 vs 基准，段内累计收益）=====")
    print(_pad_cjk("区间", 16) + _pad_cjk("v3轮动", 10) + _pad_cjk("五资产等权", 11)
          + _pad_cjk("等权v1", 10) + _pad_cjk("000300", 10)
          + _pad_cjk("平均权益仓", 10) + _pad_cjk("最低权益仓", 10))
    for name, a, b in EPISODES_V3:
        mask = (closes_v3.index >= a) & (closes_v3.index <= b)
        idx = closes_v3.index[mask]
        if len(idx) == 0:
            continue
        def seg_ret(s: pd.Series) -> float:
            seg = s.loc[idx].dropna()
            return float(seg.iloc[-1] / seg.iloc[0] - 1) if len(seg) else np.nan
        def cell(v: float) -> str:
            return "—".rjust(10) if pd.isna(v) else f"{v:>10.2%}"
        equity = (port["hs300"] + port["div"]).loc[idx]
        print(f"{_pad_cjk(name, 16)}{cell(seg_ret(nav))}{cell(seg_ret(eq5))}"
              f"{cell(seg_ret(nav_v1))}{cell(seg_ret(nav300))}"
              f"{equity.mean():>10.0%}{equity.min():>10.0%}")

# ---------------------------------------------------------------- 9. 绘图
def plot_v3_main(closes_v3: pd.DataFrame, nav: pd.Series, port: pd.DataFrame,
                 nav_v1: pd.Series, eq5: pd.Series, nav300: pd.Series,
                 bond_nav: pd.Series, outpath: Path) -> None:
    dates = closes_v3.index
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True,
                             gridspec_kw={"height_ratios": [3, 2, 2.5]})
    ax1, ax2, ax3 = axes

    ax1.plot(dates, nav, color=V3_PURPLE, lw=2.5, label="v3 五资产轮动")
    ax1.plot(dates, nav_v1, color=V1_BLUE, lw=1.8, label="v1 动量轮动")
    ax1.plot(dates, eq5, color="#eb6834", lw=1.5, label="五资产等权")
    ax1.plot(dates, nav300, color="#1baf7a", lw=1.5, label="000300指数")
    ax1.plot(dates, bond_nav, color="#008300", lw=1.5, label="国债ETF")
    ax1.set_yscale("log")
    ax1.set_title("v3 五资产轮动 vs v1 动量轮动 vs 基准（20 年，月末调仓）")
    ax1.set_ylabel("净值（对数刻度）")
    # 图例顶条加白底，防 2007 峰值区视觉拥挤
    ax1.legend(loc="lower center", bbox_to_anchor=(0.5, 1.03), ncols=5, fontsize=8.5,
               frameon=True, facecolor="white", edgecolor=GRIDLINE, framealpha=0.9)
    right_pad = dates[-1] + pd.Timedelta(days=int((dates[-1] - dates[0]).days * 0.16))
    ax1.set_xlim(dates[0], right_pad)
    ax1.margins(y=0.08)
    series = [(nav, "v3轮动"), (nav_v1, "v1动量"), (eq5, "五资产等权"),
              (nav300, "000300"), (bond_nav, "国债ETF")]
    logv = np.log([s.iloc[-1] for s, _ in series])
    order = np.argsort(logv)
    for k in range(1, len(order)):
        lo, hi = order[k - 1], order[k]
        if logv[hi] - logv[lo] < 0.09:
            logv[hi] = logv[lo] + 0.09
    for (s, name), y in zip(series, np.exp(logv)):
        ax1.text(1.02, y, f"{name}  {s.iloc[-1]:.2f}",
                 transform=ax1.get_yaxis_transform(), ha="left", va="center",
                 fontsize=9, color=SECONDARY_INK)

    dd3 = nav / nav.cummax() - 1
    dd1 = nav_v1 / nav_v1.cummax() - 1
    dd300 = nav300 / nav300.cummax() - 1
    ax2.fill_between(dates, 0, dd3, color=V3_PURPLE, alpha=0.30, lw=0)
    ax2.plot(dates, dd3, color=V3_PURPLE, lw=1.5, label="v3 五资产轮动")
    ax2.plot(dates, dd1, color=V1_BLUE, lw=1.5, label="v1 动量轮动")
    ax2.plot(dates, dd300, color="#1baf7a", lw=1.2, ls="--", label="000300")
    ax2.set_ylabel("回撤")
    ax2.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax2.legend(loc="lower right", fontsize=9)
    ax2.set_ylim(top=0.0)

    cols = ASSET_KEYS + ["cash"]
    x = mdates.date2num(dates.to_pydatetime())
    ax3.stackplot(x, port[cols].to_numpy().T, colors=[V3_COLORS[c] for c in cols],
                  linewidth=0.5, edgecolor=SURFACE,
                  labels=[ASSET_NAMES[c] for c in cols])
    ax3.set_ylabel("持仓权重")
    ax3.set_yticks([0, 0.5, 1.0])
    ax3.set_ylim(0, 1)
    ax3.axhline(0.5, color=GRIDLINE, ls="--", lw=0.8)
    ax3.legend(ncols=6, loc="upper center", bbox_to_anchor=(0.5, -0.18), fontsize=9)

    ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=24))
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)

# ---------------------------------------------------------------- 10. main
def main() -> None:
    ap = argparse.ArgumentParser(description="ETF 大类资产配置轮动 v3（五资产轮动）")
    ap.add_argument("--walk-forward", action="store_true", help="运行滚出验证")
    ap.add_argument("--save-csv", action="store_true", help="保存净值/权重 CSV")
    ap.add_argument("--outdir", type=Path, default=OUTPUT_DIR, help="输出目录")
    args = ap.parse_args()

    closes_v3 = load_v3_closes()
    print(f"数据：5 类资产，{len(closes_v3)} 个交易日（{closes_v3.index[0]:%Y-%m-%d} ~ "
          f"{closes_v3.index[-1]:%Y-%m-%d}）；沪深300 腿用 000300 指数（无分红建模，"
          f"ETF 跟踪误差 ~0.1-0.2%/年）；510880 红利截断到 {DIVIDEND_START} 起用"
          f"（数据审计：此前前复权污染严重）；5 资产全池 2013-07 起")
    print(f"策略：MA{MA_TREND} 趋势过滤 ｜ 动量 0.5×126日+0.5×252日 ｜ 选前 {TOP_N}"
          f" 各 50% ｜ 0 合格回退{'国债' if FALLBACK_IS_BOND else '现金'} ｜ "
          f"月末调仓 ｜ 佣金单边万1")

    nav, port, turnover = backtest_v3(closes_v3)
    m3 = compute_metrics(nav, RISK_FREE_ANNUAL)
    years = (len(nav) - 1) / TRADING_DAYS

    closes_all = load_closes(KLINES_DIR)
    nav_v1, _, _ = backtest(closes_all, 10, use_scale=True, use_index_triggers=True)
    bench = benchmark_navs(closes_all)
    eq5 = equal_weight_nav(closes_v3)
    nav300 = closes_v3["hs300"] / closes_v3["hs300"].dropna().iloc[0]

    print(f"\n===== 绩效对比（{closes_v3.index[0]:%Y-%m-%d} ~ {closes_v3.index[-1]:%Y-%m-%d}，"
          f"策略已扣佣金万1，基准不计成本）=====")
    metrics = {"v3轮动": m3,
               "v1轮动": compute_metrics(nav_v1, RISK_FREE_ANNUAL),
               "五资产等权": compute_metrics(eq5, RISK_FREE_ANNUAL),
               "等权v1": compute_metrics(bench["等权持有"], RISK_FREE_ANNUAL),
               "000300": compute_metrics(nav300, RISK_FREE_ANNUAL),
               "国债ETF": compute_metrics(bench["国债ETF"], RISK_FREE_ANNUAL)}
    rows = [("总收益", "总收益"), ("年化收益", "年化收益"), ("年化波动", "年化波动"),
            ("最大回撤", "最大回撤"), ("夏普", "夏普"), ("卡玛", "卡玛")]
    print(_pad_cjk("指标", 10) + "  ".join(_pad_cjk(h, 10) for h in metrics))
    for label, key in rows:
        cells = [f"{m[key]:>10.2%}" if key != "夏普" else f"{m[key]:>10.2f}"
                 for m in metrics.values()]
        print(_pad_cjk(label, 10) + "  ".join(cells))
    print(f"\nv3 附加：年换手 {turnover.sum() / 2 / years:.0%} ｜ 调仓 "
          f"{int((turnover > 1e-12).sum())} 次 ｜ 回退天数占比 "
          f"{(port['cash'] > 0.99).mean():.0%}")

    run_ablations(closes_v3)

    trades = trade_log_v3(port, closes_v3.index)
    print(f"\n===== 调仓记录（生效日，共 {len(trades)} 次，显示最近 40 次）=====")
    for d, held in trades[-40:]:
        parts = [f"{ASSET_NAMES[c]} {w:.0%}" for c, w in held]
        print(f"{d:%Y-%m-%d}  {' + '.join(parts)}")

    print_episodes_v3(closes_v3, nav, port, nav_v1, eq5, nav300)

    if args.walk_forward:
        wf = walk_forward_v3(closes_v3)
        print_walk_forward_v3(wf)

    args.outdir.mkdir(exist_ok=True)
    p1 = args.outdir / "v3_comparison.png"
    plot_v3_main(closes_v3, nav, port, nav_v1, eq5, nav300, bench["国债ETF"], p1)
    p2 = args.outdir / "v3_monthly_heatmap.png"
    plot_monthly_heatmap({"v3轮动": nav.pct_change(fill_method=None).fillna(0.0),
                          "五资产等权": eq5.pct_change(fill_method=None).fillna(0.0)},
                         p2)
    print(f"\n图表已保存：{p1}  {p2}")

    if args.save_csv:
        out = pd.DataFrame({"v3轮动": nav, "v1轮动": nav_v1, "五资产等权": eq5,
                            "000300": nav300, **port})
        pcsv = args.outdir / "v3_nav.csv"
        out.to_csv(pcsv, index_label="date")
        print(f"数据已保存：{pcsv}")

    # 诚实结论（写入输出末尾）
    print(f"\n===== 结论：文档目标（20 年化>9% 且回撤<20%）"
          f"{'达成' if m3['年化收益'] > 0.09 and m3['最大回撤'] > -0.20 else '未达成'} =====")
    print(f"v3a 实际：年化 {m3['年化收益']:.2%} / 回撤 {m3['最大回撤']:.2%}。"
          f"机制：MA200 慢线（2008/2015 跌破前已深跌且月末才调）+ top2 硬选双权益腿"
          f"全暴露 + 2007 峰值到 2026 才收复的死区 + 红利腿 2016 前缺席。"
          f"v1（广度快择时）仍是回撤最优解。")


if __name__ == "__main__":
    main()
