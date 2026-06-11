"""
保守抄底策略回测 (SPX 10y)

信号（3 因子 AND，全部在 train 集 12.5% 分位取阈值，绝对无未来函数）：
    1. 量比 vol_ratio > 1.16
    2. 跌破 MA20: ma20_dev <= -2.7%
    3. 5 日波动率 vol_5d > 21% (年化)

回测方式：信号触发后持有 20 个交易日，期间不再加新仓。
        同时可视化资金曲线（信号触发日→t+20 的累计收益串联）。
        画三张图：
            (a) 资金曲线 - 完整 10 年，对比『策略 vs 买入持有』
            (b) test 段资金曲线 + 每次触发点标注
            (c) 每次触发的 20 日前瞻收益分布 + 胜率/均值/中位数文字
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

# 中文字体设置（macOS）
plt.rcParams["font.sans-serif"] = ["Heiti SC", "Hei", "STHeiti",
                                   "Hiragino Sans GB", "PingFang HK",
                                   "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

CSV_PATH = Path(__file__).with_name("download_data_spx_10y.csv")
OUT_PNG = Path(__file__).with_name("strategy_conservative.png")

# ---------- 数据切分 ----------
TRAIN_END = pd.Timestamp("2021-12-31")
VAL_END = pd.Timestamp("2023-12-31")
HOLD_DAYS = 20        # 持有期
INITIAL_CAPITAL = 100  # 起点资金 100

# ---------- 策略参数（来自 buy_the_dip.py 的 train 12.5% 分位） ----------
# 这些阈值是『在 train 段上』算出来的，全集回测无未来函数
RULE_Q = 0.125
THR_VOL_RATIO = 1.16       # 量比 >= 1.16
THR_MA20_DEV  = -0.027     # 跌破 MA20 超过 2.7%
THR_VOL_5D    = 0.21       # 5 日年化波动率 >= 21%


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("Date").reset_index(drop=True).copy()
    df["ret_1d"] = df["Close"].pct_change()
    df["ma20_dev"]  = df["Close"] / df["Close"].rolling(20).mean() - 1
    df["vol_5d"]    = df["ret_1d"].rolling(5).std() * np.sqrt(252)
    df["vol_ratio"] = df["Volume"] / df["Volume"].rolling(20).mean()
    # 前瞻 20 日收益
    df["fwd_20"] = df["Close"].shift(-HOLD_DAYS) / df["Close"] - 1
    return df


def assign_period(d: pd.Timestamp) -> str:
    if d <= TRAIN_END:
        return "train"
    if d <= VAL_END:
        return "val"
    return "test"


def in_position_until(close_idx: int, held_until: set[int]) -> bool:
    """如果当前日期被任何未平仓信号『覆盖』，就跳过（不重复入场）"""
    return any(close_idx <= e for e in held_until)


def main() -> None:
    df = pd.read_csv(CSV_PATH, parse_dates=["Date"])
    df = build_features(df)
    df["period"] = df["Date"].apply(assign_period)

    # ---------- 1) 信号识别 ----------
    sig = (
        (df["vol_ratio"] >= THR_VOL_RATIO) &
        (df["ma20_dev"]  <= THR_MA20_DEV)  &
        (df["vol_5d"]    >= THR_VOL_5D)    &
        df["vol_ratio"].notna() &
        df["ma20_dev"].notna()  &
        df["vol_5d"].notna()
    )

    # ---------- 2) 串行回测（避免重叠持仓） ----------
    # 在全集（含 train+val+test）上找信号，触发后持有 20 天，期间不再加仓
    trades = []           # 每次交易记录
    in_pos_until = -1     # 当前持仓到期日（行号）
    for i in range(len(df)):
        if i <= in_pos_until:
            continue
        if not sig.iloc[i]:
            continue
        entry_idx = i
        exit_idx = i + HOLD_DAYS
        if exit_idx >= len(df):
            break
        trades.append({
            "entry_date":  df["Date"].iloc[entry_idx],
            "exit_date":   df["Date"].iloc[exit_idx],
            "entry_px":    float(df["Close"].iloc[entry_idx]),
            "exit_px":     float(df["Close"].iloc[exit_idx]),
            "ret":         float(df["fwd_20"].iloc[entry_idx]),
            "period":      df["period"].iloc[entry_idx],
            "vol_ratio":   float(df["vol_ratio"].iloc[entry_idx]),
            "ma20_dev":    float(df["ma20_dev"].iloc[entry_idx]),
            "vol_5d":      float(df["vol_5d"].iloc[entry_idx]),
        })
        in_pos_until = exit_idx

    trades_df = pd.DataFrame(trades)
    print("=" * 78)
    print("保守抄底策略回测 - SPX 10y")
    print("=" * 78)
    print(f"策略: 量比 >= {THR_VOL_RATIO}  AND  跌破 MA20 <= {THR_MA20_DEV:.1%}  "
          f"AND  5日波动率 >= {THR_VOL_5D:.0%}")
    print(f"持有期: {HOLD_DAYS} 日 | 起点资金: {INITIAL_CAPITAL}")
    print(f"交易次数: {len(trades_df)}")
    if len(trades_df) == 0:
        print("无交易")
        return

    # ---------- 3) 统计 ----------
    print("\n[1] 交易明细（按入场日期排序）")
    show = trades_df.copy()
    show["ret_%"]      = (show["ret"] * 100).round(2)
    show["vol_ratio"]  = show["vol_ratio"].round(3)
    show["ma20_dev"]   = (show["ma20_dev"] * 100).round(2)
    show["vol_5d"]     = (show["vol_5d"] * 100).round(1)
    print(show[["entry_date", "exit_date", "period",
                "vol_ratio", "ma20_dev", "vol_5d", "ret_%"]].to_string(index=False))

    # 分段统计
    print("\n[2] 分段统计")
    for p in ["train", "val", "test", "all"]:
        sub = trades_df if p == "all" else trades_df[trades_df["period"] == p]
        if len(sub) == 0:
            print(f"  {p:>5s}: 0 次")
            continue
        win  = (sub["ret"] > 0).mean() * 100
        mean = sub["ret"].mean() * 100
        med  = sub["ret"].median() * 100
        print(f"  {p:>5s}: n={len(sub):>3d}  胜率={win:5.1f}%  "
              f"平均收益={mean:+6.2f}%  中位数={med:+6.2f}%")

    # ---------- 4) 资金曲线（与买入持有对比） ----------
    # 策略曲线：每次交易按 ret 复利累乘
    strat_eq = [INITIAL_CAPITAL]
    for r in trades_df["ret"]:
        strat_eq.append(strat_eq[-1] * (1 + r))
    # 时间序列：每次出仓日对应一个权益点
    eq_dates = [df["Date"].iloc[0]] + list(trades_df["exit_date"])
    strat_eq = pd.Series(strat_eq, index=pd.to_datetime(eq_dates))

    # 买入持有基准
    bench = df[["Date", "Close"]].set_index("Date")["Close"]
    bench_eq = bench / bench.iloc[0] * INITIAL_CAPITAL

    # 策略权益与基准『同日』对齐（向前填充）
    strat_eq_aligned = bench_eq.copy()
    strat_eq_aligned[:] = np.nan
    for d, v in strat_eq.items():
        if d in strat_eq_aligned.index:
            strat_eq_aligned.loc[d:] = v
    strat_eq_aligned = strat_eq_aligned.ffill().fillna(INITIAL_CAPITAL)

    # 关键指标
    strat_total = strat_eq.iloc[-1] / INITIAL_CAPITAL - 1
    bench_total = bench_eq.iloc[-1] / INITIAL_CAPITAL - 1
    print(f"\n[3] 资金曲线对比")
    print(f"  策略最终权益: {strat_eq.iloc[-1]:8.2f}  (累计 {strat_total*100:+.1f}%)")
    print(f"  基准最终权益: {bench_eq.iloc[-1]:8.2f}  (累计 {bench_total*100:+.1f}%)")
    print(f"  超额: {(strat_total - bench_total)*100:+.1f} pp")

    # ---------- 5) 画图 ----------
    fig = plt.figure(figsize=(15, 11))
    gs  = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.25)
    ax1 = fig.add_subplot(gs[0, :])    # 完整资金曲线
    ax2 = fig.add_subplot(gs[1, 0])    # test 段资金曲线 + 触发点
    ax3 = fig.add_subplot(gs[1, 1])    # 每次交易收益柱
    ax4 = fig.add_subplot(gs[2, 0])    # 分段胜率/平均收益
    ax5 = fig.add_subplot(gs[2, 1])    # 触发时市场状态散点

    # ---- (a) 完整资金曲线 ----
    ax1.plot(bench_eq.index, bench_eq.values,
             color="#888", lw=1.5, label="买入持有基准")
    ax1.plot(strat_eq_aligned.index, strat_eq_aligned.values,
             color="#1f77b4", lw=2.2, label="保守抄底策略")
    # 标注每个平仓点
    for _, t in trades_df.iterrows():
        ax1.axvline(t["entry_date"], color="#d62728", alpha=0.12, lw=0.6)
    # 标注 train/val/test 区间
    ax1.axvspan(df["Date"].iloc[0],  TRAIN_END, alpha=0.04, color="blue",   label="训练集")
    ax1.axvspan(TRAIN_END,          VAL_END,  alpha=0.04, color="orange", label="验证集")
    ax1.axvspan(VAL_END, df["Date"].iloc[-1], alpha=0.04, color="green",  label="测试集")
    ax1.set_title("保守抄底策略 vs 买入持有 - 完整 10 年", fontsize=13)
    ax1.set_ylabel("资金 (起点=100)")
    ax1.legend(loc="upper left", fontsize=9, ncol=3)
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_locator(mdates.YearLocator(2))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # ---- (b) test 段详细资金曲线 ----
    test_start = VAL_END
    test_bench = bench_eq[bench_eq.index > test_start]
    test_strat = strat_eq_aligned[strat_eq_aligned.index > test_start]
    test_trades = trades_df[trades_df["period"] == "test"]
    ax2.plot(test_bench.index, test_bench.values,
             color="#888", lw=1.5, label="买入持有")
    ax2.plot(test_strat.index, test_strat.values,
             color="#1f77b4", lw=2.5, marker="o", ms=4, label="保守抄底")
    for _, t in test_trades.iterrows():
        ax2.scatter([t["entry_date"]], [test_strat.asof(t["entry_date"])
                    if t["entry_date"] in test_strat.index else np.nan],
                    color="#d62728", s=70, zorder=5, marker="^")
        ax2.annotate(f"{t['ret']*100:+.1f}%",
                     xy=(t["exit_date"],
                         test_strat.asof(t["exit_date"])),
                     xytext=(0, 8), textcoords="offset points",
                     ha="center", fontsize=8, color="#1f77b4")
    ax2.set_title(f"测试集详细 (val/test 边界={test_start.date()}, "
                  f"test 段 {len(test_trades)} 笔交易)", fontsize=12)
    ax2.set_ylabel("资金 (起点=100)")
    ax2.legend(loc="upper left", fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=30, ha="right")

    # ---- (c) 每次交易 20 日收益柱 ----
    colors = ["#2ca02c" if r > 0 else "#d62728"
              for r in trades_df["ret"]]
    bars = ax3.bar(range(len(trades_df)), trades_df["ret"] * 100,
                   color=colors, alpha=0.75, edgecolor="black", lw=0.5)
    ax3.axhline(0, color="black", lw=0.8)
    ax3.set_xticks(range(len(trades_df)))
    ax3.set_xticklabels([d.strftime("%Y-%m")
                         for d in trades_df["entry_date"]],
                        rotation=45, ha="right", fontsize=8)
    ax3.set_title(f"每次交易 20 日前瞻收益  "
                  f"(绿={int((trades_df['ret']>0).sum())} 胜 / "
                  f"红={int((trades_df['ret']<=0).sum())} 负)", fontsize=12)
    ax3.set_ylabel("收益 (%)")
    ax3.grid(True, alpha=0.3, axis="y")

    # ---- (d) 分段表现 ----
    periods = ["train", "val", "test"]
    win_rates = []
    mean_rets = []
    n_trades  = []
    for p in periods:
        sub = trades_df[trades_df["period"] == p]
        n_trades.append(len(sub))
        if len(sub) == 0:
            win_rates.append(0)
            mean_rets.append(0)
        else:
            win_rates.append((sub["ret"] > 0).mean() * 100)
            mean_rets.append(sub["ret"].mean() * 100)
    x = np.arange(len(periods))
    width = 0.35
    bars1 = ax4.bar(x - width/2, win_rates, width,
                    color="#1f77b4", alpha=0.8, label="胜率 (%)")
    bars2 = ax4.bar(x + width/2, mean_rets, width,
                    color="#ff7f0e", alpha=0.8, label="平均收益 (%)")
    ax4.axhline(50, color="gray", ls="--", lw=1, alpha=0.6)
    ax4.axhline(0,  color="black", lw=0.6)
    for b, v in zip(bars1, win_rates):
        ax4.text(b.get_x() + b.get_width()/2, v + 1.2, f"{v:.1f}%",
                 ha="center", fontsize=9)
    for b, v in zip(bars2, mean_rets):
        ax4.text(b.get_x() + b.get_width()/2, v + 1.2 if v >= 0 else v - 4,
                 f"{v:+.1f}%", ha="center", fontsize=9)
    ax4.set_xticks(x)
    ax4.set_xticklabels([f"{p}\n(n={n})" for p, n in zip(periods, n_trades)])
    ax4.set_title("分段表现对比", fontsize=12)
    ax4.set_ylabel("胜率 / 平均收益 (%)")
    ax4.legend(loc="upper right", fontsize=9)
    ax4.grid(True, alpha=0.3, axis="y")

    # ---- (e) 触发时市场状态散点 ----
    sc = ax5.scatter(trades_df["ma20_dev"] * 100, trades_df["vol_ratio"],
                     c=trades_df["ret"] * 100, cmap="RdYlGn",
                     s=80, edgecolor="black", lw=0.5, alpha=0.85)
    plt.colorbar(sc, ax=ax5, label="20日收益 (%)")
    ax5.axvline(THR_MA20_DEV * 100, color="blue",   ls="--",
                lw=1, alpha=0.5, label="MA20 阈值")
    ax5.axhline(THR_VOL_RATIO,      color="orange", ls="--",
                lw=1, alpha=0.5, label="量比阈值")
    ax5.set_xlabel("ma20_dev (%)")
    ax5.set_ylabel("量比 vol_ratio")
    ax5.set_title("触发时市场状态分布", fontsize=12)
    ax5.legend(loc="upper left", fontsize=8)
    ax5.grid(True, alpha=0.3)

    fig.suptitle(
        f"保守抄底策略回测  |  信号: 量比≥{THR_VOL_RATIO} & 跌破MA20≤{THR_MA20_DEV:.1%} & 5日波≥{THR_VOL_5D:.0%}  |  持有{HOLD_DAYS}日",
        fontsize=14, y=0.998, weight="bold",
    )

    plt.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
    print(f"\n图表已保存: {OUT_PNG}")


if __name__ == "__main__":
    main()
