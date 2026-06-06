"""
计算 SPX 近 10 年最大 5 次回撤 (Max Drawdown)，打印详细信息并可视化。

回撤定义：
  drawdown(t) = (price(t) - peak_to_date(t)) / peak_to_date(t)
  max drawdown = drawdown 时间序列上的最小值
  每次回撤区间 = [peak_date, trough_date]
  recovery_date = 之后价格首次 >= peak 的交易日（若未恢复则 None）
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 中文字体设置（macOS）：Heiti SC 优先，找不到时 fallback
plt.rcParams["font.sans-serif"] = ["Heiti SC", "Hei", "STHeiti",
                                   "Hiragino Sans GB", "PingFang HK",
                                   "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

CSV_PATH = Path(__file__).with_name("download_data_spx_10y.csv")
OUT_PNG = Path(__file__).with_name("max_drawdowns.png")


def compute_max_drawdowns(df: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    """返回 top_n 大的回撤，按回撤幅度（负数）从大到小排序。

    列：peak_date, peak_price, trough_date, trough_price,
        drawdown_pct, drawdown_days, recovery_date, recovery_days
    """
    df = df.sort_values("Date").reset_index(drop=True)
    price = df["Close"].values
    dates = df["Date"].values

    # 运行峰值 (前一日及之前最高收盘价)
    running_max = np.maximum.accumulate(price)
    dd = (price - running_max) / running_max  # <= 0

    # 找所有局部回撤区间：
    # 一次回撤起点 = running_max 创新高的那一天
    # 一次回撤结束 = 该次回撤的最低点（trough），或之后 running_max 再创新高
    n = len(price)
    events: list[dict] = []
    i = 0
    while i < n:
        # 找这一段回撤：起点为 i（这里 running_max 创新高或就是起点）
        peak_idx = i
        peak_price = price[i]
        peak_date = dates[i]

        # 往后扫，直到 trough 或 running_max 回到 >= peak
        trough_idx = peak_idx
        trough_price = price[peak_idx]
        j = peak_idx + 1
        while j < n:
            if price[j] >= peak_price:
                # 创新高：从这里开始新一段
                break
            if price[j] < trough_price:
                trough_price = price[j]
                trough_idx = j
            j += 1

        if trough_idx > peak_idx:
            dd_pct = (trough_price - peak_price) / peak_price
            recovery_idx = None
            if j < n:
                # 从 trough 之后首次回到 peak_price 的位置
                rec = np.where(price[trough_idx + 1 :] >= peak_price)[0]
                if len(rec):
                    recovery_idx = trough_idx + 1 + rec[0]
            events.append(
                {
                    "peak_idx": peak_idx,
                    "peak_date": pd.Timestamp(dates[peak_idx]),
                    "peak_price": peak_price,
                    "trough_idx": trough_idx,
                    "trough_date": pd.Timestamp(dates[trough_idx]),
                    "trough_price": trough_price,
                    "drawdown_pct": dd_pct,
                    "recovery_idx": recovery_idx,
                    "recovery_date": (
                        pd.Timestamp(dates[recovery_idx])
                        if recovery_idx is not None
                        else pd.NaT
                    ),
                }
            )
            # 下一段从 j 开始
            i = j if j > trough_idx else trough_idx + 1
        else:
            i += 1

    res = pd.DataFrame(events)
    if res.empty:
        return res

    res["drawdown_days"] = (res["trough_date"] - res["peak_date"]).dt.days
    res["recovery_days"] = np.where(
        res["recovery_date"].isna(),
        np.nan,
        (res["recovery_date"] - res["trough_date"]).dt.days,
    )
    res = res.sort_values("drawdown_pct").reset_index(drop=True)
    return res.head(top_n)


def main() -> None:
    df = pd.read_csv(CSV_PATH, parse_dates=["Date"])
    print("=" * 78)
    print(f"数据: {CSV_PATH.name}  ({len(df)} 行, {df['Date'].min().date()} → {df['Date'].max().date()})")
    print("=" * 78)

    top5 = compute_max_drawdowns(df, top_n=5)

    print(f"\n最大 5 次回撤（按回撤幅度排序）：\n")
    show = top5.copy()
    show["drawdown_pct"] = show["drawdown_pct"] * 100
    show["peak_price"] = show["peak_price"].round(2)
    show["trough_price"] = show["trough_price"].round(2)
    show["recovery_date"] = show["recovery_date"].dt.strftime("%Y-%m-%d").fillna("未恢复")
    show["peak_date"] = show["peak_date"].dt.strftime("%Y-%m-%d")
    show["trough_date"] = show["trough_date"].dt.strftime("%Y-%m-%d")
    print(
        show[
            [
                "peak_date",
                "peak_price",
                "trough_date",
                "trough_price",
                "drawdown_pct",
                "drawdown_days",
                "recovery_date",
                "recovery_days",
            ]
        ].to_string(index=False)
    )

    # 总结性指标
    mdd = top5["drawdown_pct"].iloc[0] * 100
    print(f"\n  → 历史最大回撤 (MDD): {mdd:.2f}%")

    # ---------- 可视化 ----------
    df = df.sort_values("Date").reset_index(drop=True)
    price = df["Close"].values
    dates = df["Date"].values
    running_max = np.maximum.accumulate(price)
    dd_curve = (price - running_max) / running_max * 100

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(13, 8), sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )

    # 上：价格 + 高亮每次回撤区间
    ax1.plot(dates, price, color="#1f4e79", lw=1.2, label="SPX Close")
    ax1.plot(dates, running_max, color="#888", lw=0.8, ls="--", label="Running Peak")
    colors = ["#c0392b", "#e67e22", "#d4ac0d", "#27ae60", "#2980b9"]
    for i, (_, row) in enumerate(top5.iterrows()):
        mask = (dates >= row["peak_date"]) & (dates <= row["recovery_date"])
        if row["recovery_date"] is pd.NaT:
            mask = (dates >= row["peak_date"])
        ax1.axvspan(
            row["peak_date"],
            row["trough_date"] if row["recovery_date"] is pd.NaT else row["recovery_date"],
            alpha=0.15,
            color=colors[i],
            label=f"#{i+1} DD: {row['drawdown_pct']*100:.1f}%",
        )
        # 标注 peak 和 trough
        ax1.scatter([row["peak_date"]], [row["peak_price"]], color=colors[i], s=40, zorder=5)
        ax1.scatter([row["trough_date"]], [row["trough_price"]], color=colors[i], s=40, marker="v", zorder=5)

    ax1.set_ylabel("收盘价")
    ax1.set_title("SPX (^GSPC) – 近 10 年最大 5 次回撤", fontsize=13)
    ax1.legend(loc="upper left", fontsize=8, ncol=2)
    ax1.grid(alpha=0.3)

    # 下：回撤曲线 + 标记 Top5 的 trough
    ax2.fill_between(dates, dd_curve, 0, color="#c0392b", alpha=0.3, label="回撤 %")
    ax2.plot(dates, dd_curve, color="#c0392b", lw=0.8)
    for i, (_, row) in enumerate(top5.iterrows()):
        ax2.scatter(
            [row["trough_date"]],
            [row["drawdown_pct"] * 100],
            color=colors[i],
            s=70,
            zorder=5,
            edgecolors="black",
        )
        ax2.annotate(
            f"#{i+1} {row['drawdown_pct']*100:.1f}%",
            xy=(row["trough_date"], row["drawdown_pct"] * 100),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            fontsize=9,
            color=colors[i],
            weight="bold",
        )
    ax2.set_ylabel("回撤 (%)")
    ax2.set_xlabel("日期")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=130)
    print(f"\n图表已保存: {OUT_PNG}")


if __name__ == "__main__":
    main()
