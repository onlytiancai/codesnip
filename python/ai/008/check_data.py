"""
检查 008/download_data_spx_10y.csv 数据质量与基本统计。
"""

from pathlib import Path

import numpy as np
import pandas as pd

CSV_PATH = Path(__file__).with_name("download_data_spx_10y.csv")


def main() -> None:
    df = pd.read_csv(CSV_PATH, parse_dates=["Date"])
    print("=" * 70)
    print(f"文件: {CSV_PATH.name}  ({CSV_PATH.stat().st_size/1024:.1f} KB)")
    print("=" * 70)

    # ---------- 1. 基本形状 ----------
    print(f"\n[1] 形状与列")
    print(f"  行数: {len(df)}")
    print(f"  列:   {list(df.columns)}")
    print(f"  dtype:\n{df.dtypes.to_string()}")

    # ---------- 2. 时间范围 ----------
    print(f"\n[2] 时间范围")
    print(f"  起始: {df['Date'].min().date()}")
    print(f"  结束: {df['Date'].max().date()}")
    span = (df["Date"].max() - df["Date"].min()).days
    years = span / 365.25
    print(f"  跨度: {span} 天 ≈ {years:.2f} 年")

    # ---------- 3. 缺失值 ----------
    print(f"\n[3] 缺失值")
    na = df.isna().sum()
    print(f"  NaN 总数: {int(na.sum())}")
    if na.sum() > 0:
        print(na[na > 0].to_string())
    else:
        print("  ✓ 无缺失值")

    # ---------- 4. 重复 ----------
    dup = df.duplicated(subset=["Date"]).sum()
    print(f"\n[4] 重复日期: {dup}  (基于 Date 唯一性)")

    # ---------- 5. 时间连续性（是否漏日） ----------
    print(f"\n[5] 交易日连续性")
    all_days = pd.date_range(df["Date"].min(), df["Date"].max(), freq="B")  # 工作日
    missing = sorted(set(all_days) - set(df["Date"]))
    print(f"  实际交易日: {len(df)}")
    print(f"  工作日(剔除周末): {len(all_days)}")
    print(f"  缺失工作日(节假日/数据缺失): {len(missing)}")
    if missing and len(missing) <= 20:
        print(f"  缺失日期: {[d.date().isoformat() for d in missing]}")
    elif missing:
        print(f"  前 10 个缺失: {[d.date().isoformat() for d in missing[:10]]} ...")

    # 最大跳日间隔
    gaps = df["Date"].diff().dt.days.dropna()
    big_gaps = gaps[gaps > 5]  # 超过 5 天的间隔（> 周末+1天）
    print(f"  相邻交易日最大间隔: {int(gaps.max())} 天")
    print(f"  间隔 > 5 天的次数: {len(big_gaps)}（基本对应长假期）")

    # ---------- 6. 价格合理性 ----------
    print(f"\n[6] 价格合理性 (OHLC 关系)")
    bad_high = (df["High"] < df[["Open", "Close", "Low"]].max(axis=1)).sum()
    bad_low = (df["Low"] > df[["Open", "Close", "High"]].min(axis=1)).sum()
    neg = (df[["Open", "High", "Low", "Close", "Adj Close"]] <= 0).sum().sum()
    print(f"  High 小于 Open/Close/Low 任一的行: {bad_high}")
    print(f"  Low  大于 Open/Close/High 任一的行: {bad_low}")
    print(f"  非正价格数: {neg}")
    if bad_high + bad_low + neg == 0:
        print("  ✓ OHLC 关系全部正常")

    # ---------- 7. 成交量 ----------
    print(f"\n[7] 成交量 Volume")
    print(f"  最小: {df['Volume'].min():,.0f}")
    print(f"  最大: {df['Volume'].max():,.0f}")
    print(f"  中位: {df['Volume'].median():,.0f}")
    zero_vol = (df["Volume"] == 0).sum()
    print(f"  为 0 的天数: {zero_vol}")

    # ---------- 8. 价格统计 ----------
    print(f"\n[8] 价格统计")
    desc = df[["Open", "High", "Low", "Close", "Adj Close"]].describe()
    print(desc.to_string())

    # ---------- 9. 收益率 ----------
    df = df.sort_values("Date").reset_index(drop=True)
    ret = df["Close"].pct_change().dropna()
    log_ret = np.log(df["Close"] / df["Close"].shift(1)).dropna()
    print(f"\n[9] 日收益率 (Close->Close)")
    print(f"  N:           {len(ret)}")
    print(f"  均值:        {ret.mean()*100:.4f}%")
    print(f"  标准差:      {ret.std()*100:.4f}%")
    print(f"  年化波动率:  {ret.std()*np.sqrt(252)*100:.2f}%")
    print(f"  最小:        {ret.min()*100:.2f}%   ({df.loc[ret.idxmin(),'Date'].date()})")
    print(f"  最大:        {ret.max()*100:.2f}%   ({df.loc[ret.idxmax(),'Date'].date()})")
    print(f"  累计收益:    {(df['Close'].iloc[-1]/df['Close'].iloc[0] - 1)*100:.2f}%")
    print(f"  CAGR:        {((df['Close'].iloc[-1]/df['Close'].iloc[0])**(1/years) - 1)*100:.2f}%")

    # 极端日
    print(f"\n  涨幅 Top 5:")
    top = df.assign(r=ret).nlargest(5, "r")[["Date", "Close", "r"]]
    for _, r in top.iterrows():
        print(f"    {r['Date'].date()}  Close={r['Close']:.2f}  {r['r']*100:+.2f}%")
    print(f"  跌幅 Top 5:")
    bot = df.assign(r=ret).nsmallest(5, "r")[["Date", "Close", "r"]]
    for _, r in bot.iterrows():
        print(f"    {r['Date'].date()}  Close={r['Close']:.2f}  {r['r']*100:+.2f}%")

    # ---------- 10. Close vs Adj Close ----------
    diff = (df["Close"] - df["Adj Close"]).abs()
    diff_pct = (df["Close"] / df["Adj Close"] - 1).abs()
    print(f"\n[10] Close vs Adj Close")
    print(f"  绝对差最大: {diff.max():.4f}")
    print(f"  相对差最大: {diff_pct.max()*100:.4f}%")
    print(f"  恒等行数:   {(diff < 1e-6).sum()}/{len(df)}")

    # ---------- 总结 ----------
    print(f"\n{'='*70}\n结论")
    print("=" * 70)
    issues = []
    if na.sum() > 0:
        issues.append("存在缺失值")
    if dup:
        issues.append("存在重复日期")
    if bad_high or bad_low or neg:
        issues.append("存在 OHLC 不一致")
    if zero_vol:
        issues.append("存在零成交量日")
    if not issues:
        print("  ✓ 数据干净，可直接用于研究/回测。")
    else:
        for x in issues:
            print(f"  ✗ {x}")


if __name__ == "__main__":
    main()
