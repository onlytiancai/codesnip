"""
Buy-the-dip 信号研究 (SPX 10y)

目标：发现『历史上什么条件下抄底大概率赚钱』，并严格做样本外验证。

信号：只用 SPX 自身的 OHLCV 衍生量，零未来函数。

切分（防过拟合）：
  Train: 2016-01-01 ~ 2021-12-31  → 选规则/阈值
  Val:   2022-01-01 ~ 2023-12-31  → 微调
  Test:  2024-01-01 ~ 2026-06-05   → 完全 holdout，只跑一次

每个信号：在 t 日触发，记录未来 5/10/20 日前瞻收益。
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
OUT_PNG = Path(__file__).with_name("buy_the_dip.png")

# ---------- 数据切分 ----------
TRAIN_END = pd.Timestamp("2021-12-31")
VAL_END = pd.Timestamp("2023-12-31")

# ---------- 候选因子（可解释的、纯时序的） ----------
FACTOR_DEFS = {
    "ret_1d":     ("单日收益",        lambda d: d["ret_1d"]),
    "ret_5d":     ("5日累计收益",     lambda d: d["Close"].pct_change(5)),
    "ret_10d":    ("10日累计收益",    lambda d: d["Close"].pct_change(10)),
    "ret_20d":    ("20日累计收益",    lambda d: d["Close"].pct_change(20)),
    "ma20_dev":   ("Close/MA20 - 1",  lambda d: d["Close"] / d["Close"].rolling(20).mean() - 1),
    "ma60_dev":   ("Close/MA60 - 1",  lambda d: d["Close"] / d["Close"].rolling(60).mean() - 1),
    "dd_20d":     ("20日新高回撤",    lambda d: d["Close"] / d["Close"].rolling(20).max() - 1),
    "vol_5d":     ("5日已实现波动率", lambda d: d["ret_1d"].rolling(5).std() * np.sqrt(252)),
    "vol_ratio":  ("量比(V/MA20V)",   lambda d: d["Volume"] / d["Volume"].rolling(20).mean()),
    "down_streak":("连续下跌天数",    None),  # 特殊处理
}

# 阈值方向：'low' 表示因子越低越该抄底（典型：跌幅、跌破均线、波动率）；
#           'high' 表示因子越高越该抄底（典型：量比放大）
FACTOR_DIRECTION = {
    "ret_1d": "low", "ret_5d": "low", "ret_10d": "low", "ret_20d": "low",
    "ma20_dev": "low", "ma60_dev": "low", "dd_20d": "low",
    "vol_5d": "high",  # 高波动=恐慌，抄底胜率更高（用 train 验证）
    "vol_ratio": "high",
    "down_streak": "low",
}

FORWARD_HORIZONS = [5, 10, 20]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("Date").reset_index(drop=True).copy()
    df["ret_1d"] = df["Close"].pct_change()
    for k, (_, fn) in FACTOR_DEFS.items():
        if fn is None:
            continue
        df[k] = fn(df)
    # 连续下跌天数
    is_down = (df["ret_1d"] < 0).fillna(False)
    df["down_streak"] = (
        is_down.groupby((~is_down).cumsum().fillna(0)).cumcount() + 1
    ) * is_down
    df["down_streak"] = df["down_streak"].fillna(0).astype(int)

    # 前瞻收益（用于评估，不用于信号生成）
    for h in FORWARD_HORIZONS:
        df[f"fwd_{h}"] = df["Close"].shift(-h) / df["Close"] - 1
    return df


def assign_period(d: pd.Timestamp) -> str:
    if d <= TRAIN_END:
        return "train"
    if d <= VAL_END:
        return "val"
    return "test"


def evaluate_threshold(factor_vals: pd.Series, fwd: pd.Series,
                       direction: str, q: float) -> dict:
    """对单个因子，在分位 q 处取阈值，统计触发后前瞻收益。"""
    v = factor_vals.dropna()
    thr = v.quantile(q) if direction == "low" else v.quantile(1 - q)
    mask = (factor_vals <= thr) if direction == "low" else (factor_vals >= thr)
    sub = fwd[mask].dropna()
    if len(sub) == 0:
        return {"thr": thr, "n": 0}
    return {
        "thr": thr,
        "n": int(len(sub)),
        "mean": float(sub.mean()),
        "median": float(sub.median()),
        "win_rate": float((sub > 0).mean()),
    }


def scan_factor(df: pd.DataFrame, factor: str, period: str) -> pd.DataFrame:
    direction = FACTOR_DIRECTION[factor]
    sub = df[df["period"] == period]
    rows = []
    quantiles = [0.05, 0.10, 0.15, 0.20]
    for h in FORWARD_HORIZONS:
        fwd = sub[f"fwd_{h}"]
        fv = sub[factor]
        for q in quantiles:
            res = evaluate_threshold(fv, fwd, direction, q)
            res.update({"factor": factor, "period": period,
                        "horizon": h, "q": q})
            rows.append(res)
    return pd.DataFrame(rows)


def select_rules_on_train(df: pd.DataFrame) -> list[dict]:
    """在 train 上按『前瞻 10 日 win_rate > 55% 且 n >= 8』挑选稳健规则。

    为避免过拟合：
      - 只看前瞻 10 日
      - 至少 8 次触发（统计有意义）
      - 至少 2 个分位 q 同时满足
    """
    rules = []
    for fac in FACTOR_DEFS:
        scan = scan_factor(df, fac, "train")
        good_qs = scan[(scan["horizon"] == 10)
                       & (scan["win_rate"] >= 0.55)
                       & (scan["n"] >= 8)]["q"].tolist()
        if len(good_qs) >= 2:
            # 选最稳健的中位 q
            best_q = float(np.median(good_qs))
            rules.append({
                "factor": fac,
                "direction": FACTOR_DIRECTION[fac],
                "q": best_q,
                "train_winning_qs": good_qs,
            })
    return rules


def apply_rule(df: pd.DataFrame, rule: dict) -> pd.Series:
    """返回 bool Series：哪些日期触发了抄底信号。"""
    thr = df.loc[df["period"] == "train", rule["factor"]].quantile(
        rule["q"] if rule["direction"] == "low" else 1 - rule["q"]
    )
    v = df[rule["factor"]]
    if rule["direction"] == "low":
        return v <= thr
    return v >= thr


def evaluate_rule(df: pd.DataFrame, rule: dict) -> pd.DataFrame:
    """对一条规则在三个 period 上做前瞻收益评估。"""
    sig = apply_rule(df, rule)
    rows = []
    for period in ["train", "val", "test"]:
        sub = df[df["period"] == period].copy()
        sub_sig = sig.loc[sub.index]
        for h in FORWARD_HORIZONS:
            fwd = sub.loc[sub_sig, f"fwd_{h}"].dropna()
            if len(fwd) == 0:
                continue
            rows.append({
                "factor": rule["factor"],
                "q": rule["q"],
                "period": period,
                "horizon": h,
                "n": int(len(fwd)),
                "mean_ret_%": float(fwd.mean() * 100),
                "median_ret_%": float(fwd.median() * 100),
                "win_rate_%": float((fwd > 0).mean() * 100),
            })
    return pd.DataFrame(rows)


def main() -> None:
    df = pd.read_csv(CSV_PATH, parse_dates=["Date"])
    df = build_features(df)
    df["period"] = df["Date"].apply(assign_period)

    print("=" * 78)
    print("Buy-the-Dip 信号研究 - SPX 10y")
    print("=" * 78)
    print(f"Train: {(df['period']=='train').sum()}  "
          f"Val: {(df['period']=='val').sum()}  "
          f"Test: {(df['period']=='test').sum()}")

    # ---------- Step 1: 训练集扫描 ----------
    print("\n[Step 1] 在 train 上扫描所有因子 × 分位 × 持有期")
    all_scan = pd.concat(
        [scan_factor(df, f, "train") for f in FACTOR_DEFS],
        ignore_index=True,
    )
    # 展示 train 上前瞻 10 日最稳健的 (高 win_rate, n>=8)
    top_train = (
        all_scan[(all_scan["horizon"] == 10) & (all_scan["n"] >= 8)]
        .sort_values(["win_rate", "n"], ascending=[False, False])
        .head(15)
    )
    show = top_train[["factor", "q", "thr", "n", "mean", "win_rate"]].copy()
    show["thr"] = show["thr"].round(4)
    show["mean"] = (show["mean"] * 100).round(3)
    show["win_rate"] = (show["win_rate"] * 100).round(1)
    show.columns = ["factor", "q", "threshold", "n", "fwd10d_mean_%", "fwd10d_win_%"]
    print(show.to_string(index=False))

    # ---------- Step 2: 选稳健规则 ----------
    rules = select_rules_on_train(df)
    print(f"\n[Step 2] 在 train 上挑出 {len(rules)} 条稳健规则 "
          f"(前瞻10日 win_rate>=55% 且 n>=8 的 q 至少 2 个):")
    for r in rules:
        print(f"  - {r['factor']:12s}  q={r['q']:.2f}  "
              f"direction={r['direction']}  "
              f"good_qs={[round(x,2) for x in r['train_winning_qs']]}")

    # ---------- Step 3: 三段评估 ----------
    print(f"\n[Step 3] 在 train / val / test 上评估每条规则")
    summary = pd.concat([evaluate_rule(df, r) for r in rules], ignore_index=True)
    summary["mean_ret_%"] = summary["mean_ret_%"].round(3)
    summary["median_ret_%"] = summary["median_ret_%"].round(3)
    summary["win_rate_%"] = summary["win_rate_%"].round(1)
    print(summary.to_string(index=False))

    # 寻找同时三段都正且 test 胜率 > 50% 的规则
    for h in FORWARD_HORIZONS:
        piv = (summary[summary["horizon"] == h]
               .pivot_table(index="factor", columns="period",
                            values=["win_rate_%", "mean_ret_%", "n"]))
        print(f"\n[Step 3b] 前瞻 {h} 日各 period 横向对比：")
        print(piv.round(2).to_string())

    # ---------- Step 4: 组合规则 (投票) ----------
    if rules:
        sig_all = pd.DataFrame({r["factor"]: apply_rule(df, r) for r in rules})
        df["n_signals"] = sig_all.sum(axis=1)
        print("\n[Step 4] 多信号投票 (要求至少 K 个因子同时触发) 的前瞻 10 日表现：")
        vote_rows = []
        for k in range(1, len(rules) + 1):
            mask = df["n_signals"] >= k
            for period in ["train", "val", "test"]:
                fwd = df.loc[mask & (df["period"] == period), "fwd_10"].dropna()
                if len(fwd) == 0:
                    continue
                vote_rows.append({
                    "K": k, "period": period, "n": int(len(fwd)),
                    "mean_%": float(fwd.mean() * 100),
                    "win_%": float((fwd > 0).mean() * 100),
                })
        vote_df = pd.DataFrame(vote_rows)
        print(vote_df.round(2).to_string(index=False))

    # ---------- Step 5: 可视化 ----------
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    # (a) 价格 + 触发信号
    ax = axes[0, 0]
    ax.plot(df["Date"], df["Close"], color="#1f4e79", lw=1.0)
    if rules:
        for r in rules:
            sig = apply_rule(df, r)
            ax.scatter(df.loc[sig, "Date"], df.loc[sig, "Close"],
                       s=10, alpha=0.5, label=r["factor"])
    ax.set_title("(a) SPX 价格 + 抄底信号触发点")
    ax.set_ylabel("收盘价")
    ax.legend(fontsize=7, loc="upper left", title="因子")
    ax.grid(alpha=0.3)

    # (b) 各规则在 3 个 period 上的前瞻 10 日胜率
    ax = axes[0, 1]
    if rules:
        s10 = summary[summary["horizon"] == 10]
        x = np.arange(len(rules))
        w = 0.25
        for i, period in enumerate(["train", "val", "test"]):
            ax.bar(x + i * w, s10[s10["period"] == period]["win_rate_%"],
                   w, label={"train": "训练集", "val": "验证集", "test": "测试集"}[period])
        ax.axhline(50, color="gray", ls="--", lw=0.8)
        ax.set_xticks(x + w)
        ax.set_xticklabels([r["factor"] for r in rules], rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("前瞻 10 日胜率 (%)")
        ax.set_title("(b) 各规则在三个时段的胜率")
        ax.legend()
        ax.grid(alpha=0.3, axis="y")

    # (c) 投票策略累计收益
    ax = axes[1, 0]
    if rules:
        for k in [1, max(1, len(rules) // 2), len(rules)]:
            mask = df["n_signals"] >= k
            test_period = df[df["period"] == "test"].copy()
            sig_in_test = mask.loc[test_period.index]
            equity = (1 + test_period["ret_1d"].fillna(0) * sig_in_test.shift(1).fillna(False)).cumprod()
            ax.plot(test_period["Date"], equity * 100,
                    label=f"K≥{k} (测试集, 触发{int(sig_in_test.sum())}次)")
        # 基准 buy & hold
        test_period["bh"] = (1 + test_period["ret_1d"].fillna(0)).cumprod()
        ax.plot(test_period["Date"], test_period["bh"] * 100,
                color="black", ls="--", label="买入持有 (测试集)")
    ax.set_title("(c) 投票策略在测试集上的资金曲线")
    ax.set_ylabel("净值 (起点=100)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # (d) 所有信号触发的『前瞻 N 日平均收益』分布
    ax = axes[1, 1]
    if rules:
        sig_any = pd.DataFrame({r["factor"]: apply_rule(df, r) for r in rules}).any(axis=1)
        for i, h in enumerate(FORWARD_HORIZONS):
            vals = df.loc[sig_any, f"fwd_{h}"].dropna() * 100
            ax.hist(vals, bins=20, alpha=0.5, label=f"前瞻 {h} 日")
        ax.axvline(0, color="red", lw=0.8)
    ax.set_title("(d) 触发信号后的前瞻收益分布")
    ax.set_xlabel("前瞻收益 (%)")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=130)
    print(f"\n图表已保存: {OUT_PNG}")

    # ---------- 最终结论 ----------
    print("\n" + "=" * 78)
    print("结论")
    print("=" * 78)
    if rules:
        # 样本外稳健：在 val 和 test 上 win_rate 同时 >= 50% 的规则（任一持有期）
        s_all = summary[summary["period"].isin(["val", "test"])]
        win_piv = s_all.pivot_table(
            index=["factor", "horizon"], columns="period", values="win_rate_%"
        ).dropna()
        robust = win_piv[(win_piv["val"] >= 50) & (win_piv["test"] >= 50)]
        if len(robust):
            print("样本外稳健（val & test 胜率均 >= 50%）的规则：")
            robust_sorted = robust.sort_values(["horizon", "test"], ascending=[True, False])
            for (fac, h), row in robust_sorted.iterrows():
                print(f"  - {fac:12s}  持有期={h:>2d}日  "
                      f"val_win={row['val']:.1f}%  test_win={row['test']:.1f}%")
        else:
            print("没有规则在 val 和 test 上同时 win_rate >= 50%。")

        # 跨期平均胜率（val+test 均值）最高的 3 条
        avg = (s_all.groupby(["factor", "horizon"])["win_rate_%"]
               .mean().sort_values(ascending=False).head(5))
        print("\n  样本外(val+test)平均胜率 Top 5:")
        for (fac, h), v in avg.items():
            print(f"    {fac:12s}  h={h:>2d}  avg_win={v:.1f}%")
    else:
        print("在 train 上没有任何规则满足稳健条件，建议放宽阈值或延长持有期。")


if __name__ == "__main__":
    main()
