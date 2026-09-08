"""Walk-Forward 滚动样本外（WFA）主框架。

核心流程：
    1. 数据按月分桶，调仓日 = 每月最后一个交易日
    2. 每个调仓日：取过去 train_months 月数据 → 估计 μ/Σ → 求最优权重
    3. 持仓至下一调仓日，记录 OOS 收益
    4. 输出 IS / OOS 指标 + 衰减率 + 权重历史

参数：
    train_months: 训练窗口（默认 60）
    test_months:  OOS 推进步长（默认 1）
    w_max:        单只权重上限（默认 0.30）
    objectives:   目标列表（默认三个全跑）
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from tqdm import tqdm

from .covariance import ledoit_wolf_cov, estimate_mu
from .optimizers import optimize_portfolio
from .metrics import full_metrics, decay_rate


# ---------------------------------------------------------------------------
# 数据类
# ---------------------------------------------------------------------------

@dataclass
class WFAOutput:
    """WFA 运行结果。"""

    metrics: pd.DataFrame            # 每个 (date, objective) 一行：IS/OOS 指标 + 衰减率
    weights: pd.DataFrame            # 每个 (date, objective) 一行：每只 ETF 的权重
    oos_returns: pd.DataFrame        # 每个 (date, objective) 一列：OOS 期间组合日收益


# ---------------------------------------------------------------------------
# 主函数
# ---------------------------------------------------------------------------

def walk_forward(
    prices: pd.DataFrame,
    objectives: Iterable[str] = ("sharpe", "calmar", "minvar"),
    train_months: int = 60,
    test_months: int = 1,
    w_max: float = 0.30,
    halflife: int = 120,
    rf: float = 0.025,
    show_progress: bool = True,
) -> WFAOutput:
    """滚动样本外优化。

    Args:
        prices: 宽表 (T, n) 收盘价，前复权。
        objectives: 优化目标，可选 'sharpe' / 'calmar' / 'minvar'。
        train_months: 训练窗口（月数）。
        test_months:  OOS 推进步长（月数）。
        w_max: 单只权重上限。
        halflife: EWM 半衰期（仅 sharpe/minvar 用）。
        rf: 无风险利率。
        show_progress: 是否显示 tqdm 进度。

    Returns:
        WFAOutput。
    """
    daily_ret = prices.pct_change().dropna()
    monthly_close = prices.resample("ME").last().dropna(how="all")
    # 月末调仓日：每月最后一个交易日
    rebalance_dates = monthly_close.index.tolist()
    # 起点：第一个 rebalance 必须保证之前有 train_months 月数据
    first_valid_idx = train_months
    if first_valid_idx >= len(rebalance_dates):
        raise ValueError(
            f"数据不足：仅有 {len(rebalance_dates)} 个月末点，"
            f"但需要 train_months={train_months}"
        )
    rebalance_dates = rebalance_dates[first_valid_idx:]

    objectives = list(objectives)
    metrics_rows = []
    weight_rows = []
    oos_ret_dict = {obj: [] for obj in objectives}

    iter_dates = tqdm(rebalance_dates, desc="WFA 滚动") if show_progress else rebalance_dates
    for i, rebal_date in enumerate(iter_dates):
        # 训练窗口：rebal_date 前推 train_months 月
        train_end = rebal_date
        train_start = rebal_date - pd.DateOffset(months=train_months)
        train_prices = prices.loc[train_start:train_end].iloc[:-1]   # 不含 rebal 当日
        if len(train_prices) < 60:
            continue
        train_ret = train_prices.pct_change().dropna()
        # 估计参数
        try:
            cov = ledoit_wolf_cov(train_ret)
            mu = estimate_mu(train_ret, halflife=halflife)
        except Exception as e:
            print(f"[WFA] 协方差估计失败 ({rebal_date.date()}): {e}")
            continue

        # OOS 测试窗口：rebal_date 到下次调仓日
        if i + 1 < len(rebalance_dates):
            test_end = rebalance_dates[i + 1]
        else:
            test_end = rebal_date + pd.DateOffset(months=test_months)
        test_prices = prices.loc[rebal_date:test_end].iloc[1:]
        if len(test_prices) < 2:
            continue
        test_ret = test_prices.pct_change().dropna()

        # IS 组合收益（按训练期）
        is_port = (train_ret * 0).copy()  # placeholder
        is_ret = pd.Series(0.0, index=train_ret.index)
        for obj in objectives:
            # 优化权重
            if obj == "calmar":
                w = optimize_portfolio(
                    mu, cov, returns_for_calmar=train_ret,
                    objective="calmar", w_max=w_max,
                )
            else:
                w = optimize_portfolio(
                    mu, cov, objective=obj, w_max=w_max, rf=rf,
                )
            # IS 收益
            is_port_ret = (train_ret * w).sum(axis=1)
            is_m = full_metrics(is_port_ret, rf=rf)
            # OOS 收益
            oos_port_ret = (test_ret * w).sum(axis=1)
            oos_m = full_metrics(oos_port_ret, rf=rf)

            metrics_rows.append({
                "date":      rebal_date,
                "objective": obj,
                "is_sharpe": is_m.get("sharpe"),
                "is_calmar": is_m.get("calmar"),
                "is_mdd":    is_m.get("max_drawdown"),
                "is_annret": is_m.get("annual_return"),
                "oos_sharpe": oos_m.get("sharpe"),
                "oos_calmar": oos_m.get("calmar"),
                "oos_mdd":    oos_m.get("max_drawdown"),
                "oos_annret": oos_m.get("annual_return"),
                "decay":      decay_rate(is_m.get("sharpe", 0.0), oos_m.get("sharpe", 0.0)),
            })
            weight_rows.append({
                "date":      rebal_date,
                "objective": obj,
                **{code: float(wc) for code, wc in zip(prices.columns, w)},
            })
            oos_ret_dict[obj].append(
                oos_port_ret.rename(rebal_date.strftime("%Y-%m-%d"))
            )

    metrics = pd.DataFrame(metrics_rows)
    weights = pd.DataFrame(weight_rows)
    oos_returns = pd.concat(
        [pd.concat(vals, axis=1) for vals in oos_ret_dict.values() if vals],
        axis=1,
        keys=[obj for obj, vals in oos_ret_dict.items() if vals],
    ) if any(oos_ret_dict.values()) else pd.DataFrame()

    return WFAOutput(metrics=metrics, weights=weights, oos_returns=oos_returns)


# ---------------------------------------------------------------------------
# 稳健度评分（路线 B 核心）
# ---------------------------------------------------------------------------

def score_robustness(
    weights: pd.DataFrame,
    min_weight: float = 0.01,
) -> pd.DataFrame:
    """对每个 ETF 计算稳健度评分。

    Args:
        weights: wfa 输出的 weight DataFrame（含 date / objective + 各 ETF 列）。
        min_weight: 判定"被赋权"的最小阈值。

    Returns:
        DataFrame, index=ETF, columns=[freq, avg_weight, stability, n_obj]。
    """
    code_cols = [c for c in weights.columns if c not in ("date", "objective")]
    if not code_cols:
        return pd.DataFrame()

    # 每个 ETF 在所有 (date, objective) 组合中的统计
    nonzero_freq = (weights[code_cols] > min_weight).mean()
    non_zero = weights[code_cols].where(weights[code_cols] > min_weight)
    avg_weight = non_zero.mean()
    weight_std = non_zero.std()
    stability = avg_weight / weight_std.replace(0, np.nan)
    # 出现在多少个 objective 中
    n_obj = (
        weights[code_cols]
        .groupby(weights["objective"])
        .apply(lambda d: (d > min_weight).any())
        .sum()
    )

    out = pd.DataFrame({
        "freq":       nonzero_freq,
        "avg_weight": avg_weight,
        "stability":  stability,
        "n_obj":      n_obj,
    }).dropna(subset=["freq"]).sort_values(
        ["freq", "avg_weight"], ascending=False
    )
    return out
