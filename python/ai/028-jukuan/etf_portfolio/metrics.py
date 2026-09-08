"""指标与衰减率计算。

封装 empyrical 的常用指标；提供衰减率（样本内 vs 样本外夏普）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import empyrical as ep


def full_metrics(returns: pd.Series, rf: float = 0.025) -> dict:
    """对组合日收益 Series 计算一组常用指标。

    Args:
        returns: 日收益 Series。
        rf: 无风险利率（年化）。

    Returns:
        dict with keys: annual_return, annual_vol, sharpe, sortino, calmar,
        max_drawdown, stability.
    """
    if returns is None or len(returns) == 0:
        return {}
    return {
        "annual_return": float(ep.annual_return(returns)),
        "annual_vol":    float(ep.annual_volatility(returns)),
        "sharpe":        float(ep.sharpe_ratio(returns, risk_free=rf)),
        "sortino":       float(ep.sortino_ratio(returns, risk_free=rf)),
        "calmar":        float(ep.calmar_ratio(returns)),
        "max_drawdown":  float(ep.max_drawdown(returns)),
        "stability":     float(ep.stability_of_timeseries(returns)),
    }


def decay_rate(is_sharpe: float, oos_sharpe: float) -> float:
    """衰减率 = (IS Sharpe - OOS Sharpe) / |IS Sharpe|。

    解读：
        < 30%   泛化优秀
        30–60%  尚可
        > 70%   严重过拟合，建议丢弃
    """
    if not np.isfinite(is_sharpe) or abs(is_sharpe) < 1e-6:
        return np.nan
    return (is_sharpe - oos_sharpe) / abs(is_sharpe)


def classify_decay(decay: float) -> str:
    """衰减率分档。"""
    if not np.isfinite(decay):
        return "未知"
    if decay < 0.30:
        return "优秀"
    if decay < 0.60:
        return "尚可"
    return "过拟合"
