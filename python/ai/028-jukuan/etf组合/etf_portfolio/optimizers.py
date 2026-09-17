"""组合优化求解器。

目标函数：
    1. neg_sharpe - 负夏普比率（最大化）
    2. neg_calmar - 负卡玛比率（最大化），MDD 路径依赖显式计算
    3. portfolio_variance - 组合方差（最小化）

约束：单只 w ∈ [w_min, w_max]，Σ w = 1。
求解器：scipy.optimize.minimize + SLSQP。
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Optional


# ---------------------------------------------------------------------------
# 目标函数
# ---------------------------------------------------------------------------

def neg_sharpe(w: np.ndarray, mu: np.ndarray, cov: np.ndarray, rf: float = 0.025) -> float:
    """负夏普 = -(w·μ - rf) / sqrt(w·Σw)。"""
    port_ret = float(w @ mu)
    port_vol = float(np.sqrt(max(w @ cov @ w, 1e-12)))
    return -(port_ret - rf) / port_vol


def neg_calmar(w: np.ndarray, returns_matrix: pd.DataFrame, freq: int = 252) -> float:
    """负卡玛 = -年化收益 / |最大回撤|。

    注意：MDD 是路径依赖指标，必须用训练窗口的日收益序列构造组合收益。
    """
    port_ret = (returns_matrix.values * w).sum(axis=1)
    nav = np.exp(np.cumsum(port_ret))
    running_max = np.maximum.accumulate(nav)
    dd = (nav - running_max) / running_max
    mdd = float(-dd.min())
    ann_ret = float(port_ret.mean() * freq)
    if mdd < 1e-6:
        return 0.0
    return -ann_ret / mdd


def portfolio_variance(w: np.ndarray, cov: np.ndarray) -> float:
    """组合方差。"""
    return float(w @ cov @ w)


# ---------------------------------------------------------------------------
# 通用求解包装
# ---------------------------------------------------------------------------

def optimize_portfolio(
    mu: np.ndarray,
    cov: np.ndarray,
    returns_for_calmar: Optional[pd.DataFrame] = None,
    objective: str = "sharpe",
    w_max: float = 0.30,
    w_min: float = 0.0,
    rf: float = 0.025,
    ftol: float = 1e-9,
    maxiter: int = 500,
) -> np.ndarray:
    """通用 SLSQP 优化。

    Args:
        mu: (n,) 年化均值向量。
        cov: (n, n) 年化协方差矩阵。
        returns_for_calmar: 仅 `objective='calmar'` 时使用，(T, n) 日收益矩阵。
        objective: 'sharpe' / 'calmar' / 'minvar'。
        w_max, w_min: 单只权重上下界。
        rf: 无风险利率（仅 sharpe 用）。
        ftol: 收敛容差。
        maxiter: 最大迭代。

    Returns:
        (n,) 权重向量（和为 1）。SLSQP 失败则回退到等权。
    """
    n = len(mu)
    bounds = [(w_min, w_max)] * n
    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]

    if objective == "sharpe":
        obj = lambda w: neg_sharpe(w, mu, cov, rf=rf)
    elif objective == "calmar":
        if returns_for_calmar is None:
            raise ValueError("calmar 目标必须提供 returns_for_calmar")
        obj = lambda w: neg_calmar(w, returns_for_calmar)
    elif objective == "minvar":
        obj = lambda w: portfolio_variance(w, cov)
    else:
        raise ValueError(f"未知 objective: {objective}")

    w0 = np.ones(n) / n
    res = minimize(
        obj, w0, method="SLSQP", bounds=bounds, constraints=constraints,
        options={"ftol": ftol, "maxiter": maxiter},
    )
    if res.success and np.all(res.x >= -1e-6) and np.all(res.x <= 1 + 1e-6):
        w = np.clip(res.x, 0, None)
        return w / w.sum()
    # 失败回退
    return np.ones(n) / n
