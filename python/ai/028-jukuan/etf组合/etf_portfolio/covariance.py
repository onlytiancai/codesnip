"""协方差矩阵估计。

主要内容：
    1. ledoit_wolf_cov - 调用 sklearn 的 LedoitWolf 收缩估计（首选）
    2. manual_ledoit_wolf - 手写 Ledoit-Wolf 单因子收缩（备选）
    3. estimate_mu - 指数加权均值（EWM），缓解历史均值噪声
"""
import numpy as np
import pandas as pd

try:
    from sklearn.covariance import LedoitWolf
    _HAS_SKLEARN = True
except Exception:  # pragma: no cover
    _HAS_SKLEARN = False


# ---------------------------------------------------------------------------
# Ledoit-Wolf 协方差
# ---------------------------------------------------------------------------

def ledoit_wolf_cov(returns_df: pd.DataFrame) -> pd.DataFrame:
    """对日收益矩阵做 Ledoit-Wolf 收缩估计。

    Args:
        returns_df: (T, n) DataFrame，行为日期，列为资产。
    Returns:
        (n, n) DataFrame。
    """
    if not _HAS_SKLEARN:
        return manual_ledoit_wolf(returns_df)
    arr = returns_df.values
    lw = LedoitWolf().fit(arr)
    return pd.DataFrame(lw.covariance_, index=returns_df.columns, columns=returns_df.columns)


def manual_ledoit_wolf(returns_df: pd.DataFrame) -> pd.DataFrame:
    """手写 Ledoit-Wolf 收缩到单因子（平均方差）。

    Σ_shrunk = δ * F + (1 - δ) * S，其中 F = μ * I，μ = trace(S)/n。
    δ 由 Ledoit & Wolf (2004) 最优公式给出。
    """
    X = returns_df.values
    T, n = X.shape
    X = X - X.mean(axis=0, keepdims=True)
    S = (X.T @ X) / T
    mu = float(np.trace(S)) / n
    F = mu * np.eye(n)
    # pi_hat: 方差估计的渐近方差
    X2 = X ** 2
    pi_mat = (X2.T @ X2) / T - S ** 2
    pi_hat = float(np.sum(pi_mat)) / n
    # gamma: 目标矩阵与样本矩阵的"距离"
    diff = S - F
    gamma = float(np.linalg.norm(diff, "fro") ** 2) / n
    if pi_hat <= 0:
        delta = 0.0
    else:
        delta = max(0.0, min(1.0, (pi_hat - gamma) / (pi_hat * (T - 1) / T)))
    shrunk = delta * F + (1 - delta) * S
    return pd.DataFrame(shrunk, index=returns_df.columns, columns=returns_df.columns)


# ---------------------------------------------------------------------------
# 均值估计
# ---------------------------------------------------------------------------

def estimate_mu(returns_df: pd.DataFrame, halflife: int = 120, freq: int = 252) -> np.ndarray:
    """指数加权移动均值 → 年化。

    Args:
        returns_df: (T, n) DataFrame 日收益。
        halflife: 半衰期（默认 120 个交易日 ≈ 6 个月）。
        freq: 年化乘数（默认 252）。

    Returns:
        (n,) 年化均值向量。
    """
    ewm = returns_df.ewm(halflife=halflife, adjust=False).mean()
    mu_daily = ewm.iloc[-1].values
    return mu_daily * freq
