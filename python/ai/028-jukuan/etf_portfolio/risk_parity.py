"""风险平价组合（对照组）。

经典风险平价：每只资产对组合风险的边际贡献相等。

    w_i * (Σ w)_i = w_j * (Σ w)_j  (对所有 i, j)

实现：Spinu (2013) 提出的迭代算法，或 Newton 求解。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def risk_parity(cov: np.ndarray, max_iter: int = 1000, tol: float = 1e-8) -> np.ndarray:
    """Newton 迭代求解风险平价权重。

    Args:
        cov: (n, n) 年化协方差矩阵。
        max_iter: 最大迭代次数。
        tol: 收敛容差。

    Returns:
        (n,) 权重向量（和为 1）。
    """
    n = cov.shape[0]
    w = np.ones(n) / n
    for _ in range(max_iter):
        # 组合方差与边际风险贡献
        port_var = float(w @ cov @ w)
        if port_var < 1e-12:
            return np.ones(n) / n
        sigma_p = np.sqrt(port_var)
        mrc = (cov @ w) / sigma_p           # 边际风险贡献
        rc = w * mrc                         # 风险贡献
        # 目标：所有 rc 相等 → rc / w 相等
        target = rc.sum() / n                # 平均风险贡献
        # Newton 更新：w ← w * target / rc
        ratio = np.divide(target, rc, out=np.ones_like(rc), where=rc > 1e-12)
        w_new = w * ratio
        # 归一化
        s = w_new.sum()
        if s <= 0:
            return np.ones(n) / n
        w_new = w_new / s
        if np.linalg.norm(w_new - w, ord=1) < tol:
            return w_new
        w = w_new
    return w
