"""数据加载与收益转换。

设计原则：
    1. 所有 `get_price` 调用在 notebook 内通过聚宽 magic 完成；本模块提供编排函数。
    2. 拉取结果落盘到 parquet（key=起始日_截止日_代码列表hash），后续阶段直接读盘，避免重复拉数据。
    3. 收益转换：`pct_change` / `np.log` / 月末重采样，统一用日频。
"""
import hashlib
from datetime import date, datetime
from pathlib import Path
from typing import Iterable, List, Optional, Union

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 拉价
# ---------------------------------------------------------------------------

def fetch_prices(
    codes: Iterable[str],
    end_date: Union[date, str],
    lookback_days: int = 3 * 365,
    start_date: Union[date, Optional[str]] = None,
    fields: Optional[List[str]] = None,
) -> pd.DataFrame:
    """批量拉 ETF 日收盘价（DataFrame: index=日期, columns=代码）。

    Args:
        codes: ETF 代码列表。
        end_date: 截止日。
        lookback_days: 回看天数（仅在 start_date 为 None 时生效）。
        start_date: 起始日；如提供则覆盖 lookback_days。
        fields: 字段列表，默认 ['close']。
    """
    if fields is None:
        fields = ["close"]
    if isinstance(end_date, str):
        end_date = datetime.strptime(end_date, "%Y-%m-%d").date()
    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date, "%Y-%m-%d").date()

    frames = []
    for code in codes:
        kwargs = dict(
            security=code,
            end_date=end_date,
            frequency="daily",
            fields=fields,
            skip_paused=True,
            fq="pre",
            panel=False,
        )
        if start_date is not None:
            kwargs["start_date"] = start_date
            kwargs.pop("count", None)
        else:
            kwargs["count"] = lookback_days
        px = get_price(**kwargs)  # noqa: F821 (聚宽 magic)
        if px is None or px.empty:
            continue
        # 多字段时取指定
        if isinstance(px, pd.DataFrame) and "close" in fields and len(fields) == 1:
            px = px.rename(columns={"close": code})
            frames.append(px)
        else:
            frames.append(px.rename(columns={f: f"{code}_{f}" for f in px.columns}))
    if not frames:
        return pd.DataFrame()
    wide = pd.concat(frames, axis=1).sort_index()
    # 只保留收盘价列
    close_cols = [c for c in wide.columns if c in list(codes)]
    if close_cols:
        wide = wide[close_cols]
    return wide


# ---------------------------------------------------------------------------
# 收益转换
# ---------------------------------------------------------------------------

def to_returns(prices: pd.DataFrame, log: bool = False) -> pd.DataFrame:
    """价格转日收益。

    Args:
        prices: 宽表 DataFrame。
        log: True 用对数收益 `np.log(p_t / p_{t-1})`；False 用简单收益。
    """
    if log:
        return np.log(prices / prices.shift(1)).dropna(how="all")
    return prices.pct_change().dropna(how="all")


def to_monthly_returns(returns_daily: pd.DataFrame) -> pd.DataFrame:
    """日收益 → 月收益（按月累加对数收益 = 简单月复利）。"""
    # 用对数相加 = 简单月收益（仅在 log 收益下精确；这里用 (1+r).prod()-1 保持一致）
    monthly = (1 + returns_daily).resample("M").prod() - 1
    return monthly.dropna(how="all")


# ---------------------------------------------------------------------------
# Parquet 缓存
# ---------------------------------------------------------------------------

def _hash_codes(codes: Iterable[str]) -> str:
    joined = "|".join(sorted(codes))
    return hashlib.md5(joined.encode("utf-8")).hexdigest()[:8]


def save_parquet(df: pd.DataFrame, path: Union[Path, str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)


def load_parquet(path: Union[Path, str]) -> pd.DataFrame:
    return pd.read_parquet(path)


def cache_key(codes: Iterable[str], end_date: Union[date, str]) -> str:
    if isinstance(end_date, date):
        end_date = end_date.isoformat()
    return _hash_codes(codes) + "_" + end_date
