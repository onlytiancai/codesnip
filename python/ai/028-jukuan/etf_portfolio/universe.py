"""候选 ETF 池预过滤。

筛选逻辑：
    1. 上市满 N 年（默认 5 年）
    2. 截至筛选日仍在市（end_date 远在未来）
    3. 近一年日均成交额 > 阈值（默认 5000 万）
    4. 两两相关系数 > 阈值时，保留近一年成交额更高者（默认 0.92）

数据源：聚宽研究环境 `get_security_info` / `get_price`，须在 notebook 内调用。
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Iterable

import pandas as pd


# ---------------------------------------------------------------------------
# 数据类
# ---------------------------------------------------------------------------

@dataclass
class CandidateETFs:
    """预过滤结果。

    Attributes:
        kept: 通过全部筛选的 ETF 代码列表（按原序）。
        dropped: 被剔除的 {code: reason} 字典。
        corr_matrix: 相关矩阵（仅含 kept），None 表示未做相关去重。
    """

    kept: list[str]
    dropped: dict[str, str]
    corr_matrix: pd.DataFrame | None = None


# ---------------------------------------------------------------------------
# 加载初筛清单
# ---------------------------------------------------------------------------

def load_initial_universe(json_path: str | Path) -> list[dict]:
    """从 `etf_universe.json` 加载手工整理的初筛池。

    返回元素格式：{'code': str, 'name': str, 'category': str, 'list_date': str}。
    """
    with open(json_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    return cfg["etfs"]


# ---------------------------------------------------------------------------
# 步骤 1+2：上市年限 + 在市状态
# ---------------------------------------------------------------------------

def filter_by_listing(
    codes: Iterable[str],
    as_of: date | str,
    min_years: float = 5.0,
) -> tuple[list[str], dict[str, str]]:
    """按上市年限 + 仍在市过滤。

    Args:
        codes: 候选代码列表。
        as_of: 筛选基准日（datetime/date 或 'YYYY-MM-DD'）。
        min_years: 最小上市年限。

    Returns:
        (kept, dropped_reason)。
    """
    if isinstance(as_of, str):
        as_of = datetime.strptime(as_of, "%Y-%m-%d").date()
    kept, dropped = [], {}
    for code in codes:
        try:
            info = get_security_info(code)  # noqa: F821 (聚宽 magic)
        except Exception as e:  # pragma: no cover - 聚宽内核专属
            dropped[code] = f"get_security_info 失败: {e}"
            continue
        if info.end_date != date(2200, 1, 1):
            dropped[code] = f"已退市 (end_date={info.end_date})"
            continue
        years = (as_of - info.start_date).days / 365.25
        if years < min_years:
            dropped[code] = f"上市 {years:.2f} 年 < {min_years}"
            continue
        kept.append(code)
    return kept, dropped


# ---------------------------------------------------------------------------
# 步骤 3：日均成交额
# ---------------------------------------------------------------------------

def filter_by_turnover(
    codes: Iterable[str],
    as_of: date | str,
    min_avg_money: float = 5.0e7,
    lookback_days: int = 250,
) -> tuple[list[str], dict[str, str], dict[str, float]]:
    """按近一年日均成交额过滤。

    Args:
        codes: 候选代码列表。
        as_of: 筛选基准日。
        min_avg_money: 最小日均成交额（元）。
        lookback_days: 回看天数。

    Returns:
        (kept, dropped_reason, adv_map) - adv_map 是代码 → 日均成交额。
    """
    if isinstance(as_of, str):
        as_of = datetime.strptime(as_of, "%Y-%m-%d").date()
    kept, dropped = [], {}
    adv_map: dict[str, float] = {}
    for code in codes:
        try:
            px = get_price(  # noqa: F821 (聚宽 magic)
                code,
                end_date=as_of,
                count=lookback_days,
                frequency="daily",
                fields=["money"],
                skip_paused=True,
                panel=False,
            )
        except Exception as e:  # pragma: no cover
            dropped[code] = f"get_price 失败: {e}"
            continue
        if px is None or px.empty:
            dropped[code] = "无成交数据"
            continue
        avg_money = float(px["money"].mean())
        adv_map[code] = avg_money
        if avg_money < min_avg_money:
            dropped[code] = f"日均成交 {avg_money/1e6:.1f} M < {min_avg_money/1e6:.1f} M"
            continue
        kept.append(code)
    return kept, dropped, adv_map


# ---------------------------------------------------------------------------
# 步骤 4：相关去重
# ---------------------------------------------------------------------------

def fetch_returns_for_codes(
    codes: Iterable[str],
    as_of: date | str,
    lookback_days: int = 750,
) -> pd.DataFrame:
    """拉一批 ETF 的日收益矩阵（按 inner join 对齐共同日期）。

    返回 columns=codes, index=日期, values=日收益率（pct_change, dropna）。

    注：用 750 天（约 3 年）日收益估相关矩阵，比 60 个月窗更稳。
    """
    if isinstance(as_of, str):
        as_of = datetime.strptime(as_of, "%Y-%m-%d").date()
    frames = []
    for code in codes:
        px = get_price(  # noqa: F821
            code,
            end_date=as_of,
            count=lookback_days,
            frequency="daily",
            fields=["close"],
            skip_paused=True,
            fq="pre",
            panel=False,
        )
        if px is None or px.empty:
            continue
        ret = px["close"].pct_change().rename(code)
        frames.append(ret)
    if not frames:
        return pd.DataFrame()
    wide = pd.concat(frames, axis=1).dropna(how="any")
    return wide


def drop_high_corr(
    codes: Iterable[str],
    returns: pd.DataFrame,
    adv_map: dict[str, float],
    corr_thresh: float = 0.92,
) -> tuple[list[str], pd.DataFrame]:
    """两两相关系数 > 阈值时，保留日均成交更高者。

    返回 (final_codes, corr_matrix)。
    """
    if returns.empty:
        return list(codes), pd.DataFrame()
    corr = returns.corr()
    survivors = set(corr.columns)
    pairs = []
    for i, a in enumerate(corr.columns):
        for b in corr.columns[i + 1:]:
            if a in survivors and b in survivors and corr.loc[a, b] > corr_thresh:
                pairs.append((a, b, float(corr.loc[a, b])))
    for a, b, _rho in pairs:
        if a not in survivors or b not in survivors:
            continue
        adv_a = adv_map.get(a, 0.0)
        adv_b = adv_map.get(b, 0.0)
        drop = a if adv_a < adv_b else b
        survivors.discard(drop)
    return sorted(survivors), corr


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def filter_candidates(
    initial_universe: list[dict],
    as_of: date | str,
    min_years: float = 5.0,
    min_avg_money: float = 5.0e7,
    corr_thresh: float = 0.92,
    lookback_days: int = 250,
    corr_lookback_days: int = 750,
    verbose: bool = True,
) -> CandidateETFs:
    """完整预过滤流水线。

    Args:
        initial_universe: 来自 `etf_universe.json` 的初筛池。
        as_of: 筛选基准日。
        min_years: 最小上市年限。
        min_avg_money: 最小日均成交额（元）。
        corr_thresh: 相关去重阈值。
        lookback_days: 成交额回看天数。
        corr_lookback_days: 相关矩阵回看天数。
        verbose: 是否打印剔除明细。

    Returns:
        CandidateETFs。
    """
    codes = [x["code"] for x in initial_universe]
    if verbose:
        print(f"[universe] 候选池 {len(codes)} 只，基准日 {as_of}")

    # 1+2. 上市 + 在市
    kept, dropped = filter_by_listing(codes, as_of, min_years)
    if verbose:
        print(f"[universe] 上市 + 在市过滤后剩 {len(kept)} 只（剔除 {len(dropped)}）")

    # 3. 成交额
    kept2, dropped2, adv_map = filter_by_turnover(kept, as_of, min_avg_money, lookback_days)
    dropped.update(dropped2)
    if verbose:
        print(f"[universe] 成交额过滤后剩 {len(kept2)} 只（剔除 {len(dropped2)}）")

    # 4. 相关去重
    returns = fetch_returns_for_codes(kept2, as_of, corr_lookback_days)
    final, corr = drop_high_corr(kept2, returns, adv_map, corr_thresh)
    if verbose:
        print(f"[universe] 相关去重后剩 {len(final)} 只")

    # 记录相关去重的剔除
    if not returns.empty:
        for c in kept2:
            if c not in final:
                # 找出和谁相关
                rho_max = corr[c].drop(c).max() if c in corr.columns else 0.0
                peer = corr[c].drop(c).idxmax() if c in corr.columns else "?"
                dropped[c] = f"与 {peer} 相关 {rho_max:.3f} > {corr_thresh}"
                if peer in dropped and "相关" in dropped[peer]:
                    dropped.pop(peer, None)
    return CandidateETFs(kept=final, dropped=dropped, corr_matrix=corr)
