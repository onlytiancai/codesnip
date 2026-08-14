"""批量获取宽基 ETF 日 K 线（前复权），落到 .cache/klines/ 下 CSV。

用法示例
--------
# 默认：最近 3 年，全部 ETF
python fetch_etf_klines.py

# 指定时间窗
python fetch_etf_klines.py --start-date 2024-01-01 --end-date 2025-12-31

# 增量：从每个 ETF 已落盘文件的最新一日往后补到昨天
python fetch_etf_klines.py --incremental

# 往前扩展：把现有 CSV 补到最近 10 年，prepend 到现有数据（不动现有数据）
python fetch_etf_klines.py --lookback-years 10

# 质量检查：只扫描 .cache/klines/*.csv，不下载
python fetch_etf_klines.py --quality-check

# 质量检查 + 子区间 + 出图
python fetch_etf_klines.py --quality-check --qc-start-date 2024-01-01 --qc-end-date 2024-12-31 --qc-plot

# 只取两只 ETF
python fetch_etf_klines.py --codes 510050.SH,159915.SZ
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from eltdx import TdxClient

# ---- ETF 列表 ---------------------------------------------------------------
# 全市场宽基 ETF，覆盖大盘/中盘/小盘/创业板/科创板
DEFAULT_ETFS: list[str] = [
    "510050.SH",   # 华夏上证50ETF        上证50
    "510300.SH",   # 华泰柏瑞沪深300ETF   沪深300
    "510500.SH",   # 南方中证500ETF       中证500
    "159845.SZ",   # 富国中证1000ETF      中证1000
    "159915.SZ",   # 易方达创业板ETF      创业板指
    "588000.SH",   # 华夏科创50ETF        科创50
    "511010.SH",   # 国债ETF              避险资产（熊市替代空仓）
    "000300.SH",   # 沪深300指数          对照基准（指数本身）
]

# 指数代码前缀（CSI / 深证综指）：eltdx 解析必须 kind="index"，否则
# 解码层会因 record 格式不同抛 `invalid kline date`。
# 000xxx.SH = 上证综指/中证/上证风格指数；399xxx.SZ = 深证综指/创业板等。
_INDEX_PREFIXES: tuple[str, ...] = ("000", "399")


def _kind_for(combined: str) -> str:
    """`000300.SH` -> `index`，其余 -> `stock`。"""
    code = combined.split(".", 1)[0]
    return "index" if code.startswith(_INDEX_PREFIXES) else "stock"


# 每次拉一页最多 800 根 K 线，eltdx 的硬上限
PAGE_SIZE = 800

# 默认数据目录
DEFAULT_DATA_DIR = Path(__file__).parent / ".cache" / "klines"

# 默认出图目录（--qc-plot）
DEFAULT_PLOT_DIR = Path(__file__).parent / ".cache" / "qc_plots"

# 中国时区（eltdx 返回的 datetime 用的就是这个 tz）
CST = timezone(timedelta(hours=8))


# ---------------------------------------------------------------------------
# 名称查找：从 .cache/eltdx_codes.json（cache_codes.py 生成的）读
# ---------------------------------------------------------------------------
def _load_name_lookup() -> dict[str, str]:
    """`sh510050` -> `上证50ETF华夏`，找不到返回 `?`。

    复用 cache_codes.py 写下的本地代码表，省一次网络请求。
    """
    cache_path = Path(__file__).parent / ".cache" / "eltdx_codes.json"
    if not cache_path.exists():
        return {}
    try:
        data = json.loads(cache_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    return {
        c["full_code"]: c["name"]
        for c in data.get("codes", [])
        if c.get("full_code") and c.get("name")
    }


NAMES: dict[str, str] = _load_name_lookup()


def _name_for(combined: str) -> str:
    """`510050.SH` -> `上证50ETF华夏`（缓存里没有返回 `?`）。"""
    return NAMES.get(to_eltdx_code(combined), "?")


def _label(combined: str, width: int = 26) -> str:
    """`510050.SH 上证50ETF华夏` 左对齐到指定宽度（中文按字符计）。"""
    name = _name_for(combined)
    return f"{combined} {name}".ljust(width)


# ---------------------------------------------------------------------------
# 代码转换
# ---------------------------------------------------------------------------
def to_eltdx_code(combined: str) -> str:
    """`510050.SH` -> `sh510050`，大小写无依赖。"""
    code, _, market = combined.partition(".")
    if not market:
        raise ValueError(f"无效代码（缺少市场后缀）: {combined}")
    return f"{market.lower()}{code}"


def csv_path(data_dir: Path, combined: str) -> Path:
    return data_dir / f"{combined}.csv"


# ---------------------------------------------------------------------------
# 单只 ETF 拉取
# ---------------------------------------------------------------------------
@dataclass
class FetchResult:
    code: str
    fetched: int = 0          # 本次新拉取的 bar 数
    kept: int = 0             # 落盘后文件总行数
    skipped_dup: int = 0      # 增量时去重丢弃的旧重复日期
    error: str | None = None


def _bar_to_row(bar) -> dict:
    """KlineBar -> dict；统一保留毫精度价格，转 float 写 CSV。"""
    t: datetime = bar.time
    if t.tzinfo is not None:
        t = t.astimezone(CST)
    return {
        "date": t.strftime("%Y-%m-%d"),
        "open": float(bar.open),
        "high": float(bar.high),
        "low": float(bar.low),
        "close": float(bar.close),
        "volume": int(bar.volume_lots),    # volume_lots = 手数（100 股/手），比 raw 直观
        "amount": float(bar.amount),
    }


def _is_placeholder(row: dict) -> bool:
    """eltdx 在"今天"还没收盘时，会返回一行占位：OHLC 全部等于前收，
    volume=0, amount=0。这行不是真实数据，应当丢弃。
    """
    return (
        row["volume"] == 0
        and row["amount"] == 0.0
        and row["open"] == row["high"] == row["low"] == row["close"]
    )


def _strip_trailing_placeholders(rows: list[dict]) -> list[dict]:
    """从尾部连续剔除占位行。"""
    while rows and _is_placeholder(rows[-1]):
        rows.pop()
    return rows


def fetch_one(
    client: TdxClient,
    combined: str,
    start_d: date,
    end_d: date,
) -> pd.DataFrame:
    """拉取 [start_d, end_d] 区间的全部日 K（按 800/页翻页）。

    翻页方向：eltdx 的 start 参数语义是"跳过最新的 N 根"，因此 start=0 取
    最新一页；series.bars 内部按**时间升序**排列（最旧 -> 最新）。
    所以下一页要 start += count，越往后越旧；遇到"翻页拉到的所有 bar
    都早于 start_d"就可以停。

    末尾占位行兜底：即使 end_d 设到今天，eltdx 在盘中也会返回一行"今日"
    占位（OHLC=前收, vol=0, amt=0）；本函数在 finalize 之前会把它剔掉。
    """
    eltdx_code = to_eltdx_code(combined)
    kind = _kind_for(combined)
    # 指数没有分红/拆分，前复权无意义；eltdx 对 kind="index" 用 `none` 也不会出错。
    adjust = "qfq" if kind == "stock" else "none"
    rows: list[dict] = []

    start = 0
    while True:
        series = client.get_kline(
            "day",
            eltdx_code,
            start=start,
            count=PAGE_SIZE,
            adjust=adjust,
            kind=kind,
        )
        bars = series.bars
        if not bars:
            break

        page_min = date.fromisoformat(_bar_to_row(bars[0])["date"])
        # 整页都早于 start_d -> 整页丢弃，且后续页只会更旧，可以停
        if page_min > end_d:
            # 整页都比 end_d 晚（理论上不会，因为 start=0 是最新页）
            pass
        elif page_min >= start_d or any(
            date.fromisoformat(_bar_to_row(b)["date"]) >= start_d for b in bars
        ):
            for bar in bars:
                row = _bar_to_row(bar)
                bar_date = date.fromisoformat(row["date"])
                if bar_date < start_d or bar_date > end_d:
                    continue
                rows.append(row)
        else:
            # 整页都早于 start_d，无需再翻
            break

        start += len(bars)
        if len(bars) < PAGE_SIZE:
            break
        # 20 年 ≈ 5040 bar；用 8000 上限兜底极端长历史（指数回溯到 2005）。
        if start > 8000:
            break

    rows = _strip_trailing_placeholders(rows)
    return _finalize(rows)


def _finalize(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(
        rows,
        columns=["date", "open", "high", "low", "close", "volume", "amount"],
    )
    if df.empty:
        return df
    df = (
        df.drop_duplicates(subset=["date"], keep="last")
        .sort_values("date")
        .reset_index(drop=True)
    )
    return df


def save_csv(df_new: pd.DataFrame, path: Path, incremental: bool) -> tuple[int, int]:
    """落盘；返回 (新增条数, 去重丢弃数)。

    - `incremental=False`：覆盖写
    - `incremental=True`：跟旧数据合并，按 date 去重（旧优先 keep="last"`）
    """
    if not incremental or not path.exists():
        df_new.to_csv(path, index=False)
        return len(df_new), 0

    df_old = pd.read_csv(path)
    kept_old = len(df_old)
    last_old_date = pd.to_datetime(df_old["date"]).max().date() if kept_old else None

    if last_old_date is not None:
        df_new = df_new[pd.to_datetime(df_new["date"]).dt.date > last_old_date]

    merged = pd.concat([df_old, df_new], ignore_index=True)
    merged = merged.drop_duplicates(subset=["date"], keep="last").sort_values("date").reset_index(drop=True)
    merged.to_csv(path, index=False)

    added = len(df_new)
    dup_dropped = (kept_old + len(df_new)) - len(merged)
    return added, dup_dropped


def prepend_csv(df_new: pd.DataFrame, path: Path) -> tuple[int, int]:
    """把 df_new 拼到现有 CSV 前面（新数据在前），按 date 去重（新数据优先）。

    用于 `--lookback-years` 扩展早期数据；现有数据不动。
    返回 (df_new 行数, 去重丢弃行数)。
    """
    if df_new.empty:
        return 0, 0

    if path.exists():
        df_old = pd.read_csv(path)
        kept_old = len(df_old)
    else:
        df_old = pd.DataFrame(columns=df_new.columns)
        kept_old = 0

    # df_new 在前 → keep="first" 保留新数据；如果新旧同日，新值生效（一般行情数据日期
    # 一致则值一致，但理论上 eltdx 翻页边界可能让某些日期恰好重复出现一次）
    merged = pd.concat([df_new, df_old], ignore_index=True)
    merged = merged.drop_duplicates(subset=["date"], keep="first").sort_values("date").reset_index(drop=True)
    merged.to_csv(path, index=False)

    added = len(df_new)
    dup_dropped = (kept_old + len(df_new)) - len(merged)
    return added, dup_dropped


# ---------------------------------------------------------------------------
# 增量模式：决定每只 ETF 的实际起止
# ---------------------------------------------------------------------------
def resolve_incremental_range(path: Path) -> tuple[date, date] | None:
    """读已有 CSV 的最大日期，返回 (start, yesterday)；不存在返回 None。

    截止到昨天：eltdx 在盘中/盘前对"今日"只返回占位行，避开。
    """
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if df.empty:
        return None
    last = pd.to_datetime(df["date"]).max().date()
    return last + timedelta(days=1), date.today() - timedelta(days=1)


# ---------------------------------------------------------------------------
# 质量检查
# ---------------------------------------------------------------------------
@dataclass
class DescriptiveStats:
    """单只 ETF 的描述性统计：缺失值、价格、收益、波动、回撤、成交量。"""

    n: int = 0
    days_span: int = 0           # 起止日期差（日历日）
    missing: dict[str, int] = field(default_factory=dict)   # {列名: NaN 数}

    # close 描述
    close_mean: float = 0.0
    close_std: float = 0.0
    close_min: float = 0.0
    close_max: float = 0.0

    # 日对数收益描述
    ret_mean: float = 0.0        # 日对数收益均值
    ret_std: float = 0.0         # 日对数收益标准差
    ret_min: float = 0.0         # 最差单日收益
    ret_max: float = 0.0         # 最佳单日收益

    # 年化与回撤
    annual_vol: float = 0.0      # 年化波动率 = ret_std * sqrt(252)
    annual_ret: float = 0.0      # 年化收益（几何）
    max_drawdown: float = 0.0    # 最大回撤（负数）
    max_dd_start: str = ""       # 回撤起点（峰值日）
    max_dd_end: str = ""         # 回撤终点（谷底日）

    # 成交量（手数）
    vol_mean: float = 0.0
    vol_median: float = 0.0
    vol_min: float = 0.0
    vol_max: float = 0.0

    @property
    def has_data(self) -> bool:
        return self.n > 0


@dataclass
class QCRow:
    code: str
    rows: int
    first: str | None
    last: str | None
    issues: list[str]
    stats: DescriptiveStats = field(default_factory=DescriptiveStats)


def _descriptive_stats(df: pd.DataFrame) -> DescriptiveStats:
    """从价格序列算描述统计。空 df 返回默认全零 stats。"""
    s = DescriptiveStats()
    s.n = len(df)
    if df.empty:
        return s

    s.days_span = (
        pd.to_datetime(df["date"].iloc[-1]) - pd.to_datetime(df["date"].iloc[0])
    ).days

    # 缺失值
    for col in ["open", "high", "low", "close", "volume", "amount"]:
        if col in df.columns:
            s.missing[col] = int(df[col].isna().sum())

    if "close" not in df.columns:
        return s

    close = df["close"]
    s.close_mean = float(close.mean())
    s.close_std = float(close.std())
    s.close_min = float(close.min())
    s.close_max = float(close.max())

    # 日对数收益（学术标准）
    log_ret = np.log(close / close.shift(1)).dropna()
    if not log_ret.empty:
        s.ret_mean = float(log_ret.mean())
        s.ret_std = float(log_ret.std())
        s.ret_min = float(log_ret.min())
        s.ret_max = float(log_ret.max())
        s.annual_vol = s.ret_std * np.sqrt(252)

        # 年化收益（几何年化）：(end/start)^(252/n) - 1
        # 注意：早期版本用 cum_log * (252/n) 即"连续复利率"近似，
        # 在收益大时（>30%）会显著低估（科创50 翻倍会被报成 +83%）。
        # 这里改回标准几何年化：先 cum_log 再 exp 回来 - 1。
        n_days = len(log_ret)
        if n_days > 0 and close.iloc[0] > 0:
            cum_log = float(np.log(close.iloc[-1] / close.iloc[0]))
            s.annual_ret = float(np.exp(cum_log * (252 / n_days)) - 1)

        # 最大回撤：直接在 close 上算（(close - peak) / peak），单位是百分比，
        # 下界 -100%（资产归零）。不要在 cum_log 上算，因为 cum_log 的起点 0
        # 对应的是被前复权压扁的首日价，qfq 多年的分红回填会让 cum_log 出现
        # 远超真实回撤的"虚高峰"，导致回撤被算成 -200%、-300% 这种不存在的值。
        running_peak = close.cummax()
        drawdown = (close - running_peak) / running_peak
        dd_min = float(drawdown.min())
        s.max_drawdown = dd_min

        dd_end_pos = int(drawdown.idxmin())
        if dd_end_pos > 0:
            peak_pos = int(running_peak.iloc[: dd_end_pos + 1].idxmax())
            s.max_dd_start = str(df["date"].iloc[peak_pos])
            s.max_dd_end = str(df["date"].iloc[dd_end_pos])

    # 成交量
    if "volume" in df.columns:
        v = df["volume"].dropna()
        if not v.empty:
            s.vol_mean = float(v.mean())
            s.vol_median = float(v.median())
            s.vol_min = float(v.min())
            s.vol_max = float(v.max())

    return s


def quality_check(
    path: Path,
    start_date: date | None = None,
    end_date: date | None = None,
) -> QCRow:
    issues: list[str] = []
    code = path.stem

    if not path.exists():
        return QCRow(code=code, rows=0, first=None, last=None, issues=["文件不存在"])

    df = pd.read_csv(path)
    if df.empty:
        return QCRow(code=code, rows=0, first=None, last=None, issues=["文件为空"])

    # 按日期范围过滤（[start, end] 全闭区间）
    if start_date is not None or end_date is not None:
        d = pd.to_datetime(df["date"]).dt.date
        mask = pd.Series(True, index=df.index)
        if start_date is not None:
            mask &= d >= start_date
        if end_date is not None:
            mask &= d <= end_date
        df = df[mask].reset_index(drop=True)
        if df.empty:
            return QCRow(
                code=code, rows=0, first=None, last=None,
                issues=[f"区间 {start_date or '-'} ~ {end_date or '-'} 内无数据"],
            )

    n = len(df)
    first = str(df["date"].iloc[0])
    last = str(df["date"].iloc[-1])

    # 1. 必需列
    need_cols = {"date", "open", "high", "low", "close", "volume", "amount"}
    miss = need_cols - set(df.columns)
    if miss:
        issues.append(f"缺列: {sorted(miss)}")

    # 2. 空值
    null_cols = [c for c in need_cols if c in df.columns and df[c].isna().any()]
    if null_cols:
        issues.append(f"含 NaN 列: {null_cols}")

    # 3. OHLC 关系
    if not df.empty and {"open", "high", "low", "close"}.issubset(df.columns):
        bad_hi = df[df["high"] < df[["open", "close"]].max(axis=1)]
        bad_lo = df[df["low"] > df[["open", "close"]].min(axis=1)]
        if not bad_hi.empty:
            issues.append(f"high < max(O,C) 的行 {len(bad_hi)} 行")
        if not bad_lo.empty:
            issues.append(f"low  > min(O,C) 的行 {len(bad_lo)} 行")

    # 4. 负数/极端
    if "volume" in df.columns:
        neg_v = (df["volume"] < 0).sum()
        if neg_v:
            issues.append(f"volume<0 行 {neg_v}")
        zero_v = (df["volume"] == 0).sum()
        if zero_v > n * 0.3:
            issues.append(f"volume=0 占比 {zero_v/n:.1%}")

    # 5. 重复日期
    dup = df["date"].duplicated().sum()
    if dup:
        issues.append(f"重复日期 {dup} 条")

    # 6. 日期连续性（仅在节假日会有连续空缺；> 30 个日历日断档告警）
    if n > 1:
        dates = pd.to_datetime(df["date"]).sort_values().reset_index(drop=True)
        gaps = dates.diff().dropna().dt.days
        big_gap = gaps[gaps > 30]   # 大于 30 个日历日（含长假）
        if not big_gap.empty:
            worst = big_gap.max()
            issues.append(f"最大日期间隔 {worst} 天 (>30)，疑似断档")

    # 7. 价格极端波动：阈值放到 50%，因为 ETF 前复权遇分红/份额折算
    #    时会出现整段比例重映射（例如 1:5 折算会"涨"400%）。
    if {"open", "close"}.issubset(df.columns):
        prev_close = df["close"].shift(1)
        chg = (df["close"] - prev_close) / prev_close
        extreme = chg[(chg.abs() > 0.50) & prev_close.notna()]
        if not extreme.empty:
            issues.append(
                f"单日涨跌 > 50% 共 {len(extreme)} 行，"
                "可能是复权跳变（正常）或脏数据（需复核）"
            )

    stats = _descriptive_stats(df)
    return QCRow(code=code, rows=n, first=first, last=last, issues=issues, stats=stats)


# ---------------------------------------------------------------------------
# 相关性矩阵（多只 ETF 联合分析）
# ---------------------------------------------------------------------------
# 最少需要多少个交易日才能让 Pearson/Spearman 相关系数有统计意义
# 经验值：30 个观测起步，< 30 时两个点的 Pearson 必为 ±1（任何两点共线），
# 全矩阵会显示成"全是 1.000"的伪相关，没意义。
MIN_OBS_FOR_CORR = 30


def load_close_series(
    data_dir: Path,
    codes: list[str],
    start_date: date | None = None,
    end_date: date | None = None,
) -> pd.DataFrame:
    """读每只 ETF 的 close 序列，按日期 inner join 对齐，返回 (date-indexed) DataFrame。
    缺失或空的 ETF 直接跳过；start_date / end_date 把每只 ETF 各自截到区间内再
    inner join，保证对齐样本来自用户指定的窗口。
    """
    frames: dict[str, pd.Series] = {}
    for c in codes:
        path = csv_path(data_dir, c)
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if df.empty or "close" not in df.columns:
            continue
        if start_date is not None or end_date is not None:
            d = pd.to_datetime(df["date"]).dt.date
            mask = pd.Series(True, index=df.index)
            if start_date is not None:
                mask &= d >= start_date
            if end_date is not None:
                mask &= d <= end_date
            df = df[mask]
            if df.empty:
                continue
        s = pd.Series(
            df["close"].values,
            index=pd.to_datetime(df["date"]),
            name=c,
        )
        frames[c] = s
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames.values(), axis=1, join="inner")
    out.columns = list(frames.keys())
    out.index.name = "date"
    return out


def correlation_matrices(closes: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """对日对数收益算 Pearson / Spearman 相关系数矩阵。"""
    if closes.empty or len(closes) < 2:
        return {}
    log_ret = np.log(closes / closes.shift(1)).dropna()
    if log_ret.empty:
        return {}
    return {
        "pearson": log_ret.corr(method="pearson"),
        "spearman": log_ret.corr(method="spearman"),
        "log_returns": log_ret,
    }


# ---------------------------------------------------------------------------
# QC 输出格式化
# ---------------------------------------------------------------------------
def _fmt_pct(x: float, signed: bool = False) -> str:
    if not np.isfinite(x):
        return "    -"
    sign = "+" if signed and x >= 0 else ""
    return f"{sign}{x * 100:5.2f}%"


def _fmt_num(x: float, decimals: int = 2) -> str:
    if not np.isfinite(x):
        return "    -"
    return f"{x:.{decimals}f}"


def _fmt_int(x: int) -> str:
    if x == 0:
        return "0"
    if x >= 10_000_000:
        return f"{x / 1_000_000:.1f}M"
    if x >= 10_000:
        return f"{x / 1_000:.1f}K"
    return f"{x}"


def _print_one_etf(row: QCRow) -> None:
    """打印单只 ETF 的检查结果 + 描述统计。"""
    tag = "OK " if not row.issues else "WARN"
    issues_str = f"  {'; '.join(row.issues)}" if row.issues else ""
    print(f"[{tag}] {_label(row.code)} rows={row.rows:<6} "
          f"{row.first or '-'} -> {row.last or '-'}{issues_str}")

    if not row.stats.has_data or row.stats.n < 2:
        return

    s = row.stats
    miss_parts = [f"{k}={v}" for k, v in s.missing.items() if v > 0]
    miss_str = ", ".join(miss_parts) if miss_parts else "无"
    print(f"        区间 {s.days_span} 日历日；缺失值 {miss_str}")
    print(f"        close     : mean={_fmt_num(s.close_mean)} "
          f"std={_fmt_num(s.close_std)} "
          f"min={_fmt_num(s.close_min)} max={_fmt_num(s.close_max)}")
    print(f"        日对数收益: mean={_fmt_pct(s.ret_mean, signed=True)} "
          f"std={_fmt_pct(s.ret_std)} "
          f"min={_fmt_pct(s.ret_min, signed=True)} "
          f"max={_fmt_pct(s.ret_max, signed=True)}")
    print(f"        年化      : 收益={_fmt_pct(s.annual_ret, signed=True)} "
          f"波动={_fmt_pct(s.annual_vol)}")
    if s.max_dd_start and s.max_dd_end:
        print(f"        最大回撤  : {_fmt_pct(s.max_drawdown, signed=True)} "
              f"({s.max_dd_start} ~ {s.max_dd_end})")
    # 成交量统一换算成"百万手"
    vm, vmd, vmin, vmax = (s.vol_mean / 1e6, s.vol_median / 1e6,
                            s.vol_min / 1e6, s.vol_max / 1e6)
    print(f"        成交量(百万手): mean={vm:6.2f} "
          f"median={vmd:6.2f} min={vmin:6.2f} max={vmax:6.2f}")


def _print_corr_matrix(name: str, df: pd.DataFrame) -> None:
    """打印相关系数矩阵，对角线 1.000，三位小数。"""
    print(f"\n--- {name} ---")
    codes = df.columns.tolist()
    header = "            " + "  ".join(f"{c:>10}" for c in codes)
    print(header)
    for i, c in enumerate(codes):
        vals = "  ".join(f"{df.iloc[i, j]:>10.3f}" for j in range(len(codes)))
        print(f"{c:<12} {vals}")


def _display_width(s: str) -> int:
    """估算字符串在等宽终端的显示宽度：ASCII 1 列，CJK 2 列。"""
    return sum(2 if ord(c) > 127 else 1 for c in s)


def _pad(s: str, width: int) -> str:
    """按显示宽度左对齐 padding 到 width。"""
    pad_count = width - _display_width(s)
    return s + " " * max(0, pad_count)


def _print_overview_table(rows: list[QCRow]) -> None:
    """紧凑的概述表格：代码 | 名称 | 条数 | 日期范围 | 年化 | 回撤 | 波动。"""
    header_cells = ["代码", "名称", "条数", "日期范围", "年化收益", "最大回撤", "波动率"]
    widths = [12, 18, 6, 28, 9, 9, 8]

    head_line = "  ".join(_pad(c, w) for c, w in zip(header_cells, widths))
    print(head_line)
    print("  ".join("-" * w for w in widths))

    for row in rows:
        s = row.stats
        if s.has_data and s.n >= 2:
            ret = _fmt_pct(s.annual_ret, signed=True)
            dd = _fmt_pct(s.max_drawdown, signed=True) if s.max_drawdown != 0 else "    -"
            vol = _fmt_pct(s.annual_vol)
            span = f"{row.first or '-'} ~ {row.last or '-'}"
        else:
            ret = dd = vol = "-"
            span = (f"{row.first or '-'} ~ {row.last or '-'}"
                    if (row.first or row.last) else "-")
        cells = [
            row.code,
            _name_for(row.code),
            str(row.rows),
            span,
            ret,
            dd,
            vol,
        ]
        print("  ".join(_pad(c, w) for c, w in zip(cells, widths)))


# ---------------------------------------------------------------------------
# matplotlib 出图（--qc-plot）
# ---------------------------------------------------------------------------
# 中文字体回退顺序：PingFang SC → Hiragino Sans GB → Heiti TC → DejaVu Sans
# 最后一个兜底保证一定有可用字体；CJK 字符画不出来时退化为方框，但不会崩。
_CN_FONT_CANDIDATES = ["PingFang SC", "Hiragino Sans GB", "Heiti TC"]
# macOS 系统里 PingFang.ttc 的常见路径；matplotlib fontManager 默认不索引它，
# 需要显式 addfont 一次才能按 "PingFang SC" 这种 family name 解析。
_PINGFANG_TTC_CANDIDATES = [
    "/System/Library/AssetsV2/com_apple_MobileAsset_Font8/"
    "86ba2c91f017a3749571a82f2c6d890ac7ffb2fb.asset/AssetData/PingFang.ttc",
    "/System/Library/PrivateFrameworks/FontServices.framework/Resources/"
    "Reserved/PingFangUI.ttc",
    "/Library/Fonts/PingFang.ttc",
]


def _setup_chinese_font():
    """一次性设好 matplotlib 中文字体；使用 Agg backend（headless / cron 安全）。

    探测流程：
    1. 显式 addfont 系统里的 PingFang.ttc（若存在），让 "PingFang SC" 可解析
    2. 按 _CN_FONT_CANDIDATES 顺序 findfont，第一个能解析的胜出
    3. 全失败时退到 DejaVu Sans，并把缺失字体的警告打到 stderr
    """
    import matplotlib

    matplotlib.use("Agg")  # 非交互 backend；先于 pyplot import 设，避免污染默认
    import matplotlib.font_manager as fm
    import matplotlib.pyplot as plt

    # 先尝试把系统里的 PingFang.ttc 注册进 fontManager（一次性；后续 plt
    # 共享这个 manager，无需重复 addfont）。
    for path in _PINGFANG_TTC_CANDIDATES:
        if Path(path).exists():
            try:
                fm.fontManager.addfont(path)
            except Exception:  # noqa: BLE001
                pass

    # 探测：哪个候选名真的能被解析
    chosen: str | None = None
    for name in _CN_FONT_CANDIDATES:
        try:
            fm.findfont(name, fallback_to_default=False)
            chosen = name
            break
        except Exception:  # noqa: BLE001
            continue

    fallback_chain = _CN_FONT_CANDIDATES + ["DejaVu Sans"]
    if chosen is not None:
        sans_list = [chosen] + [n for n in fallback_chain if n != chosen]
    else:
        sans_list = fallback_chain

    plt.rcParams["font.sans-serif"] = sans_list
    plt.rcParams["axes.unicode_minus"] = False
    if chosen is None:
        print(
            "[WARN] 中文字体回退链全部未命中；CJK 字符将显示为方框。"
            "可在 _CN_FONT_CANDIDATES / _PINGFANG_TTC_CANDIDATES 里追加系统字体路径。",
            file=sys.stderr,
        )
    return plt


def _plot_one_etf_chart(
    df: pd.DataFrame,
    code: str,
    save_dir: Path,
    stats: DescriptiveStats | None = None,
) -> Path | None:
    """为单只 ETF 画 close 曲线 + 标注区间收益 / 最大回撤，存 PNG。

    返回保存路径；df 空 / 缺 close 时返回 None 不画。
    """
    if df.empty or "close" not in df.columns or "date" not in df.columns:
        return None

    plt = _setup_chinese_font()

    dates = pd.to_datetime(df["date"])
    closes = df["close"].astype(float).to_numpy()
    p0, p1 = float(closes[0]), float(closes[-1])
    period_ret = (p1 / p0 - 1) if p0 > 0 else 0.0

    fig, ax = plt.subplots(figsize=(11, 5.2))

    ax.plot(dates, closes, linewidth=1.3, color="#1f77b4", label="close（前复权）")
    ax.fill_between(dates, closes, alpha=0.10, color="#1f77b4")

    ax.scatter([dates.iloc[0]], [p0], color="#2ca02c", s=55, zorder=5,
               edgecolors="white", linewidths=1.2)
    ax.scatter([dates.iloc[-1]], [p1], color="#d62728", s=55, zorder=5,
               edgecolors="white", linewidths=1.2)
    ax.annotate(
        f"{dates.iloc[0].date()}\n{p0:.3f}",
        xy=(dates.iloc[0], p0), xytext=(8, -28),
        textcoords="offset points", fontsize=8, color="#2ca02c",
    )
    ax.annotate(
        f"{dates.iloc[-1].date()}\n{p1:.3f}",
        xy=(dates.iloc[-1], p1), xytext=(-90, 8),
        textcoords="offset points", fontsize=8, color="#d62728",
    )

    name = _name_for(code)
    period_str = f"{dates.iloc[0].date()} ~ {dates.iloc[-1].date()}"
    title_extra = ""
    if stats is not None and stats.has_data:
        title_extra = (
            f"  |  收益 {stats.annual_ret * 100:+.2f}%/年  "
            f"波动 {stats.annual_vol * 100:.2f}%  "
            f"回撤 {_fmt_pct(stats.max_drawdown, signed=True)}"
        )
    ax.set_title(f"{code}  {name}\n区间 {period_str}（{len(df)} bar）"
                 f"  |  区间收益 {period_ret * 100:+.2f}%{title_extra}",
                 fontsize=11)
    ax.set_xlabel("日期")
    ax.set_ylabel("收盘价（前复权 qfq）")
    ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.6)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.85)

    fig.autofmt_xdate()
    fig.tight_layout()

    save_dir.mkdir(parents=True, exist_ok=True)
    out_path = save_dir / f"{code}.png"
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def _plot_corr_heatmap(
    matrices: dict[str, pd.DataFrame],
    save_dir: Path,
) -> Path | None:
    """把 Pearson 相关系数矩阵画成热力图 PNG，文件名 corr_pearson.png。"""
    if "pearson" not in matrices or matrices["pearson"].empty:
        return None

    plt = _setup_chinese_font()

    corr = matrices["pearson"].to_numpy()
    codes = matrices["pearson"].columns.tolist()

    fig, ax = plt.subplots(figsize=(7.5, 6))
    im = ax.imshow(corr, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Pearson r")

    ax.set_xticks(range(len(codes)))
    ax.set_yticks(range(len(codes)))
    ax.set_xticklabels(codes, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(codes, fontsize=9)

    for i in range(len(codes)):
        for j in range(len(codes)):
            v = corr[i, j]
            color = "white" if abs(v) > 0.6 else "black"
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    color=color, fontsize=8)

    ax.set_title("跨 ETF Pearson 相关系数矩阵", fontsize=12)
    fig.tight_layout()

    save_dir.mkdir(parents=True, exist_ok=True)
    out_path = save_dir / "corr_pearson.png"
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def _plot_combined_performance(
    closes: pd.DataFrame,
    save_dir: Path,
) -> Path | None:
    """把所有 ETF 的 close 归一化到起点=100，画在同一张图上（性能对比图）。

    用途：把不同价格区间、不同上市时间的几只宽基 ETF 拉到同一个比较基准下，
    一图看清谁跑得更好。closes 来自 load_close_series（已经按 inner join 对齐
    + 应用了 qc-start-date/qc-end-date），所以画出来的样本就是用户指定的窗口。

    返回保存路径；列数 < 2 或全空时返回 None 不画。
    """
    if closes.empty or len(closes.columns) < 2:
        return None

    plt = _setup_chinese_font()

    # 归一化：每只 ETF 的 close / 自己首日 close * 100
    normed = closes / closes.iloc[0] * 100.0

    fig, ax = plt.subplots(figsize=(11, 5.8))

    # 调色板：tab10 前 N 色，区分度高
    cmap = plt.get_cmap("tab10")
    for i, code in enumerate(closes.columns):
        name = _name_for(code)
        ax.plot(
            closes.index, normed[code],
            linewidth=1.4, color=cmap(i), label=f"{code} {name}",
        )

    # 在每个序列终点标累计收益%
    for i, code in enumerate(closes.columns):
        final = float(normed[code].iloc[-1])
        ret_pct = final - 100.0
        ax.scatter([closes.index[-1]], [final], color=cmap(i), s=42,
                   zorder=5, edgecolors="white", linewidths=1.0)
        ax.annotate(
            f"{ret_pct:+.1f}%",
            xy=(closes.index[-1], final),
            xytext=(8, 0),
            textcoords="offset points",
            fontsize=8, color=cmap(i), va="center",
        )

    # 参考线：起点 = 100
    ax.axhline(100, color="grey", linestyle=":", linewidth=0.7, alpha=0.7)

    first_date = closes.index[0]
    last_date = closes.index[-1]
    first_str = first_date.date() if hasattr(first_date, "date") else str(first_date)
    last_str = last_date.date() if hasattr(last_date, "date") else str(last_date)
    ax.set_title(
        f"宽基 ETF 归一化性能对比（起点 = 100）  |  "
        f"区间 {first_str} ~ {last_str}（{len(closes)} bar，对齐 {len(closes.columns)} 只 ETF）",
        fontsize=11,
    )
    ax.set_xlabel("日期")
    ax.set_ylabel("归一化指数（首日 = 100）")
    ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.6)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.85)

    fig.autofmt_xdate()
    fig.tight_layout()

    save_dir.mkdir(parents=True, exist_ok=True)
    out_path = save_dir / "combined.png"
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def print_quality_report(
    data_dir: Path,
    codes: list[str],
    start_date: date | None = None,
    end_date: date | None = None,
    plot_dir: Path | None = None,
) -> None:
    """质量检查总入口：概述表 + 单只描述统计 + 跨 ETF 相关性矩阵。

    start_date / end_date：把每只 ETF 各自截到区间内再做检查；用于"只想看
    最近 1 年波动率"或"对比 2018 熊市 vs 2020 牛市"等子区间分析。

    plot_dir：非 None 时，给每只 ETF 画一张 close 曲线 PNG，外加一张
    Pearson 相关性热力图；空 / 缺数据则跳过对应那一张。
    """
    range_note = ""
    if start_date is not None or end_date is not None:
        range_note = f" | 区间 {start_date or '-'} ~ {end_date or '-'}"
    print(f"== 质量检查：扫描 {data_dir}{range_note} ==")
    rows = [quality_check(csv_path(data_dir, c), start_date, end_date) for c in codes]
    all_issues = 0

    print("\n--- 概述 ---")
    _print_overview_table(rows)

    print("\n--- 单只 ETF 检查 + 描述统计 ---")
    for row in rows:
        _print_one_etf(row)
        if row.issues:
            all_issues += 1

    # 跨 ETF 相关性（同样按区间过滤，保证对齐样本来自指定窗口）
    closes = load_close_series(data_dir, codes, start_date, end_date)
    matrices: dict[str, pd.DataFrame] = {}
    corr_skipped_reason: str | None = None
    if closes.empty or len(closes.columns) < 2:
        corr_skipped_reason = "数据不足，跳过相关性"
    else:
        matrices = correlation_matrices(closes)
        if not matrices:
            corr_skipped_reason = "收益序列为空，跳过相关性"
        else:
            n_obs = len(matrices["log_returns"])
            if n_obs < MIN_OBS_FOR_CORR:
                corr_skipped_reason = (
                    f"对齐后只有 {n_obs} 个交易日，少于阈值 {MIN_OBS_FOR_CORR}；"
                    "Pearson/Spearman 几乎恒为 ±1，跳过输出"
                )
            else:
                print(f"\n--- 跨 ETF 相关性（日对数收益，n={n_obs} 个交易日对齐）---")
                _print_corr_matrix("Pearson 相关系数", matrices["pearson"])
                _print_corr_matrix("Spearman 秩相关", matrices["spearman"])

    if corr_skipped_reason:
        print(f"\n[相关性] {corr_skipped_reason}。")

    print(f"\n共 {len(codes)} 只，{all_issues} 只有问题。")

    # ---- 出图（matplotlib）----
    if plot_dir is None:
        return
    plot_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n== 出图：写入 {plot_dir}（matplotlib Agg）==")
    saved: list[Path] = []
    for code, qrow in zip(codes, rows):
        df = pd.read_csv(csv_path(data_dir, code))
        if start_date is not None or end_date is not None:
            d = pd.to_datetime(df["date"]).dt.date
            mask = pd.Series(True, index=df.index)
            if start_date is not None:
                mask &= d >= start_date
            if end_date is not None:
                mask &= d <= end_date
            df = df[mask]
        path = _plot_one_etf_chart(df, code, plot_dir, stats=qrow.stats)
        if path is not None:
            saved.append(path)
            print(f"  - {code:<14} -> {path.name}")
        else:
            print(f"  - {code:<14} (空 / 缺数据，跳过)")
    if matrices and "pearson" in matrices and not matrices["pearson"].empty:
        heat = _plot_corr_heatmap(matrices, plot_dir)
        if heat is not None:
            saved.append(heat)
            print(f"  - corr heatmap  -> {heat.name}")
    if not closes.empty and len(closes.columns) >= 2:
        combo = _plot_combined_performance(closes, plot_dir)
        if combo is not None:
            saved.append(combo)
            print(f"  - combined perf -> {combo.name}")
    if saved:
        print(f"\n共生成 {len(saved)} 张 PNG：")
        for p in saved:
            print(f"  {p}")
    else:
        print("\n未生成任何 PNG。")


# ---------------------------------------------------------------------------
# 往前扩展模式（--lookback-years）
# ---------------------------------------------------------------------------
def _run_lookback(
    client: TdxClient,
    data_dir: Path,
    codes: list[str],
    years: int,
    end_d: date,
) -> None:
    """往前扩展到 N 年：只下载缺失的早期数据，prepend 到现有 CSV。

    每只 ETF 独立判断：现有 first_date 已经早于等于 target_first 时 skip。
    否则拉 [target_first, existing_first - 1] 的数据，prepend。
    """
    target_first = end_d - timedelta(days=365 * years)
    print(f"== 扩展到最近 {years} 年（目标最早 {target_first}） ==")

    for combined in codes:
        path = csv_path(data_dir, combined)
        existing_first = None
        if path.exists():
            df = pd.read_csv(path)
            if not df.empty:
                existing_first = pd.to_datetime(df["date"]).min().date()

        if existing_first and existing_first <= target_first:
            print(f"[skip]   {_label(combined)} 已覆盖到 {existing_first}，无需扩展")
            continue
        if existing_first and existing_first > target_first:
            # 上市日已经晚于目标起点：现有数据就是该 ETF 能拿到的最早，
            # 再往前 eltdx 也没有 bar；不必再尝试 fetch。
            print(f"[skip]   {_label(combined)} 上市于 {existing_first}，已是最早，无需扩展")
            continue

        fetch_s = target_first
        fetch_e = (existing_first - timedelta(days=1)) if existing_first else end_d
        print(f"[prepend] {_label(combined)} 拉 {fetch_s} -> {fetch_e} ...", end=" ", flush=True)

        df_new = fetch_one(client, combined, fetch_s, fetch_e)
        if df_new.empty:
            print("无数据")
            continue

        added, dup = prepend_csv(df_new, path)
        kept = len(pd.read_csv(path))
        print(f"+{added} 行（去重 {dup}），文件 {kept} 行")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="批量拉宽基 ETF 日 K 线（前复权 qfq）")
    p.add_argument("--start-date", help="起始日期 YYYY-MM-DD，默认：今天 - 3 年")
    p.add_argument("--end-date", help="结束日期 YYYY-MM-DD，默认：今天 - 1 天")
    p.add_argument("--codes", help="逗号分隔的 ETF 列表，默认全部")
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="CSV 落盘目录")
    p.add_argument("--incremental", action="store_true",
                   help="增量模式：每个文件从已有最新日期 +1 补到昨天")
    p.add_argument("--lookback-years", type=int, default=None,
                   help="往前扩展到 N 年：只下载缺失的早期数据，prepend 到现有 CSV；现有数据不动")
    p.add_argument("--quality-check", action="store_true",
                   help="只做质量检查，不下载")
    p.add_argument("--qc-start-date", default=None,
                   help="质量检查的起始日期 YYYY-MM-DD（含），仅与 --quality-check 一起生效")
    p.add_argument("--qc-end-date", default=None,
                   help="质量检查的结束日期 YYYY-MM-DD（含），仅与 --quality-check 一起生效")
    p.add_argument("--qc-plot", action="store_true",
                   help="质量检查时为每只 ETF 生成 close 曲线 PNG + 相关性热力图，matplotlib Agg backend")
    p.add_argument("--qc-plot-dir", type=Path, default=DEFAULT_PLOT_DIR,
                   help="qc-plot 的 PNG 输出目录，默认 <项目根>/.cache/qc_plots")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    # 日期默认值
    today = date.today()
    if args.end_date:
        end_d = datetime.strptime(args.end_date, "%Y-%m-%d").date()
    else:
        # 默认截止到昨天：eltdx 在盘中/盘前对"今日"只返回一行占位
        # （OHLC=前收, volume=0, amount=0），避开它才能保证落盘的都是真实数据。
        # 如果用户显式给了 --end-date，尊重用户。
        end_d = today - timedelta(days=1)
    start_d = (
        datetime.strptime(args.start_date, "%Y-%m-%d").date()
        if args.start_date
        else end_d - timedelta(days=365 * 3)
    )
    if start_d > end_d:
        print(f"ERROR: start_date({start_d}) > end_date({end_d})", file=sys.stderr)
        return 2

    etfs = (
        [c.strip() for c in args.codes.split(",") if c.strip()]
        if args.codes
        else list(DEFAULT_ETFS)
    )

    args.data_dir.mkdir(parents=True, exist_ok=True)

    # ----- 质量检查模式 -----
    if args.quality_check:
        qc_s = (
            datetime.strptime(args.qc_start_date, "%Y-%m-%d").date()
            if args.qc_start_date else None
        )
        qc_e = (
            datetime.strptime(args.qc_end_date, "%Y-%m-%d").date()
            if args.qc_end_date else None
        )
        if qc_s is not None and qc_e is not None and qc_s > qc_e:
            print(f"ERROR: --qc-start-date({qc_s}) > --qc-end-date({qc_e})", file=sys.stderr)
            return 2
        plot_dir = args.qc_plot_dir if args.qc_plot else None
        print_quality_report(args.data_dir, etfs, qc_s, qc_e, plot_dir)
        return 0

    # ----- lookback 扩展模式 -----
    if args.lookback_years is not None:
        with TdxClient(timeout=8) as client:
            _run_lookback(client, args.data_dir, etfs, args.lookback_years, end_d)
        return 0

    # ----- 下载模式 -----
    end_note = "" if args.end_date else "（默认截止昨天，规避 eltdx 当日占位）"
    print(f"== 拉取窗口 {start_d} -> {end_d}{end_note} | 模式："
          f"{'incremental' if args.incremental else 'full'} ==")

    results: list[FetchResult] = []

    with TdxClient(timeout=8) as client:
        for combined in etfs:
            res = FetchResult(code=combined)
            path = csv_path(args.data_dir, combined)
            try:
                if args.incremental:
                    rng = resolve_incremental_range(path)
                    if rng is None:
                        s, e = start_d, end_d
                        mode = "full"
                    else:
                        s, e = rng
                        mode = "inc"
                        if s > e:
                            print(f"[skip] {_label(combined)} 已是最新（last={s - timedelta(days=1)}）")
                            results.append(res)
                            continue
                else:
                    s, e = start_d, end_d
                    mode = "full"

                print(f"[{mode}] {_label(combined)}  {s} -> {e} ...", end=" ", flush=True)
                df_new = fetch_one(client, combined, s, e)

                if df_new.empty:
                    print("无数据")
                    if not args.incremental and path.exists():
                        df_existing = pd.read_csv(path)
                        res.kept = len(df_existing)
                    results.append(res)
                    continue

                added, dup = save_csv(df_new, path, incremental=args.incremental)
                res.fetched = added
                res.skipped_dup = dup
                res.kept = len(pd.read_csv(path))
                print(f"+{res.fetched} 行（去重 {res.skipped_dup}），文件 {res.kept} 行")
            except Exception as ex:  # noqa: BLE001
                res.error = f"{type(ex).__name__}: {ex}"
                print(f"FAIL: {res.error}")
            results.append(res)

    print("\n== 汇总 ==")
    ok = sum(1 for r in results if r.error is None)
    fail = len(results) - ok
    total_new = sum(r.fetched for r in results)
    print(f"成功 {ok}/{len(results)}，失败 {fail}；本次新增 bar 总数 {total_new}")
    if fail:
        for r in results:
            if r.error:
                print(f"  - {_label(r.code)}: {r.error}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
