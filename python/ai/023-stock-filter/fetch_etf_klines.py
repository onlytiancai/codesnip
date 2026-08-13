"""批量获取宽基 ETF 日 K 线（前复权），落到 .cache/klines/ 下 CSV。

用法示例
--------
# 默认：最近 3 年，全部 ETF
python fetch_etf_klines.py

# 指定时间窗
python fetch_etf_klines.py --start-date 2024-01-01 --end-date 2025-12-31

# 增量：从每个 ETF 已落盘文件的最新一日往后补到今天
python fetch_etf_klines.py --incremental

# 质量检查：只扫描 .cache/klines/*.csv，不下载
python fetch_etf_klines.py --quality-check

# 只取两只 ETF
python fetch_etf_klines.py --codes 510050.SH,159915.SZ
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

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
]

# 每次拉一页最多 800 根 K 线，eltdx 的硬上限
PAGE_SIZE = 800

# 默认数据目录
DEFAULT_DATA_DIR = Path(__file__).parent / ".cache" / "klines"

# 中国时区（eltdx 返回的 datetime 用的就是这个 tz）
CST = timezone(timedelta(hours=8))


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
    # bar.time 是带 tz 的 datetime；CSV 落日期字符串
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
    """
    eltdx_code = to_eltdx_code(combined)
    rows: list[dict] = []

    start = 0
    while True:
        series = client.get_kline(
            "day",
            eltdx_code,
            start=start,
            count=PAGE_SIZE,
            adjust="qfq",
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
        if start > 5000:
            break

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
    """落盘；返回 (新增条数, 去重丢弃数)。"""
    if not incremental or not path.exists():
        df_new.to_csv(path, index=False)
        return len(df_new), 0

    # 增量：读旧，合并，按 date 去重（旧优先），重写
    df_old = pd.read_csv(path)
    kept_old = len(df_old)

    # 旧文件里最大日期作为哨兵，避免无限增长
    last_old_date = pd.to_datetime(df_old["date"]).max().date() if kept_old else None

    if last_old_date is not None:
        df_new = df_new[pd.to_datetime(df_new["date"]).dt.date > last_old_date]

    merged = pd.concat([df_old, df_new], ignore_index=True)
    merged = merged.drop_duplicates(subset=["date"], keep="last").sort_values("date").reset_index(drop=True)
    merged.to_csv(path, index=False)

    added = len(df_new)
    dup_dropped = (kept_old + len(df_new)) - len(merged)
    return added, dup_dropped


# ---------------------------------------------------------------------------
# 增量模式：决定每只 ETF 的实际起止
# ---------------------------------------------------------------------------
def resolve_incremental_range(path: Path) -> tuple[date, date] | None:
    """读已有 CSV 的最大日期，返回 (start, today)；不存在返回 None。"""
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if df.empty:
        return None
    last = pd.to_datetime(df["date"]).max().date()
    # 从 max+1 开始；eltdx 会按"最新优先"返回，所以 +1 足够
    return last + timedelta(days=1), date.today()


# ---------------------------------------------------------------------------
# 质量检查
# ---------------------------------------------------------------------------
@dataclass
class QCRow:
    code: str
    rows: int
    first: str | None
    last: str | None
    issues: list[str]


def quality_check(path: Path) -> QCRow:
    issues: list[str] = []
    code = path.stem

    if not path.exists():
        return QCRow(code=code, rows=0, first=None, last=None, issues=["文件不存在"])

    df = pd.read_csv(path)
    if df.empty:
        return QCRow(code=code, rows=0, first=None, last=None, issues=["文件为空"])

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

    # 6. 日期连续性（仅在节假日会有连续空缺；> 10 个交易日断档告警）
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

    return QCRow(code=code, rows=n, first=first, last=last, issues=issues)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="批量拉宽基 ETF 日 K 线（前复权 qfq）")
    p.add_argument("--start-date", help="起始日期 YYYY-MM-DD，默认：今天 - 3 年")
    p.add_argument("--end-date", help="结束日期 YYYY-MM-DD，默认：今天")
    p.add_argument("--codes", help="逗号分隔的 ETF 列表，默认全部")
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="CSV 落盘目录")
    p.add_argument("--incremental", action="store_true",
                   help="增量模式：每个文件从已有最新日期 +1 补到今天")
    p.add_argument("--quality-check", action="store_true",
                   help="只做质量检查，不下载")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    # 日期默认值
    today = date.today()
    end_d = datetime.strptime(args.end_date, "%Y-%m-%d").date() if args.end_date else today
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
        print(f"== 质量检查：扫描 {args.data_dir} ==")
        all_issues = 0
        for combined in etfs:
            row = quality_check(csv_path(args.data_dir, combined))
            tag = "OK " if not row.issues else "WARN"
            print(f"[{tag}] {row.code:<10} rows={row.rows:<6} "
                  f"{row.first or '-'} -> {row.last or '-'} "
                  f"{'; '.join(row.issues) if row.issues else ''}")
            if row.issues:
                all_issues += 1
        print(f"\n共 {len(etfs)} 只，{all_issues} 只有问题。")
        return 0

    # ----- 下载模式 -----
    print(f"== 拉取窗口 {start_d} -> {end_d}（含今天） | 模式："
          f"{'incremental' if args.incremental else 'full'} ==")

    results: list[FetchResult] = []

    with TdxClient(timeout=8) as client:
        for combined in etfs:
            res = FetchResult(code=combined)
            path = csv_path(args.data_dir, combined)
            try:
                # 决定本只的实际区间
                if args.incremental:
                    rng = resolve_incremental_range(path)
                    if rng is None:
                        s, e = start_d, end_d
                        mode = "full"
                    else:
                        s, e = rng
                        mode = "inc"
                        if s > e:
                            print(f"[skip] {combined} 已是最新（last={s - timedelta(days=1)}）")
                            results.append(res)
                            continue
                else:
                    s, e = start_d, end_d
                    mode = "full"

                print(f"[{mode}] {combined}  {s} -> {e} ...", end=" ", flush=True)
                df_new = fetch_one(client, combined, s, e)

                if df_new.empty:
                    print("无数据")
                    if not args.incremental and path.exists():
                        # 文件不动
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

    # 汇总
    print("\n== 汇总 ==")
    ok = sum(1 for r in results if r.error is None)
    fail = len(results) - ok
    total_new = sum(r.fetched for r in results)
    print(f"成功 {ok}/{len(results)}，失败 {fail}；本次新增 bar 总数 {total_new}")
    if fail:
        for r in results:
            if r.error:
                print(f"  - {r.code}: {r.error}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())