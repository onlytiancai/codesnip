"""
下载 SPX（标普500）指数最近 10 年的日线数据到 CSV。
数据源：Yahoo Finance（ticker: ^GSPC）
需要代理：在 shell 中导出 http(s)_proxy 后运行
"""

import os
import time
from datetime import date, timedelta

import pandas as pd
import yfinance as yf

OUT_CSV = __file__.replace(".py", "_spx_10y.csv")
TICKER = "^GSPC"

END = date.today()
START = END - timedelta(days=10 * 365)  # 近 10 年

print(f"下载 {TICKER} 日线数据：{START} -> {END}")
print(f"代理: http={os.environ.get('http_proxy')}  https={os.environ.get('https_proxy')}")


def fetch_with_retry(ticker: str, start: str, end: str, max_retries: int = 5) -> pd.DataFrame:
    delay = 5
    last_err: Exception | None = None
    for i in range(1, max_retries + 1):
        try:
            df = yf.download(
                ticker,
                start=start,
                end=end,
                interval="1d",
                auto_adjust=False,
                progress=False,
                threads=False,  # 避免并发触发限流
            )
            if df is not None and not df.empty:
                return df
            last_err = RuntimeError("返回空数据")
        except Exception as e:  # noqa: BLE001
            last_err = e
        print(f"  第 {i}/{max_retries} 次失败：{last_err}，{delay}s 后重试...")
        time.sleep(delay)
        delay = min(delay * 2, 60)
    raise RuntimeError(f"重试 {max_retries} 次后仍失败：{last_err}")


df = fetch_with_retry(TICKER, START.isoformat(), END.isoformat())

# yfinance 1.x 单 ticker 也可能返回 MultiIndex，扁平化
if isinstance(df.columns, pd.MultiIndex):
    df.columns = [c[0] for c in df.columns]

df.index.name = "Date"
df = df.reset_index()

df = df[["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]]
df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")

df.to_csv(OUT_CSV, index=False)

print(f"\n已保存 {len(df)} 行 -> {OUT_CSV}")
print(df.head().to_string(index=False))
print("...")
print(df.tail().to_string(index=False))
