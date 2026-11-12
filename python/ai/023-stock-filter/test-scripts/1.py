from eltdx import TdxClient

with TdxClient(timeout=3) as client:
    quote = client.get_quote(["sz000001", "sh600000"])
    bars = client.get_kline("day", "sz000001", count=30)
    minute = client.get_minute("sz000001")
    ticks = client.get_history_trade_day("sz000001", "2026-05-20")

print(quote[0])
print(bars.bars[-1])
