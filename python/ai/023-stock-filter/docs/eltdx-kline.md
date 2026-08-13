# TdxClient 获取 K 线数据

eltdx 是通达信在线行情协议的 Python 封装，核心入口是 `TdxClient`。本文档讲清楚**只用库 API**怎么把 K 线拉下来——不依赖脚本、不依赖代理，方便排查问题或临时取数。

## 一、核心概念

`TdxClient` 是一个**带 socket 池的同步客户端**：

- 默认走 7709 端口，连通达信公网主站（`116.205.183.150:7709` 等一组 IP）
- 内部维护 TCP 长连接、心跳和请求池；调用 `get_kline` 是一次 RPC，不是重新建连
- 推荐用 `with` 上下文包起来，离开时自动关连接

```python
from eltdx import TdxClient

with TdxClient(timeout=8) as client:
    series = client.get_kline("day", "sh510050", count=30, adjust="qfq")
    for bar in series.bars:
        print(bar.time.date(), bar.open, bar.high, bar.low, bar.close, bar.volume_lots)
```

## 二、`get_kline` 签名

```
get_kline(
    arg1,                     # period: "day" | "week" | "month" | "m5" | "m15" | "m30" | "m60" ...
    arg2=None,                # code: "sh510050" / "sz159915"
    *,
    start: int = 0,          # 跳过最新的 N 根
    count: int = 800,        # 本次最多取多少根（eltdx 硬上限 800）
    kind: str = "stock",      # 品种类型，可省略
    adjust: str | None = None,   # "qfq" 前复权 / "hfq" 后复权 / None 不复权
    anchor_date=None,        # 锚定日期（不复权时使用）
    include_raw: bool = False,
)
```

**关键参数说明**：

| 参数 | 含义 | 常用值 |
|---|---|---|
| `period` | K 线周期 | `"day"`（日）、`"week"`、`"month"`、`"m5"`（5 分钟）… |
| `code` | 6 位市场代码，**小写市场前缀** | `sh510050`、`sz159915`、`bj83xxxx` |
| `start` | "跳过最新的 N 根" —— 翻页用 | `0` 取最新一页 |
| `count` | 本页根数 | 最多 800 |
| `adjust` | 复权方式 | `"qfq"` 前复权（推荐）/ `"hfq"` 后复权 / `None` 不复权 |

> **代码格式注意**：库用的是 `sh510050` / `sz159915` 这种小写前缀，不是 `510050.SH`。交易所后缀要先做转换。

## 三、返回结构 `KlineSeries`

```python
series = client.get_kline("day", "sh510050", count=5, adjust="qfq")
series.code           # 'sh510050'
series.period_name    # 'day'
series.adjust_mode    # 'qfq'
series.count          # 5
series.bars           # list[KlineBar] —— 重点关注
series.raw_payload    # 原始协议字节（仅 include_raw=True 时有内容）
```

**`KlineBar` 字段**：

| 字段 | 类型 | 含义 |
|---|---|---|
| `time` | `datetime`（带 tz=`Asia/Shanghai`） | 收盘时间 |
| `open` / `high` / `low` / `close` | `float`（元） | OHLC 价 |
| `open_price_milli` 等 4 个 | `int`（毫精度） | 整数化的 OHLC，免浮点误差 |
| `volume_lots` | `float`（手） | 成交量（1 手 = 100 份） |
| `volume_raw` | `int` | 原始字段，含义因品种略变 |
| `amount` | `float`（元） | 成交额 |
| `last_close_price_milli` | `int` | 昨收（毫精度），首日可能为 None |
| `up_count` / `down_count` | `int` | 连阳/连阴天数 |

**翻页顺序**：eltdx 的 `start` 语义是"跳过最新的 N 根"：

- `start=0, count=800` → 最新 800 根
- `start=800, count=800` → 第 801 ~ 1600 根（更老）
- `series.bars` 内部按**时间升序**排列（最旧 → 最新），不是从新到旧

> ⚠️ **这是脚本开发中最容易踩的坑**。如果按"从新到旧"假设去翻页或过滤，第一根就是最旧的，导致过滤逻辑直接短路。

## 四、翻页范式（取多年数据）

eltdx 单次最多 800 根；要取 3 年（约 730 个交易日）一页就够，但更早的数据需要翻页：

```python
from datetime import date

with TdxClient(timeout=8) as client:
    rows = []
    start = 0
    PAGE = 800
    while True:
        s = client.get_kline("day", "sh510050", start=start, count=PAGE, adjust="qfq")
        if not s.bars:
            break
        rows.extend(s.bars)
        start += len(s.bars)   # 关键：start += count，不是 +1
        if len(s.bars) < PAGE:
            break              # 不到一页说明已到头
```

**早停优化**：如果 `start=800` 这一页的所有 bar 都早于目标 start_date，可以直接 break 不用再翻。

## 五、按日期范围过滤

```python
def bars_in_range(bars, start_d: date, end_d: date):
    """eltdx 返回的 bars 是时间升序，直接按日期切即可。"""
    out = []
    for b in bars:
        d = b.time.date()
        if d < start_d:
            continue       # 太早
        if d > end_d:
            continue       # 太晚（理论上不会出现，start=0 时最新页不会超过今天）
        out.append(b)
    return out
```

不要用 `time` 字段的 `<` 比较做"短路 return"——因为 bars 是升序，遇到更早的应该跳过（continue）而不是返回，因为后面还有更新的。

## 六、复权方式对比

| 模式 | 最近价 | 历史价 | 适用场景 |
|---|---|---|---|
| `None`（不复权） | 等于原始收盘 | 等于原始收盘 | 分红再投、回测真实账户 |
| `"qfq"`（前复权） | 等于原始收盘 | 按除权除息反向调整 | **回测、可视化（推荐）** |
| `"hfq"`（后复权） | 按除权除息正向调整 | 等于原始首日 | 看长期趋势图 |

> ETF 前复权遇分红/份额折算时会出现整段比例重映射的视觉跳变（例如 1:5 折算看起来"涨"400%），这是正常的复权计算结果，不是数据问题。

## 七、其它相关方法

```python
# 1) 行情快照（最新价、买卖五档等）
quote = client.get_quote(["sh510050", "sz159915"])
for q in quote:
    print(q.code, q.price, q.open, q.high, q.low, q.volume)

# 2) 今日分时（5 分钟 K 线粒度）
minute = client.get_minute("sh510050")
for m in minute.bars:
    print(m.time, m.price, m.volume_lots)

# 3) 某日成交明细（逐笔，最多 2000 条）
ticks = client.get_history_trade_day("sh510050", "2026-05-20")
for t in ticks:
    print(t.time, t.price, t.volume, t.bs_flag)
```

## 八、常见错误

| 现象 | 原因 |
|---|---|
| `ConnectionClosedError: unable to connect to any 7709 host` | 网络到不了通达信主站（被墙/防火墙/路由问题）。库本身无代理参数 |
| `ConnectionClosedError: 7709 socket closed by remote peer` | TDX 服务端拒绝了连接，可能短时间内请求过多被限速 |
| `series.bars` 看起来顺序"倒过来" | 错觉，实际是升序，最旧的 `bars[0]`，最新的 `bars[-1]` |
| 取到 `start=0` 整页都被早于目标日期过滤掉 | 把"升序"当成"降序"了 — 见上条 |

## 九、参考

- 项目脚本：`test-scripts/cache_codes.py`、`fetch_etf_klines.py`
- eltdx 源码：`~/.pyenv/versions/qlib/lib/python3.11/site-packages/eltdx/`
- 通达信协议：<https://github.com/electkismet/eltdx>