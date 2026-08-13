# fetch_etf_klines.py 使用文档

批量获取宽基 ETF 日 K 线（前复权）的脚本，覆盖六只核心宽基指数对应的 ETF：上证50、沪深300、中证500、中证1000、创业板指、科创50。

## 一、用途

回测/分析时一次性把多只宽基 ETF 的历史 K 线拉到本地 CSV 里。三种典型场景：

1. **首次建库** —— 拉最近 3 年数据落到 `.cache/klines/`，每只 ETF 一份 CSV
2. **每日增量** —— 自动识别每只 ETF 已落盘文件的最新一日，往后补到今天
3. **质量复核** —— 对已落盘文件做 OHLC 关系、断档、复权跳变等检查，不下载

## 二、依赖与运行环境

| 依赖 | 说明 |
|---|---|
| Python | `~/.pyenv/versions/qlib/bin/python`（项目默认） |
| `eltdx` | `pip install eltdx`，提供 `TdxClient` |
| `pandas` | 已在 qlib 环境内 |
| 网络 | 需能直连通达信主站（7709/TCP） |

> 本脚本**不带代理**。如果你在墙内，请自行配置全局代理或改用其它数据源。

## 三、目录与文件

| 路径 | 内容 |
|---|---|
| `fetch_etf_klines.py` | 脚本本体（项目根目录） |
| `.cache/klines/<代码>.csv` | 每只 ETF 一份 CSV，例如 `510050.SH.csv` |
| `.cache/` | 已被 `.gitignore` 忽略 |

CSV 列：`date, open, high, low, close, volume, amount`

| 列 | 类型 | 含义 |
|---|---|---|
| `date` | `YYYY-MM-DD` | 交易日（按 CST 时区） |
| `open` / `high` / `low` / `close` | `float` | OHLC 价（前复权 qfq） |
| `volume` | `int` | 成交量，**单位为"手"**（1 手 = 100 份） |
| `amount` | `float` | 成交额（元） |

## 四、ETF 默认列表

| 代码 | 名称 | 跟踪指数 |
|---|---|---|
| `510050.SH` | 华夏上证50ETF | 上证50 |
| `510300.SH` | 华泰柏瑞沪深300ETF | 沪深300 |
| `510500.SH` | 南方中证500ETF | 中证500 |
| `159845.SZ` | 富国中证1000ETF | 中证1000 |
| `159915.SZ` | 易方达创业板ETF | 创业板指 |
| `588000.SH` | 华夏科创50ETF | 科创50 |

列表在脚本顶部 `DEFAULT_ETFS`，可改。

## 五、命令行用法

### 5.1 首次拉取（默认最近 3 年）

```bash
~/.pyenv/versions/qlib/bin/python fetch_etf_klines.py
```

输出示例：

```
== 拉取窗口 2023-08-14 -> 2026-08-13（含今天） | 模式：full ==
[full] 510050.SH  2023-08-14 -> 2026-08-13 ... +727 行（去重 0），文件 727 行
[full] 510300.SH  2023-08-14 -> 2026-08-13 ... +727 行（去重 0），文件 727 行
...
== 汇总 ==
成功 6/6，失败 0；本次新增 bar 总数 4362
```

### 5.2 指定时间窗

```bash
~/.pyenv/versions/qlib/bin/python fetch_etf_klines.py \
  --start-date 2024-01-01 --end-date 2025-12-31
```

### 5.3 增量更新（每日跑一次）

```bash
~/.pyenv/versions/qlib/bin/python fetch_etf_klines.py --incremental
```

- 每只 ETF **独立判断**起点：读现有 CSV 的最大日期，从 `+1` 开始拉
- 已是最新的会被 `[skip]`，不做无效请求
- 落盘前按日期去重（`keep=last`，新数据优先），不会出现重复

### 5.4 质量检查（不下载）

```bash
~/.pyenv/versions/qlib/bin/python fetch_etf_klines.py --quality-check
```

只扫描 `.cache/klines/*.csv`，对每份文件做 7 项检查并按需 WARN：

```
[OK ] 510050.SH  rows=727  2023-08-14 -> 2026-08-13
[WARN] 159915.SZ rows=727  2023-08-14 -> 2026-08-13  单日涨跌 > 50% 共 1 行，可能是复权跳变（正常）或脏数据（需复核）
...
```

### 5.5 子集

```bash
~/.pyenv/versions/qlib/bin/python fetch_etf_klines.py \
  --codes 510050.SH,159915.SZ
```

### 5.6 自定义数据目录

```bash
~/.pyenv/versions/qlib/bin/python fetch_etf_klines.py \
  --data-dir /path/to/your/klines
```

## 六、参数总览

| 参数 | 默认 | 说明 |
|---|---|---|
| `--start-date` | `今天 - 3 年` | 起始日期 `YYYY-MM-DD` |
| `--end-date` | `今天` | 结束日期 `YYYY-MM-DD` |
| `--codes` | 全部 6 只 | 逗号分隔 ETF 列表 |
| `--data-dir` | `<项目根>/.cache/klines` | CSV 落盘目录 |
| `--incremental` | 关 | 增量模式：每只 ETF 从已有最新日期往后补 |
| `--quality-check` | 关 | 只做质量检查，不下载 |

## 七、质量检查项

| # | 检查 | 触发 WARN 的条件 |
|---|---|---|
| 1 | 必需列 | 缺 `date/open/high/low/close/volume/amount` 任一列 |
| 2 | 空值 | 上述列出现 `NaN` |
| 3 | OHLC 关系 | 出现 `high < max(O,C)` 或 `low > min(O,C)` |
| 4 | 负数 volume | `volume < 0` 任何一行 |
| 5 | volume=0 占比 | > 30%（正常 ETF 极少出现零成交日） |
| 6 | 重复日期 | 同一日期出现 ≥ 2 次 |
| 7 | 30+ 天断档 | 相邻 bar 日期差 > 30 个日历日（长假最大 10 天） |
| 8 | 单日 > 50% 涨跌 | ETF 前复权遇分红/折算时会出现 400% 之类视觉跳变，是正常的；阈值放 50% 兼顾 |

## 八、典型工作流

### 8.1 一次性把宽基 ETF 历史数据落盘

```bash
~/.pyenv/versions/qlib/bin/python fetch_etf_klines.py
# 一两分钟拿到 6 只 ETF × 727 根 K 线
```

### 8.2 配 cron 每天增量

```bash
# crontab -e
30 16 * * 1-5 cd /Users/huhao/src/codesnip/python/ai/023-stock-filter && \
  ~/.pyenv/versions/qlib/bin/python fetch_etf_klines.py --incremental \
  >> .cache/incremental.log 2>&1
```

### 8.3 拉数据后跑回测 / 画图

```python
import pandas as pd

df = pd.read_csv(".cache/klines/510300.SH.csv", parse_dates=["date"])
print(df.head())
print(f"区间: {df['date'].min().date()} -> {df['date'].max().date()}, 共 {len(df)} 根")

# 计算对数收益
import numpy as np
df["log_ret"] = np.log(df["close"] / df["close"].shift(1))
print(f"年化波动率: {df['log_ret'].std() * np.sqrt(252):.2%}")
```

## 九、故障排查

| 现象 | 排查方向 |
|---|---|
| `ConnectionClosedError: unable to connect to any 7709 host` | 网络不通 TDX 主站。`nc -zv 116.205.183.150 7709` 看是否能连 |
| 全部 ETF 显示 `无数据` | `fetch_one` 的过滤/翻页逻辑出错。先看 `--quality-check` 输出确认 CSV 没坏；若是首次拉取 + 没数据，重点看 eltdx 翻页 |
| `--incremental` 没 skip 也没新增 | 现有 CSV 的 `date` 列格式异常；脚本假设 `YYYY-MM-DD` |
| `--quality-check` 大量 WARN | 可能 eltdx 改了字段名或单位；对照 `docs/eltdx-kline.md` 第三节检查脚本里的 `_bar_to_row` |

## 十、相关文档

- `docs/eltdx-kline.md` —— `TdxClient` 库层面的 K 线拉取教程（讲清翻页和复权的坑）
- `docs/cache_codes.md` —— 另一份脚本的文档，可作为风格参考
- `README.md` —— 项目整体说明