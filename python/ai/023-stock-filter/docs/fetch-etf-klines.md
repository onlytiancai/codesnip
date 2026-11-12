# fetch_etf_klines.py 使用文档

批量获取宽基 ETF 日 K 线（前复权）的脚本，覆盖六只核心宽基指数对应的 ETF：上证50、沪深300、中证500、中证1000、创业板指、科创50。

## 一、用途

回测/分析时一次性把多只宽基 ETF 的历史 K 线拉到本地 CSV 里。四种典型场景：

1. **首次建库** —— 拉最近 3 年数据落到 `.cache/klines/`，每只 ETF 一份 CSV
2. **每日增量** —— 自动识别每只 ETF 已落盘文件的最新一日，往后补到今天
3. **往前扩展** —— 已落盘只到 3 年时，补拉早期数据 prepend 到现有 CSV（不动现有数据），拉满到 10 年
4. **质量复核** —— 对已落盘文件做 OHLC 关系、断档、复权跳变等检查，不下载

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
| `.cache/eltdx_codes.json` | 共享代码表（由 `test-scripts/cache_codes.py refresh` 生成），用作"代码 → 中文名"查找表；找不到时显示 `?` |
| `.cache/` | 已被 `.gitignore` 忽略 |

CSV 列：`date, open, high, low, close, volume, amount`

| 列 | 类型 | 含义 |
|---|---|---|
| `date` | `YYYY-MM-DD` | 交易日（按 CST 时区） |
| `open` / `high` / `low` / `close` | `float` | OHLC 价（前复权 qfq） |
| `volume` | `int` | 成交量，**单位为"手"**（1 手 = 100 份） |
| `amount` | `float` | 成交额（元） |

## 四、ETF 默认列表

| 代码 | 名称（eltdx 缓存里） | 跟踪指数 |
|---|---|---|
| `510050.SH` | 上证50ETF华夏 | 上证50 |
| `510300.SH` | 沪深300ETF华泰柏 | 沪深300 |
| `510500.SH` | 中证500ETF南方 | 中证500 |
| `159845.SZ` | 中证1000ETF华夏 | 中证1000 |
| `159915.SZ` | 创业板ETF易方达 | 创业板指 |
| `588000.SH` | 科创50ETF华夏 | 科创50 |

> 表中"名称"是 eltdx 代码表里的简称，跟交易所/基金公告里的全称略有差异（顺序与字眼都不同）。脚本运行时直接从 `.cache/eltdx_codes.json` 读这份简称，找不到就显示 `?`。

列表在脚本顶部 `DEFAULT_ETFS`，可改。

## 五、命令行用法

### 5.1 首次拉取（默认最近 3 年，截止到昨天）

```bash
~/.pyenv/versions/qlib/bin/python fetch_etf_klines.py
```

> **截止到昨天**：eltdx 在盘中/盘前对"今日"只返回一行占位（OHLC=前收、volume=0、amount=0），不是真实数据。脚本默认 `end_d = today - 1` 避开它。`fetch_one` 内还有兜底过滤：如果用户显式给了 `--end-date=今天`，也会把那行占位剔掉。

输出示例：

```
== 拉取窗口 2023-08-13 -> 2026-08-12（默认截止昨天，规避 eltdx 当日占位） | 模式：full ==
[full] 510050.SH  2023-08-13 -> 2026-08-12 ... +726 行（去重 0），文件 726 行
[full] 510300.SH  2023-08-13 -> 2026-08-12 ... +726 行（去重 0），文件 726 行
...
== 汇总 ==
成功 6/6，失败 0；本次新增 bar 总数 4356
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

不带 `--qc-start-date/--qc-end-date` 时，默认扫描每只 ETF 的**全部历史**。要限定子区间，见 [§5.8](#58-质量检查指定日期区间)。

扫描 `.cache/klines/*.csv`，输出三块信息：

1. **概述表**：每只 ETF 一行的紧凑汇总 —— 代码 / 名称 / 条数 / 日期范围 / 年化收益 / 最大回撤 / 波动率
2. **单只 ETF 的检查项 + 描述统计**：缺失值、价格/收益的 mean/std/min/max、年化收益与波动率、最大回撤及对应日期窗口、成交量分布
3. **跨 ETF 相关性矩阵**：按日期 inner join 对齐后，对日对数收益计算 Pearson + Spearman 相关

代码旁的简称取自 `.cache/eltdx_codes.json`；找不到时显示 `?`。

示例输出片段（10 年数据，6 只 ETF 全部对齐 1301 个交易日）：

```
--- 概述 ---
代码          名称                条数    日期范围                      年化收益   最大回撤   波动率
------------  ------------------  ------  ----------------------------  ---------  ---------  --------
510050.SH     上证50ETF华夏       2426    2016-08-15 ~ 2026-08-12       + 5.28%    -58.30%    20.11%
510300.SH     沪深300ETF华泰柏    2426    2016-08-15 ~ 2026-08-12       + 5.74%    -59.33%    20.94%
510500.SH     中证500ETF南方      2426    2016-08-15 ~ 2026-08-12       + 3.99%    -54.14%    23.16%
159845.SZ     中证1000ETF华夏     1302    2021-03-31 ~ 2026-08-12       + 5.77%    -61.23%    24.35%
159915.SZ     创业板ETF易方达     2425    2016-08-15 ~ 2026-08-12       + 5.65%    -83.44%    29.08%
588000.SH     科创50ETF华夏       1393    2020-11-16 ~ 2026-08-12       + 4.06%    -90.73%    31.73%

--- 单只 ETF 检查 + 描述统计 ---
[OK ] 510050.SH 上证50ETF华夏    rows=2426   2016-08-15 -> 2026-08-12
        区间 3649 日历日；缺失值 无
        close     : mean=2.57 std=0.40 min=1.71 max=3.76
        日对数收益: mean=+ 0.02% std= 1.27% min=-8.68% max=+ 9.33%
        年化      : 收益=+ 5.28% 波动=20.11%
        最大回撤  : -58.30% (2021-02-10 ~ 2024-01-17)
        成交量(百万手): mean=  6.82 median=  5.89 min=  0.78 max= 68.47

--- Pearson 相关系数（n=1301 交易日对齐）---
             510050.SH   510300.SH   510500.SH   159845.SZ   159915.SZ   588000.SH
510050.SH         1.000       0.936       0.674       0.566       0.665       0.537
510300.SH         0.936       1.000       0.829       0.753       0.849       0.699
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

### 5.7 往前扩展历史数据（仅补缺）

```bash
~/.pyenv/versions/qlib/bin/python fetch_etf_klines.py --lookback-years 10
```

适用场景：之前只拉了 3 年，现在想补到 10 年，又不想覆盖掉现有 CSV。

行为说明：

- **不动现有数据**：只在每个 ETF 现有 CSV 的"前头"插入缺失的早期数据；现有数据完全保留
- **按 ETF 独立判断**：每只 ETF 读完自己现有 CSV 的最早日期，若已经早于目标起点就 `[skip]`，否则只拉 `[目标最早, 现有最早-1]` 这段区间
- **按日期去重（新数据优先）**：理论上同一天的复权数据 eltdx 给的值一致，但翻页边界可能出现一次重复，新数据用 `keep="first"` 生效
- **不影响上市时间晚的 ETF**：159845.SZ（中证1000ETF华夏 2021-03-31 上市）和 588000.SH（科创50ETF华夏 2020-11-16 上市）即使配 10 年也只会拉出上市后到目标之间的数据

### 5.8 质量检查指定日期区间

```bash
~/.pyenv/versions/qlib/bin/python fetch_etf_klines.py \
  --quality-check --qc-start-date 2024-01-01 --qc-end-date 2024-12-31
```

适用场景：只想看最近一年 / 某次牛熊市窗口的回撤与相关性，不想被全部 10 年历史稀释掉。

行为说明：

- **按 ETF 各自截取**：每只 ETF 先按 `[start, end]` 闭区间截取自己的 CSV，再做检查与描述统计
- **不影响相关性矩阵的对齐样本**：`load_close_series` 也按区间截，再 inner join，所以 Pearson / Spearman 的 n 来自你指定的窗口
- **单边 OK**：`--qc-start-date` 与 `--qc-end-date` 都可以单独给，留空那一边默认无穷大 / -∞
- **空区间是合法结果**：区间内没数据会打 `WARN ... 内无数据`，并不报错
- **校验**：`start > end` 直接退出码 2，与 `--start-date/--end-date` 的下载模式一致

实操举例：把窗口限定到 2024 年，6 只 ETF 都剩 242 行（2024 年交易日数），年化收益、波动、回撤、相关性都只在 2024 年样本上算：

```
== 质量检查：扫描 /…/.cache/klines | 区间 2024-01-01 ~ 2024-12-31 ==

--- 概述 ---
代码          名称                条数    日期范围                      年化收益   最大回撤   波动率
------------  ------------------  ------  ----------------------------  ---------  ---------  --------
510050.SH     上证50ETF华夏       242     2024-01-02 ~ 2024-12-31       +20.16%    -12.00%    20.56%
510300.SH     沪深300ETF华泰柏    242     2024-01-02 ~ 2024-12-31       +19.16%    -14.08%    23.71%
510500.SH     中证500ETF南方      242     2024-01-02 ~ 2024-12-31       + 7.90%    -21.92%    30.62%
…
--- 跨 ETF 相关性（日对数收益，n=242 个交易日对齐）---
```

实操举例（从 3 年扩到 10 年）：

```
== 扩展到最近 10 年（目标最早 2016-08-14） ==
[prepend] 510050.SH 上证50ETF华夏  拉 2016-08-14 -> 2023-08-13 ... +1700 行（去重 0），文件 2426 行
[prepend] 510300.SH 沪深300ETF华泰柏  拉 2016-08-14 -> 2023-08-13 ... +1700 行（去重 0），文件 2426 行
[prepend] 510500.SH 中证500ETF南方  拉 2016-08-14 -> 2023-08-13 ... +1700 行（去重 0），文件 2426 行
[prepend] 159845.SZ 中证1000ETF华夏  拉 2016-08-14 -> 2023-08-13 ... +576 行（去重 0），文件 1302 行
[prepend] 159915.SZ 创业板ETF易方达  拉 2016-08-14 -> 2023-08-13 ... +1699 行（去重 0），文件 2425 行
[prepend] 588000.SH 科创50ETF华夏  拉 2016-08-14 -> 2023-08-13 ... +667 行（去重 0），文件 1393 行
```

> 为什么用 prepend 而不是从头下载？回到头来重拉 10 年再写 csv 显然更浪费（已有 726 行完全有效），prepend 只需要拉缺失的[2016-08-14, 2023-08-13]这段就能补齐，现存数据一个字都不动。

### 5.9 质量检查生成曲线图（不下载，只画图）

```bash
~/.pyenv/versions/qlib/bin/python fetch_etf_klines.py \
  --quality-check --qc-start-date 2024-01-01 --qc-end-date 2024-12-31 \
  --qc-plot
```

适用场景：质量检查的数字结果看着别扭，想直接看曲线 / 看区间起点终点 / 看相关性热力图。开关不影响数字输出，只是在终端报告之上**额外**往目录里写 PNG。

行为说明：

- **默认输出目录**：`<项目根>/.cache/qc_plots/`，可用 `--qc-plot-dir` 覆盖
- **每只 ETF 一张 PNG**：`<code>.png`，内容为 close 曲线 + 区间首末标记 + 标题（区间收益、年化收益、波动、最大回撤）
- **额外一张相关性热力图**：`corr_pearson.png`，跨 ETF 日对数收益 Pearson 矩阵
- **一张归一化性能对比图**：`combined.png`，把所有 ETF 的 close 按"首日 = 100"归一化后画在同一个坐标轴上，颜色区分 + 终点标注累计收益%——一眼看清哪只 ETF 在区间内跑得最好
- **强制 Agg backend**：脚本内部 `matplotlib.use("Agg")`，不需要 GUI / X server；适合 cron / SSH 跑
- **中文字体回退**：依次尝试 `PingFang SC → Hiragino Sans GB → Heiti TC → DejaVu Sans`。macOS 系统里的 PingFang.ttc 默认不在 matplotlib fontManager 索引里，脚本会主动 `addfont` 一次让它可解析。三档全挂时退到 DejaVu Sans，并把 WARN 打到 stderr（曲线图里的 CJK 字会变方框但不崩）
- **目录不存在会自动建**：`--qc-plot-dir` 指向的路径若不存在会 `mkdir -p`
- **空数据安全**：单只 ETF 在区间内没数据时不会单独画图，也不报错；只影响那一张 PNG 缺席

实操举例（一年窗口）：

```
== 质量检查：扫描 /…/.cache/klines | 区间 2024-01-01 ~ 2024-12-31 ==
…（跟 §5.4 一样的数字输出）…
[plot] 510050.SH -> /…/.cache/qc_plots/510050.SH.png
[plot] 510300.SH -> /…/.cache/qc_plots/510300.SH.png
…
[plot] 相关性热力图 -> /…/.cache/qc_plots/corr_pearson.png
[plot] 归一化性能对比 -> /…/.cache/qc_plots/combined.png
== 共生成 8 张 PNG ==
```

## 六、参数总览

| 参数 | 默认 | 说明 |
|---|---|---|
| `--start-date` | `end_d - 3 年` | 起始日期 `YYYY-MM-DD` |
| `--end-date` | `今天 - 1 天`（昨天） | 结束日期 `YYYY-MM-DD` |
| `--codes` | 全部 6 只 | 逗号分隔 ETF 列表 |
| `--data-dir` | `<项目根>/.cache/klines` | CSV 落盘目录 |
| `--incremental` | 关 | 增量模式：每只 ETF 从已有最新日期往后补 |
| `--quality-check` | 关 | 只做质量检查，不下载 |
| `--qc-start-date` | 无 | 质量检查起始日期 `YYYY-MM-DD`（含），仅与 `--quality-check` 一起生效；按 `[start, end]` 闭区间截取每只 ETF 后再算检查与相关性 |
| `--qc-end-date` | 无 | 质量检查结束日期 `YYYY-MM-DD`（含），同 `--qc-start-date`；单边可独立给 |
| `--qc-plot` | 关 | 质量检查生成曲线图：每只 ETF 一张 close 曲线 PNG + 一张相关性热力图 + 一张归一化性能对比图（combined.png）；强制 Agg backend，中文字体自动回退；仅与 `--quality-check` 一起生效 |
| `--qc-plot-dir` | `<项目根>/.cache/qc_plots` | `--qc-plot` 输出目录；不存在会自动建 |
| `--lookback-years` | 关 | 往前扩展到 N 年：只补缺，prepend 到现有 CSV；现有数据不动 |

## 七、质量检查项

### 7.1 检查类（触发 WARN 即可能脏数据）

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

### 7.2 描述统计（每只 ETF 一段）

| 字段 | 含义 |
|---|---|
| 区间 | 起止日期差（日历日） |
| 缺失值 | 各列 NaN 数量 |
| close mean/std/min/max | 收盘价的均值/标准差/最小/最大 |
| 日对数收益 mean/std/min/max | `log(close_t / close_{t-1})` 的统计量 |
| 年化收益 | `exp(log(close_end / close_start) × 252/n_days) - 1`（几何年化） |
| 年化波动率 | `日对数收益 std × sqrt(252)` |
| 最大回撤 | 累计对数收益曲线峰值到谷值的最大跌幅（负数） |
| 成交量 mean/median/min/max | 单位"百万手" |

### 7.3 跨 ETF 相关性

对所有 ETF 的 close 序列按日期 inner join 对齐后，对日对数收益计算两个相关矩阵：

- **Pearson 相关系数**：衡量线性相关，对极端值敏感
- **Spearman 秩相关**：衡量单调相关，对极端值稳健

> inner join 会丢掉"某只 ETF 没交易但其它有"的日期，所以 n 可能小于单只 ETF 的总行数。示例中 6 只 ETF 共 726 根 bar，对齐后剩 725 个交易日（一般只有 1~2 天差异）。
>
> **样本量阈值**：`MIN_OBS_FOR_CORR = 30`。如果某只 ETF 数据偏短（比如测试覆盖成 3 行），inner join 后只剩 2~3 个交易日；这时 Pearson/Spearman 数学上必为 ±1（任意两点共线），全矩阵显示成"全是 1.000"会误读为高度相关。脚本在 n < 30 时会跳过矩阵输出并打 WARN 提示。

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

### 8.3 把现有数据从 3 年扩到 10 年

```bash
~/.pyenv/versions/qlib/bin/python fetch_etf_klines.py --lookback-years 10
# 只下载缺失的早期数据，prepend 到现有 CSV；现有数据不动
```

### 8.4 拉数据后跑回测 / 画图

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