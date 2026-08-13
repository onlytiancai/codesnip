# 宽基ETF多因子轮动策略回测 — 使用文档

## 概述

`backtest_etf_rotation.py` 是 A 股宽基 ETF 多因子轮动策略的第一版（MVP）回测脚本，
依据策略规范 `docs/豆包ETF轮动策略描述.md` 实现。

核心要点：
- **6 只宽基 ETF** 月末调仓（动量 60% + 下行波动率 30% + 流动性 10%）
- **3 触发熊市择时** → 缩权益 → 剩余资金配 **国债 ETF**
- **5% 调仓阈值 + 单标的 40% 上限 + 波动率倒数加权**
- 输出：业绩指标、因子 IC、场景分段、权重时序、换手率、7 张图

## 安装依赖

```bash
~/.pyenv/versions/qlib/bin/python -m pip install empyrical
```

只用 empyrical 一个库（其它 pandas / numpy / matplotlib / seaborn / scipy 项目已有）。

## 数据准备

### 1. 拉 ETF 日 K 线

策略需要 6 只宽基 ETF + 1 只国债 ETF 的日 K 线（前复权 qfq）。
脚本 `fetch_etf_klines.py` 已经默认包含 511010.SH（国债 ETF），
直接跑：

```bash
~/.pyenv/versions/qlib/bin/python fetch_etf_klines.py --lookback-years 10
```

输出落到 `.cache/klines/<code>.csv`，列：`date, open, high, low, close, volume, amount`。

### 2. 沪深300指数

**eltdx 服务端对 `sh000300` 日 K 有协议 bug**（`ProtocolError: invalid kline date`），
所以脚本默认回落 ETF 代理：**`510300.SH`**（华泰柏瑞沪深300ETF），
与沪深300指数相关系数 > 0.999，足以驱动 MA/波动率触发。

如未来 eltdx 修复，可自行准备 `.cache/index/000300.SH.csv`，
列结构：`date, close`。回测脚本会优先使用。

## 运行

```bash
~/.pyenv/versions/qlib/bin/python backtest_etf_rotation.py \
  --start-date 2021-04-01 --end-date 2026-08-12 \
  --output-dir .cache/backtest/
```

### CLI 参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `--start-date` | `2021-04-01` | 回测起始（避开冷启动期） |
| `--end-date` | `2026-08-12` | 回测截止 |
| `--data-dir` | `.cache/klines/` | ETF CSV 目录 |
| `--index-csv` | （无） | 沪深300指数 CSV；不存在回落 ETF 代理 |
| `--output-dir` | `.cache/backtest/` | 业绩报告 + 出图目录 |
| `--slippage-bps` | `15` | 单边滑点（基点） |
| `--rebal-threshold` | `0.05` | 5% 调仓阈值 |
| `--max-weight` | `0.40` | 单标的权重上限 |
| `--mom-window` | `120` | 动量回看窗口 |
| `--down-vol-window` | `60` | 下行波动窗口 |
| `--liq-window` | `20` | 流动性窗口 |
| `--vol-window` | `60` | 加权用总波动窗口 |
| `--bear-trigger1` | `2` | 触发1：池内站上60日均线数量 ≤ N |
| `--bear-vol` | `0.25` | 触发2：沪深300 60日年化波动率 > X |
| `--bear-ma` | `120` | 触发3：沪深300 收盘 < MA_N |

## 输出文件

```
.cache/backtest/
├── metrics.txt             # 业绩指标（年化、Sharpe、Calmar、α/β 等）
├── trade_log.csv           # 每次调仓的买卖明细 + 滑点成本
├── weights.csv             # 调仓日 × ETF 权重时序（含跳过的）
├── turnover.csv            # 每日换手率
├── factor_ic.csv           # 综合得分 Pearson + Spearman IC 时序
├── regime_breakdown.csv    # 牛/震/熊分组年化、回撤、夏普
├── bear_scale.csv          # 每日 bear scale（额外，调试用）
├── nav.png                 # 净值曲线 + 沪深300 + bear 区间灰底
├── drawdown.png            # 回撤曲线
├── monthly_returns.png     # 月度收益热力图
├── ic.png                  # IC 时序 + 滚动 12m IR
├── weights_heatmap.png     # 调仓日 × ETF 权重堆叠热力图
├── turnover.png            # 月度换手率柱状图
└── regime.png              # 牛/震/熊 分段柱状图
```

## 策略核心算法（实现要点）

### 1. 因子计算
```
mom      = close / close.shift(120) - 1                         # 120 日动量
down_vol = std(neg_returns).rolling(60) * sqrt(252)             # 60 日下行波动率
liq      = amount.rolling(20).mean()                            # 20 日均成交额
vol      = std(log_returns).rolling(60) * sqrt(252)             # 60 日年化波动率（用于加权）
```

### 2. 截面打分
```
z_mom, z_dv, z_liq = 各自做横截面 Z-score
score = 0.6·z_mom - 0.3·z_dv + 0.1·z_liq

# 硬过滤：跌破 60 日均线 → 分数 ×0.5；下行波动 > 40% → 分数置 NaN（剔除）
```

### 3. 权重构造
```
weight_i ∝ max(0, score_i) / max(vol_i, 5%)
归一化到 sum=1，再 cap 在 40% 上限（迭代摊薄）
```

### 4. 熊市择时（3 触发任一命中）
```
触发1：池内站上 60 日均线数量 ≤ 2
触发2：沪深300 60 日年化波动率 > 25%
触发3：沪深300 收盘 < MA120

Scale 映射：0 触发 → 1.0；1 → 0.6；2 → 0.4；3 → 0.3
新权益权重 = 目标权重 × scale；剩余 1-scale 配国债 ETF
```

### 5. 调仓执行
- 候选调仓日 = 每月最后一个交易日
- 触发判断：任一标的 |new_w - current_w| > 5%
- 触发则双边扣滑点 0.15%
- 首次建仓（从 0 → 目标权重）不算换手、不扣滑点

## 已知限制

- **冷启动期**：159845.SZ 上市于 2021-03，588000.SH 上市于 2020-11。
  默认 `--start-date 2021-04-01` 已规避。
- **过度拟合**：6 个标的样本极少，因子权重（60/30/10）来自策略文档先验，
  回测只是验证合理性，不做参数寻优。
- **数据缺失**：511010.SH 上市时间晚于宽基；脚本兼容「ETF 在某调仓日
  尚未上市 → 不入选打分截面」。
- **eltdx 服务端 bug**：`sh000300` 日 K 解码失败，故回落 510300.SH ETF 代理。
- **首次建仓处理**：第一次调仓日从 0 → 目标权重不算换手、滑点。

## MVP 范围内、不在计划内的功能（v2 候选）

- Smart Beta 子策略独立回测
- 风险平价 / cvxpy 优化权重
- 动态因子 IC 加权（用滚动 IC 调整 60/30/10 权重）
- 指数成分股广度因子（进阶增强）
- 多空对冲 / 杠杆配置

## 复现命令速查

```bash
# 1. 安装
~/.pyenv/versions/qlib/bin/python -m pip install empyrical

# 2. 拉数据
~/.pyenv/versions/qlib/bin/python fetch_etf_klines.py --lookback-years 10

# 3. 跑回测
~/.pyenv/versions/qlib/bin/python backtest_etf_rotation.py \
  --start-date 2021-04-01 --end-date 2026-08-12 \
  --output-dir .cache/backtest/
```