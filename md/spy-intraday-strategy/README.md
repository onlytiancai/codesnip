# SPY日内量化策略研究与回测

基于文档 [1.md](1.md) 的理论分析，对SPY日内量化策略进行实证研究。

## 📊 数据概况

| 文件 | 类型 | 条数 | 时间范围 |
|------|------|------|----------|
| `spy_hourly.csv` | 小时线 | 2,533 | 2025-01-07 ~ 2026-06-23 |
| `spy_1min_recent.csv` | 分钟线 | 2,730 | 2026-06-12 ~ 2026-06-23 |
| `spy_daily.csv` | 日线 | 1,254 | 2021-06-24 ~ 2026-06-23 |

> Yahoo Finance限制：1分钟数据最多7天，小时数据最多730天

## 📁 文件说明

### 数据获取
```bash
# 下载所有类型数据
python download_spy_data.py --type all

# 下载特定类型
python download_spy_data.py --type hourly --days 365
python download_spy_data.py --type 1min
python download_spy_data.py --type daily --years 5
```

### 数据质量检查
```bash
# 检查目录下所有数据
python check_data_quality.py --dir .

# 检查单个文件
python check_data_quality.py spy_hourly.csv
```

### 策略回测
```bash
# 使用小时数据回测所有策略
python backtest.py --data spy_hourly.csv

# 使用1分钟数据回测
python backtest.py --data spy_1min_recent.csv

# 回测指定策略
python backtest.py --data spy_hourly.csv --strategy vwap combo

# 自定义参数
python backtest.py --data spy_hourly.csv --capital 50000 --commission 0.0003
```

## 📈 策略列表

| 策略 | 原理 | 特点 |
|------|------|------|
| **VWAP均值回归** | 价格偏离VWAP后均值回归 | 机构常用，逻辑清晰 |
| **RSI均值回归** | RSI超买超卖后反转 | 信号频繁，需严格过滤 |
| **趋势动量** | 趋势延续时追涨杀跌 | 高胜率但回撤大 |
| **ORB突破** | 开盘区间突破 | 经典日内策略 |
| **VWAP+动量组合** | VWAP偏离 + 动量确认 | 多因子共振，稳定性较高 |

## 🔬 回测结果（spy_hourly.csv - 1年数据）

| 策略 | 交易次数 | 胜率 | 总盈亏 | 夏普比率 | 最大回撤 |
|------|---------|------|--------|---------|---------|
| **VWAP+动量组合** | 91 | 53.8% | **+$8,709** | **1.25** | -10.5% |
| RSI均值回归 | 283 | 56.9% | -$6,295 | -0.61 | -7.6% |
| 趋势动量 | 144 | 36.1% | -$12,553 | -1.53 | -23.2% |
| VWAP均值回归 | 104 | 44.2% | -$14,642 | -1.95 | -15.9% |

> 初始资金 $100,000，佣金0.05%，滑点0.02%

## ⚠️ 重要结论

### 理论要点（来自1.md）

1. **量价≠全部信息**：ES期货、期权GEX/DEX、订单流等都是增量信息
2. **十年数据≠稳定策略**：市场结构变化大，过拟合风险高
3. **Regime切换关键**：不同市场状态需要不同策略
4. **交易成本敏感**：每日3-5次交易，成本控制至关重要

### 实证发现

1. **VWAP+动量组合**是唯一正收益策略（夏普1.25）
2. 单因子策略在样本内普遍表现不佳
3. ORB策略在当前数据上无有效信号（参数需调整）
4. 样本期较短（1年），结论需谨慎看待

### 数据局限性

- Yahoo Finance数据有延迟，不是严格tick数据
- 1分钟数据仅有7天，统计意义有限
- 建议使用付费数据源（AlgoSeek、TickData、IB API）做更严格验证

## 📚 进一步研究方向

1. **Regime分类**：区分Trend Day / Mean Reversion Day / Range-bound Day
2. **GEX/DEX因子**：结合期权市场数据判断价格引力
3. **跨市场验证**：结合ES期货、VIX、债券数据
4. **Walk-Forward回测**：滚动训练避免过拟合
5. **实盘验证**：小资金实盘测试策略稳定性