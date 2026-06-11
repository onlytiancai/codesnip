---
title: "Doubao SPX Iron Condor 实战对话"
type: source
tags: [spx, iron-condor, greeks, delta, gamma, hedging]
date: 2026-04-15
source_file: raw/doubao-ad4d476d06810.md
---

## Summary

这是一份来自豆包 AI 聊天记录的期权专家对话，聚焦 SPX 铁鹰策略的实战分析。涵盖 Greeks（Delta、Gamma）的量化解读、铁鹰调仓策略、SPY 与 SPX Call 对冲对比、以及专业基金应对单边行情的方法论。

## Key Claims

- 铁鹰策略 Delta=-3.13、Gamma=-0.0284 时，SPX 涨 100 点组合亏损约 313 美元
- Delta≥5 或 Gamma≥0.1 是铁鹰的硬性风险阈值，意味着 100 点波动即可吞噬大部分利润
- SPY 对冲需要 40 股（而非 4 股）才能完全抵消 Delta=-4 的敞口，因 SPX 乘数为 100 而 SPY 无乘数
- 深度虚值 Put（5800/5900）的 Delta 差值小，轻度虚值 Call（7300/7400）的 Delta 差值大，根源在于"虚实程度"不同
- Gamma Cash 是动态对冲 Delta 时产生的现金流，本质是利用高抛低吸获利

## Key Quotes

> "Delta≥5 → 标的 100 点波动就能吞噬大部分利润；Gamma≥0.1 → 标的 50 点波动就能让 Delta 恶化到风险失控的程度"

> "SPY 是'静态、高成本、零维护'的对冲，Call 是'动态、低成本、高维护'的对冲"

> "专业对冲基金在遇到铁鹰大幅单边上涨时，不赌方向，只控风险"

## Greeks 量化逻辑

| 阈值 | 含义 | SPX 波动影响 |
|------|------|-------------|
| Delta≥5 | SPX 每波动 1 点，组合价值变动≥5 美元 | 100 点波动 = 500 美元亏损 |
| Gamma≥0.1 | SPX 每波动 1 点，Delta 变动 0.1 | 50 点波动后 Delta 从 -3 变为 -8 |

## 对冲方案对比

| 维度 | 4 股 SPY | 8 张 Delta=0.5 SPX Call |
|------|----------|-------------------------|
| Delta 贡献 | +4 | +4 |
| Gamma 影响 | 0 | +0.16 |
| 时间价值损耗 | 无 | 有 |
| 调整频率 | 低 | 高 |

## 铁鹰调整思路分类

1. **保守型**：部分止盈、缩小跨度、移动执行价
2. **激进型**：加仓同策略、转裸卖宽跨
3. **中性型**：单边移仓、Delta 中性对冲、滚动到期

## Connections

- [[IronCondor]] — 核心策略
- [[Gamma风险]] — Gamma 量化和阈值逻辑
- [[Delta对冲]] — SPY vs SPX Call 对冲对比
- [[GammaCash]] — Gamma Cash 概念解析

## Contradictions

- 本对话中 SPY 对冲数量（40 股）与之前计算（3-4 股）有重大差异——差异来源：SPX 期权乘数为 100 而非 1，导致实际所需 SPY 数量应为 Delta × 10
