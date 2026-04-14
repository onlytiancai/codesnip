---
title: "IV Rank 与 IV Percentile"
type: concept
tags: [options, implied-volatility, trading-conditions]
sources: [spx-iron-condor-ihuhao]
last_updated: 2026-04-14
---

## 概述

**IV Rank（隐含波动率排名）** 和 **IV Percentile（隐含波动率百分位）** 是用于评估当前隐含波动率（IV）水平的量化指标，帮助交易者判断当前期权定价是否处于高位或低位。

## IV Rank vs IV Percentile

- **IV Rank**：将当前 IV 与过去一年最高和最低 IV 进行比较，例如 IV Rank = 80 表示当前 IV 处于过去一年 80% 的位置
- **IV Percentile**：计算过去一年 IV 处于当前水平或更低水平的时间百分比，更适合捕捉 IV 分布

## Iron Condor 入场标准

根据 [[spx-iron-condor-ihuhao]]，Iron Condor 的理想 IV 环境：

| 指标 | 入场要求 | 最佳环境 |
|---|---|---|
| IV Percentile | > 40 | > 50 |
| VIX | > 20 | 22–28 |

## 使用原则

- **高 IV 环境**（IV Rank/Percentile 高）：权利金更丰厚，适合卖出期权（收取更多权利金）
- **低 IV 环境**：权利金较低，不适合做 Iron Condor，应考虑其他策略
- **IV Spike**：刚经历 IV 急剧上升后不适合追入，应等待 IV 稳定

## 与 VIX 的关系

VIX 是衡量市场恐慌预期的指标，通常与 IV Percentile 一起使用：
- VIX > 20 表示市场有一定波动率
- VIX 22–28 是 Iron Condor 的理想区间
- VIX < 15 通常表示市场过于平静，权利金不足

## 相关概念

- [[IronCondor]] — 使用 IV Rank/Percentile 筛选入场时机的主要策略
- [[Vega风险]] — IV 变化对期权组合的影响
