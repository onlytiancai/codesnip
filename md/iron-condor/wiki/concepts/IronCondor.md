---
title: "Iron Condor（铁秃鹰）"
type: concept
tags: [options, neutral-strategy, defined-risk, credit-spread]
sources: [iron-condor-optionstrategiesinsider, iron-condor-tastylive]
last_updated: 2026-04-14
---

## 概述

**Iron Condor（铁秃鹰）** 是一种方向中性、风险有限的期权策略，收益来源于标的资产在到期前维持在一个价格区间内波动。构建方式为同时卖出价外（OTM）熊市看涨价差（Bear Call Spread）和价外牛市看跌价差（Bull Put Spread），两个价差同一到期日。

## 合约结构

该策略由四个期权腿组成：

1. **卖出 1 张 OTM 看跌期权**（较高的空头行权价，靠近当前价格）
2. **买入 1 张 OTM 看跌期权**（更低行权价，限定看跌侧最大亏损）
3. **卖出 1 张 OTM 看涨期权**（较低的空头行权价，靠近当前价格）
4. **买入 1 张 OTM 看涨期权**（更高行权价，限定看涨侧最大亏损）

## 关键属性

| 属性 | 值 |
|---|---|
| 方向偏向 | 中性（Neutral） |
| 最大盈利 | 建仓时收取的净权利金 |
| 最大亏损 | 最宽价差宽度 − 净权利金 |
| 理想 IV 环境 | 高 IV（权利金更丰厚） |
| 距到期日 | 建议 30–60 天 |
| 成功率 | 行权价越价外，成功率越高 |

## 入场规则（tastylive）

> "我们以收取价差宽度 1/3 的权利金为目标来入场。例如，3 点宽的价差，我们希望收取 $1.00 的权利金。这能带来约 67% 的成功率。"

## 盈亏平衡点

- **上方盈亏平衡点** = 看涨空头行权价 + 净权利金
- **下方盈亏平衡点** = 看跌空头行权价 − 净权利金

## 风险管理

- 达到**最大盈利的 50%** 时平仓，锁定利润并提升长期胜率
- 将未测试（盈利）侧向股价方向移仓，收取额外权利金
- 将未测试侧移至与测试侧相同的行权价，即形成 **[[IronFly]]**
- 在到期日前平仓或调整，避免被行权（Assignment）风险

## 与其他策略的关系

- Iron Condor 本质上是**风险限定的 Strangle** — 两翼的多头期权限制了空头 Strangle 的亏损
- 与 [[IronButterfly]] 相关 — Iron Butterfly 两侧空头行权价相同（ATM），权利金更高但盈亏平衡点更差
- 与 [[ReverseIronCondor]] 相关 — 反向策略，以借方入场，在趋势行情中盈利
- [[BearCallSpread]] 和 [[BullPutSpread]] 是其两个组成价差

## 相关概念

- [[IronCondor]]（自身）
- [[IronButterfly]] — ATM 变体，权利金更高但盈亏平衡点更差
- [[IronFly]] — 将 Iron Condor 一侧移仓至测试侧行权价后形成
- [[ReverseIronCondor]] — 反向借方策略，适用于趋势行情
- [[BearCallSpread]] — 看跌腿
- [[BullPutSpread]] — 看涨腿
- [[Straddle]] — 中性无限风险版本
- [[Strangle]] — 更宽、更便宜但风险更大的版本
