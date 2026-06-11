---
title: "Reverse Iron Condor（反向铁秃鹰）"
type: concept
tags: [options, directional-strategy, debit-spread, volatility-strategy]
sources: [iron-condor-tastylive]
last_updated: 2026-04-14
---

## 概述

**Reverse Iron Condor（反向铁秃鹰）**（又称 Long Iron Condor）是一种风险有限、收益有限的期权策略，以**净借方（Net Debit）** 入场，在标的价格出现大幅单边波动时盈利。它是标准 [[IronCondor]] 的完全反向 — Iron Condor 期望标的价格区间内波动，Reverse Iron Condor 则期望大幅单边行情。

## 合约结构

Reverse Iron Condor 由两个借方价差组成：

**看跌侧（看跌方向）：**
- 买入 1 张较低价外看跌期权（靠近当前价格）
- 卖出 1 张更低行权价看跌期权（降低持仓成本）

**看涨侧（看涨方向）：**
- 买入 1 张较高价外看涨期权（靠近当前价格）
- 卖出 1 张更高行权价看涨期权（降低持仓成本）

## 与 Iron Condor 的区别

| 特征 | Iron Condor | Reverse Iron Condor |
|---|---|---|
| 入场方式 | 净信用（收取权利金） | 净借方（支付权利金） |
| 市场预期 | 中性 / 区间震荡 | 波动性 / 趋势行情 |
| 最大盈利 | 限定为收取的权利金 | 有限（但大于 Iron Condor） |
| 最大亏损 | 价差宽度 − 收取权利金 | 限定为支付的借方金额 |
| 盈利触发条件 | 股价保持在两个空头行权价之间 | 股价突破任一价差 |

## 使用场景

在预期标的出现大幅波动时入场 — 例如财报前、FDA 审批等重大事件前后。交易者先支付借方金额，需要其中一个价差变为价内（ITM）且价值超过所支付的借方金额才能盈利。

## 相关概念

- [[IronCondor]] — 反向策略（信用 vs 借方）
- [[IronButterfly]] — 结构相似的 ATM 变体
- [[Strangle]] — 反向 Strangle（借方型）理论收益无限
