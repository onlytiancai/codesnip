---
title: "Iron Butterfly（铁蝴蝶）"
type: concept
tags: [options, neutral-strategy, defined-risk]
sources: [iron-condor-tastylive]
last_updated: 2026-04-14
---

## 概述

**Iron Butterfly（铁蝴蝶）** 是一种方向中性、风险有限的期权策略，与 [[IronCondor]] 类似，但区别在于其空头看跌和空头看涨行权价相同（ATM，即平价）。这使得最大盈利区间集中在一个价格点而非一个区间。

## 合约结构

Iron Butterfly 使用三个行权价（共四个腿）：

1. **卖出 1 张 ATM 看跌期权**
2. **买入 1 张 OTM 看跌期权**（较低行权价，限定看跌侧风险）
3. **卖出 1 张 ATM 看涨期权**（与 ATM 看跌同一行权价）
4. **买入 1 张 OTM 看涨期权**（较高行权价，限定看涨侧风险）

注意：空头看跌和空头看涨使用相同行权价 — 这是与 Iron Condor 的核心结构差异。

## Iron Condor 与 Iron Butterfly 对比

| 特征 | Iron Condor | Iron Butterfly |
|---|---|---|
| 空头行权价 | 两个不同行权价（OTM） | 同一行权价（ATM） |
| 盈利区间 | 介于两个空头行权价之间 | 单一行权价 |
| 最大权利金 | 较低 | 较高 |
| 盈亏平衡点 | 相距更远 | 更近 |
| 到期时变为价内（ITM）风险 | 较低 | 较高（几乎总有一侧 ITM） |

Iron Butterfly 因两侧空头均为 ATM，能收到更多权利金，但盈亏平衡点更差，盈利区间更窄。

## 与 Iron Fly 的关系

当把 [[IronCondor]] 的一侧移仓至另一侧相同的空头行权价时，就变成了 [[IronFly]] — 与 Iron Butterfly 结构完全相同。

## 相关概念

- [[IronCondor]] — 更宽区间变体；OTM 行权价提供更高成功率
- [[IronFly]] — Iron Condor 移仓后形成；行权价相同则结构等价
