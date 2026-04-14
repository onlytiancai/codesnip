---
title: "Bull Put Spread（牛市看跌价差）"
type: concept
tags: [options, bullish-strategy, credit-spread]
sources: [iron-condor-optionstrategiesinsider]
last_updated: 2026-04-14
---

## 概述

**Bull Put Spread（牛市看跌价差）** 是一种看涨期权策略，操作方式为卖出较高行权价看跌期权，同时买入更低行权价看跌期权（相同到期日）。它是 [[IronCondor]] 中看跌（put）那一侧的两个腿。

## 合约结构

- **卖出 1 张 OTM 看跌期权**（较高行权价，靠近当前价格）
- **买入 1 张 OTM 看跌期权**（较低行权价，距离当前价格更远）

若股价大幅下跌，多头看跌期权限定亏损。

## 关键属性

| 属性 | 值 |
|---|---|
| 方向偏向 | 看涨（Neutral 到小幅看涨均可获利） |
| 最大盈利 | 收取的净权利金 |
| 最大亏损 | 价差宽度 − 净权利金 |
| 理想 IV 环境 | 高 IV（权利金更丰厚） |

## 在 Iron Condor 中的角色

在 [[IronCondor]] 中，Bull Put Spread 位于结构的下方（看跌侧）。当股价在到期时保持在空头看跌行权价以上时盈利。

## 相关概念

- [[IronCondor]] — Bull Put Spread 是其组成部分之一
- [[BearCallSpread]] — Iron Condor 中对应另一侧的看涨腿
