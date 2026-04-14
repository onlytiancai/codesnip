---
title: "Bear Call Spread（熊市看涨价差）"
type: concept
tags: [options, bearish-strategy, credit-spread]
sources: [iron-condor-optionstrategiesinsider]
last_updated: 2026-04-14
---

## 概述

**Bear Call Spread（熊市看涨价差）** 是一种看跌期权策略，操作方式为卖出低行权价看涨期权，同时买入更高行权价看涨期权（相同到期日）。它是 [[IronCondor]] 中看涨（call）那一侧的两个腿。

## 合约结构

- **卖出 1 张 OTM 看涨期权**（较低行权价，靠近当前价格）
- **买入 1 张 OTM 看涨期权**（较高行权价，距离当前价格更远）

若股价大幅上涨，多头看涨期权限定亏损。

## 关键属性

| 属性 | 值 |
|---|---|
| 方向偏向 | 看跌（Neutral 到小幅看跌均可获利） |
| 最大盈利 | 收取的净权利金 |
| 最大亏损 | 价差宽度 − 净权利金 |
| 理想 IV 环境 | 高 IV（权利金更丰厚） |

## 在 Iron Condor 中的角色

在 [[IronCondor]] 中，Bear Call Spread 位于结构的上方（看涨侧）。当股价在到期时保持在空头看流行权价以下时盈利。

## 相关概念

- [[IronCondor]] — Bear Call Spread 是其组成部分之一
- [[BullPutSpread]] — Iron Condor 中对应另一侧的看跌腿
