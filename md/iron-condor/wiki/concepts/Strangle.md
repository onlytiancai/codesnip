---
title: "Strangle（宽跨式组合）"
type: concept
tags: [options, neutral-strategy]
sources: [iron-condor-optionstrategiesinsider]
last_updated: 2026-04-14
---

## 概述

**Strangle（宽跨式组合）** 是一种中性策略，与 [[Straddle]] 类似，但两张期权的行权价不同（均为 OTM，即价外）。Iron Condor 本质上就是一个加了两侧多头保护的 Strangle — 两翼（long wings）限制了风险。

## Strangle 与 Iron Condor 对比

| 特征 | Strangle | Iron Condor |
|---|---|---|
| 合约腿数 | 2 | 4 |
| 最大亏损（空头） | 无限 | 有限 |
| 建仓成本 | 较低（无多头保护） | 较高（需支付两翼保费） |
| 风险特征 | 风险无限 | 风险有限 |

空头 Strangle 是 Iron Condor 的起点 — Iron Condor 在其基础上买入更价外的期权来限定最大亏损。

## 相关概念

- [[IronCondor]] — Iron Condor 是风险限定的 Strangle
- [[Straddle]] — ATM 版本，两张期权行权价相同
