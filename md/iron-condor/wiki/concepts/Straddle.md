---
title: "Straddle（跨式组合）"
type: concept
tags: [options, neutral-strategy]
sources: [iron-condor-optionstrategiesinsider]
last_updated: 2026-04-14
---

## 概述

**Straddle（跨式组合）** 是一种中性期权策略，同时买入（或卖出）相同行权价、相同到期日的看涨和看跌期权。与 [[IronCondor]] 不同的是，买入 Straddle 风险无限（权利金损失），卖出 Straddle 风险亦无限（标的价格大幅波动时亏损无上限）。

## Straddle 与 Iron Condor 对比

| 特征 | Straddle | Iron Condor |
|---|---|---|
| 合约腿数 | 2 | 4 |
| 最大亏损（多头） | 支付的权利金 | 有限（多方 Iron Condor） |
| 最大亏损（空头） | 无限 | 限定为价差宽度 |
| 盈利区间 | 无限（多头跨式） | 限定为收取的权利金 |

Iron Condor 被视为卖出 Straddle 的更保守替代方案 — 因为两侧的多头翼（long wings）将风险限制在了可预见的范围内。

## 相关概念

- [[IronCondor]] — Iron Condor 是风险限定的 Strangle（更宽版本的 Straddle）
- [[Strangle]] — 类似但行权价不同；盈利区间更宽但风险更高
