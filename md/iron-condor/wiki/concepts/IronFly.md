---
title: "Iron Fly（铁飞）"
type: concept
tags: [options, neutral-strategy, defined-risk]
sources: [iron-condor-tastylive]
last_updated: 2026-04-14
---

## 概述

**Iron Fly（铁飞）** 是 [[IronCondor]] 的一种变体。当 Iron Condor 的一侧被测试（可能亏损），而另一侧盈利且处于较远价外时，交易者将盈利侧的空头行权价移仓至测试侧相同的行权价，从而将盈利区间压缩为单一价格——即形成 Iron Fly。它是通过主动管理而非直接新建仓形成的。

## 形成方式

当 [[IronCondor]] 某一侧被测试（可能陷入困境），另一侧盈利且位于较远价外时，交易者可以将盈利侧的短行权价向测试侧的短行权价移动。当两侧空头行权价落在同一价格时，结构就变成了 Iron Fly。

## 与 Iron Condor 的关系

Iron Fly 与 [[IronButterfly]] 结构完全等价 — 两者都是空头看跌和空头看涨在同一行权价（ATM）。区别在于到达路径：

- **[[IronButterfly]]**：直接作为新仓位入场
- **Iron Fly**：从已有 Iron Condor 通过移仓演变而来

## 相关概念

- [[IronCondor]] — 母策略；移仓一侧即形成 Iron Fly
- [[IronButterfly]] — 一旦两侧空头行权价相同，结构上完全等价
