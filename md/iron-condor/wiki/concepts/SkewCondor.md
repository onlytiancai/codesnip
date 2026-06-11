---
title: "Skew Condor（不对称铁秃鹰）"
type: concept
tags: [options, iron-condor, skew, risk-management]
sources: [spx-iron-condor-ihuhao]
last_updated: 2026-04-14
---

## 概述

**Skew Condor（不对称铁秃鹰）** 是 Iron Condor 的一种变体，通过调整看涨侧和看跌侧的宽度来利用期权 Skew 特性。由于市场下跌快而猛、上涨慢而磨的特性，put side 需要更远的保护。

## 构建原则

根据 [[spx-iron-condor-ihuhao]]：

| 侧 | 位置 | 说明 |
|---|---|---|
| Put side | 更远 | 需要更多保护 |
| Call side | 可以稍近 | 上涨速度较慢 |

## 市场不对称性

> "下跌：快 + 猛；上涨：慢 + 磨"

这意味着：
- Put side 被测试到的概率更高
- Put side 一旦被测试，恶化速度更快
- 因此 put wing 需要更宽的缓冲

## 与标准 Iron Condor 的区别

| 属性 | 标准 Iron Condor | Skew Condor |
|---|---|---|
| Put wing 宽度 | 与 Call wing 相等 | 更宽 |
| Call wing 宽度 | 与 Put wing 相等 | 可以稍窄 |
| 风险分布 | 对称 | 不对称 |
| 适用场景 | 标准市场环境 | 高波动或偏斜市场 |

## 风险管理

- 即使是不对称结构，每侧最大亏损仍然是固定的
- 需要配合严格的仓位控制（单笔风险 ≤ 账户 30–50%）
- 临近 short strike 时需及时减仓或调整

## 相关概念

- [[IronCondor]] — 基础策略
- [[Gamma风险]] — 临近行权价时的风险
- [[Vega风险]] — IV 变化的风险
