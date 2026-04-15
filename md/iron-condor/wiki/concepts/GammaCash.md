---
title: "Gamma Cash"
type: concept
tags: [options, greeks, gamma, hedging]
sources: [doubao-ad4d476d06810]
last_updated: 2026-04-15
---

## 概述

**Gamma Cash（Gamma 现金流）** 是动态 Delta 对冲过程中产生的现金流，本质是利用标的价格波动中高抛低吸获得的收益（或损失）。常见于做市商和高频交易策略。

## Gamma Cash 的产生机制

持有期权组合时，Gamma 决定 Delta 的变化速度。为维持 Delta 中性（控制标的价格波动风险），需要动态调整标的资产持仓：

- **标的上涨**：Delta 变大 → 卖出标的，获得现金
- **标的下跌**：Delta 变小 → 买入标的，支出现金

## 计算公式（简化版）

Gamma Cash 变动 ≈ (1/2) × Γ × (ΔS)²

其中：
- Γ = 组合 Gamma
- ΔS = 标的价格波动幅度

## 实例说明

假设组合 Gamma = 0.1（SPX 涨 1 点，Delta 增加 0.1）：

1. **标的先涨 5 点**：Delta 增加 0.1×5=0.5 → 卖出 0.5 份标的，获得现金
2. **标的再跌 5 点**：Delta 减少 0.1×5=0.5 → 买入 0.5 份标的，支出现金
3. **净收益**：高抛低吸的价差收益

## 与 Iron Condor 的关系

对于卖出铁鹰/宽跨式策略：
- 组合 Gamma 为**负**（卖出期权主导）
- 标的波动越大，Gamma Cash 的**亏损**可能越严重
- 这也是为什么 Gamma 风险需要及时对冲

## 相关概念

- [[Gamma风险]] — Gamma 导致的风险放大
- [[Delta对冲]] — Gamma 影响 Delta 后的对冲操作
- [[IronCondor]] — 承受负 Gamma 的典型策略
