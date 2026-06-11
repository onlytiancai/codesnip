---
title: "Delta 对冲"
type: concept
tags: [options, greeks, delta, hedging]
sources: [doubao-ad4d476d06810]
last_updated: 2026-04-15
---

## 概述

**Delta 对冲**是通过调整标的资产或期权头寸，使组合 Delta 接近零，从而消除标的价格波动对组合价值的影响。常见工具包括 SPY 正股、SPX 期货、或期权。

## Delta 对冲的计算逻辑

### 核心公式

对冲数量 = （目标 Delta - 当前组合 Delta） / 对冲工具 Delta

### SPY 对冲 SPX Iron Condor

SPX 期权的合约乘数为 **100 美元/点**，因此：

- 组合 Delta = -4 表示 SPX 涨 1 点，组合亏损 4×100 = 400 美元
- 1 股 SPY 股价 ≈ SPX÷10，SPX 涨 1 点 SPY 涨 0.1 美元
- 所需 SPY 股数 = 400 ÷ 0.1 = **40 股**（而非 4 股）

> 关键点：Delta 数值已包含 SPX 期权 100 倍乘数，SPY 无乘数，两者单位不同

## SPY 对冲 vs SPX Call 对冲

| 维度 | SPY 对冲 | SPX Call 对冲 |
|------|----------|---------------|
| Gamma | 0（标的资产，Gamma=0） | >0（期权有 Gamma） |
| 时间价值损耗 | 无 | 有（权利金衰减） |
| 调整频率 | 低 | 高（需跟踪 Delta 变化） |
| 极端行情表现 | 恒定对冲效果 | 可能失效（Call 变实值后 Delta 趋近 1） |
| 初期成本 | 高（需买入大量正股） | 低（仅需权利金） |

## Delta 中性的动态维护

即使初始实现 Delta 中性，后续仍需跟踪：

- **SPX 涨跌 50-100 点后重新计算** Delta
- **Gamma > 0 时**：标的上涨会导致组合 Delta 变大（更正值），标的下跌会导致更负
- **SPY 对冲优势**：Delta 恒定，调整逻辑简单直接

## 与 Iron Condor 的关系

Iron Condor 卖出期权承受负 Gamma，当标的价格向某一侧移动时：
- 该侧空头期权变实值，Delta 快速负向增加
- 需要通过对冲将 Delta 拉回中性

专业基金在触发阈值（Delta≤-20 或 ≥20）时会启动对冲流程。

## 相关概念

- [[Gamma风险]] — Delta 变化的加速度
- [[IronCondor]] — 承受 Delta 敞口的策略
- [[GammaCash]] — 动态 Delta 对冲产生的现金流
