---
title: "Vega 风险"
type: concept
tags: [options, greeks, vega, risk]
sources: [spx-iron-condor-ihuhao]
last_updated: 2026-04-14
---

## 概述

**Vega（ν）** 衡量期权价格对隐含波动率（IV）变化的敏感度。对于 Iron Condor 卖出期权持仓而言，Vega 风险是指 IV 上升时，期权组合可能同时遭受价格损失和 IV 膨胀带来的双重打击。

## Vega 风险的特点

根据 [[spx-iron-condor-ihuhao]]：

> "IV 上升 → 双杀（价格 + IV）"

- **空头 Vega 为负**：卖出期权意味着当 IV 上升时，期权价格会上涨（不利于空头持仓）
- **双杀效应**：IV 上升不仅增加持仓成本，还会抵消已有的账面盈利
- **波动率聚集**：市场波动往往呈聚集状态，IV 上升后往往持续较高

## Iron Condor 中的 Vega 风险来源

1. **系统性风险事件**：FOMC、CPI、地缘政治等导致 IV 全面上升
2. **行业特定事件**：单一股票或板块的新闻导致个别 IV 上升
3. **到期前 IV Spike**：临近到期时 IV 异常上升

## 风险管理措施

- **IV 环境筛选**：仅在 IV Percentile > 40 时建仓
- **VIX 门槛**：VIX > 20 时建仓（理想 22–28）
- **避免重大事件**：FOMC、CPI 前不宜建仓
- **及时调整**：IV 异常上升时考虑减仓

## Vega 风险 vs Gamma 风险

| 风险类型 | 触发条件 | 影响 |
|---|---|---|
| Gamma 风险 | 价格接近 short strike | PnL 加速恶化 |
| Vega 风险 | IV 上升 | 期权价格全面上涨 |

## 与 IV Rank 的关系

- **高 IV Rank（> 50）**：IV 已经处于高位，Vega 风险相对较低（IV 上升空间有限）
- **低 IV Rank（< 30）**：IV 处于低位，Vega 风险较高（IV 可能急剧上升）

## 相关概念

- [[Gamma风险]] — 价格接近行权价时的风险
- [[IVRank]] — 判断 IV 水平的指标
- [[IronCondor]] — 承受 Vega 风险的主要策略
