---
title: "Gamma 风险"
type: concept
tags: [options, greeks, gamma, risk]
sources: [spx-iron-condor-ihuhao]
last_updated: 2026-04-14
---

## 概述

**Gamma（Γ）** 衡量期权 Delta 相对于标的资产价格变化的变动率。对于 Iron Condor 持仓而言，Gamma 风险是指临近到期或价格接近行权价时，PnL 可能加速恶化的风险。

## Gamma 风险的特点

根据 [[spx-iron-condor-ihuhao]]：

> "临近 strike → PnL 加速恶化"

- **高 Gamma 区域**：当标的资产价格接近空头行权价（short strike）时，该侧的 Gamma 最高
- **短期权 Gamma 为负**：卖出期权（收取权利金）意味着承受负 Gamma，价格向不利方向移动时损失加速
- **时间衰减非线性和**：临近到期时，即使标的价格小幅波动，期权价格也可能大幅变动

## Iron Condor 中的 Gamma 风险来源

1. **Short Strike 附近**：价格触及 short strike 时，该侧空头期权开始快速亏损
2. **到期日临近**：DTE < 14 天时，Gamma 效应急剧放大
3. **价差宽度收窄**：如果价格向 short strike 移动，未测试侧的 value 在减少

## 风险管理措施

- **提前退出**：当价格接近 short strike 时，考虑减仓或调整
- **避免临近到期建仓**：建议 45–70 DTE 入场，避开高 Gamma 期
- **设置硬性止损**：当 loss = 2x credit 时无条件出场

## Gamma vs 其他 Greeks

| Greek | 衡量 | Iron Condor 角色 |
|---|---|---|
| Delta (Δ) | 价格敏感度 | 确定 short strike 位置 |
| Gamma (Γ) | Delta 变化率 | 临近行权价时加速亏损 |
| Theta (Θ) | 时间衰减 | 主要收益来源 |
| Vega (ν) | IV 变化 | IV 上升时双杀 |

## 相关概念

- [[Vega风险]] — IV 上升带来的双重风险
- [[IronCondor]] — 承受 Gamma 风险的主要策略
- [[SkewCondor]] — 通过不对称结构降低 Gamma 风险
