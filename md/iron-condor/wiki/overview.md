---
title: "概述"
type: synthesis
tags: []
sources: [iron-condor-optionstrategiesinsider, iron-condor-tastylive, spx-iron-condor-ihuhao]
last_updated: 2026-04-14
---

# 概述

*本页面由 LLM 维护，每次 ingestion 时更新，以反映所有来源的综合内容。*

## 核心主题

本 wiki 目前涵盖的主题领域：**美股期权策略**，具体包括中性及方向性信用/借方价差策略，适用于区间震荡或波动行情中的盈利于机构及散户交易者。

## 综合摘要

**Iron Condor（铁秃鹰）** 是一种方向中性、风险有限的期权策略，由四腿构成——同时卖出 OTM 熊市看涨价差（Bear Call Spread）和 OTM 牛市看跌价差（Bull Put Spread）。当标的股票在到期前维持在看涨价差与看跌价差之间时盈利。主要优势包括：风险有上限、行权价设置足够价外时有高胜率、建仓时收取净权利金。最佳使用环境为高隐含波动率（IV）、距到期日 30–60 天。

**入场策略** 各来源略有差异：optionstrategiesinsider 建议 30–60 天到期；tastylive 以收取价差宽度 1/3 的权利金为目标，成功率约 67%。两者均认同**50% 最大盈利平仓规则**，以提升长期胜率。

**管理方式** 包括将未测试（盈利）侧向股价方向移仓以收取额外权利金。当移仓至与测试侧相同行权价时，Iron Condor 即演变为 **[[IronFly]]**（结构上等同于 [[IronButterfly]]）。

**Reverse Iron Condor（反向铁秃鹰）** 则是其反向——以借方入场，在价格出现大幅单边波动时盈利。

相关策略：[[IronButterfly]]（ATM 行权价，权利金更高，盈亏平衡点更差）、[[Straddle]]（无限风险）、[[Strangle]]（无保护翼的风险无限策略）。

## 来源

- [[iron-condor-optionstrategiesinsider]] — 结构、盈亏平衡公式、XYZ $100 实例、30–60 DTE 入场窗口
- [[iron-condor-tastylive]] — 全面指南、1/3 宽度入场规则、$500/$50 实例、50% 盈利目标、移仓管理、反向铁秃鹰、铁飞
- [[spx-iron-condor-ihuhao]] — 方法论、IV Rank/VIX 筛选标准、Skew Condor、Gamma/Vega 风险管理、实盘 Checklist

## 风险管理与系统化

除基本策略框架外，新增 [[spx-iron-condor-ihuhao]] 提供了更量化的入场和风险管理标准：

- **IV 环境筛选**：IV Percentile > 40（最好 > 50），VIX > 20（理想 22–28）
- **Short strike 量化**：Δ = 0.10–0.15，POP ≈ 70–80%
- **仓位控制**：单笔最大风险 ≤ 账户 30–50%，建议 4–6 组（$50k 账户）
- **入场标准**：credit ≥ width 的 20–30%
- **退出规则**：止盈 50–70%，止损 2x credit

## Greeks 风险管理

Iron Condor 主要暴露于以下风险：

- **Gamma 风险** — 临近 short strike 时 PnL 加速恶化，需要设置硬性止损
- **Vega 风险** — IV 上升导致双杀，应在高 IV 环境建仓并避免重大事件窗口
