---
title: "SPX Iron Condor 方法论与实盘 Checklist"
type: source
tags: [SPX, iron-condor, trading-system, checklist]
date: 2026-03-19
source_file: raw/spx-iron-condor-ihuhao.md
---

## Summary
整理了一套可长期复用的 SPX Iron Condor 方法论与实盘 checklist，涵盖环境判断、结构设计、收益过滤、仓位控制和退出机制五大步骤，以及风险管理认知和实盘操作清单。

## Key Claims
- Iron Condor 本质：**卖波动（IV） + 吃时间（Theta） + 赌不极端行情**
- 入场环境要求：IV Percentile > 40（最好 > 50）、VIX > 20（理想 22–28）、无超大事件
- Short strike 标准：Δ = 0.10–0.15，POP ≈ 70–80%
- Spread 宽度建议：保守 100点，激进 50点
- Credit 入场标准：≥ width 的 20–30%
- 仓位控制：单笔最大风险 ≤ 账户 30–50%，建议 4–6 组（$50k 账户）
- 止盈规则：收益达到 50–70% 时平仓
- 止损规则：loss = 2x credit 时出场
- Put side 应比 Call side 更远，形成**不对称 condor（skew condor）**

## Key Quotes
> "高IV时，卖远端、做宽、轻仓；赚时间的钱，不赌方向；一旦错，快速认错。"

> "胜率高 = 安全" 是最大误区——小赚很多次，一次爆亏全吐回。

> 永远记住：下跌快而猛，上涨慢而磨。所以 put side 一定要更保守。

## 新增概念
- **[[IVRank]]** — IV Rank 与 IV Percentile 用于判断当前隐含波动率环境
- **[[SkewCondor]]** — 不对称 condor，put side 更远，call side 稍近
- **[[Gamma风险]]** — 临近行权价时 PnL 加速恶化的风险
- **[[Vega风险]]** — IV 上升导致双杀（价格 + IV）的风险

## 与现有来源的关系
- 与 [[IronCondor]] 策略框架一致，但提供了更具体的量化参数
- DTE 范围（45–70）与 [[iron-condor-tastylive]] 建议的 30–60 天略有差异
- 止盈规则（50–70%）比 [[iron-condor-tastylive]] 的 50% 更灵活

## Contradictions
- DTE 范围：本文建议 45–70 天，其他来源（[[iron-condor-tastylive]]、[[iron-condor-optionstrategiesinsider]]）建议 30–60 天
- 止盈目标：本文建议 50–70%，[[iron-condor-tastylive]] 建议 50%
