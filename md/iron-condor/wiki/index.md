# Wiki 索引

本文件由 LLM 维护，每次 ingestion 时更新。

## 概述
- [Overview](overview.md) — 跨所有来源的持续综合

## 来源
- [Iron Condor 策略指南](sources/iron-condor-optionstrategiesinsider.md) — 中性期权策略，涵盖结构、盈亏平衡点、P/L 和交易实例
- [Iron Condor：全面指南](sources/iron-condor-tastylive.md) — 全面指南，包含入场策略、$500 详细实例、管理规则、反向铁秃鹰
- [SPX Iron Condor 方法论与实盘 Checklist](sources/spx-iron-condor-ihuhao.md) — 系统化方法论，含五大步骤、量化参数和风险管理原则
- [Doubao SPX Iron Condor 实战对话](sources/doubao-ad4d476d06810.md) — Greeks 量化逻辑、Delta/Gamma 对冲详解、专业基金调仓策略

## 实体

## 概念
- [IronCondor（铁秃鹰）](concepts/IronCondor.md) — 方向中性、风险有限的期权策略，由熊市看涨价差和牛市看跌价差同时构成
- [IronButterfly（铁蝴蝶）](concepts/IronButterfly.md) — Iron Condor 的 ATM 变体，权利金更高但盈利区间更窄
- [IronFly（铁飞）](concepts/IronFly.md) — Iron Condor 两侧空头行权价收敛至相同水平；结构与铁蝴蝶等价
- [ReverseIronCondor（反向铁秃鹰）](concepts/ReverseIronCondor.md) — 以借方入场的反向策略，适用于趋势行情
- [BearCallSpread（熊市看涨价差）](concepts/BearCallSpread.md) — 看跌信用价差；Iron Condor 的看涨侧
- [BullPutSpread（牛市看跌价差）](concepts/BullPutSpread.md) — 看涨信用价差；Iron Condor 的看跌侧
- [Straddle（跨式组合）](concepts/Straddle.md) — ATM 中性策略，多头风险无限，空头风险亦无限
- [Strangle（宽跨式组合）](concepts/Strangle.md) — OTM 中性策略；Iron Condor 是其风险限定版本
- [IVRank](concepts/IVRank.md) — IV Rank 与 IV Percentile，用于判断当前隐含波动率环境
- [SkewCondor（不对称铁秃鹰）](concepts/SkewCondor.md) — 利用市场不对称性的 Iron Condor 变体
- [Gamma风险](concepts/Gamma风险.md) — 临近行权价时 PnL 加速恶化的风险
- [Vega风险](concepts/Vega风险.md) — IV 上升导致双杀的风险
- [Delta对冲](concepts/Delta对冲.md) — SPY 与 SPX Call 对冲对比、Delta 中性动态维护
- [GammaCash](concepts/GammaCash.md) — 动态 Delta 对冲产生的现金流概念

## 综合分析
