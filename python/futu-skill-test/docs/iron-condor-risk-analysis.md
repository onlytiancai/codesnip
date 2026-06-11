# Iron Condor 风险分析经验总结

## 常见错误

### 1. Greeks 符号搞混
- **Long Call**: Delta+, Gamma+, Theta-, Vega+
- **Long Put**: Delta-, Gamma+, Theta-, Vega+
- **Short Call**: Delta-, Gamma-, Theta+, Vega-
- **Short Put**: Delta+, Gamma-, Theta+, Vega-
- **铁鹰组合**: 净 Gamma 通常为负（Short Gamma），净 Theta 为正

### 2. ITM/OTM 判断错误
- **Put**: strike > 标价 → OTM（不会行权，short赚钱）；strike < 标价 → ITM
- **Call**: strike < 标价 → ITM（可能行权，short亏损）；strike > 标价 → OTM
- **Short Put**: 标的价格 > strike = OTM = 赚钱状态
- **Short Call**: 标的价格 > strike = ITM = 亏损状态
- 不要把"接近"误认为"深度价内"，确认相对位置再下结论

### 3. 忽视 Theta 实际收益和合约乘数
- 组合净 Theta 为正不代表每天都赚钱
- Greeks 数据（Theta/Vega/Gamma）是**每股**数据，需乘以合约乘数（SPX期权 = 100）才是实际盈亏
- 要计算 Theta 收益占净权利金的比例，避免高估持有价值
- 例：Theta $1.58/天 × 100 = $158/天，16天 = $2,528

### 4. 移仓不是万能解药
- 每次移仓都有成本，反复移仓会侵蚀利润
- 如果趋势方向明确，考虑直接平仓而非不断追涨

## 分析要点

### 先确认基础事实
1. 标的价格 vs strike 的相对位置（ITM/OTM）
2. 各腿的 Delta 正负
3. 再进行 Greeks 计算和风险分析

**顺序错误**：先算 Greeks 后判断 ITM/OTM，导致方向判断错误

### 风险等级排序（从高到低）
1. **Call 边被逼近**: 标的价格距 short call strike < 3% 且趋势向上 = 最大威胁
2. **负 Gamma**: 标的大幅波动对组合不利
3. **VIX 位置**: >25 高波动环境，short vega 承压
4. **Theta**: 需乘以合约乘数计算实际收益，距到期 <15天时仍可能有意义

### 核心指标判断
- **Delta ≈ 0**: 方向风险低
- **负 Gamma**: 剧烈波动对你不利（最大威胁）
- **正 Theta**: 时间流逝对你有利，但需计算绝对收益
- **负 Vega**: VIX 上涨时承压

### 决策优先级
1. 先判断 Theta 收益是否值得继续持有
2. 再评估 Gamma 风险（wing 是否安全）
3. 最后考虑 Vega 暴露（当前 VIX 位置）

### 移仓时机
- Wing 被逼近（距标的价格 <3%）时优先考虑平仓或移仓
- 距到期 >30天可以给 Theta 更多时间补偿风险
- 距到期 <15天 Theta 收益有限，优先考虑风控
