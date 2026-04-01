# Iron Condor 风险分析经验总结

## 常见错误

### 1. Greeks 符号搞混
- **Long Call**: Delta+, Gamma+, Theta-, Vega+
- **Long Put**: Delta-, Gamma+, Theta-, Vega+
- **Short Call**: Delta-, Gamma-, Theta+, Vega-
- **Short Put**: Delta+, Gamma-, Theta+, Vega-
- **铁鹰组合**: 净 Gamma 通常为负（Short Gamma），净 Theta 为正

### 2. ITM/OTM 判断错误
- 不要把"接近"误认为"深度价内"
- 确认 strike 与标的价格的相对位置再下结论

### 3. 忽视 Theta 实际收益
- 组合净 Theta 为正不代表每天都赚钱
- 要计算 Theta 收益占净权利金的比例，避免高估持有价值

### 4. 移仓不是万能解药
- 每次移仓都有成本，反复移仓会侵蚀利润
- 如果趋势方向明确，考虑直接平仓而非不断追涨

## 分析要点

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
