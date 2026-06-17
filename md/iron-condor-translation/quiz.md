# 铁鹰式期权学习测试 / Iron Condor Quiz

> 基于 `iron-condor-bilingual.md` 的章节练习。每题 1 分（多选 2 分），提交后查看每章节 + 总体正确率，错题有详细解析。

<!-- section: intro -->

## 引言 / Introduction

铁鹰（Iron Condor）由 4 个期权腿组成，本质上是牛市认沽价差 + 熊市认购价差的组合，在标的横盘时获取时间价值收益。

::: quiz q-intro-1 single
id: q-intro-1
section: intro
difficulty: easy
tags: definition, structure
question: |
  铁鹰（Iron Condor）本质上是哪两种价差的组合？
options:
  - { key: A, text: "牛市认购价差 + 熊市认沽价差" }
  - { key: B, text: "牛市认沽价差 + 熊市认购价差" }
  - { key: C, text: "跨式（Straddle）+ 宽跨式（Strangle）" }
  - { key: D, text: "蝶式（Butterfly）+ 日历价差（Calendar）" }
answer: B
explanation: |
  原文："Iron Condor is actually a combination of a **Bull Put Spread** and a **Bear Call Spread**"，
  对应中文"**牛市认沽价差 + 熊市认购价差**"。这是铁鹰最核心的定义。
:::

::: quiz q-intro-2 single
id: q-intro-2
section: intro
difficulty: easy
tags: abc-example, profit
question: |
  在 ABC 股票的铁鹰例子中（4 张合约 10 月到期、$50 行权价），若 ABC 收盘价在到期日落在 4 个行权价之间，账户最大总收益是多少？
options:
  - { key: A, text: "$100" }
  - { key: B, text: "$200" }
  - { key: C, text: "$300" }
  - { key: D, text: "$500" }
answer: B
explanation: |
  ABC 例子：$50/$45 Bull Put Spread 收 $100 + $55/$60 Bear Call Spread 收 $100 = **最大总收益 $200**。
  两个价差各贡献 $100 净权利金。
:::

::: quiz q-intro-3 multiple
id: q-intro-3
section: intro
difficulty: medium
tags: scenarios, management
question: |
  作者指出"管理得当"的话，铁鹰在 5 种股价走势下都可能有收益。下列哪些走势属于这 5 种？（多选）
options:
  - { key: A, text: "Up a lot — 大涨" }
  - { key: B, text: "Up a little — 小涨" }
  - { key: C, text: "Essentially flat — 几乎横盘" }
  - { key: D, text: "Down a lot — 大跌" }
answer: A, B, C, D
explanation: |
  原文列出 5 种：Up a lot / Up a little / Essentially flat / Down a little / Down a lot。
  "管理得当"意味着通过滚动、调整等手段让原本"灾难"的方向也能止损可控。**大跌** 通过滚动 Put 端往下移、买入更深 Put 锁定等方法也可处理。
:::

<!-- section: bear-call -->

## 熊市认购价差 / Bear Call Spread

卖出低行权价 Call + 买入更高行权价 Call，构成中性偏空策略。权利金为净收入。

::: quiz q-bc-1 single
id: q-bc-1
section: bear-call
difficulty: easy
tags: structure
question: |
  熊市认购价差（Bear Call Spread）的结构是？
options:
  - { key: A, text: "买低 Call + 卖高 Call" }
  - { key: B, text: "卖低 Call + 买高 Call" }
  - { key: C, text: "买 Put + 卖 Put" }
  - { key: D, text: "买 Call + 买 Put" }
answer: B
explanation: |
  卖出行权价较低的 Call（收权利金）+ 买入行权价较高的 Call（限制风险） = **Bear Call Spread**。
  与"牛市认沽价差"（买低 Put + 卖高 Put）镜像对称。
:::

::: quiz q-bc-2 single
id: q-bc-2
section: bear-call
difficulty: medium
tags: profit, abc-example
question: |
  ABC 熊市认购价差例子：10 月到期的 $50 Call 卖价 $2，$55 Call 买价 $1，每张合约净收到的权利金是？
options:
  - { key: A, text: "$50" }
  - { key: B, text: "$100" }
  - { key: C, text: "$200" }
  - { key: D, text: "$300" }
answer: B
explanation: |
  净权利金 = ($2 − $1) × 100 股 = **$100**。最大风险 = ($55 − $50 − $1) × 100 = $400。
  与下方的 Bull Put Spread 共同构成铁鹰的 Call 端。
:::

<!-- section: combine -->

## 组合：完整的铁鹰 / Putting It All Together

两笔价差合并后，最大风险资本略少、收益翻倍。

::: quiz q-combine-1 single
id: q-combine-1
section: combine
difficulty: easy
tags: payoff
question: |
  ABC 例子合成完整铁鹰后，最大亏损为多少？
options:
  - { key: A, text: "$200" }
  - { key: B, text: "$300" }
  - { key: C, text: "$400" }
  - { key: D, text: "$700" }
answer: B
explanation: |
  表格原文：最大收益 **$200** / 最大亏损 **$300** / 风险资本 $500 / 回报率 66.67%。
  风险资本 $500 − 最大收益 $200 = **最大亏损 $300**。
:::

::: quiz q-combine-2 single
id: q-combine-2
section: combine
difficulty: easy
tags: return-rate
question: |
  ABC 铁鹰例子的潜在回报率（最大收益 / 最大亏损）约为多少？
options:
  - { key: A, text: "20%" }
  - { key: B, text: "40%" }
  - { key: C, text: "66.67%" }
  - { key: D, text: "100%" }
answer: C
explanation: |
  回报率 = 最大收益 / 最大亏损 = $200 / $300 ≈ **66.67%**。
  这个数字原文明确给出，是铁鹰吸引力的关键指标。
:::

::: quiz q-combine-3 multiple
id: q-combine-3
section: combine
difficulty: medium
tags: advantage
question: |
  关于铁鹰相比单独 Bull Put 或 Bear Call 的优势，下列哪些说法是正确的？（多选）
options:
  - { key: A, text: "在同等风险下，收益近乎翻倍" }
  - { key: B, text: "实际占用风险资本反而略少" }
  - { key: C, text: "Gamma 风险完全消除" }
  - { key: D, text: "可以从两个方向获利" }
answer: A, B, D
explanation: |
  原文：'essentially receive **double the income** for the same amount of risk'，且 'capital at risk is slightly less'，上下行两侧均可获利。
  C 错：铁鹰仍有显著 Gamma 风险（详见 Gamma 章节），只是 long gamma 不存在而已。
:::

<!-- section: timing -->

## 何时建仓铁鹰 / When to Enter Iron Condors

建仓时机的 4 种场景：区间扩张之后、波动率飙升时、IV Rank 较高、随时建仓保持一致。

<!-- ### 区间扩张之后 / After a Range Expansion -->

::: quiz q-timing-1 single
id: q-timing-1
section: timing
difficulty: medium
tags: range-expansion
question: |
  区间扩张（Range Expansion）后建仓铁鹰的核心逻辑是？
options:
  - { key: A, text: "标的刚大涨 / 大跌，预期回到震荡" }
  - { key: B, text: "标的刚创新低，应该抄底" }
  - { key: C, text: "波动率在历史最低点" }
  - { key: D, text: "VIX 在 10 以下" }
answer: A
explanation: |
  区间扩张后标的常进入"盘整"阶段，配合较高 IV，建仓铁鹰收割时间价值。
  原文以 RUT 2016-11 大选后 8 个月横盘为例。
:::

<!-- ### 波动率飙升时 / On a Volatility Spike -->

::: quiz q-timing-2 single
id: q-timing-2
section: timing
difficulty: medium
tags: vol-spike, vega
question: |
  遇到市场大跌、VIX 飙升时，作者建议的最佳做法是？
options:
  - { key: A, text: "立刻重仓做空波动率" }
  - { key: B, text: "等上一两周再行动，避开 V 形反转" }
  - { key: C, text: "立刻买入跨式做多波动率" }
  - { key: D, text: "完全不做，等一年" }
answer: B
explanation: |
  原文：'A good idea is to **wait a week or two** after those big vol spikes, you don't necessarily want to get the absolute top in volatility.'
  波动率剧烈飙升后常快速回落（VIX 期货升水结构逆转），过早建仓容易买在 IV 高点。
:::

<!-- ### 根据 IV Rank / Based on IV Rank -->

::: quiz q-timing-3 single
id: q-timing-3
section: timing
difficulty: easy
tags: iv-rank
question: |
  根据原文，文中推荐的免费 IV Rank 扫描器来自哪个网站？
options:
  - { key: A, text: "optionstradingiq.com" }
  - { key: B, text: "marketchameleon.com" }
  - { key: C, text: "ivolatility.com" }
  - { key: D, text: "CBOE.com" }
answer: B
explanation: |
  原文（L216）：'A good free scanner for IV Rank is available from
  **www.marketchameleon.com**.'
  这是一家提供免费 IV Rank 数据的扫描器网站，还提供低 IV Rank 股票筛选。
:::

<!-- ### 随时建仓 / Any Time -->

::: quiz q-timing-4 multiple
id: q-timing-4
section: timing
difficulty: medium
tags: consistency, sizing
question: |
  关于"随时建仓保持一致性"的原则，下列哪些做法被作者推荐？（多选）
options:
  - { key: A, text: "分批建仓（25% 一批）" }
  - { key: B, text: "完全不在 IV 极低时建仓" }
  - { key: C, text: "在不同行权价、到期日轮动" }
  - { key: D, text: "机械执行交易计划" }
answer: A, C, D
explanation: |
  原文强调 'be consistent'：每批 25%、不同 strike/expiry 轮换、机械执行计划。
  B 错：作者恰恰在 IV 极低时也建仓（如 25%），不放弃任一时点。
:::

<!-- section: timeframe -->

## 长期还是短期铁鹰 / Long-term or Short-term Iron Condors

作者偏好 90 日长期铁鹰，因其在 2018-03 暴跌中表现远优于月度/周度。

::: quiz q-tf-1 single
id: q-tf-1
section: timeframe
difficulty: easy
tags: preference
question: |
  作者本人在文章中表示更偏好哪种到期时间的铁鹰？
options:
  - { key: A, text: "周度（Weekly）" }
  - { key: B, text: "月度（Monthly）" }
  - { key: C, text: "长期（Long-term）" }
  - { key: D, text: "LEAPS（>1 年）" }
answer: C
explanation: |
  原文：'**I much prefer long term iron condors to short term condors**'
  （我**更偏好长期铁鹰，而非短期**）。转向长期最初是因搬回墨尔本的时差问题，
  长期铁鹰走得慢、对调整及时性要求低。
  文章用 2018-03 案例对比了 90 日、月度、周度 3 种铁鹰，**90 day condor performed by far the best**。
:::

::: quiz q-tf-2 single
id: q-tf-2
section: timeframe
difficulty: medium
tags: case-study
question: |
  在 2018-03 月美股闪崩的案例中（2018-03-20 建仓 RUT 铁鹰），90 日铁鹰相比月度/周度铁鹰的核心优势是？
options:
  - { key: A, text: "收益更高" }
  - { key: B, text: "有更多时间调整 / 避免 Gamma 周冲击" }
  - { key: C, text: "行权价更宽" }
  - { key: D, text: "占用资金更少" }
answer: B
explanation: |
  长期铁鹰在 30 天后才有最大 Theta 加速、Gamma 风险主要落在最后一周。
  暴跌发生时距到期还远，可滚动下移或加倍对冲；月度/周度直接面临 Gamma 灾难。
:::

<!-- section: contango -->

## 理解升水与贴水 / Understanding Contango and Backwardation

VIX 期货的升水（Contango）和贴水（Backwardation）反映市场对未来波动的预期。

::: quiz q-cont-1 single
id: q-cont-1
section: contango
difficulty: easy
tags: definition
question: |
  升水（Contango）状态下 VIX 期货曲线的形态是？
options:
  - { key: A, text: "近月高于远月" }
  - { key: B, text: "近月低于远月" }
  - { key: C, text: "完全水平" }
  - { key: D, text: "随机波动" }
answer: B
explanation: |
  Contango = **近低远高**，市场平静、预期未来波动温和。Backwardation 才是近高远低（市场恐慌）。
  Contango 是常态，做空 VIX 的策略（如铁鹰）受益于期货回归现货。
:::

::: quiz q-cont-2 single
id: q-cont-2
section: contango
difficulty: easy
tags: definition
question: |
  贴水（Backwardation）表示？
options:
  - { key: A, text: "市场对未来充满信心" }
  - { key: B, text: "市场恐慌、近期波动率高于远期" }
  - { key: C, text: "VIX 期货价格平稳" }
  - { key: D, text: "适合立即建仓铁鹰" }
answer: B
explanation: |
  Backwardation = 近月 > 远月，反映市场短期恐慌。
  贴水状态通常持续较短，是建仓铁鹰的好窗口（IV 高 + 市场回归平静）。
:::

::: quiz q-cont-3 single
id: q-cont-3
section: contango
difficulty: hard
tags: volpocalypse
question: |
  2018-02 "Volpocalypse" 事件中，VIX 单日涨幅约为？
options:
  - { key: A, text: "20%" }
  - { key: B, text: "50%" }
  - { key: C, text: "115.6%" }
  - { key: D, text: "300%" }
answer: C
explanation: |
  原文（L375）：'when VIX spiked an almighty **115.6%** from **17.31 to 37.32**'.
  这是教科书级"短 Gamma 爆仓"案例：反向 VIX ETN（XIV）当日清零。
  在此之前（不计 1987 崩盘），最大单日涨幅是 2007 年 2 月的 **64.20%**。
:::

<!-- section: legging -->

## 分腿建仓 / Legging In to an Iron Condor

分腿建仓可提高收益但单边裸奔风险显著。

::: quiz q-leg-1 single
id: q-leg-1
section: legging
difficulty: easy
tags: definition
question: |
  "分腿建仓"（Legging In）指的是？
options:
  - { key: A, text: "4 个腿全部一次性下单" }
  - { key: B, text: "分多次下单，先建一侧等股价配合再建另一侧" }
  - { key: C, text: "用限价单分批平仓" }
  - { key: D, text: "用市价单快速建仓" }
answer: B
explanation: |
  分腿建仓 = 拆开 Put/Call 两边分别建仓，追求更优价格但承担单边风险。
  进阶技巧：只对老练交易者。
:::

::: quiz q-leg-2 multiple
id: q-leg-2
section: legging
difficulty: medium
tags: risk
question: |
  关于分腿建仓的风险，下列哪些说法正确？（多选）
options:
  - { key: A, text: "单边裸奔（naked）风险显著增加" }
  - { key: B, text: "另一侧对冲完全失效，承压方独自扛" }
  - { key: C, text: "完全没有额外风险" }
  - { key: D, text: "可能因行情剧烈无法成交另一侧" }
answer: A, B, D
explanation: |
  原文（L425-429）：'If a trader is bullish they might start by selling a bull put spread.
  Then, if the market declines, that spread is placed under pressure with **no offsetting gains**
  from the declining price of the bear call spread.'
  也就是说当一侧承压时，**另一侧并没有提供对冲收益**（"单边裸奔"）。
  反之亦然，上涨时 Put 端独自受益 / 风险。行情剧烈时另一侧可能无法成交。
  C 错：分腿必然带来额外风险。
:::

<!-- section: strikes -->

## 选择行权价 / Selecting Iron Condor Strikes

3 种选价方法：Delta 法、标准差法、技术分析。

::: quiz q-strike-1 single
id: q-strike-1
section: strikes
difficulty: easy
tags: delta-method
question: |
  作者推荐用 Delta 法时，短腿（卖出腿）的 Delta 约为多少？
options:
  - { key: A, text: "5 Delta" }
  - { key: B, text: "10-15 Delta" }
  - { key: C, text: "30 Delta" }
  - { key: D, text: "50 Delta" }
answer: B
explanation: |
  原文：'I like to sell options that are around the **10-15 Delta** area'.
  对应大约 1 个标准差外。Delta 越高，胜率越低但单笔收益越高；反之亦然。
:::

::: quiz q-strike-2 single
id: q-strike-2
section: strikes
difficulty: easy
tags: methods
question: |
  文中提到的 3 种选行权价方法是？
options:
  - { key: A, text: "Delta 法、标准差法、技术分析" }
  - { key: B, text: "P/E 法、PEG 法、DCF 法" }
  - { key: C, text: "线性回归、移动平均、布林带" }
  - { key: D, text: "随机、激进、保守" }
answer: A
explanation: |
  原文（L448-451）给出 3 种思路：
  1. **Use delta**（如卖 10/15 Delta）
  2. **Use standard deviation**（卖出现价 ±1~2 个标准差之外的行权价）
  3. **Use technical analysis**（参考支撑位、阻力位、趋势线）
  关键不是用哪种，而是选定后**保持一致**。
:::

<!-- section: delta-dollars -->

## Delta Dollars：最重要的指标 / Delta Dollars May Be the Single Most Important Metric

Delta Dollars = 持仓 Delta × 标的价格，反映方向性美元敞口。

::: quiz q-dd-1 single
id: q-dd-1
section: delta-dollars
difficulty: easy
tags: formula
question: |
  "Delta Dollars" 的计算公式是？
options:
  - { key: A, text: "Delta × 标的当前价格 × 100" }
  - { key: B, text: "Delta × 标的当前价格" }
  - { key: C, text: "Delta × 行权价" }
  - { key: D, text: "Gamma × 100" }
answer: B
explanation: |
  原文定义：'**Delta dollars** is quite simply the **position delta × the underlying price**'.
  注意：账户层级的 Delta Dollars 不乘 100（不是单张合约）。
:::

::: quiz q-dd-2 single
id: q-dd-2
section: delta-dollars
difficulty: medium
tags: threshold
question: |
  作者给自己定的 Delta Dollars 警戒线（占账户净值的百分比）约为？
options:
  - { key: A, text: "100%" }
  - { key: B, text: "150-200%" }
  - { key: C, text: "300%" }
  - { key: D, text: "无限制" }
answer: B
explanation: |
  原文：'I like to set a rule that I don't let my delta dollar exposure get above **150% of my account size**'，实战中 200% 是他比较舒服的位置。
  警戒线触发 → 减仓或 Delta 对冲。
:::

::: quiz q-dd-3 single
id: q-dd-3
section: delta-dollars
difficulty: hard
tags: case-study
question: |
  RUT 2018 实战案例中，账户约 $20,000，Delta Dollars 触及多少时触发调整？
options:
  - { key: A, text: "$20,000（100%）" }
  - { key: B, text: "$30,000（150%）" }
  - { key: C, text: "$40,000（200%）" }
  - { key: D, text: "$60,000（300%）" }
answer: C
explanation: |
  原文：'I had a delta dollar exposure of about **$40,000**' 触发调整。
  $40,000 / $20,000 = 200%，处于作者个人舒适区上限。
:::

<!-- section: delta-hedge -->

## Delta 对冲 / Delta Hedging

紧急止血技巧：买 1 份 Put 把净 Delta 拉回中性。

::: quiz q-dh-1 single
id: q-dh-1
section: delta-hedge
difficulty: easy
tags: method
question: |
  标的下跌时，作者推荐的快速 Delta 对冲方法是？
options:
  - { key: A, text: "再卖一份 Put" }
  - { key: B, text: "买入约 20 Delta 的 Put 中和" }
  - { key: C, text: "用期货做空" }
  - { key: D, text: "完全平仓" }
answer: B
explanation: |
  原文案例：净 Delta 跌至 −200 → 买 1 份 20 Delta 的 Put → 中和 −20 净 Delta。
  这是短期应急工具，不是长期仓位。
:::

::: quiz q-dh-2 single
id: q-dh-2
section: delta-hedge
difficulty: medium
tags: exit-strategy
question: |
  关于 Delta 对冲 Put 的退出，作者建议？
options:
  - { key: A, text: "持有到期" }
  - { key: B, text: "标的反弹后小亏或平价平仓" }
  - { key: C, text: "立即卖出获利" }
  - { key: D, text: "滚动到下一月份" }
answer: B
explanation: |
  原文：'When the market bounces, you'll likely be able to get out of that hedge for a small loss or even at breakeven.'
  对冲 Put 的成本是机会成本，方向反转后及时退出。
:::

<!-- section: idx-vs-etf -->

## 指数期权还是 ETF 期权？/ Should You Trade Index or ETF Options?

7 维度对比 SPX vs SPY：流动性、结算、佣金、税务、资金、分红、提前行权。

::: quiz q-ie-1 multiple
id: q-ie-1
section: idx-vs-etf
difficulty: medium
tags: liquidity, capital
question: |
  关于 SPX（指数期权）与 SPY（ETF 期权）的对比，下列哪些说法正确？（多选）
options:
  - { key: A, text: "SPY 的价格大约是 SPX 的 1/10" }
  - { key: B, text: "SPY 单张名义小，更适合小资金账户" }
  - { key: C, text: "两者在主流指数上流动性差异不大" }
  - { key: D, text: "新手或担心流动性时优先选 ETF" }
answer: A, B, C, D
explanation: |
  原文（L659-660）'The **SPY** ETF is approximately **1/10 the value** of the **SPX** Index.'（SPY 大约是 SPX 的 1/10）。
  原文（L610）'When comparing liquidity on the major indexes, there is not much difference
  between index options and ETF options as both are very, very liquid.'（主流指数上两者都极好，差异不大）。
  原文（L607）'Traders who are worried about liquidity, or are just starting out,
  should stick to the **ETFs** as there will be less slippage.'（担心流动性或新手优先 ETF）。
:::

::: quiz q-ie-2 single
id: q-ie-2
section: idx-vs-etf
difficulty: easy
tags: tax
question: |
  指数期权（SPX 等）的税务优势主要体现在？
options:
  - { key: A, text: "完全免税" }
  - { key: B, text: "适用 60% 长期 + 40% 短期资本利得混合税率（Section 1256）" }
  - { key: C, text: "20% 优惠税率" }
  - { key: D, text: "可抵减工资" }
answer: B
explanation: |
  美国境内 Section 1256 合约（指数期权）享受 **60% 长期 + 40% 短期** 混合税率，无论持仓多久。
  ETF 期权（SPY）按正常短期资本利得征税（高税率）。
:::

::: quiz q-ie-3 single
id: q-ie-3
section: idx-vs-etf
difficulty: easy
tags: early-assignment
question: |
  关于提前行权（Early Assignment）风险，下列哪一类期权通常**不**面临此风险？
options:
  - { key: A, text: "SPY 美式 ETF 期权" }
  - { key: B, text: "RUT 欧式指数期权" }
  - { key: C, text: "OEX 美式指数期权" }
  - { key: D, text: "QQQ 美式 ETF 期权" }
answer: B
explanation: |
  **欧式** 行权的指数期权（RUT、SPX、NDX、MXN）无法被提前行权。
  美式 ETF 期权（SPY/QQQ）和美式指数期权（OEX）有提前行权风险，临到期 ITM 短腿需警惕。
:::

<!-- section: small-account -->

## 小资金如何做铁鹰 / How to Trade Iron Condors with a Small Account

起步 $2,000-$10,000；心理上 $200k 与 $10k 不可同日而语。

::: quiz q-sa-1 single
id: q-sa-1
section: small-account
difficulty: easy
tags: capital
question: |
  根据原文，小资金铁鹰的最低起步资金约为？
options:
  - { key: A, text: "$500" }
  - { key: B, text: "$2,000" }
  - { key: C, text: "$5,000 ~ $10,000" }
  - { key: D, text: "$100,000+" }
answer: B
explanation: |
  原文（L716）：'You can even start trading with as little as **$2,000**.'
  （**$2,000** 就能开始）。
  同时原文（L712）也建议理想情况下"**$5,000 to $10,000**"起步。
  即便手头有 $200,000，也建议从 $10,000 起步慢慢建仓。
:::

::: quiz q-sa-2 multiple
id: q-sa-2
section: small-account
difficulty: medium
tags: rules
question: |
  关于小资金做铁鹰的核心原则，下列哪些说法正确？（多选）
options:
  - { key: A, text: "高 IV Rank 时才建仓" }
  - { key: B, text: "优先选择流动性好的 ETF 期权" }
  - { key: C, text: "远离周度期权（Gamma 风险大）" }
  - { key: D, text: "保持 30-40% 现金缓冲" }
answer: A, B, C, D
explanation: |
  原文 9 条要点涵盖以上所有：IV Rank>50 / 选 ETF / 30-45 DTE 而非周度 / 30-40% 现金缓冲 / 严格止损等。
  小账户更要严守纪律，避免一次回撤出局。
:::

<!-- section: flash-crash -->

## 如何在闪崩中存活 / How to Survive a Flash Crash

闪崩应对 4 招：止损单、VIX Call、更虚值 Put、Delta 对冲。

::: quiz q-fc-1 single
id: q-fc-1
section: flash-crash
difficulty: easy
tags: stop-loss
question: |
  作者在 4 种闪崩应对方法中，最推荐的第一道防线是？
options:
  - { key: A, text: "VIX Call" }
  - { key: B, text: "止损单（Stop Loss）" }
  - { key: C, text: "买入更虚值 Put" }
  - { key: D, text: "Delta 对冲" }
answer: B
explanation: |
  原文：'The first one is to **use stop losses**'.
  止损单是铁鹰闪崩存活的最重要工具，事先设定阈值。
:::

::: quiz q-fc-2 multiple
id: q-fc-2
section: flash-crash
difficulty: medium
tags: methods
question: |
  文中提到的闪崩应对方法包括？（多选）
options:
  - { key: A, text: "止损单" }
  - { key: B, text: "买入 VIX Call（保险）" }
  - { key: C, text: "买入深度虚值 Put" }
  - { key: D, text: "Delta 对冲" }
answer: A, B, C, D
explanation: |
  4 种方法：① 止损单 ② VIX Call ③ 更虚值 Put ④ Delta 对冲。
  适合不同账户规模和经验水平。
:::

<!-- section: gamma -->

## 理解 Gamma 风险 / Understanding Gamma Risk

Gamma = Delta 的变化率；临到期 + ATM 时最大。

::: quiz q-gamma-1 single
id: q-gamma-1
section: gamma
difficulty: easy
tags: definition
question: |
  Gamma 在下列哪种情况下最大？
options:
  - { key: A, text: "深度虚值期权" }
  - { key: B, text: "平值（ATM）期权" }
  - { key: C, text: "深度实值期权" }
  - { key: D, text: "与 moneyness 无关" }
answer: B
explanation: |
  Gamma 在 ATM 最大、临到期最大。ATM 期权 Delta 接近 0.5，对标的价格变动最敏感。
:::

::: quiz q-gamma-2 single
id: q-gamma-2
section: gamma
difficulty: easy
tags: timing
question: |
  "Gamma 周"（Gamma Week）通常指到期前多久？
options:
  - { key: A, text: "最后 1 周" }
  - { key: B, text: "最后 2-3 周" }
  - { key: C, text: "前 1 周" }
  - { key: D, text: "整个月" }
answer: A
explanation: |
  Gamma 风险主要在到期前 **最后一周** 显著放大。长期铁鹰（90 DTE）大部分时间 Gamma 较低。
:::

<!-- section: adjusting -->

## 调整铁鹰 / Adjusting Iron Condors

调整必须写进交易计划，"希望不是策略"。

::: quiz q-adj-1 single
id: q-adj-1
section: adjusting
difficulty: easy
tags: rule
question: |
  作者关于"调整"的核心建议是？
options:
  - { key: A, text: "事前写进交易计划" }
  - { key: B, text: "等亏损扩大再决定" }
  - { key: C, text: "完全不平仓" }
  - { key: D, text: "每月调一次" }
answer: A
explanation: |
  原文（L840）：'What you don't want to do, is **close your eyes, cross your fingers and hope that the position comes back into profit**.
  **Hope is not a strategy.**'
  调整规则必须提前写入交易计划，临场决策容易情绪化。
:::

::: quiz q-adj-2 multiple
id: q-adj-2
section: adjusting
difficulty: medium
tags: methods
question: |
  文中提及的铁鹰调整方法包括？（多选）
options:
  - { key: A, text: "滚动到下个月份" }
  - { key: B, text: "下移行权价" }
  - { key: C, text: "加倍建仓" }
  - { key: D, text: "买入 VIX Call 对冲" }
answer: A, B, C, D
explanation: |
  文中介绍 9 种调整方法：rolling for a credit、rolling down、doubling up、买 VIX Call、买 Put 保护、加仓、关单、调整宽度等。
  关键：每种方法对应不同市况。
:::

<!-- section: returns -->

## 铁鹰能做到 10% 月收益吗？/ Are 10% Returns Possible with Iron Condors?

技术上可以，但需要 100% 满仓——爆仓只是时间问题。

::: quiz q-ret-1 single
id: q-ret-1
section: returns
difficulty: easy
tags: allocation
question: |
  作者建议铁鹰的合理仓位比例约为账户净值的多少？
options:
  - { key: A, text: "5%" }
  - { key: B, text: "20%" }
  - { key: C, text: "50%" }
  - { key: D, text: "100%" }
answer: B
explanation: |
  原文：'Personally, I think **about a 20% allocation is good**'.
  10%/月收益 = 100% 满仓 + 高 IV Rank 持续——爆仓只是时间问题。
:::

::: quiz q-ret-2 single
id: q-ret-2
section: returns
difficulty: medium
tags: principle
question: |
  关于"高波动加仓、低波动降仓"的原则，原文给出的解释是？
options:
  - { key: A, text: "可以放大收益" }
  - { key: B, text: "波动率与收益存在正相关" }
  - { key: C, text: "做空波动率在高 IV 时盈利概率最大" }
  - { key: D, text: "低 IV 时建仓成本最低" }
answer: C
explanation: |
  原文逻辑：高波动率时期权权利金贵、回归概率大；低波动率时权利金便宜、回归概率小。
  所以 **动态调整仓位** 优于固定仓位。
:::

<!-- section: examples -->

## 铁鹰实战案例 / Iron Condor Examples

两个案例：RUT 长期铁鹰 2018-01-30 建仓、NFLX 铁鹰 2018-07-18 建仓。

::: quiz q-ex-1 single
id: q-ex-1
section: examples
difficulty: medium
tags: rut-case
question: |
  RUT 2018-01-30 长期铁鹰案例中，建仓后发生了什么调整动作？
options:
  - { key: A, text: "立即止损平仓" }
  - { key: B, text: "2-8 整体下移 + 2-14 加倍建仓" }
  - { key: C, text: "持有到期" }
  - { key: D, text: "切换为跨式" }
answer: B
explanation: |
  原文：2-8 RUT 暴跌 → 整体下移铁鹰；2-14 继续跌 → 加倍建仓 Put 端；4-18 反弹 → 平仓共赚 $1,075。
:::

::: quiz q-ex-2 single
id: q-ex-2
section: examples
difficulty: medium
tags: nflx-case
question: |
  NFLX 2018-07-18 铁鹰案例中，最终净收益约为？
options:
  - { key: A, text: "$100" }
  - { key: B, text: "$295" }
  - { key: C, text: "$1,000" }
  - { key: D, text: "$2,000" }
answer: B
explanation: |
  原文：NFLX 案例 8-8 平仓，**net profit $295**。
  同时展示硬止损规则：单笔 $1,000 触及即止损。
:::

::: quiz q-ex-3 single
id: q-ex-3
section: examples
difficulty: medium
tags: stop-loss
question: |
  作者对单笔铁鹰的硬止损阈值（无论怎么想扛）是多少？
options:
  - { key: A, text: "$200" }
  - { key: B, text: "$500" }
  - { key: C, text: "$1,000" }
  - { key: D, text: "无硬止损" }
answer: C
explanation: |
  原文（L1032）：'I have a hard stop loss of **$1,000 per Iron Condor trade**'.
  这是风控底线——"Hope is not a strategy" 的具体化。
:::

::: quiz q-ex-4 single
id: q-ex-4
section: examples
difficulty: hard
tags: adjustment
question: |
  实战案例中，Call 端下移（Rolling Down）的常见原因不包括以下哪项？
options:
  - { key: A, text: "标的大涨突破短腿" }
  - { key: B, text: "希望多收权利金" }
  - { key: C, text: "把即将进入 Gamma 周的铁鹰下移" }
  - { key: D, text: "调整后形成新的铁鹰结构" }
answer: B
explanation: |
  Rolling Down 的目的是**让铁鹰远离危险区 / 推迟 Gamma 周**，不是为了多收权利金。
  标的大涨时 Call 端下移，新铁鹰 delta 偏移更小。
:::

<!-- section: conclusion -->

## 结语 / Conclusion

经验无可替代；小资金起步盯风险；铁鹰是核心策略。

::: quiz q-con-1 single
id: q-con-1
section: conclusion
difficulty: easy
tags: principle
question: |
  作者对铁鹰交易者最核心的建议是？
options:
  - { key: A, text: "赚快钱，all-in 策略" }
  - { key: B, text: "从小资金开始、严守风控" }
  - { key: C, text: "跟着别人的单做" }
  - { key: D, text: "每天都交易" }
answer: B
explanation: |
  原文结语：'start with a small account ... learn risk management first'。
  作者签名：**Trade safe!**
:::

::: quiz q-con-2 single
id: q-con-2
section: conclusion
difficulty: easy
tags: mindset
question: |
  作者的签名（落款）是？
options:
  - { key: A, text: "Happy trading!" }
  - { key: B, text: "Trade safe!" }
  - { key: C, text: "To the moon!" }
  - { key: D, text: "YOLO!" }
answer: B
explanation: |
  原文结尾签名：**Trade safe! / Gav.**
  反映作者的风险优先理念。
:::
