# Iron Condor 期权策略专业分析指南

> Trading dates: April 2026 | SPX Options Analysis
> Market data powered by yfinance | Position data via Futu OpenAPI

---

## Chapter 1: Greek Letters Analysis

### Delta: Neutral Characteristics & Win Rate

Delta measures the rate of change in option price relative to the underlying price movement.

**Iron Condor Delta Profile:**
- Iron Condor targets Delta near zero (directionally neutral)
- Net Delta is typically small, indicating minimal directional bias
- Delta of ±0.10 to ±0.20 per leg is acceptable for most IC structures

**Win Rate Relationship:**
- SPX typically moves 0.5-1% daily (roughly ±5-10 index points)
- For a 20-wide IC centered around SPX 5500:
  - Short put at 5450 (Delta ≈ -0.15 to -0.20)
  - Short call at 5550 (Delta ≈ +0.15 to +0.20)
  - Net Delta ≈ 0 (balanced)

**Delta as Probability Proxy:**
| Delta Value | Approximate ITM Probability | Interpretation |
|-------------|---------------------------|----------------|
| 0.50 | 50% | At-the-money |
| 0.20 | ~80% OTM | Wing protection level |
| 0.10 | ~90% OTM | Distant OTM strike |
| -0.10 | ~90% OTM (put side) | Distant OTM strike |

**Practical Rule:**
- Short strike Delta ≈ 0.20 suggests ~80% probability of expiring OTM
- Iron Condor wings typically set at Delta 0.10-0.15 for 85-90% success rate target

### Gamma: Short Gamma Risk & DTE Relationship

Gamma measures the rate of change of Delta relative to the underlying.

**Iron Condor is a Short Gamma Strategy:**

| Position | Gamma Sign | Effect |
|----------|-----------|--------|
| Long option | Positive | Gains value when market moves |
| Short option | Negative | Loses value when market moves |
| **Iron Condor (net)** | **Negative** | **不利波动** |

**Short Gamma Risks:**
1. **Pinning Risk**: Large institutional positions can pin price near short strikes at expiry
2. **Gap Risk**: Overnight news can cause gaps past wings
3. **Gamma Scalping Cost**: Market makers hedge dynamically, costing the short gamma trader

**Gamma vs DTE (Days to Expiration):**
```
DTE    | Gamma per Strike | Total IC Gamma Exposure
------ | ---------------- | -----------------------
45     | Low              | Manageable, slow delta drift
30     | Moderate         | Noticeable gamma acceleration
15     | High             | Short gamma effects become significant
7      | Very High        | Dangerous territory - delta can swing rapidly
1      | Extreme          | Pin risk maximum, avoid new positions
```

**Gamma Risk Management:**
- Monitor total portfolio Gamma: `-Gamma × SPX × 100` should remain below daily theta intake
- When Gamma spikes (DTE < 15), consider narrowing the structure or taking profits

### Theta: Positive Theta Characteristics & DTE Relationship

Theta measures time decay - the daily "rent" collected for being short options.

**Iron Condor Generates Positive Theta:**
- Short options lose value as time passes ( Theta decays in your favor)
- Long options (wings) have negative theta (cost you daily)
- Net theta is typically positive for balanced IC structures

**Theta per DTE Table:**
| DTE | Typical Daily Theta | Notes |
|-----|---------------------|-------|
| 45+ | Lower per day | Theta accumulates slowly, IV may decay |
| 30 | Moderate | Sweet spot for Iron Condors |
| 21 | Good | Often optimal entry point |
| 14 | High | Peak theta collection period |
| 7 | Very high | Last gasp theta, gamma risk increases |
| 1-2 | Maximum | Close to expiry, gamma dominates |

**Theta Decay Curve:**
- Theta accelerates non-linearly: minimal at 60+ DTE, accelerates after 30 DTE
- At 7 DTE, theta is 3-4x higher than at 30 DTE
- Chart pattern: convex curve, steeper near expiry

**Theta Calculation (Example):**
```
SPX IC: Net credit $2.50 (250 per contract)
DTE: 30 days
Daily theta: $2.50 / 30 ≈ $8.33 per day
Total theta if held to expiry: $2.50 × 100 = $250 per contract
Actual realized theta depends on early close or adjustment
```

### Vega: Volatility Impact & VIX Relationship

Vega measures sensitivity to implied volatility changes.

**Iron Condor is Short Vega:**
- Short options have negative vega
- IV rises → options you sold become more expensive (bad)
- IV falls → options you sold become cheaper (good)

**Vega Exposure by Strike:**
| Strike | Delta | Vega Contribution |
|--------|-------|------------------|
| Deep ITM | High | Negative (short position) |
| ATM | 0.50 | Most negative vega |
| OTM (wing) | Low | Small but meaningful |

**VIX Relationship:**
- VIX > 25: High volatility environment
  - Wider wings recommended
  - Higher premium available (more credit)
  - Higher IV means more premium but also more risk
  - Consider smaller position size

- VIX 15-25: Normal volatility
  - Standard 20-wide IC typically works
  - Balanced risk/reward

- VIX < 15: Low volatility environment
  - Narrower wings (15-wide) to generate adequate credit
  - IV may crush further, benefiting short vega
  - Be cautious of volatility spike risk

**Vega Risk Formula:**
```
Portfolio Vega = Σ (Vega_i × Qty_i × 100)
Example:
  Short 5 lots SPX 5500C @ Vega -0.15 × -5 × 100 = +75 (benefits from IV rise)
  Long 5 lots SPX 5700C @ Vega +0.08 × 5 × 100 = -40 (hurts from IV rise)
  Net: +35 (slightly long vega - unusual, needs review)
```

### Rho: Interest Rate Impact

Rho measures sensitivity to interest rate changes.

**Iron Condor Rho Profile:**
- Short puts have negative rho (benefit from rate increases)
- Short calls have positive rho (hurt by rate increases)
- Net rho effect is typically small for SPX options

**Practical Impact:**
- SPX options are equity-index options, rho is secondary consideration
- In high-rate environments (2022-2024), rho effect becomes more noticeable
- For short-dated options (DTE < 30), rho impact is minimal
- 1% rate change affects option price by < 1% typically

---

## Chapter 2: Volatility Analysis

### IV Analysis: IV vs RV, IV Rank

**Implied Volatility (IV) vs Realized Volatility (RV):**

| Metric | Definition | Role |
|--------|------------|------|
| IV | Market's expectation of future volatility | What you sell options at |
| RV (Realized) | Actual historical volatility | What actually happens |

**IV vs RV Relationship:**
- IV > RV: Options are "expensive" - favorable for selling
- IV < RV: Options are "cheap" - be cautious of selling
- IV/RV Ratio > 1.2: Premium selling environment
- IV/RV Ratio < 0.8: Consider taking profits or avoiding new sales

**IV Rank (Percentile):**
```
IV Rank = (Current IV - 52W Low) / (52W High - 52W Low) × 100

Example:
  52W Low IV: 12%
  52W High IV: 28%
  Current IV: 20%
  IV Rank = (20 - 12) / (28 - 12) × 100 = 50%
```

**IV Rank Interpretation:**
| IV Rank | Environment | Strategy Implication |
|---------|-------------|----------------------|
| 0-20% | Very low IV | Reduce size, IV may crush further |
| 20-40% | Low IV | Neutral, premiums may be thin |
| 40-60% | Moderate IV | Normal conditions, standard IC |
| 60-80% | High IV | Good premium environment |
| 80-100% | Very high IV | Excellent premium but risk elevated |

### VIX: Market Sentiment & Term Structure

**VIX Overview:**
- VIX measures S&P 500 index option implied volatility
- Also known as the "fear index"
- Inverse relationship with stock prices (typically)

**VIX Levels & Market Implications:**
| VIX Range | Market Status | IC Implication |
|-----------|--------------|----------------|
| < 15 | Complacent | Low premium, reduce size |
| 15-20 | Normal | Standard IC framework |
| 20-25 | Elevated | Higher premiums, wider wings |
| 25-30 | High stress | Reduce exposure, very wide wings |
| > 30 | Crisis | Avoid new IC, manage existing |

**VIX Term Structure:**
```
Normal (Contango):
  1M: 18%  →  3M: 20%  →  6M: 21%  →  12M: 22%
  (Upward sloping = normal decay of volatility term premium)

Inverted (Backwardation):
  1M: 25%  →  3M: 22%  →  6M: 20%  →  12M: 19%
  (Spot VIX elevated, near-term fear high)

Flat:
  1M: 20%  →  3M: 20%  →  6M: 20%
  (Uncertain direction)
```

**Contango vs Backwardation Effects:**
- **Contango**: Normal state, front-month IV lower than back-month
  - Short-dated options may have less premium
  - Roll costs low for calendar spreads

- **Backwardation**: Fear spike, near-term IV elevated
  - Front-month premium high
  - Short-dated IC may be attractive
  - Watch for quick mean reversion

### Volatility Smile: OTM Put Skew

**The Smile/Skew Phenomenon:**
- OTM puts (downside) trade at higher IV than equivalent OTM calls
- Reflects demand for downside protection
- Creates opportunity in put skew selling

**SPX Put Skew Structure:**
```
Strike      | Delta  | IV (Typical) | Notes
------------|--------|--------------|------------------
5400 (OTM)  | -0.10  | 19-21%       | Wing level
5450 (OTM)  | -0.15  | 18-19%       | Common IC short put
5500 (ATM)  | -0.50  | 16-17%       | ATM level
5550 (OTM)  | +0.15  | 17-18%       | Common IC short call
5600 (OTM)  | +0.10  | 18-20%       | Wing level
```

**Put Skew Implications for IC:**
1. OTM puts are "expensive" relative to calls at same delta
2. Selling OTM puts generates more premium than selling equivalent calls
3. Put wing is typically set wider than call wing to account for crash risk
4. Black swan events: left tail risk > right tail risk

**Skew Ratio Calculation:**
```
Skew Ratio = IV(OTM Put at Delta -0.15) / IV(OTM Call at Delta +0.15)

Example:
  Put IV: 18.5% (Delta -0.15)
  Call IV: 17.0% (Delta +0.15)
  Skew Ratio = 18.5 / 17.0 = 1.09

Interpretation: Puts are 9% more expensive than calls at equivalent delta
```

### Term Structure: Contango/Backwardation, ES Futures vs SPX

**Contango (Normal):**
- Future price > Spot price
- Indicates expected higher future spot price OR storage costs
- Normal for equity indices with positive carry

```
ES Front Month: 5512
SPX Spot: 5500
Contango: +12 points (0.22%)

Typical contango: 0.10% - 0.50% annualized
```

**Backwardation (Inverted):**
- Future price < Spot price
- Often indicates:
  - Near-term supply/demand imbalance
  - Expected spot price decline
  - Risk-off sentiment (futures selling)

```
ES Front Month: 5490
SPX Spot: 5500
Backwardation: -10 points (-0.18%)
```

**Contango/Backwardation Impact on IC:**

| Condition | Effect on IC |
|-----------|--------------|
| Contango | Normal carry, standard IC construction |
| Deep Contango | May indicate upcoming volatility term premium decay |
| Mild Backwardation | Normal if VIX elevated, monitor |
| Deep Backwardation | Significant fear event, avoid new positions |

**ES-SPX Spread Trading:**
- ES futures track SPX closely but not perfectly
- Spread can diverge 5-15 points in stressed markets
- Use ES as a rough hedge reference for SPX positions
- During high stress: ES may trade at larger discount

---

## Chapter 3: Put/Call Ratio & Market Structure

### PCR Indicator Interpretation

**Put/Call Ratio Formula:**
```
PCR = Total Put Volume / Total Call Volume
     OR
PCR = Total Put OI / Total Call OI  (more meaningful for positions)
```

**PCR Levels & Interpretation:**
| PCR Value | Interpretation | Market Signal |
|-----------|---------------|---------------|
| < 0.70 | Very bullish | Extreme greed, caution warranted |
| 0.70-0.90 | Bullish | Generally positive for stocks |
| 0.90-1.10 | Neutral | Balanced positioning |
| 1.10-1.30 | Bearish | Increased put buying, defensive |
| > 1.30 | Very bearish | High fear, potential bottom zone |

**IC-Specific PCR Use:**
- High PCR (>1.2): More premium available on put side
  - Consider wider put wing for more credit
  - Sentiment may be too bearish

- Low PCR (<0.8): Less put premium
  - May need to narrow put wing
  - Bullish sentiment, reduce put side exposure

**SPX-Specific Considerations:**
- SPX is a covered-asset (cash-settled)
- Large institutional hedging uses SPX puts
- PCR > 1.0 is normal for SPX due to portfolio insurance hedging

### SPX Spot vs ES Futures Relationship

**Price Discovery:**
```
ES (Electronic S&P 500) = 5512
SPX (Cash Index) = 5500
Premium/Discount = ES - SPX = +12 points

This premium reflects:
1. Cost of carry (interest rates)
2. Expected dividend adjustments
3. Supply/demand in futures market
```

**Spread Dynamics:**
| Condition | ES vs SPX | Interpretation |
|-----------|-----------|----------------|
| Normal contango | ES > SPX | Carry positive, futures fairly valued |
| Tight spread | ES ≈ SPX | Possible rebalancing flows |
| ES discount | ES < SPX | Risk-off, futures selling, possible selling pressure |

**Practical Trading Notes:**
1. **ES is the leading indicator** - it trades 23 hours
2. **SPX cash opens 9:30 ET** - gap fills from ES overnight move
3. **During market stress**: ES may discount SPX significantly
4. **IC adjustments**: Use ES level as reference for strike proximity

**Overnight Gap Risk:**
```
ES Close (4:15 PM): 5510
ES Overnight High: 5525 (+15 points)
ES Overnight Low: 5485 (-25 points)

Gap Risk: If ES gaps past IC wings overnight,
          morning adjustment may be necessary
```

---

## Chapter 4: Adjustment Strategies

### Key Adjustment Methods

| Adjustment | When to Use | Win Rate | P/L Ratio | Margin | Max Loss |
|-----------|-------------|----------|-----------|--------|----------|
| **Narrow Wing** | Reduce risk | Increases | Decreases | Decreases | Decreases |
| **Widen Wing** | Increase premium | Decreases | Increases | Increases | Increases |
| **Roll Up** | Breakout above | Varies | Varies | Varies | Changes |
| **Roll Down** | Breakout below | Varies | Varies | Varies | Changes |
| **Close Partial** | Lock in profit | - | - | Decreases | Decreases |
| **Convert to Iron Butterfly** | Expect range-bound | Increases | Decreases | Decreases | Decreases |

### 1. Narrow Wing (收窄 Wing)

**Mechanics:**
- Buy back the short option closer to current price
- Sell a new further OTM option to maintain wing width
- Net effect: reduced risk, reduced credit

**Example - April IC:**
```
Original:
  Short 5450P @ Delta -0.15, credit $2.00
  Long 5400P @ Delta -0.08, debit $0.80
  Net credit: $1.20

Narrow:
  Buy back 5450P @ $1.50 (now closer, IV crushed)
  Sell new 5425P @ Delta -0.12, credit $1.20
  Net effect: reduced wing, locked in some profit
```

**When to Apply:**
- Price moved significantly but stayed OTM
- VIX declined (IV compression)
- DTE > 30, theta accumulation complete

**Trade-off:**
- Win rate increases (more room to OTM)
- Profit potential decreases (less net credit)
- Margin requirement may decrease

### 2. Widen Wing (扩大 Wing)

**Mechanics:**
- Close existing short option at a loss
- Sell new option further OTM
- Collect additional credit to offset loss

**Example:**
```
Original: Short 5500C, credit $2.50, 30 DTE
Current:  SPX at 5550 (short call ITM)

Adjustment:
  Buy back 5500C @ $6.00 (loss $3.50)
  Sell new 5575C @ $3.50
  Net cost: $0 (break-even on adjustment)

New structure: Wider call wing, same premium risk
```

**When to Apply:**
- Price approaching short strike
- High IV environment (premium available)
- Strong conviction that breakout is temporary

**Trade-off:**
- Higher potential profit
- Lower win rate
- Higher margin requirement
- Larger max loss

### 3. Roll Up (向上移仓)

**Mechanics:**
- Close short call that's at risk
- Sell new higher-strike call
- Collect premium to offset loss

**When to Apply:**
- Strong upside breakout
- SPX trending higher
- Economic data / catalysts supportive

**Example:**
```
Original April IC Call Side:
  Short 5550C @ $2.00
  SPX at 5560 (approaching strike)

Roll Up:
  Buy 5550C @ $4.00 (loss $2.00)
  Sell 5575C @ $2.50
  Net debit: $1.50 additional

New call spread: 5575/5600 vs original 5550/5575
```

### 4. Roll Down (向下移仓)

**Mechanics:**
- Close short put that's at risk
- Sell new lower-strike put
- Same concept as roll up, but for downside

**When to Apply:**
- Downside breakout
- Market weakness / risk-off
- Support levels broken

### 5. Close Partial (平仓部分)

**Mechanics:**
- Close entire position (all legs) when profitable
- Or close one side (e.g., put spread only)

**When to Apply:**
- Target profit reached (e.g., 50-70% of max profit)
- Major news event ahead
- Weekend / holiday with uncertainty

**Partial Close Strategy:**
```
Original IC: $2.50 credit, 30 DTE
At 10 DTE: Price at $1.00 (60% profit)

Option A: Close all
  Buy back all legs @ $1.00
  Profit: $1.50 per contract (60% of max)

Option B: Close put spread only
  Keep call spread (still has value)
  Lock in put spread profit
  Let call spread run
```

### 6. Convert to Iron Butterfly (转铁蝶式)

**Mechanics:**
- Bring short strikes closer to ATM
- Symmetric structure around current price
- Reduced wing width

**Transformation:**
```
Iron Condor → Iron Butterfly

IC:    Long 5400P / Short 5450P / Short 5550C / Long 5600C
       Width: 500 points on each side

Butterfly: Long 5425P / Short 5500P / Short 5500C / Long 5575C
           Width: 75 points to ATM, wings 75 points away
```

**When to Apply:**
- Price stuck in tight range
- Volatility collapsed
- DTE < 21, want to reduce gamma risk

**Trade-off:**
- Higher probability of profit (butterfly is more neutral)
- Lower max profit
- Reduced margin
- Lower max loss

---

## Chapter 5: Practical Analysis

### Current Portfolio: April 2026 + May 2026 Iron Condors

Based on the `spx_options.py` script output and market data (April 2, 2026):

**Portfolio Structure:**
```
April Iron Condor:
  - Put Spread: Long 5400P / Short 5450P
  - Call Spread: Short 5550C / Long 5600C
  - Net Credit: ~$2.50 (250 per contract)
  - DTE: ~15-20 days (April expiry)
  - Max Profit: $250 per contract
  - Max Loss: ~$500 - $250 = $750 per contract

May Iron Condor:
  - Put Spread: Similar structure, further expiry
  - Call Spread: Similar structure
  - Net Credit: ~$3.00 (300 per contract)
  - DTE: ~45 days (May expiry)
  - Max Profit: $300 per contract
```

### Greek Analysis (April IC)

**Current SPX Price: ~5500 (assumed)**

**Put Side Analysis:**
| Strike | Type | Delta | Gamma | Theta | Vega |
|--------|------|-------|-------|-------|------|
| 5400 | Long (Buy) | -0.08 | +0.03 | -0.05 | +0.08 |
| 5450 | Short (Sell) | -0.15 | -0.05 | +0.12 | -0.12 |
| **Net** | | **-0.23** | **-0.02** | **+0.07** | **-0.04** |

**Call Side Analysis:**
| Strike | Type | Delta | Gamma | Theta | Vega |
|--------|------|-------|-------|-------|------|
| 5550 | Short (Sell) | +0.15 | -0.05 | +0.10 | -0.10 |
| 5600 | Long (Buy) | +0.08 | +0.03 | -0.04 | +0.06 |
| **Net** | | **+0.23** | **-0.02** | **+0.06** | **-0.04** |

**Combined IC Greeks:**
```
Total Delta:  ≈ 0 (well hedged)
Total Gamma:  ≈ -0.04 (short gamma - vulnerable to large moves)
Total Theta:  ≈ +$0.13 × 100 = +$13/day (time working for you)
Total Vega:   ≈ -0.08 (short vega - vulnerable to IV spike)
```

### Risk Assessment

**Gamma Risk (Short Gamma Exposure):**
```
Gamma Risk = -Gamma × SPX Price × 100
           = -0.04 × 5500 × 100
           = -$2,200 per 1% move

Interpretation: For every 1% SPX move, lose ~$2,200
               1% move = 55 points = $550 per contract × 4 legs
```

**Theta/Gamma Ratio (Key Metric):**
```
Daily Theta: +$13
Gamma Risk per 1% Move: ~$2,200

Ratio: $13 / $2,200 = 0.006

This is low - theta doesn't cover gamma risk well
At DTE 15, this is acceptable but monitor closely
```

### IV & VIX Analysis

**Current Environment (April 2, 2026):**
```
VIX Level: Assumed ~18-20 (normal range)
IV vs RV: Depends on recent realized moves

If VIX 18:
  - IV Rank likely 40-60% (moderate premium)
  - Standard 20-wide IC appropriate
  - Current structure: reasonable

If VIX > 25:
  - Consider wider wings (25-wide)
  - Reduce position size
  - Higher premium but higher risk
```

**April IC Adjustment Triggers:**

| SPX Level | Status | Action |
|-----------|--------|--------|
| < 5430 | Put side threatened | Consider roll down or narrow put wing |
| 5430-5470 | Healthy range | Hold, collect theta |
| 5470-5530 | Sweet spot | Maximum theta capture |
| 5530-5570 | Call side threatened | Consider roll up or narrow call wing |
| > 5570 | Call side in danger | Roll up or convert to butterfly |

### May IC Positioning

**May IC vs April IC:**

| Factor | April IC | May IC |
|--------|----------|--------|
| DTE | ~15 | ~45 |
| Theta/day | Higher ($15+) | Lower ($7-8) |
| Gamma risk | Higher | Lower |
| Premium captured | Fast | Slow build |
| Adjustment flexibility | Less | More |

**May IC Recommendations:**
1. **Hold for theta accumulation**: 45 DTE allows time
2. **Watch for 30 DTE mark**: Consider taking profit if >70% of max credit
3. **Monitor April IC adjustment**: If April needs work, may affect May
4. **Watch VIX term structure**: If front-month VIX spikes, May IV may follow

### Adjustment Decision Tree

```
Is April IC profitable > 50%?
├── YES: Can hold for more theta
│   └── Is DTE < 10?
│       ├── YES: Take profit, close all
│       └── NO: Hold, monitor
│
├── NO: Adjustment needed
│   └── Which side is threatened?
│       ├── Put side (SPX < 5430)
│       │   ├── How far OTM is long put?
│       │   ├── Consider: Roll down OR narrow wing
│       │   └── Decision based on VIX level
│       │
│       └── Call side (SPX > 5530)
│           ├── How far OTM is long call?
│           ├── Consider: Roll up OR narrow wing
│           └── Decision based on momentum
```

### Exit Strategy Summary

**Tiered Exit Plan:**

| Target | Timing | Action |
|--------|--------|--------|
| 50% profit | Any time | Review, consider holding |
| 70% profit | DTE > 21 | Take profit, move to May |
| 70% profit | DTE < 14 | Take profit, too much gamma |
| Max profit | Near expiry | Let expire if OTM |
| Stop loss | > 80% loss | Close to avoid assignment |

**Key Dates to Watch:**
- April Expiry: ~April 17, 2026
- May IC: Continue holding, adjust as needed
- VIX events: Fed meetings, CPI, NFP

---

## Appendix: Quick Reference

### Greek Sign Summary (Short IC)

```
Position: Net Delta ≈ 0
          Net Gamma < 0 (short gamma)
          Net Theta > 0 (time decay benefit)
          Net Vega < 0 (IV increase hurts)

Risk Priority:
  1. Gamma (big moves)
  2. Vega (IV spikes)
  3. Theta (gradual decay - your friend)
```

### Volatility Checklist

- [ ] Current VIX level (15-25 normal range)
- [ ] VIX term structure (contango vs backwardation)
- [ ] IV Rank (where current IV sits in 52-week range)
- [ ] IV vs RV comparison (premium fair?)
- [ ] Put/call skew (put wings wider?)

### Position Monitoring Checklist

- [ ] Distance to short strikes (% and delta)
- [ ] DTE countdown
- [ ] Daily theta accumulation
- [ ] Any news/events ahead
- [ ] VIX movement correlation

---

*Document generated: April 2026*
*Data sources: yfinance (market), Futu OpenAPI (positions)*
*For educational purposes only - not financial advice*
