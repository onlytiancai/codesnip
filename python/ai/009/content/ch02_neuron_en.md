# Chapter 2: One Neuron = One Function

> **Back-link**: This chapter uses the "function" concept from [0.1](#01-function-throw-in-a-number-spit-out-a-number).

## 2.1 Decomposing the neuron formula

Translating the "spirit" from last chapter into math:

$$
y = \sigma(w \cdot x + b)
$$

**4 parameters**:
- $x$: input (a number or vector)
- $w$: weight (same shape as $x$, determines each input's importance)
- $b$: bias (the "threshold" — bigger $b$ means harder to activate)
- $\sigma$: activation function (squashes any real number to (0, 1))

::: chart caption="Neuron = 4-step pipeline"
![Neuron formula](assets/images/ch02_neuron_function.png)
:::

---

## 2.2 What are weights $w$ and bias $b$?

**Weight $w$** (weight):
- "Importance" — bigger $w$ means more important input
- $w > 0$: excitatory (this feature makes activation more likely)
- $w < 0$: inhibitory (this feature makes activation less likely)
- $w = 0$: this input is "ignored"

**Bias $b$** (bias):
- "Threshold" — bigger $b$ makes activation harder
- $b = 0$: threshold is 0
- $b = -10$: low threshold, easy to activate
- $b = 10$: high threshold, almost never activates

**Real example**: "should I go out to play?"
- $x_1$ = weather quality (0=bad, 1=good)
- $x_2$ = temperature comfort (0~1)
- $x_3$ = has homework (0=no, 1=yes)
- $w = [0.5, 0.3, -0.8]$ (weather matters, homework is a minus)
- $b = -0.2$ (low threshold, most cases go out)
- Output: $y = \sigma(0.5·x_1 + 0.3·x_2 - 0.8·x_3 - 0.2)$
- $y$ closer to 1 → "go out"; closer to 0 → "stay home"

---

## 2.3 Example: turning 3 inputs into 1 number

**Scenario**: should I go out today? 3 factors: temperature, humidity, weekend.

| Input | Meaning | Value |
|---|---|---|
| $x_1$ | Temperature (0=cold, 1=hot) | 0.8 |
| $x_2$ | Humidity (0=dry, 1=wet) | 0.3 |
| $x_3$ | Is weekend (0=weekday, 1=weekend) | 1.0 |

Weights (assumed learned):
- $w_1 = 0.5$ (temperature matters)
- $w_2 = -0.3$ (humidity is bad)
- $w_3 = 0.4$ (weekend bonus)
- $b = -0.2$

**Calculation**:
1. Weighted sum: $sum = 0.5·0.8 + (-0.3)·0.3 + 0.4·1.0 + (-0.2) = 0.4 - 0.09 + 0.4 - 0.2 = 0.51$
2. Activation: $y = \sigma(0.51) \approx 0.625$
3. Interpretation: $y = 0.625$, leans toward "go out" (anything > 0.5 supports going out)

::: formula
$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$
:::

::: quiz q2-1 single
In the neuron formula $\sigma(w \cdot x + b)$, which of the 4 parts ($x, w, b, \sigma$) is most like a "threshold"?
- A: $x$ (input)
- B: $w$ (weight)
- C: $b$ (bias)
- D: $\sigma$ (activation function)

answer: C

> The bias $b$ controls the "threshold" — bigger $b$ means the neuron is harder to activate.
:::

::: quiz q2-2 single
What is the range of $y = \sigma(w \cdot x + b)$?
- A: Any real number $(-\infty, +\infty)$
- B: $[0, 1]$
- C: $(0, 1)$ (open interval)
- D: $\{-1, +1\}$

answer: C

> σ squashes any real number to the open interval (0, 1) (never quite 0 or 1, but can get arbitrarily close).
:::

::: quiz q2-3 short placeholder="Explain weight in 1 sentence to a friend"
Explain "what is a weight" to your friend in **one sentence**.

> Reference answer (not unique):
> - "A weight is the importance score of a feature — high score means it has a big impact on the final result; negative score means it's a deduction."
> - "A weight is 'favoritism' — features with large weights get more 'attention' from the neuron."
> - "A weight is a volume knob — bigger weight, louder feature."

---

## Summary

> 🎯 A neuron = a function of 4 parameters ($x, w, b, \sigma$). Next, we make this function "learn" — i.e., automatically adjust $w$ and $b$.
