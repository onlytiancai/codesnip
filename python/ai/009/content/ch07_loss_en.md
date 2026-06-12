# Chapter 7: Loss Function — How Wrong Are We?

## 7.1 Three kinds of error

**Loss function** = measure of "how different prediction is from truth".

**3 common error metrics** (per sample):

| Name | Formula | Characteristic |
|---|---|---|
| Absolute error MAE | $L = \|y - \hat{y}\|$ | Simple, not differentiable at 0 |
| Squared error MSE | $L = (y - \hat{y})^2$ | Differentiable, amplifies large errors |
| Cross-entropy CE | $L = -[y\log\hat{y} + (1-y)\log(1-\hat{y})]$ | Default for binary classification |

**Binary Cross-Entropy** (BCE) details:
- Applies to binary classification (y=0 or 1)
- True $y=1$: $L = -\log(\hat{y})$, smaller when $\hat{y} \to 1$
- True $y=0$: $L = -\log(1-\hat{y})$, smaller when $\hat{y} \to 0$

---

## 7.2 Binary Cross-Entropy: intuition + formula

**Intuition**: BCE comes from "information theory"
- True $y=1$, prediction $\hat{y}$ close to 1 → small loss
- True $y=0$, prediction $\hat{y}$ close to 0 → small loss
- **Completely wrong** ($\hat{y} \to 0$ but $y=1$) → loss $\to +\infty$

::: chart caption="BCE loss: more accurate prediction = smaller loss"
![BCE](assets/images/ch07_bce_curve.png)
:::

**Examples**:
- $y=1, \hat{y}=0.99$: BCE = $-\log(0.99) \approx$ **0.01** (nearly perfect)
- $y=1, \hat{y}=0.5$: BCE = $-\log(0.5) \approx$ **0.69** (random guess)
- $y=1, \hat{y}=0.01$: BCE = $-\log(0.01) \approx$ **4.6** (catastrophic)

---

## 7.3 Why BCE for classification, not MSE

| Dimension | MSE | BCE |
|---|---|---|
| Formula | $(y - \hat{y})^2$ | $-[y\log\hat{y} + (1-y)\log(1-\hat{y})]$ |
| Gradient at 0 | Small → slow learning | Large → fast learning |
| Extreme values | Gradient saturation | Always effective |
| Probabilistic interpretation | None | Strong (KL divergence) |

**Vanishing gradient problem** (classic pitfall of MSE + sigmoid):
- When $\sigma$ output is near 0 or 1, $\sigma'$ is near 0
- MSE gradient passes through $\sigma'$ → also near 0 → weights barely update → can't learn
- BCE gradient passes through $\sigma'$ → **cancels it out** → no vanishing ✓

**This is why classification (probability output) almost always uses BCE, not MSE**.

::: quiz q7-1 single
True $y=1$, prediction $\hat{y}=0.01$, BCE loss is approximately?
- A: 0.01
- B: 4.6
- C: 0.99
- D: 0

answer: B

> $L = -\log(0.01) = -\log(10^{-2}) = 2 \log 10 \approx 4.6$. Predicted 0.01 but truth is 1 → catastrophic.
:::

::: quiz q7-2 multiple
What must a loss function satisfy? (Multiple choice)
- A: Differentiable (in most cases)
- B: Bigger is better
- C: Reflect "how wrong the prediction is"
- D: Always positive output

answer: A,C,D

> A ✓: need gradients → must be differentiable.
> B ✗: **smaller is better** — training minimizes loss.
> C ✓: the whole point — measure wrongness.
> D ✓: loss $\geq 0$ (distance).
:::

::: quiz q7-3 short placeholder="Explain why BCE resists vanishing gradient"
**Question**: If we replace BCE with MSE ($(y - \hat{y})^2$) and use a sigmoid output, what vanishing-gradient problem arises?

> Reference answer (not unique):
> Assume $y=1$, $\hat{y} = \sigma(z) \approx 0$ (z is very small).
> - MSE gradient: $\frac{\partial L}{\partial z} = 2(\sigma(z) - y) \cdot \sigma'(z) = 2(0 - 1) \cdot \sigma'(z) \approx -2\,\sigma'(z)$.
> - When z is small, $\sigma'(z) \approx 0$ → gradient ≈ 0 → weights barely update.
>
> Compare with BCE gradient: $\frac{\partial L}{\partial z} = \sigma(z) - y = 0 - 1 = -1$ (does **not** contain $\sigma'$).
> - Even when z is small, the gradient is still -1 → learning is not "stuck".
>
> **BCE's clever "cancelling"**: it eliminates $\sigma'$ in the gradient, bypassing the vanishing-gradient problem.
:::

---

## Summary

> 🎯 Loss = "how wrong". **For classification, prefer BCE** — avoids sigmoid + MSE's vanishing gradient.
> Next chapter: use "loss" to **adjust weights** — backpropagation.
