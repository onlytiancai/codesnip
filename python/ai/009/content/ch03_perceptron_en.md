# Chapter 3: The Perceptron — A Simple Learning Machine

> **Back-link**: Uses the "vector" concept from [0.4](#04-vector-a-column-of-numbers-describing-an-object).

## 3.1 The step function: outputs only 0 or 1

The **perceptron** (1958, Rosenblatt) is the "oldest" learning algorithm.

It's like the neuron from Chapter 2, but uses a different activation:

$$
y = \text{step}(w \cdot x + b) = \begin{cases} 1 & \text{if } w \cdot x + b \geq 0 \\ 0 & \text{if } w \cdot x + b < 0 \end{cases}
$$

**Step function**:
- Input $\geq 0$ → output 1
- Input < 0 → output 0
- Only "yes or no" — binary classification

---

## 3.2 Learning rule: adjust weights when wrong

The perceptron's "learning" is super simple:

> **When wrong, push the weights in the right direction.**

Formula (for one sample):

$$
w \leftarrow w + \alpha \cdot (y - \hat{y}) \cdot x
$$

- $y$ is the true label (0 or 1)
- $\hat{y}$ is the prediction
- $(y - \hat{y})$ is the error ($-1, 0, +1$)
- $x$ is the input
- $\alpha$ is the learning rate (step size, e.g., 0.1)

**Intuition**:
- Predicted 0, true 1 (miss) → error $+1$ → **increase $w$**
- Predicted 1, true 0 (false alarm) → error $-1$ → **decrease $w$**
- Predicted correctly → error 0 → $w$ unchanged

**Bias also updates**: $b \leftarrow b + \alpha \cdot (y - \hat{y})$

---

## 3.3 The AND truth table: learned in 3 minutes

**AND** (logical AND): output 1 only when both inputs are 1.

| $x_1$ | $x_2$ | $y$ (true) |
|---|---|---|
| 0 | 0 | 0 |
| 0 | 1 | 0 |
| 1 | 0 | 0 |
| 1 | 1 | 1 |

::: chart caption="AND data + learned boundary (red dashed line)"
![Perceptron boundary](assets/images/ch03_perceptron_boundary.png)
:::

**Key observation**: AND is **linearly separable** — a single line (here $0.3·x_1 + 0.2·x_2 - 0.2 = 0$) separates $(1,1)$ from the other three points.

---

## 3.4 OR is also simple

**OR** (logical OR): output 1 if any input is 1.

| $x_1$ | $x_2$ | $y$ |
|---|---|---|
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 1 |

OR is also linearly separable. The perceptron learns it in minutes.

::: chart caption="AND and OR are both linearly separable"
![AND/OR](assets/images/ch03_and_or_scatter.png)
:::

::: quiz q3-1 single
In the perceptron formula $\text{step}(w \cdot x + b)$, what is $\text{step}(z)$?
- A: Sum
- B: Output 1 if $\geq 0$, else 0
- C: Maximum
- D: Random

answer: B

> step is the unit step function: $z \geq 0$ outputs 1, $z < 0$ outputs 0.
:::

::: quiz q3-2 single
What happens if the learning rate $\alpha$ is too large?
- A: Faster convergence
- B: More stable training
- C: May "swing" near the optimum
- D: Auto-zero

answer: C

> Learning rate too large → step too big → overshoots the optimum → oscillates (or even diverges).
> Too small → too slow to converge.
> Rule of thumb: start with 0.1, watch the loss curve.
:::

::: quiz q3-3 short placeholder="Answer about the perceptron's learning rule"
**Question**: Why doesn't the perceptron update weights when it gets the answer right? What would happen if it did update even on correct answers?

> Reference answer (not unique):
> When the answer is right, $y - \hat{y} = 0$, so the update is $0 \cdot x = 0$ — nothing changes. If we forced an update (even adding tiny noise), it's like perturbing correct answers, causing the model to oscillate on already-correct samples and never converge.
>
> In short: **make right answers more right, fix wrong answers faster** — only adjust what's wrong.
:::

---

## Summary

> 🎯 The perceptron is the "oldest" learning algorithm — just step function + "fix it when wrong".
> **Limit**: can only learn **linearly separable** problems. Next chapter we see how serious this is.
