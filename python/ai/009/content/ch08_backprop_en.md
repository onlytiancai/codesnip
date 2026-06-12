# Chapter 8: Backprop — Tracing the Blame

> **Back-link**: Uses "chain rule + partial derivative" from [0.6](#06-partial-derivative--chain-rule-signal-in-nested-machines).

## 8.1 Chain rule: sensitivity in nested machines

**Backpropagation** = use the **chain rule** to trace from loss $L$ back to $W_1$.

**Why we need it**:
- Forward: $x \to z_1 \to a_1 \to z_2 \to a_2 = \hat{y} \to L$ (compute loss)
- Backward: $L \to \ldots \to a_2 \to \ldots \to z_1 \to \ldots \to W_1$ (compute each W's "influence" on $L$)
- Knowing the "influence" lets us "adjust $W$ to decrease $L$"

**Chain rule review** (0.6):
- $\frac{\partial L}{\partial W_1} = \frac{\partial L}{\partial z_2} \cdot \frac{\partial z_2}{\partial a_1} \cdot \frac{\partial a_1}{\partial z_1} \cdot \frac{\partial z_1}{\partial W_1}$
- 4 partial derivatives **multiplied** = A's influence on D = A→B × B→C × C→D

---

## 8.2 From $L$ back to $W_1$ (4 formulas)

**2-4-1 MLP backprop** (single sample, BCE loss):

**Step 1**: $L$'s partial wrt $a_2$ ("how wrong is $a_2$")
$$
\frac{\partial L}{\partial a_2} = \frac{a_2 - y}{a_2 (1 - a_2)}
$$

**Step 2**: $a_2$'s partial wrt $z_2$ = $\sigma$'s derivative
$$
\frac{\partial L}{\partial z_2} = \frac{\partial L}{\partial a_2} \cdot \sigma'(z_2) = (a_2 - y)
$$
> **Note**: BCE + σ combination makes $\sigma'$ cancel out, giving the beautiful simple form $a_2 - y$.

**Step 3**: $z_2$'s partial wrt $W_2$ = $a_1$
$$
\frac{\partial L}{\partial W_2} = \frac{\partial L}{\partial z_2} \cdot a_1^T = (a_2 - y) \cdot a_1^T
$$

**Step 4**: $z_2$'s partial wrt $a_1$ = $W_2$
$$
\frac{\partial L}{\partial a_1} = W_2^T \cdot \frac{\partial L}{\partial z_2} = W_2^T \cdot (a_2 - y)
$$

**Step 5**: $a_1$'s partial wrt $z_1$
$$
\frac{\partial L}{\partial z_1} = \frac{\partial L}{\partial a_1} \cdot \sigma'(z_1)
$$

**Step 6**: $z_1$'s partial wrt $W_1$ = $x$
$$
\frac{\partial L}{\partial W_1} = \frac{\partial L}{\partial z_1} \cdot x^T
$$

**Update weights** (gradient descent):
$$
W_1 \leftarrow W_1 - \alpha \cdot \frac{\partial L}{\partial W_1}, \quad W_2 \leftarrow W_2 - \alpha \cdot \frac{\partial L}{\partial W_2}
$$

---

## 8.3 Hand calculation: 2-4-1 MLP backward for XOR

::: chart caption="Backpropagation: signal flows from L backward to W1"
![Backprop](assets/images/ch08_chain_rule.png)
:::

::: quiz q8-1 single
What is backpropagation essentially?
- A: Running the data backward
- B: Chain rule + gradient descent
- C: Reinitialization
- D: Random guessing

answer: B

> Backprop = chain rule (compute gradients) + gradient descent (update weights). Both are essential.
:::

::: quiz q8-2 single
Learning rate $\alpha = 0.1$, after 1000 steps, loss is still oscillating. What might be the cause?
- A: Data is wrong
- B: Learning rate too large or network structure wrong
- C: Batch too large
- D: Labels are reversed

answer: B

> Loss oscillation = steps too big, overshooting optimum. Most common: $\alpha$ too large.
> Fix: try smaller $\alpha$ (e.g., 0.01), or use Adam.
> Note: bigger batch makes loss **more stable** (averages out noise).
:::

---

## Summary

> 🎯 Backprop = chain rule from $L$ back to $W_1$. BCE + σ's combination makes $dz_2 = a_2 - y$ elegantly simple.
> Next chapter: how to **loop** forward + backward, forming the training loop.
