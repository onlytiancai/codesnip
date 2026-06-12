# Chapter 6: Forward Propagation — Feeding the Data In

> **Back-link**: Heavily uses "matrix multiplication" from [0.5](#05-matrix-and-matrix-multiplication-the-magic-of-tables).

## 6.1 Step 1: $z_1 = X \cdot W_1 + b_1$

**Forward propagation** = the complete computation from input to output.

**2-4-1 MLP forward** (single sample):

$$
\begin{aligned}
z^{(1)} &= W^{(1)} \cdot x + b^{(1)} & \text{linear (input → hidden)} \\
a^{(1)} &= \sigma(z^{(1)}) & \text{activation (4-dim vector)} \\
z^{(2)} &= W^{(2)} \cdot a^{(1)} + b^{(2)} & \text{linear (hidden → output)} \\
a^{(2)} &= \sigma(z^{(2)}) & \text{activation} \\
\hat{y} &= a^{(2)} & \text{prediction}
\end{aligned}
$$

**Shape changes** (single sample $x$ is a 2-dim vector):
- $x$: shape $(2,)$
- $W_1$: shape $(4, 2)$
- $z_1 = W_1 \cdot x + b_1$: shape $(4,)$
- $a_1 = \sigma(z_1)$: shape $(4,)$
- $W_2$: shape $(1, 4)$
- $z_2 = W_2 \cdot a_1 + b_2$: shape $(1,)$
- $a_2 = \sigma(z_2)$: shape $(1,)$ — one number, the prediction $\hat{y}$

---

## 6.2 Step 2: $a_1 = \sigma(z_1)$ (activation appears)

**Activation function** $\sigma$: squashes any real number to $(0, 1)$.

**Why do we need activation?**
- Without $\sigma$ → the network is just "stacked linear transformations" → still linear → equivalent to 1 layer
- With $\sigma$ → **nonlinearity** → multi-layer matters → can learn curves

---

## 6.3 Step 3: $z_2 = a_1 \cdot W_2 + b_2 \to a_2 = \sigma(z_2)$

Same as step 1, but with $W_2$ and $a_1$ as input.

**Note shape**:
- $a_1$ is 4-dim → need $W_2$ shape $(1, 4)$: 1 output neuron × 4 inputs
- $z_2$ is 1-dim → $\sigma(z_2)$ is 1 number → that's the prediction

---

## 6.4 Hand calculation: 2-4-1 MLP for XOR sample (0, 1)

**Setup** (for teaching, **not** learned values):

$$
W^{(1)} = \begin{bmatrix} 0.5 & 0.7 \\ -0.3 & 0.8 \\ 0.6 & -0.2 \\ 0.4 & 0.9 \end{bmatrix}, \quad b^{(1)} = \begin{bmatrix} -0.1 \\ 0.2 \\ -0.1 \\ 0.3 \end{bmatrix}
$$

$$
W^{(2)} = \begin{bmatrix} 0.5 & -0.4 & 0.6 & 0.3 \end{bmatrix}, \quad b^{(2)} = -0.2
$$

**Input** $x = [0, 1]$ (XOR true = 1)

**Step 1**: $z_1 = W_1 \cdot x + b_1$
$$
z_1 = \begin{bmatrix} 0.6 \\ 1.0 \\ -0.3 \\ 1.2 \end{bmatrix}
$$

**Step 2**: $a_1 = \sigma(z_1) \approx [0.646, 0.731, 0.426, 0.768]$

**Step 3**: $z_2 = W_2 \cdot a_1 + b_2 \approx 0.232$

**Step 4**: $a_2 = \sigma(z_2) \approx 0.558$

**Prediction** $\hat{y} \approx 0.558$. XOR true is 1, prediction 0.558 → **close to 1** → correct ✓ (not very precise, but right direction)

::: chart caption="Forward pass computation graph"
![Forward computation graph](assets/images/ch06_compute_graph.png)
:::

::: quiz q6-1 single
A 2-4-1 MLP processes 1 sample. What is the shape of $a_1$?
- A: $(2,)$
- B: $(4,)$
- C: $(1,)$
- D: $(2, 4)$

answer: B

> 4 hidden neurons → each outputs 1 number → $a_1$ is a 4-dim vector, shape $(4,)$.
> Note: single sample, so no batch dimension.
:::

::: quiz q6-2 single
Why apply sigmoid after $z_1$?
- A: Make numbers smaller
- B: Introduce nonlinearity (line becomes curve)
- C: Force 0/1 output
- D: Speed up computation

answer: B

> Without activation = stacked linear transforms = equivalent to 1 layer = can't solve XOR.
> Activation introduces **nonlinearity** = multi-layer matters.
:::

::: quiz q6-3 short placeholder="Explain the key role of activation function"
**Question**: If we replace sigmoid with the identity function $f(z) = z$, can the MLP still solve XOR? Why?

> Reference answer (not unique):
> No. **The identity function is linear** ($f(z) = z$ has no "nonlinearity"). If the hidden layer uses identity activation, then:
>
> Layer 1: $a_1 = z_1 = W_1 \cdot x + b_1$ (linear)
> Layer 2: $\hat{y} = z_2 = W_2 \cdot a_1 + b_2 = W_2 \cdot (W_1 \cdot x + b_1) + b_2 = (W_2 W_1) \cdot x + (W_2 b_1 + b_2)$ (still linear)
>
> Equivalent to a single-layer network → can't solve XOR.
>
> **Conclusion**: the "nonlinearity" of the activation function is the **fundamental reason** MLPs can learn complex functions.
:::

::: quiz q6-4 short placeholder="Hand-compute XOR sample (0, 0)"
Using the $W_1, b_1, W_2, b_2$ above, **hand-compute** XOR's 2nd sample $(0, 0)$ and give an approximate $\hat{y}$.
> Hint: $z_1 = W_1 \cdot [0, 0] + b_1 = b_1$, $a_1 = \sigma(b_1)$, $z_2 = W_2 \cdot a_1 + b_2$.

> Reference answer:
> Step 1: $z_1 = b_1 = [-0.1, 0.2, -0.1, 0.3]$
> Step 2: $a_1 = \sigma(z_1) \approx [0.475, 0.550, 0.475, 0.574]$
> Step 3: $z_2 = 0.5·0.475 + (-0.4)·0.550 + 0.6·0.475 + 0.3·0.574 + (-0.2)$
>           $\approx 0.2375 - 0.220 + 0.285 + 0.1722 - 0.2$
>           $\approx 0.275$
> Step 4: $a_2 = \sigma(0.275) \approx 0.568$... wait, XOR(0,0) truth is 0, so prediction should be close to 0
>
> This parameter set isn't perfect (it's just for teaching). The learned $W$ would give a more accurate result.
> The point is: **you walked through the 4 steps**.

---

## Summary

> 🎯 Forward pass = 4 steps: $X \cdot W_1 + b_1 \to \sigma \to a_1 \cdot W_2 + b_2 \to \sigma = \hat{y}$
> Key: the activation $\sigma$ is the source of nonlinearity — without it, MLP = single layer.
> Next chapter: how to **measure "how wrong"** — the loss function.
