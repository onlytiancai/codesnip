# Chapter 9: Gradient Descent + Training Loop

## 9.1 Gradient descent: walking down the steepest slope

**Gradient descent** = the core optimization algorithm.

**Basic idea**:
- Loss $L$ is a function of $W$ (a "mountain")
- Gradient $\partial L / \partial W$ = the "steepness" and "direction" of the slope
- Walk one step in the negative gradient direction → $L$ decreases → optimized

**Update formula**:
$$
W \leftarrow W - \alpha \cdot \frac{\partial L}{\partial W}
$$

- $\alpha$ (alpha) = learning rate = step size
- Gradient $\partial L / \partial W$ is computed by backprop
- Each step, $W$ updates once

::: chart caption="Gradient descent: walking down the negative gradient"
![Gradient descent](assets/images/ch09_gradient_descent.png)
:::

---

## 9.2 Learning rate: step size

**Effect of learning rate $\alpha$**:

| $\alpha$ | Effect |
|---|---|
| Too small | Super slow convergence, stuck on "halfway up the mountain" |
| Too large | "Swinging" near the optimum, divergence |
| Just right | Smooth descent to the valley |

**How to choose $\alpha$?**
- Rule of thumb: start with 0.1
- Watch the loss curve: ↓ steady = good; ↓ oscillating = too big; ↓ not moving = too small
- Modern optimizers (Adam) adapt $\alpha$ automatically

---

## 9.3 Training loop: forward → loss → backward → update

**One complete epoch**:

```python
for epoch in range(num_epochs):
    for x_batch, y_batch in dataloader:
        # 1. Forward
        y_pred = model(x_batch)
        # 2. Compute loss
        loss = criterion(y_pred, y_batch)
        # 3. Backward
        loss.backward()
        # 4. Update (gradient descent)
        optimizer.step()
        # 5. Zero gradients
        optimizer.zero_grad()
```

**Key points**:
- Each epoch processes the **entire dataset** once
- All 4 steps are **essential** (training fails if any is missing)
- `zero_grad()` before backward, otherwise gradients accumulate

---

## 9.4 Loss curve & decision boundary: visualizing "learning"

**Training 200 steps** (using the 2-4-1 MLP + BCE loss from Chapter 6):

::: train-demo :steps=200 :lr=0.5
Train a 2-4-1 MLP to solve XOR. Click "Start training" to see the loss curve drop and accuracy rise.
:::

**Key observations**:
- **Loss curve**: drops from 0.7 (random) to near 0
- **Accuracy curve**: rises from 50% to 100%
- **Decision boundary**: goes from "almost random" to "a curve" (Chapter 10 details)

::: chart caption="200 training steps: Loss ↓, Accuracy ↑"
![Loss curve](assets/images/ch09_loss_curve.png)
:::

---

::: quiz q9-1 single
In the gradient descent update $W \leftarrow W - \alpha \cdot \frac{\partial L}{\partial W}$, what role does $\alpha$ play?
- A: The gradient itself
- B: Step size
- C: Loss value
- D: Bias

answer: B

> $\alpha$ = learning rate = step size. Controls how far to go each step.
:::

::: quiz q9-2 single
During training, loss stays at 0.69 forever (for binary classification). Most likely cause?
- A: Learning rate too large
- B: Learning rate too small
- C: Network outputs all 0.5 (learned nothing)
- D: Dataset too large

answer: C

> Loss = 0.69 = $-\log(0.5)$ = random guessing level. Network isn't learning at all.
> Causes: wrong model structure / reversed labels / training broken.
> Debug: print intermediate values, try a simpler model.
:::

---

## Summary

> 🎯 Training = 4-step loop (forward → loss → backward → update). **Learning rate $\alpha$** is the most important hyperparameter.
> Next chapter: tie it all together — hand-code a complete MLP to solve XOR, and issue the certificate.
