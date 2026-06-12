# Chapter 10: Hand-coded MLP Solves XOR + Certificate

## 10.1 Complete numpy code (less than 50 lines)

```python
import numpy as np

# 1. Data: XOR
X = np.array([[0,0], [0,1], [1,0], [1,1]], dtype=np.float32)
Y = np.array([[0], [1], [1], [0]], dtype=np.float32)

# 2. Initialize parameters (random)
np.random.seed(42)
W1 = np.random.randn(2, 4) * 0.5
b1 = np.zeros((1, 4))
W2 = np.random.randn(4, 1) * 0.5
b2 = np.zeros((1, 1))

# 3. Activation
def sigmoid(z): return 1 / (1 + np.exp(-z))
def sigmoid_deriv(a): return a * (1 - a)

# 4. Training loop
lr = 0.5
for epoch in range(2000):
    # Forward
    z1 = X @ W1 + b1
    a1 = sigmoid(z1)
    z2 = a1 @ W2 + b2
    y_hat = sigmoid(z2)
    
    # Loss (BCE simplified)
    loss = -np.mean(Y * np.log(y_hat + 1e-9) + (1 - Y) * np.log(1 - y_hat + 1e-9))
    
    # Backward
    dz2 = y_hat - Y                       # BCE + σ simplification
    dW2 = a1.T @ dz2 / len(X)
    db2 = np.mean(dz2, axis=0, keepdims=True)
    dz1 = (dz2 @ W2.T) * sigmoid_deriv(a1)
    dW1 = X.T @ dz1 / len(X)
    db1 = np.mean(dz1, axis=0, keepdims=True)
    
    # Update
    W1 -= lr * dW1; b1 -= lr * db1
    W2 -= lr * dW2; b2 -= lr * db2

# 5. Test
print("Final predictions:")
print(np.round(y_hat, 3))
print("Loss:", round(loss, 4))
```

---

## 10.2 Training 2000 steps: watch loss drop

**Expected results**:
- Initial loss ≈ 0.69 (random)
- After 200 steps ≈ 0.30
- After 2000 steps ≈ 0.02

**Predictions** (XOR true = $[0, 1, 1, 0]$):

```
[[0.014]    ← (0,0) → close to 0 ✓
 [0.987]    ← (0,1) → close to 1 ✓
 [0.987]    ← (1,0) → close to 1 ✓
 [0.013]]   ← (1,1) → close to 0 ✓
```

**All 4 correct, loss ≈ 0.02** → training succeeded!

---

## 10.3 Test: 4 points all correct

| Input | True | Predicted | Correct? |
|---|---|---|---|
| (0, 0) | 0 | 0.014 | ✓ |
| (0, 1) | 1 | 0.987 | ✓ |
| (1, 0) | 1 | 0.987 | ✓ |
| (1, 1) | 0 | 0.013 | ✓ |

**100% accuracy**.

---

## 10.4 Decision boundary evolution

::: chart caption="1000 training steps: decision boundary from nothing to a curve"
![Decision boundary](assets/images/ch10_decision_boundary.png)
:::

**3 phases**:
- **Before training**: boundary is nearly a straight line (single-layer perceptron)
- **Mid training**: boundary starts to curve
- **After training**: boundary forms the "XOR shape" — enclosing (0,1) and (1,0)

---

## Summary: Neural network = a bunch of $w$ + a bunch of $\sigma$ + gradient descent

**The whole tutorial condensed**:

1. **Neuron** = weighted sum + activation $\sigma$
2. **Network** = many neurons in multiple layers
3. **Forward** = $X \cdot W_1 + b_1 \to \sigma \to a_1 \cdot W_2 + b_2 \to \sigma = \hat{y}$
4. **Loss** = BCE (measure wrongness)
5. **Backward** = chain rule to compute gradients
6. **Update** = $W \leftarrow W - \alpha \cdot \partial L / \partial W$
7. **Loop** = repeat steps 3-6

**What we skipped** (for further study):
- Convolutional networks (CNN, for images)
- Recurrent networks (RNN, for sequences)
- Transformers (attention mechanism)
- Adam / Momentum optimizers
- Dropout / BatchNorm (regularization)
- GPU acceleration / distributed training
- Self-supervised learning / reinforcement learning

**What you learned = the answer to AI interview question #1: hand-coding an MLP.** ✓

---

## 🎓 Issue Certificate

Complete all 10 chapters + 60% correct answers → **unlock the certificate**!

Click "Certificate" in the sidebar or the "Next Chapter" button to enter.

---

::: quiz q10-1 single
After training, the 2-4-1 MLP predicts close to (0, 1, 1, 0) for XOR's 4 points. This means?
- A: MLP successfully solved XOR
- B: MLP failed, should output all 0
- C: MLP failed, should output all 1
- D: Model is underfitting

answer: A

> Predictions (0, 1, 1, 0) match XOR's truth (0, 1, 1, 0) exactly → **XOR solved**.
> This is what MLP does that single-layer perceptron can't — multi-layer + nonlinear activation.
:::

::: quiz q10-2 multiple
In the 4-step training loop, what does the "backward" step compute? (Multiple choice)
- A: Model prediction $\hat{y}$
- B: Loss $L$
- C: Gradient $\partial L / \partial W$
- D: Bias gradient $\partial L / \partial b$

answer: C,D

> Backpropagation computes the **gradient of every parameter**: $\partial L / \partial W$ (C ✓) and $\partial L / \partial b$ (D ✓).
> A is forward; B is loss computation; backward is C+D.
:::

::: quiz q10-3 short placeholder="Summarize this tutorial"
Summarize in **2-3 sentences** the most important lesson of this tutorial.

> Reference answer (not unique):
> - "A neural network = a bunch of weights $w$ + activation $\sigma$ + gradient descent. Training is the loop: forward, backward, update. Simple, but can solve complex problems."
> - "Neural networks aren't a black box — they're matrix multiplication + sigmoid + chain rule. Hand-compute one 2-4-1 MLP backward pass, and you'll fully understand it."
> - "Hand-coding an MLP = entry ticket for AI jobs. XOR is the simplest test — solve XOR with a 2-4-1, and you've mastered all the basics of deep learning."
:::

---

## 🎉 Congratulations on completing the tutorial!

> If you've done all 32 questions with ≥ 60% correct:
> 1. Click "🌙/☀️" in the top right to confirm theme preference
> 2. Click "中文/English" to confirm language
> 3. Click "🎓 Certificate" in the sidebar
> 4. Enter your name, download the PNG certificate!

**To continue learning**:
- Book: *Deep Learning from Scratch* (Saito Koki)
- Course: [3Blue1Brown's Neural Network series](https://www.3blue1brown.com/topics/neural-networks)
- Practice: Kaggle starter competitions (Titanic / MNIST)

— Tutorial complete —
