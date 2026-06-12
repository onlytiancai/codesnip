# Chapter 5: MLP — Stacking Neurons into a Network

> **Back-link**: Uses the "matrix" concept from [0.5](#05-matrix-and-matrix-multiplication-the-magic-of-tables).

## 5.1 Topology: 2 → 4 → 1

**MLP** (Multi-Layer Perceptron) = multiple perceptrons stacked into layers.

**Simplest MLP**: 2 → 4 → 1
- **Input layer**: 2 neurons ($x_1, x_2$)
- **Hidden layer**: 4 neurons ($h_1, h_2, h_3, h_4$)
- **Output layer**: 1 neuron ($\hat{y}$)
- **Fully connected**: every neuron in one layer connects to every neuron in the next

::: chart caption="2 → 4 → 1 MLP topology"
![MLP topology](assets/images/ch05_mlp_topology_en.png)
:::

**Parameter count**:
- $W_1$: 4×2 = 8 weights
- $b_1$: 4×1 = 4 biases
- $W_2$: 1×4 = 4 weights
- $b_2$: 1×1 = 1 bias
- **Total: 17 parameters**

---

## 5.2 "Two lines" combined = a curve

**Why can MLP solve XOR?**

**Intuition**:
- 1 perceptron can only draw **one line**
- 4 perceptrons can draw **4 lines**
- Combining 4 lines (OR) = **a curve**

**Implementation**:
- Hidden layer's 4 neurons each draw a line
- Output layer ORs them together
- The union of 4 lines can enclose XOR's "on" region

::: chart caption="Combination of 4 hidden lines = a curve (XOR region)"
![Two lines compose](assets/images/ch05_two_lines_compose.png)
:::

---

## 5.3 What the hidden layer does: automatic feature learning

**What does the hidden layer "hide"?**
- Its input is "the previous layer's output" — training data doesn't directly tell it what to compute
- But what it learns are "intermediate features" that help solve the problem

**Example** (handwritten digit recognition):
- Input layer: 784 pixel values
- Hidden layer 1: learns "edges" (horizontal, vertical, diagonal)
- Hidden layer 2: learns "parts" (circles, line intersections)
- Hidden layer 3: learns "concepts" (the shape of "0", the shape of "8")
- Output layer: decides which digit (0~9)

**Automatic feature engineering**:
- Old way: hand-design features (SIFT, HOG) → slow, expert-dependent
- Neural networks: **learn** features → fast, no expert needed
- This is the core advantage of deep learning

::: quiz q5-1 single
How many layers does a 2 → 4 → 1 MLP have?
- A: 2
- B: 3
- C: 4
- D: 5

answer: B

> The "2 → 4 → 1" is 3 segments: input + hidden + output = 3 layers.
> Note: only 2 weight matrices (W1, W2), because the input layer is just a placeholder. But "how many layers" usually counts the input layer.
:::

::: quiz q5-2 multiple
What advantages does MLP have over single-layer perceptron? (Multiple choice)
- A: Can learn XOR
- B: Can learn curved boundaries
- C: Faster training
- D: Hidden layer automatically finds features

answer: A,B,D

> A ✓: can learn XOR (multi-layer + activation).
> B ✓: can learn curved boundaries.
> C ✗: training is **slower** — more parameters, gradients to compute.
> D ✓: hidden layer auto-feature-learning (core deep learning advantage).
:::

::: quiz q5-3 short placeholder="Explain why it's called a hidden layer"
Explain: **why is it called a "hidden" layer?** Are its inputs/outputs visible to the training data?

> Reference answer (not unique):
> The "hidden" in "hidden layer" means — its output is **not directly the data label**, but the "intermediate features" it has learned. Training data has two visible pieces of info: "input" and "label". The hidden layer sits in the middle, inventing its own "internal language".
> For example, recognizing cats: training data has "image" (input) and "is it a cat" (label), but no "shape of cat's ear" label — that label is something the hidden layer **learns by itself**.

---

## Summary

> 🎯 MLP = multiple perceptrons stacked + activation. **Key capability**: curved boundaries (XOR), automatic feature learning.
> Next chapter: how MLP **computes** predictions — **forward propagation**.
