# Chapter 4: The XOR Problem — Where Perceptrons Fail

## 4.1 The XOR truth table: 4 points on a diagonal

**XOR** (exclusive or): output 1 when inputs **differ**, 0 when **same**.

| $x_1$ | $x_2$ | $y$ |
|---|---|---|
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |

Visually: 4 points on a diagonal. $(0,0)$ and $(1,1)$ are "off" (y=0); $(0,1)$ and $(1,0)$ are "on" (y=1).

---

## 4.2 Geometric intuition: no single line can separate

::: chart caption="XOR: 3 attempted lines all fail"
![XOR](assets/images/ch04_xor_scatter.png)
:::

::: perceptron-playground
:::

**How can the 4 points be split?** Drag the sliders to move the line—
- **AND**: put (1,1) in the bright region, the other 3 in the dark region
- **OR**: put (0,0) in the dark region, the other 3 in the bright region
- **XOR**: try it — can you classify all 4 points correctly?

**Try**:
- A line with positive slope? $(0,1)$ and $(1,0)$ on one side, but $(0,0)$ and $(1,1)$ end up on the same side ✗
- A line with negative slope? Same problem ✗
- Any line? Always wrong ✗

**Conclusion**: **XOR is not linearly separable**. The perceptron can **never** learn XOR.

---

## 4.3 Historical context: Minsky and the AI winter

**1969**: AI godfather **Marvin Minsky** and **Seymour Papert** published *Perceptrons*, **rigorously proving** that single-layer perceptrons can't solve XOR.

The book's impact:
- Before 1969, perceptrons were widely seen as promising
- After 1969, **AI field entered a winter** (1970~1986)
- 17 years with almost no neural network research
- **1986**: Hinton et al. proposed **backpropagation** (Chapter 8), and multi-layer perceptrons (MLP) could solve XOR — neural networks "revived"

**Minsky's limitation**:
- He proved that **single-layer** perceptrons can't solve XOR
- But **multi-layer** perceptrons (MLP) can!
- Minsky's book also mentioned "just add a layer" — but **no algorithm existed to train multi-layer networks at the time**
- Backpropagation appeared in 1986

**Key timeline**:
- 1943: McCulloch & Pitts propose the neuron math model
- 1958: Rosenblatt invents the perceptron
- 1969: Minsky's *Perceptrons* → AI winter begins
- 1986: Hinton et al. propose backpropagation → neural networks revive
- 2012: AlexNet dominates ImageNet → deep learning explodes

::: quiz q4-1 single
Why is XOR not separable?
- A: Formula error
- B: Data error
- C: No line can separate the diagonal pattern
- D: Learning rate too small

answer: C

> XOR's 4 points lie on a diagonal: $(0,0), (1,1)$ on one side (y=0); $(0,1), (1,0)$ on the other (y=1). Any line passes through both groups.
:::

::: quiz q4-2 multiple
What did Minsky's *Perceptrons* (1969) actually say? (Multiple choice)
- A: Single-layer perceptrons can't solve XOR
- B: Multi-layer perceptrons can, but no training algorithm existed
- C: Neural networks are useless forever
- D: All of AI should be abandoned

answer: A,B

> A ✓: proved single-layer perceptrons can't solve XOR (math fact).
> B ✓: mentioned multi-layer can, but no backprop algorithm then.
> C ✗: Minsky never said "forever useless" — only pointed out current limitations.
> D ✗: Minsky didn't oppose all of AI.
:::

::: quiz q4-3 short placeholder="Explain in 1 sentence to a friend"
Explain to a friend in **one sentence**: why does XOR stump the perceptron?

> Reference answer (not unique):
> - "XOR's 4 points are on a diagonal — no matter how you draw a straight line, both classes are on the same side."
> - "XOR requires 'different = 1' — that rule is not the job of a straight line."
> - "A perceptron only draws straight lines to separate two sides; XOR's boundary is curved, so a straight line can't do it."
:::

---

## Summary

> 🎯 Single-layer perceptrons can only solve **linearly separable** problems. XOR shows us: **we need multiple layers**.
> Next chapter: stack multiple neurons to see how XOR can be solved.
