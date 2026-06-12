# Chapter 1: What is a Neural Network?

> **This chapter has zero math.** We use a "decision spirit" story to introduce concepts. The formulas come in later chapters.

## 1.1 A neuron: weighted sum + activation

Imagine a **little spirit** making a decision:

- It hears **3 numbers** (e.g., temperature 30°, humidity 80%, is it weekend? 1)
- Each number has an **importance weight** (e.g., temperature is very important, weight 0.5; day of week less so, weight 0.1)
- The spirit "weighted-sums" all numbers
- Then a **switch function** decides: "go out" or "stay home"

**That's a neuron!** It does 2 things:
1. **Weighted sum**: $sum = w_1 \cdot x_1 + w_2 \cdot x_2 + w_3 \cdot x_3 + b$
2. **Activation**: $y = \sigma(sum)$ (σ squashes any number into 0~1)

::: chart caption="Structure of a single neuron"
![Neuron cartoon](assets/images/ch01_neuron_cartoon_en.png)
:::

**Simplified example** (imagine each number is "evidence", weight is "importance"):
- $x_1 = 30$ (temperature), $w_1 = 0.5$ → contributes 15
- $x_2 = 80$ (humidity), $w_2 = 0.2$ → contributes 16
- $x_3 = 1$ (weekend), $w_3 = 0.3$ → contributes 0.3
- $b = -20$ (the "threshold")
- sum = $15 + 16 + 0.3 - 20 = 11.3$
- $\sigma(11.3) \approx 0.9999$ (strongly "go out")

---

## 1.2 Networks: many neurons connected

One neuron can only make a simple "yes/no" decision. **Complex decisions need many neurons "voting"**.

For example, deciding "is this image a cat or a dog":
- Row 1 neurons: detect "ear shape" (pointed? round?)
- Row 2 neurons: detect "eye size"
- Row 3 neurons: detect "whiskers"
- ...
- Final neuron: combines all info, outputs "cat" or "dog"

**Each row = a layer**. Many layers stacked = **neural network**.

**Intuition**:
- Layer 1 learns "low-level features" (edges, colors)
- Middle layers learn "mid-level features" (ears, eyes)
- Final layer learns "high-level concepts" ("cat")

---

## 1.3 Neural networks vs. the brain

**Important clarification**: Neural networks are **not** copies of the brain.

| Dimension | Brain Neuron | Artificial Neuron |
|---|---|---|
| Count | ~86 billion | Hundreds of millions (largest) |
| Signal | Electrochemical pulse | Numbers (0~1) |
| Connection | Dynamic growth | Fixed values |
| Learning | Synaptic plasticity | Gradient descent |
| Speed | ~milliseconds | ~nanoseconds |

**Their relationship**:
- Brain neurons inspired artificial neurons (McCulloch & Pitts, 1943)
- But they work **very differently**
- Artificial neural networks are **mathematical models**, not biological ones

::: quiz q1-1 single
What two things does a neuron essentially do?
- A: Weighted sum + activation
- B: Matrix multiplication + pooling
- C: Convolution + fully connected
- D: Sort + average

answer: A

> Neuron = weighted sum + activation. These two operations form the "fundamental particle" of all neural networks.
:::

::: quiz q1-2 multiple
What similarities do neurons and brain cells have? (Multiple choice)
- A: Both have multiple input signals
- B: Both can "learn"
- C: Same structure
- D: Both use electrical signals

answer: A,B

> Similar: inputs and learning (A, B ✓).
> Different: structure differs (C ✗); brain uses electrochemical pulses, not just electrical (D ✗).
:::

---

## Summary

> 🎯 **The most important takeaway**: a neural network = many "weighted sum + activation" decision-makers, connected to make complex decisions.
>
> Next chapter, we translate "the spirit" into math.
