# Math Prep: 6 Tools You'll Need

> This chapter is an **independent toolbox**. Each later chapter links back to it when needed. Read it once; don't worry about understanding everything deeply — impressions are enough.

We only need 6 tools to understand neural networks:

| Section | Tool | Purpose |
|---|---|---|
| 0.1 | Function | Input → Output |
| 0.2 | Coordinate System | Plot numbers on a map |
| 0.3 | Slope | How "steep" is a function |
| 0.4 | Vector | A column of numbers describing an object |
| 0.5 | Matrix | Tables + "batch math" |
| 0.6 | Partial Derivative + Chain Rule | Sensitivity in nested machines |

---

## 0.1 Function: Throw in a number, spit out a number

> **Analogy: a vending machine.** Drop in a coin (input), press a button (formula $f$), and out comes a drink (output).

::: chart caption="A function is a machine: number in, number out"
![Function machine](assets/images/ch00_function_machine.png)
:::

**Form**: $y = f(x)$, read "y equals f of x".

**Example**: $f(x) = 2x + 1$ (double then add 1)

| Input $x$ | Calculation | Output $y$ |
|---|---|---|
| 0 | $2·0 + 1$ | 1 |
| 1 | $2·1 + 1$ | 3 |
| 3 | $2·3 + 1$ | 7 |

**Key concepts**:
- **Domain**: all possible inputs
- **Range**: all possible outputs
- **Independent vs dependent variable**: $x$ is independent (it varies freely); $y$ depends on $x$

::: quiz q0-1-1 single
Which of the following is **not** a valid function?
- A: $f(x) = 3x + 2$
- B: $f(x) = x^2$ (square)
- C: $f(x) = $ "if x is even return 0, else 1" (a "patched" function)
- D: $f(x) = $ "x is a number, return x+1; otherwise throw an error"

answer: D

> A function must produce a **unique** output for every input. Option D doesn't define an output for non-number inputs (throwing an error is not an output), so it isn't a complete function.
:::

::: quiz q0-1-2 single
$f(x) = 3x - 5$, find $f(4)$.
- A: 7
- B: 12
- C: -7
- D: 5

answer: A

> Substitute: $3 \times 4 - 5 = 12 - 5 = 7$. Order of operations: multiplication before subtraction.
:::

---

## 0.2 Coordinate System: Plotting numbers on a map

> **Analogy: a movie theater seat.** Row number (x-axis) and column number (y-axis) — two numbers locate a seat.

::: chart caption="Cartesian coordinate system: x and y axes"
![Coordinate system](assets/images/ch00_coordinate_grid.png)
:::

**Cartesian coordinate system**:
- **x-axis**: positive to the right
- **y-axis**: positive upward
- **Origin $(0, 0)$**: where the two axes meet
- **Point $(a, b)$**: x-coordinate $a$, y-coordinate $b$
- **4 quadrants**: top-right (+,+), top-left (-,+), bottom-left (-,-), bottom-right (+,-)

**Why do we need a coordinate system?**
Because a function $f(x) = y$ is a "input → output" relationship. Plotting all $(x, y)$ in a coordinate system lets you see the function's "shape".

::: quiz q0-2-1 single
Which quadrant is the point $(3, -2)$ in?
- A: First quadrant (top-right)
- B: Second quadrant (top-left)
- C: Third quadrant (bottom-left)
- D: Fourth quadrant (bottom-right)

answer: D

> $x=3 > 0$ (right), $y=-2 < 0$ (down), so it's bottom-right → fourth quadrant.
:::

---

## 0.3 Slope: How steep is a function at a point

> **Analogy: a car's speedometer.** A speedometer doesn't show position; it shows "how fast position is changing". The derivative = "how fast you're going at a given instant".

::: chart caption="f(x) = x² and its tangent line at x=1.5 (slope=3)"
![Slope](assets/images/ch00_slope_tangent.png)
:::

**What is slope**:
- Slope = "how much $y$ changes" / "how much $x$ changes"
- The line $y = mx + b$ has slope $m$
- Positive slope: function goes up to the right; negative: down

**Example**:
- $y = 2x + 1$: slope $m=2$, each step right, $y$ goes up 2
- $y = -3x + 5$: slope $m=-3$, each step right, $y$ goes down 3

**Slope at a point (= derivative)**:
- $f(x) = x^2$, slope $f'(x) = 2x$
- $f(3) = 9$, $f'(3) = 6$ (at $x=3$, the function rises at rate 6)

::: quiz q0-3-1 single
What is the slope of the line $y = 2x - 1$?
- A: 2
- B: -1
- C: 1
- D: 0.5

answer: A

> In $y = mx + b$, the slope is $m$. Here $m=2$.
:::

---

## 0.4 Vector: A column of numbers describing an object

> **Analogy: a personal profile card.** Height, weight, age, grade — 4 numbers describe a person.

::: chart caption="Vector: an ordered column of numbers"
![Vector](assets/images/ch00_vector_personal.png)
:::

**Vector definition**:
- **An ordered collection of numbers**
- Written as $[a, b, c]$
- **Dimensionality** = how many numbers

**Examples**:
- $[165, 50, 12, 92]$: 4-dimensional vector describing "Xiao Ming"
- $[3, 5]$: 2-dimensional vector (can be plotted on a coordinate system)
- $[28, 28, 1]$: shape of a handwritten digit image (28×28 pixels, 1 channel)

**Why do neural networks need vectors?**
- The "input" of a neural network is a vector
- An image = 784-dimensional vector (28×28 pixels flattened)
- A student's profile = 4-dimensional vector

::: quiz q0-4-1 single
What dimension is $[3, 5, 7, 9]$?
- A: 3
- B: 4
- C: 5
- D: 9

answer: B

> Count the numbers: 3, 5, 7, 9 — 4 numbers, so it's 4-dimensional.
:::

---

## 0.5 Matrix and Matrix Multiplication: The magic of tables

> **Analogy 1: an Excel spreadsheet.** Each row is a student, each column is a subject.
> **Analogy 2: "batch math".** An invoice × a tax rate = the actual amount paid.

::: chart caption="Matrix multiplication: (2×2) × (2×1) = (2×1)"
![Matrix multiplication](assets/images/ch00_matrix_multiplication.png)
:::

**Matrix shape**:
- (rows × columns): 2 rows, 3 columns = $(2, 3)$
- **Each row** is a vector, **each column** is also a vector

**Matrix multiplication rules**:
- **Shape matching**: left matrix's column count = right matrix's row count
- Result shape: (left rows, right columns)
- **Each cell = left's row · right's column** (dot product = sum of element-wise products)

**Example**: $(2×2) \times (2×1) = (2×1)$

$$
\begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix} \times \begin{bmatrix} 5 \\ 6 \end{bmatrix} = \begin{bmatrix} 1·5+2·6 \\ 3·5+4·6 \end{bmatrix} = \begin{bmatrix} 17 \\ 39 \end{bmatrix}
$$

**Why do neural networks need matrices?**
- Batch data (multiple samples) is a matrix: each row is a sample
- Weight $W_1$ = matrix: each row connects to a neuron in the next layer
- The whole forward pass = a chain of matrix multiplications + activations

::: quiz q0-5-1 single
(2 × 3) matrix × (3 × 4) matrix = ?
- A: (2, 4)
- B: (3, 3)
- C: (2, 3)
- D: (3, 4)

answer: A

> Left matrix: 2 rows, 3 columns. Right matrix: 3 rows, 4 columns.
> Result: (left rows, right columns) = (2, 4).
> Check: left cols (3) = right rows (3) ✓
:::

---

## 0.6 Partial Derivative + Chain Rule: Signal in nested machines

> **Analogy: an airplane's 3 knobs** (throttle, rudder, elevator). Each knob affects "altitude" but with different sensitivity. **Partial derivative** = sensitivity of one knob.
>
> **Chain rule**: A affects B, B affects C → A's effect on C = (A→B's effect) × (B→C's effect).

::: chart caption="Chain rule: signal flows backward through nested functions"
![Chain rule](assets/images/ch00_chain_rule.png)
:::

**Partial derivative**: $\partial f / \partial x$ reads "f's partial derivative with respect to x" — hold all other variables fixed, see how $f$ changes as $x$ changes.

**Example**: $f(x, y) = 3x + 5y$
- $\partial f / \partial x = 3$ (hold $y$, $x$ increases by 1, $f$ increases by 3)
- $\partial f / \partial y = 5$

**Chain rule**: $(f \circ g)'(x) = f'(g) \cdot g'(x)$

**Example**: $f(g) = g^2$, $g(x) = 2x + 1$, find $(f \circ g)'(x)$
- $f'(g) = 2g$, $g'(x) = 2$
- $(f \circ g)'(x) = 2(2x+1) \times 2 = 8x + 4$

**Why do neural networks need it?**
- A neural network is a nested function: $x \to z_1 \to a_1 \to z_2 \to a_2 = \hat{y}$
- Backpropagation = tracing from the loss $L$ of $\hat{y}$ back to the first weight $W_1$
- Each tracing step uses the chain rule

::: quiz q0-6-1 single
$f(x, y) = 4x + 7y$. Find $\partial f / \partial x$.
- A: 4
- B: 7
- C: 11
- D: 0

answer: A

> Hold $y$ fixed, only $x$ varies. $f = 4x + (\text{stuff not depending on } x)$, so $\partial f / \partial x = 4$.
:::

---

## Summary

> 🎯 **These 6 tools are all the math you'll need to understand neural networks.** If you didn't fully understand them on the first read, don't worry — the later chapters will use them repeatedly, deepening your understanding each time.

**Now back to the main course**: in the next chapter, we meet the neural network.
