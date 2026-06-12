# 第 8 章 反向传播：从错到对的"归因"

> **回链**：本章重点用 0.6 的"链式法则 + 偏导"（[第 0 章 0.6 节](#06-偏导--链式法则嵌套机器的信号)）。

## 8.1 链式法则：嵌套机器的灵敏度

**反向传播**（backpropagation）= 用**链式法则**从损失 L 反推到 W1。

**为什么需要它？**
- 前向：x → z1 → a1 → z2 → a2 = ŷ → L（计算损失）
- 反向：L → ... → a2 → ... → z1 → ... → W1（计算每个 W 对 L 的"影响"）
- 知道了"影响"才能"调整 W 让 L 减小"

**链式法则复习**（0.6 节）：
- $\frac{\partial L}{\partial W_1} = \frac{\partial L}{\partial z_2} \cdot \frac{\partial z_2}{\partial a_1} \cdot \frac{\partial a_1}{\partial z_1} \cdot \frac{\partial z_1}{\partial W_1}$
- 4 个偏导**相乘** = A 对 D 的影响 = A→B × B→C × C→D

---

## 8.2 从 L 一路反推到 W1（4 个公式）

**2-4-1 MLP 的反向传播**（单样本，BCE 损失）：

**Step 1**：L 对 a2 的偏导（"a2 错了多少"）
$$
\frac{\partial L}{\partial a_2} = \frac{a_2 - y}{a_2 (1 - a_2)}
$$

**Step 2**：a2 对 z2 的偏导（"z2 变化时 a2 怎么变"）= σ 的导数
$$
\frac{\partial L}{\partial z_2} = \frac{\partial L}{\partial a_2} \cdot \sigma'(z_2) = (a_2 - y)
$$
> **注意**：BCE + σ 的组合让 σ 导数被约掉，得到漂亮简洁的形式 $a_2 - y$。

**Step 3**：z2 对 W2 的偏导（"W2 变化时 z2 怎么变"）= a1
$$
\frac{\partial L}{\partial W_2} = \frac{\partial L}{\partial z_2} \cdot a_1^T = (a_2 - y) \cdot a_1^T
$$

**Step 4**：z2 对 a1 的偏导（"a1 变化时 z2 怎么变"）= W2
$$
\frac{\partial L}{\partial a_1} = W_2^T \cdot \frac{\partial L}{\partial z_2} = W_2^T \cdot (a_2 - y)
$$

**Step 5**：a1 对 z1 的偏导
$$
\frac{\partial L}{\partial z_1} = \frac{\partial L}{\partial a_1} \cdot \sigma'(z_1)
$$

**Step 6**：z1 对 W1 的偏导 = x
$$
\frac{\partial L}{\partial W_1} = \frac{\partial L}{\partial z_1} \cdot x^T
$$

**更新权重**（梯度下降）：
$$
W_1 \leftarrow W_1 - \alpha \cdot \frac{\partial L}{\partial W_1}, \quad W_2 \leftarrow W_2 - \alpha \cdot \frac{\partial L}{\partial W_2}
$$

---

## 8.3 手算一遍：2-4-1 MLP 反向算 XOR

**承第 6 章**：预测 ŷ ≈ 0.558，XOR 真值 y=1。

**Step 1**：$\frac{\partial L}{\partial z_2} = a_2 - y = 0.558 - 1 = -0.442$

**Step 2**：$a_1 = [0.646, 0.731, 0.426, 0.768]$
$\frac{\partial L}{\partial W_2} = -0.442 \cdot [0.646, 0.731, 0.426, 0.768] = [-0.286, -0.323, -0.188, -0.339]$
（每个 W2 元素一个偏导）

**Step 3**：$\frac{\partial L}{\partial a_1} = W_2^T \cdot (-0.442)$
$W_2 = [0.5, -0.4, 0.6, 0.3]$
$\frac{\partial L}{\partial a_1} = [-0.221, 0.177, -0.265, -0.133]$

**Step 4**：$\sigma'(z_1) = \sigma(z_1) \cdot (1 - \sigma(z_1)) = a_1 \cdot (1 - a_1)$
$= [0.646·0.354, 0.731·0.269, 0.426·0.574, 0.768·0.232]$
$= [0.229, 0.197, 0.244, 0.178]$

$\frac{\partial L}{\partial z_1} = \frac{\partial L}{\partial a_1} \cdot \sigma'(z_1) = [-0.0506, 0.0349, -0.0647, -0.0237]$

**Step 5**：$x = [0, 1]$
$\frac{\partial L}{\partial W_1} = \frac{\partial L}{\partial z_1} \cdot x^T$（外积）
$$
\frac{\partial L}{\partial W_1} = \begin{bmatrix} -0.0506 \\ 0.0349 \\ -0.0647 \\ -0.0237 \end{bmatrix} \cdot [0, 1] = \begin{bmatrix} 0 & -0.0506 \\ 0 & 0.0349 \\ 0 & -0.0647 \\ 0 & -0.0237 \end{bmatrix}
$$

**更新**（α=0.1）：$W_1^{new} = W_1 - 0.1 \cdot \frac{\partial L}{\partial W_1}$

看：第 1 列（W1 的 x1 列）没变（因为 x1=0），第 2 列（W1 的 x2 列）变了一点——**x1=0 不贡献梯度**，这跟 006.md 提到的"w[0] 一动没动"是同一个现象。

::: chart caption="反向传播：信号从 L 反向流到 W1"
![反向传播](assets/images/ch08_chain_rule.png)
:::

---

## 8.4 验证：手算 vs PyTorch autograd

**用 PyTorch 验证**（如果你有环境）：

```python
import torch
import torch.nn as nn

torch.manual_seed(42)
model = nn.Sequential(nn.Linear(2, 4), nn.Sigmoid(), nn.Linear(4, 1), nn.Sigmoid())
x = torch.tensor([[0.0, 1.0]])
y = torch.tensor([[1.0]])
loss = nn.BCELoss()(model(x), y)
loss.backward()
print("dL/dW1 =", model[0].weight.grad)
print("dL/dW2 =", model[2].weight.grad)
```

**应该跟我们手算的结果一致**（在浮点精度范围内）。

::: quiz q8-1 single
反向传播的本质是？
- A: 反向跑一遍数据
- B: 链式法则 + 梯度下降
- C: 重新初始化
- D: 随机猜测

answer: B

> 反向传播 = 链式法则（算梯度）+ 梯度下降（更新权重）。两个一起，缺一不可。
:::

::: quiz q8-2 single
学习率 $\alpha = 0.1$，跑了 1000 步，loss 还在震荡。可能的原因是？
- A: 数据错了
- B: 学习率太大或网络结构不对
- C: batch 太大
- D: 标签反了

answer: B

> Loss 震荡 = 步子太大跳过了最优点。最常见原因：学习率 α 太大。
> 解决方法：α 调小（比如 0.01）；或者用 Adam 这种自适应优化器。
> 注意：batch 太大反而让 loss **更稳**（平均掉了噪声）。
:::

::: quiz q8-3 multiple
反向传播的步骤有？（多选）
- A: 前向算损失
- B: 反向算梯度
- C: 用梯度更新权重
- D: 删除一半神经元

answer: A,B,C

> 标准流程：A → B → C → 重复。
> A：x 输入，算了 ŷ 和损失 L。
> B：从 L 一路链式法则反推到每个 W。
> C：$W \leftarrow W - \alpha \cdot \frac{\partial L}{\partial W}$。
> D ✗：删除神经元是 dropout 做的事，不是反向传播。
:::

::: quiz q8-4 short placeholder="用 1 句话告诉 6 岁的弟弟"
请用**一句话**向你 6 岁的弟弟解释：什么是"训练神经网络"？

> 参考答案（不唯一）：
> - "训练神经网络就是'猜 → 看错多少 → 调整猜法'，猜得越多次就越准"
> - "就像你学骑车——先试着骑（猜），看会不会摔（算错），摔了就调整姿势（改权重），慢慢就会骑了"
> - "训练就是'考试 + 改错'——做一道题（猜），对答案（算错），把错题改正（调整）"
:::

---

## 小结

> 🎯 反向传播 = 链式法则从 L 反推到 W1。BCE + σ 的组合让 $dz_2 = a_2 - y$ 极简。
> 下一章我们看怎么**循环**前向+反向，构成训练循环。
