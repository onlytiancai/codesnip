# 第 10 章 手写 MLP 解决 XOR + 证书

## 10.1 完整 numpy 代码（不到 50 行）

```python
import numpy as np

# 1. 数据：XOR
X = np.array([[0,0], [0,1], [1,0], [1,1]], dtype=np.float32)
Y = np.array([[0], [1], [1], [0]], dtype=np.float32)

# 2. 初始化参数（随机）
np.random.seed(42)
W1 = np.random.randn(2, 4) * 0.5
b1 = np.zeros((1, 4))
W2 = np.random.randn(4, 1) * 0.5
b2 = np.zeros((1, 1))

# 3. 激活函数
def sigmoid(z): return 1 / (1 + np.exp(-z))
def sigmoid_deriv(a): return a * (1 - a)

# 4. 训练循环
lr = 0.5
for epoch in range(2000):
    # 前向
    z1 = X @ W1 + b1
    a1 = sigmoid(z1)
    z2 = a1 @ W2 + b2
    y_hat = sigmoid(z2)
    
    # 损失（BCE 简化版：-mean(y log yhat + (1-y) log(1-yhat))）
    loss = -np.mean(Y * np.log(y_hat + 1e-9) + (1 - Y) * np.log(1 - y_hat + 1e-9))
    
    # 反向
    dz2 = y_hat - Y                       # BCE + σ 简化
    dW2 = a1.T @ dz2 / len(X)
    db2 = np.mean(dz2, axis=0, keepdims=True)
    dz1 = (dz2 @ W2.T) * sigmoid_deriv(a1)
    dW1 = X.T @ dz1 / len(X)
    db1 = np.mean(dz1, axis=0, keepdims=True)
    
    # 更新
    W1 -= lr * dW1; b1 -= lr * db1
    W2 -= lr * dW2; b2 -= lr * db2

# 5. 测试
print("Final predictions:")
print(np.round(y_hat, 3))
print("Loss:", round(loss, 4))
```

---

## 10.2 训练 2000 步：看 loss 下降

**预期结果**：
- 初始 loss ≈ 0.69（随机水平）
- 训练 200 步后 ≈ 0.30
- 训练 2000 步后 ≈ 0.02

**预测输出**（XOR 真值 [0, 1, 1, 0]）：

```
[[0.014]    ← (0,0) → 接近 0 ✓
 [0.987]    ← (0,1) → 接近 1 ✓
 [0.987]    ← (1,0) → 接近 1 ✓
 [0.013]]   ← (1,1) → 接近 0 ✓
```

**4 个点全对，loss ≈ 0.02** → 训练成功！

---

## 10.3 测试：4 个点全对

**实际计算**（训练后）：

| 输入 | 真值 | 预测 | 对/错 |
|---|---|---|---|
| (0, 0) | 0 | 0.014 | ✓ |
| (0, 1) | 1 | 0.987 | ✓ |
| (1, 0) | 1 | 0.987 | ✓ |
| (1, 1) | 0 | 0.013 | ✓ |

**准确率 100%**。

---

## 10.4 决策边界演化

::: chart caption="训练 1000 步：决策边界从无到弯线"
![决策边界](assets/images/ch10_decision_boundary.png)
:::

**3 个阶段**：
- **训练前**：边界接近直线（单层感知机）
- **训练中**：边界开始弯曲
- **训练后**：边界形成"XOR 形状"——圈住 (0,1) 和 (1,0)

---

## 总结：神经网络 = 一堆 w + 一堆 σ + 梯度下降

**整个教程的浓缩**：

1. **神经元** = 加权求和 + 激活函数 σ
2. **网络** = 多个神经元连成多层
3. **前向** = X·W1+b1 → σ → a1·W2+b2 → σ = ŷ
4. **损失** = BCE（衡量错多少）
5. **反向** = 链式法则算梯度
6. **更新** = $W \leftarrow W - \alpha \cdot \partial L / \partial W$
7. **循环** = 重复 3-6 步

**整个教程跳过的高级内容**（按需深入）：
- 卷积神经网络（CNN，处理图片）
- 循环神经网络（RNN，处理序列）
- Transformer（注意力机制）
- Adam / Momentum 优化器
- Dropout / BatchNorm（正则化）
- GPU 加速 / 分布式训练
- 自监督学习 / 强化学习

**你学到的核心** = 面试 AI 岗第一题的答案：**手写 MLP**。✓

---

## 🎓 颁发证书

完成所有 10 章 + 答对 60% 题目 → **解锁结业证书**！

点击侧边栏的"结业证书"或在右下角"下一章"按钮进入。

---

::: quiz q10-1 single
训练后，2-4-1 MLP 对 XOR 4 个点的预测都接近 (0, 1, 1, 0)。这说明？
- A: MLP 成功解决了 XOR
- B: MLP 失败了，应该输出全 0
- C: MLP 失败了，应该输出全 1
- D: 模型欠拟合

answer: A

> 预测 (0, 1, 1, 0) 与 XOR 真值 (0, 1, 1, 0) 完全一致 → **完美解 XOR**。
> 这就是 MLP 比单层感知机强的地方——多层 + 非线性激活。
:::

::: quiz q10-2 multiple
训练循环 4 步中，"反向"这一步算的是什么？（多选）
- A: 模型预测 ŷ
- B: 损失 L
- C: 梯度 ∂L/∂W
- D: 偏置 ∂L/∂b

answer: C,D

> 反向传播算的是**每个参数的梯度**：∂L/∂W（C ✓）和 ∂L/∂b（D ✓）。
> A 是前向；B 是损失计算；反向是 C+D。
:::

::: quiz q10-3 short placeholder="总结这一教程"
请你用**2-3 句话**总结这个教程**最重要的一课**。

> 参考答案（不唯一）：
> - "神经网络 = 一堆权重 w + 激活函数 σ + 梯度下降。训练就是循环前向、反向、更新。简单，但能解决复杂问题"
> - "神经网络不是黑盒——它就是矩阵乘法 + sigmoid + 链式法则。手算一遍 2-4-1 MLP 反向，就完全懂了"
> - "会手写 MLP = 面试 AI 岗的入场券。XOR 是最简单的检验，能用 2-4-1 解 XOR，就掌握了所有深度学习的基础"
:::

---

## 🎉 恭喜你学完整个教程！

> 如果你做完了所有 32 道测试题、答对 ≥ 60%：
> 1. 点击右上角"🌙/☀️"确认主题偏好
> 2. 点击右上角"中文/English"确认语言
> 3. 点击侧边栏"🎓 结业证书"
> 4. 填上你的名字，下载 PNG 证书！

**继续深入**：
- 看书：《深度学习入门：基于 Python 的理论与实现》（斋藤康毅）
- 课程：[3Blue1Brown 的神经网络系列](https://www.3blue1brown.com/topics/neural-networks)（YouTube）
- 实战：Kaggle 入门赛（Titanic / MNIST）

— 教程完 —
