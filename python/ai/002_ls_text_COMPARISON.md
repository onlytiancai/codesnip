# 语言模型对比：最小二乘法 vs Softmax + 交叉熵

## 概述

本文档对比了两种训练 n-gram 语言模型的方法：
1. **最小二乘法 (Least Squares)** - 原始方法
2. **Softmax + 交叉熵损失** - 概率方法

## 方法 1: 最小二乘法 (Least Squares)

### 原理
```
给定训练样本 (x_i, y_i)，目标是最小化：
L = ||Y - W @ X.T||²_F

直接求解：
W = Y @ X.T @ (X @ X.T)^(-1)
```

### 特点
- ✅ 直接求解，无需迭代
- ✅ 计算简单，一行代码解决
- ✅ 对于小数据集很快（O(n³)矩阵求逆）
- ❌ 输出不是概率（可能为负，可能 > 1）
- ❌ 无法应用概率论的理论框架
- ❌ 对异常值敏感
- ❌ 无法自然地处理多分类的概率

### 公式
```
前向：y_hat = W @ x
损失：L = ||y - y_hat||²
```

### 代码示例
```python
# 矩阵形式求解
XXT = X @ X.T
W = Y @ X.T @ np.linalg.pinv(XXT)  # 一行求解

# 预测
y_pred = x @ W.T
```

---

## 方法 2: Softmax + 交叉熵损失

### 原理
```
使用 logistic 回归框架，但输出为概率分布：

前向传播：
z = W @ x + b              (logits, shape: [V])
p = softmax(z)             (概率, shape: [V])

损失函数（交叉熵）：
L = -sum(y_true * log(p))  (KL散度的特殊情况)

反向传播：
dL/dz = p - y_true
dL/dW = x @ (p - y_true)^T
dL/db = p - y_true
```

### 特点
- ✅ 输出为有效的概率分布（和为1，都在[0,1]）
- ✅ 理论基础清晰（最大似然估计）
- ✅ 可以自然地处理概率相关的推理
- ✅ 梯度反向传播简洁（dL/dz = p - y_true）
- ✅ 可扩展性好，易于添加正则化、dropout等
- ✅ 对异常值不敏感（softmax 的平滑性）
- ❌ 需要迭代训练，收敛速度取决于超参数
- ❌ 需要选择学习率、初始化等超参数
- ❌ 计算复杂度O(n*m*k)（n=epochs, m=samples, k=features）

### 数值稳定的实现
```python
def softmax(z):
    """避免数值溢出"""
    z_max = np.max(z, axis=-1, keepdims=True)
    exp_z = np.exp(z - z_max)
    return exp_z / np.sum(exp_z, axis=-1, keepdims=True)

def cross_entropy_loss(y_true, y_pred):
    """交叉熵损失"""
    m = y_true.shape[0]
    loss = -np.sum(y_true * np.log(y_pred + 1e-8)) / m
    return loss
```

### 梯度下降更新
```python
# 前向传播
logits = X @ W + b        # (T, V)
probs = softmax(logits)   # (T, V)

# 反向传播
dlogits = probs - Y       # (T, V)
dW = X.T @ dlogits       # (2V, V)
db = np.sum(dlogits, axis=0)  # (V,)

# 更新
W -= learning_rate * dW
b -= learning_rate * db
```

---

## 实验结果对比

### 数据集信息
- 样本数：26
- 输入维度：46（2个词的 one-hot，各23维）
- 输出维度：23（词表大小）

### 结果

| 指标 | 最小二乘法 | Softmax+交叉熵 |
|-----|---------|-------------|
| 训练准确率 | 88.46% | 88.46% (200 epochs) |
| 输出形式 | 任意实数 | 概率分布 |
| 训练时间 | 毫秒级 | 秒级 |
| 可解释性 | 低（不是概率） | 高（概率） |
| 可扩展性 | 差 | 优 |

### 预测示例

**最小二乘法：**
```
我 喜欢 -> 学习 (score: 0.5000)
学习 是 -> 一个 (score: 1.0000)
```

**Softmax + 交叉熵：**
```
我 喜欢 -> 学习 (confidence: 0.4930)
学习 是 -> 一个 (confidence: 0.9883)
```

---

## 何时选择哪种方法？

### 使用最小二乘法
- 数据集非常小（< 100样本）且需要快速反馈
- 问题已知线性可分
- 需要绝对的计算速度

### 使用 Softmax + 交叉熵
- ✅ **推荐用于现代深度学习**
- 数据集中等规模以上
- 需要概率输出
- 需要可解释的模型置信度
- 计划进行进一步的扩展（如添加隐藏层）
- 需要正则化或其他技巧来防止过拟合

---

## 理论洞察

### 为什么 Softmax 优于直接输出？

1. **概率性质**
   - Softmax 保证 Σp_i = 1，每个 p_i ∈ [0,1]
   - 可以进行 log-likelihood 最大化

2. **梯度流**
   - 交叉熵 + softmax 的梯度非常简洁：dL/dz = p - y
   - 这是最优的梯度形式（不受饱和问题困扰）

3. **贝叶斯解释**
   - 最大化交叉熵等价于最大似然估计
   - 具有清晰的概率论基础

4. **可扩展性**
   - 易于添加隐藏层（神经网络）
   - 易于添加 L1/L2 正则化
   - 易于添加 dropout、batch norm 等技巧

---

## 文件说明

- `002_ls_text_05.py` - 原始的最小二乘法实现
- `002_ls_text_05_softmax.py` - Softmax + 交叉熵的纯实现
- `002_ls_text_05_comparison.py` - 两种方法的详细对比

### 运行方式

```bash
# 运行原始最小二乘法
python 002_ls_text_05.py

# 运行 softmax 版本
python 002_ls_text_05_softmax.py

# 运行对比版本（推荐）
python 002_ls_text_05_comparison.py
```

---

## 总结

| 特性 | 最小二乘 | Softmax |
|-----|--------|--------|
| 理论基础 | 线性回归 | 最大似然估计 |
| 输出 | 实数 | 概率 |
| 训练 | 直接求解 | 迭代优化 |
| 计算成本 | O(n³) 求逆 | O(n*m*k)*迭代次数 |
| 现代应用 | 很少 | 标准方法 |

**推荐：**使用 **Softmax + 交叉熵** 作为现代机器学习的标准方法。它提供更好的理论基础、可解释性和可扩展性。

---

## 🔑 核心差异

### 最小二乘法
```python
W = Y @ X.T @ np.linalg.pinv(X @ X.T)  # 直接求解
y_pred = x @ W.T  # 输出任意实数
```

### Softmax + 交叉熵
```python
logits = x @ W + b
probs = softmax(logits)  # 输出概率分布
loss = -sum(y_true * log(probs))  # 交叉熵损失
dW = X.T @ (probs - Y)  # 梯度简洁优雅
```

## 📊 实验结果

两种方法在此数据集上的训练准确率都是 **88.46%**，但：
- ✅ Softmax 输出有意义的概率（0.49 vs 0.5000）
- ✅ Softmax 有清晰的理论基础
- ✅ Softmax 易于扩展（可加隐藏层、正则化等）

运行 002_ls_text_05_comparison.py 可以看到完整的对比！