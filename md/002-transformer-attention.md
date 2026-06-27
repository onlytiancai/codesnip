# Transformer 注意力机制

> 注意力机制是 Transformer 的核心创新。在它出现之前，RNN 处理长序列时几乎无力应对"远距离依赖"问题——比如句子开头的词和结尾的词往往有语义关联，但 RNN 很难记住那么久之前的信息。注意力机制用一种"查询匹配"的思路，让任意两个位置之间可以直接建立联系，不受距离限制。本文用图文、公式和代码，带你从零理解它。

---

## 1. 从一个问题开始：代词指代

看这个句子：

> The animal didn't cross the street because **it** was tired.

当模型读到 `it` 这个词时，它需要知道 `it` 指的是谁。这是个看似简单、实则微妙的问题，`it` 既可以指 `animal`，也可以指 `street`，但只有 `animal` 符合"能感到疲劳"这个语义。

这就是典型的**远距离依赖**：关键信息（`animal`）和当前词（`it`）之间隔了 6 个词。传统 RNN 必须靠"记性"熬过这 6 步才能建立联系，而注意力机制换了一种思路——

> **直接问："it 指的是谁？"然后去所有词里找答案。**

---

## 2. QKV 的诞生：把"问"和"答"拆成三个角色

注意力机制的核心是把每次"查询"拆成三个角色：

- **Query（查询）**：我当前需要什么信息？比如 `it` 在问"代词指的是谁"
- **Key（键）**：我提供什么特征供别人匹配？比如 `animal` 说"我有生命、我能疲劳"
- **Value（值）**：如果匹配成功，我真正传递什么内容？比如 `animal` 的语义信息

继续用代词的例子，Query 是 `it` 发出的请求，Key 是候选词提供的特征描述，Value 是实际要被 `it` 吸收的信息。

### 2.1 用向量来表示 Query、Key、Value

每个词的 Query、Key、Value 都不是凭空产生的——它们都来自同一个词的"向量表示"（embedding），经过不同的投影矩阵得到：

$$
\mathbf{q}_i = W_Q \mathbf{x}_i, \quad \mathbf{k}_i = W_K \mathbf{x}_i, \quad \mathbf{v}_i = W_V \mathbf{x}_i
$$

这三个矩阵分别问不同的问题：

| 矩阵 | 在问什么 |
|---|---|
| $W_Q$ | 我**想找什么**特征 |
| $W_K$ | 我**能提供**什么特征供别人匹配 |
| $W_V$ | 我**能贡献**什么内容 |

![embedding 经过 W_Q / W_K / W_V 投影得到 QKV](images/002/09_qkv_pipeline.png)

---

## 3. 引入数学：向量和矩阵乘法

在继续之前，我们需要一点点数学工具。

### 3.1 向量与 embedding

**向量**是有顺序的多个数排成一列。n 维向量是 $\mathbb{R}^n$ 里的一个元素：

$$
\mathbf{x} = \begin{bmatrix} x_1 \\ x_2 \\ \vdots \\ x_n \end{bmatrix} \in \mathbb{R}^n
$$

Transformer 里，每个 token 先被转换成一个向量（**embedding**）。我们用一个简化的 4 维 0/1 向量做教学演示：

| dim | 语义含义 | animal | street | it |
|---|---|:-:|:-:|:-:|
| 0 | 是不是名词 | 1 | 1 | 0 |
| 1 | 有没有生命 | 1 | 0 | 0 |
| 2 | 能不能感到疲劳 | 1 | 0 | 0 |
| 3 | 是不是代词 | 0 | 0 | 1 |

> 真实模型的 embedding 是 768、1024、4096 维的浮点数，**每一维没有具体语义**——它们是"训练"出来的"潜在特征"。我们这里用 4 维 0/1 向量只是为了能"手算"和"画图"，机制完全等价。

![7 token × 4 维 embedding 热图](images/002/05_embedding_heatmap.png)

---

### 3.2 矩阵 × 向量：行作为"语义探针"

**点积**（dot product）是两个向量对应分量相乘后求和：

$$
\mathbf{a} \cdot \mathbf{b} = \sum_i a_i b_i
$$

例如 $[1, 2, 3] \cdot [4, 5, 6] = 1\times4 + 2\times5 + 3\times6 = 32$。

---

给定 $m \times n$ 矩阵 $W$ 和 $n$ 维向量 $\mathbf{x}$，乘积 $W\mathbf{x}$ 是一个 $m$ 维向量，**第 i 维 = 矩阵第 i 行与 $\mathbf{x}$ 的点积**：

$$
W\mathbf{x} = \begin{bmatrix}
\mathbf{w}_1 \cdot \mathbf{x} \\
\mathbf{w}_2 \cdot \mathbf{x} \\
\vdots \\
\mathbf{w}_m \cdot \mathbf{x}
\end{bmatrix}
$$

**直觉**：把矩阵的每一行想象成一个"语义探针"——它只关心向量里它"想问"的那几个维度。

$W_K$ 的每一行也是一个"语义探针"。以图中的 $W_K$ 为例：

- 第 1 行 `[1,0,0,0]` 只问 dim0（名词性）
- 第 2 行 `[0,1,0,0]` 只问 dim1（有生命）
- 第 3 行 `[0,0,1,0]` 只问 dim2（能感觉累）

$W_K$ 与 $\text{embedding}(\text{animal}) = [1,1,1,0]$ 相乘后，得到 $K(\text{animal})$：

$$
K(\text{animal}) = \begin{bmatrix}
\mathbf{w}_1 \cdot \text{embedding} \\
\mathbf{w}_2 \cdot \text{embedding} \\
\mathbf{w}_3 \cdot \text{embedding}
\end{bmatrix}
= \begin{bmatrix}
1 \\ 1 \\ 1
\end{bmatrix}
$$

$W_K$ 与 $\text{embedding}(\text{street}) = [1,0,0,0]$ 相乘后，得到 $K(\text{street})$：

$$
K(\text{street}) = \begin{bmatrix}
1 \\ 0 \\ 0
\end{bmatrix}
$$

**$K$ 的每个维度就是 $W_K$ 对应行"问"出的那个语义特征的强度**——$K(\text{animal})$ dim1=1 说明 animal 的"有生命"特征很强，$K(\text{street})$ dim2=0 说明 street 的"能感觉累"特征很弱。

![矩阵乘向量：行作为"语义探针"](images/002/02_matvec.png)

---

### 3.3 点积的几何含义：方向越一致，值越大

点积有一个漂亮的几何解释：

$$
\mathbf{a} \cdot \mathbf{b} = |\mathbf{a}|\,|\mathbf{b}|\cos\theta
$$

其中 $\theta$ 是 $\mathbf{a}$、$\mathbf{b}$ 的夹角。**点积越大 = 两向量方向越一致 = 越"像"**：

- $\theta = 0°$（同向）→ $\cos\theta = 1$ → 点积最大
- $\theta = 90°$（垂直）→ $\cos\theta = 0$ → 点积为 0
- $\theta = 180°$（反向）→ $\cos\theta = -1$ → 点积最小

在注意力机制里，我们用 Q·K 衡量"Q 想要的特征"和"K 提供的特征"有多一致。完全一致 = 完全匹配，垂直 = 毫不相关。

![点积 = |a||b|cosθ + Q·K(animal) vs Q·K(street) 对比](images/002/03_dot_product_angle.png)

---

## 4. 匹配过程：Query 找 Key

回到代词指代的例子。`it` 需要找到它的先行词（Query），而 `animal` 和 `street` 各自提供自己的特征（Key）。

### 4.1 设计 W_Q 和 W_K

我们给 demo 设计三个投影矩阵：

```python
W_Q = [
    [0, 0, 0, 0],   # Q dim0 = 0
    [0, 0, 0, 1],   # Q dim1 ← 代词 dim3=1 ⇒ 1×1=1
    [0, 0, 0, 0],   # Q dim2 = 0
    [0, 0, 0, 0],
]
W_K = [
    [1, 0, 0, 0],   # K dim0 ← embedding dim0（名词）
    [0, 1, 0, 0],   # K dim1 ← embedding dim1（有生命）
    [0, 0, 1, 0],   # K dim2 ← embedding dim2（能累）
    [0, 0, 0, 0],
]
```

**W_Q 的设计意图**：`Q(it)` 只看 `it` 的 dim3（代词），然后"翻译"成 dim1（有生命）——意思就是"代词在找有生命的东西"。

**W_K 的设计意图**：保留 embedding 的前 3 维，让"名词"、"有生命"、"能累"三个特征都能在 K 里被检索到。

投影结果：

| token | Q = W_Q·x | K = W_K·x |
|---|---|---|
| animal | [0, 0, 0, 0] | [1, 1, 1, 0] |
| street | [0, 0, 0, 0] | [1, 0, 0, 0] |
| it | [0, 1, 0, 0] | [0, 0, 0, 0] |

注意 $Q(it) = [0, 1, 0, 0]$——它"想要"对方的 dim1（"有生命"）特征；$K(animal) = [1, 1, 1, 0]$ 在 dim1 上是 1，匹配；$K(street) = [1, 0, 0, 0]$ 在 dim1 上是 0，不匹配。

---

### 4.2 Q·K 分数：匹配度量化

现在让 `it` 的 Query 和所有词的 Key 做点积——这就是**匹配度打分**：

$$
s_{ij} = \mathbf{q}_i \cdot \mathbf{k}_j
$$

对 `it`（位置 6）和所有 7 个 token 算分：

| token | $Q(it)·K(t)$ | 解读 |
|---|:-:|---|
| The | 0.00 | |
| **animal** | **1.00** | **匹配！** Q(it) 想要"有生命"，animal 在 dim1 上是 1 |
| didn't | 0.00 | |
| cross | 0.00 | |
| the | 0.00 | |
| street | 0.00 | street 在 dim1 上是 0，不匹配 |
| it | 0.00 | 它自己也不在 dim1 上响应 |

![Q(it)·K(t) 的 7 个分数条形图](images/002/06_qk_scores.png)

---

## 5. 从分数到权重：softmax

光有分数不够——我们需要一个"概率分布"，把所有分数转成"权重"，表示每个词被注意的程度。

### 5.1 softmax 函数

**softmax** 把 K 个实数（logits）变成 K 个加起来等于 1 的非负数：

$$
\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_{j=1}^{K} e^{x_j}}
$$

**两个关键性质**：

1. **指数放大差距**：差距 1 的两个 logit，softmax 后权重比 ≈ 2.7；差距 5 的两个，权重比 ≈ 148。所以 softmax 会"奖励强者、惩罚弱者"——这正是注意力想要的"集中注意"行为。
2. **平移不变**：所有 logit 加上同一个常数 $c$，softmax 输出不变。这就是为什么实现里要 `x - max(x)` 防止 $e^x$ 数值溢出。

![softmax 曲线 + 7 token logits→权重](images/002/04_softmax.png)

### 5.2 代入 demo 数据

$$
\text{scores} = [0, 1, 0, 0, 0, 0, 0] \quad\Rightarrow\quad
\text{softmax} = [0.1147, 0.3118, 0.1147, 0.1147, 0.1147, 0.1147, 0.1147]
$$

验算：$e^1 = 2.718$，$e^0 = 1$，分母 $= 2.718 + 6 = 8.718$。

- $0.3118 = 2.718 / 8.718$ ← animal
- $0.1147 = 1 / 8.718$ ← 其余 6 个

**attention 权重**：

| token | $\alpha$ (softmax) |
|---|:-:|
| The | 0.1147 |
| **animal** | **0.3118** ← 高度集中 |
| didn't | 0.1147 |
| cross | 0.1147 |
| the | 0.1147 |
| street | 0.1147 |
| it | 0.1147 |
| **合计** | **1.0000** |

![softmax 后 7 个权重](images/002/07_attn_weights.png)

---

## 6. 提取 Value：加权求和得到输出

权重算出来了，现在用权重对所有 Value 做**加权求和**——这就是最终输出：

$$
\mathbf{o}_i = \sum_{j=1}^{n} \alpha_{ij} \mathbf{v}_j
$$

### 6.1 Value 的设计

```python
W_V = [
    [1, 0, 0, 0],   # V dim0 ← embedding dim0（名词）
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
]
```

**W_V 的设计意图**：只保留 dim0（名词身份）——意思就是"我只能告诉别人：我是名词"。

投影结果：

| token | V = W_V·x |
|---|---|
| animal | [1, 0, 0, 0] |
| street | [1, 0, 0, 0] |
| it | [0, 0, 0, 0] |
| 其它 | [0, 0, 0, 0] |

### 6.2 为什么 W_V 要砍到只剩 dim0？

这是注意力机制里**最精彩的设计哲学**之一。W_V 只贡献"是不是名词"这一维，目的是让 `output(it)` 不会"覆盖" `it` 自己的"我是代词"这个身份：

- 如果 W_V = 单位阵（保留所有 dim），那 `output(it)` 会几乎等于 `V(animal)`——`it` 和 `animal` 变得不可区分，**代词这个语法角色就丢了**。
- 如果 W_V 只贡献"代词"那一维，那 `output(it)[3] = α_total × 0 < 1`——**反而稀释**了原始 embedding 里"我是代词"的信号。

让 V 只携带"外部信息"（来自其他 token 的），原始 embedding 才能保留"我是谁"——这两股信息在下一层**拼起来**才有意义。

### 6.3 实际计算

$$
\mathbf{o}_{it} = \sum_{j} \alpha_j \mathbf{v}_j
$$

展开：

```
output(it) = Σ α·V
            = 0.1147·V(The)    + 0.3118·V(animal)  + 0.1147·V(didn't)
            + 0.1147·V(cross) + 0.1147·V(the)     + 0.1147·V(street)
            + 0.1147·V(it)

每维求和：
  dim0 = 0.1147·0 + 0.3118·1 + 0.1147·0 + 0.1147·0 + 0.1147·0 + 0.1147·1 + 0.1147·0
       = 0.3118 + 0.1147
       = 0.4265   ≈ 0.43

  dim1 = 0  （所有 V 在 dim1 都是 0）
  dim2 = 0
  dim3 = 0

output(it) = [0.43, 0, 0, 0]
```

![output[0] = 0.43 的瀑布图分解](images/002/08_output_decomposition.png)

---

## 7. output 解读：0.43 是什么意思

`output(it) = [0.43, 0, 0, 0]` 这个 4 维向量**每一维代表什么**？

| dim | 含义 | 值 | 解释 |
|---|---|:-:|---|
| 0 | 是不是名词 | **0.43** | 几乎等于 $\alpha(animal) + \alpha(street) = 0.3118 + 0.1147 = 0.4265$ |
| 1 | 有生命 | 0 | W_V 把这一维砍成 0，V 在这一维始终是 0 |
| 2 | 能累 | 0 | 同上 |
| 3 | 代词 | 0 | 同上 |

> 0.43 这个数值**只回答了一个问题**：「在 `it` 这个位置，模型有多大把握认为它指向某个名词？」
> 答案 ≈ 43%——也就是那两个被识别为"名词"的 token（animal + street）的注意力权重之和。

**这一层 attention 没有回答"it 到底是 animal 还是 street"**——它只说"it 大概率指一个名词"。把 animal 从 street 里挑出来的工作**留给下一层 Transformer**：

```
层 1 (我们这层):  output(it) = [0.43, 0, 0, 0]
                   "it 指向某个名词，43% 把握"

层 2:  把 output + embedding 拼起来
       Q(动物检测) · K(animal) ↑↑
       输出 "animal 是那个名词"  ← 完成判断
```

**每一层只解决一小步**——**第 1 层不能抢第 2 层的活**。如果第 1 层就把"是 animal"写进 output，第 2 层就不知道有 2 个候选、不知道存在歧义、也不知道该保留什么原始信息。

---

## 8. 完整流程回顾

把前三节串起来，Self-Attention 实际上只做三件事：

1. **算 Q·K 分数**：对查询位置 $i$ 和所有键位置 $j$ 算点积 $s_{ij} = \mathbf{q}_i \cdot \mathbf{k}_j$
2. **softmax 得到权重**：对每个查询位置 i，把所有分数过 softmax $\alpha_{ij} = \frac{e^{s_{ij}}}{\sum_{j'} e^{s_{ij'}}}$
3. **加权求和**：每个输出位置 i 用权重对所有 Value 加权求和 $\mathbf{o}_i = \sum_{j=1}^{n} \alpha_{ij} \mathbf{v}_j$

完整公式：

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right) V
$$

> 真实实现里通常还会**除以 $\sqrt{d_k}$**（$d_k$ = K 的维度），这是下一节的内容。

直接跑 [`code/attention_demo.py`](code/attention_demo.py) 会得到以下完整数据流。本节所有数字都和代码输出一一对应。

### 8.1 7 个 token 的 embedding

| token | dim0 (名词) | dim1 (有生命) | dim2 (能累) | dim3 (代词) |
|---|:-:|:-:|:-:|:-:|
| The | 0 | 0 | 0 | 0 |
| animal | **1** | **1** | **1** | 0 |
| didn't | 0 | 0 | 0 | 0 |
| cross | 0 | 0 | 0 | 0 |
| the | 0 | 0 | 0 | 0 |
| street | **1** | 0 | 0 | 0 |
| it | 0 | 0 | 0 | **1** |

### 8.2 Q·K 分数与 softmax 权重

| token | $Q(it)·K(t)$ | $\alpha$ (softmax) | 解读 |
|---|:-:|:-:|---|
| The | 0.00 | 0.1147 | 基线 |
| **animal** | **1.00** | **0.3118** | **匹配上 dim1** |
| didn't | 0.00 | 0.1147 | 基线 |
| cross | 0.00 | 0.1147 | 基线 |
| the | 0.00 | 0.1147 | 基线 |
| street | 0.00 | 0.1147 | 名词但无生命 |
| it | 0.00 | 0.1147 | 它自己也不在 dim1 上响应 |
| **合计** | — | **1.0000** | softmax 必为概率分布 |

---

## 9. 进阶内容

### 9.1 Scaled Dot-Product：为什么除以 $\sqrt{d_k}$

原论文公式是：

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right) V
$$

**为什么要除以 $\sqrt{d_k}$？** 防止 softmax 梯度消失。

直觉：Q 和 K 的每个元素都是均值为 0、方差为 1 的随机变量时，$QK^\top$ 的方差是 $d_k$。$d_k$ 越大（比如 512），Q·K 的绝对值越大，softmax 越接近 one-hot，**梯度越接近 0**。除以 $\sqrt{d_k}$ 把方差拉回 1，softmax 输出分布更"软"，梯度更健康。

> Demo 里 $d_k = 4$、$\sqrt{d_k} = 2$，除以 2 只是把分数整体缩小，softmax 后权重分布形状不变。所以教学 demo 可以省略这一步。

### 9.2 Masking：把"未来 token"屏蔽掉

Decoder 用的是 **Causal Self-Attention**——生成第 i 个 token 时**不能看到第 i+1, i+2, ... 个 token**（否则就是作弊）。

实现方法：在 softmax 之前把"未来位置"的分数置为 $-\infty$：

$$
s_{ij} = \begin{cases}
\mathbf{q}_i \cdot \mathbf{k}_j, & j \le i \\
-\infty, & j > i
\end{cases}
$$

$e^{-\infty} = 0$，所以这些位置的权重自动为 0。

### 9.3 Multi-Head：8 个 head 并行关注不同子空间

单个 attention 头只能学一种"找法"。**Multi-Head** 把 $d$ 维切分成 h 份（比如 768 维切 12 个 64 维 head），每个 head **独立**做一次 attention，最后拼回来：

$$
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) W_O
$$

$$
\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)
$$

直觉：8 个 head 可能分别学到"指代关系"、"句法结构"、"位置关系"……，每个 head 关注一个子空间。

![Multi-Head Attention 简化示意](images/002/10_multihead.png)

### 9.4 PyTorch 等价实现

```python
import torch
import torch.nn.functional as F
import math


def self_attention(X, W_Q, W_K, W_V):
    """
    X:    (seq_len, d_model)  7×4
    W_*:  (d_model, d_k)      4×4
    返回: (seq_len, d_model)  7×4
    """
    Q = X @ W_Q          # (7,4)
    K = X @ W_K          # (7,4)
    V = X @ W_V          # (7,4)

    d_k = Q.size(-1)
    scores = (Q @ K.T) / math.sqrt(d_k)         # (7,7)
    attn   = F.softmax(scores, dim=-1)          # (7,7)
    output = attn @ V                           # (7,4)
    return output, attn


# 复用 demo 的 W 和 embedding
# output[6]（= it 那一行）就对应 demo 里的 [0.43, 0, 0, 0]
```

完整版（带 mask、multi-head、LayerNorm）见 [`transformer.md`](transformer.md)。

---

## 10. 小结

**三句话回顾**：

1. **Self-Attention = 软数据库查询**：用 Q 匹配所有 K 得到权重，再用权重从 V 那里"取回"信息——可以理解为"可微分的模糊查表"。
2. **Q/K/V 是同源的三个翻译**：都是 embedding 经过不同 $W$ 投影的结果，分别回答"找什么"、"提供什么"、"贡献什么"。
3. **output 是"我注意到的"，不是"我注意到的谁"**：单层 attention 只收集证据，最终判断要交给多层堆叠。