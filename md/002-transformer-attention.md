# Transformer 注意力机制（图文 + 公式 + 代码版）

## 1. 前置数学（可跳过的进阶阅读）

如果你已经熟悉向量、矩阵、点积、softmax，可以直接跳到 §2。下面给"想补数学的人"用。

### 1.1 标量与向量

**标量**（scalar）就是一个数。**向量**（vector）是有顺序的多个数排成一列。n 维向量是 $\mathbb{R}^n$ 里的一个元素：

$$
\mathbf{x} = \begin{bmatrix} x_1 \\ x_2 \\ \vdots \\ x_n \end{bmatrix} \in \mathbb{R}^n
$$

**几何解释**：2D 向量可以画成从原点出发的有向箭头，箭头长度 = 向量模长 $|\mathbf{x}| = \sqrt{x_1^2 + x_2^2}$，箭头方向 = 各个分量的比例关系。

![向量作为 2D 平面上的箭头](images/002/01_vector_geometry.png)

> n 维向量没法画（n>3 就画不出来了），但**代数规则完全相同**——所以「embedding 是 4 维向量」和「平面上的 2D 箭头」本质上是同一回事，只是维度不同。

### 1.2 矩阵 × 向量：行作为"询问器"

给定 $m \times n$ 矩阵 $W$ 和 $n$ 维向量 $\mathbf{x}$，乘积 $W\mathbf{x}$ 是一个 $m$ 维向量，**第 i 维 = 矩阵第 i 行 · 向量**：

$$
W\mathbf{x} = \begin{bmatrix}
\rule[0.5ex]{1em}{0.4pt}\;\mathbf{w}_1\;\rule[0.5ex]{1em}{0.4pt} \\
\rule[0.5ex]{1em}{0.4pt}\;\mathbf{w}_2\;\rule[0.5ex]{1em}{0.4pt} \\
\vdots \\
\rule[0.5ex]{1em}{0.4pt}\;\mathbf{w}_m\;\rule[0.5ex]{1em}{0.4pt}
\end{bmatrix}
\mathbf{x} = \begin{bmatrix}
\mathbf{w}_1 \cdot \mathbf{x} \\
\mathbf{w}_2 \cdot \mathbf{x} \\
\vdots \\
\mathbf{w}_m \cdot \mathbf{x}
\end{bmatrix}
$$

**直觉**：把矩阵的每一行想象成一个"询问器"——它只关心向量里它"想问"的那几个维度。比如我们想让第 2 行去问"这个 token 有没有生命"，就把第 2 行设成 `[0, 1, 0, 0]`，于是 $\mathbf{w}_2 \cdot \mathbf{x} = x_2$（embedding 的"有生命"维度）。

![矩阵乘向量：行作为"询问器"](images/002/02_matvec.png)

在注意力机制里，$W_Q$、$W_K$、$W_V$ 就是三个这样的"询问器矩阵"——它们从同一个 embedding 出发，各自问不同的问题。

### 1.3 点积 = $|\mathbf{a}||\mathbf{b}|\cos\theta$，几何意义

两个向量的点积：

$$
\mathbf{a} \cdot \mathbf{b} = \sum_i a_i b_i = |\mathbf{a}|\,|\mathbf{b}|\cos\theta
$$

其中 $\theta$ 是 $\mathbf{a}$、$\mathbf{b}$ 的夹角。**点积越大 = 两向量方向越一致 = 越"像"**。

- $\theta = 0°$（同向）→ $\cos\theta = 1$ → 点积最大
- $\theta = 90°$（垂直）→ $\cos\theta = 0$ → 点积为 0
- $\theta = 180°$（反向）→ $\cos\theta = -1$ → 点积最小（负数）

![点积 = |a||b|cosθ + Q·K(animal) vs Q·K(street) 对比](images/002/03_dot_product_angle.png)

> 在注意力机制里，我们用 Q·K 衡量"Q 想要的特征"和"K 提供的特征"有多一致。完全一致 = 完全匹配，垂直 = 毫不相关。

### 1.4 softmax：从实数到概率分布

softmax 把 K 个实数（logits）变成 K 个加起来等于 1 的非负数：

$$
\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_{j=1}^{K} e^{x_j}}
$$

**两个关键性质**：

1. **指数放大差距**：差距 1 的两个 logit，softmax 后权重比 ≈ 2.7；差距 5 的两个，权重比 ≈ 148。所以 softmax 会"奖励强者、惩罚弱者"——这正是注意力想要的"集中注意"行为。
2. **平移不变**：所有 logit 加上同一个常数 $c$，softmax 输出不变。这就是为什么实现里要 `x - max(x)` 防止 $e^x$ 数值溢出。

![softmax 曲线 + 7 token logits→权重](images/002/04_softmax.png)

**代入 demo 的 7 个 Q·K 分数**：

$$
\text{scores} = [0, 1, 0, 0, 0, 0, 0] \quad\Rightarrow\quad
\text{softmax} = [0.1147, 0.3118, 0.1147, 0.1147, 0.1147, 0.1147, 0.1147]
$$

验算：$e^1 = 2.718$，$e^0 = 1$，分母 $= 2.718 + 6 = 8.718$。

- $0.3118 = 2.718 / 8.718$ ← animal
- $0.1147 = 1 / 8.718$ ← 其余 6 个

---

## 2. Embedding：从 token 到 4 维向量

Transformer 的输入是 token 序列（"The animal didn't cross the street because **it** was tired." → 7 个 token）。**每个 token 要先变成一个向量**才能进入注意力层。

我们用教学版的 4 维 0/1 向量作 embedding（真实模型是几百维的浮点数）：

| dim | 含义 | 备注 |
|---|---|---|
| 0 | 是不是名词 | 名词=1，动词/虚词=0 |
| 1 | 有没有生命 | 活物=1，无生命=0 |
| 2 | 能不能感到疲劳 | 能累=1，不能累=0 |
| 3 | 是不是代词 | 代词=1，其他=0 |

![7 token × 4 维 embedding 热图](images/002/05_embedding_heatmap.png)

> 真实模型的 embedding 是 768、1024、4096 维的浮点数，**每一维没有具体语义**——它们是"训练"出来的"潜在特征"。我们这里用 4 维 0/1 向量只是为了能"手算"和"画图"，机制完全等价。

---

## 3. QKV 三个投影：W_Q / W_K / W_V

Q/K/V 不是凭空冒出来的——它们都是**同一个 embedding 向量经过不同的"翻译"得到**的：

$$
\mathbf{q}_i = W_Q \mathbf{x}_i, \quad \mathbf{k}_i = W_K \mathbf{x}_i, \quad \mathbf{v}_i = W_V \mathbf{x}_i
$$

三个矩阵各自问不同的问题：

| 矩阵 | 在问什么 | 输出维度（demo） |
|---|---|---|
| $W_Q$ | 我**想找什么**特征 | 4 维 |
| $W_K$ | 我**能提供**什么特征供别人匹配 | 4 维 |
| $W_V$ | 我**能贡献**什么内容 | 4 维 |

### 3.1 demo 里的 W 设计

```python
W_Q = [
    [0, 0, 0, 0],   # Q dim0 = 0
    [0, 0, 0, 1],   # Q dim1 ← 代词 dim3=1 ⇒ 1×1=1
    [0, 0, 0, 0],   # Q dim2 = 0  ← 不取"能累"
    [0, 0, 0, 0],
]
W_K = [
    [1, 0, 0, 0],   # K dim0 ← embedding dim0（名词）
    [0, 1, 0, 0],   # K dim1 ← embedding dim1（有生命）
    [0, 0, 1, 0],   # K dim2 ← embedding dim2（能累）
    [0, 0, 0, 0],
]
W_V = [
    [1, 0, 0, 0],   # V dim0 ← embedding dim0（名词）
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
]
```

**W_Q 的设计意图**：`Q(it)` 只看 `it` 的 dim3（代词），然后"翻译"成 dim1（有生命）——意思就是"代词在找有生命的东西"。

**W_K 的设计意图**：保留 embedding 的前 3 维，让"名词"、"有生命"、"能累"三个特征都能在 K 里被检索到。

**W_V 的设计意图**：**只保留 dim0（名词身份）**——意思就是"我只能告诉别人：我是名词"。

### 3.2 为什么 W_V 要砍到只剩 dim0？

这是 001 文档里**最精彩的设计哲学讨论**之一。W_V 只贡献"是不是名词"这一维，目的是让 `output(it)` 不会"覆盖" `it` 自己的"我是代词"这个身份：

- 如果 W_V = 单位阵（保留所有 dim），那 `output(it)` 会几乎等于 `V(animal)`——`it` 和 `animal` 变得不可区分，**代词这个语法角色就丢了**。
- 如果 W_V 只贡献"代词"那一维，那 `output(it)[3] = α_total × 0 < 1`——**反而稀释**了原始 embedding 里"我是代词"的信号。

让 V 只携带"外部信息"（来自其他 token 的），原始 embedding 才能保留"我是谁"——这两股信息在下一层**拼起来**才有意义。详见 §6。

---

## 4. Self-Attention 完整流程

把前三节串起来，Self-Attention 实际上只做三件事：

### 4.1 步骤 1：算所有 Q·K 分数

对查询位置 $i$ 和所有键位置 $j$ 算点积：

$$
s_{ij} = \mathbf{q}_i \cdot \mathbf{k}_j
$$

在 demo 里查询位置固定为 `it`（位置 6），得到 7 个分数：

![Q(it)·K(t) 的 7 个分数条形图](images/002/06_qk_scores.png)

```
The       0.00
animal    1.00   ← Q(it) 想要"有生命"，animal 在 dim1 上是 1
didn't    0.00
cross     0.00
the       0.00
street    0.00   ← street 在 dim1 上是 0，不匹配
it        0.00
```

### 4.2 步骤 2：softmax 得到权重

对每个查询位置 i，把所有分数过 softmax：

$$
\alpha_{ij} = \frac{e^{s_{ij}}}{\sum_{j'} e^{s_{ij'}}}
$$

> 真实实现里通常还会**除以 $\sqrt{d_k}$**（$d_k$ = K 的维度），见 §7.1。demo 里没做这一步，因为 $d_k=4$、$\sqrt{4}=2$，效果只是"把分数整体除以 2"，softmax 后权重分布形状不变。

![softmax 后 7 个权重](images/002/07_attn_weights.png)

```
The       0.1147
animal    0.3118   ← 高度集中
didn't    0.1147
cross     0.1147
the       0.1147
street    0.1147
it        0.1147
```

### 4.3 步骤 3：α·V 加权求和

每个输出位置 i 用权重对所有 Value 加权求和：

$$
\mathbf{o}_i = \sum_{j=1}^{n} \alpha_{ij} \mathbf{v}_j
$$

在 demo 里，$\mathbf{o}_{it} = \sum_{j} \alpha_j \mathbf{v}_j = [0.43, 0, 0, 0]$，具体分解见 §5 / §6。

### 4.4 整体管道

![Self-Attention 完整流程](images/002/09_qkv_pipeline.png)

---

## 5. 完整 demo 复现

直接跑 [`code/attention_demo.py`](code/attention_demo.py) 会得到以下完整数据流。本节所有数字都和代码输出一一对应。

### 5.1 7 个 token 的 embedding

| token | dim0 (名词) | dim1 (有生命) | dim2 (能累) | dim3 (代词) |
|---|:-:|:-:|:-:|:-:|
| The | 0 | 0 | 0 | 0 |
| animal | **1** | **1** | **1** | 0 |
| didn't | 0 | 0 | 0 | 0 |
| cross | 0 | 0 | 0 | 0 |
| the | 0 | 0 | 0 | 0 |
| street | **1** | 0 | 0 | 0 |
| it | 0 | 0 | 0 | **1** |

### 5.2 Q / K / V 投影

经过 $W_Q$、$W_K$、$W_V$ 投影后（仅列出有意义的 token）：

| token | Q = W_Q·x | K = W_K·x | V = W_V·x |
|---|---|---|---|
| animal | [0, 0, 0, 0] | [1, 1, 1, 0] | [1, 0, 0, 0] |
| street | [0, 0, 0, 0] | [1, 0, 0, 0] | [1, 0, 0, 0] |
| it | [0, 1, 0, 0] | [0, 0, 0, 0] | [0, 0, 0, 0] |
| 其它 4 个 | [0, 0, 0, 0] | [0, 0, 0, 0] | [0, 0, 0, 0] |

注意 $Q(it) = [0, 1, 0, 0]$——它"想要"对方的 dim1（"有生命"）特征；$K(animal) = [1, 1, 1, 0]$ 在 dim1 上是 1，匹配；$K(street) = [1, 0, 0, 0]$ 在 dim1 上是 0，不匹配。

### 5.3 Q·K 分数与 softmax 权重

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

**验算**：$\alpha(animal) = e^1 / (e^1 + 6e^0) = 2.718 / 8.718 = 0.3118$；其它 6 个各 $1/8.718 = 0.1147$。

### 5.4 output(it) 怎么算出来的

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

## 6. output 解读：为什么是 0.43

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

> 引用 001 的金句：**output 应该回答"我注意到了什么"，而不是"我注意到了谁"**。让 output 直接表达"it = animal"在技术上可行，但破坏了 Transformer 的分层抽象。

---

## 7. 进阶：让 attention 真正可用

### 7.1 Scaled Dot-Product：为什么除以 $\sqrt{d_k}$

原论文公式是：

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right) V
$$

**为什么要除以 $\sqrt{d_k}$？** 防止 softmax 梯度消失。

直觉：Q 和 K 的每个元素都是均值为 0、方差为 1 的随机变量时，$QK^\top$ 的方差是 $d_k$。$d_k$ 越大（比如 512），Q·K 的绝对值越大，softmax 越接近 one-hot，**梯度越接近 0**。除以 $\sqrt{d_k}$ 把方差拉回 1，softmax 输出分布更"软"，梯度更健康。

> Demo 里 $d_k = 4$、$\sqrt{d_k} = 2$，除以 2 只是把分数整体缩小，softmax 后权重分布形状不变。所以教学 demo 可以省略这一步。

### 7.2 Masking：把"未来 token"屏蔽掉

Decoder 用的是 **Causal Self-Attention**——生成第 i 个 token 时**不能看到第 i+1, i+2, ... 个 token**（否则就是作弊）。

实现方法：在 softmax 之前把"未来位置"的分数置为 $-\infty$：

$$
s_{ij} = \begin{cases}
\mathbf{q}_i \cdot \mathbf{k}_j, & j \le i \\
-\infty, & j > i
\end{cases}
$$

$e^{-\infty} = 0$，所以这些位置的权重自动为 0。

### 7.3 Multi-Head：8 个 head 并行关注不同子空间

单个 attention 头只能学一种"找法"。**Multi-Head** 把 $d$ 维切分成 h 份（比如 768 维切 12 个 64 维 head），每个 head **独立**做一次 attention，最后拼回来：

$$
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) W_O
$$

$$
\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)
$$

直觉：8 个 head 可能分别学到"指代关系"、"句法结构"、"位置关系"……，每个 head 关注一个子空间。

![Multi-Head Attention 简化示意](images/002/10_multihead.png)

### 7.4 PyTorch 等价实现

下面是上面 demo 思路的 PyTorch 等价版（与 [`transformer.md:25`](transformer.md#L25) 风格一致，可对照阅读）：

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

## 8. 小结

**三句话回顾**：

1. **Self-Attention = 软数据库查询**：用 Q 匹配所有 K 得到权重，再用权重从 V 那里"取回"信息——可以理解为"可微分的模糊查表"。
2. **Q/K/V 是同源的三个翻译**：都是 embedding 经过不同 $W$ 投影的结果，分别回答"找什么"、"提供什么"、"贡献什么"。
3. **output 是"我注意到的"，不是"我注意到的谁"**：单层 attention 只收集证据，最终判断要交给多层堆叠。

**下一步学习路径**：

- 看 [`transformer.md`](transformer.md) — 完整 Self-Attention 类 + Multi-Head 实现
- 读原论文 *Attention Is All You Need* ([arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762))
- 自己改 [`code/attention_demo.py`](code/attention_demo.py) 里的 $W$ 矩阵、换句子、换维度，观察 attention 权重怎么变

---

## 附录 A：图清单

| 图 | 章节 | 文件 |
|---|---|---|
| 1 | §1.1 标量与向量 | [01_vector_geometry.png](images/002/01_vector_geometry.png) |
| 2 | §1.2 矩阵乘向量 | [02_matvec.png](images/002/02_matvec.png) |
| 3 | §1.3 点积几何 | [03_dot_product_angle.png](images/002/03_dot_product_angle.png) |
| 4 | §1.4 softmax | [04_softmax.png](images/002/04_softmax.png) |
| 5 | §2 Embedding | [05_embedding_heatmap.png](images/002/05_embedding_heatmap.png) |
| 6 | §4.1 Q·K 分数 | [06_qk_scores.png](images/002/06_qk_scores.png) |
| 7 | §4.2 softmax 权重 | [07_attn_weights.png](images/002/07_attn_weights.png) |
| 8 | §5 / §6 output 分解 | [08_output_decomposition.png](images/002/08_output_decomposition.png) |
| 9 | §4.4 整体流程 | [09_qkv_pipeline.png](images/002/09_qkv_pipeline.png) |
| 10 | §7.3 Multi-Head | [10_multihead.png](images/002/10_multihead.png) |

**重生成**：`cd scripts/002 && python gen_all.py`（详见 [scripts/002/README.md](scripts/002/README.md)）。
