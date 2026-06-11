# Transformer 原理详解

# Transformer 原理详解

## 📐 整体架构

Transformer 采用 **Encoder-Decoder** 架构，完全摒弃了 RNN 和 CNN：

```
┌─────────────────────────────────────────────────────────────┐
│                        Transformer                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   输入                    Encoder              Decoder      │
│   序列  ──────►  [Multi-Head Attn + FFN]  ──────►  输出      │
│                 [Multi-Head Attn + FFN]  ──────►  序列      │
│                 [Multi-Head Attn + FFN]  ──────►           │
│                           ×N                               │
└─────────────────────────────────────────────────────────────┘
```

* * *

## 1️⃣ Self-Attention 机制

### 核心思想

通过 **Query (Q)、Key (K)、Value (V)** 三个向量，让序列中每个位置都能"关注"到序列中的所有位置。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super().__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        
        assert self.head_dim * heads == embed_size, "embed_size must be divisible by heads"
        
        # 三个线性变换得到 Q, K, V
        self.W_Q = nn.Linear(embed_size, embed_size)
        self.W_K = nn.Linear(embed_size, embed_size)
        self.W_V = nn.Linear(embed_size, embed_size)
        self.W_O = nn.Linear(embed_size, embed_size)
    
    def forward(self, Q, K, V, mask=None):
        """
        Q, K, V: (batch_size, seq_len, embed_size)
        返回: (batch_size, seq_len, embed_size)
        """
        batch_size = Q.size(0)
        seq_len = Q.size(1)
        
        # 线性变换 + 分头
        Q = self.W_Q(Q).view(batch_size, seq_len, self.heads, self.head_dim).transpose(1, 2)
        K = self.W_K(K).view(batch_size, seq_len, self.heads, self.head_dim).transpose(1, 2)
        V = self.W_V(V).view(batch_size, seq_len, self.heads, self.head_dim).transpose(1, 2)
        
        # ====== 核心：Scaled Dot-Product Attention ======
        
        # 1. 计算 Q 和 K 的点积注意力分数
        attention_scores = torch.matmul(Q, K.transpose(-2, -1))  # (B, H, L, L)
        
        # 2. 缩放（防止点积过大导致 softmax 梯度消失）
        attention_scores = attention_scores / math.sqrt(self.head_dim)
        
        # 3. 可选：应用掩码
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        
        # 4. Softmax 得到注意力权重
        attention_weights = F.softmax(attention_scores, dim=-1)  # (B, H, L, L)
        
        # 5. 加权求和得到输出
        context = torch.matmul(attention_weights, V)  # (B, H, L, D)
        
        # 合并多头
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_size)
        
        # 最终线性变换
        output = self.W_O(context)
        
        return output, attention_weights
```

### 注意力计算图解

```
注意力机制
                    
    Q: Query ─────┐
                  ├──► Softmax ───► Weighted Sum ───► Output
    K: Key   ─────┤      ▲
                  │      │
    V: Value  ────┘      │
                         │
                 除以 √d_k (缩放)
```

* * *

## 2️⃣ Multi-Head Attention（多头注意力）

### 为什么要多头？

-   不同的注意力头可以关注不同类型的信息
-   有的头关注语法，有的关注语义，有的关注位置...

```
┌─────────────────────────────────────────────────────────────┐
│                    Multi-Head Attention                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│      输入 X ───► Split into h heads ──► 并行计算注意力         │
│      │              │                    │                  │
│      │              ▼                    ▼                  │
│      │         ┌─────────┐         ┌─────────┐              │
│      │         │ Head 1  │         │ Head 2  │   ...        │
│      │         │ SelfAttn│         │ SelfAttn│              │
│      │         └────┬────┘         └────┬────┘              │
│      │              │                    │                  │
│      │              └──────► Concat ◄────┘                  │
│      │                         │                            │
│      │                         ▼                            │
│      │                   Linear ───► 输出                    │
│      │                                                      │
│      └─────────────────────────────────────► 残差连接        │
└─────────────────────────────────────────────────────────────┘
```
```python
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, num_heads, dropout=0.1):
        super().__init__()
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads
        
        self.attn = SelfAttention(embed_size, num_heads)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embed_size)
    
    def forward(self, x, mask=None):
        # 残差连接 + Layer Norm
        x_norm = self.layer_norm(x)
        attn_output, weights = self.attn(x_norm, x_norm, x_norm, mask)
        
        # 残差连接
        x = x + self.dropout(attn_output)
        
        return x, weights
```

* * *

## 3️⃣ 位置编码 (Positional Encoding)

### 为什么需要？

Transformer 没有循环结构，无法知道词语的顺序，所以需要显式加入位置信息。

```python
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # 创建位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)
        
        # 计算频率
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        
        # 奇偶位置使用 sin/cos
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数维度
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数维度
        
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)
```

### 位置编码可视化

```
位置 1: [sin, cos, sin, cos, ...]
位置 2: [sin², cos², sin², cos², ...]
位置 3: [sin³, cos³, sin³, cos³, ...]
         ↓
    每个位置有独特的编码模式
```

* * *

## 4️⃣ Feed Forward Network（前馈网络）

每个 Transformer block 中还有一个两层全连接网络：

```python
class FeedForward(nn.Module):
    def __init__(self, embed_size, ff_dim, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(embed_size, ff_dim)
        self.linear2 = nn.Linear(ff_dim, embed_size)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embed_size)
    
    def forward(self, x):
        # 残差连接
        x_norm = self.layer_norm(x)
        ff_output = self.linear2(self.dropout(F.relu(self.linear1(x_norm))))
        return x + self.dropout(ff_output)  # 残差连接
```

* * *

## 5️⃣ 完整的 Encoder Block

```python
class EncoderBlock(nn.Module):
    def __init__(self, embed_size, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(embed_size, num_heads, dropout)
        self.feed_forward = FeedForward(embed_size, ff_dim, dropout)
    
    def forward(self, x, mask=None):
        # 1. Multi-Head Self-Attention
        x, attn_weights = self.attention(x, mask)
        
        # 2. Feed Forward + 残差
        x = self.feed_forward(x)
        
        return x, attn_weights


class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_size, num_heads, num_layers, 
                 ff_dim, max_len, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.pos_encoding = PositionalEncoding(embed_size, max_len, dropout)
        
        self.layers = nn.ModuleList([
            EncoderBlock(embed_size, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])
        
        self.layer_norm = nn.LayerNorm(embed_size)
    
    def forward(self, x, mask=None):
        # 词嵌入 + 位置编码
        x = self.embedding(x) * math.sqrt(self.embed_size)
        x = self.pos_encoding(x)
        
        # 通过所有 Encoder 层
        attn_weights_list = []
        for layer in self.layers:
            x, attn_weights = layer(x, mask)
            attn_weights_list.append(attn_weights)
        
        return self.layer_norm(x), attn_weights_list
```

* * *

## 6️⃣ Decoder 结构

Decoder 比 Encoder 多了一个 **Cross-Attention** 层：

```python
class DecoderBlock(nn.Module):
    def __init__(self, embed_size, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(embed_size, num_heads, dropout)
        self.cross_attention = MultiHeadAttention(embed_size, num_heads, dropout)
        self.feed_forward = FeedForward(embed_size, ff_dim, dropout)
    
    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        # 1. Masked Self-Attention（防止看到未来位置）
        x, _ = self.self_attention(x, tgt_mask)
        
        # 2. Cross-Attention（关注 Encoder 输出）
        x, attn_weights = self.cross_attention(x, encoder_output, encoder_output, src_mask)
        
        # 3. Feed Forward
        x = self.feed_forward(x)
        
        return x, attn_weights


def create_tgt_mask(seq_len, device):
    """创建下三角掩码，防止看到未来位置"""
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
    return mask == 0
```

* * *

## 7️⃣ 完整的 Transformer

```python
class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, embed_size=512,
                 num_heads=8, num_layers=6, ff_dim=2048, 
                 max_len=5000, dropout=0.1):
        super().__init__()
        
        self.encoder = Encoder(src_vocab_size, embed_size, num_heads, 
                               num_layers, ff_dim, max_len, dropout)
        self.decoder = Decoder(tgt_vocab_size, embed_size, num_heads,
                               num_layers, ff_dim, max_len, dropout)
        self.output_layer = nn.Linear(embed_size, tgt_vocab_size)
    
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        # Encoder
        encoder_output, enc_attn = self.encoder(src, src_mask)
        
        # Decoder
        decoder_output, dec_attn = self.decoder(tgt, encoder_output, 
                                                src_mask, tgt_mask)
        
        # 输出层
        output = self.output_layer(decoder_output)
        
        return output, enc_attn, dec_attn
```

* * *

## 🔍 工作流程图解

```
输入: "The cat sat on the mat"
                    │
                    ▼
┌─────────────────────────────────────┐
│             Embedding + Pos Enc     │
└─────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────┐
│           Encoder Stack ×6          │
│  ┌─────────────────────────────┐    │
│  │  Multi-Head Self-Attn       │    │
│  │  (每个词关注所有词)          │    │
│  └─────────────────────────────┘    │
│              │                      │
│              ▼                      │
│  ┌─────────────────────────────┐    │
│  │  Feed Forward Network       │    │
│  └─────────────────────────────┘    │
└─────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────┐
│           Decoder Stack ×6          │
│  ┌─────────────────────────────┐    │
│  │  Masked Self-Attn           │    │
│  │  (只看前面)                  │    │
│  └─────────────────────────────┘    │
│              │                      │
│              ▼                      │
│  ┌─────────────────────────────┐    │
│  │  Cross-Attn (关注Encoder)   │    │
│  └─────────────────────────────┘    │
│              │                      │
│              ▼                      │
│  ┌─────────────────────────────┐    │
│  │  Feed Forward Network       │    │
│  └─────────────────────────────┘    │
└─────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────┐
│            Linear + Softmax         │
└─────────────────────────────────────┘
                    │
                    ▼
         输出概率分布
```

* * *

## 📊 Transformer vs 传统模型对比

| 特性 | RNN | CNN | Transformer |
| --- | --- | --- | --- |
| **并行计算** | ❌ 串行 | ✅ 部分并行 | ✅ 完全并行 |
| **长距离依赖** | ⚠️ 梯度消失 | ⚠️ 多层才能捕获 | ✅ 直接连接 |
| **全局信息** | ❌ 难以捕获 | ⚠️ 需多空洞卷积 | ✅ 自注意力 |
| **时间复杂度** | O(n) | O(n) | O(n²) |

* * *

## 💡 关键创新点总结

1.  **Self-Attention** - 任意位置直接交互，无视距离
2.  **Multi-Head** - 多角度理解，不同头学不同模式
3.  **位置编码** - 弥补无序列结构的缺陷
4.  **残差连接** - 缓解深层网络训练困难
5.  **Layer Normalization** - 稳定训练

有任何具体部分想深入了解吗？比如注意力可视化的实际效果，或者训练技巧等？