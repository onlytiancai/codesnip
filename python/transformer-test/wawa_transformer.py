import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# 位置编码：为序列添加位置信息
class PositionalEncoding(nn.Module):
    """
    Implements the positional encoding as described in "Attention is All You Need".
    Args:
        d_model (int): The dimension of the model (embedding size). 决定每个位置编码向量的长度，需与输入特征维度一致。
        max_len (int, optional): The maximum length of input sequences. 默认为5000，表示能支持的最大序列长度。
        dropout (float, optional): Dropout probability applied after adding positional encoding. 默认为0.1，用于防止过拟合。
    Attributes:
        dropout (nn.Dropout): Dropout layer applied to the output.
        pe (torch.Tensor): Positional encoding buffer of shape (max_len, 1, d_model).
    Methods:
        forward(x):
            Adds positional encoding to the input tensor and applies dropout.
            Args:
                x (torch.Tensor): Input tensor of shape (seq_len, batch_size, d_model).
            Returns:
                torch.Tensor: Output tensor with positional encoding added, same shape as input.
    """

    def __init__(self, d_model, max_len=5000, dropout=0.1):
        # 位置编码采用三角函数（正弦和余弦），其核心思想是为每个序列位置生成唯一的向量，
        # 并且这种编码方式允许模型在推理时泛化到比训练时更长的序列。
        # 数学原理如下：
        # 1. 对于每个位置 pos 和每个维度 i，编码定义为：
        #    PE(pos, 2i)   = sin(pos / (10000^(2i/d_model)))
        #    PE(pos, 2i+1) = cos(pos / (10000^(2i/d_model)))
        # 2. 这样设计有两个好处：
        #    a) 不同位置的编码向量彼此不同，且相似位置的编码向量距离较近，有助于模型捕捉相对和绝对位置信息。
        #    b) 任意两个位置的编码可以通过线性变换表达其相对距离（即 sin(a+b) = sin(a)cos(b) + cos(a)sin(b)），
        #       这使得模型能推断出序列中元素之间的相对顺序。
        # 3. 采用不同频率的三角函数（通过指数缩放）可以让模型在不同维度上感知不同粒度的位置信息，
        #    低维度变化慢，关注全局顺序，高维度变化快，关注局部顺序。
        
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # 初始化位置编码矩阵
        # position: 创建一个形状为 (max_len, 1) 的张量，表示序列中每个位置的索引（从0到max_len-1）
        # 例如，如果max_len=5，则position为[[0], [1], [2], [3], [4]]
        position = torch.arange(max_len).unsqueeze(1)
        
        # div_term: 形状为 (d_model/2,) 的张量，用于控制不同维度的正弦/余弦函数的频率
        # torch.arange(0, d_model, 2) 生成偶数索引（如0,2,4,...），
        # -math.log(10000.0) / d_model 是缩放因子，使得高维度变化更快
        # 这样每个维度的周期都不同，有助于模型区分不同位置
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        
        # pe: 初始化一个全零张量，形状为 (max_len, 1, d_model)
        # 用于存储每个位置的编码，1是为了后续和batch维度对齐
        pe = torch.zeros(max_len, 1, d_model)
        
        # 偶数维度（0,2,4,...）用正弦函数编码位置信息
        # position * div_term 会广播成 (max_len, d_model/2) 的形状
        # 这样每个位置的每个偶数维度都有唯一的正弦值
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        
        # 奇数维度（1,3,5,...）用余弦函数编码位置信息
        # 这样每个位置的每个奇数维度都有唯一的余弦值
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        
        # register_buffer: 把pe注册为模型的“缓冲区”
        # 这样pe不会作为参数参与训练（不会被优化），但会随模型保存和加载
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (seq_len, batch_size, d_model)
        # 将位置编码加到输入x上，pe[:x.size(0)]会自动匹配序列长度
        # 这样每个位置都获得了唯一的位置信息
        x = x + self.pe[:x.size(0)]
        # 应用dropout，防止过拟合
        return self.dropout(x)

# 多头注意力机制
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        """
        初始化多头自注意力机制的参数。
        Args:
            d_model (int): 输入特征的维度，即模型的隐藏层维度。
            num_heads (int): 注意力头的数量，将输入特征分为多少个头进行并行计算。
        Raises:
            AssertionError: 如果 d_model 不能被 num_heads 整除，则抛出异常。
        Attributes:
            d_model (int): 输入特征的维度。
            num_heads (int): 注意力头的数量。
            d_k (int): 每个注意力头的特征维度，等于 d_model // num_heads。
            w_q (nn.Linear): 查询（Q）的线性变换层。
            w_k (nn.Linear): 键（K）的线性变换层。
            w_v (nn.Linear): 值（V）的线性变换层。
            w_o (nn.Linear): 输出的线性变换层。
        """
        super().__init__()
        assert d_model % num_heads == 0, "d_model必须能被num_heads整除"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # 每个头的维度
        
        # 线性变换层（Q、K、V和输出）
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        
        # 线性变换并分多头 (batch_size, num_heads, seq_len, d_k)
        q = self.w_q(q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.w_k(k).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.w_v(v).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # 计算注意力分数 (batch_size, num_heads, seq_len_q, seq_len_k)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)  # 掩码位置设为负无穷
        
        # 注意力权重归一化
        attn = F.softmax(scores, dim=-1)
        
        # 加权求和并拼接多头结果
        output = torch.matmul(attn, v)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        return self.w_o(output), attn

# 前馈神经网络
class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.fc2(self.dropout(F.relu(self.fc1(x))))

# 编码器层
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask):
        # 自注意力子层
        attn_output, _ = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout1(attn_output))
        
        # 前馈子层
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout2(ff_output))
        return x

# 解码器层
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)  # 掩蔽自注意力
        self.cross_attn = MultiHeadAttention(d_model, num_heads)  # 编码器-解码器注意力
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x, enc_output, self_mask, cross_mask):
        # 掩蔽自注意力子层（防止关注未来信息）
        attn_output, _ = self.self_attn(x, x, x, self_mask)
        x = self.norm1(x + self.dropout1(attn_output))
        
        # 编码器-解码器注意力子层
        attn_output, _ = self.cross_attn(x, enc_output, enc_output, cross_mask)
        x = self.norm2(x + self.dropout2(attn_output))
        
        # 前馈子层
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout3(ff_output))
        return x

# 完整Transformer模型
class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, num_layers=6,
                 num_heads=8, d_ff=2048, dropout=0.1):
        super().__init__()
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, dropout=dropout)
        
        # 堆叠编码器层和解码器层
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])
        
        self.fc = nn.Linear(d_model, tgt_vocab_size)  # 输出层

    def forward(self, src, tgt, src_mask, tgt_mask, cross_mask):
        # 编码器部分
        enc_output = self.encoder_embedding(src)
        enc_output = self.pos_encoding(enc_output.transpose(0, 1)).transpose(0, 1)  # 调整维度
        for layer in self.encoder_layers:
            enc_output = layer(enc_output, src_mask)
        
        # 解码器部分
        dec_output = self.decoder_embedding(tgt)
        dec_output = self.pos_encoding(dec_output.transpose(0, 1)).transpose(0, 1)
        for layer in self.decoder_layers:
            dec_output = layer(dec_output, enc_output, tgt_mask, cross_mask)
        
        # 输出预测结果
        return self.fc(dec_output)

# 示例用法
if __name__ == "__main__":
    src_vocab_size = 1000  # 源语言词汇表大小
    tgt_vocab_size = 1000  # 目标语言词汇表大小
    model = Transformer(src_vocab_size, tgt_vocab_size)
    
    # 随机生成输入（batch_size=2, src_seq_len=10, tgt_seq_len=8）
    src = torch.randint(0, src_vocab_size, (2, 10))
    tgt = torch.randint(0, tgt_vocab_size, (2, 8))
    
    # 生成掩码（简单示例，实际需根据padding和序列方向生成）
    src_mask = torch.ones(2, 1, 1, 10)  # 编码器掩码
    tgt_mask = torch.ones(2, 1, 8, 8)   # 解码器自注意力掩码
    cross_mask = torch.ones(2, 1, 8, 10)# 编码器-解码器掩码
    
    # 模型前向传播
    output = model(src, tgt, src_mask, tgt_mask, cross_mask)
    print("输出形状:", output.shape)  # 应为 (2, 8, 1000)