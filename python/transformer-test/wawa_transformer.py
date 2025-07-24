import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# 位置编码：为序列添加位置信息
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # 初始化位置编码矩阵
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)  # 偶数维度用正弦
        pe[:, 0, 1::2] = torch.cos(position * div_term)  # 奇数维度用余弦
        self.register_buffer('pe', pe)  # 不参与训练的参数

    def forward(self, x):
        # x: (seq_len, batch_size, d_model)
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

# 多头注意力机制
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
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