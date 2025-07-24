import torch
import torch.optim as optim
import torch.nn as nn
from wawa_transformer import Transformer
import numpy as np

# -------------------------- 数据准备 --------------------------
# 生成训练数据：数字序列 -> 字母序列（1->A, 2->B, ..., 5->E）
# 输入格式：[1,2,3] -> 输出格式：[<SOS>,A,B,C,<EOS>] （<SOS>=6, <EOS>=7）
train_data = [
    (torch.tensor([1, 2, 3]), torch.tensor([6, 1, 2, 3, 7])),  # 1,2,3 -> SOS,A,B,C,EOS
    (torch.tensor([2, 4, 5]), torch.tensor([6, 2, 4, 5, 7])),  # 2,4,5 -> SOS,B,D,E,EOS
    (torch.tensor([1, 5]), torch.tensor([6, 1, 5, 7])),        # 1,5 -> SOS,A,E,EOS
    (torch.tensor([3, 4, 2, 1]), torch.tensor([6, 3, 4, 2, 1, 7])),
]

# 词汇表大小（源：6种，目标：8种）
src_vocab_size = 6
tgt_vocab_size = 8

# -------------------------- 掩码工具函数 --------------------------
def create_pad_mask(seq, pad_idx=0):
    """生成填充掩码（屏蔽PAD位置）"""
    return (seq != pad_idx).unsqueeze(1).unsqueeze(2)  # 形状：(batch_size, 1, 1, seq_len)

def create_target_mask(seq):
    """生成目标序列掩码（防止解码器关注未来信息）"""
    batch_size, seq_len = seq.size()
    # 上三角矩阵（对角线及以下为1，以上为0）
    mask = (torch.triu(torch.ones(seq_len, seq_len)) == 1).transpose(0, 1)
    mask = mask.unsqueeze(0).unsqueeze(0)  # 形状：(1, 1, seq_len, seq_len)
    return mask & create_pad_mask(seq)  # 结合填充掩码

# -------------------------- 训练配置 --------------------------
model = Transformer(
    src_vocab_size=src_vocab_size,
    tgt_vocab_size=tgt_vocab_size,
    d_model=64,  # 简化模型，减小维度
    num_layers=2,
    num_heads=2,
    d_ff=128
)
criterion = nn.CrossEntropyLoss(ignore_index=0)  # 忽略PAD的损失
optimizer = optim.Adam(model.parameters(), lr=0.001)
epochs = 100  # 训练轮次

# -------------------------- 训练过程 --------------------------
for epoch in range(epochs):
    model.train()
    total_loss = 0
    
    for src, tgt in train_data:
        # 处理输入：增加batch维度（batch_size=1）
        src = src.unsqueeze(0)  # 形状：(1, src_seq_len)
        tgt_input = tgt[:-1].unsqueeze(0)  # 目标输入（不含最后一个EOS）
        tgt_label = tgt[1:].unsqueeze(0)   # 目标标签（不含第一个SOS）
        
        # 生成掩码
        src_mask = create_pad_mask(src)
        tgt_mask = create_target_mask(tgt_input)
        cross_mask = src_mask  # 编码器-解码器掩码复用源序列掩码
        
        # 前向传播
        output = model(src, tgt_input, src_mask, tgt_mask, cross_mask)
        
        # 计算损失
        loss = criterion(
            output.contiguous().view(-1, tgt_vocab_size),  # 展平为(batch*seq_len, vocab)
            tgt_label.contiguous().view(-1)  # 展平为(batch*seq_len,)
        )
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    # 每20轮打印一次损失
    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_data):.4f}")

# -------------------------- 测试推理 --------------------------
def translate(src_seq):
    """用训练好的模型生成目标序列"""
    model.eval()
    with torch.no_grad():
        src = src_seq.unsqueeze(0)  # 增加batch维度
        src_mask = create_pad_mask(src)
        
        # 初始化目标序列（从<SOS>开始）
        tgt_seq = torch.tensor([[6]], dtype=torch.long)  # 6是<SOS>
        
        # 生成序列（最多生成10个token）
        for _ in range(10):
            tgt_mask = create_target_mask(tgt_seq)
            output = model(src, tgt_seq, src_mask, tgt_mask, src_mask)
            next_token = output[:, -1, :].argmax(dim=-1, keepdim=True)  # 取最后一个token的预测
            tgt_seq = torch.cat([tgt_seq, next_token], dim=1)
            
            # 如果生成EOS，停止
            if next_token.item() == 7:
                break
        
        return tgt_seq.squeeze(0).tolist()  # 去除batch维度

# 测试案例：输入 [3,4]（预期输出：SOS,C,D,EOS -> [6,3,4,7]）
test_src = torch.tensor([3, 4])
predicted_tgt = translate(test_src)

# 映射为字母（1→A,2→B,3→C,4→D,5→E,6→SOS,7→EOS）
char_map = {1:'A',2:'B',3:'C',4:'D',5:'E',6:'<SOS>',7:'<EOS>'}
predicted_chars = [char_map[token] for token in predicted_tgt]

print("\n测试输入（数字）：", test_src.tolist())
print("模型输出（字母）：", predicted_chars)