import torch
import torch.nn as nn
import math
import numpy as np

def detailed_positional_encoding_demo():
    """
    详细演示位置编码(PositionalEncoding)的__init__函数中每一步的计算过程
    
    位置编码是Transformer模型的核心组件之一，它为序列中的每个位置生成唯一的向量，
    让模型能够理解元素在序列中的位置信息。
    """
    print("=" * 80)
    print("详细演示位置编码(PositionalEncoding)的计算过程")
    print("=" * 80)
    
    # 设置参数 - 这些参数与原始代码中的__init__函数参数对应
    d_model = 8    # 模型维度，即每个位置编码向量的长度（原代码中的d_model）
    max_len = 10   # 最大序列长度（原代码中的max_len）
    dropout = 0.1  # Dropout概率（原代码中的dropout）
    
    print(f"参数设置:")
    print(f"  d_model = {d_model}  # 模型维度，每个位置编码向量有{d_model}个数值")
    print(f"  max_len = {max_len}   # 最大序列长度，支持最多{max_len}个位置")
    print(f"  dropout = {dropout}  # Dropout概率，用于防止过拟合")
    print()
    
    # ========== 步骤1: 创建位置索引张量 ==========
    print("步骤1: 创建位置索引张量")
    print("-" * 40)
    
    # 原代码: position = torch.arange(max_len).unsqueeze(1)
    print("原代码: position = torch.arange(max_len).unsqueeze(1)")
    print()
    
    # 详细分解这一步
    print("1.1 首先创建位置索引序列:")
    position_1d = torch.arange(max_len)
    print(f"torch.arange({max_len}) = {position_1d}")
    print(f"  - 这创建了一个一维张量，包含从0到{max_len-1}的整数")
    print(f"  - 张量形状: {position_1d.shape}")
    print(f"  - 张量类型: {position_1d.dtype}")
    print()
    
    print("1.2 然后使用unsqueeze(1)增加一个维度:")
    position = position_1d.unsqueeze(1)
    print(f"position_1d.unsqueeze(1) = ")
    print(position)
    print(f"  - unsqueeze(1)在第1个位置（从0开始计数）插入一个新维度")
    print(f"  - 原形状: {position_1d.shape} -> 新形状: {position.shape}")
    print(f"  - 这样做是为了后续的广播运算（broadcasting）")
    print()
    
    # ========== 步骤2: 计算频率分母项 ==========
    print("步骤2: 计算频率分母项（div_term）")
    print("-" * 40)
    
    # 原代码: div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
    print("原代码: div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))")
    print()
    
    print("2.1 创建偶数索引序列:")
    even_indices = torch.arange(0, d_model, 2)
    print(f"torch.arange(0, {d_model}, 2) = {even_indices}")
    print(f"  - 从0开始，到{d_model-1}结束，步长为2，得到偶数索引")
    print(f"  - 这些索引对应sin函数的维度位置")
    print(f"  - 张量形状: {even_indices.shape}")
    print()
    
    print("2.2 计算缩放因子:")
    log_10000 = math.log(10000.0)
    scale_factor = -log_10000 / d_model
    print(f"math.log(10000.0) = {log_10000:.6f}")
    print(f"scale_factor = -math.log(10000.0) / d_model = {scale_factor:.6f}")
    print(f"  - 这个因子控制不同维度的频率变化速度")
    print(f"  - 负号使得频率随维度增加而递减")
    print()
    
    print("2.3 计算每个维度的指数:")
    exponents = even_indices * scale_factor
    print(f"exponents = even_indices * scale_factor = {exponents}")
    print(f"  - 将偶数索引乘以缩放因子")
    print(f"  - 这些值将作为exp函数的输入")
    print()
    
    print("2.4 计算最终的频率分母项:")
    div_term = torch.exp(exponents)
    print(f"div_term = torch.exp(exponents) = {div_term}")
    print(f"  - 对每个指数取自然指数")
    print(f"  - 这些值控制sin/cos函数的频率")
    print(f"  - 值越大，频率越高，变化越快")
    print(f"  - 张量形状: {div_term.shape}")
    print()
    
    # ========== 步骤3: 初始化位置编码矩阵 ==========
    print("步骤3: 初始化位置编码矩阵")
    print("-" * 40)
    
    # 原代码: pe = torch.zeros(max_len, 1, d_model)
    print("原代码: pe = torch.zeros(max_len, 1, d_model)")
    print()
    
    pe = torch.zeros(max_len, 1, d_model)
    print(f"pe = torch.zeros({max_len}, 1, {d_model})")
    print(f"pe.shape = {pe.shape}")
    print(f"  - 创建一个全零的三维张量")
    print(f"  - 第0维({max_len}): 序列长度，每个位置一行")
    print(f"  - 第1维(1): 批次维度，这里设为1是为了后续广播")
    print(f"  - 第2维({d_model}): 特征维度，每个位置编码的向量长度")
    print()
    print("初始的pe矩阵（全零）:")
    print(pe.squeeze())  # 去掉中间的维度1，便于查看
    print()
    
    # ========== 步骤4: 计算正弦编码（偶数维度） ==========
    print("步骤4: 计算正弦编码（偶数维度）")
    print("-" * 40)
    
    # 原代码: pe[:, 0, 0::2] = torch.sin(position * div_term)
    print("原代码: pe[:, 0, 0::2] = torch.sin(position * div_term)")
    print()
    
    print("4.1 理解position * div_term的广播运算:")
    print(f"position.shape = {position.shape}  # ({max_len}, 1)")
    print(f"div_term.shape = {div_term.shape}  # ({len(div_term)},)")
    print()
    print("position:")
    print(position)
    print()
    print("div_term:")
    print(div_term)
    print()
    
    print("4.2 执行广播乘法:")
    position_div_product = position * div_term
    print("position * div_term = ")
    print(position_div_product)
    print(f"  - 结果形状: {position_div_product.shape}")
    print(f"  - 每一行代表一个位置，每一列代表一个频率")
    print(f"  - position[i] * div_term[j] = 位置i在频率j下的角度")
    print()
    
    print("4.3 计算正弦值:")
    sin_values = torch.sin(position_div_product)
    print("torch.sin(position * div_term) = ")
    print(sin_values)
    print(f"  - 对每个角度计算正弦值")
    print(f"  - 这些值将填入pe矩阵的偶数维度")
    print()
    
    print("4.4 将正弦值填入pe矩阵的偶数位置:")
    pe[:, 0, 0::2] = sin_values
    print("pe[:, 0, 0::2] = sin_values")
    print(f"  - 切片0::2表示从第0维开始，每隔2个取一个（即偶数位置）")
    print(f"  - 偶数维度索引: {list(range(0, d_model, 2))}")
    print()
    print("填入正弦值后的pe矩阵:")
    print(pe.squeeze())
    print()
    
    # ========== 步骤5: 计算余弦编码（奇数维度） ==========
    print("步骤5: 计算余弦编码（奇数维度）")
    print("-" * 40)
    
    # 原代码: pe[:, 0, 1::2] = torch.cos(position * div_term)
    print("原代码: pe[:, 0, 1::2] = torch.cos(position * div_term)")
    print()
    
    print("5.1 计算余弦值:")
    cos_values = torch.cos(position_div_product)
    print("torch.cos(position * div_term) = ")
    print(cos_values)
    print(f"  - 使用相同的角度计算余弦值")
    print(f"  - 这些值将填入pe矩阵的奇数维度")
    print()
    
    print("5.2 将余弦值填入pe矩阵的奇数位置:")
    pe[:, 0, 1::2] = cos_values
    print("pe[:, 0, 1::2] = cos_values")
    print(f"  - 切片1::2表示从第1维开始，每隔2个取一个（即奇数位置）")
    print(f"  - 奇数维度索引: {list(range(1, d_model, 2))}")
    print()
    print("最终的位置编码矩阵:")
    print(pe.squeeze())
    print()
    
    # ========== 步骤6: 分析结果 ==========
    print("步骤6: 分析位置编码结果")
    print("-" * 40)
    
    print("6.1 维度含义分析:")
    for dim in range(d_model):
        if dim % 2 == 0:  # 偶数维度
            freq_idx = dim // 2
            freq = div_term[freq_idx].item()
            print(f"  维度{dim} (偶数): sin函数，频率={freq:.6f}，周期={2*math.pi/freq:.2f}")
        else:  # 奇数维度
            freq_idx = dim // 2
            freq = div_term[freq_idx].item()
            print(f"  维度{dim} (奇数): cos函数，频率={freq:.6f}，周期={2*math.pi/freq:.2f}")
    print()
    
    print("6.2 每个位置的编码向量:")
    pe_final = pe.squeeze()
    for pos in range(min(5, max_len)):  # 只显示前5个位置
        print(f"  位置{pos}: {pe_final[pos].tolist()}")
    print()
    
    print("6.3 相邻位置的差异:")
    for pos in range(min(3, max_len-1)):
        diff = pe_final[pos+1] - pe_final[pos]
        print(f"  位置{pos+1} - 位置{pos} = {diff.tolist()}")
    print()
    
    # ========== 步骤7: 模拟真实使用场景 ==========
    print("步骤7: 模拟在神经网络中的使用")
    print("-" * 40)
    
    print("7.1 创建Dropout层:")
    dropout_layer = nn.Dropout(p=dropout)
    print(f"dropout_layer = nn.Dropout(p={dropout})")
    print(f"  - Dropout层会随机将一些元素设为0，概率为{dropout}")
    print(f"  - 这有助于防止过拟合")
    print()
    
    print("7.2 模拟输入数据:")
    # 假设输入是一个词嵌入矩阵，形状为(seq_len, batch_size, d_model)
    seq_len = 5
    batch_size = 2
    x = torch.randn(seq_len, batch_size, d_model)
    print(f"输入x的形状: {x.shape}")
    print("输入x (随机词嵌入):")
    print(x)
    print()
    
    print("7.3 添加位置编码:")
    # 取对应长度的位置编码
    pe_for_input = pe[:seq_len]  # 形状: (seq_len, 1, d_model)
    print(f"pe_for_input的形状: {pe_for_input.shape}")
    print("pe_for_input:")
    print(pe_for_input.squeeze())
    print()
    
    print("7.4 广播相加:")
    x_with_pe = x + pe_for_input
    print("x + pe_for_input 的结果:")
    print(x_with_pe)
    print(f"  - 形状: {x_with_pe.shape}")
    print(f"  - pe_for_input会自动广播到匹配x的batch_size维度")
    print()
    
    print("7.5 应用Dropout:")
    # 注意：在训练模式下dropout才会生效
    dropout_layer.train()
    x_final = dropout_layer(x_with_pe)
    print("应用dropout后的结果:")
    print(x_final)
    print(f"  - 一些元素可能被设为0（取决于随机性）")
    print()
    
    # ========== 总结 ==========
    print("=" * 80)
    print("总结：位置编码的核心思想")
    print("=" * 80)
    print("""
1. 位置信息的重要性:
   - Transformer没有循环结构，无法自然感知位置信息
   - 位置编码为每个位置提供唯一的"身份证"

2. 数学设计的巧妙性:
   - 使用sin/cos函数，保证每个位置的编码都不同
   - 不同频率的组合，让模型能感知不同尺度的位置关系
   - 三角函数的周期性质，有助于模型理解相对位置

3. 实现细节:
   - position张量: 位置索引(0, 1, 2, ...)
   - div_term张量: 频率控制(高维度变化快，低维度变化慢)
   - 偶数维度用sin，奇数维度用cos，配对形成复数样式
   - 通过广播运算高效计算所有位置-频率组合

4. 使用方式:
   - 直接加到词嵌入上，不需要额外的参数训练
   - 通过register_buffer保存，不参与梯度更新
   - 支持任意长度的序列（只要不超过max_len）

这种设计让Transformer能够理解"第3个词"、"距离当前词5个位置"等概念，
是Transformer成功的关键因素之一。
""")

if __name__ == "__main__":
    detailed_positional_encoding_demo()
