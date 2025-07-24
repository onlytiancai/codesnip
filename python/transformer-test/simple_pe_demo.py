import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from wawa_transformer import PositionalEncoding
import math

# 设置matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

def simple_positional_encoding_demo():
    """简化的位置编码演示，重点理解核心概念"""
    print("=" * 60)
    print("位置编码(PositionalEncoding)核心概念演示")
    print("=" * 60)
    
    # 使用小参数便于理解
    d_model = 8  # 只用8维便于观察
    max_len = 10
    
    print(f"简化参数: d_model={d_model}, max_len={max_len}")
    print()
    
    # 创建位置编码
    pos_encoding = PositionalEncoding(d_model, max_len, dropout=0.0)
    
    # 显示完整的位置编码矩阵
    print("完整的位置编码矩阵 (位置 × 维度):")
    print("每行代表一个位置的编码向量")
    print("pos\\dim", end="")
    for dim in range(d_model):
        print(f"\tdim_{dim}", end="")
    print()
    
    for pos in range(max_len):
        print(f"pos_{pos}", end="")
        for dim in range(d_model):
            val = pos_encoding.pe[pos, 0, dim].item()
            print(f"\t{val:6.3f}", end="")
        print()
    
    print("\n" + "="*50)
    print("观察规律:")
    print("1. 偶数列(0,2,4,6)是sin函数值")
    print("2. 奇数列(1,3,5,7)是cos函数值") 
    print("3. 从左到右，波动频率越来越快")
    print("4. 每一行(位置)都有独特的编码模式")
    print("="*50)
    
    # 手动验证几个关键值
    print("\n手动验证位置编码公式:")
    print("公式: PE(pos,2i) = sin(pos/10000^(2i/d_model))")
    print("     PE(pos,2i+1) = cos(pos/10000^(2i/d_model))")
    print()
    
    test_pos = 3
    print(f"验证位置 {test_pos} 的编码:")
    for i in range(0, d_model, 2):
        # 计算div_term = 1 / 10000^(2i/d_model)
        div_term = math.exp(i * (-math.log(10000.0) / d_model))
        
        # sin值 (偶数维度)
        manual_sin = math.sin(test_pos * div_term)
        model_sin = pos_encoding.pe[test_pos, 0, i].item()
        
        print(f"维度{i} (sin): 手动计算={manual_sin:.6f}, 模型输出={model_sin:.6f}")
        
        # cos值 (奇数维度)
        if i + 1 < d_model:
            manual_cos = math.cos(test_pos * div_term)
            model_cos = pos_encoding.pe[test_pos, 0, i + 1].item()
            print(f"维度{i+1} (cos): 手动计算={manual_cos:.6f}, 模型输出={model_cos:.6f}")
    
    # 可视化不同维度的波形
    print("\n生成简化的波形图...")
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    positions = np.arange(max_len)
    
    for dim in range(d_model):
        row = dim // 4
        col = dim % 4
        
        values = pos_encoding.pe[:max_len, 0, dim].numpy()
        axes[row, col].plot(positions, values, 'o-', linewidth=2, markersize=6)
        
        func_type = "sin" if dim % 2 == 0 else "cos"
        freq_factor = dim // 2
        axes[row, col].set_title(f'Dim {dim} ({func_type})\nFreq factor: {freq_factor}')
        axes[row, col].set_xlabel('Position')
        axes[row, col].set_ylabel('Value')
        axes[row, col].grid(True, alpha=0.3)
        axes[row, col].set_ylim(-1.1, 1.1)
    
    plt.tight_layout()
    plt.savefig('/home/haohu/src/codesnip/python/transformer-test/simple_pe_demo.png', dpi=150)
    plt.show()
    print("简化演示图已保存为 simple_pe_demo.png")
    
    # 分析位置编码的实际应用效果
    print("\n" + "="*50)
    print("位置编码的实际应用效果分析:")
    print("="*50)
    
    # 模拟两个相同单词在不同位置的情况
    word_embedding = torch.randn(1, d_model)  # 假设某个单词的embedding
    
    print("假设我们有一个单词，它的embedding是固定的:")
    print(f"单词embedding: {word_embedding.squeeze().numpy()}")
    print()
    
    print("但这个单词出现在不同位置时，加上位置编码后的最终表示:")
    for pos in [0, 2, 5]:
        # 获取该位置的位置编码
        pos_enc = pos_encoding.pe[pos, 0, :].unsqueeze(0)
        
        # 单词embedding + 位置编码
        final_repr = word_embedding + pos_enc
        
        print(f"位置{pos}: {final_repr.squeeze().numpy()}")
    
    print("\n观察: 相同的单词在不同位置会有不同的最终表示!")
    print("这样Transformer就能区分'我爱你'和'你爱我'中'我'和'你'的不同位置。")

def demonstrate_position_similarity():
    """演示位置之间的相似性关系"""
    print("\n" + "="*60)
    print("位置相似性分析")
    print("="*60)
    
    d_model = 64
    max_len = 50
    pos_encoding = PositionalEncoding(d_model, max_len, dropout=0.0)
    
    # 选择一个参考位置
    ref_pos = 10
    ref_encoding = pos_encoding.pe[ref_pos, 0, :]
    
    print(f"以位置{ref_pos}为参考，计算其他位置与它的相似度:")
    print("位置\t欧氏距离\t余弦相似度")
    
    for pos in range(0, min(20, max_len)):
        if pos != ref_pos:
            current_encoding = pos_encoding.pe[pos, 0, :]
            
            # 欧氏距离
            euclidean = torch.norm(ref_encoding - current_encoding).item()
            
            # 余弦相似度
            cosine = torch.cosine_similarity(
                ref_encoding.unsqueeze(0), 
                current_encoding.unsqueeze(0)
            ).item()
            
            print(f"{pos}\t{euclidean:.3f}\t\t{cosine:.3f}")
    
    print(f"\n观察规律:")
    print(f"1. 离参考位置{ref_pos}越近的位置，欧氏距离越小，余弦相似度越大")
    print(f"2. 这种渐变的相似性帮助模型理解序列中的位置关系")
    print(f"3. 不会出现跳跃式的突变，保证了位置信息的平滑性")

if __name__ == "__main__":
    # 运行简化演示
    simple_positional_encoding_demo()
    
    # 演示位置相似性
    demonstrate_position_similarity()
    
    print("\n" + "="*60)
    print("总结: 位置编码的核心价值")
    print("="*60)
    print("1. 给每个位置分配独特但相关的'身份证'")
    print("2. 相近位置有相似的编码，远程位置编码差异较大")
    print("3. 使用数学函数(sin/cos)确保编码的规律性和可预测性")
    print("4. 不需要训练参数，是纯数学计算")
    print("5. 可以处理训练时未见过的更长序列")
    print("="*60)
