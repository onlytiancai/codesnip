import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from wawa_transformer import PositionalEncoding
import math

# 设置matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Arial Unicode MS']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 测试位置编码的基本功能和特性
def test_positional_encoding():
    print("=" * 60)
    print("位置编码(PositionalEncoding)原理测试")
    print("=" * 60)
    
    # 测试参数
    d_model = 512
    max_len = 100
    
    # 创建位置编码实例
    pos_encoding = PositionalEncoding(d_model, max_len, dropout=0.0)
    
    print(f"模型维度 d_model: {d_model}")
    print(f"最大序列长度 max_len: {max_len}")
    print(f"位置编码矩阵形状: {pos_encoding.pe.shape}")
    
    # 测试1: 验证位置编码的基本性质
    print("\n" + "="*50)
    print("测试1: 位置编码的基本性质")
    print("="*50)
    
    # 创建一个简单的输入序列
    seq_len = 10
    batch_size = 2
    x = torch.randn(seq_len, batch_size, d_model)
    
    # 应用位置编码
    output = pos_encoding(x)
    
    print(f"输入序列形状: {x.shape}")
    print(f"输出序列形状: {output.shape}")
    print(f"位置编码是否改变了序列长度: {x.shape == output.shape}")
    
    # 测试2: 验证不同位置的编码是否不同
    print("\n" + "="*50)
    print("测试2: 验证不同位置的编码唯一性")
    print("="*50)
    
    # 提取前5个位置的编码
    pos_encodings = pos_encoding.pe[:5, 0, :]  # 形状: (5, d_model)
    
    # 计算任意两个位置编码的余弦相似度
    print("前5个位置编码的余弦相似度矩阵:")
    for i in range(5):
        similarities = []
        for j in range(5):
            if i == j:
                similarities.append(1.0)
            else:
                # 计算余弦相似度
                sim = torch.cosine_similarity(pos_encodings[i:i+1], pos_encodings[j:j+1])
                similarities.append(sim.item())
        print(f"位置{i}: {[f'{s:.3f}' for s in similarities]}")
    
    # 测试3: 验证三角函数编码的周期性
    print("\n" + "="*50)
    print("测试3: 三角函数编码的周期性分析")
    print("="*50)
    
    # 分析前几个维度的编码值
    positions = torch.arange(20)
    print("前20个位置在不同维度的编码值:")
    print("位置\\维度", end="\t")
    for dim in [0, 1, 2, 3, 10, 11]:
        print(f"dim_{dim}", end="\t")
    print()
    
    for pos in range(10):  # 只显示前10个位置
        print(f"pos_{pos}", end="\t\t")
        for dim in [0, 1, 2, 3, 10, 11]:
            val = pos_encoding.pe[pos, 0, dim].item()
            print(f"{val:.3f}", end="\t")
        print()
    
    # 测试4: 可视化位置编码模式
    print("\n" + "="*50)
    print("测试4: 位置编码的可视化分析")
    print("="*50)
    
    # 提取用于可视化的数据
    viz_len = 50
    viz_dims = 64
    viz_data = pos_encoding.pe[:viz_len, 0, :viz_dims].numpy()
    
    print(f"可视化数据形状: {viz_data.shape}")
    print("生成位置编码热力图...")
    
    # 创建热力图
    plt.figure(figsize=(12, 8))
    plt.imshow(viz_data.T, cmap='RdBu', aspect='auto')
    plt.colorbar(label='Encoding Value')
    plt.xlabel('Position')
    plt.ylabel('Dimension')
    plt.title('Positional Encoding Heatmap\n(Blue: Negative, Red: Positive)')
    plt.tight_layout()
    plt.savefig('/home/haohu/src/codesnip/python/transformer-test/positional_encoding_heatmap.png', dpi=150)
    print("热力图已保存为 positional_encoding_heatmap.png")
    
    # 测试5: 分析相对位置关系
    print("\n" + "="*50)
    print("测试5: 相对位置关系分析")
    print("="*50)
    
    # 验证位置编码是否能表达相对位置关系
    def analyze_relative_distance(pos1, pos2):
        """分析两个位置之间的编码距离"""
        enc1 = pos_encoding.pe[pos1, 0, :]
        enc2 = pos_encoding.pe[pos2, 0, :]
        
        # 欧几里得距离
        euclidean_dist = torch.norm(enc1 - enc2).item()
        # 余弦相似度
        cosine_sim = torch.cosine_similarity(enc1.unsqueeze(0), enc2.unsqueeze(0)).item()
        
        return euclidean_dist, cosine_sim
    
    print("位置对之间的编码距离分析:")
    print("位置对\t\t欧几里得距离\t余弦相似度")
    
    test_pairs = [(0, 1), (0, 2), (0, 5), (0, 10), (5, 6), (5, 10), (10, 20)]
    for pos1, pos2 in test_pairs:
        if pos2 < max_len:
            eucl_dist, cos_sim = analyze_relative_distance(pos1, pos2)
            print(f"({pos1}, {pos2})\t\t{eucl_dist:.3f}\t\t{cos_sim:.3f}")
    
    # 测试6: 手动验证三角函数公式
    print("\n" + "="*50)
    print("测试6: 手动验证三角函数编码公式")
    print("="*50)
    
    # 手动计算第5个位置的前几个维度的编码
    pos = 5
    print(f"手动计算位置 {pos} 的编码值:")
    print("维度\t预期值(手动)\t实际值(模型)\t误差")
    
    for i in range(0, min(8, d_model), 2):
        # 计算 div_term
        div_term = math.exp(i * (-math.log(10000.0) / d_model))
        
        # 偶数维度用sin
        expected_sin = math.sin(pos * div_term)
        actual_sin = pos_encoding.pe[pos, 0, i].item()
        
        # 奇数维度用cos
        if i + 1 < d_model:
            expected_cos = math.cos(pos * div_term)
            actual_cos = pos_encoding.pe[pos, 0, i + 1].item()
            
            print(f"{i}(sin)\t{expected_sin:.6f}\t{actual_sin:.6f}\t{abs(expected_sin - actual_sin):.8f}")
            print(f"{i+1}(cos)\t{expected_cos:.6f}\t{actual_cos:.6f}\t{abs(expected_cos - actual_cos):.8f}")
    
    # 测试7: 不同模型维度的影响
    print("\n" + "="*50)
    print("测试7: 不同模型维度对位置编码的影响")
    print("="*50)
    
    dimensions = [64, 128, 256, 512]
    pos = 10
    
    print("在位置10处，不同d_model的编码前4个维度值:")
    print("d_model\tdim_0\t\tdim_1\t\tdim_2\t\tdim_3")
    
    for dim in dimensions:
        pe_temp = PositionalEncoding(dim, max_len, dropout=0.0)
        values = pe_temp.pe[pos, 0, :4]
        print(f"{dim}\t{values[0]:.4f}\t\t{values[1]:.4f}\t\t{values[2]:.4f}\t\t{values[3]:.4f}")
    
    print("\n" + "="*60)
    print("位置编码测试完成!")
    print("关键发现:")
    print("1. 位置编码为每个位置生成唯一向量")
    print("2. 使用sin/cos函数确保平滑的位置关系")
    print("3. 不同频率的波形在不同维度上编码不同粒度的位置信息")
    print("4. 相邻位置的编码相似，距离远的位置编码差异大")
    print("5. 位置编码不需要训练，是固定的数学函数")
    print("="*60)

# 额外的可视化函数
def plot_specific_dimensions():
    """绘制特定维度的位置编码曲线"""
    print("\n生成特定维度的位置编码曲线图...")
    
    d_model = 512
    max_len = 100
    pos_encoding = PositionalEncoding(d_model, max_len, dropout=0.0)
    
    positions = np.arange(max_len)
    
    # 选择几个不同频率的维度来可视化
    dims_to_plot = [0, 1, 4, 5, 16, 17, 64, 65]
    
    plt.figure(figsize=(15, 10))
    
    for i, dim in enumerate(dims_to_plot):
        plt.subplot(2, 4, i + 1)
        values = pos_encoding.pe[:, 0, dim].numpy()
        plt.plot(positions, values)
        plt.title(f'Dimension {dim} {"(sin)" if dim % 2 == 0 else "(cos)"}')
        plt.xlabel('Position')
        plt.ylabel('Encoding Value')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/home/haohu/src/codesnip/python/transformer-test/positional_encoding_curves.png', dpi=150)
    print("维度曲线图已保存为 positional_encoding_curves.png")

if __name__ == "__main__":
    # 运行基本测试
    test_positional_encoding()
    
    # 生成可视化图表
    plot_specific_dimensions()
    
    print("\n所有测试完成! 查看生成的图片以更好地理解位置编码的工作原理。")
