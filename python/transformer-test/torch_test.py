import torch
import numpy as np

def test_arange():
    """测试torch.arange函数的各种用法"""
    print("=" * 50)
    print("测试 torch.arange 函数")
    print("=" * 50)
    
    # 基本用法：生成0到9的整数
    tensor1 = torch.arange(10)
    print(f"arange(10): {tensor1}")
    print(f"shape: {tensor1.shape}, dtype: {tensor1.dtype}\n")
    
    # 指定起始和结束值
    tensor2 = torch.arange(2, 10)
    print(f"arange(2, 10): {tensor2}")
    print(f"shape: {tensor2.shape}\n")
    
    # 指定步长
    tensor3 = torch.arange(0, 10, 2)
    print(f"arange(0, 10, 2): {tensor3}")
    print(f"shape: {tensor3.shape}\n")
    
    # 浮点数范围
    tensor4 = torch.arange(0.0, 5.0, 0.5)
    print(f"arange(0.0, 5.0, 0.5): {tensor4}")
    print(f"shape: {tensor4.shape}, dtype: {tensor4.dtype}\n")
    
    # 指定数据类型
    tensor5 = torch.arange(10, dtype=torch.float32)
    print(f"arange(10, dtype=float32): {tensor5}")
    print(f"dtype: {tensor5.dtype}\n")
    
    # 指定设备（如果有GPU的话）
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tensor6 = torch.arange(5, device=device)
    print(f"arange(5, device={device}): {tensor6}")
    print(f"device: {tensor6.device}\n")


def test_unsqueeze():
    """测试torch.unsqueeze函数的各种用法"""
    print("=" * 50)
    print("测试 torch.unsqueeze 函数")
    print("=" * 50)
    
    # 创建一个基础张量
    base_tensor = torch.arange(6)
    print(f"原始张量: {base_tensor}")
    print(f"原始形状: {base_tensor.shape}\n")
    
    # 在不同维度上添加维度
    for dim in range(-2, 2):
        try:
            unsqueezed = torch.unsqueeze(base_tensor, dim)
            print(f"unsqueeze(dim={dim}): {unsqueezed}")
            print(f"新形状: {unsqueezed.shape}\n")
        except Exception as e:
            print(f"dim={dim} 时出错: {e}\n")
    
    # 使用2D张量测试
    tensor_2d = torch.arange(12).reshape(3, 4)
    print(f"2D张量: \n{tensor_2d}")
    print(f"2D张量形状: {tensor_2d.shape}\n")
    
    # 在不同维度添加维度
    for dim in range(-4, 4):
        try:
            unsqueezed_2d = torch.unsqueeze(tensor_2d, dim)
            print(f"2D张量 unsqueeze(dim={dim}): 形状 {unsqueezed_2d.shape}")
        except Exception as e:
            print(f"2D张量 dim={dim} 时出错: {e}")
    
    print()


def test_combined_usage():
    """测试arange和unsqueeze的组合使用"""
    print("=" * 50)
    print("测试 arange 和 unsqueeze 的组合使用")
    print("=" * 50)
    
    # 创建位置索引（常用于Transformer等模型）
    seq_len = 8
    positions = torch.arange(seq_len)
    print(f"序列位置: {positions}")
    print(f"形状: {positions.shape}")
    
    # 为批处理添加维度
    positions_batch = positions.unsqueeze(0)  # 添加batch维度
    print(f"添加batch维度: {positions_batch}")
    print(f"新形状: {positions_batch.shape}")
    
    # 为特征维度添加维度
    positions_feature = positions.unsqueeze(-1)  # 添加feature维度
    print(f"添加feature维度: {positions_feature}")
    print(f"新形状: {positions_feature.shape}")
    
    # 同时添加多个维度
    positions_multi = positions.unsqueeze(0).unsqueeze(-1)
    print(f"添加batch和feature维度: 形状 {positions_multi.shape}")
    
    # 创建网格索引（用于2D位置编码）
    x_coords = torch.arange(4).unsqueeze(1)  # (4, 1)
    y_coords = torch.arange(3).unsqueeze(0)  # (1, 3)
    
    print(f"\nX坐标: \n{x_coords}")
    print(f"X坐标形状: {x_coords.shape}")
    print(f"Y坐标: \n{y_coords}")
    print(f"Y坐标形状: {y_coords.shape}")
    
    # 广播创建网格
    x_grid = x_coords.expand(4, 3)
    y_grid = y_coords.expand(4, 3)
    
    print(f"\nX网格: \n{x_grid}")
    print(f"Y网格: \n{y_grid}")


def test_practical_examples():
    """实际应用示例"""
    print("=" * 50)
    print("实际应用示例")
    print("=" * 50)
    
    # 示例1: 创建注意力掩码的位置索引
    seq_len = 5
    positions = torch.arange(seq_len)
    
    # 创建下三角掩码（用于因果注意力）
    mask_positions_i = positions.unsqueeze(1)  # (seq_len, 1)
    mask_positions_j = positions.unsqueeze(0)  # (1, seq_len)
    
    causal_mask = mask_positions_i >= mask_positions_j
    print("因果注意力掩码:")
    print(f"位置i (列): {mask_positions_i.squeeze()}")
    print(f"位置j (行): {mask_positions_j.squeeze()}")
    print(f"掩码 (i >= j): \n{causal_mask}")
    
    # 示例2: 批处理的序列索引
    batch_size = 3
    seq_len = 4
    
    # 为每个批次创建相同的位置索引
    positions = torch.arange(seq_len).unsqueeze(0).expand(batch_size, seq_len)
    print(f"\n批处理位置索引 (batch_size={batch_size}, seq_len={seq_len}):")
    print(positions)
    
    # 示例3: 创建频率索引（用于位置编码）
    d_model = 8
    freqs = torch.arange(0, d_model, 2)  # 偶数维度
    print(f"\n频率索引 (d_model={d_model}): {freqs}")
    
    # 计算位置编码的频率
    positions = torch.arange(seq_len).unsqueeze(1)  # (seq_len, 1)
    freqs = freqs.unsqueeze(0)  # (1, d_model//2)
    
    angles = positions * freqs
    print(f"角度矩阵形状: {angles.shape}")
    print(f"角度矩阵: \n{angles}")


if __name__ == "__main__":
    # 检查PyTorch版本
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"GPU数量: {torch.cuda.device_count()}")
    print()
    
    # 运行所有测试
    test_arange()
    test_unsqueeze()
    test_combined_usage()
    test_practical_examples()
    
    print("=" * 50)
    print("所有测试完成！")
    print("=" * 50)