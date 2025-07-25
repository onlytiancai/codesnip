import torch
import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

def demonstrate_linear_transformation_property():
    """
    演示位置编码的线性变换特性：
    任意两个位置的编码可以通过线性变换表达其相对距离
    """
    print("=" * 80)
    print("位置编码的线性变换特性演示")
    print("=" * 80)
    
    # 简化参数便于理解
    d_model = 4  # 只用4维便于观察
    max_len = 10
    
    # 手动计算位置编码
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(max_len).unsqueeze(1).float()
    
    # 计算频率项
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                        -(np.log(10000.0) / d_model))
    
    print(f"频率项 div_term: {div_term}")
    print(f"对应周期: {2 * np.pi / div_term}")
    
    # 计算位置编码
    pe[:, 0::2] = torch.sin(position * div_term)  # 偶数维度用sin
    pe[:, 1::2] = torch.cos(position * div_term)  # 奇数维度用cos
    
    print("\n位置编码矩阵 (前5个位置):")
    print("位置 | dim0(sin) | dim1(cos) | dim2(sin) | dim3(cos)")
    print("-" * 55)
    for i in range(5):
        print(f"{i:3d}  | {pe[i,0]:8.4f} | {pe[i,1]:8.4f} | {pe[i,2]:8.4f} | {pe[i,3]:8.4f}")
    
    # 演示核心特性：相对位置的线性表达
    print("\n" + "=" * 50)
    print("核心特性演示：相对位置的线性表达")
    print("=" * 50)
    
    # 选择两个具体位置进行演示
    pos_a = 2
    pos_b = 5
    relative_distance = pos_b - pos_a  # 相对距离为3
    
    print(f"\n位置 {pos_a} 的编码: {pe[pos_a]}")
    print(f"位置 {pos_b} 的编码: {pe[pos_b]}")
    print(f"相对距离: {relative_distance}")
    
    # 关键：验证三角恒等式
    print(f"\n验证三角恒等式：sin(a+b) = sin(a)cos(b) + cos(a)sin(b)")
    print(f"验证三角恒等式：cos(a+b) = cos(a)cos(b) - sin(a)sin(b)")
    
    # 对于第一个频率（第0和1维度）
    freq_0 = div_term[0].item()
    
    # 位置a的角度
    angle_a = pos_a * freq_0
    # 位置b的角度  
    angle_b = pos_b * freq_0
    # 相对角度
    angle_diff = relative_distance * freq_0
    
    print(f"\n第一个频率 freq_0 = {freq_0:.6f}")
    print(f"位置{pos_a}的角度: {angle_a:.6f}")
    print(f"位置{pos_b}的角度: {angle_b:.6f}")
    print(f"角度差: {angle_diff:.6f}")
    
    # 验证sin恒等式
    sin_a = np.sin(angle_a)
    cos_a = np.cos(angle_a)
    sin_b = np.sin(angle_b)
    cos_b = np.cos(angle_b)
    sin_diff = np.sin(angle_diff)
    cos_diff = np.cos(angle_diff)
    
    # 使用三角恒等式计算位置b的sin值
    sin_b_calculated = sin_a * cos_diff + cos_a * sin_diff
    cos_b_calculated = cos_a * cos_diff - sin_a * sin_diff
    
    print(f"\n验证 sin({angle_b:.3f}):")
    print(f"  直接计算: {sin_b:.6f}")
    print(f"  恒等式计算: {sin_b_calculated:.6f}")
    print(f"  误差: {abs(sin_b - sin_b_calculated):.8f}")
    
    print(f"\n验证 cos({angle_b:.3f}):")
    print(f"  直接计算: {cos_b:.6f}")
    print(f"  恒等式计算: {cos_b_calculated:.6f}")
    print(f"  误差: {abs(cos_b - cos_b_calculated):.8f}")
    
    # 关键洞察：线性变换矩阵
    print(f"\n" + "=" * 50)
    print("关键洞察：线性变换矩阵")
    print("=" * 50)
    
    print(f"对于相对距离 {relative_distance}，我们可以构造一个线性变换矩阵：")
    
    # 构造旋转矩阵（对于每个频率）
    for freq_idx, freq in enumerate(div_term):
        angle = relative_distance * freq
        cos_angle = torch.cos(angle)
        sin_angle = torch.sin(angle)
        rotation_matrix = torch.tensor([
            [cos_angle, -sin_angle],
            [sin_angle, cos_angle]
        ])
        
        print(f"\n频率 {freq_idx} (freq={freq:.4f}) 的旋转矩阵:")
        print(f"[[{rotation_matrix[0,0]:8.4f}, {rotation_matrix[0,1]:8.4f}]")
        print(f" [{rotation_matrix[1,0]:8.4f}, {rotation_matrix[1,1]:8.4f}]]")
        
        # 验证变换
        pos_a_pair = pe[pos_a, freq_idx*2:freq_idx*2+2]  # [sin, cos]
        pos_b_pair = pe[pos_b, freq_idx*2:freq_idx*2+2]  # [sin, cos]
        
        # 应用线性变换
        transformed = torch.matmul(rotation_matrix, pos_a_pair)
        
        print(f"位置{pos_a}的编码对: [{pos_a_pair[0]:8.4f}, {pos_a_pair[1]:8.4f}]")
        print(f"变换后的结果: [{transformed[0]:8.4f}, {transformed[1]:8.4f}]")
        print(f"位置{pos_b}的编码对: [{pos_b_pair[0]:8.4f}, {pos_b_pair[1]:8.4f}]")
        print(f"变换误差: {torch.norm(transformed - pos_b_pair):.8f}")

def demonstrate_relative_position_learning():
    """
    演示模型如何利用这种特性学习相对位置关系
    """
    print(f"\n" + "=" * 50)
    print("模型如何利用相对位置信息")
    print("=" * 50)
    
    d_model = 8
    max_len = 20
    
    # 创建位置编码
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(max_len).unsqueeze(1).float()
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                        -(np.log(10000.0) / d_model))
    
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    
    # 模拟模型学习到的线性变换参数
    print("假设模型学习到检测'相对距离为3'的线性变换...")
    
    # 检查所有相对距离为3的位置对
    relative_dist = 3
    pairs = [(i, i + relative_dist) for i in range(max_len - relative_dist)]
    
    print(f"\n相对距离为{relative_dist}的位置对及其变换结果:")
    print("位置对 | 变换前    | 变换后    | 目标位置  | 误差")
    print("-" * 60)
    
    for i, (pos_a, pos_b) in enumerate(pairs[:5]):  # 只显示前5对
        # 对于第一个频率维度
        freq = div_term[0]
        angle = relative_dist * freq
        cos_angle = torch.cos(angle)
        sin_angle = torch.sin(angle)
        rotation_matrix = torch.tensor([
            [cos_angle, -sin_angle],
            [sin_angle, cos_angle]
        ])
        
        pos_a_encoding = pe[pos_a, 0:2]  # 前两维
        pos_b_encoding = pe[pos_b, 0:2]  # 前两维
        transformed = torch.matmul(rotation_matrix, pos_a_encoding)
        error = torch.norm(transformed - pos_b_encoding)
        
        print(f"({pos_a:2d},{pos_b:2d}) | [{pos_a_encoding[0]:6.3f},{pos_a_encoding[1]:6.3f}] | "
              f"[{transformed[0]:6.3f},{transformed[1]:6.3f}] | [{pos_b_encoding[0]:6.3f},{pos_b_encoding[1]:6.3f}] | {error:.6f}")

def create_visualization():
    """
    创建可视化图表来展示线性变换特性
    """
    print("Generating visualization charts...")
    
    d_model = 4
    max_len = 10
    
    # Calculate position encoding
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(max_len).unsqueeze(1).float()
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                        -(np.log(10000.0) / d_model))
    
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Position encoding sin/cos values vs position
    positions = range(max_len)
    ax1.plot(positions, pe[:, 0], 'o-', label='Dimension 0 (sin)', linewidth=2)
    ax1.plot(positions, pe[:, 1], 's-', label='Dimension 1 (cos)', linewidth=2)
    ax1.set_title('Position Encoding Values vs Position')
    ax1.set_xlabel('Position')
    ax1.set_ylabel('Encoding Value')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Representation in complex plane (first two dimensions)
    ax2.scatter(pe[:, 0], pe[:, 1], c=positions, cmap='viridis', s=100)
    for i in range(max_len):
        ax2.annotate(f'{i}', (pe[i, 0], pe[i, 1]), 
                    xytext=(5, 5), textcoords='offset points')
    ax2.set_title('Position Encoding in Complex Plane')
    ax2.set_xlabel('Dimension 0 (sin)')
    ax2.set_ylabel('Dimension 1 (cos)')
    ax2.grid(True, alpha=0.3)
    ax2.axis('equal')
    
    # 3. Linear transformation demo: relative distance of 3
    relative_dist = 3
    freq = div_term[0]
    angle = relative_dist * freq
    
    # Original positions
    orig_x = pe[1:max_len-relative_dist, 0]
    orig_y = pe[1:max_len-relative_dist, 1]
    
    # Transformed positions
    cos_angle = torch.cos(angle)
    sin_angle = torch.sin(angle)
    rotation_matrix = torch.tensor([
        [cos_angle, -sin_angle],
        [sin_angle, cos_angle]
    ])
    
    transformed_points = []
    for i in range(1, max_len-relative_dist):
        point = pe[i, 0:2]
        transformed = torch.matmul(rotation_matrix, point)
        transformed_points.append(transformed)
    
    transformed_points = torch.stack(transformed_points)
    
    # Target positions (actual positions with relative distance of 3)
    target_x = pe[1+relative_dist:max_len, 0]
    target_y = pe[1+relative_dist:max_len, 1]
    
    ax3.scatter(orig_x, orig_y, c='blue', label='Original positions', s=100, alpha=0.7)
    ax3.scatter(transformed_points[:, 0], transformed_points[:, 1], 
               c='red', label='Transformed positions', s=100, alpha=0.7, marker='^')
    ax3.scatter(target_x, target_y, c='green', label='Target positions', s=100, alpha=0.7, marker='s')
    
    # Draw arrows to show transformation
    for i in range(len(orig_x)):
        ax3.arrow(orig_x[i], orig_y[i], 
                 transformed_points[i, 0] - orig_x[i], 
                 transformed_points[i, 1] - orig_y[i],
                 head_width=0.02, head_length=0.03, fc='orange', ec='orange', alpha=0.6)
    
    ax3.set_title(f'Linear Transformation Demo: Relative Distance = {relative_dist}')
    ax3.set_xlabel('Dimension 0 (sin)')
    ax3.set_ylabel('Dimension 1 (cos)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Transformation errors for different relative distances
    relative_distances = range(1, 6)
    errors = []
    
    for rd in relative_distances:
        angle = rd * freq
        cos_angle = torch.cos(angle)
        sin_angle = torch.sin(angle)
        rotation_matrix = torch.tensor([
            [cos_angle, -sin_angle],
            [sin_angle, cos_angle]
        ])
        
        total_error = 0
        count = 0
        for i in range(max_len - rd):
            pos_a_encoding = pe[i, 0:2]
            pos_b_encoding = pe[i + rd, 0:2]
            transformed = torch.matmul(rotation_matrix, pos_a_encoding)
            error = torch.norm(transformed - pos_b_encoding)
            total_error += error
            count += 1
        
        avg_error = total_error / count if count > 0 else 0
        errors.append(avg_error)
    
    ax4.bar(relative_distances, errors, alpha=0.7)
    ax4.set_title('Transformation Errors for Different Relative Distances')
    ax4.set_xlabel('Relative Distance')
    ax4.set_ylabel('Average Transformation Error')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('positional_encoding_linear_transformation.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Visualization chart saved as 'positional_encoding_linear_transformation.png'")

if __name__ == "__main__":
    demonstrate_linear_transformation_property()
    demonstrate_relative_position_learning()
    create_visualization()
    
    print(f"\n" + "=" * 80)
    print("总结：位置编码线性变换特性的意义")
    print("=" * 80)
    print("""
1. 数学基础：
   - 位置编码使用sin/cos函数，满足三角恒等式
   - sin(a+b) = sin(a)cos(b) + cos(a)sin(b)
   - cos(a+b) = cos(a)cos(b) - sin(a)sin(b)

2. 线性变换特性：
   - 任意两个位置的编码差可以表示为旋转矩阵
   - 相同相对距离的位置对具有相同的变换矩阵
   - 这使得模型可以学习"相对位置模式"

3. 模型学习的好处：
   - 模型可以学会识别特定的相对距离关系
   - 泛化能力：训练时学到的相对位置关系可用于推理时的新位置
   - 位置无关性：相同的相对关系在序列的任何位置都适用

4. 实际应用：
   - 语法依赖：动词和宾语的相对位置关系
   - 长距离依赖：指代消解中的先行词关系
   - 序列模式：重复出现的结构模式识别
""")