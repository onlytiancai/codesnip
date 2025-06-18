import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import matplotlib

# 设置matplotlib支持中文的字体和负号显示
matplotlib.rcParams['font.family'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False 

# 定义目标函数 y = x² + sinx
def target_function(x):
    return x**2 + np.sin(x)

# 计算函数在某点的导数（梯度）
def calculate_gradient(x, h=1e-6):
    # 用数值差分法近似导数：f'(x) ≈ [f(x+h) - f(x)] / h
    return (target_function(x + h) - target_function(x)) / h

def gradient_descent(initial_x, learning_rate=0.1, min_step=1e-6, max_iterations=1000):
    x = initial_x
    x_history = [x]
    y_history = [target_function(x)]
    step_history = [learning_rate]
    gradient_history = [calculate_gradient(x)]
    
    iteration = 0
    while iteration < max_iterations:
        # 计算当前点的梯度
        gradient = calculate_gradient(x)
        
        # 动态调整学习率（步长）：更频繁地调整学习率以展示变化
        if iteration > 0:
            # 检查梯度方向是否变化
            if gradient * gradient_history[-1] < 0:
                learning_rate *= 0.5
            # 每5次迭代额外减小一次学习率，使变化更明显
            elif iteration % 5 == 0:
                learning_rate *= 0.9
        
        # 梯度下降更新位置
        new_x = x - learning_rate * gradient
        
        # 记录历史数据
        x_history.append(new_x)
        y_history.append(target_function(new_x))
        step_history.append(learning_rate)
        gradient_history.append(gradient)
        
        # 检查是否收敛（步长足够小）
        if abs(new_x - x) < min_step:
            break
            
        x = new_x
        iteration += 1
    
    return x, target_function(x), x_history, y_history, step_history

# 可视化函数曲线和梯度下降过程（静态图）
def visualize_gradient_descent(x_history, y_history, step_history):
    # 计算每一步的梯度
    gradient_history = [calculate_gradient(x) for x in x_history]
    
    # 生成函数曲线数据
    x_range = np.linspace(-5, 5, 1000)
    y_range = target_function(x_range)
    
    # 创建图形 - 左侧上下两个图表，右侧表格
    fig = plt.figure(figsize=(16, 10))
    
    # 创建网格布局：2行2列，右侧列合并为一个区域
    gs = fig.add_gridspec(2, 2, width_ratios=[2, 1.5], height_ratios=[3, 2])
    
    # 左上角：函数曲线
    ax1 = fig.add_subplot(gs[0, 0])
    # 左下角：学习率图表
    ax2 = fig.add_subplot(gs[1, 0])
    # 右侧整列：表格
    ax3 = fig.add_subplot(gs[:, 1])
    
    # 绘制函数曲线和下降轨迹
    ax1.plot(x_range, y_range, 'b-', label='y = x^2 + sin(x)')
    ax1.plot(x_history, y_history, 'r.-', label='梯度下降路径')
    ax1.plot(x_history[-1], y_history[-1], 'go', label=f'最终点: ({x_history[-1]:.4f}, {y_history[-1]:.4f})')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('函数曲线和梯度下降路径')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    # 绘制步长变化 - 改进显示方式
    iterations = range(len(step_history))
    
    # 使用条形图显示学习率变化
    ax2.bar(iterations, step_history, color='g', alpha=0.7, width=0.8)
    
    # 添加数据点和连线，使变化更明显
    ax2.plot(iterations, step_history, 'ro-', markersize=4)
    
    # 在每个变化点添加数值标签
    for i, rate in enumerate(step_history):
        if i == 0 or rate != step_history[i-1]:  # 只在初始点和变化点添加标签
            ax2.annotate(f'{rate:.4f}', 
                        xy=(i, rate),
                        xytext=(0, 5),  # 文本偏移
                        textcoords='offset points',
                        ha='center')
    
    # 设置y轴范围，确保变化可见
    min_rate = min(step_history)
    max_rate = max(step_history)
    if min_rate == max_rate:  # 如果学习率没有变化
        ax2.set_ylim(min_rate * 0.5, min_rate * 1.5)  # 创建一些空间
    else:
        margin = (max_rate - min_rate) * 0.2
        ax2.set_ylim(min_rate - margin, max_rate + margin)
    
    ax2.set_xlabel('迭代次数')
    ax2.set_ylabel('学习率')
    ax2.set_title('学习率变化')
    ax2.grid(True, linestyle='--', alpha=0.6)
    
    # 创建表格显示迭代数据
    ax3.axis('tight')
    ax3.axis('off')
    
    # 准备表格数据
    table_data = []
    for i in range(len(x_history)):
        table_data.append([
            i,  # 迭代次数
            f"{x_history[i]:.6f}",  # x值
            f"{y_history[i]:.6f}",  # y值
            f"{gradient_history[i]:.6f}",  # 梯度
            f"{step_history[i]:.6f}"  # 学习率
        ])
    
    # 创建表格
    table = ax3.table(
        cellText=table_data,
        colLabels=['迭代', 'x', 'y', '梯度', '学习率'],
        loc='center',
        cellLoc='center',
        colWidths=[0.1, 0.225, 0.225, 0.225, 0.225]
    )
    
    # 设置表格样式
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)  # 调整表格大小
    
    # 设置表格标题
    ax3.set_title('梯度下降迭代过程数据表', pad=20)
    
    plt.tight_layout()
    plt.show()

# 使用动画可视化梯度下降过程
def animate_gradient_descent(x_history, y_history, step_history):
    # 计算显示范围 - 只关注迭代区域
    x_min, x_max = min(x_history), max(x_history)
    y_min, y_max = min(y_history), max(y_history)
    
    # 添加边距，确保点不会太靠近边缘
    x_margin = (x_max - x_min) * 0.5
    y_margin = (y_max - y_min) * 0.5
    
    # 确保至少有一定的显示范围
    if x_margin < 0.5:
        x_margin = 0.5
    if y_margin < 0.5:
        y_margin = 0.5
    
    # 生成函数曲线数据 - 只在关注区域生成更密集的点
    x_range = np.linspace(x_min - x_margin, x_max + x_margin, 1000)
    y_range = target_function(x_range)
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 绘制函数曲线（固定不变的部分）
    ax.plot(x_range, y_range, 'b-', linewidth=2, label='y = x² + sin(x)')
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.set_title('梯度下降动画演示', fontsize=14)
    
    # 初始点 - 增大点的大小和轨迹的可见性
    point, = ax.plot([], [], 'ro', markersize=12)
    path, = ax.plot([], [], 'r-', linewidth=2, alpha=0.7)
    
    # 添加当前点的投影线
    v_line, = ax.plot([], [], 'r--', alpha=0.5)
    h_line, = ax.plot([], [], 'r--', alpha=0.5)
    
    # 添加文本信息 - 放大字体
    iteration_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=12, 
                            bbox=dict(facecolor='white', alpha=0.7))
    
    # 设置坐标轴范围 - 聚焦在迭代区域
    ax.set_xlim(x_min - x_margin, x_max + x_margin)
    ax.set_ylim(y_min - y_margin, y_max + y_margin)
    
    # 添加网格线
    ax.grid(True, linestyle='--', alpha=0.6)
    
    # 添加图例
    ax.legend(fontsize=12)
    
    def init():
        point.set_data([], [])
        path.set_data([], [])
        v_line.set_data([], [])
        h_line.set_data([], [])
        iteration_text.set_text('')
        return point, path, v_line, h_line, iteration_text
    
    def update(frame):
        # 更新点的位置
        x_data = x_history[:frame+1]
        y_data = y_history[:frame+1]
        
        # 确保数据是序列，即使只有一个点
        point.set_data([x_data[-1]], [y_data[-1]])
        path.set_data(x_data, y_data)
        
        # 添加投影线
        current_x = x_data[-1]
        current_y = y_data[-1]
        v_line.set_data([current_x, current_x], [y_min - y_margin, current_y])
        h_line.set_data([x_min - x_margin, current_x], [current_y, current_y])
        
        # 更新迭代信息
        gradient = calculate_gradient(current_x)
        iteration_text.set_text(f'迭代: {frame}/{len(x_history)-1}\n'
                               f'x: {current_x:.4f}, y: {current_y:.4f}\n'
                               f'梯度: {gradient:.4f}')
        
        return point, path, v_line, h_line, iteration_text
    
    # 创建动画 - 放慢速度
    frames = len(x_history)
    ani = FuncAnimation(fig, update, frames=frames, init_func=init, 
                        interval=400, blit=True)
    
    plt.tight_layout()
    plt.show()
    
    return ani

# 运行梯度下降算法
if __name__ == "__main__":
    # 使用随机初始点
    initial_x = np.random.uniform(-5, 5)
    print(f"随机初始点: x = {initial_x:.4f}, y = {target_function(initial_x):.4f}")
    
    # 执行梯度下降，使用较大的学习率和适当的收敛阈值，确保有足够的迭代次数
    optimal_x, optimal_y, x_history, y_history, step_history = gradient_descent(
        initial_x, learning_rate=0.5, min_step=1e-5, max_iterations=30
    )
    
    print(f"找到极值点: x = {optimal_x:.6f}, y = {optimal_y:.6f}")
    print(f"迭代次数: {len(x_history) - 1}")
    print(f"学习率变化: 初始值={step_history[0]:.4f}, 最终值={step_history[-1]:.4f}")
    
    # 可视化静态图，显示学习率变化
    visualize_gradient_descent(x_history, y_history, step_history)
    
    # 如果需要动画，取消下面的注释
    # animate_gradient_descent(x_history, y_history, step_history)