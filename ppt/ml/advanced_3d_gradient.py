import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches

# 定义不同的损失函数用于演示
def rosenbrock(x, y):
    """Rosenbrock函数 - 经典优化测试函数"""
    return (1 - x)**2 + 100 * (y - x**2)**2

def himmelblau(x, y):
    """Himmelblau函数 - 多峰函数"""
    return (x**2 + y - 11)**2 + (x + y**2 - 7)**2

def simple_quadratic(x, y):
    """简单二次函数"""
    return x**2 + y**2

# 计算梯度
def gradient_rosenbrock(x, y):
    dx = -2 * (1 - x) - 400 * x * (y - x**2)
    dy = 200 * (y - x**2)
    return dx, dy

def gradient_simple(x, y):
    return 2*x, 2*y

# 梯度下降算法
def gradient_descent(func, grad_func, start, lr=0.01, steps=100):
    path = [start]
    x, y = start
    
    for _ in range(steps):
        dx, dy = grad_func(x, y)
        x -= lr * dx
        y -= lr * dy
        path.append((x, y))
    
    return np.array(path)

# 创建网格
x = np.linspace(-2, 2, 100)
y = np.linspace(-1, 3, 100)
X, Y = np.meshgrid(x, y)

# 选择函数（可以切换）
Z = simple_quadratic(X, Y)
grad_func = gradient_simple
start_point = (1.5, 2.5)
lr = 0.1

# 执行梯度下降
path = gradient_descent(simple_quadratic, grad_func, start_point, lr, 50)
z_path = [simple_quadratic(p[0], p[1]) for p in path]

# 创建综合可视化
fig = plt.figure(figsize=(15, 10))

# 1. 3D表面图
ax1 = fig.add_subplot(221, projection='3d')
surf = ax1.plot_surface(X, Y, Z, alpha=0.6, cmap='viridis')
ax1.plot(path[:, 0], path[:, 1], z_path, 'ro-', markersize=3, linewidth=2)
ax1.set_title('3D表面 + 梯度下降路径')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('损失值')

# 2. 等高线图
ax2 = fig.add_subplot(222)
contour = ax2.contour(X, Y, Z, levels=20, cmap='viridis')
ax2.clabel(contour, inline=True, fontsize=8)
ax2.plot(path[:, 0], path[:, 1], 'ro-', markersize=4, linewidth=2)
ax2.set_title('等高线视图')
ax2.set_xlabel('X')
ax2.set_ylabel('Y')

# 3. 梯度向量场
ax3 = fig.add_subplot(223)
x_sparse = np.linspace(-2, 2, 20)
y_sparse = np.linspace(-1, 3, 20)
X_sparse, Y_sparse = np.meshgrid(x_sparse, y_sparse)
DX, DY = grad_func(X_sparse, Y_sparse)
ax3.quiver(X_sparse, Y_sparse, -DX, -DY, alpha=0.6)
ax3.contour(X, Y, Z, levels=10, alpha=0.3)
ax3.plot(path[:, 0], path[:, 1], 'ro-', markersize=4, linewidth=2)
ax3.set_title('梯度向量场')
ax3.set_xlabel('X')
ax3.set_ylabel('Y')

# 4. 损失下降曲线
ax4 = fig.add_subplot(224)
ax4.plot(z_path, 'b-', linewidth=2)
ax4.set_title('损失值随迭代变化')
ax4.set_xlabel('迭代次数')
ax4.set_ylabel('损失值')
ax4.grid(True)

plt.tight_layout()
plt.show()

# 交互式3D动画
fig2 = plt.figure(figsize=(12, 8))
ax = fig2.add_subplot(111, projection='3d')

def animate_3d(frame):
    ax.clear()
    
    # 绘制表面
    surf = ax.plot_surface(X, Y, Z, alpha=0.4, cmap='viridis')
    
    # 当前路径
    current_path = path[:frame+1]
    current_z = z_path[:frame+1]
    
    if len(current_path) > 1:
        ax.plot(current_path[:, 0], current_path[:, 1], current_z, 
                'ro-', markersize=4, linewidth=2, alpha=0.8)
    
    # 当前点
    if frame < len(path):
        ax.scatter(path[frame, 0], path[frame, 1], z_path[frame], 
                  color='red', s=100, alpha=1)
        
        # 添加梯度向量
        if frame > 0:
            dx, dy = grad_func(path[frame, 0], path[frame, 1])
            ax.quiver(path[frame, 0], path[frame, 1], z_path[frame],
                     -dx*0.1, -dy*0.1, 0, color='yellow', arrow_length_ratio=0.1)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('损失值')
    ax.set_title(f'3D梯度下降动画 - 步骤 {frame}/{len(path)-1}')
    
    # 设置视角
    ax.view_init(elev=30, azim=45)

# 创建动画
ani = FuncAnimation(fig2, animate_3d, frames=len(path), interval=300, repeat=True)
plt.show()

print(f"起始点: ({start_point[0]:.2f}, {start_point[1]:.2f})")
print(f"终点: ({path[-1, 0]:.2f}, {path[-1, 1]:.2f})")
print(f"起始损失: {z_path[0]:.4f}")
print(f"最终损失: {z_path[-1]:.4f}")