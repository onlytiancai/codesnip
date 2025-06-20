import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# 生成简单的二次函数数据
np.random.seed(42)
X = np.linspace(-2, 2, 50)
Y = 3 * X**2 + 2 * X + 1 + np.random.normal(0, 0.5, len(X))

# 损失函数：对于 y = ax^2 + bx + c，只优化 a 和 b（固定 c=1）
def loss_function(a, b):
    y_pred = a * X**2 + b * X + 1
    return np.mean((y_pred - Y)**2)

# 计算梯度
def compute_gradients(a, b):
    y_pred = a * X**2 + b * X + 1
    error = y_pred - Y
    da = 2 * np.mean(error * X**2)
    db = 2 * np.mean(error * X)
    return da, db

# 创建参数网格用于绘制损失面
a_range = np.linspace(1, 5, 50)
b_range = np.linspace(0, 4, 50)
A, B = np.meshgrid(a_range, b_range)
Z = np.array([[loss_function(a, b) for a in a_range] for b in b_range])

# 梯度下降
a, b = 1.0, 0.0  # 初始参数
lr = 0.01
path = [(a, b, loss_function(a, b))]

for _ in range(100):
    da, db = compute_gradients(a, b)
    a -= lr * da
    b -= lr * db
    path.append((a, b, loss_function(a, b)))

path = np.array(path)

# 3D可视化
fig = plt.figure(figsize=(12, 5))

# 静态3D图
ax1 = fig.add_subplot(121, projection='3d')
ax1.plot_surface(A, B, Z, alpha=0.6, cmap='viridis')
ax1.plot(path[:, 0], path[:, 1], path[:, 2], 'ro-', markersize=3, linewidth=2)
ax1.set_xlabel('参数 a')
ax1.set_ylabel('参数 b')
ax1.set_zlabel('损失值')
ax1.set_title('3D梯度下降路径')

# 等高线图
ax2 = fig.add_subplot(122)
contour = ax2.contour(A, B, Z, levels=20)
ax2.clabel(contour, inline=True, fontsize=8)
ax2.plot(path[:, 0], path[:, 1], 'ro-', markersize=4, linewidth=2)
ax2.set_xlabel('参数 a')
ax2.set_ylabel('参数 b')
ax2.set_title('等高线视图')

plt.tight_layout()
plt.show()

# 动画版本
fig2 = plt.figure(figsize=(10, 8))
ax = fig2.add_subplot(111, projection='3d')

def animate(frame):
    ax.clear()
    ax.plot_surface(A, B, Z, alpha=0.3, cmap='viridis')
    
    # 显示到当前帧的路径
    current_path = path[:frame+1]
    ax.plot(current_path[:, 0], current_path[:, 1], current_path[:, 2], 
            'ro-', markersize=4, linewidth=2)
    
    # 当前点
    if frame < len(path):
        ax.scatter(path[frame, 0], path[frame, 1], path[frame, 2], 
                  color='red', s=100, alpha=1)
    
    ax.set_xlabel('参数 a')
    ax.set_ylabel('参数 b')
    ax.set_zlabel('损失值')
    ax.set_title(f'3D梯度下降动画 - 步骤 {frame}')

ani = FuncAnimation(fig2, animate, frames=len(path), interval=200, repeat=True)
plt.show()