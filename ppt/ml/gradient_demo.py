import sys
import matplotlib
import numpy as np
import matplotlib.pyplot as plt

if sys.platform.startswith('win'):
    matplotlib.rcParams['font.family'] = ['SimHei'] # Windows的中文字体
elif sys.platform.startswith('darwin'):
    matplotlib.rcParams['font.family'] = ['Arial Unicode MS'] # Mac的中文字体
matplotlib.rcParams['axes.unicode_minus'] = False 

# 定义多元函数 f(x,y) = x^2 + y^2 - 2*x*y
def f(x, y):
    return x**2 + y**2 - 2*x*y

# 计算梯度 ∇f = (∂f/∂x, ∂f/∂y)
def gradient(x, y):
    df_dx = 2*x - 2*y  # ∂f/∂x
    df_dy = 2*y - 2*x  # ∂f/∂y
    return df_dx, df_dy

# 创建网格
x = np.linspace(-3, 3, 100)
y = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(x, y)
Z = f(X, Y)

# 创建图形
fig = plt.figure(figsize=(15, 5))

# 子图1: 3D曲面图
ax1 = fig.add_subplot(131, projection='3d')
ax1.plot_surface(X, Y, Z, alpha=0.7, cmap='viridis')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('f(x,y)')
ax1.set_title('函数曲面')

# 子图2: 等高线图
ax2 = fig.add_subplot(132)
contour = ax2.contour(X, Y, Z, levels=20)
ax2.clabel(contour, inline=True, fontsize=8)
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_title('等高线图')

# 子图3: 梯度向量场
ax3 = fig.add_subplot(133)
# 创建稀疏网格用于显示梯度向量
x_sparse = np.linspace(-3, 3, 15)
y_sparse = np.linspace(-3, 3, 15)
X_sparse, Y_sparse = np.meshgrid(x_sparse, y_sparse)

# 计算梯度
U, V = gradient(X_sparse, Y_sparse)

# 绘制等高线
ax3.contour(X, Y, Z, levels=20, alpha=0.5)
# 绘制梯度向量场
ax3.quiver(X_sparse, Y_sparse, U, V, alpha=0.8, color='red')
ax3.set_xlabel('x')
ax3.set_ylabel('y')
ax3.set_title('梯度向量场')

plt.tight_layout()
plt.show()

# 在特定点计算梯度
point_x, point_y = 1, 2
grad_x, grad_y = gradient(point_x, point_y)
print(f"在点({point_x}, {point_y})处:")
print(f"函数值: f({point_x}, {point_y}) = {f(point_x, point_y)}")
print(f"梯度: ∇f = ({grad_x}, {grad_y})")
print(f"梯度模长: |∇f| = {np.sqrt(grad_x**2 + grad_y**2):.3f}")