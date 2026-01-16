import numpy as np
import matplotlib.pyplot as plt
from data import area, distance, price

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 构造设计矩阵 X
n = len(area)
X = np.column_stack([
    np.ones(n),
    area,
    distance
])

# 计算最小二乘解
beta_hat = np.linalg.inv(X.T @ X) @ X.T @ price
print("最小二乘估计 beta_hat：")
print(f"截距: {beta_hat[0]:.2f}")
print(f"面积系数: {beta_hat[1]:.2f}")
print(f"距离系数: {beta_hat[2]:.2f}")

# 定义误差平方和函数 SSE
def sse(beta0, beta1):
    """计算给定 beta0（截距）和 beta1（面积系数）时的误差平方和
       假设距离系数为固定值 beta_hat[2]"""
    beta = np.array([beta0, beta1, beta_hat[2]])
    y_hat = X @ beta
    return np.sum((price - y_hat) ** 2)

# 创建参数网格
beta0_range = np.linspace(beta_hat[0] - 100, beta_hat[0] + 100, 50)
beta1_range = np.linspace(beta_hat[1] - 2, beta_hat[1] + 2, 50)
beta0_grid, beta1_grid = np.meshgrid(beta0_range, beta1_range)

# 计算每个参数组合的 SSE
sse_grid = np.zeros_like(beta0_grid)
for i in range(beta0_grid.shape[0]):
    for j in range(beta0_grid.shape[1]):
        sse_grid[i, j] = sse(beta0_grid[i, j], beta1_grid[i, j])

# 3D 可视化碗状图形
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# 绘制 SSE 曲面（碗状）
surface = ax.plot_surface(
    beta0_grid, beta1_grid, sse_grid,
    cmap='viridis',
    alpha=0.8,
    rstride=1,
    cstride=1,
    linewidth=0.5,
    edgecolor='lightgray'
)

# 标记最小点
ax.scatter(
    beta_hat[0], beta_hat[1], sse(beta_hat[0], beta_hat[1]),
    color='red',
    s=200,
    marker='*',
    label=f'最小点 (β₀, β₁) = ({beta_hat[0]:.2f}, {beta_hat[1]:.2f})'
)

# 添加颜色条
fig.colorbar(surface, ax=ax, shrink=0.5, aspect=10, label='误差平方和 (SSE)')

# 设置标签和标题
ax.set_xlabel('截距 β₀', fontsize=12, labelpad=10)
ax.set_ylabel('面积系数 β₁', fontsize=12, labelpad=10)
ax.set_zlabel('误差平方和 (SSE)', fontsize=12, labelpad=10)
ax.set_title('最小二乘法：误差平方和函数的碗状曲面', fontsize=14, pad=20)

# 设置视角
ax.view_init(elev=30, azim=135)

# 添加图例
ax.legend(loc='upper right', fontsize=10)

plt.tight_layout()
plt.show()