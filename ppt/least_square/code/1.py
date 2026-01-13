import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# -----------------------------
# 1. 构造真实感数据
# -----------------------------
np.random.seed(0)

n = 30  # 样本数

# 两个真实特征：面积 & 距离
area = np.random.uniform(50, 150, n)        # 平方米
distance = np.random.uniform(1, 15, n)      # 公里

# 真实参数（假设）
beta_true = np.array([2.0, 0.05, -0.8])  # 截距、面积系数、距离系数

# 构造设计矩阵
X = np.column_stack([
    np.ones(n),
    area,
    distance
])

# 房价（加入噪声）
noise = np.random.normal(0, 2, n)
y = X @ beta_true + noise

# -----------------------------
# 2. 最小二乘解
# -----------------------------
beta_hat = np.linalg.inv(X.T @ X) @ X.T @ y
y_hat = X @ beta_hat
r = y - y_hat

print("最小二乘估计 beta_hat：")
print(beta_hat)

# 验证正交性：X^T r ≈ 0
print("\nX^T r（应接近 0）：")
print(X.T @ r)

# -----------------------------
# 3. 几何可视化（只画两个特征）
# -----------------------------
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# 原始数据点
ax.scatter(area, distance, y, color='blue', label='观测 y')

# 拟合平面
area_grid, dist_grid = np.meshgrid(
    np.linspace(area.min(), area.max(), 20),
    np.linspace(distance.min(), distance.max(), 20)
)

y_plane = (
    beta_hat[0]
    + beta_hat[1] * area_grid
    + beta_hat[2] * dist_grid
)

ax.plot_surface(area_grid, dist_grid, y_plane,
                alpha=0.4, color='orange')

# 残差向量（从投影点到真实 y）
for i in range(n):
    ax.plot(
        [area[i], area[i]],
        [distance[i], distance[i]],
        [y_hat[i], y[i]],
        color='red'
    )

ax.set_xlabel("面积 (㎡)")
ax.set_ylabel("距离 (km)")
ax.set_zlabel("房价 (万元)")
ax.set_title("最小二乘几何解释：y 投影到 X 的列空间")

ax.legend()
plt.show()
