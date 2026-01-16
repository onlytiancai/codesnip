import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


from data import area, distance, price

n = len(area)

# -----------------------------
# 2. 构造设计矩阵
# -----------------------------
X = np.column_stack([
    np.ones(n),
    area,
    distance
])

# -----------------------------
# 3. 最小二乘解
# -----------------------------
beta_hat = np.linalg.inv(X.T @ X) @ X.T @ price
price_hat = X @ beta_hat
r = price - price_hat

print("最小二乘估计 beta_hat：")
print(beta_hat)
print("\n截距:", beta_hat[0])
print("面积系数:", beta_hat[1])
print("距离系数:", beta_hat[2])

# -----------------------------
# 4. 3D 可视化
# -----------------------------
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# 原始数据点
ax.scatter(area, distance, price, color='blue', s=100, alpha=0.8, label='观测房价')

# 原点标记
ax.scatter(0, 0, 0, color='red', s=100, alpha=1.0, label='原点')

# 为每个数据点添加坐标标签
for i in range(n):
    ax.text3D(
        area[i], distance[i], price[i] + 20,  # 文本位置上移20个单位
        f'({area[i]}m², {distance[i]}km, {price[i]}w)',
        fontsize=8,
        color='black',
        ha='center', va='bottom'
    )

# 拟合平面
area_grid, dist_grid = np.meshgrid(
    np.linspace(area.min() - 10, area.max() + 10, 20),
    np.linspace(distance.min() - 1, distance.max() + 1, 20)
)

price_plane = (
    beta_hat[0]
    + beta_hat[1] * area_grid
    + beta_hat[2] * dist_grid
)

ax.plot_surface(area_grid, dist_grid, price_plane,
               color='orange', alpha=0.4, rstride=1, cstride=1,
               linewidth=0.5, edgecolor='lightgray')

# 残差向量
for i in range(n):
    ax.plot(
        [area[i], area[i]],
        [distance[i], distance[i]],
        [price[i], price_hat[i]],
        color='red', linestyle='--', linewidth=1.5
    )

# 设置标签和标题
ax.set_xlabel("面积 (㎡)", fontsize=12, labelpad=10)
ax.set_ylabel("到市中心距离 (公里)", fontsize=12, labelpad=10)
ax.set_zlabel("房价 (万元)", fontsize=12, labelpad=10)
ax.set_title("最小二乘法拟合：房价与面积、到市中心距离的关系", fontsize=14, pad=20)

# 设置坐标轴范围
ax.set_xlim(area.min() - 10, area.max() + 10)
ax.set_ylim(distance.min() - 1, distance.max() + 1)
ax.set_zlim(price.min() - 50, price.max() + 50)

# 添加图例
ax.legend(loc='upper right', fontsize=10)

# 调整视角
ax.view_init(elev=20, azim=130)

plt.tight_layout()
plt.show()