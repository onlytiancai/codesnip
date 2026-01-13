import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


# -----------------------------
# 1. 构造 X 和 y
# -----------------------------
X = np.array([
    [1, 0],
    [0, 1],
    [1, 1]
], dtype=float)

y = np.array([2, 1, 4], dtype=float)

# -----------------------------
# 2. 最小二乘解
# -----------------------------
beta_hat = np.linalg.inv(X.T @ X) @ X.T @ y
y_proj = X @ beta_hat
r = y - y_proj

# 验证正交性
print("X^T r =", X.T @ r)

# -----------------------------
# 3. 构造列空间平面
# -----------------------------
u, v = X[:, 0], X[:, 1]

a = np.linspace(-3, 3, 10)
b = np.linspace(-3, 3, 10)
A, B = np.meshgrid(a, b)

plane_x = A * u[0] + B * v[0]
plane_y = A * u[1] + B * v[1]
plane_z = A * u[2] + B * v[2]

# -----------------------------
# 4. 3D 可视化
# -----------------------------
fig = plt.figure(figsize=(9, 7))
ax = fig.add_subplot(111, projection='3d')

# 列空间平面
ax.plot_surface(
    plane_x, plane_y, plane_z,
    alpha=0.25
)

# y 向量（蓝色）
ax.quiver(
    0, 0, 0,
    y[0], y[1], y[2],
    color='blue',
    linewidth=2,
    label='y'
)

# 投影向量 Xβ（绿色）
ax.quiver(
    0, 0, 0,
    y_proj[0], y_proj[1], y_proj[2],
    color='green',
    linewidth=2,
    label='Xβ'
)

# 残差向量 r（红色）
ax.quiver(
    y_proj[0], y_proj[1], y_proj[2],
    r[0], r[1], r[2],
    color='red',
    linewidth=2,
    label='r = y − Xβ'
)

ax.set_xlabel("x₁")
ax.set_ylabel("x₂")
ax.set_zlabel("x₃")
ax.set_title("最小二乘法的几何解释")

ax.legend()
plt.show()
