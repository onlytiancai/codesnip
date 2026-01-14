from data import area, distance, price
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 创建设计矩阵X，包含截距项、面积和距离
n = len(price)  # n=12个样本，n维空间
p = 3           # p=3个特征（截距、面积、距离）
X = np.column_stack((np.ones(n), area, distance))
y = price.reshape(-1, 1)

# 使用最小二乘法计算系数β
beta = np.linalg.lstsq(X, y, rcond=None)[0]

# 计算投影向量Xβ
y_pred = X @ beta

# 计算残差向量r
r = y - y_pred

# 使用PCA将n维空间降维到3维以便可视化
# 我们需要降维的是由X的列向量、y、投影向量和残差向量张成的空间
# 创建一个包含所有向量的矩阵
combined = np.column_stack((X, y, y_pred, r))

# 进行PCA降维到3维
pca = PCA(n_components=3)
# combined.T 是 (p+3)个n维向量的集合，我们需要对这些向量进行降维
combined_3d = pca.fit_transform(combined.T).T

# 从降维后的数据中提取各向量
# X的列向量在降维后的表示
X_3d = combined_3d[:, :p]
# y向量在降维后的表示
y_3d = combined_3d[:, p]
# 投影向量Xβ在降维后的表示
proj_3d = combined_3d[:, p+1]
# 残差向量r在降维后的表示
res_3d = combined_3d[:, p+2]

# 创建三维图形
fig = plt.figure(figsize=(15, 12))
ax = fig.add_subplot(111, projection='3d')

# 绘制原点
ax.scatter(0, 0, 0, c='k', marker='o', s=50, label='原点')

# 绘制X的列向量（张成子空间的基）
for i in range(p):
    # 绘制向量
    ax.quiver(0, 0, 0, X_3d[0, i], X_3d[1, i], X_3d[2, i], 
              arrow_length_ratio=0.1, color=['r', 'g', 'b'][i], linewidth=2, 
              label=f'X的第{i+1}列向量')
    
    # 为向量添加标注
    vector_name = ['截距项列', '面积列', '距离列'][i]
    ax.text(X_3d[0, i] * 1.1, X_3d[1, i] * 1.1, X_3d[2, i] * 1.1, 
            f'X_{i+1}: {vector_name}', fontsize=11, 
            color=['r', 'g', 'b'][i], fontweight='bold')

# 绘制y向量
ax.quiver(0, 0, 0, y_3d[0], y_3d[1], y_3d[2], 
          arrow_length_ratio=0.1, color='purple', linewidth=3, label='y向量')
# 为y向量添加标注
ax.text(y_3d[0] * 1.1, y_3d[1] * 1.1, y_3d[2] * 1.1, 
        'y: 真实数据向量', fontsize=12, color='purple', fontweight='bold')

# 绘制投影向量Xβ
ax.quiver(0, 0, 0, proj_3d[0], proj_3d[1], proj_3d[2], 
          arrow_length_ratio=0.1, color='orange', linewidth=3, label='投影向量Xβ')
# 为投影向量添加标注
ax.text(proj_3d[0] * 1.1, proj_3d[1] * 1.1, proj_3d[2] * 1.1, 
        'Xβ: y在span{X}上的投影', fontsize=12, color='orange', fontweight='bold')

# 绘制残差向量r = y - Xβ
ax.quiver(proj_3d[0], proj_3d[1], proj_3d[2], 
          res_3d[0], res_3d[1], res_3d[2], 
          arrow_length_ratio=0.1, color='cyan', linewidth=3, label='残差向量r = y - Xβ')
# 为残差向量添加标注
ax.text(proj_3d[0] + res_3d[0] * 1.1, proj_3d[1] + res_3d[1] * 1.1, proj_3d[2] + res_3d[2] * 1.1, 
        'r = y - Xβ: 残差向量', fontsize=12, color='cyan', fontweight='bold')

# 绘制span{X}子空间平面
# 平面由X的列向量张成，我们需要计算平面的方程
# 平面过原点，因为X包含截距项

# 计算平面的法向量
# 取X_3d的两个列向量的叉积
v1 = X_3d[:, 1]
v2 = X_3d[:, 2]
normal = np.cross(v1, v2)
normal = normal / np.linalg.norm(normal)  # 归一化

# 生成平面上的点
# 使用网格点
xx, yy = np.meshgrid(np.linspace(-2, 2, 10), np.linspace(-2, 2, 10))

# 计算平面上的z坐标
# 平面方程：normal[0]*x + normal[1]*y + normal[2]*z = 0
if normal[2] != 0:
    zz = (-normal[0] * xx - normal[1] * yy) / normal[2]
    # 限制平面范围，使其与向量长度相匹配
    max_val = np.max(np.abs(np.concatenate([X_3d.flatten(), y_3d, proj_3d, res_3d]))) * 1.5
    ax.plot_surface(xx, yy, zz, alpha=0.2, color='yellow', 
                    linewidth=0, antialiased=False, 
                    label='span{X}子空间平面')
    # 为平面添加标注
    ax.text(0, 0, zz[5, 5] * 0.9, 'span{X}: 由X的列向量张成的子空间', 
            fontsize=12, color='gold', fontweight='bold', 
            bbox=dict(facecolor='white', alpha=0.8, boxstyle='round'))
else:
    # 处理z=0的特殊情况
    ax.plot_wireframe(xx, yy, np.zeros_like(xx), alpha=0.2, color='yellow')
    # 为平面添加标注
    ax.text(0, 0, -0.1, 'span{X}: 由X的列向量张成的子空间', 
            fontsize=12, color='gold', fontweight='bold', 
            bbox=dict(facecolor='white', alpha=0.8, boxstyle='round'))

# 添加标签和标题
ax.set_xlabel('PCA维度1', fontsize=12)
ax.set_ylabel('PCA维度2', fontsize=12)
ax.set_zlabel('PCA维度3', fontsize=12)
ax.set_title('最小二乘法的几何解释 (n维空间降维可视化)', fontsize=16)

# 设置坐标轴范围
all_vectors = np.concatenate([X_3d.flatten(), y_3d, proj_3d, res_3d])
max_range = np.max(np.abs(all_vectors)) * 1.5
ax.set_xlim(-max_range, max_range)
ax.set_ylim(-max_range, max_range)
ax.set_zlim(-max_range, max_range)

# 添加图例
ax.legend(loc='best', fontsize=10)

# 调整视角
ax.view_init(elev=25, azim=45)

# 显示网格
ax.grid(True, alpha=0.3)

# 显示图形
plt.tight_layout()
plt.show()

# 打印系数和验证正交性
print("回归系数 β:")
print(f"截距: {beta[0][0]:.4f}")
print(f"面积系数: {beta[1][0]:.4f}")
print(f"距离系数: {beta[2][0]:.4f}")

# 验证残差与X的列正交
print(f"\n验证残差与X的列正交:")
for i in range(p):
    orthogonality = np.dot(r.flatten(), X[:, i])
    print(f"残差与X第{i+1}列的点积: {orthogonality:.10f} (应接近0)")

# 验证残差与投影向量正交
orth_proj_res = np.dot((X @ beta).flatten(), r.flatten())
print(f"\n投影向量与残差向量的点积: {orth_proj_res:.10f} (应接近0)")

# 打印残差信息
print(f"\n残差平方和: {np.sum(r**2):.4f}")
print(f"平均残差: {np.mean(r):.4f}")