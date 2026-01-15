from data import area, distance, price
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 1. 面积与房价的散点图
plt.figure(figsize=(10, 6))
plt.scatter(area, price, color='blue', s=100, alpha=0.8)
plt.title('房屋面积与房价的关系')
plt.xlabel('面积（平方米）')
plt.ylabel('房价（万元）')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('面积与房价散点图.png', dpi=300)

# 2. 距离与房价的散点图
plt.figure(figsize=(10, 6))
plt.scatter(distance, price, color='red', s=100, alpha=0.8)
plt.title('距离市中心距离与房价的关系')
plt.xlabel('距离（公里）')
plt.ylabel('房价（万元）')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('距离与房价散点图.png', dpi=300)

# 3. 3D散点图：同时展示面积、距离和房价
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(area, distance, price, c=price, cmap='viridis', s=200, alpha=0.8)
ax.set_title('面积、距离与房价的3D关系')
ax.set_xlabel('面积（平方米）')
ax.set_ylabel('距离（公里）')
ax.set_zlabel('房价（万元）')
fig.colorbar(scatter, ax=ax, label='房价（万元）')
plt.tight_layout()
plt.savefig('3D散点图.png', dpi=300)

# 4. 气泡图：用气泡大小表示房价，颜色表示距离
plt.figure(figsize=(12, 8))
scatter = plt.scatter(area, distance, s=price*2, c=distance, cmap='coolwarm', alpha=0.7, edgecolors='black')
plt.title('面积、距离与房价的气泡图')
plt.xlabel('面积（平方米）')
plt.ylabel('距离（公里）')
plt.colorbar(scatter, label='距离（公里）')
# 添加图例
for i, (a, d, p) in enumerate(zip(area, distance, price)):
    plt.text(a, d, f'{p}万', ha='center', va='center', fontsize=9, color='white', fontweight='bold')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('气泡图.png', dpi=300)

# 5. 相关性热力图
# 创建数据框
import pandas as pd
df = pd.DataFrame({'面积': area, '距离': distance, '房价': price})
# 计算相关系数
corr_matrix = df.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, square=True, linewidths=0.5, cbar_kws={'label': '相关系数'})
plt.title('面积、距离与房价的相关性热力图')
plt.tight_layout()
plt.savefig('相关性热力图.png', dpi=300)

# 6. 分组散点图：按距离分组展示面积与房价的关系
plt.figure(figsize=(12, 8))
distance_groups = np.unique(distance)
colors = ['blue', 'green', 'red', 'purple']

for i, d in enumerate(distance_groups):
    mask = (distance == d)
    plt.scatter(area[mask], price[mask], color=colors[i], s=150, alpha=0.8, label=f'距离：{d}公里')

plt.title('按距离分组的面积与房价关系')
plt.xlabel('面积（平方米）')
plt.ylabel('房价（万元）')
plt.legend(title='距离分组')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('分组散点图.png', dpi=300)

# 显示所有图表
plt.show()