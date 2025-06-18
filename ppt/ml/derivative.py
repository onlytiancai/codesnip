import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# 设置matplotlib支持中文的字体和负号显示
matplotlib.rcParams['font.family'] = ['SimHei'] # 或者你系统支持的中文 C:\Windows\Fonts
matplotlib.rcParams['axes.unicode_minus'] = False 

# 1. 定义一元二次函数
def f(x):
    """
    一元二次函数: f(x) = x^2 - 2x + 1
    """
    return x**2 - 2*x + 1

# 2. 定义导函数
def df(x):
    """
    一元二次函数 f(x) 的导函数: f'(x) = 2x - 2
    """
    return 2*x - 2

# 3. 生成 x 值范围
x_values = np.linspace(-5, 5, 100)

# 4. 计算对应的 y 值和导函数的 y 值
y_values = f(x_values)
df_values = df(x_values)

# --- 计算并绘制从0到5每隔0.5的点和切线 ---
tangent_points = np.arange(0, 5.1, 0.5)  # 从0到5，每隔0.5
tangent_colors = plt.cm.tab10(np.linspace(0, 1, len(tangent_points)))  # 生成不同颜色

# 存储每个点的信息，用于后续绘图
tangent_data = []

for i, x_tangent in enumerate(tangent_points):
    y_tangent = f(x_tangent)  # 原函数在该点的y值
    slope = df(x_tangent)     # 导函数在该点的y值，即切线斜率
    
    # 切线方程：y - y_tangent = slope * (x - x_tangent)
    y_tangent_line = slope * (x_values - x_tangent) + y_tangent
    
    tangent_data.append({
        'x': x_tangent,
        'y': y_tangent,
        'slope': slope,
        'y_line': y_tangent_line,
        'color': tangent_colors[i]
    })
# -----------------------------------------------

# 5. 绘图
plt.figure(figsize=(10, 6)) 

# 绘制原函数图像
plt.plot(x_values, y_values, label='$f(x) = x^2 - 2x + 1$', color='blue', linestyle='-')

# 绘制导函数图像
plt.plot(x_values, df_values, label="$f'(x) = 2x - 2$", color='red', linestyle='--')

# 绘制从0到5每隔0.5的点和切线
for i, data in enumerate(tangent_data):
    # 为所有切线添加图例
    plt.plot(x_values, data['y_line'], linestyle=':', linewidth=1.5, color=data['color'],
            label=f'切线在 $x={data["x"]}$ 处 (斜率={data["slope"]:.1f})')
    
    # 标记切点
    plt.plot(data['x'], data['y'], 'o', markersize=6, color=data['color'])

plt.axhline(0, color='gray', linewidth=0.8) # x轴
plt.axvline(0, color='gray', linewidth=0.8) # y轴

# 添加标题和标签
plt.title('一元二次函数及其导函数和多条切线图像', fontsize=16)
plt.xlabel('x', fontsize=12)
plt.ylabel('y', fontsize=12)

# 添加图例，调整为多列显示以适应更多图例项
plt.legend(fontsize=9, ncol=2, loc='upper right')

# 添加网格线
plt.grid(True, linestyle=':', alpha=0.7)

# 设置 x 和 y 轴的显示范围，让图像更清晰
plt.xlim(-3, 5)
plt.ylim(-2, 5)


# 显示图像
plt.show()