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

# --- 计算并绘制在 x=0 处的切线 ---
x0_tangent = 0 
y0_tangent = f(x0_tangent) 
slope0 = df(x0_tangent) 
y_tangent0 = slope0 * (x_values - x0_tangent) + y0_tangent
# -----------------------------------------------

# --- 额外部分：计算并绘制在 x=1.5 处的切线 ---
x1_tangent = 1.5 # 选择另一个点
y1_tangent = f(x1_tangent) # 原函数在该点的y值
slope1 = df(x1_tangent) # 导函数在该点的y值，即切线斜率

# 切线方程：y - y1 = slope1 * (x - x1)
y_tangent1 = slope1 * (x_values - x1_tangent) + y1_tangent
# -----------------------------------------------

# 5. 绘图
plt.figure(figsize=(10, 6)) 

# 绘制原函数图像
plt.plot(x_values, y_values, label='$f(x) = x^2 - 2x + 1$', color='blue', linestyle='-')

# 绘制导函数图像
plt.plot(x_values, df_values, label="$f'(x) = 2x - 2$", color='red', linestyle='--')

# 绘制 x=0 处的切线
plt.plot(x_values, y_tangent0, label=f'切线在 $x={x0_tangent}$ 处 (斜率={slope0})', color='green', linestyle=':', linewidth=2)
plt.plot(x0_tangent, y0_tangent, 'go', markersize=8) # 标记切点

# 绘制 x=1.5 处的切线
plt.plot(x_values, y_tangent1, label=f'切线在 $x={x1_tangent}$ 处 (斜率={slope1})', color='purple', linestyle=':', linewidth=2)
plt.plot(x1_tangent, y1_tangent, 'o', markersize=8, color='purple') # 标记切点，使用紫色标记点

plt.axhline(0, color='gray', linewidth=0.8) # x轴
plt.axvline(0, color='gray', linewidth=0.8) # y轴

# 添加标题和标签
plt.title('一元二次函数及其导函数和多条切线图像', fontsize=16)
plt.xlabel('x', fontsize=12)
plt.ylabel('y', fontsize=12)

# 添加图例
plt.legend(fontsize=10)

# 添加网格线
plt.grid(True, linestyle=':', alpha=0.7)

# 设置 x 和 y 轴的显示范围，让图像更清晰
plt.xlim(-3, 5)
plt.ylim(-2, 5)


# 显示图像
plt.show()