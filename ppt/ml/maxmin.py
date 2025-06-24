import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# 设置matplotlib以支持中文显示和负号
matplotlib.rcParams['font.family'] = ['SimHei'] # 尝试使用SimHei（黑体），如果你的系统没有，请替换为其他中文字体
matplotlib.rcParams['axes.unicode_minus'] = False # 解决负号显示为方块的问题

# 1. 定义函数 f(x)
def f(x):
    """
    一个具有多个极值的四次多项式函数:
    f(x) = x^4 - 4x^3 - 2x^2 + 12x + 5
    """
    return x**4 - 4*x**3 - 2*x**2 + 12*x + 5

# 2. 定义函数的一阶导数 df_dx(x)
def df_dx(x):
    """
    f(x) 的一阶导数: f'(x) = 4x^3 - 12x^2 - 4x + 12
    """
    return 4*x**3 - 12*x**2 - 4*x + 12

# 3. 定义函数的二阶导数 d2f_dx2(x)
def d2f_dx2(x):
    """
    f(x) 的二阶导数: f''(x) = 12x^2 - 24x - 4
    """
    return 12*x**2 - 24*x - 4

# --- 查找临界点 ---
# 临界点是 f'(x) = 0 的解。
# 对于多项式，可以使用 numpy.roots 来找到根。
# 导函数的系数为 [4, -12, -4, 12]
derivative_coeffs = [4, -12, -4, 12]
critical_points_raw = np.roots(derivative_coeffs)

# 过滤出实数根，并进行四舍五入以处理浮点数精度问题
critical_points = np.array([p.real for p in critical_points_raw if np.isclose(p.imag, 0)])
critical_points.sort() # 排序，方便后续处理和显示

# 存储极值点信息
extrema_info = []

# --- 分类临界点并计算极值 ---
print("--- 极值点分析 ---")
for cp in critical_points:
    f_val = f(cp)
    d2f_val = d2f_dx2(cp)
    
    type_of_extremum = ""
    marker_color = 'k' # Default to black
    
    if d2f_val > 0:
        type_of_extremum = "局部最小值"
        marker_color = 'g' # 绿色表示局部最小值
    elif d2f_val < 0:
        type_of_extremum = "局部最大值"
        marker_color = 'r' # 红色表示局部最大值
    else: # d2f_val == 0, 二阶导数测试无法判断
        type_of_extremum = "可能是拐点或更高阶极值点 (二阶导数测试无效)"
        marker_color = 'b' # 蓝色表示无法判断
    
    extrema_info.append({
        'x': cp,
        'f(x)': f_val,
        'type': type_of_extremum,
        'marker_color': marker_color
    })
    print(f"在 x = {cp:.4f} 处: f(x) = {f_val:.4f}, 类型: {type_of_extremum}")

# --- 确定全局最大值和最小值 ---
# 对于这个四次函数 (x^4 项系数为正), f(x) -> 无穷大 当 x -> +/- 无穷大
# 所以没有全局最大值。全局最小值是所有局部最小值中的最小者。
global_min_value = float('inf')
global_max_value = float('-inf') # 初始化为负无穷大，但这个函数没有全局最大值

for info in extrema_info:
    if info['type'] == "局部最小值":
        if info['f(x)'] < global_min_value:
            global_min_value = info['f(x)']
    # 如果函数有全局最大值，会在所有局部最大值中取最大，或者在边界点取值
    # 对于本例函数，没有全局最大值

print(f"\n全局最小值: {global_min_value:.4f}")
print("全局最大值: 该函数没有全局最大值 (当 x 趋近于正负无穷时，f(x) 趋近于无穷大)。")


# --- 可视化 ---
# 生成 x 值范围，用于绘制函数图像
x_plot = np.linspace(-3.5, 4.5, 500) # 扩大范围以更好地展示函数行为
y_plot = f(x_plot)

plt.figure(figsize=(12, 7))

# 绘制函数 f(x) 的图像
plt.plot(x_plot, y_plot, label='函数 $f(x) = x^4 - 4x^3 - 2x^2 + 12x + 5$', color='blue', linewidth=2)

# 标记并标注极值点
for info in extrema_info:
    plt.plot(info['x'], info['f(x)'], marker='o', markersize=8, color=info['marker_color'],
             label=f'{info["type"]} ({info["x"]:.2f}, {info["f(x)"]:.2f})')
    
    # 添加文字标注，使其不重叠且易读
    text_offset_y = 15 if '最大' in info['type'] else -20 # 局部最大值文字在上，局部最小值文字在下
    text_offset_x = 0.2 if info['x'] < 0 else -0.2 # 根据 x 坐标调整文本位置
    
    plt.annotate(f'{info["type"]}\n({info["x"]:.2f}, {info["f(x)"]:.2f})',
                 xy=(info['x'], info['f(x)']),
                 xytext=(info['x'] + text_offset_x, info['f(x)'] + text_offset_y),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=0.5, headwidth=5),
                 horizontalalignment='center', verticalalignment='center',
                 fontsize=10, bbox=dict(boxstyle="round,pad=0.3", fc="yellow", ec="black", lw=0.5, alpha=0.7))


# 添加 x 轴和 y 轴的参考线
plt.axhline(0, color='gray', linewidth=0.8, linestyle='--')
plt.axvline(0, color='gray', linewidth=0.8, linestyle='--')

# 添加图表标题、轴标签、图例和网格
plt.title('带有多个极值的函数 $f(x)$ 及其极值点', fontsize=18)
plt.xlabel('x', fontsize=14)
plt.ylabel('f(x)', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, linestyle=':', alpha=0.7)

# 调整图表显示范围，确保所有关键点可见
plt.xlim(x_plot.min(), x_plot.max())
plt.ylim(y_plot.min() - 20, y_plot.max() + 20) # 留出一些边距

plt.show()