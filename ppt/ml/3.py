import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# --- 设置 matplotlib 以支持中文显示和负号 ---
# 这对于图表标题和标签中的中文显示至关重要。
matplotlib.rcParams['font.family'] = ['SimHei'] # 尝试使用SimHei（黑体），如果你的系统没有，请替换为其他中文字体，如'Microsoft YaHei'
matplotlib.rcParams['axes.unicode_minus'] = False # 解决负号显示为方块的问题

# --- 1. 生成模拟数据 ---
# 我们假设数据的真实来源是一个简单的一元二次函数 y = true_a * x^2
true_a = 2.0 # 设定真实的 'a' 值
np.random.seed(42) # 设置随机种子，以便每次运行结果一致
x_data = np.linspace(-5, 5, 50) # 在 -5 到 5 之间生成 50 个 x 值
# 根据真实 'a' 生成 y 值，并加入一些随机噪声，模拟真实世界数据的不完美性
y_data = true_a * x_data**2 + np.random.normal(0, 5, 50) # 噪声标准差为 5

# --- 2. 定义误差函数（均方误差 MSE） ---
def mse_loss(a_param, x_vals, y_vals):
    """
    计算给定参数 'a' 时，模型 y = a*x^2 的均方误差损失。
    
    参数:
    a_param (float): 模型的参数 'a'。
    x_vals (array): 输入 x 值的数据。
    y_vals (array): 真实的 y 值数据。
    
    返回:
    float: 均方误差值。
    """
    predictions = a_param * x_vals**2 # 根据当前 'a' 值计算预测的 y 值
    errors = y_vals - predictions # 计算每个点的误差 (真实值 - 预测值)
    mse = np.mean(errors**2) # 计算误差的平方的平均值
    return mse

# --- 3. 计算不同 'a' 值对应的损失 ---
# 定义一个 'a' 参数的探索范围，以便绘制损失函数的形状。
# 我们选择一个比真实值更大的范围，以清晰展示抛物线形状。
a_values_to_explore = np.linspace(-5, 10, 200) # 在 -5 到 10 之间生成 200 个 'a' 值
# 对每个 'a' 值，计算其对应的 MSE 损失
loss_values = [mse_loss(a, x_data, y_data) for a in a_values_to_explore]

# --- 4. 绘图：误差函数图像 ---
plt.figure(figsize=(10, 6)) # 设置图表大小

# 绘制参数 'a' 与损失值之间的关系曲线
plt.plot(a_values_to_explore, loss_values, color='red', linewidth=2, label='均方误差损失函数')

# --- 标记图表上的关键点和信息 ---
# 标记真实的参数 'a' 及其对应的损失点
true_param_loss = mse_loss(true_a, x_data, y_data)
plt.axvline(true_a, color='gray', linestyle='--', label=f'真实函数参数 $a={true_a}$')
plt.plot(true_a, true_param_loss, 'ro', markersize=8, label=f'真实参数点 (损失={true_param_loss:.2f})')

# 找到并标记损失函数的最小值点，这代表最佳拟合参数 'a'
min_loss_idx = np.argmin(loss_values) # 找到损失值最小的索引
optimal_a = a_values_to_explore[min_loss_idx] # 对应的 'a' 值就是最佳拟合参数
optimal_loss = loss_values[min_loss_idx] # 最小损失值

plt.plot(optimal_a, optimal_loss, 'go', markersize=8, label=f'最佳拟合参数 $a={optimal_a:.2f}$ (最小损失={optimal_loss:.2f})')
plt.axvline(optimal_a, color='green', linestyle=':', label=f'最佳拟合参数 $a={optimal_a:.2f}$')

# 添加图表标题、轴标签和图例
plt.title('简化一元二次函数 $y = ax^2$ 的均方误差损失函数', fontsize=16)
plt.xlabel('参数 $a$ 的值', fontsize=12)
plt.ylabel('均方误差 (MSE)', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, linestyle=':', alpha=0.7) # 添加网格线
plt.ylim(bottom=0) # 确保Y轴从0开始，因为损失值不会是负数

plt.show()

# --- 5. 额外绘图：展示拟合效果 ---
# 这一部分是为了让你更直观地看到拟合后的函数与原始数据和真实函数的关系
plt.figure(figsize=(10, 6))

# 绘制原始数据点
plt.scatter(x_data, y_data, label='原始数据点 (含噪声)', s=20, color='blue', alpha=0.7)

# 绘制真实的原始函数曲线（我们用来生成数据的函数）
x_fit_range = np.linspace(min(x_data), max(x_data), 100) # 用于绘制平滑曲线的 x 范围
y_true_curve = true_a * x_fit_range**2
plt.plot(x_fit_range, y_true_curve, label=f'真实函数 $y={true_a}x^2$', color='green', linestyle='--', linewidth=2)

# 绘制根据最小损失找到的最佳参数 'a' 所拟合出的函数曲线
y_fitted_curve = optimal_a * x_fit_range**2
plt.plot(x_fit_range, y_fitted_curve, label=f'拟合函数 $y={optimal_a:.2f}x^2$', color='red', linewidth=2)

plt.title('原始数据与拟合曲线', fontsize=16)
plt.xlabel('x', fontsize=12)
plt.ylabel('y', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, linestyle=':', alpha=0.7)
plt.show()