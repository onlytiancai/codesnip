import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# matplotlib.rcParams['font.family'] = ['SimHei'] # 或者你系统支持的中文 C:\Windows\Fonts
if sys.platform.startswith('win'):
    matplotlib.rcParams['font.family'] = ['SimHei'] # Windows的中文字体
elif sys.platform.startswith('darwin'):
    matplotlib.rcParams['font.family'] = ['Arial Unicode MS'] # Mac的中文字体
matplotlib.rcParams['axes.unicode_minus'] = False 

# --- 1. 生成模拟数据 (实际应用中，这些是你的100个输入点) ---
# 假设真实的一元二次函数是 y = 2x^2 - 3x + 5
true_a, true_b, true_c = 2, -3, 5

# 生成100个 x 值
np.random.seed(42) # 为了结果可复现性
x_data = np.linspace(-10, 10, 100)

# 生成对应的 y 值，并加入一些随机噪声模拟真实世界数据
y_data = true_a * x_data**2 + true_b * x_data + true_c + np.random.normal(0, 5, 100) # 加入标准差为5的噪声

# --- 2. 构建矩阵 X 和向量 y ---
# X 矩阵的每一行是 [x_i^2, x_i, 1]
# np.vstack 将多个一维数组按行堆叠起来
# .T 是转置操作
X_matrix = np.vstack([x_data**2, x_data, np.ones(len(x_data))]).T
# y 向量就是 y_data
y_vector = y_data

# --- 3. 使用 NumPy 的最小二乘函数求解 ---
# np.linalg.lstsq 是专门用于解决线性最小二乘问题的函数
# 返回值：
#   coefficients: 最小二乘解 (a, b, c)
#   residuals: 残差平方和 (通常不关心)
#   rank: 矩阵 X 的秩
#   singular_values: X 的奇异值
coefficients, residuals, rank, singular_values = np.linalg.lstsq(X_matrix, y_vector, rcond=None)

# 提取求解出的 a, b, c
a_found, b_found, c_found = coefficients

print(f"原始参数: a={true_a}, b={true_b}, c={true_c}")
print(f"拟合结果: a={a_found:.4f}, b={b_found:.4f}, c={c_found:.4f}")

# --- 4. 可视化结果 ---
plt.figure(figsize=(10, 6))

# 绘制原始数据点
plt.scatter(x_data, y_data, label='原始数据点 (含噪声)', s=15, color='blue', alpha=0.6)

# 绘制拟合出的曲线
x_fit = np.linspace(min(x_data), max(x_data), 100)
y_fit = a_found * x_fit**2 + b_found * x_fit + c_found
plt.plot(x_fit, y_fit, color='red', label=f'拟合曲线: y = {a_found:.2f}x^2 + {b_found:.2f}x + {c_found:.2f}', linewidth=2)

# 绘制真实的原始曲线 (用于比较，实际应用中你不知道这个)
y_true = true_a * x_fit**2 + true_b * x_fit + true_c
plt.plot(x_fit, y_true, color='green', linestyle='--', label='真实函数 (无噪声)', linewidth=1)


plt.title('一元二次函数拟合 (最小二乘法)')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()