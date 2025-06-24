import sys
import matplotlib
import numpy as np
import matplotlib.pyplot as plt

if sys.platform.startswith('win'):
    matplotlib.rcParams['font.family'] = ['SimHei'] # Windows的中文字体
elif sys.platform.startswith('darwin'):
    matplotlib.rcParams['font.family'] = ['Arial Unicode MS'] # Mac的中文字体
matplotlib.rcParams['axes.unicode_minus'] = False 

# 定义x范围
x = np.linspace(-2*np.pi, 2*np.pi, 1000)

# 原函数: y = x² + sin(x)
y = x**2 + np.sin(x)

# 一阶导数: y' = 2x + cos(x)
y_prime = 2*x + np.cos(x)

# 二阶导数: y'' = 2 - sin(x)
y_double_prime = 2 - np.sin(x)

# 创建图像
plt.figure(figsize=(12, 8))

# 绘制三个函数
plt.subplot(3, 1, 1)
plt.plot(x, y, 'b-', linewidth=2, label='$y = x^2 + \sin x$')
plt.title('原函数')
plt.grid(True, alpha=0.3)
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(x, y_prime, 'r-', linewidth=2, label="$y' = 2x + \cos x$")
plt.title('一阶导数')
plt.grid(True, alpha=0.3)
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(x, y_double_prime, 'g-', linewidth=2, label="$y'' = 2 - \sin x$")
plt.title('二阶导数')
plt.xlabel('x')
plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()
plt.show()