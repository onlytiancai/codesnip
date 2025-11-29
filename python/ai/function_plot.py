import numpy as np
import matplotlib.pyplot as plt

# 定义自变量范围
x = np.linspace(-10, 10, 400)

# 定义函数
f = x**2
g = 2*x
f_plus_g = f + g
two_f = 2 * f

# 绘图
plt.figure(figsize=(8, 6))

plt.plot(x, f, label='f = x^2')
plt.plot(x, g, label='g = 2x')
plt.plot(x, f_plus_g, label='f + g')
plt.plot(x, two_f, label='2f')

plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.title('Functions Visualization')
plt.grid(True)

plt.show()
