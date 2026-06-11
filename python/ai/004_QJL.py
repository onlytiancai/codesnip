import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)

d = 512
trials = 500

errors_naive = []
errors_qjl = []

for _ in range(trials):
    x = np.random.randn(d)
    y = np.random.randn(d)

    # 原始内积
    true_dot = np.dot(x, y)

    # 低bit量化 (模拟3bit)
    scale = np.max(np.abs(x))
    q = np.round(x / scale * 3) / 3 * scale

    # naive量化内积
    naive_dot = np.dot(q, y)

    # QJL: 记录误差符号
    error = x - q
    sign = np.sign(error)

    # 随机投影矩阵
    R = np.random.choice([-1,1], size=(d,))

    # 1bit correction
    correction = np.dot(sign * R, y * R) / d

    qjl_dot = naive_dot + correction

    errors_naive.append(naive_dot - true_dot)
    errors_qjl.append(qjl_dot - true_dot)

plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.hist(errors_naive, bins=50)
plt.title("Naive Quantization Error")

plt.subplot(1,2,2)
plt.hist(errors_qjl, bins=50)
plt.title("QJL Corrected Error")

plt.show()