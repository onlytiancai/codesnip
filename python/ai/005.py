import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

# 维度
d = 256

# -----------------------------
# 1. 构造一个只有单维离群值的向量
# -----------------------------
x = np.zeros(d)

# 第0维是极端离群值
x[0] = 100

# -----------------------------
# 2. 随机正交旋转 (PolarQuant)
# -----------------------------
R = np.random.randn(d, d)
Q, _ = np.linalg.qr(R)

x_rot = Q @ x

# -----------------------------
# 3. 可视化
# -----------------------------
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(x)
plt.title("Before Rotation (Outlier in one dimension)")
plt.xlabel("Dimension")
plt.ylabel("Value")

plt.subplot(1,2,2)
plt.plot(x_rot)
plt.title("After Random Rotation (Spread across dimensions)")
plt.xlabel("Dimension")
plt.ylabel("Value")

plt.show()

# -----------------------------
# 4. 打印统计信息
# -----------------------------
print("Before rotation")
print("max abs:", np.max(np.abs(x)))
print("non-zero dims:", np.sum(np.abs(x) > 1e-6))

print("\nAfter rotation")
print("max abs:", np.max(np.abs(x_rot)))
print("non-zero dims:", np.sum(np.abs(x_rot) > 1e-6))