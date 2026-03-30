import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# -----------------------------
# 1. 生成带强离群值的数据
# -----------------------------
d = 256
n = 5000

# 基础高斯
X = np.random.normal(0, 1, size=(n, d))

# 强制制造离群维度（模拟attention outliers）
outlier_dims = [0, 1, 2, 3]

# 放大方差
for dim in outlier_dims:
    X[:, dim] = np.random.normal(0, 15, size=n)

# 再加入极端离群点
for dim in outlier_dims:
    idx = np.random.choice(n, size=50, replace=False)
    X[idx, dim] += np.random.normal(0, 60, size=50)

# -----------------------------
# 2. PolarQuant 随机旋转
# -----------------------------
R = np.random.randn(d, d)
Q, _ = np.linalg.qr(R)

X_rot = X @ Q

# -----------------------------
# 3. 统计函数
# -----------------------------
def compute_stats(X):
    skews = []
    kurts = []
    outliers = []

    for i in range(X.shape[1]):
        x = X[:, i]
        mean = np.mean(x)
        std = np.std(x)

        skew = np.mean(((x - mean) / std) ** 3)
        kurt = np.mean(((x - mean) / std) ** 4) - 3
        outlier_ratio = np.mean(np.abs(x - mean) > 3 * std)

        skews.append(skew)
        kurts.append(kurt)
        outliers.append(outlier_ratio)

    return np.array(skews), np.array(kurts), np.array(outliers)

skew_before, kurt_before, out_before = compute_stats(X)
skew_after, kurt_after, out_after = compute_stats(X_rot)

# -----------------------------
# 4. 打印统计结果
# -----------------------------
print("\n===== PolarQuant Statistics Comparison =====\n")

print("Average Skewness")
print("Before:", np.mean(skew_before))
print("After :", np.mean(skew_after))

print("\nAverage Kurtosis")
print("Before:", np.mean(kurt_before))
print("After :", np.mean(kurt_after))

print("\nAverage >3σ Outlier Ratio")
print("Before:", np.mean(out_before))
print("After :", np.mean(out_after))

print("\nMax Kurtosis (worst dimension)")
print("Before:", np.max(kurt_before))
print("After :", np.max(kurt_after))

# -----------------------------
# 5. 直方图可视化（最坏维度）
# -----------------------------
worst_dim = np.argmax(kurt_before)

plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.hist(X[:, worst_dim], bins=80)
plt.title(f"Before Rotation (dim={worst_dim})")

plt.subplot(1,2,2)
plt.hist(X_rot[:, worst_dim], bins=80)
plt.title(f"After Rotation (dim={worst_dim})")

plt.show()

# -----------------------------
# 6. Kurtosis 全维度对比图
# -----------------------------
plt.figure(figsize=(10,5))
plt.plot(sorted(kurt_before), label="Before")
plt.plot(sorted(kurt_after), label="After")
plt.title("Kurtosis Distribution Across Dimensions")
plt.legend()
plt.show()

# -----------------------------
# 7. Outlier ratio 对比
# -----------------------------
plt.figure(figsize=(10,5))
plt.plot(sorted(out_before), label="Before")
plt.plot(sorted(out_after), label="After")
plt.title(">3σ Outlier Ratio Across Dimensions")
plt.legend()
plt.show()