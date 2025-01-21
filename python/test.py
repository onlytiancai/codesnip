import numpy as np
# 假设幂迭代得到的特征值和特征向量
eigenvalues_power = [2.0, 3.0]
eigenvectors_power = [[0.7071, 0.8944],  # v1
                      [0.7071, 0.4472]]  # v2 (列向量)

# 转换为 numpy 数组
eigenvectors_power = np.array(eigenvectors_power).T

# 验证特征方程
for i in range(len(eigenvalues_power)):
    lhs = np.dot(A, eigenvectors_power[:, i])  # A @ v
    rhs = eigenvalues_power[i] * eigenvectors_power[:, i]  # λ * v
    print(f"验证特征值 {eigenvalues_power[i]} 对应的特征向量:")
    print("A @ v:", lhs)
    print("λ * v:", rhs)
    print("误差:", np.linalg.norm(lhs - rhs))
