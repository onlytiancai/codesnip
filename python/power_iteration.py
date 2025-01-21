def matrix_multiplication(A, B):
    """Perform matrix multiplication"""
    result = [[0 for _ in range(len(B[0]))] for _ in range(len(A))]
    for i in range(len(A)):
        for j in range(len(B[0])):
            for k in range(len(B)):
                result[i][j] += A[i][k] * B[k][j]
    return result

def transpose(matrix):
    """Transpose a matrix"""
    return [[matrix[j][i] for j in range(len(matrix))] for i in range(len(matrix[0]))]

def power_iteration(A, num_iterations: int = 100):
    """Perform power iteration to find the largest eigenvalue and its eigenvector"""
    n = len(A)
    # Start with a random vector
    b_k = [1] * n  # Initial guess
    for _ in range(num_iterations):
        # Calculate the matrix-by-vector product Ab
        b_k1 = matrix_multiplication(A, [[b] for b in b_k])
        
        # Find the norm (magnitude) of b_k1
        b_k1_norm = sum(x[0] ** 2 for x in b_k1) ** 0.5
        
        # Normalize the vector
        b_k = [x[0] / b_k1_norm for x in b_k1]
    
    # Approximate eigenvalue
    eigenvalue = sum(b_k[i] * sum(A[i][j] * b_k[j] for j in range(n)) for i in range(n))
    return eigenvalue, b_k

def eigen_decomposition(A):
    """Perform Eigen Decomposition"""
    n = len(A)
    eigenvalues = []
    eigenvectors = []

    for _ in range(n):
        # Get the largest eigenvalue and corresponding eigenvector
        eigenvalue, eigenvector = power_iteration(A)
        eigenvalues.append(eigenvalue)
        eigenvectors.append(eigenvector)

        # Deflate the matrix A by subtracting the outer product of the eigenvector
        # A' = A - λ vv^T
        outer_product = [[eigenvalue * eigenvector[i] * eigenvector[j] for j in range(n)] for i in range(n)]

        # Update matrix A
        A = [[A[i][j] - outer_product[i][j] for j in range(n)] for i in range(n)]

    return eigenvalues, eigenvectors

# Example usage
A = [[4, -2],
     [1, 1]]
print('A:', A)
eigenvalues, eigenvectors = eigen_decomposition(A)

print("Eigenvalues:", eigenvalues)
print("Eigenvectors:", eigenvectors)

# ---------------- 使用特征方程验证幂迭代得到的特征值和特征向量 
import numpy as np
eigenvectors = np.array(eigenvectors).T
for i in range(len(eigenvalues)):
    lhs = np.dot(A, eigenvectors[:, i])  # A @ v
    rhs = eigenvalues[i] * eigenvectors[:, i]  # λ * v
    print(f"验证特征值 {eigenvalues[i]} 对应的特征向量:")
    print("A @ v:", lhs)
    print("λ * v:", rhs)
    print("误差:", np.linalg.norm(lhs - rhs))

# --------------- 和 numpy 结果对比
eigenvalues, eigenvectors = np.linalg.eig(A)
print("特征值:")
print(eigenvalues)
print("\n特征向量:")
print(eigenvectors)

eigenvectors = np.array(eigenvectors).T
for i in range(len(eigenvalues)):
    lhs = np.dot(A, eigenvectors[:, i])  # A @ v
    rhs = eigenvalues[i] * eigenvectors[:, i]  # λ * v
    print(f"验证特征值 {eigenvalues[i]} 对应的特征向量:")
    print("A @ v:", lhs)
    print("λ * v:", rhs)
    print("误差:", np.linalg.norm(lhs - rhs))

# --- 使用 numpy 的幂迭代

def power_iteration(A, tol=1e-8, max_iter=1000):
    """计算矩阵的主特征值和对应特征向量"""
    n = A.shape[0]
    b_k = np.random.rand(n)  # 初始化随机向量
    b_k /= np.linalg.norm(b_k)  # 归一化

    for _ in range(max_iter):
        # 矩阵-向量乘法
        b_k1 = np.dot(A, b_k)

        # 归一化向量
        b_k1_norm = np.linalg.norm(b_k1)
        b_k1 /= b_k1_norm

        # 检查收敛
        if np.linalg.norm(b_k1 - b_k) < tol:
            break
        b_k = b_k1

    eigenvalue = np.dot(b_k.T, np.dot(A, b_k))  # 计算特征值
    return eigenvalue, b_k


# 测试代码
A = np.array([[4, -2],
              [1,  1]])

eigenvalue, eigenvector = power_iteration(A)

print("幂迭代计算的特征值:", eigenvalue)
print("幂迭代计算的特征向量:", eigenvector)
