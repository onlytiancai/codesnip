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
        outer_product = matrix_multiplication(
            [[eigenvector[i] * eigenvector[j] for j in range(n)] for i in range(n)],
            [[eigenvalue]]
        )
        
        A = [[A[i][j] - outer_product[i][j] for j in range(n)] for i in range(n)]
        
    return eigenvalues, eigenvectors

# Example usage
A = [[4, -2],
     [1, 1]]

eigenvalues, eigenvectors = eigen_decomposition(A)

print("Eigenvalues:", eigenvalues)
print("Eigenvectors:", eigenvectors)
