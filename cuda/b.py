from numba import cuda
import numpy as np
import math
from time import time

@cuda.jit
def matmul(A, B, C):
    """  矩阵乘法 C = A * B
    """
    # Numba库提供了更简易的计算方法
    # x, y = cuda.grid(2)
    # 具体计算公式如下
    row = cuda.threadIdx.x + cuda.blockDim.x * cuda.blockIdx.x
    col = cuda.threadIdx.y + cuda.blockDim.y * cuda.blockIdx.y


    if row < C.shape[0] and col < C.shape[1]:
        tmp = 0.
        for k in range(A.shape[1]):
            tmp += A[row, k] * B[k, col]
        C[row, col] = tmp

def main():
    # 初始化矩阵
    M = 6000
    N = 4800
    P = 4000
    A = np.random.random((M, N)) # 随机生成的 [M x N] 矩阵
    B = np.random.random((N, P)) # 随机生成的 [N x P] 矩阵

    start = time()
    A = cuda.to_device(A)
    B = cuda.to_device(B)
    C_gpu = cuda.device_array((M, P))

    # 执行配置
    threads_per_block = (16, 16)
    blocks_per_grid_x = int(math.ceil(A.shape[0] / threads_per_block[0]))
    blocks_per_grid_y = int(math.ceil(B.shape[1] / threads_per_block[1]))
    blocksPerGrid = (blocks_per_grid_x, blocks_per_grid_y)

    # 启动核函数
    matmul[blocksPerGrid, threads_per_block](A, B, C_gpu)

    # 数据拷贝
    C = C_gpu.copy_to_host()
    cuda.synchronize()

    print("gpu matmul time :" + str(time() - start))

    start = time()
    C_cpu = np.empty((M, P), np.float)
    np.matmul(A, B, C_cpu)
    print("cpu matmul time :" + str(time() - start))

    # 验证正确性
    if np.allclose(C_cpu, C):
        print("gpu result correct")

if __name__ == "__main__":
    main()
