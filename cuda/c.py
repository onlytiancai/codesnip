from numba import cuda, float32
import numpy as np
import math
from time import time

# thread per block
# 每个block有 BLOCK_SIZE x BLOCK_SIZE 个元素
BLOCK_SIZE = 16

@cuda.jit
def matmul(A, B, C):
    """  矩阵乘法 C = A * B
    """
    row = cuda.threadIdx.x + cuda.blockDim.x * cuda.blockIdx.x
    col = cuda.threadIdx.y + cuda.blockDim.y * cuda.blockIdx.y

    if row < C.shape[0] and col < C.shape[1]:
        tmp = 0.
        for k in range(A.shape[1]):
            tmp += A[row, k] * B[k, col]
        C[row, col] = tmp

@cuda.jit
def matmul_shared_memory(A, B, C):
    """
    使用Shared Memory的矩阵乘法 C = A * B
    """
    # 在Shared Memory中定义向量
    # 向量可被整个Block的所有Thread共享
    # 必须声明向量大小和数据类型
    sA = cuda.shared.array(shape=(BLOCK_SIZE, BLOCK_SIZE), dtype=float32)
    sB = cuda.shared.array(shape=(BLOCK_SIZE, BLOCK_SIZE), dtype=float32)

    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    row = cuda.threadIdx.x + cuda.blockDim.x * cuda.blockIdx.x
    col = cuda.threadIdx.y + cuda.blockDim.y * cuda.blockIdx.y

    if row >= C.shape[0] and col >= C.shape[1]:
        # 当(x, y)越界时退出
        return

    tmp = 0.
    # 以一个 BLOCK_SIZE x BLOCK_SIZE 为单位
    for m in range(math.ceil(A.shape[1] / BLOCK_SIZE)):
        sA[tx, ty] = A[row, ty + m * BLOCK_SIZE]
        sB[tx, ty] = B[tx + m * BLOCK_SIZE, col]
        # 线程同步，等待Block中所有Thread预加载结束
        # 该函数会等待所有Thread执行完之后才执行下一步
        cuda.syncthreads()
        # 此时已经将A和B的子矩阵拷贝到了sA和sB

        # 计算Shared Memory中的向量点积
        # 直接从Shard Memory中读取数据的延迟很低
        for n in range(BLOCK_SIZE):
            tmp += sA[tx, n] * sB[n, ty]

        # 线程同步，等待Block中所有Thread计算结束
        cuda.syncthreads()

    # 循环后得到每个BLOCK的点积之和
    C[row, col] = tmp

def main():
    # 初始化矩阵
    M = 6000
    N = 4800
    P = 4000
    A = np.random.random((M, N)) # 随机生成的 [M x N] 矩阵
    B = np.random.random((N, P)) # 随机生成的 [N x P] 矩阵

    A_device = cuda.to_device(A)
    B_device = cuda.to_device(B)
    C_device = cuda.device_array((M, P)) # [M x P] 矩阵

    # 执行配置
    threads_per_block = (BLOCK_SIZE, BLOCK_SIZE)
    blocks_per_grid_x = int(math.ceil(A.shape[0] / BLOCK_SIZE))
    blocks_per_grid_y = int(math.ceil(B.shape[1] / BLOCK_SIZE))
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

    start = time()
    matmul[blocks_per_grid, threads_per_block](A_device, B_device, C_device)
    cuda.synchronize()
    print("matmul time :" + str(time() - start))

    start = time()
    matmul_shared_memory[blocks_per_grid, threads_per_block](A_device, B_device, C_device)
    cuda.synchronize()
    print("matmul with shared memory time :" + str(time() - start))
    C = C_device.copy_to_host()

if __name__ == "__main__":
    main()
