from numba import cuda
import numpy as np
import math
from time import time

@cuda.jit
def gpu_add(a, b, result, n):
    idx = cuda.threadIdx.x + cuda.blockDim.x * cuda.blockIdx.x
    if idx < n :
        result[idx] = a[idx] + b[idx]

def main():
    n = 20000000
    x = np.arange(n).astype(np.int32)
    y = 2 * x

    start = time()
    x_device = cuda.to_device(x)
    y_device = cuda.to_device(y)
    out_device = cuda.device_array(n)

    threads_per_block = 1024
    blocks_per_grid = math.ceil(n / threads_per_block)

    # 使用默认流
    gpu_add[blocks_per_grid, threads_per_block](x_device, y_device, out_device, n)
    gpu_result = out_device.copy_to_host()
    cuda.synchronize()
    print("gpu vector add time " + str(time() - start))

    start = time()

    # 使用5个stream
    number_of_streams = 5
    # 每个stream处理的数据量为原来的 1/5
    # 符号//得到一个整数结果
    segment_size = n // number_of_streams

    # 创建5个cuda stream
    stream_list = list()
    for i in range (0, number_of_streams):
        stream = cuda.stream()
        stream_list.append(stream)

    threads_per_block = 1024
    # 每个stream的处理的数据变为原来的1/5
    blocks_per_grid = math.ceil(segment_size / threads_per_block)
    streams_out_device = cuda.device_array(segment_size)
    streams_gpu_result = np.empty(n)

    # 启动多个stream
    for i in range(0, number_of_streams):
        # 传入不同的参数，让函数在不同的流执行
        x_i_device = cuda.to_device(x[i * segment_size : (i + 1) * segment_size], stream=stream_list[i])
        y_i_device = cuda.to_device(y[i * segment_size : (i + 1) * segment_size], stream=stream_list[i])

        gpu_add[blocks_per_grid, threads_per_block, stream_list[i]](
                x_i_device,
                y_i_device,
                streams_out_device,
                segment_size)

        streams_gpu_result[i * segment_size : (i + 1) * segment_size] = streams_out_device.copy_to_host(stream=stream_list[i])

    cuda.synchronize()
    print("gpu streams vector add time " + str(time() - start))

if __name__ == "__main__":
    main()
