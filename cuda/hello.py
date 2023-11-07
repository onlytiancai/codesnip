import numpy as np
import numba
from numba import cuda
from numba import jit
import timeit

print('np version', np.__version__)
print('numba version', numba.__version__)

cuda.detect()

x = np.arange(1000000).reshape(1000, 1000)

def go(a):
    trace = 0.0
    for i in range(a.shape[0]):
        trace += np.tanh(a[i, i]) # 计算数组所有元素的双曲正切
    return a + trace


go_fast = jit(nopython=True)(go)
'''该函数首次调用将被编译成机器码'''

print('go:', timeit.timeit('go(x)', 'from __main__ import go, x', number=1))
go_fast(x) # 预热
print('go_fast:', timeit.timeit('go_fast(x)', 'from __main__ import go_fast,x', number=1))
