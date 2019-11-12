import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf

import time
def time_matmul(x):
    start = time.time()
    for loop in range(1000):
        tf.matmul(x, x)
    result = time.time() - start
    print('1000 loops: {:0}ms'.format(1000*result))

# 强制使用CPU
print('On CPU:')
with tf.device('CPU:0'):
    x = tf.random.uniform([1000, 1000])
    # 使用断言验证当前是否为CPU0
    assert x.device.endswith('CPU:0')
    time_matmul(x)

# 如果存在GPU,强制使用GPU
if tf.test.is_gpu_available():
    print('On GPU:')
    with tf.device('GPU:0'):
        x = tf.random.uniform([1000, 1000])
    # 使用断言验证当前是否为GPU0
    assert x.device.endswith('GPU:0')
    time_matmul(x)
