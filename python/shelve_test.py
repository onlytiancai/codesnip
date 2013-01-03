# -*- coding: utf -*-
'''
测试shelve的安全性，如断电，kill -9，数据会不会损坏

### 测试方法：

    1. 起一个进程不断的进行写入测试，然后两秒后kill -9
    1. 连续测试多次后再读出数据，看能否读出数据，并数据总量不少
    1. 具体逻辑见shelve_test.sh

### 测试结果

    1. kill -9后数据不会破坏
    1. 重要数据的保存后要调用sync()，否则可能有数据没有保存下来
'''

import shelve

d = shelve.open('d')


def test_write():
    while True:
        for i in range(100):
            key, value = str(i), i
            d[key] = value
            d.sync()


def test_read():
    print '*' * 20, 'read', str(len(d))
    print d.items()
    d.close()

if __name__ == '__main__':
    import sys
    arg = sys.argv[1]
    if arg == 'write':
        test_write()
    else:
        test_read()
