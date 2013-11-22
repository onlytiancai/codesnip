# -*- coding: utf-8 -*-
from decimal import Decimal
from multiprocessing import Pool

num_rec = 10000
width = Decimal(1.0) / Decimal(num_rec)
cpu_count = 4

def calc_pi():
    '串行计算元周率'
    sum_value = Decimal(0.0) 
    for i in xrange(num_rec):
        mid = Decimal(i + 0.5) * width
        height = Decimal(4.0) / (Decimal(1.0) + mid * mid)
        sum_value += height

    pi = width * sum_value
    return pi

def calc_sum(args):
    '并行计算元周率中每个线程执行的分解求和'
    begin, end = args
    sum_value = Decimal(0.0) 

    for i in xrange(begin, end):
        mid = Decimal(i + 0.5) * width
        height = Decimal(4.0) / (Decimal(1.0) + mid * mid)
        sum_value += height
    return sum_value

def split_data(length, n):
    '''并行数据分解，把计算量分解成n块
    >>> split_data(10, 4)
    [(0, 3), (3, 6), (6, 9), (9, 10)]
    >>> split_data(101, 4)
    [(0, 26), (26, 52), (52, 78), (78, 101)]
    '''
    if n > length: raise ValueError()

    result = []
    for i in xrange(length):
        block_size = (length / n + 1)
        if i % block_size == 0:
            begin = i
            end = i + block_size
            end = length if end > length else end
            result.append((begin, end))
    return result

def calc_pi2():
    '并行计算元周率'
    pool = Pool(processes=cpu_count)
    result = pool.map(calc_sum, split_data(num_rec, cpu_count))
    return width * sum(result)

if __name__ == '__main__':
    import doctest
    doctest.testmod()

    import time

    t = time.time()
    print 'pi1=', calc_pi(), time.time() - t 

    t = time.time()
    print 'pi2=', calc_pi2(), time.time() - t
