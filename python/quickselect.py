# -*- coding:utf-8 -*-
'''
description:
    找出arr数组中第k个最小元素
    Martin Fowler："在你不知道如何测试代码之前，就不该编写程序。而一旦你完成了程序，测试代码也应该完成。除非测试成功，你不能认为你编写出了可以工作的程序。"
'''
import logging

__author__ = 'wawa'
__version__ = '0.1'

def select_bysort(arr, k):
    '先排序，然后直接选出第k个元素'
    sorted_arr = sorted(arr) 
    return sorted_arr[k - 1]

def median3(arr, left, right):
    '求一个数组的最左，最右，中间三个位置的数的中值，并把最小的放在最左边，最大的放在最右边，中值放在倒数第二位'
    center = (left+right) / 2
    if arr[center] < arr[left]:
        arr[center], arr[left] = arr[left], arr[center]
    if arr[right] < arr[left]:
        arr[right], arr[left] = arr[left], arr[right]
    if arr[right] < arr[center]:
        arr[right], arr[center] = arr[center], arr[right]

    arr[center], arr[right - 1] = arr[right - 1], arr[center]
    return arr[right - 1]

def quickselect(arr, k, left=None, right=None):
    '''
    Mark Alllen Weiss《数据结构与算法分析》里的算法，大概原理类似于快速排序，最坏情况是O(N**2),平均复杂度是O(N)
    用500w大小的数组测试，用内置排序算法排序后取第k个最小值需要7秒，用该算法需要3秒多一点，性能约翻了一倍。
    python内置的排序算法是Tim Peter写的，基本上是宇宙中性能最好的通用排序算法，http://en.wikipedia.org/wiki/Timsort
    算法描述大致如下：
        令|S|为S中元素的个数
        1、如果|S|=1,那么k肯定是1，直接返回，如果|S|很小，直接排序后返回第k个最小元素
        2、选取一个枢纽元v∈S,（这里我们用三数中值法取的枢纽元）
        3、将S-|v|分割成S1和S2，和快速排序一样
        4、如果k<=|S1|，那么第k个最小元素肯定在S1中，这种情况下返回quickselect(S1, k)
            如果k=1+|S1|,那么枢纽元就是第k个最小元素，直接返回
            否则，第k个最小元就在S2中，他是S2中第（k-|S1| - 1）个最小元，再做一次递归并返回quickselect(S2, k-|S1|-1)
    '''
    logging.debug('quickselect:arr=%s k=%s left=%s right=%s', arr, k, left, right)

    if not arr: raise ValueError('arr is empty.')
    if k <= 0: raise ValueError('k error.')
    if not left: left = 0  
    if not right: right = len(arr) - 1
    if right - left + 1 == 1: return arr[0]
    if right - left + 1 <= 3:
        arr[left:right+1] = sorted(arr[left:right+1]) 
        return arr[k - 1]

    pivot = median3(arr, left, right) 

    i, j = left, right - 1 
    while True:
        while True:
            i = i + 1
            if arr[i] >= pivot:break 
        while True:
            j = j - 1
            if arr[j] <= pivot:break 
        if i < j:
            arr[i], arr[j] = arr[j], arr[i]
        else:
            break

    arr[i], arr[right - 1] = arr[right - 1], arr[i]

    if k <= i:
        return quickselect(arr, k, left, i - 1)
    elif k > i + 1:
        return quickselect(arr, k, i + 1, right)
    
    return arr[k-1]
    
import unittest

class Median3TestCase(unittest.TestCase):
    test_value_map = [
        ([1,2,3],2),
        ([3,2,1],2),
        ([3,3,3],3),
        ([2,3,3,3,5,1],2),
            ]
    def test_median3(self):
        'median3 test'
        for arr, expected in self.test_value_map:
            self.assertEqual(median3(arr, 0, len(arr) - 1), expected)

class QuickSelectTestCase(unittest.TestCase):
    test_value_map = [
        ([8,1,4,9,6,3,5,2,7,0], 5, 4),
        ([1,2,3,4,5], 3, 3),
        ([5,4,3,2,1], 3, 3),
        ([1,3,2,5,4], 3, 3),
        ([1,3,2,5,4], 5, 5),
        ([1,3,2,5,4], 1, 1),
        ([1,3,3,2,5,4], 3, 3),
        ([1,3,3,2,5,4], 6, 5),
        
    ]
    def setUp(self):
        import random
        arr = range(1, 100)
        random.shuffle(arr)
        self.test_value_map.append((arr, 98, 98))

    def inner_test(self, quickselect_func):
        for arr, k, expected in self.test_value_map:
            result = quickselect_func(arr, k)
            self.assertEqual(result, expected)

    def test_select_bysort(self):
        'select_bysort test'
        self.inner_test(select_bysort) 

    def test_quickselect(self):
        'quickselect test'
        self.inner_test(quickselect) 

def performance_test():
    logging.getLogger().setLevel(logging.ERROR)

    from datetime import datetime
    def time_test(test_name):
        def inner(f):
            def inner2(*args, **kargs):
                start_time = datetime.now()
                result = f(*args, **kargs)
                print '%s take time:%s' % (test_name, (datetime.now() - start_time))
                return result
            return inner2
        return inner

    @time_test("gen_test_data")
    def gen_test_data():
        import random
        test_data_count = 500*10000
        arr = range(1, test_data_count)
        random.shuffle(arr)
        k = random.randint(1, test_data_count)
        return arr, k, k

    @time_test("test_by_select_bysort")
    def test_by_select_bysort(arr, k, expected):
        result = select_bysort(arr, k)
        print 'test_by_select_bysort:%s %s' % (k, 'successed' if expected == result else 'faild')
    
    @time_test("test_by_quickselect")
    def test_by_quickselect(arr, k, expected):
        result = quickselect(arr, k)
        print 'test_by_quickselect:%s %s' % (k, 'successed' if expected == result else 'faild')

    arr, k, expected = gen_test_data()
    test_by_select_bysort(arr, k, expected)
    test_by_quickselect(arr, k, expected)

if __name__ == '__main__':
    performance_test()

    logging.getLogger().setLevel(logging.NOTSET)
    unittest.main(verbosity=2)
