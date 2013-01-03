# -*- coding: utf-8 -*-
'''
原题：http://weibo.com/1915548291/zcIjIuzVVA

#面试编程题#给定一个数t，以及n个整数，在这n个整数中找到相加之和为t的所有组合，
例如t = 4，n = 6，这6个数为[4, 3, 2, 2, 1, 1]，这样输出就有4个不同的组合，
它们的相加之和为4：4, 3+1, 2+2, and 2+1+1。请设计一个高效算法实现这个需求。
'''

import itertools


def get_result(t, arr):
    '比较暴力比较搓的版本'
    results = set()
    for i in range(1, len(arr) + 1):
        for result in itertools.combinations(arr, i):
            if sum(result) == t:
                results.add(result)
    return results
        

if __name__ == '__main__':
    for result in get_result(4, [4, 3, 2, 2, 1, 1]):
        print '+'.join(map(str,result))
