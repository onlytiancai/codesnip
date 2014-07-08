# -*- coding: utf-8 -*-
'''
用动态规划求斐波那契数列比用普通递归快很多
'''


def fib(n):
    if n == 0: return 0
    if n == 1: return 1
    return fib(n - 1) + fib(n - 2)


def cache(func):
    cached = {}

    def inner(*args):
        if args in cached:
            return cached[args]
        else:
            ret = func(*args)
            cached[args] = ret
            return ret

    return inner


@cache
def fib2(n):
    if n == 0: return 0
    if n == 1: return 1
    return fib2(n - 1) + fib2(n - 2)


if __name__ == '__main__':
    # unit test
    for i in range(30):
        assert fib2(i) == fib(i)

    # benchmark test
    import timeit
    print timeit.timeit('fib2(200)', setup="from __main__ import fib2", number=1)
    print timeit.timeit('fib(30)', setup="from __main__ import fib", number=1)
